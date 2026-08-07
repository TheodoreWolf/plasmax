"""Vmapped direct-gradient baselines for policies and actuator schedules.

``direct_policy`` learns a deterministic residual MLP and
``direct_knots_{1,10,100}`` optimizes a piecewise-linear open-loop actuator
schedule. Both methods differentiate through short simulator windows and cut
gradients between windows, avoiding unstable long-horizon BPTT while carrying
the physical state through the complete episode.

Every accelerator job contains independent optimizer states for all training
seeds. The transform order remains
``vmap(seed -> vmap(value_and_grad(single_rollout)))``; differentiating outside
the rollout vmap is invalid for TORAX's adaptive-loop custom JVP.
"""

# ruff: noqa: E402

from __future__ import annotations

import dataclasses
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, NamedTuple

from scripts._runtime import set_default_xla_flags

set_default_xla_flags("--xla_gpu_enable_command_buffer=")

import jax
import jax.numpy as jnp
import numpy as np
import optax
import tyro
from envelope import Continuous
from flax import serialization

from experiments.studies.baseline_study import (
    indexed_keys,
    run_slug,
    seed_keys,
    validate_reward,
)
from experiments.studies.direct_gradient import (
    apply_updates_with_backoff,
    make_knot_chunk,
)
from experiments.studies.optimize_open_loop import _make_rollout, _setpoint_theta_row
from experiments.studies.train_backprop_policy import (
    BatchMetrics,
    ResidualPolicy,
    policy_action,
)
from experiments.studies.transfer_eval import transfer_metrics, write_transfer_summary
from plasmax.environment.factory import load_env
from plasmax.environment.registry import resolve_backend
from plasmax.wrappers import unwrap_to_env_state
from training.envelope_gymnax import EnvelopeGymnax, to_typed_key
from training.vmap_logging import SeedBufferLogger

Algorithm = Literal[
    "direct_policy",
    "direct_knots_1",
    "direct_knots_10",
    "direct_knots_100",
]


@dataclasses.dataclass
class EnvConfig:
    env_setup: str = "iter/hybrid/flattop"
    backend: str = "bohm_gyrobohm"
    transfer_backend: str | None = None
    reward: str | None = None
    variant: Literal["oracle", "realistic"] = "realistic"
    disruption_penalty: float | None = None
    eval_n_envs: int = 16
    eval_seed: int = 10_000
    transfer_n_envs: int = 128


@dataclasses.dataclass
class DirectConfig:
    total_timesteps: int = 10_000_000
    eval_freq: int = 1_000_000
    # Zero selects 64 rollouts per optimizer seed for every direct method.
    # This keeps the GH200 saturated and gives knot schedules the same
    # simulator batch size as the feedback policy.
    num_rollouts: int = 0
    gradient_horizon: int = 32
    # SHAC used 2e-3 for smoother rigid-body simulators. TORAX's stiff
    # implicit solve produced non-finite follow-up gradients at both 2e-3 and
    # 5e-4. A ten-seed SPARC pilot remained finite at every checkpoint with
    # 1e-6, so use that conservative package default.
    policy_learning_rate: float = 1e-6
    knot_learning_rate: float = 5e-2
    hidden_sizes: tuple[int, ...] = (64, 64)
    grad_clip: float = 1.0
    nonfinite_backoff_factor: float = 0.5
    min_update_scale: float = 1e-3
    remat: bool = True


@dataclasses.dataclass
class WandbConfig:
    project: str = "plasmax"
    entity: str = "flair"
    group: str = "debug"
    mode: Literal["online", "offline", "disabled"] = "online"
    tags: tuple[str, ...] = ()


@dataclasses.dataclass
class Config:
    algorithm: Algorithm = "direct_policy"
    env: EnvConfig = dataclasses.field(default_factory=EnvConfig)
    direct: DirectConfig = dataclasses.field(default_factory=DirectConfig)
    wandb: WandbConfig = dataclasses.field(default_factory=WandbConfig)
    seed: int = 0
    num_seeds: int = 10
    history_dir: str | None = None
    study: str = "debug"
    strict_phase_reward: bool = True


class ChunkAux(NamedTuple):
    next_carry: Any
    rollout_return: jax.Array
    alive_steps: jax.Array


class NativePolicyCarry(NamedTuple):
    obs: jax.Array
    env_state: Any
    alive: jax.Array


class NativePolicyStep(NamedTuple):
    reward: jax.Array
    action: jax.Array
    done: jax.Array
    alive: jax.Array


@dataclasses.dataclass(frozen=True)
class NativePolicyChunk:
    initialize: Callable[[jax.Array], NativePolicyCarry]
    run: Callable[[Any, NativePolicyCarry], tuple[jax.Array, Any]]


@dataclasses.dataclass(frozen=True)
class NativePolicyObjective:
    evaluate: Callable[[Any, jax.Array], tuple[jax.Array, BatchMetrics]]


def _make_native_policy_chunk(
    env,
    policy: ResidualPolicy,
    action_setpoint: jax.Array,
    chunk_steps: int,
    *,
    remat: bool,
    policy_obs_indices: tuple[int, ...] | None = None,
) -> NativePolicyChunk:
    """Build a direct-policy chunk without Gymnax auto-reset branches."""

    def step(current_params, carry, _):
        obs, env_state, alive = carry

        def active_step(_):
            policy_obs = (
                obs
                if policy_obs_indices is None
                else obs[jnp.asarray(policy_obs_indices, dtype=jnp.int32)]
            )
            action = policy_action(
                policy,
                current_params,
                action_setpoint,
                policy_obs,
            )
            next_state, info = env.step(env_state, action)
            done = info.terminated | info.truncated
            return (
                NativePolicyCarry(info.obs, next_state, ~done),
                NativePolicyStep(info.reward, action, done, jnp.asarray(True)),
            )

        def inactive_step(_):
            return (
                NativePolicyCarry(obs, env_state, jnp.asarray(False)),
                NativePolicyStep(
                    jnp.zeros_like(unwrap_to_env_state(env_state).plasma.t),
                    jnp.zeros_like(action_setpoint),
                    jnp.asarray(False),
                    jnp.asarray(False),
                ),
            )

        return jax.lax.cond(alive, active_step, inactive_step, operand=None)

    scan_step = jax.checkpoint(step) if remat else step

    def initialize(key):
        env_state, info = env.init(key)
        return NativePolicyCarry(info.obs, env_state, jnp.asarray(True))

    def run(current_params, carry):
        next_carry, trajectory = jax.lax.scan(
            lambda state, unused: scan_step(current_params, state, unused),
            carry,
            None,
            length=chunk_steps,
        )
        return jnp.sum(trajectory.reward), (next_carry, trajectory)

    return NativePolicyChunk(initialize=initialize, run=run)


def _make_native_policy_objective(
    chunk: NativePolicyChunk,
) -> NativePolicyObjective:
    def single_loss(current_params, key):
        rollout_return, (_, trajectory) = chunk.run(
            current_params,
            chunk.initialize(key),
        )
        metrics = BatchMetrics(
            returns=rollout_return,
            alive_steps=jnp.sum(trajectory.alive),
        )
        return -rollout_return, metrics

    batch_evaluate = jax.vmap(single_loss, in_axes=(None, 0))

    def evaluate(current_params, keys):
        losses, metrics = batch_evaluate(current_params, keys)
        return jnp.mean(losses), metrics

    return NativePolicyObjective(evaluate=evaluate)


def _num_rollouts(cfg: Config) -> int:
    if cfg.direct.num_rollouts > 0:
        return cfg.direct.num_rollouts
    return 64


def _knots(algorithm: Algorithm) -> int:
    if not algorithm.startswith("direct_knots_"):
        raise ValueError(f"{algorithm!r} is not a knot-schedule algorithm")
    return int(algorithm.rsplit("_", 1)[-1])


def _backend_name(alias_or_path: str) -> str:
    return Path(resolve_backend(alias_or_path)).stem


def _load_envelope(cfg: Config, backend: str):
    return load_env(
        cfg.env.env_setup,
        backend,
        reward=cfg.env.reward,
        variant=cfg.env.variant,
        disruption_penalty=cfg.env.disruption_penalty,
    )


def _largest_divisor_at_most(value: int, limit: int) -> int:
    if value <= 0 or limit <= 0:
        raise ValueError("value and limit must be positive")
    for candidate in range(min(value, limit), 0, -1):
        if value % candidate == 0:
            return candidate
    raise AssertionError("one divides every positive integer")


def _training_key_bank(
    optimizer_seed_keys: jax.Array,
    update_or_pass: int,
    num_rollouts: int,
) -> jax.Array:
    """Return ``(num_seeds, num_rollouts)`` stable training keys."""

    def for_seed(key):
        return indexed_keys(jax.random.fold_in(key, update_or_pass), num_rollouts)

    return jax.vmap(for_seed)(optimizer_seed_keys)


def _broadcast_eval_keys(
    eval_seed: int,
    num_seeds: int,
    eval_n_envs: int,
) -> jax.Array:
    keys = indexed_keys(jax.random.PRNGKey(eval_seed), eval_n_envs)
    return jnp.broadcast_to(keys, (num_seeds, *keys.shape))


def _to_typed_key_bank(keys: jax.Array) -> jax.Array:
    """Convert a ``(seeds, rollouts, 2)`` legacy key bank for Envelope."""
    return jax.vmap(jax.vmap(to_typed_key))(keys)


def _select_seed_pytrees(mask: np.ndarray, new: Any, old: Any) -> Any:
    mask_array = jnp.asarray(mask)

    def select(new_leaf, old_leaf):
        shape = (mask_array.shape[0],) + (1,) * (new_leaf.ndim - 1)
        return jnp.where(mask_array.reshape(shape), new_leaf, old_leaf)

    return jax.tree.map(select, new, old)


def _log_evaluation(
    logger: SeedBufferLogger,
    train_steps: int,
    returns: np.ndarray,
    lengths: np.ndarray,
    best_returns: np.ndarray,
    *,
    grad_norm: np.ndarray | None = None,
    grads_finite: np.ndarray | None = None,
    update_scale: np.ndarray | None = None,
) -> None:
    metrics = {
        "evaluation/return_mean": returns.mean(axis=1),
        "evaluation/return_std": returns.std(axis=1),
        "evaluation/return_min": returns.min(axis=1),
        "evaluation/return_max": returns.max(axis=1),
        "evaluation/episode_length_mean": lengths.mean(axis=1),
        "evaluation/best_return_mean": best_returns,
    }
    metrics["train/grad_norm"] = (
        np.zeros(logger.num_seeds) if grad_norm is None else grad_norm
    )
    metrics["train/grads_finite"] = (
        np.ones(logger.num_seeds) if grads_finite is None else grads_finite
    )
    metrics["train/update_scale"] = (
        np.ones(logger.num_seeds) if update_scale is None else update_scale
    )
    logger.log_batch(train_steps, metrics)


def _make_optimizer(
    learning_rate: float,
    grad_clip: float,
    *,
    policy: bool,
) -> optax.GradientTransformation:
    transforms: list[optax.GradientTransformation] = []
    if grad_clip > 0:
        transforms.append(optax.clip_by_global_norm(grad_clip))
    if policy:
        # SHAC's robust general setting for direct simulator policy gradients.
        transforms.append(optax.adam(learning_rate, b1=0.7, b2=0.95))
    else:
        transforms.append(optax.adam(learning_rate))
    return optax.chain(*transforms)


def _run_policy(
    cfg: Config,
    logger: SeedBufferLogger,
    env,
    episode_steps: int,
) -> tuple[Any, dict[str, Any]]:
    action_space = env.action_space
    if not isinstance(action_space, Continuous):
        raise ValueError("direct pathwise policy gradients require continuous actions")
    num_rollouts = _num_rollouts(cfg)
    gradient_horizon = _largest_divisor_at_most(
        episode_steps,
        cfg.direct.gradient_horizon,
    )
    chunks_per_pass = episode_steps // gradient_horizon
    optimizer_seed_keys = seed_keys(cfg.seed, cfg.num_seeds)
    eval_keys = _to_typed_key_bank(
        _broadcast_eval_keys(
            cfg.env.eval_seed,
            cfg.num_seeds,
            cfg.env.eval_n_envs,
        )
    )

    setpoint_key = jax.random.fold_in(jax.random.key(cfg.seed), 0x5E7)
    _, setpoint_info = env.init(setpoint_key)
    # Keep boundary setpoints infinitesimally inside the action box. SPARC's
    # ECCD source starts at exactly zero power, where the derivative with
    # respect to its independently controlled deposition radius is undefined.
    # The 1e-3 normalized inset is only 5 kW on its 10 MW range.
    action_setpoint = jnp.tanh(_setpoint_theta_row(env, setpoint_key))
    policy = ResidualPolicy(
        action_dim=action_space.shape[0],
        hidden_sizes=cfg.direct.hidden_sizes,
    )
    dummy_obs = jnp.zeros(
        setpoint_info.obs.shape,
        dtype=jnp.float32,
    )
    init_keys = jax.vmap(lambda key: jax.random.fold_in(key, 0x1A17))(
        optimizer_seed_keys
    )
    params = jax.vmap(lambda key: policy.init(key, dummy_obs)["params"])(init_keys)
    optimizer = _make_optimizer(
        cfg.direct.policy_learning_rate,
        cfg.direct.grad_clip,
        policy=True,
    )
    opt_states = jax.vmap(optimizer.init)(params)
    rollback_params = params
    rollback_opt_states = opt_states
    update_scales = jnp.ones((cfg.num_seeds,), dtype=jnp.float32)

    chunk = _make_native_policy_chunk(
        env,
        policy,
        action_setpoint,
        gradient_horizon,
        remat=cfg.direct.remat,
    )
    full_chunk = _make_native_policy_chunk(
        env,
        policy,
        action_setpoint,
        episode_steps,
        remat=cfg.direct.remat,
    )
    full_objective = _make_native_policy_objective(full_chunk)

    def single_chunk_loss(current_params, carry):
        rollout_return, (next_carry, trajectory) = chunk.run(current_params, carry)
        return -rollout_return, ChunkAux(
            next_carry=next_carry,
            rollout_return=rollout_return,
            alive_steps=jnp.sum(trajectory.alive),
        )

    per_rollout_value_and_grad = jax.vmap(
        jax.value_and_grad(single_chunk_loss, has_aux=True),
        in_axes=(None, 0),
    )

    def update_seed(
        current_params,
        current_opt_state,
        current_rollback_params,
        current_rollback_opt_state,
        current_update_scale,
        carries,
    ):
        (losses_and_aux, per_rollout_grads) = per_rollout_value_and_grad(
            current_params,
            carries,
        )
        losses, aux = losses_and_aux
        grads = jax.tree.map(lambda value: jnp.mean(value, axis=0), per_rollout_grads)
        (
            next_params,
            next_opt_state,
            next_rollback_params,
            next_rollback_opt_state,
            next_update_scale,
            finite,
        ) = apply_updates_with_backoff(
            optimizer,
            grads,
            current_opt_state,
            current_params,
            current_rollback_opt_state,
            current_rollback_params,
            current_update_scale,
            backoff_factor=cfg.direct.nonfinite_backoff_factor,
            min_update_scale=cfg.direct.min_update_scale,
        )
        return (
            next_params,
            next_opt_state,
            next_rollback_params,
            next_rollback_opt_state,
            next_update_scale,
            aux.next_carry,
            jnp.mean(losses),
            jnp.mean(aux.rollout_return),
            optax.global_norm(grads),
            finite,
        )

    update = jax.jit(jax.vmap(update_seed))
    initialize_carries = jax.jit(jax.vmap(jax.vmap(chunk.initialize)))

    def evaluate_seed(current_params, keys):
        _, metrics = full_objective.evaluate(current_params, keys)
        return metrics

    evaluate = jax.jit(jax.vmap(evaluate_seed))

    def run_evaluation(current_params):
        metrics: BatchMetrics = evaluate(current_params, eval_keys)
        jax.block_until_ready(metrics.returns)
        return np.asarray(metrics.returns), np.asarray(metrics.alive_steps)

    returns, lengths = run_evaluation(params)
    best_returns = returns.mean(axis=1)
    best_params = params
    _log_evaluation(logger, 0, returns, lengths, best_returns)

    train_steps = 0
    update_index = 0
    pass_index = 0
    next_eval = cfg.direct.eval_freq
    last_grad_norm = np.full(cfg.num_seeds, np.nan)
    last_grads_finite = np.ones(cfg.num_seeds)
    last_update_scale = np.ones(cfg.num_seeds)
    start = time.monotonic()
    while train_steps < cfg.direct.total_timesteps:
        rollout_keys = _to_typed_key_bank(
            _training_key_bank(
                optimizer_seed_keys,
                pass_index,
                num_rollouts,
            )
        )
        carries = initialize_carries(rollout_keys)
        for _ in range(chunks_per_pass):
            if train_steps >= cfg.direct.total_timesteps:
                break
            (
                params,
                opt_states,
                rollback_params,
                rollback_opt_states,
                update_scales,
                carries,
                _,
                _,
                grad_norm,
                grads_finite,
            ) = update(
                params,
                opt_states,
                rollback_params,
                rollback_opt_states,
                update_scales,
                carries,
            )
            jax.block_until_ready(grad_norm)
            carries = jax.tree.map(jax.lax.stop_gradient, carries)
            update_index += 1
            train_steps += gradient_horizon * num_rollouts
            last_grad_norm = np.asarray(grad_norm)
            last_grads_finite = np.asarray(grads_finite, dtype=np.float64)
            last_update_scale = np.asarray(update_scales, dtype=np.float64)
            should_evaluate = (
                train_steps >= next_eval or train_steps >= cfg.direct.total_timesteps
            )
            if should_evaluate:
                returns, lengths = run_evaluation(params)
                current_returns = returns.mean(axis=1)
                improved = current_returns > best_returns
                best_params = _select_seed_pytrees(improved, params, best_params)
                best_returns = np.maximum(best_returns, current_returns)
                _log_evaluation(
                    logger,
                    train_steps,
                    returns,
                    lengths,
                    best_returns,
                    grad_norm=last_grad_norm,
                    grads_finite=last_grads_finite,
                    update_scale=last_update_scale,
                )
                while next_eval <= train_steps:
                    next_eval += cfg.direct.eval_freq
        pass_index += 1

    return best_params, {
        "episode_steps": episode_steps,
        "gradient_horizon": gradient_horizon,
        "num_rollouts": num_rollouts,
        "optimizer_updates": update_index,
        "actual_train_steps": train_steps,
        "train_seconds": time.monotonic() - start,
        "best_returns": best_returns,
        "action_setpoint": np.asarray(action_setpoint),
        "min_update_scale": float(np.min(last_update_scale)),
    }


def _make_knot_parameterization(env, key, episode_steps: int, requested_knots: int):
    """Build a schedule with no more controls than simulator transitions."""
    effective_knots = min(requested_knots, episode_steps)
    theta_row = _setpoint_theta_row(env, key)
    theta0 = jnp.broadcast_to(theta_row, (effective_knots, theta_row.shape[0]))
    if effective_knots == 1:

        def to_actions(theta):
            return jnp.broadcast_to(jnp.tanh(theta[0]), (episode_steps, theta.shape[1]))

        return to_actions, theta0, effective_knots

    knot_positions = jnp.linspace(0.0, episode_steps - 1, effective_knots)
    step_positions = jnp.arange(episode_steps, dtype=knot_positions.dtype)

    def to_actions(theta):
        interpolate_actuator = jax.vmap(
            lambda values: jnp.interp(step_positions, knot_positions, values),
            in_axes=1,
            out_axes=1,
        )
        return jnp.tanh(interpolate_actuator(theta))

    return to_actions, theta0, effective_knots


def _run_knots(
    cfg: Config,
    logger: SeedBufferLogger,
    env,
    episode_steps: int,
) -> tuple[jax.Array, dict[str, Any]]:
    requested_knots = _knots(cfg.algorithm)
    num_rollouts = _num_rollouts(cfg)
    gradient_horizon = _largest_divisor_at_most(
        episode_steps,
        cfg.direct.gradient_horizon,
    )
    chunks_per_pass = episode_steps // gradient_horizon
    optimizer_seed_keys = seed_keys(cfg.seed, cfg.num_seeds)
    parameterization_key = jax.random.fold_in(jax.random.key(cfg.seed), 0x5E7)
    to_actions, theta0, effective_knots = _make_knot_parameterization(
        env,
        parameterization_key,
        episode_steps,
        requested_knots,
    )
    theta = jnp.broadcast_to(theta0, (cfg.num_seeds, *theta0.shape))
    optimizer = _make_optimizer(
        cfg.direct.knot_learning_rate,
        cfg.direct.grad_clip,
        policy=False,
    )
    opt_states = jax.vmap(optimizer.init)(theta)
    rollback_theta = theta
    rollback_opt_states = opt_states
    update_scales = jnp.ones((cfg.num_seeds,), dtype=jnp.float32)
    evaluation_rollout = _make_rollout(env, False)
    chunk = make_knot_chunk(
        env,
        to_actions,
        gradient_horizon,
        remat=cfg.direct.remat,
    )

    def single_chunk_loss(current_theta, carry, start_step):
        rollout_return, (next_carry, trajectory) = chunk.run(
            current_theta,
            carry,
            start_step,
        )
        return -rollout_return, ChunkAux(
            next_carry=next_carry,
            rollout_return=rollout_return,
            alive_steps=jnp.sum(trajectory.alive),
        )

    # TORAX's adaptive-loop custom VJP supports vmap(grad(single rollout)), not
    # grad(vmap(rollout)). Keep value_and_grad inside both rollout and seed vmaps.
    per_rollout_value_and_grad = jax.vmap(
        jax.value_and_grad(single_chunk_loss, has_aux=True),
        in_axes=(None, 0, None),
    )

    def update_seed(
        current_theta,
        current_opt_state,
        current_rollback_theta,
        current_rollback_opt_state,
        current_update_scale,
        carries,
        start_step,
    ):
        (losses_and_aux, per_rollout_grads) = per_rollout_value_and_grad(
            current_theta,
            carries,
            start_step,
        )
        losses, aux = losses_and_aux
        grads = jax.tree.map(lambda value: jnp.mean(value, axis=0), per_rollout_grads)
        (
            next_theta,
            next_opt_state,
            next_rollback_theta,
            next_rollback_opt_state,
            next_update_scale,
            finite,
        ) = apply_updates_with_backoff(
            optimizer,
            grads,
            current_opt_state,
            current_theta,
            current_rollback_opt_state,
            current_rollback_theta,
            current_update_scale,
            backoff_factor=cfg.direct.nonfinite_backoff_factor,
            min_update_scale=cfg.direct.min_update_scale,
        )
        return (
            next_theta,
            next_opt_state,
            next_rollback_theta,
            next_rollback_opt_state,
            next_update_scale,
            aux.next_carry,
            jnp.mean(losses),
            jnp.mean(aux.rollout_return),
            jnp.mean(aux.alive_steps),
            optax.global_norm(grads),
            finite,
        )

    update = jax.jit(jax.vmap(update_seed, in_axes=(0, 0, 0, 0, 0, 0, None)))
    initialize_carries = jax.jit(jax.vmap(jax.vmap(chunk.initialize)))
    eval_keys = _to_typed_key_bank(
        _broadcast_eval_keys(
            cfg.env.eval_seed,
            cfg.num_seeds,
            cfg.env.eval_n_envs,
        )
    )

    def evaluate_one(current_theta, key):
        _, aux = evaluation_rollout(to_actions(current_theta), key)
        return aux["raw_cum"], jnp.sum(aux["valid"])

    evaluate = jax.jit(jax.vmap(jax.vmap(evaluate_one, in_axes=(None, 0))))

    def run_evaluation(current_theta):
        returns, lengths = evaluate(current_theta, eval_keys)
        jax.block_until_ready(returns)
        return np.asarray(returns), np.asarray(lengths)

    returns, lengths = run_evaluation(theta)
    best_returns = returns.mean(axis=1)
    best_theta = theta
    _log_evaluation(logger, 0, returns, lengths, best_returns)

    train_steps = 0
    update_index = 0
    pass_index = 0
    next_eval = cfg.direct.eval_freq
    last_grad_norm = np.full(cfg.num_seeds, np.nan)
    last_grads_finite = np.ones(cfg.num_seeds)
    last_update_scale = np.ones(cfg.num_seeds)
    start = time.monotonic()
    while train_steps < cfg.direct.total_timesteps:
        rollout_keys = _to_typed_key_bank(
            _training_key_bank(
                optimizer_seed_keys,
                pass_index,
                num_rollouts,
            )
        )
        carries = initialize_carries(rollout_keys)
        for chunk_index in range(chunks_per_pass):
            if train_steps >= cfg.direct.total_timesteps:
                break
            start_step = jnp.asarray(chunk_index * gradient_horizon)
            (
                theta,
                opt_states,
                rollback_theta,
                rollback_opt_states,
                update_scales,
                carries,
                _,
                _,
                _,
                grad_norm,
                grads_finite,
            ) = update(
                theta,
                opt_states,
                rollback_theta,
                rollback_opt_states,
                update_scales,
                carries,
                start_step,
            )
            jax.block_until_ready(grad_norm)
            carries = jax.tree.map(jax.lax.stop_gradient, carries)
            update_index += 1
            train_steps += gradient_horizon * num_rollouts
            last_grad_norm = np.asarray(grad_norm)
            last_grads_finite = np.asarray(grads_finite, dtype=np.float64)
            last_update_scale = np.asarray(update_scales, dtype=np.float64)
            should_evaluate = (
                train_steps >= next_eval or train_steps >= cfg.direct.total_timesteps
            )
            if should_evaluate:
                returns, lengths = run_evaluation(theta)
                current_returns = returns.mean(axis=1)
                improved = current_returns > best_returns
                best_theta = _select_seed_pytrees(improved, theta, best_theta)
                best_returns = np.maximum(best_returns, current_returns)
                _log_evaluation(
                    logger,
                    train_steps,
                    returns,
                    lengths,
                    best_returns,
                    grad_norm=last_grad_norm,
                    grads_finite=last_grads_finite,
                    update_scale=last_update_scale,
                )
                while next_eval <= train_steps:
                    next_eval += cfg.direct.eval_freq
        pass_index += 1

    return best_theta, {
        "episode_steps": episode_steps,
        "gradient_horizon": gradient_horizon,
        "requested_knots": requested_knots,
        "effective_knots": effective_knots,
        "num_rollouts": num_rollouts,
        "optimizer_updates": update_index,
        "actual_train_steps": train_steps,
        "train_seconds": time.monotonic() - start,
        "best_returns": best_returns,
        "min_update_scale": float(np.min(last_update_scale)),
    }


def _evaluate_policy_on_env(
    cfg: Config,
    env,
    parameters: Any,
    episode_steps: int,
    eval_keys: jax.Array,
    action_setpoint: jax.Array,
) -> tuple[np.ndarray, np.ndarray]:
    current_obs_dim = env.observation_space.shape[0]
    expected_obs_dim = parameters["Dense_0"]["kernel"].shape[-2]
    policy_obs_indices: tuple[int, ...] | None = None
    if current_obs_dim != expected_obs_dim:
        if current_obs_dim != expected_obs_dim + 1:
            raise ValueError(
                "policy checkpoint observation width is incompatible with the env: "
                f"checkpoint={expected_obs_dim}, env={current_obs_dim}"
            )
        ip_slice = env.obs_layout().slice_of("Ip")
        if ip_slice.stop - ip_slice.start != 1:
            raise ValueError(f"expected scalar Ip observation, got {ip_slice}")
        policy_obs_indices = tuple(
            index for index in range(current_obs_dim) if index != ip_slice.start
        )
        print(
            "Adapting pre-Ip policy checkpoint to current observations by "
            f"dropping Ip at index {ip_slice.start}.",
            flush=True,
        )
    policy = ResidualPolicy(
        action_dim=env.action_space.shape[0],
        hidden_sizes=cfg.direct.hidden_sizes,
    )
    chunk = _make_native_policy_chunk(
        env,
        policy,
        action_setpoint,
        episode_steps,
        remat=cfg.direct.remat,
        policy_obs_indices=policy_obs_indices,
    )
    objective = _make_native_policy_objective(chunk)

    def evaluate_seed(current_params, keys):
        _, metrics = objective.evaluate(current_params, keys)
        return metrics.returns, metrics.alive_steps

    returns, lengths = jax.jit(jax.vmap(evaluate_seed))(parameters, eval_keys)
    jax.block_until_ready(returns)
    return np.asarray(returns), np.asarray(lengths)


def _evaluate_knots_on_env(
    cfg: Config,
    env,
    parameters: jax.Array,
    episode_steps: int,
    eval_keys: jax.Array,
) -> tuple[np.ndarray, np.ndarray]:
    parameterization_key = jax.random.fold_in(jax.random.key(cfg.seed), 0x5E7)
    to_actions, _, _ = _make_knot_parameterization(
        env,
        parameterization_key,
        episode_steps,
        _knots(cfg.algorithm),
    )
    rollout = _make_rollout(env, False)

    def evaluate_one(current_theta, key):
        _, aux = rollout(to_actions(current_theta), key)
        return aux["raw_cum"], jnp.sum(aux["valid"])

    evaluate = jax.jit(jax.vmap(jax.vmap(evaluate_one, in_axes=(None, 0))))
    returns, lengths = evaluate(parameters, eval_keys)
    jax.block_until_ready(returns)
    return np.asarray(returns), np.asarray(lengths)


def _evaluate_transfer(
    cfg: Config,
    source_env,
    parameters: Any,
    summary: dict[str, Any],
) -> dict[str, float | str]:
    if cfg.env.transfer_backend is None:
        raise ValueError("transfer_backend is required for transfer evaluation")

    source_backend = _backend_name(cfg.env.backend)
    target_backend = _backend_name(cfg.env.transfer_backend)
    print(f"Loading transfer env ({target_backend})...", flush=True)
    target_env = _load_envelope(cfg, cfg.env.transfer_backend)
    target_episode_steps = EnvelopeGymnax(
        target_env
    ).default_params.max_steps_in_episode
    source_episode_steps = int(summary["episode_steps"])
    if target_episode_steps != source_episode_steps:
        raise ValueError(
            "source and target backends must have the same episode horizon; "
            f"got {source_episode_steps} and {target_episode_steps}"
        )

    eval_keys = _to_typed_key_bank(
        _broadcast_eval_keys(
            cfg.env.eval_seed,
            cfg.num_seeds,
            cfg.env.transfer_n_envs,
        )
    )
    start = time.monotonic()
    if cfg.algorithm == "direct_policy":
        action_setpoint = jnp.asarray(summary["action_setpoint"])
        source_returns, source_lengths = _evaluate_policy_on_env(
            cfg,
            source_env,
            parameters,
            source_episode_steps,
            eval_keys,
            action_setpoint,
        )
        target_returns, target_lengths = _evaluate_policy_on_env(
            cfg,
            target_env,
            parameters,
            target_episode_steps,
            eval_keys,
            action_setpoint,
        )
    else:
        source_returns, source_lengths = _evaluate_knots_on_env(
            cfg,
            source_env,
            parameters,
            source_episode_steps,
            eval_keys,
        )
        target_returns, target_lengths = _evaluate_knots_on_env(
            cfg,
            target_env,
            parameters,
            target_episode_steps,
            eval_keys,
        )
    eval_seconds = time.monotonic() - start
    metrics = transfer_metrics(
        source_backend,
        target_backend,
        source_returns,
        source_lengths,
        target_returns,
        target_lengths,
        eval_seconds,
    )
    print(
        f"Transfer {source_backend} -> {target_backend}: "
        f"source return {metrics['transfer/source_return_mean']:.3f}, "
        f"target return {metrics['transfer/target_return_mean']:.3f} "
        f"(ratio {metrics['transfer/return_ratio']:.3f})",
        flush=True,
    )
    return metrics


def _save_checkpoint(
    out_dir: str | None,
    run_name: str,
    cfg: Config,
    parameters: Any,
    summary: dict[str, Any],
) -> None:
    if out_dir is None:
        return
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "config_json": np.asarray(json.dumps(dataclasses.asdict(cfg), sort_keys=True)),
        "summary_json": np.asarray(
            json.dumps(
                summary,
                sort_keys=True,
                default=lambda value: np.asarray(value).tolist(),
            )
        ),
    }
    if cfg.algorithm == "direct_policy":
        payload["policy_params"] = np.frombuffer(
            serialization.to_bytes(parameters),
            dtype=np.uint8,
        )
    else:
        payload["theta"] = np.asarray(parameters)
    checkpoint = out / f"{run_name}_checkpoint.npz"
    with checkpoint.open("wb") as checkpoint_file:
        np.savez(checkpoint_file, **payload)
    print(f"Saved checkpoint: {checkpoint}")


def main(cfg: Config) -> None:
    if cfg.num_seeds <= 0:
        raise ValueError("num_seeds must be positive")
    if cfg.direct.total_timesteps <= 0 or cfg.direct.eval_freq <= 0:
        raise ValueError("total_timesteps and eval_freq must be positive")
    if not 0.0 < cfg.direct.nonfinite_backoff_factor < 1.0:
        raise ValueError("nonfinite_backoff_factor must be between zero and one")
    if not 0.0 < cfg.direct.min_update_scale <= 1.0:
        raise ValueError("min_update_scale must be in (0, 1]")
    if cfg.strict_phase_reward:
        validate_reward(cfg.env.env_setup, cfg.env.reward, cfg.env.backend)

    run_name = run_slug(
        cfg.algorithm,
        cfg.env.env_setup,
        cfg.env.backend,
        cfg.env.variant,
        cfg.env.reward,
        cfg.num_seeds,
    )
    if cfg.env.transfer_backend is not None:
        run_name = f"{run_name}-to-{_backend_name(cfg.env.transfer_backend)}"
    logger = SeedBufferLogger(
        num_seeds=cfg.num_seeds,
        seed_ids=tuple(range(cfg.seed, cfg.seed + cfg.num_seeds)),
        run_name=run_name,
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        group=cfg.wandb.group,
        mode=cfg.wandb.mode,
        config=dataclasses.asdict(cfg),
        out_dir=cfg.history_dir,
        job_type=cfg.algorithm,
        tags=(cfg.study, cfg.algorithm, *cfg.wandb.tags),
    )
    envelope_env = _load_envelope(cfg, cfg.env.backend)
    gymnax_env = EnvelopeGymnax(envelope_env)
    episode_steps = gymnax_env.default_params.max_steps_in_episode
    logger.start_time = time.time()

    if cfg.algorithm == "direct_policy":
        parameters, summary = _run_policy(
            cfg,
            logger,
            envelope_env,
            episode_steps,
        )
    else:
        parameters, summary = _run_knots(
            cfg,
            logger,
            envelope_env,
            episode_steps,
        )
    transfer_summary: dict[str, float | str] = {}
    if cfg.env.transfer_backend is not None:
        transfer_summary = _evaluate_transfer(
            cfg,
            envelope_env,
            parameters,
            summary,
        )
        summary.update(transfer_summary)
        write_transfer_summary(cfg.history_dir, run_name, transfer_summary)
    logger.log_once(
        {
            "run/episode_steps": summary["episode_steps"],
            "run/num_rollouts": summary["num_rollouts"],
            "run/optimizer_updates": summary["optimizer_updates"],
            "run/actual_train_steps": summary["actual_train_steps"],
            "run/min_update_scale": summary["min_update_scale"],
            "time/train_s": summary["train_seconds"],
            **(
                {"run/gradient_horizon": summary["gradient_horizon"]}
                if cfg.algorithm == "direct_policy"
                else {
                    "run/gradient_horizon": summary["gradient_horizon"],
                    "run/requested_knots": summary["requested_knots"],
                    "run/effective_knots": summary["effective_knots"],
                }
            ),
            **transfer_summary,
        }
    )
    _save_checkpoint(cfg.history_dir, run_name, cfg, parameters, summary)
    logger.finish()


if __name__ == "__main__":
    main(tyro.cli(Config))
