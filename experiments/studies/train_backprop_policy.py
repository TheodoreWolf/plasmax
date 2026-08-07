"""Train a feedback policy by differentiating reward through TORAX.

Unlike PPO, this uses the simulator's pathwise derivative directly: a
deterministic policy acts on each observation, ``jax.lax.scan`` rolls the
closed loop forward, and reverse-mode AD differentiates cumulative reward
through the complete simulator-policy trajectory.

The policy is initialized as a zero residual around the environment's reset
actuator setpoint. This matters because TORAX rate-limits each physical action
around the previous action; a generic zero action can start outside that window
and receive an exactly zero gradient.

``--rollout-steps`` controls the finite differentiated rollout horizon. It is
deliberately separate from the discharge length configured by the environment
YAML. Rewards after the first terminal transition are masked because
The ``EnvelopeGymnax`` compatibility adapter auto-resets on ``done``.

Set ``--truncation-steps`` to apply online truncated backpropagation through
time. The simulator state is carried through the complete rollout, but one Adam
update is applied after each chunk and the state is detached before the next
chunk. For example, a 4,400-step rollout with 100-step truncation applies 44
updates per training pass while no gradient spans more than 100 simulator steps.

Examples::

    uv run python experiments/studies/train_backprop_policy.py
    uv run python experiments/studies/train_backprop_policy.py \
        --env iter/hybrid/flattop --backend bohm_gyrobohm --rollout-steps 100
    uv run python experiments/studies/train_backprop_policy.py \
        --rollout-steps 50 --num-rollouts 4 --iters 100 --wandb
    uv run python experiments/studies/train_backprop_policy.py \
        --rollout-steps 4400 --truncation-steps 100 --iters 1

Direct gradients require continuous actions. Quantized ``MultiDiscrete``
actions, hard disruption thresholds, and actuator clipping are not smooth; use
a continuous environment and a phase reward with smooth safety barriers when
those boundaries matter.
"""

from __future__ import annotations

import dataclasses
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
import tyro
from flax import linen as nn
from flax import serialization
from gymnax.environments.spaces import Box

from plasmax.environment import factory as scenario_config
from plasmax.wrappers import unwrap_to_env_state
from scripts.project_paths import wandb_dir
from training.envelope_gymnax import EnvelopeGymnax


@dataclasses.dataclass
class Args:
    env: str = "iter/hybrid/flattop"
    backend: str = "bohm_gyrobohm"
    reward: str | None = None
    variant: Literal["oracle", "realistic"] = "realistic"
    # Omitted values inherit the task YAML; explicit zero disables the penalty.
    disruption_penalty: float | None = None

    # Static differentiated rollout horizon. This does not alter t_final.
    rollout_steps: int = 50
    # TBPTT chunk length. Zero differentiates the complete rollout at once.
    truncation_steps: int = 0
    # Independent training trajectories, freshly sampled each training pass.
    num_rollouts: int = 1
    # Fixed held-out trajectories used for frozen full-rollout evaluation.
    eval_rollouts: int = 1
    # Independent base seed for the fixed held-out evaluation trajectories.
    eval_seed: int = 10_000
    # Evaluate the frozen policy every N completed training passes.
    eval_every_passes: int = 1
    # Number of training passes. Without truncation, each pass is one update.
    iters: int = 30
    # Adam step size.
    learning_rate: float = 1e-3
    # Width of each hidden MLP layer; pass no values for a linear policy.
    hidden_sizes: tuple[int, ...] = (64, 64)
    # Global gradient-norm clip; set to 0 to disable.
    grad_clip: float = 1.0
    # Base seed for initialization and the stream of training rollout keys.
    seed: int = 0
    # Rematerialize each simulator step during the backward pass to save memory.
    remat: bool = True

    out: str = "outputs/backprop_policy.npz"
    wandb: bool = False
    wandb_project: str = "plasmax"
    wandb_entity: str = "flair"
    wandb_group: str = "debug"
    wandb_mode: Literal["online", "offline"] = "online"


class ResidualPolicy(nn.Module):
    """MLP producing a normalized-action residual around the reset setpoint."""

    action_dim: int
    hidden_sizes: tuple[int, ...]

    @nn.compact
    def __call__(self, obs: jax.Array) -> jax.Array:
        x = obs
        for width in self.hidden_sizes:
            x = nn.swish(nn.Dense(width)(x))
        return nn.Dense(
            self.action_dim,
            kernel_init=nn.initializers.zeros_init(),
            bias_init=nn.initializers.zeros_init(),
            name="action_residual",
        )(x)


class RolloutStep(NamedTuple):
    """Per-step outputs stacked by the differentiated rollout."""

    reward: jax.Array
    action: jax.Array
    done: jax.Array
    alive: jax.Array


class RolloutCarry(NamedTuple):
    """Differentiable simulator carry propagated between rollout chunks."""

    obs: jax.Array
    env_state: Any
    key: jax.Array
    alive: jax.Array


class ChunkOutput(NamedTuple):
    """Final carry and transitions emitted by one rollout chunk."""

    carry: RolloutCarry
    trajectory: RolloutStep


class BatchMetrics(NamedTuple):
    """Per-rollout summaries returned alongside the scalar training loss."""

    returns: jax.Array
    alive_steps: jax.Array


@dataclasses.dataclass(frozen=True)
class PolicyObjective:
    """Batched policy objective and its explicitly per-rollout gradient."""

    evaluate: Callable[[Any, jax.Array], tuple[jax.Array, BatchMetrics]]
    value_and_grad: Callable[
        [Any, jax.Array],
        tuple[tuple[jax.Array, BatchMetrics], Any],
    ]


@dataclasses.dataclass(frozen=True)
class PolicyChunk:
    """Initializer and fixed-length chunk runner for online TBPTT."""

    initialize: Callable[[jax.Array], RolloutCarry]
    run: Callable[[Any, RolloutCarry], tuple[jax.Array, ChunkOutput]]


@dataclasses.dataclass
class OptimizationResult:
    """Final policy, host-side training history, and best-eval diagnostics."""

    params: Any
    history: list[dict[str, float | int]]
    baseline_return: float
    best_return: float
    best_iter: int
    final_return: float
    final_return_std: float


def policy_action(
    policy: ResidualPolicy,
    params: Any,
    action_setpoint: jax.Array,
    obs: jax.Array,
) -> jax.Array:
    """Applies a normalized-action residual around the exact reset setpoint."""
    residual = policy.apply({"params": params}, obs)
    return jnp.clip(action_setpoint + residual, -1.0, 1.0)


def initial_action_setpoint(env: Any, env_params: Any, key: jax.Array) -> jax.Array:
    """Returns a differentiable normalized reset actuator setpoint.

    Boundary values are inset by 1e-3. In particular, a zero-power Gaussian
    source has an undefined sensitivity to its independently controlled
    deposition location; the negligible inset keeps that paired gradient
    finite without materially changing the operating point.
    """
    _, state = env.reset(key, env_params)
    state = unwrap_to_env_state(state)
    setpoint = jnp.asarray(
        env.envelope_env.from_physical(state.prev_action), dtype=jnp.float32
    )
    return jnp.clip(setpoint, -0.999, 0.999)


def make_policy_chunk(
    env: Any,
    env_params: Any,
    policy: ResidualPolicy,
    action_setpoint: jax.Array,
    chunk_steps: int,
    *,
    remat: bool,
) -> PolicyChunk:
    """Builds reset and fixed-length continuation functions for a rollout."""

    def step(
        params: Any,
        carry: RolloutCarry,
        _: None,
    ) -> tuple[RolloutCarry, RolloutStep]:
        obs, env_state, key, alive = carry
        key, step_key = jax.random.split(key)
        action = policy_action(policy, params, action_setpoint, obs)
        next_obs, next_state, reward, done, _ = env.step(
            step_key, env_state, action, env_params
        )
        masked_reward = jnp.where(alive, reward, jnp.zeros_like(reward))
        transition = RolloutStep(
            reward=masked_reward,
            action=action,
            done=done,
            alive=alive,
        )
        next_carry = RolloutCarry(
            obs=next_obs,
            env_state=next_state,
            key=key,
            alive=alive & ~done,
        )
        return next_carry, transition

    scan_step = jax.checkpoint(step) if remat else step

    def initialize(key: jax.Array) -> RolloutCarry:
        reset_key, transition_key = jax.random.split(key)
        obs, env_state = env.reset(reset_key, env_params)
        return RolloutCarry(
            obs=obs,
            env_state=env_state,
            key=transition_key,
            alive=jnp.asarray(True),
        )

    def run(
        params: Any,
        carry: RolloutCarry,
    ) -> tuple[jax.Array, ChunkOutput]:
        next_carry, trajectory = jax.lax.scan(
            lambda c, x: scan_step(params, c, x),
            carry,
            None,
            length=chunk_steps,
        )
        return jnp.sum(trajectory.reward), ChunkOutput(next_carry, trajectory)

    return PolicyChunk(initialize=initialize, run=run)


def make_policy_rollout(
    env: Any,
    env_params: Any,
    policy: ResidualPolicy,
    action_setpoint: jax.Array,
    rollout_steps: int,
    *,
    remat: bool,
) -> Callable[[Any, jax.Array], tuple[jax.Array, RolloutStep]]:
    """Builds a pure single-trajectory rollout with a static scan length."""
    chunk = make_policy_chunk(
        env,
        env_params,
        policy,
        action_setpoint,
        rollout_steps,
        remat=remat,
    )

    def rollout(params: Any, key: jax.Array) -> tuple[jax.Array, RolloutStep]:
        rollout_return, output = chunk.run(params, chunk.initialize(key))
        return rollout_return, output.trajectory

    return rollout


def make_policy_objective(
    rollout: Callable[[Any, jax.Array], tuple[jax.Array, RolloutStep]],
) -> PolicyObjective:
    """Builds a mean-return objective with differentiation inside ``vmap``.

    TORAX's adaptive loop has a custom JVP containing ``lax.cond``. Applying
    reverse mode outside a rollout-level ``vmap`` batches that conditional and
    triggers a missing ``stop_gradient`` transpose rule in JAX 0.10.1. Instead,
    each scalar rollout loss is differentiated first, those independent
    value-and-gradient calls are vectorized, and their gradients are averaged.
    This is mathematically identical to differentiating the mean loss while
    preserving the transform order required by TORAX.
    """

    def single_loss(
        params: Any,
        key: jax.Array,
    ) -> tuple[jax.Array, BatchMetrics]:
        rollout_return, trajectory = rollout(params, key)
        metrics = BatchMetrics(
            returns=rollout_return,
            alive_steps=jnp.sum(trajectory.alive),
        )
        return -rollout_return, metrics

    batch_evaluate = jax.vmap(single_loss, in_axes=(None, 0))
    batch_value_and_grad = jax.vmap(
        jax.value_and_grad(single_loss, has_aux=True),
        in_axes=(None, 0),
    )

    def evaluate(
        params: Any,
        keys: jax.Array,
    ) -> tuple[jax.Array, BatchMetrics]:
        losses, metrics = batch_evaluate(params, keys)
        return jnp.mean(losses), metrics

    def value_and_grad(
        params: Any,
        keys: jax.Array,
    ) -> tuple[tuple[jax.Array, BatchMetrics], Any]:
        (losses, metrics), per_rollout_grads = batch_value_and_grad(params, keys)
        grads = jax.tree.map(
            lambda gradient: jnp.mean(gradient, axis=0),
            per_rollout_grads,
        )
        return (jnp.mean(losses), metrics), grads

    return PolicyObjective(evaluate=evaluate, value_and_grad=value_and_grad)


def _training_rollout_keys(
    training_key: jax.Array,
    training_pass: int,
    num_rollouts: int,
) -> jax.Array:
    """Returns a reproducible fresh rollout-key bank for one training pass."""
    pass_key = jax.random.fold_in(training_key, training_pass)
    return jax.random.split(pass_key, num_rollouts)


def _should_evaluate(
    completed_passes: int,
    total_passes: int,
    eval_every_passes: int,
) -> bool:
    """Returns whether this completed pass is on cadence or is the final pass."""
    return completed_passes % eval_every_passes == 0 or completed_passes == total_passes


def optimize_policy(
    initial_params: Any,
    objective: PolicyObjective,
    training_key: jax.Array,
    evaluation_keys: jax.Array,
    *,
    rollout_steps: int,
    num_rollouts: int,
    iters: int,
    eval_every_passes: int,
    learning_rate: float,
    grad_clip: float,
    on_iter: Callable[[dict[str, float | int]], None] | None = None,
    on_evaluate: Callable[[dict[str, float | int]], None] | None = None,
) -> OptimizationResult:
    """Runs full-rollout pathwise updates and returns the final policy."""
    transforms = []
    if grad_clip > 0:
        transforms.append(optax.clip_by_global_norm(grad_clip))
    transforms.append(optax.adam(learning_rate))
    optimizer = optax.apply_if_finite(
        optax.chain(*transforms),
        max_consecutive_errors=iters + 1,
    )

    params = initial_params
    opt_state = optimizer.init(params)

    @jax.jit
    def update(
        current_params: Any,
        current_opt_state: Any,
        rollout_keys: jax.Array,
    ):
        (loss, metrics), grads = objective.value_and_grad(
            current_params,
            rollout_keys,
        )
        updates, next_opt_state = optimizer.update(
            grads, current_opt_state, current_params
        )
        next_params = optax.apply_updates(current_params, updates)
        grads_finite = jnp.all(
            jnp.stack([jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree.leaves(grads)])
        )
        return (
            next_params,
            next_opt_state,
            loss,
            metrics,
            optax.global_norm(grads),
            grads_finite,
        )

    evaluate_full = jax.jit(
        lambda current_params: objective.evaluate(current_params, evaluation_keys)
    )
    history: list[dict[str, float | int]] = []
    best_return = -np.inf
    best_iter = -1
    best_params = params

    def record_training(
        training_pass: int,
        simulator_steps: int,
        loss: jax.Array,
        metrics: BatchMetrics,
        grad_norm: jax.Array | float,
        grads_finite: jax.Array,
        elapsed: float,
    ) -> None:
        returns = np.asarray(metrics.returns)
        alive_steps = np.asarray(metrics.alive_steps)
        mean_return = float(returns.mean())
        entry: dict[str, float | int] = {
            "iter": training_pass - 1,
            "pass": training_pass,
            "update": training_pass,
            "simulator_steps": simulator_steps,
            "loss": float(loss),
            "return_mean": mean_return,
            "return_std": float(returns.std()),
            "return_min": float(returns.min()),
            "return_max": float(returns.max()),
            "alive_steps_mean": float(alive_steps.mean()),
            "grad_norm": float(grad_norm),
            "grads_finite": int(grads_finite),
            "seconds": elapsed,
        }
        history.append(entry)
        if on_iter is not None:
            on_iter(entry)
        compile_tag = "compile+" if training_pass == 1 else ""
        finite_tag = "" if bool(grads_finite) else "  SKIPPED_NONFINITE_GRAD"
        print(
            f"[{training_pass - 1:3d}] return={mean_return:+.5f} "
            f"(+/- {entry['return_std']:.5f})  |grad|={entry['grad_norm']:.2e} "
            f"alive={entry['alive_steps_mean']:.1f}  "
            f"({compile_tag}{elapsed:.1f}s){finite_tag}"
        )

    def record_evaluation(
        completed_passes: int,
        simulator_steps: int,
    ) -> tuple[float, float]:
        nonlocal best_params, best_return, best_iter
        full_loss, full_metrics = evaluate_full(params)
        jax.block_until_ready(full_loss)
        returns = np.asarray(full_metrics.returns)
        alive_steps = np.asarray(full_metrics.alive_steps)
        evaluation: dict[str, float | int] = {
            "pass": completed_passes,
            "update": completed_passes,
            "simulator_steps": simulator_steps,
            "return_mean": float(returns.mean()),
            "return_std": float(returns.std()),
            "return_min": float(returns.min()),
            "return_max": float(returns.max()),
            "episode_length_mean": float(alive_steps.mean()),
        }
        if on_evaluate is not None:
            on_evaluate(evaluation)
        mean_return = float(evaluation["return_mean"])
        if mean_return > best_return:
            best_return = mean_return
            best_iter = completed_passes
            best_params = params
        print(
            f"[eval pass {completed_passes}] frozen-policy "
            f"return={mean_return:+.5f} steps={simulator_steps}"
        )
        return mean_return, float(evaluation["return_std"])

    final_return, final_return_std = record_evaluation(
        completed_passes=0,
        simulator_steps=0,
    )
    baseline_return = final_return

    for training_pass_index in range(iters):
        rollout_keys = _training_rollout_keys(
            training_key,
            training_pass_index,
            num_rollouts,
        )
        start = time.monotonic()
        params, opt_state, loss, metrics, grad_norm, grads_finite = update(
            params,
            opt_state,
            rollout_keys,
        )
        jax.block_until_ready(loss)
        elapsed = time.monotonic() - start
        completed_passes = training_pass_index + 1
        simulator_steps = completed_passes * rollout_steps * num_rollouts
        record_training(
            completed_passes,
            simulator_steps,
            loss,
            metrics,
            grad_norm,
            grads_finite,
            elapsed,
        )
        if _should_evaluate(completed_passes, iters, eval_every_passes):
            final_return, final_return_std = record_evaluation(
                completed_passes,
                simulator_steps,
            )

    print(
        f"best frozen evaluation (diagnostic): update #{best_iter} "
        f"return={best_return:+.5f} "
        f"(initial={baseline_return:+.5f}, delta={best_return - baseline_return:+.5f})"
    )
    return OptimizationResult(
        params=best_params,
        history=history,
        baseline_return=baseline_return,
        best_return=best_return,
        best_iter=best_iter,
        final_return=final_return,
        final_return_std=final_return_std,
    )


def optimize_policy_truncated(
    initial_params: Any,
    chunk: PolicyChunk,
    full_objective: PolicyObjective,
    training_key: jax.Array,
    evaluation_keys: jax.Array,
    *,
    rollout_steps: int,
    truncation_steps: int,
    num_rollouts: int,
    iters: int,
    eval_every_passes: int,
    learning_rate: float,
    grad_clip: float,
    on_iter: Callable[[dict[str, float | int]], None] | None = None,
    on_evaluate: Callable[[dict[str, float | int]], None] | None = None,
) -> OptimizationResult:
    """Runs online TBPTT and returns the final policy."""
    if rollout_steps % truncation_steps:
        raise ValueError("rollout_steps must be divisible by truncation_steps")
    chunks_per_pass = rollout_steps // truncation_steps
    total_updates = iters * chunks_per_pass

    transforms = []
    if grad_clip > 0:
        transforms.append(optax.clip_by_global_norm(grad_clip))
    transforms.append(optax.adam(learning_rate))
    optimizer = optax.apply_if_finite(
        optax.chain(*transforms),
        max_consecutive_errors=total_updates + 1,
    )

    def single_chunk_loss(
        params: Any,
        carry: RolloutCarry,
    ) -> tuple[jax.Array, tuple[RolloutCarry, BatchMetrics]]:
        chunk_return, output = chunk.run(params, carry)
        metrics = BatchMetrics(
            returns=chunk_return,
            alive_steps=jnp.sum(output.trajectory.alive),
        )
        return -chunk_return, (output.carry, metrics)

    batch_value_and_grad = jax.vmap(
        jax.value_and_grad(single_chunk_loss, has_aux=True),
        in_axes=(None, 0),
    )

    @jax.jit
    def update(
        current_params: Any,
        current_opt_state: Any,
        carries: RolloutCarry,
    ):
        (losses_and_aux, per_rollout_grads) = batch_value_and_grad(
            current_params,
            carries,
        )
        losses, (next_carries, metrics) = losses_and_aux
        grads = jax.tree.map(
            lambda gradient: jnp.mean(gradient, axis=0),
            per_rollout_grads,
        )
        updates, next_opt_state = optimizer.update(
            grads,
            current_opt_state,
            current_params,
        )
        next_params = optax.apply_updates(current_params, updates)
        grads_finite = jnp.all(
            jnp.stack([jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree.leaves(grads)])
        )
        return (
            next_params,
            next_opt_state,
            next_carries,
            jnp.mean(losses),
            metrics,
            optax.global_norm(grads),
            grads_finite,
        )

    initialize = jax.jit(jax.vmap(chunk.initialize))
    evaluate_full = jax.jit(
        lambda current_params: full_objective.evaluate(
            current_params,
            evaluation_keys,
        )
    )

    params = initial_params
    opt_state = optimizer.init(params)
    best_return = -np.inf
    best_iter = -1
    best_params = params
    history: list[dict[str, float | int]] = []
    global_update = 0

    def record_evaluation(
        completed_passes: int,
        simulator_steps: int,
    ) -> tuple[float, float]:
        nonlocal best_params, best_return, best_iter
        full_loss, full_metrics = evaluate_full(params)
        jax.block_until_ready(full_loss)
        returns = np.asarray(full_metrics.returns)
        alive_steps = np.asarray(full_metrics.alive_steps)
        evaluation: dict[str, float | int] = {
            "pass": completed_passes,
            "update": global_update,
            "simulator_steps": simulator_steps,
            "return_mean": float(returns.mean()),
            "return_std": float(returns.std()),
            "return_min": float(returns.min()),
            "return_max": float(returns.max()),
            "episode_length_mean": float(alive_steps.mean()),
        }
        if on_evaluate is not None:
            on_evaluate(evaluation)
        mean_return = float(evaluation["return_mean"])
        if mean_return > best_return:
            best_return = mean_return
            best_iter = global_update
            best_params = params
        print(
            f"[eval pass {completed_passes}] frozen-policy "
            f"return={mean_return:+.5f} steps={simulator_steps}"
        )
        return mean_return, float(evaluation["return_std"])

    final_return, final_return_std = record_evaluation(
        completed_passes=0,
        simulator_steps=0,
    )
    baseline_return = final_return

    for training_pass_index in range(iters):
        rollout_keys = _training_rollout_keys(
            training_key,
            training_pass_index,
            num_rollouts,
        )
        carries = initialize(rollout_keys)
        for chunk_index in range(chunks_per_pass):
            start = time.monotonic()
            (
                params,
                opt_state,
                next_carries,
                loss,
                metrics,
                grad_norm,
                grads_finite,
            ) = update(params, opt_state, carries)
            jax.block_until_ready(loss)
            # The next chunk sees the simulator state as a constant. Keeping this
            # outside single_chunk_loss avoids placing stop_gradient inside the
            # custom-JVP tangent program that reverse mode must transpose.
            carries = jax.tree.map(jax.lax.stop_gradient, next_carries)
            elapsed = time.monotonic() - start
            returns = np.asarray(metrics.returns)
            alive_steps = np.asarray(metrics.alive_steps)
            step_start = chunk_index * truncation_steps
            entry: dict[str, float | int] = {
                "iter": global_update,
                "pass": training_pass_index + 1,
                "update": global_update + 1,
                "chunk": chunk_index,
                "simulator_steps": (
                    (global_update + 1) * truncation_steps * num_rollouts
                ),
                "step_start": step_start,
                "step_end": step_start + truncation_steps,
                "loss": float(loss),
                "return_mean": float(returns.mean()),
                "return_std": float(returns.std()),
                "return_min": float(returns.min()),
                "return_max": float(returns.max()),
                "alive_steps_mean": float(alive_steps.mean()),
                "grad_norm": float(grad_norm),
                "grads_finite": int(grads_finite),
                "seconds": elapsed,
            }
            history.append(entry)
            if on_iter is not None:
                on_iter(entry)
            compile_tag = "compile+" if global_update == 0 else ""
            finite_tag = "" if bool(grads_finite) else "  SKIPPED_NONFINITE_GRAD"
            print(
                f"[{global_update:3d}] steps={step_start:4d}-"
                f"{step_start + truncation_steps:4d} "
                f"return={entry['return_mean']:+.5f} "
                f"|grad|={entry['grad_norm']:.2e} "
                f"alive={entry['alive_steps_mean']:.1f} "
                f"({compile_tag}{elapsed:.1f}s){finite_tag}"
            )
            global_update += 1

        completed_passes = training_pass_index + 1
        simulator_steps = global_update * truncation_steps * num_rollouts
        if _should_evaluate(completed_passes, iters, eval_every_passes):
            final_return, final_return_std = record_evaluation(
                completed_passes,
                simulator_steps,
            )

    print(
        f"best frozen evaluation (diagnostic): update #{best_iter} "
        f"return={best_return:+.5f} "
        f"(initial={baseline_return:+.5f}, delta={best_return - baseline_return:+.5f})"
    )
    return OptimizationResult(
        params=best_params,
        history=history,
        baseline_return=baseline_return,
        best_return=best_return,
        best_iter=best_iter,
        final_return=final_return,
        final_return_std=final_return_std,
    )


def _environment_signature(env: Any) -> dict[str, Any]:
    """Returns the public action/observation contract needed for safe replay."""
    params = env.default_params
    envelope_env = env.envelope_env
    base_env = envelope_env.unwrapped
    signature: dict[str, Any] = {
        "observation_shape": list(env.observation_space(params).shape),
        "action_shape": list(env.action_space(params).shape),
    }
    if hasattr(base_env, "actuator_specs"):
        signature["actuators"] = [
            dataclasses.asdict(spec) for spec in base_env.actuator_specs
        ]
    if hasattr(base_env, "profile_obs_specs"):
        signature["profile_observations"] = [
            dataclasses.asdict(spec) for spec in base_env.profile_obs_specs
        ]
        signature["scalar_observations"] = [
            dataclasses.asdict(spec) for spec in base_env.scalar_obs_specs
        ]
    if hasattr(envelope_env, "obs_layout"):
        layout = envelope_env.obs_layout()
        signature["emitted_profiles"] = list(layout.profile_names)
        signature["emitted_scalars"] = list(layout.scalar_names)
    return json.loads(json.dumps(signature, sort_keys=True))


def save_policy(
    path: str | Path,
    args: Args,
    params: Any,
    action_setpoint: jax.Array,
    trajectory: RolloutStep,
    trajectory_return: float,
    eval_mean_return: float,
    env: Any,
    result: OptimizationResult,
) -> Path:
    """Saves policy parameters, reconstruction metadata, and one rollout."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    params_bytes = np.frombuffer(serialization.to_bytes(params), dtype=np.uint8)
    history = result.history
    with out.open("wb") as checkpoint_file:
        np.savez(
            checkpoint_file,
            policy_params=params_bytes,
            action_setpoint=np.asarray(action_setpoint),
            obs_dim=np.int64(env.observation_space(env.default_params).shape[0]),
            action_dim=np.int64(env.action_space(env.default_params).shape[0]),
            hidden_sizes=np.asarray(args.hidden_sizes, dtype=np.int64),
            config_json=np.asarray(
                json.dumps(dataclasses.asdict(args), sort_keys=True)
            ),
            environment_signature_json=np.asarray(
                json.dumps(_environment_signature(env), sort_keys=True)
            ),
            requested_actions_norm=np.asarray(trajectory.action),
            requested_actions_phys=np.asarray(
                env.envelope_env.to_physical(trajectory.action)
            ),
            rewards=np.asarray(trajectory.reward),
            dones=np.asarray(trajectory.done),
            alive=np.asarray(trajectory.alive),
            trajectory_return=trajectory_return,
            eval_mean_return=eval_mean_return,
            baseline_return=result.baseline_return,
            best_eval_mean_return=result.best_return,
            best_eval_update=result.best_iter,
            history_iter=np.asarray([row["iter"] for row in history]),
            history_return=np.asarray([row["return_mean"] for row in history]),
            history_grad_norm=np.asarray([row["grad_norm"] for row in history]),
        )
    return out


def load_policy(
    path: str | Path,
    env: Any | None = None,
) -> tuple[ResidualPolicy, Any, jax.Array, dict[str, Any]]:
    """Loads a saved policy and optionally validates an environment for replay."""
    with np.load(path, allow_pickle=False) as checkpoint:
        obs_dim = int(checkpoint["obs_dim"].item())
        action_dim = int(checkpoint["action_dim"].item())
        hidden_sizes = tuple(int(x) for x in checkpoint["hidden_sizes"].tolist())
        policy = ResidualPolicy(
            action_dim=action_dim,
            hidden_sizes=hidden_sizes,
        )
        template = policy.init(
            jax.random.PRNGKey(0), jnp.zeros((obs_dim,), dtype=jnp.float32)
        )["params"]
        params = serialization.from_bytes(
            template, np.asarray(checkpoint["policy_params"]).tobytes()
        )
        action_setpoint = jnp.asarray(checkpoint["action_setpoint"], dtype=jnp.float32)
        metadata = {
            "config": json.loads(checkpoint["config_json"].item()),
            "environment_signature": json.loads(
                checkpoint["environment_signature_json"].item()
            ),
        }
    if (
        env is not None
        and _environment_signature(env) != metadata["environment_signature"]
    ):
        raise ValueError(
            "checkpoint environment signature does not match the replay env"
        )
    return policy, params, action_setpoint, metadata


def main(args: Args) -> None:
    if args.truncation_steps < 0:
        raise ValueError("truncation_steps must be non-negative")
    if args.truncation_steps > args.rollout_steps:
        raise ValueError("truncation_steps cannot exceed rollout_steps")
    if args.truncation_steps and args.rollout_steps % args.truncation_steps:
        raise ValueError("rollout_steps must be divisible by truncation_steps")
    if args.eval_every_passes <= 0:
        raise ValueError("eval_every_passes must be positive")
    if args.num_rollouts <= 0 or args.eval_rollouts <= 0 or args.iters <= 0:
        raise ValueError("num_rollouts, eval_rollouts, and iters must be positive")

    env = EnvelopeGymnax(
        scenario_config.load_env(
            args.env,
            args.backend,
            reward=args.reward,
            variant=args.variant,
            disruption_penalty=args.disruption_penalty,
        )
    )
    env_params = env.default_params
    action_space = env.action_space(env_params)
    if not isinstance(action_space, Box):
        raise ValueError(
            "direct pathwise policy gradients require a continuous Box action space"
        )

    setpoint_key, init_key, training_key = jax.random.split(
        jax.random.PRNGKey(args.seed), 3
    )
    action_setpoint = initial_action_setpoint(env, env_params, setpoint_key)
    policy = ResidualPolicy(
        action_dim=action_space.shape[0],
        hidden_sizes=args.hidden_sizes,
    )
    dummy_obs = jnp.zeros(
        env.observation_space(env_params).shape,
        dtype=jnp.float32,
    )
    initial_params = policy.init(init_key, dummy_obs)["params"]
    eval_keys = jax.random.split(
        jax.random.PRNGKey(args.eval_seed),
        args.eval_rollouts,
    )
    rollout = make_policy_rollout(
        env,
        env_params,
        policy,
        action_setpoint,
        args.rollout_steps,
        remat=args.remat,
    )
    objective = make_policy_objective(rollout)
    num_params = sum(leaf.size for leaf in jax.tree.leaves(initial_params))
    print(
        f"policy: {env.observation_space(env_params).shape[0]} obs -> "
        f"{args.hidden_sizes} -> {action_space.shape[0]} actions "
        f"({num_params} parameters)"
    )
    if args.truncation_steps:
        chunks_per_pass = args.rollout_steps // args.truncation_steps
        print(
            f"objective: mean cumulative {args.reward} over "
            f"{args.num_rollouts} rollout(s) x {args.rollout_steps} steps; "
            f"updates every {args.truncation_steps} steps "
            f"({chunks_per_pass} updates/pass)\n"
        )
    else:
        print(
            f"objective: mean cumulative {args.reward} over "
            f"{args.num_rollouts} fresh rollout(s) x "
            f"{args.rollout_steps} steps\n"
        )

    wandb_lib = None
    if args.wandb:
        import wandb as wandb_lib

        wandb_lib.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            dir=wandb_dir(),
            group=args.wandb_group,
            mode=args.wandb_mode,
            name=(
                f"tbptt{args.truncation_steps}-policy-{Path(args.env).stem}-"
                f"{Path(args.backend).stem}-r{args.num_rollouts}-"
                f"p{args.iters}-seed{args.seed}"
            ),
            config={
                **dataclasses.asdict(args),
                "method": "direct_backprop_policy",
            },
        )
        wandb_lib.define_metric("simulator_steps")
        wandb_lib.define_metric("optimizer_updates")
        wandb_lib.define_metric("training_pass")
        for namespace in ("train", "train_chunk", "evaluation", "final"):
            wandb_lib.define_metric(
                f"{namespace}/*",
                step_metric="simulator_steps",
            )

    def log_iteration(entry: dict[str, float | int]) -> None:
        assert wandb_lib is not None
        namespace = "train_chunk" if args.truncation_steps else "train"
        metadata_names = {"simulator_steps", "update", "pass"}
        wandb_lib.log(
            {
                "simulator_steps": int(entry["simulator_steps"]),
                "optimizer_updates": int(entry["update"]),
                "training_pass": int(entry["pass"]),
                **{
                    f"{namespace}/{name}": value
                    for name, value in entry.items()
                    if name not in metadata_names
                },
            }
        )

    def log_evaluation(entry: dict[str, float | int]) -> None:
        assert wandb_lib is not None
        wandb_lib.log(
            {
                "simulator_steps": int(entry["simulator_steps"]),
                "optimizer_updates": int(entry["update"]),
                "training_pass": int(entry["pass"]),
                **{
                    f"evaluation/{name}": value
                    for name, value in entry.items()
                    if name not in {"simulator_steps", "update", "pass"}
                },
            }
        )

    if args.truncation_steps:
        chunk = make_policy_chunk(
            env,
            env_params,
            policy,
            action_setpoint,
            args.truncation_steps,
            remat=args.remat,
        )
        result = optimize_policy_truncated(
            initial_params,
            chunk,
            objective,
            training_key,
            eval_keys,
            rollout_steps=args.rollout_steps,
            truncation_steps=args.truncation_steps,
            num_rollouts=args.num_rollouts,
            iters=args.iters,
            eval_every_passes=args.eval_every_passes,
            learning_rate=args.learning_rate,
            grad_clip=args.grad_clip,
            on_iter=log_iteration if wandb_lib is not None else None,
            on_evaluate=log_evaluation if wandb_lib is not None else None,
        )
    else:
        result = optimize_policy(
            initial_params,
            objective,
            training_key,
            eval_keys,
            rollout_steps=args.rollout_steps,
            num_rollouts=args.num_rollouts,
            iters=args.iters,
            eval_every_passes=args.eval_every_passes,
            learning_rate=args.learning_rate,
            grad_clip=args.grad_clip,
            on_iter=log_iteration if wandb_lib is not None else None,
            on_evaluate=log_evaluation if wandb_lib is not None else None,
        )

    # The returned parameters are the best fixed-bank checkpoint, which may not
    # be the final iterate. Re-evaluate that retained checkpoint before saving.
    evaluate_retained = jax.jit(
        lambda retained_params: objective.evaluate(retained_params, eval_keys)
    )
    _, retained_metrics = evaluate_retained(result.params)
    eval_one = jax.jit(rollout)
    trajectory_return_array, trajectory = eval_one(result.params, eval_keys[0])
    jax.block_until_ready(retained_metrics.returns)
    retained_returns = np.asarray(retained_metrics.returns)
    eval_mean_return = float(retained_returns.mean())
    eval_std_return = float(retained_returns.std())
    trajectory_return = float(trajectory_return_array)
    print(
        f"held-out return={eval_mean_return:+.5f} "
        f"(+/- {eval_std_return:.5f}, n={args.eval_rollouts})"
    )
    if wandb_lib is not None:
        total_simulator_steps = args.iters * args.rollout_steps * args.num_rollouts
        if args.truncation_steps:
            total_optimizer_updates = (
                args.iters * args.rollout_steps // args.truncation_steps
            )
        else:
            total_optimizer_updates = args.iters
        wandb_lib.log(
            {
                "simulator_steps": total_simulator_steps,
                "optimizer_updates": total_optimizer_updates,
                "training_pass": args.iters,
                "final/baseline_return": result.baseline_return,
                "final/best_eval_mean_return": result.best_return,
                "final/eval_mean_return": eval_mean_return,
                "final/eval_std_return": eval_std_return,
                "final/trajectory_return": trajectory_return,
                "final/eval_improvement": (eval_mean_return - result.baseline_return),
                "final/best_eval_improvement": (
                    result.best_return - result.baseline_return
                ),
                "final/best_eval_update": result.best_iter,
            }
        )
        wandb_lib.finish()

    out = save_policy(
        args.out,
        args,
        result.params,
        action_setpoint,
        trajectory,
        trajectory_return,
        eval_mean_return,
        env,
        result,
    )
    print(f"saved policy and rollout to {out}")


if __name__ == "__main__":
    main(tyro.cli(Args))
