"""Reusable truncated objectives for direct gradients through environments."""

import dataclasses
from collections.abc import Callable
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import optax


class KnotCarry(NamedTuple):
    env_state: Any
    alive: jax.Array
    zero_reward: jax.Array


class KnotStep(NamedTuple):
    reward: jax.Array
    alive: jax.Array


class FiniteGradientStats(NamedTuple):
    """Diagnostics from finite-only aggregation over rollout gradients."""

    nonfinite_elements: jax.Array
    total_elements: jax.Array
    rollouts_with_nonfinite: jax.Array
    total_rollouts: jax.Array
    all_missing_elements: jax.Array
    aggregate_finite: jax.Array


@dataclasses.dataclass(frozen=True)
class KnotChunk:
    initialize: Callable[[jax.Array], KnotCarry]
    run: Callable[
        [jax.Array, KnotCarry, jax.Array],
        tuple[jax.Array, tuple[KnotCarry, KnotStep]],
    ]


def finite_mean_gradients(
    per_rollout_grads: Any,
) -> tuple[Any, FiniteGradientStats]:
    """Average each gradient coordinate over its finite rollout values.

    NaN and either infinity are treated as missing. A coordinate with no
    finite rollout values is set to zero, while the returned diagnostics keep
    that loss of signal visible to the training loop.
    """
    leaves = jax.tree.leaves(per_rollout_grads)
    if not leaves:
        raise ValueError("per_rollout_grads must contain at least one leaf")
    if any(leaf.ndim < 1 for leaf in leaves):
        raise ValueError("gradient leaves must have a leading rollout axis")

    num_rollouts = leaves[0].shape[0]
    if any(leaf.shape[0] != num_rollouts for leaf in leaves[1:]):
        raise ValueError("gradient leaves must share the rollout axis size")

    finite_masks = jax.tree.map(jnp.isfinite, per_rollout_grads)

    def finite_mean(values, finite):
        finite_count = jnp.sum(finite, axis=0)
        finite_sum = jnp.sum(jnp.where(finite, values, 0), axis=0)
        divisor = jnp.maximum(finite_count, 1).astype(values.dtype)
        return finite_sum / divisor

    mean_grads = jax.tree.map(finite_mean, per_rollout_grads, finite_masks)
    mask_leaves = jax.tree.leaves(finite_masks)
    nonfinite_elements = sum(
        (jnp.sum(~finite, dtype=jnp.int32) for finite in mask_leaves),
        start=jnp.asarray(0, dtype=jnp.int32),
    )
    total_elements = jnp.asarray(
        sum(finite.size for finite in mask_leaves),
        dtype=jnp.int32,
    )
    rollout_has_nonfinite = jnp.zeros((num_rollouts,), dtype=jnp.bool_)
    all_missing_elements = jnp.asarray(0, dtype=jnp.int32)
    for finite in mask_leaves:
        rollout_has_nonfinite |= ~jnp.all(
            finite.reshape((num_rollouts, -1)),
            axis=1,
        )
        all_missing_elements += jnp.sum(
            ~jnp.any(finite, axis=0),
            dtype=jnp.int32,
        )
    aggregate_finite = jnp.all(
        jnp.stack([jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree.leaves(mean_grads)])
    )
    return mean_grads, FiniteGradientStats(
        nonfinite_elements=nonfinite_elements,
        total_elements=total_elements,
        rollouts_with_nonfinite=jnp.sum(
            rollout_has_nonfinite,
            dtype=jnp.int32,
        ),
        total_rollouts=jnp.asarray(num_rollouts, dtype=jnp.int32),
        all_missing_elements=all_missing_elements,
        aggregate_finite=aggregate_finite,
    )


def apply_updates_with_backoff(
    optimizer: optax.GradientTransformation,
    grads: Any,
    opt_state: Any,
    params: Any,
    rollback_opt_state: Any,
    rollback_params: Any,
    update_scale: jax.Array,
    *,
    backoff_factor: float,
    min_update_scale: float,
) -> tuple[Any, Any, Any, Any, jax.Array, jax.Array]:
    """Apply a finite update or roll back the update that caused bad gradients.

    A non-finite gradient is observed one optimizer step after parameters have
    entered a non-differentiable simulator region. Merely skipping that
    gradient leaves the parameters stranded there. Keep the previous finite
    point and optimizer state so the next attempt can resume from it with a
    smaller update.
    """
    finite = jnp.all(
        jnp.stack([jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree.leaves(grads)])
    )

    def apply(_):
        updates, next_opt_state = optimizer.update(grads, opt_state, params)
        scaled_updates = jax.tree.map(
            lambda update: update_scale * update,
            updates,
        )
        next_params = optax.apply_updates(params, scaled_updates)
        return (
            next_params,
            next_opt_state,
            params,
            opt_state,
            update_scale,
        )

    def rollback(_):
        next_scale = jnp.maximum(
            update_scale * backoff_factor,
            min_update_scale,
        )
        return (
            rollback_params,
            rollback_opt_state,
            rollback_params,
            rollback_opt_state,
            next_scale,
        )

    result = jax.lax.cond(finite, apply, rollback, operand=None)
    return (*result, finite)


def make_knot_chunk(
    env,
    to_actions: Callable[[jax.Array], jax.Array],
    chunk_steps: int,
    *,
    remat: bool,
) -> KnotChunk:
    """Build a truncated open-loop objective carrying state across chunks."""

    def step(carry, action):
        env_state, alive, zero_reward = carry

        def active_step(_):
            next_state, info = env.step(env_state, action)
            done = info.terminated | info.truncated
            return (
                KnotCarry(next_state, ~done, zero_reward),
                KnotStep(info.reward, jnp.asarray(True)),
            )

        def inactive_step(_):
            return (
                KnotCarry(env_state, jnp.asarray(False), zero_reward),
                KnotStep(zero_reward, jnp.asarray(False)),
            )

        return jax.lax.cond(alive, active_step, inactive_step, operand=None)

    scan_step = jax.checkpoint(step) if remat else step

    def initialize(key):
        env_state, info = env.init(key)
        return KnotCarry(
            env_state,
            jnp.asarray(True),
            jnp.zeros_like(info.reward),
        )

    def run(current_theta, carry, start_step):
        actions = jax.lax.dynamic_slice_in_dim(
            to_actions(current_theta),
            start_step,
            chunk_steps,
        )
        next_carry, trajectory = jax.lax.scan(scan_step, carry, actions)
        return jnp.sum(trajectory.reward), (next_carry, trajectory)

    return KnotChunk(initialize=initialize, run=run)


def tree_is_finite(tree: Any) -> jax.Array:
    """Whether every numeric leaf is finite (empty optimizer states are valid)."""
    leaves = jax.tree.leaves(tree)
    return (
        jnp.all(jnp.stack([jnp.all(jnp.isfinite(x)) for x in leaves]))
        if leaves
        else jnp.asarray(True)
    )


def gradient_horizon(
    episode_steps: int, total_timesteps: int, num_rollouts: int, limit: int
) -> int:
    """Choose the largest horizon that exactly tiles episodes and the budget."""
    if min(episode_steps, total_timesteps, num_rollouts, limit) <= 0:
        raise ValueError(
            "episode_steps, total_timesteps, num_rollouts and "
            "gradient_horizon must be positive"
        )
    if total_timesteps % num_rollouts:
        raise ValueError("total_timesteps must be divisible by num_rollouts")
    rollout_steps = total_timesteps // num_rollouts
    for candidate in range(min(episode_steps, rollout_steps, limit), 0, -1):
        if episode_steps % candidate == 0 and rollout_steps % candidate == 0:
            return candidate
    raise AssertionError("one divides every positive integer")


def make_optimizer(
    learning_rate: float, grad_clip: float
) -> optax.GradientTransformation:
    transforms = [optax.clip_by_global_norm(grad_clip)] if grad_clip > 0 else []
    return optax.chain(*transforms, optax.adam(learning_rate))


class PolicyOptimizerDiagnostics(NamedTuple):
    grad_norm: jax.Array
    grad_clip_scale: jax.Array
    optimizer_update_finite: jax.Array
    params_finite: jax.Array


def apply_policy_optimizer_update(
    optimizer: optax.GradientTransformation,
    grads: Any,
    opt_state: Any,
    params: Any,
    grad_clip: float,
) -> tuple[Any, Any, PolicyOptimizerDiagnostics]:
    """Apply the baseline's fixed clipped Adam step, without knot backoff."""
    updates, next_opt_state = optimizer.update(grads, opt_state, params)
    next_params = optax.apply_updates(params, updates)
    norm = optax.global_norm(grads)
    scale = (
        jnp.minimum(
            jnp.asarray(1.0, norm.dtype),
            grad_clip / jnp.maximum(norm, jnp.finfo(norm.dtype).tiny),
        )
        if grad_clip > 0
        else jnp.asarray(1.0, norm.dtype)
    )
    return (
        next_params,
        next_opt_state,
        PolicyOptimizerDiagnostics(
            norm, scale, tree_is_finite(updates), tree_is_finite(next_params)
        ),
    )


def setpoint_theta_row(env: Any, key: jax.Array) -> jax.Array:
    """Unconstrained reset setpoint, inset to preserve actuator sensitivities."""
    from plasmax.wrappers import unwrap_to_env_state

    state, _ = env.init(key)
    state = unwrap_to_env_state(state)
    low, high = env.unwrapped.action_space.low, env.unwrapped.action_space.high
    normalized = 2.0 * (state.prev_action - low) / (high - low) - 1.0
    return jnp.arctanh(jnp.clip(normalized, -0.999, 0.999))


def knot_actions(theta: jax.Array, num_steps: int) -> jax.Array:
    positions = jnp.arange(num_steps, dtype=theta.dtype)
    knots = jnp.linspace(0.0, num_steps - 1, theta.shape[0], dtype=theta.dtype)
    values = jax.vmap(
        lambda column: jnp.interp(positions, knots, column), in_axes=1, out_axes=1
    )(theta)
    return jnp.tanh(values)


def make_parameterization(
    env: Any, key: jax.Array, num_steps: int, n_knots: int
) -> tuple[Callable, jax.Array, str]:
    """Setpoint-initialized piecewise-linear unconstrained actuator knots."""
    if num_steps <= 0:
        raise ValueError("num_steps must be positive")
    count = min(n_knots, num_steps) if n_knots > 0 else num_steps
    row = setpoint_theta_row(env, key)
    theta = jnp.broadcast_to(row, (count, row.shape[0]))
    return (
        lambda values: knot_actions(values, num_steps),
        theta,
        f"{count} time-knots x {row.shape[0]} actuators",
    )
