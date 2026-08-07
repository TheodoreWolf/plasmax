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


@dataclasses.dataclass(frozen=True)
class KnotChunk:
    initialize: Callable[[jax.Array], KnotCarry]
    run: Callable[
        [jax.Array, KnotCarry, jax.Array],
        tuple[jax.Array, tuple[KnotCarry, KnotStep]],
    ]


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
