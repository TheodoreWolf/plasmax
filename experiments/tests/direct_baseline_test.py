"""Behavioral tests for truncated direct-gradient baseline chunks."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

from experiments.studies.direct_gradient import (
    apply_updates_with_backoff,
    make_knot_chunk,
)


class _Info(NamedTuple):
    reward: jax.Array
    terminated: jax.Array
    truncated: jax.Array


class _AccumulatorEnv:
    def init(self, key):
        del key
        zero = jnp.asarray(0.0)
        return zero, _Info(zero, jnp.asarray(False), jnp.asarray(False))

    def step(self, state, action):
        next_state = state + action
        return next_state, _Info(
            reward=next_state,
            terminated=jnp.asarray(False),
            truncated=jnp.asarray(False),
        )


def test_knot_chunks_carry_state_but_cut_gradients_between_windows():
    chunk = make_knot_chunk(
        _AccumulatorEnv(),
        lambda theta: theta,
        chunk_steps=2,
        remat=False,
    )
    theta = jnp.asarray([1.0, 2.0, 3.0, 4.0])
    carry = chunk.initialize(jax.random.key(0))

    (first_return, (carry, _)), first_grad = jax.value_and_grad(
        lambda value: chunk.run(value, carry, jnp.asarray(0)),
        has_aux=True,
    )(theta)
    carry = jax.tree.map(jax.lax.stop_gradient, carry)
    (second_return, _), second_grad = jax.value_and_grad(
        lambda value: chunk.run(value, carry, jnp.asarray(2)),
        has_aux=True,
    )(theta)

    np.testing.assert_allclose(first_return, 4.0, rtol=1e-7, atol=0.0)
    np.testing.assert_allclose(second_return, 16.0, rtol=1e-7, atol=0.0)
    np.testing.assert_allclose(first_grad, [2.0, 1.0, 0.0, 0.0], rtol=1e-7, atol=0.0)
    np.testing.assert_allclose(second_grad, [0.0, 0.0, 2.0, 1.0], rtol=1e-7, atol=0.0)


class _TerminatingEnv(_AccumulatorEnv):
    def step(self, state, action):
        next_state = state + action
        return next_state, _Info(
            reward=next_state,
            terminated=next_state >= 1.0,
            truncated=jnp.asarray(False),
        )


def test_knot_chunk_masks_steps_after_episode_boundary():
    chunk = make_knot_chunk(
        _TerminatingEnv(),
        lambda theta: theta,
        chunk_steps=3,
        remat=False,
    )

    rollout_return, (carry, trajectory) = chunk.run(
        jnp.ones(3),
        chunk.initialize(jax.random.key(0)),
        jnp.asarray(0),
    )

    np.testing.assert_allclose(rollout_return, 1.0, rtol=1e-7, atol=0.0)
    np.testing.assert_array_equal(trajectory.alive, [True, False, False])
    np.testing.assert_array_equal(carry.alive, False)


def test_nonfinite_gradient_rolls_back_and_retries_with_smaller_update():
    optimizer = optax.sgd(1.0)
    initial_params = jnp.asarray([1.0])
    initial_opt_state = optimizer.init(initial_params)

    (
        params,
        opt_state,
        rollback_params,
        rollback_opt_state,
        update_scale,
        finite,
    ) = apply_updates_with_backoff(
        optimizer,
        jnp.asarray([2.0]),
        initial_opt_state,
        initial_params,
        initial_opt_state,
        initial_params,
        jnp.asarray(1.0),
        backoff_factor=0.5,
        min_update_scale=0.1,
    )
    np.testing.assert_allclose(params, [-1.0], rtol=1e-7, atol=0.0)
    np.testing.assert_array_equal(finite, True)

    (
        params,
        opt_state,
        rollback_params,
        rollback_opt_state,
        update_scale,
        finite,
    ) = apply_updates_with_backoff(
        optimizer,
        jnp.asarray([jnp.nan]),
        opt_state,
        params,
        rollback_opt_state,
        rollback_params,
        update_scale,
        backoff_factor=0.5,
        min_update_scale=0.1,
    )
    np.testing.assert_allclose(params, [1.0], rtol=1e-7, atol=0.0)
    np.testing.assert_allclose(update_scale, 0.5, rtol=1e-7, atol=0.0)
    np.testing.assert_array_equal(finite, False)

    params, _, _, _, update_scale, finite = apply_updates_with_backoff(
        optimizer,
        jnp.asarray([2.0]),
        opt_state,
        params,
        rollback_opt_state,
        rollback_params,
        update_scale,
        backoff_factor=0.5,
        min_update_scale=0.1,
    )
    np.testing.assert_allclose(params, [0.0], rtol=1e-7, atol=0.0)
    np.testing.assert_allclose(update_scale, 0.5, rtol=1e-7, atol=0.0)
    np.testing.assert_array_equal(finite, True)
