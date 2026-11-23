"""Behavioral tests for truncated direct-gradient baseline chunks."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from envelope import Continuous

from agents.backprop import DeterministicPolicy, make_policy_chunk
from agents.direct_gradient import (
    apply_policy_optimizer_update,
    finite_mean_gradients,
    gradient_horizon,
    make_knot_chunk,
    make_optimizer,
)


class _Info(NamedTuple):
    reward: jax.Array
    terminated: jax.Array
    truncated: jax.Array


class _PolicyInfo(NamedTuple):
    obs: jax.Array
    reward: jax.Array
    terminated: jax.Array
    truncated: jax.Array


class _Plasma(NamedTuple):
    q_min: jax.Array
    fgw_n_e_line_avg: jax.Array


class _RolloutState(NamedTuple):
    plasma: _Plasma


class _Float64StateOpenLoopEnv:
    def init(self, key):
        del key
        plasma = _Plasma(
            q_min=jnp.asarray(1.0, dtype=jnp.float64),
            fgw_n_e_line_avg=jnp.asarray(0.5, dtype=jnp.float64),
        )
        return _RolloutState(plasma), _Info(
            reward=jnp.asarray(0.0, dtype=jnp.float32),
            terminated=jnp.asarray(False),
            truncated=jnp.asarray(False),
        )

    def step(self, state, action):
        del action
        return state, _Info(
            reward=jnp.asarray(1.0, dtype=jnp.float32),
            terminated=jnp.asarray(True),
            truncated=jnp.asarray(False),
        )


class _Float64StatePolicyEnv:
    action_space = Continuous(low=jnp.asarray([-1.0]), high=jnp.asarray([1.0]))

    def init(self, key):
        del key
        return jnp.asarray(0.0, dtype=jnp.float64), _PolicyInfo(
            obs=jnp.zeros(1, dtype=jnp.float32),
            reward=jnp.asarray(0.0, dtype=jnp.float32),
            terminated=jnp.asarray(False),
            truncated=jnp.asarray(False),
        )

    def step(self, state, action):
        del action
        return state, _PolicyInfo(
            obs=jnp.zeros(1, dtype=jnp.float32),
            reward=jnp.asarray(1.0, dtype=jnp.float32),
            terminated=jnp.asarray(True),
            truncated=jnp.asarray(False),
        )


def test_policy_chunk_masks_float64_simulator_state_with_float32_reward():
    policy = DeterministicPolicy(action_dim=1, hidden_sizes=(2,))
    params = policy.init(jax.random.key(0), jnp.zeros(1, dtype=jnp.float32))["params"]
    chunk = make_policy_chunk(
        _Float64StatePolicyEnv(),
        policy,
        chunk_steps=2,
        remat=False,
    )

    rollout_return, (_, trajectory) = jax.jit(chunk.run)(
        params,
        chunk.initialize(jax.random.key(1)),
    )

    np.testing.assert_allclose(rollout_return, 1.0, rtol=1e-7, atol=0.0)
    np.testing.assert_array_equal(trajectory.alive, [True, False])
    assert trajectory.reward.dtype == jnp.float32


def test_open_loop_rollout_masks_float64_state_with_float32_reward():
    chunk = make_knot_chunk(
        _Float64StateOpenLoopEnv(), lambda theta: theta, 2, remat=False
    )
    reward, (_, trajectory) = jax.jit(chunk.run)(
        jnp.zeros((2, 1), dtype=jnp.float32),
        chunk.initialize(jax.random.key(0)),
        jnp.asarray(0),
    )
    np.testing.assert_allclose(reward, 1.0, rtol=1e-7, atol=0.0)
    np.testing.assert_array_equal(trajectory.alive, [True, False])
    assert trajectory.reward.dtype == jnp.float32


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


def test_finite_mean_gradients_excludes_every_nonfinite_kind_per_coordinate():
    per_rollout_grads = {
        "dense": {
            "kernel": jnp.asarray(
                [
                    [1.0, jnp.nan],
                    [3.0, jnp.inf],
                    [jnp.nan, -jnp.inf],
                ]
            )
        },
        "bias": jnp.asarray(
            [
                [2.0, 4.0],
                [jnp.inf, 6.0],
                [4.0, jnp.nan],
            ]
        ),
    }

    aggregate, stats = jax.jit(finite_mean_gradients)(per_rollout_grads)

    np.testing.assert_allclose(
        aggregate["dense"]["kernel"],
        [2.0, 0.0],
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        aggregate["bias"],
        [3.0, 5.0],
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_array_equal(stats.nonfinite_elements, 6)
    np.testing.assert_array_equal(stats.total_elements, 12)
    np.testing.assert_array_equal(stats.rollouts_with_nonfinite, 3)
    np.testing.assert_array_equal(stats.total_rollouts, 3)
    np.testing.assert_array_equal(stats.all_missing_elements, 1)
    np.testing.assert_array_equal(stats.aggregate_finite, True)


def test_finite_mean_gradients_matches_mean_and_supports_seed_vmap():
    per_rollout_grads = {
        "matrix": jnp.arange(24, dtype=jnp.float32).reshape(3, 2, 4),
        "vector": jnp.asarray([[1.0], [2.0], [3.0]]),
    }
    batched = jax.tree.map(
        lambda value: jnp.stack((value, value + 10.0)),
        per_rollout_grads,
    )

    aggregate, stats = jax.jit(jax.vmap(finite_mean_gradients))(batched)
    expected = jax.tree.map(
        lambda value: jnp.mean(value, axis=1),
        batched,
    )

    for actual_leaf, expected_leaf in zip(
        jax.tree.leaves(aggregate),
        jax.tree.leaves(expected),
        strict=True,
    ):
        np.testing.assert_array_equal(actual_leaf, expected_leaf)
    np.testing.assert_array_equal(stats.nonfinite_elements, [0, 0])
    np.testing.assert_array_equal(stats.all_missing_elements, [0, 0])
    np.testing.assert_array_equal(stats.aggregate_finite, [True, True])


def test_clipped_adam_matches_optax_and_keeps_parameters_finite():
    optimizer = make_optimizer(
        1e-4,
        1.0,
    )
    expected_optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(1e-4),
    )
    params = jnp.asarray([1.0, -1.0])
    grads = jnp.asarray([3.0, 4.0])
    state = optimizer.init(params)
    expected_state = expected_optimizer.init(params)
    apply_update = jax.jit(
        lambda gradient, opt_state, parameters: apply_policy_optimizer_update(
            optimizer,
            gradient,
            opt_state,
            parameters,
            1.0,
        )
    )

    for _ in range(2):
        previous_params = params
        params, state, diagnostics = apply_update(
            grads,
            state,
            params,
        )
        expected_updates, expected_state = expected_optimizer.update(
            grads,
            expected_state,
            previous_params,
        )
        expected_params = optax.apply_updates(previous_params, expected_updates)
        np.testing.assert_allclose(params, expected_params, rtol=1e-7, atol=0.0)
        np.testing.assert_allclose(diagnostics.grad_norm, 5.0, rtol=1e-7, atol=0.0)
        np.testing.assert_allclose(
            diagnostics.grad_clip_scale,
            0.2,
            rtol=1e-7,
            atol=0.0,
        )
        np.testing.assert_array_equal(diagnostics.optimizer_update_finite, True)
        np.testing.assert_array_equal(diagnostics.params_finite, True)

    assert not np.array_equal(np.asarray(params), np.asarray([1.0, -1.0]))


def testgradient_horizon_exactly_tiles_episode_and_budget():
    horizon = gradient_horizon(
        episode_steps=150,
        total_timesteps=10_240_000,
        num_rollouts=64,
        limit=32,
    )

    assert horizon == 25
    assert 150 % horizon == 0
    assert (10_240_000 // 64) % horizon == 0


@pytest.mark.parametrize("episode_steps", [600, 4_400, 50, 4_000])
def test_selected_policy_runs_have_horizon_ten_and_one_thousand_updates(
    episode_steps: int,
):
    total_timesteps = 10_240_000
    num_rollouts = 1_024
    horizon = gradient_horizon(
        episode_steps=episode_steps,
        total_timesteps=total_timesteps,
        num_rollouts=num_rollouts,
        limit=10,
    )

    assert horizon == 10
    assert total_timesteps // (num_rollouts * horizon) == 1_000


@pytest.mark.parametrize("episode_steps", [600, 4_400, 50, 4_000])
def test_selected_policy_pilots_have_horizon_five_and_two_hundred_updates(
    episode_steps: int,
):
    total_timesteps = 1_024_000
    num_rollouts = 1_024
    horizon = gradient_horizon(
        episode_steps=episode_steps,
        total_timesteps=total_timesteps,
        num_rollouts=num_rollouts,
        limit=5,
    )

    assert horizon == 5
    assert total_timesteps // (num_rollouts * horizon) == 200
