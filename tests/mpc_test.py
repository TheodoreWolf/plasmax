"""Cheap training and frozen-inference contracts for learned-dynamics MPC."""

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from envelope import AutoResetWrapper, Continuous
from helpers import CheapBoundaryEnv

from agents.mpc import MPCAgent
from plasmax.spaces import ObsLayout
from plasmax.wrappers import QuantizeActionWrapper


class NamedBoundaryEnv(CheapBoundaryEnv):
    def obs_layout(self) -> ObsLayout:
        return ObsLayout({}, {"P_fusion": slice(0, 1)}, (), ("P_fusion",))


def _reward_fn(obs: jax.Array, action: jax.Array, next_obs: jax.Array) -> jax.Array:
    del action, next_obs
    return -jnp.sum(obs**2)


def _make_agent(**kwargs: Any) -> MPCAgent:
    defaults = dict(
        horizon=2,
        num_samples=4,
        buffer_size=16,
        hidden=8,
        train_batch_size=2,
        total_timesteps=7,
        eval_freq=3,
        eval_num_episodes=2,
        eval_max_steps=3,
    )
    defaults.update(kwargs)
    env = defaults.pop(
        "env",
        NamedBoundaryEnv(
            obs_dim=2,
            action_low=(-1.0,),
            action_high=(1.0,),
            truncate_after=2,
        ),
    )
    return MPCAgent.create(env, **defaults)


def _assert_trees_equal(left: Any, right: Any) -> None:
    for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True):
        a, b = jnp.asarray(a), jnp.asarray(b)
        if jax.dtypes.issubdtype(a.dtype, jax.dtypes.prng_key):
            np.testing.assert_array_equal(
                jax.random.key_data(a), jax.random.key_data(b)
            )
        else:
            np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-6, equal_nan=True)


class MPCAgentTest:
    def test_create_and_named_objective(self) -> None:
        agent = _make_agent()
        state = agent.init_state(jax.random.key(0))
        assert isinstance(agent.env.action_space, Continuous)
        assert agent.action_low.shape == (1,)
        assert state.buffer.size == 16
        assert agent.reward_scalar == "P_fusion"
        assert agent.reward_slice == (0, 1)
        np.testing.assert_array_equal(
            agent.reward_fn(None, None, jnp.array([2.0, 7.0])), 2.0
        )

    def test_create_rejects_autoreset_and_impossible_warmup(self) -> None:
        agent = _make_agent()
        with pytest.raises(ValueError, match="non-autoresetting"):
            _make_agent(env=AutoResetWrapper(agent.env))
        with pytest.raises(ValueError, match="train_batch_size"):
            _make_agent(buffer_size=1)
        with pytest.raises(TypeError, match="Continuous.*quantized"):
            _make_agent(env=QuantizeActionWrapper(agent.env, bin_counts=(3,)))

    def test_make_act_bounds_and_frozen_deterministic_snapshot(self) -> None:
        agent = _make_agent(planning_seed=12)
        state = agent.init_state(jax.random.key(0))
        before = jax.tree.map(lambda value: jnp.asarray(value).copy(), state)
        act = jax.jit(agent.make_act(state, deterministic=True))
        a = act(state.obs, jax.random.key(1))
        b = act(state.obs, jax.random.key(2))
        assert a.shape == (1,)
        assert np.all(np.asarray(a) >= np.asarray(agent.action_low))
        assert np.all(np.asarray(a) <= np.asarray(agent.action_high))
        np.testing.assert_array_equal(a, b)
        _assert_trees_equal(before, state)

    @pytest.mark.parametrize("boundary", ["terminate_after", "truncate_after"])
    def test_exact_budget_retains_replay_and_model_across_resets(
        self, boundary: str
    ) -> None:
        env = NamedBoundaryEnv(
            obs_dim=2,
            action_low=(-1.0,),
            action_high=(1.0,),
            **{boundary: 2},
        )
        state, results = jax.jit(_make_agent(env=env).train)(jax.random.key(0))
        np.testing.assert_array_equal(state.global_step, 7)
        np.testing.assert_array_equal(state.env_state.steps, 1)
        np.testing.assert_array_equal(state.buffer.num_entries, 7)
        np.testing.assert_array_equal(state.buffer.obs[:7, 0], [0, 1, 0, 1, 0, 1, 0])
        np.testing.assert_array_equal(
            state.buffer.next_obs[:7, 0], [1, 2, 1, 2, 1, 2, 1]
        )
        expected_done = [False, True, False, True, False, True, False]
        np.testing.assert_array_equal(
            state.buffer.done[:7],
            expected_done if boundary == "terminate_after" else [False] * 7,
        )
        np.testing.assert_array_equal(state.model_ts.step, 6)
        np.testing.assert_array_equal(results["global_step"], [0, 3, 6, 7])
        lengths, returns = results["evaluation"]
        np.testing.assert_array_equal(lengths, np.full((4, 2), 2))
        np.testing.assert_allclose(returns, 3.0, rtol=0, atol=0)
        assert not state.failed

    def test_eager_jit_vmap_and_eval_frequency_independence(self) -> None:
        def callback(agent: MPCAgent, state: Any, rng: jax.Array, _: Any):
            return agent.make_act(state)(state.obs, rng)

        agent = _make_agent(eval_callback=callback)
        keys = jax.random.split(jax.random.key(4), 2)
        eager = agent.train(keys[0])
        compiled = jax.jit(agent.train)(keys[0])
        batched = jax.jit(jax.vmap(agent.train))(keys)
        _assert_trees_equal(eager, compiled)
        _assert_trees_equal(compiled, jax.tree.map(lambda x: x[0], batched))
        _assert_trees_equal(agent.train(keys[1]), jax.tree.map(lambda x: x[1], batched))
        other_frequency = jax.jit(agent.replace(eval_freq=2).train)(keys[0])
        _assert_trees_equal(compiled[0], other_frequency[0])
        weights = batched[0].model_ts.params["params"]["Dense_0"]["kernel"]
        assert not np.allclose(weights[0], weights[1])

    def test_training_waits_for_full_batch(self) -> None:
        agent = _make_agent(total_timesteps=3, train_batch_size=4)
        state, results = jax.jit(agent.train)(jax.random.key(0))
        np.testing.assert_array_equal(state.model_ts.step, 0)
        assert np.all(np.isnan(results["model_loss"]))

    def test_failed_update_retains_last_valid_model(self) -> None:
        agent = _make_agent(lr=float("inf"))
        key = jax.random.key(0)
        initial = agent.init_state(key)
        state, results = jax.jit(agent.train)(key)
        assert state.failed
        np.testing.assert_array_equal(state.first_failure_step, 2)
        np.testing.assert_array_equal(state.global_step, 2)
        np.testing.assert_array_equal(results["global_step"], [0, 2, 2, 2])
        _assert_trees_equal(state.model_ts, initial.model_ts)

    @pytest.mark.parametrize("terminated", [False, True])
    def test_model_learning_masks_termination_but_keeps_truncation(
        self, terminated: bool
    ) -> None:
        agent = _make_agent(reward_fn=_reward_fn)
        state = agent.init_state(jax.random.key(0))
        for _ in range(8):
            state = agent.observe(
                state,
                jnp.zeros(2),
                jnp.zeros(1),
                jnp.float32(0),
                terminated=jnp.bool_(terminated),
                truncated=jnp.bool_(not terminated),
                next_obs=jnp.full(2, 2.0, jnp.float32),
            )
        _, initial_loss = agent.train_step(state, jax.random.key(3))
        for i in range(20):
            state, loss = agent.train_step(
                state, jax.random.fold_in(jax.random.key(4), i)
            )
        if terminated:
            np.testing.assert_allclose(loss, 0.0, rtol=0, atol=1e-6)
        else:
            assert 0 < loss < initial_loss
        assert agent.reward_scalar is None


class InvalidResetEnv(NamedBoundaryEnv):
    def reset(self, state: Any, key: jax.Array):
        state, info = super().reset(state, key)
        return state, info.update(terminated=jnp.bool_(True))


def test_invalid_reset_stops_without_counting_padding_as_transitions() -> None:
    env = InvalidResetEnv(
        obs_dim=2, action_low=(-1.0,), action_high=(1.0,), truncate_after=2
    )
    state, _ = jax.jit(_make_agent(env=env).train)(jax.random.key(0))
    assert state.failed
    np.testing.assert_array_equal(state.global_step, 2)
    np.testing.assert_array_equal(state.first_failure_step, 2)
    np.testing.assert_array_equal(state.buffer.num_entries, 2)


class SeedFailureEnv(NamedBoundaryEnv):
    def init(self, key: jax.Array):
        state, _ = super().init(key)
        state = state.replace(obs=state.obs.at[0].set(jax.random.bernoulli(key)))
        return state, self._info(state)

    def step(self, state: Any, action: jax.Array):
        del action
        next_state = state.replace(
            obs=state.obs.at[1].add(1),
            steps=state.steps + 1,
        )
        return next_state, self._info(next_state).update(
            reward=jnp.where(state.obs[0] > 0, jnp.float32(jnp.nan), jnp.float32(1)),
        )


def test_vmap_failure_isolated_to_one_seed() -> None:
    env = SeedFailureEnv(obs_dim=2, action_low=(-1.0,), action_high=(1.0,))
    agent = _make_agent(
        env=env,
        total_timesteps=3,
        eval_callback=lambda agent, state, rng, diagnostics: state.global_step,
    )
    keys = jax.random.split(jax.random.key(42), 6)
    state, _ = jax.jit(jax.vmap(agent.train))(keys)
    assert np.any(state.failed) and np.any(~state.failed)
    np.testing.assert_array_equal(state.global_step, jnp.where(state.failed, 1, 3))
    np.testing.assert_array_equal(
        state.first_failure_step, jnp.where(state.failed, 1, -1)
    )
    np.testing.assert_array_equal(
        state.buffer.num_entries, jnp.where(state.failed, 0, 3)
    )
