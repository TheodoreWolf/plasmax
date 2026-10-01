"""Full-episode evolution strategy contracts on a toy Envelope environment."""

import dataclasses
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from envelope import Continuous

from agents.es import ESAgent
from plasmax.spaces import ObsLayout


class _Info(NamedTuple):
    obs: jax.Array
    reward: jax.Array
    terminated: jax.Array
    truncated: jax.Array

    def update(self, **kwargs: Any) -> "_Info":
        return self._replace(**kwargs)


class _State(NamedTuple):
    position: jax.Array
    step: jax.Array
    goal: jax.Array


class _Environment:
    max_steps = 4
    action_space = Continuous(
        low=jnp.asarray([-1.0], jnp.float32),
        high=jnp.asarray([1.0], jnp.float32),
    )
    observation_space = Continuous(
        low=jnp.full(2, -jnp.inf, jnp.float32),
        high=jnp.full(2, jnp.inf, jnp.float32),
    )

    def __init__(self, terminal_at: int = 4) -> None:
        self.terminal_at = terminal_at

    def obs_layout(self) -> ObsLayout:
        return ObsLayout(
            {},
            {"position": slice(0, 1), "elapsed_time": slice(1, 2)},
            (),
            ("position", "elapsed_time"),
        )

    def init(self, key: jax.Array) -> tuple[_State, _Info]:
        state = _State(
            jnp.asarray(0.0, jnp.float32),
            jnp.asarray(0, jnp.int32),
            jax.random.normal(key, (), dtype=jnp.float32),
        )
        return state, self.info(state, jnp.asarray(0.0, jnp.float32))

    def info(self, state: _State, reward: jax.Array) -> _Info:
        return _Info(
            jnp.stack((state.position, state.step.astype(jnp.float32))),
            reward,
            state.step >= self.terminal_at,
            jnp.asarray(False),
        )

    def step(self, state: _State, action: jax.Array) -> tuple[_State, _Info]:
        state = state._replace(position=state.position + action[0], step=state.step + 1)
        reward = -jnp.square(state.position - state.goal)
        return state, self.info(state, reward)


class _ClockRewardEnvironment(_Environment):
    def step(self, state: _State, action: jax.Array) -> tuple[_State, _Info]:
        del action
        state = state._replace(step=state.step + 1)
        return state, self.info(state, state.step.astype(jnp.float32))


class _RandomRewardEnvironment(_Environment):
    def step(self, state: _State, action: jax.Array) -> tuple[_State, _Info]:
        del action
        state = state._replace(step=state.step + 1)
        return state, self.info(state, state.goal)


class _InvalidRewardEnvironment(_Environment):
    def __init__(self, invalid_reward: float) -> None:
        super().__init__()
        self.invalid_reward = invalid_reward

    def step(self, state: _State, action: jax.Array) -> tuple[_State, _Info]:
        state, info = super().step(state, action)
        reward = jnp.where(
            state.step == 3,
            jnp.asarray(self.invalid_reward, jnp.float32),
            info.reward,
        )
        return state, info.update(reward=reward)


def _agent(
    strategy: str = "open_es", parameterization: str = "policy", **kwargs: Any
) -> ESAgent:
    options = dict(
        env=_Environment(),
        strategy=strategy,
        parameterization=parameterization,
        population_size=4,
        num_rollouts=2,
        total_timesteps=64,
        eval_freq=31,
        eval_n_envs=2,
        hidden_sizes=(3,),
        num_knots=2,
        sigma=0.1,
        source_times=jnp.arange(4, dtype=jnp.float32),
    )
    options.update(kwargs)
    return ESAgent.create(**options)


def _assert_trees_close(actual: Any, expected: Any) -> None:
    assert jax.tree.structure(actual) == jax.tree.structure(expected)
    for left, right in zip(
        jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
    ):
        np.testing.assert_allclose(left, right, rtol=2e-5, atol=1e-6, equal_nan=True)


@pytest.mark.parametrize("parameterization", ["policy", "open_loop"])
def test_controller_initialization_is_independent_of_reset_defaults(
    parameterization: str,
) -> None:
    class ResetState(NamedTuple):
        prev_action: jax.Array

    class ResetDefaultsEnv(_Environment):
        def __init__(self, initial_action: float) -> None:
            super().__init__()
            self.initial_action = initial_action
            self.reset_calls = 0

        @property
        def unwrapped(self) -> "ResetDefaultsEnv":
            return self

        def init(self, key: jax.Array) -> tuple[ResetState, _Info]:
            self.reset_calls += 1
            _, info = super().init(key)
            return ResetState(jnp.asarray([self.initial_action], jnp.float32)), info

    envs = [ResetDefaultsEnv(value) for value in (-1.0, 0.8)]
    agents = [_agent(parameterization=parameterization, env=env) for env in envs]
    key = jax.random.key(3)
    states = [agent.init_state(key) for agent in agents]
    _assert_trees_close(states[0].params, states[1].params)
    assert [env.reset_calls for env in envs] == [0, 0]
    for agent, state in zip(agents, states, strict=True):
        act = agent.make_act(state)
        for obs in (jnp.zeros(2, jnp.float32), jnp.asarray([0.5, 1.5], jnp.float32)):
            np.testing.assert_array_equal(act(obs, key), jnp.zeros(1, jnp.float32))
    if parameterization == "open_loop":
        np.testing.assert_array_equal(states[0].params, np.zeros((2, 1), np.float32))
        assert states[0].params.dtype == jnp.float32


@pytest.mark.parametrize(
    "options, message",
    [
        ({"total_timesteps": 31}, "at least one complete generation"),
        ({"population_size": 3}, "population_size must be even"),
    ],
)
def test_invalid_population_or_insufficient_budget_is_rejected(
    options: dict[str, int], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _agent(**options)


@pytest.mark.parametrize("strategy", ["open_es", "cma_es"])
@pytest.mark.parametrize("parameterization", ["policy", "open_loop"])
def test_full_training_eager_jit_and_seed_vmap_agree(
    strategy: str, parameterization: str
) -> None:
    agent = _agent(strategy, parameterization)
    keys = jax.random.split(jax.random.key(7), 2)
    if parameterization == "policy":
        initial_params = [agent.init_state(key).params for key in keys]
        assert any(
            not np.array_equal(left, right)
            for left, right in zip(
                jax.tree.leaves(initial_params[0]),
                jax.tree.leaves(initial_params[1]),
                strict=True,
            )
        )
    eager = [agent.train(key) for key in keys]
    compiled = jax.jit(agent.train)(keys[0])
    batched = jax.jit(jax.vmap(agent.train))(keys)
    _assert_trees_close(eager[0], compiled)
    _assert_trees_close(compiled, jax.jit(agent.train)(keys[0]))
    for index in range(2):
        _assert_trees_close(
            eager[index], jax.tree.map(lambda leaf, i=index: leaf[i], batched)
        )

    state, results = compiled
    np.testing.assert_array_equal(state.global_step, 64)
    np.testing.assert_array_equal(state.alive_steps, 64)
    np.testing.assert_array_equal(state.update_index, 2)
    np.testing.assert_array_equal(state.failed, False)
    np.testing.assert_array_equal(results["global_step"], [0, 32, 64])
    assert any(
        not np.array_equal(left, right)
        for left, right in zip(
            jax.tree.leaves(eager[0][0].params),
            jax.tree.leaves(eager[1][0].params),
            strict=True,
        )
    )


@pytest.mark.parametrize("parameterization", ["policy", "open_loop"])
@pytest.mark.parametrize("terminal_at, expected_return", [(4, 10.0), (2, 3.0)])
def test_rollouts_sum_full_episode_and_exclude_terminal_padding(
    parameterization: str, terminal_at: int, expected_return: float
) -> None:
    agent = _agent(
        parameterization=parameterization,
        env=_ClockRewardEnvironment(terminal_at=terminal_at),
    )
    state = agent.init_state(jax.random.key(1))
    keys = jax.random.split(jax.random.key(2), 2)
    returns, lengths = jax.jit(agent._rollouts)(state.params, keys)
    np.testing.assert_allclose(returns, [expected_return] * 2, rtol=0, atol=0)
    np.testing.assert_array_equal(lengths, [terminal_at] * 2)


@pytest.mark.parametrize("strategy", ["open_es", "cma_es"])
def test_budget_counts_candidates_and_rollouts_and_stops_at_last_generation(
    strategy: str,
) -> None:
    agent = _agent(
        strategy,
        "open_loop",
        env=_Environment(terminal_at=2),
        total_timesteps=101,
        eval_freq=35,
    )
    state, results = jax.jit(agent.train)(jax.random.key(3))
    np.testing.assert_array_equal(state.global_step, 96)
    np.testing.assert_array_equal(state.alive_steps, 48)
    np.testing.assert_array_equal(state.update_index, 3)
    np.testing.assert_array_equal(results["global_step"], [0, 64, 96])


@pytest.mark.parametrize("strategy", ["open_es", "cma_es"])
def test_evaluation_cadence_does_not_change_training(strategy: str) -> None:
    dense = _agent(strategy, "open_loop", eval_freq=1)
    sparse = dataclasses.replace(dense, eval_freq=64)
    key = jax.random.key(4)
    dense_state, _ = jax.jit(dense.train)(key)
    sparse_state, _ = jax.jit(sparse.train)(key)
    _assert_trees_close(dense_state, sparse_state)


@pytest.mark.parametrize("parameterization", ["policy", "open_loop"])
def test_open_es_candidates_share_random_reset_seeds(
    parameterization: str,
) -> None:
    agent = _agent(parameterization=parameterization, env=_RandomRewardEnvironment())
    initial = agent.init_state(jax.random.key(5))
    updated = jax.jit(agent.update)(initial, jax.random.key(6))
    # Return depends on reset randomness only: all candidates must tie.
    _assert_trees_close(updated.params, initial.params)
    np.testing.assert_array_equal(updated.failed, False)
    np.testing.assert_array_equal(updated.global_step, 32)

    keys = jax.random.split(jax.random.key(6), 2)
    fresh_keys = jax.random.split(jax.random.key(7), 2)
    first, _ = agent._rollouts(initial.params, keys)
    repeated, _ = agent._rollouts(initial.params, keys)
    fresh, _ = agent._rollouts(initial.params, fresh_keys)
    np.testing.assert_array_equal(first, repeated)
    assert not np.array_equal(first, fresh)


@pytest.mark.parametrize("strategy", ["open_es", "cma_es"])
@pytest.mark.parametrize("invalid_reward", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_raw_returns_reject_generation_and_stop_later_work(
    strategy: str, invalid_reward: float
) -> None:
    agent = _agent(
        strategy,
        "open_loop",
        env=_InvalidRewardEnvironment(invalid_reward),
        total_timesteps=96,
    )
    key = jax.random.key(8)
    initial = agent.init_state(key)
    state, results = jax.jit(agent.train)(key)
    _assert_trees_close(state.params, initial.params)
    _assert_trees_close(state.es_state, initial.es_state)
    np.testing.assert_array_equal(state.failed, True)
    np.testing.assert_array_equal(state.failure_step, 32)
    np.testing.assert_array_equal(state.global_step, 32)
    np.testing.assert_array_equal(state.alive_steps, 32)
    np.testing.assert_array_equal(state.update_index, 1)
    np.testing.assert_array_equal(results["global_step"], [0, 32, 32, 32])


def test_nonfinite_proposed_update_retains_last_search_state() -> None:
    agent = _agent("open_es", "open_loop", learning_rate=float("inf"))
    initial = agent.init_state(jax.random.key(12))
    updated = jax.jit(agent.update)(initial, jax.random.key(13))
    _assert_trees_close(updated.params, initial.params)
    _assert_trees_close(updated.es_state, initial.es_state)
    np.testing.assert_array_equal(updated.failed, True)
    np.testing.assert_array_equal(updated.failure_step, 32)
    np.testing.assert_array_equal(updated.global_step, 32)
    np.testing.assert_array_equal(updated.alive_steps, 32)
    _assert_trees_close(jax.jit(agent.update)(updated, jax.random.key(14)), updated)


@pytest.mark.parametrize("strategy", ["open_es", "cma_es"])
@pytest.mark.parametrize("parameterization", ["policy", "open_loop"])
def test_inference_uses_current_mean_controller_parameters(
    strategy: str, parameterization: str
) -> None:
    agent = _agent(strategy, parameterization)
    state, _ = jax.jit(agent.train)(jax.random.key(9))
    _assert_trees_close(state.params, agent.es.get_mean(state.es_state))
    obs = jnp.asarray([0.25, 2.0], jnp.float32)
    act = agent.make_act(state)
    actual = jax.jit(act)(obs, jax.random.key(10))
    expected = agent.controller.action(state.params, obs)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)
    np.testing.assert_array_equal(actual, act(obs, jax.random.key(11)))
    assert actual.shape == agent.env.action_space.shape
    assert np.all(np.abs(actual) <= 1.0)


@pytest.mark.parametrize("strategy", ["open_es", "cma_es"])
def test_search_stays_float32_while_environment_retains_float64(
    strategy: str,
) -> None:
    class Float64Environment(_Environment):
        def init(self, key: jax.Array) -> tuple[_State, _Info]:
            state, _ = super().init(key)
            state = state._replace(
                position=state.position.astype(jnp.float64),
                goal=state.goal.astype(jnp.float64),
            )
            return state, self.info(state, jnp.asarray(0.0, jnp.float64))

    with jax.enable_x64(True):
        agent = _agent(strategy, "open_loop", env=Float64Environment())
        state = agent.init_state(jax.random.key(15))
        returns, _ = jax.jit(agent._rollouts)(
            state.params, jax.random.split(jax.random.key(16), 2)
        )
        assert returns.dtype == jnp.float64
        updated = jax.jit(agent.update)(state, jax.random.key(17))
        np.testing.assert_array_equal(updated.failed, False)
        for leaf in jax.tree.leaves((updated.params, updated.es_state)):
            if jnp.issubdtype(jnp.asarray(leaf).dtype, jnp.floating):
                assert jnp.asarray(leaf).dtype == jnp.float32
