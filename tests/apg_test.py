"""Concurrent APG gradients and receding-horizon collection contracts."""

from __future__ import annotations

from typing import Any, NamedTuple

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from envelope import Continuous, InfoContainer

from agents.apg import APGAgent


class _State(NamedTuple):
    x: jax.Array
    steps: jax.Array
    resets: jax.Array
    rng: jax.Array


class _Accumulator:
    max_steps = 5

    def __init__(
        self,
        terminal_at: int = 5,
        *,
        truncated: bool = False,
        stochastic: bool = False,
        action_bounds: tuple[float, float] = (-1.0, 1.0),
    ) -> None:
        self.terminal_at = terminal_at
        self.truncated = truncated
        self.stochastic = stochastic
        self.action_space = Continuous(
            low=jnp.asarray([action_bounds[0]], jnp.float32),
            high=jnp.asarray([action_bounds[1]], jnp.float32),
        )
        self.observation_space = Continuous(
            low=jnp.asarray([-jnp.inf], jnp.float32),
            high=jnp.asarray([jnp.inf], jnp.float32),
        )

    def _info(self, state: _State, reward: jax.Array) -> InfoContainer:
        boundary = state.steps >= self.terminal_at
        return InfoContainer(
            obs=state.x[None].astype(jnp.float32),
            reward=reward.astype(jnp.float32),
            terminated=boundary & ~jnp.asarray(self.truncated),
            truncated=boundary & jnp.asarray(self.truncated),
        ).update(termination_code=jnp.where(boundary, 1, -1))

    def init(self, rng: jax.Array) -> tuple[_State, InfoContainer]:
        state = _State(
            jnp.asarray(0.25, jnp.float32),
            jnp.int32(0),
            jnp.int32(0),
            rng,
        )
        return state, self._info(state, jnp.asarray(0.0, jnp.float32))

    def reset(self, state: _State, rng: jax.Array) -> tuple[_State, InfoContainer]:
        reset, _ = self.init(rng)
        reset = reset._replace(resets=state.resets + 1)
        return reset, self._info(reset, jnp.asarray(0.0, jnp.float32))

    def step(self, state: _State, action: jax.Array) -> tuple[_State, InfoContainer]:
        rng, noise_rng = jax.random.split(state.rng)
        noise = (
            0.01 * jax.random.normal(noise_rng, (), dtype=state.x.dtype)
            if self.stochastic
            else jnp.zeros_like(state.x)
        )
        next_state = state._replace(
            x=state.x + action[0] + noise, steps=state.steps + 1, rng=rng
        )
        # A lookahead must never include rewards beyond an episode boundary.
        reward = jnp.where(state.steps >= self.terminal_at, jnp.nan, next_state.x)
        return next_state, self._info(next_state, reward)


def _agent(**kwargs: Any) -> APGAgent:
    options = dict(
        env=_Accumulator(),
        num_rollouts=2,
        horizon=3,
        total_timesteps=4,
        eval_freq=2,
        eval_n_envs=1,
        hidden_sizes=(),
        learning_rate=0.01,
        momentum=0.0,
        remat=False,
    )
    options.update(kwargs)
    return APGAgent.create(**options)


def _assert_tree_close(actual: Any, expected: Any) -> None:
    def numeric_keys(tree: Any) -> Any:
        return jax.tree.map(
            lambda x: (
                jax.random.key_data(x)
                if hasattr(x, "dtype")
                and jax.dtypes.issubdtype(x.dtype, jax.dtypes.prng_key)
                else x
            ),
            tree,
        )

    chex.assert_trees_all_close(
        numeric_keys(actual), numeric_keys(expected), rtol=2e-5, atol=1e-6
    )


@pytest.mark.parametrize("remat", [False, True])
def test_sequence_gradient_includes_all_future_rewards_without_feedback(
    remat: bool,
) -> None:
    agent = _agent(remat=remat)
    initial = agent.init_state(jax.random.key(1))
    env_state = jax.tree.map(lambda x: x[0], initial.env_state)
    params = initial.actor_ts.params
    biases = jnp.asarray([0.2, 0.3, -0.1], jnp.float32)
    params = {
        **params,
        "action_mean": {**params["action_mean"], "bias": biases},
    }
    loss, grads = jax.value_and_grad(
        lambda p: agent._rollout_loss(p, env_state, initial.obs[0])[0]
    )(params)
    # x_(t+1) = x_t + tanh(b_t + w_t*x_0): each action affects all
    # subsequent rewards. Concurrent prediction always uses x_0 = 0.25.
    weights = jnp.asarray([3.0, 2.0, 1.0], jnp.float32)
    expected_bias_grad = -weights * (1 - jnp.tanh(biases) ** 2)
    np.testing.assert_allclose(
        loss, -0.75 - jnp.dot(weights, jnp.tanh(biases)), rtol=1e-6, atol=0
    )
    np.testing.assert_allclose(
        grads["action_mean"]["bias"], expected_bias_grad, rtol=1e-6, atol=0
    )
    np.testing.assert_allclose(
        grads["action_mean"]["kernel"],
        0.25 * expected_bias_grad[None],
        rtol=1e-6,
        atol=0,
    )


def test_update_commits_the_first_preupdate_transition_with_its_original_rng() -> None:
    agent = _agent(env=_Accumulator(stochastic=True))
    state = agent.init_state(jax.random.key(2))
    actions = jax.vmap(agent.make_act(state))(
        state.obs, jax.random.split(jax.random.key(3), agent.num_rollouts)
    )
    expected_state, expected_info = jax.vmap(agent.env.step)(state.env_state, actions)
    candidate, metrics, safe = jax.jit(agent.update)(state)
    _assert_tree_close(candidate.env_state, expected_state)
    np.testing.assert_array_equal(candidate.obs, expected_info.obs)
    np.testing.assert_array_equal(candidate.env_state.steps, [1, 1])
    np.testing.assert_array_equal(metrics["train/simulator_steps"], 6)
    np.testing.assert_array_equal(safe, True)
    assert not np.array_equal(
        candidate.actor_ts.params["action_mean"]["bias"],
        state.actor_ts.params["action_mean"]["bias"],
    )


@pytest.mark.parametrize("truncated", [False, True])
def test_lookahead_freezes_at_boundary_and_only_committed_boundary_resets(
    truncated: bool,
) -> None:
    agent = _agent(env=_Accumulator(terminal_at=2, truncated=truncated), horizon=4)
    initial = agent.init_state(jax.random.key(4))
    first, first_metrics, safe = jax.jit(agent.update)(initial)
    np.testing.assert_array_equal(safe, True)
    np.testing.assert_array_equal(first_metrics["train/simulator_steps"], 4)
    np.testing.assert_array_equal(first.env_state.steps, [1, 1])
    np.testing.assert_array_equal(first.env_state.resets, [0, 0])

    second, second_metrics, safe = jax.jit(agent.update)(first)
    np.testing.assert_array_equal(safe, True)
    np.testing.assert_array_equal(second_metrics["train/simulator_steps"], 2)
    np.testing.assert_array_equal(second.env_state.steps, [0, 0])
    np.testing.assert_array_equal(second.env_state.resets, [1, 1])
    assert np.all(np.isfinite(second_metrics["train/loss"]))


def test_action_sequence_uses_actual_bounds_and_default_midpoint() -> None:
    agent = _agent(env=_Accumulator(action_bounds=(2.0, 6.0)))
    state = agent.init_state(jax.random.key(5))
    np.testing.assert_allclose(
        agent.action_sequence(state.actor_ts.params, state.obs[0]),
        jnp.full((3, 1), 4.0),
        rtol=1e-6,
        atol=0,
    )
    gradients = jax.grad(
        lambda params: agent.action_sequence(params, state.obs[0]).sum()
    )(state.actor_ts.params)
    np.testing.assert_allclose(
        gradients["action_mean"]["bias"], jnp.full(3, 2.0), rtol=1e-6, atol=0
    )
    params = jax.tree.map(lambda x: jnp.full_like(x, 1e6), state.actor_ts.params)
    sequence = agent.action_sequence(params, state.obs[0])
    assert np.all(sequence >= 2.0) and np.all(sequence <= 6.0)


def test_policy_construction_does_not_probe_environment_defaults() -> None:
    class UninitializedAccumulator(_Accumulator):
        plasmax_config = object()

        @property
        def unwrapped(self) -> UninitializedAccumulator:
            return self

        def init(self, rng: jax.Array) -> tuple[_State, InfoContainer]:
            raise AssertionError("policy construction must not initialize the env")

    agent = _agent(env=UninitializedAccumulator())
    params = agent.policy.init(jax.random.key(5), jnp.zeros(1, jnp.float32))["params"]
    np.testing.assert_array_equal(agent.action_sequence(params, jnp.zeros(1)), 0.0)


def test_training_replans_each_step_and_reports_committed_and_lookahead_counts() -> (
    None
):
    agent = _agent(total_timesteps=3, horizon=7, eval_freq=1)
    initial = agent.init_state(jax.random.key(6))
    one, _, _ = agent.update(initial)
    two, _, _ = agent.update(one)
    final, results = jax.jit(agent.train)(jax.random.key(6))
    _assert_tree_close(final.env_state, two.env_state)
    _assert_tree_close(final.actor_ts, two.actor_ts)
    np.testing.assert_array_equal(final.env_state.steps, [2, 2])
    np.testing.assert_array_equal(final.global_step, 4)
    np.testing.assert_array_equal(final.simulator_steps, 18)
    np.testing.assert_array_equal(results["global_step"], [0, 2, 4])
    assert agent.horizon == 7


@pytest.mark.parametrize(
    ("option", "value", "message"),
    [
        ("learning_rate", 0.0, "learning_rate"),
        ("momentum", 1.0, "momentum"),
        ("grad_clip", -1.0, "grad_clip"),
    ],
)
def test_invalid_apg_options_raise(option: str, value: Any, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _agent(**{option: value})


@jax.custom_jvp
def _invalid_gradient(value: jax.Array, invalid: jax.Array) -> jax.Array:
    return value


@_invalid_gradient.defjvp
def _invalid_gradient_jvp(
    primals: tuple[jax.Array, jax.Array], tangents: tuple[jax.Array, jax.Array]
) -> tuple[jax.Array, jax.Array]:
    value, invalid = primals
    value_dot, _ = tangents
    slope = jnp.where(invalid, jnp.nan, jnp.ones_like(value))
    return value, slope * value_dot


class _InvalidGradientAccumulator(_Accumulator):
    def step(self, state: _State, action: jax.Array) -> tuple[_State, InfoContainer]:
        next_state, info = super().step(state, action)
        return next_state, info.update(
            reward=_invalid_gradient(info.reward, state.resets > 0)
        )


def test_one_nonfinite_rollout_gradient_rejects_the_entire_update() -> None:
    agent = _agent(env=_InvalidGradientAccumulator())
    initial = agent.init_state(jax.random.key(9))
    initial = initial.replace(
        env_state=initial.env_state._replace(resets=jnp.asarray([0, 1], jnp.int32))
    )
    (losses, _), per_rollout_grads = jax.vmap(
        jax.value_and_grad(agent._rollout_loss, has_aux=True), in_axes=(None, 0, 0)
    )(initial.actor_ts.params, initial.env_state, initial.obs)
    assert np.all(np.isfinite(losses))
    bias_grads = per_rollout_grads["action_mean"]["bias"]
    assert np.all(np.isfinite(bias_grads[0]))
    assert np.any(~np.isfinite(bias_grads[1]))

    final, metrics = jax.jit(agent._train_step)(initial)
    np.testing.assert_array_equal(final.failed, True)
    np.testing.assert_array_equal(metrics["train/grads_finite"], False)
    _assert_tree_close(final.actor_ts, initial.actor_ts)
    _assert_tree_close(final.env_state, initial.env_state)
    np.testing.assert_array_equal(final.failure_step, agent.num_rollouts)


def test_terminal_reset_rejects_update_and_preserves_last_active_state() -> None:
    class TerminalResetAccumulator(_Accumulator):
        def reset(self, state: _State, rng: jax.Array) -> tuple[_State, InfoContainer]:
            reset_state, info = super().reset(state, rng)
            return reset_state, info.update(terminated=jnp.asarray(True))

    agent = _agent(env=TerminalResetAccumulator(terminal_at=1))
    initial = agent.init_state(jax.random.key(10))
    failed, _ = jax.jit(agent._train_step)(initial)
    np.testing.assert_array_equal(failed.failed, True)
    np.testing.assert_array_equal(failed.failure_step, agent.num_rollouts)
    _assert_tree_close(failed.actor_ts, initial.actor_ts)
    _assert_tree_close(failed.env_state, initial.env_state)
    np.testing.assert_array_equal(failed.obs, initial.obs)

    stopped, metrics = jax.jit(agent._train_step)(failed)
    _assert_tree_close(stopped, failed)
    np.testing.assert_array_equal(metrics["train/simulator_steps"], 0)
