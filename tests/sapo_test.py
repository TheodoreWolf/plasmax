"""SAPO objectives and boundary semantics on an analytic scalar environment."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from envelope import Continuous
from flax import linen as nn

from agents.sapo import SAPOAgent, SAPOState, _soft_lambda_returns


class _State(NamedTuple):
    x: jax.Array
    steps: jax.Array
    episodes: jax.Array


class _Info(NamedTuple):
    obs: jax.Array
    reward: jax.Array
    terminated: jax.Array
    truncated: jax.Array

    def update(self, **kwargs: jax.Array) -> _Info:
        return self._replace(**kwargs)


class _Environment:
    max_steps = 3
    action_space = Continuous(
        low=jnp.asarray([-1.0], jnp.float32),
        high=jnp.asarray([1.0], jnp.float32),
    )
    observation_space = Continuous(
        low=jnp.asarray([-jnp.inf], jnp.float32),
        high=jnp.asarray([jnp.inf], jnp.float32),
    )

    def __init__(self, *, terminate: bool = False, reset_scale: float = 0.0) -> None:
        self.terminate = terminate
        self.reset_scale = reset_scale

    def _info(self, state: _State, reward: jax.Array, done: jax.Array) -> _Info:
        return _Info(
            state.x[None], reward, done & self.terminate, done & (not self.terminate)
        )

    def init(self, rng: jax.Array) -> tuple[_State, _Info]:
        del rng
        state = _State(jnp.asarray(1.0, jnp.float32), jnp.int32(0), jnp.int32(0))
        return state, self._info(state, jnp.float32(0.0), jnp.asarray(False))

    def reset(self, state: _State, rng: jax.Array) -> tuple[_State, _Info]:
        del rng
        x = state.x * self.reset_scale if self.reset_scale else jnp.float32(1.0)
        state = _State(x, jnp.int32(0), state.episodes + 1)
        return state, self._info(state, jnp.float32(0.0), jnp.asarray(False))

    def step(self, state: _State, action: jax.Array) -> tuple[_State, _Info]:
        state = state._replace(x=state.x + action[0], steps=state.steps + 1)
        return state, self._info(state, state.x, state.steps >= self.max_steps)


class _FeedbackActor(nn.Module):
    entropy: float = 0.0

    @nn.compact
    def __call__(self, obs: jax.Array, rng: jax.Array) -> tuple[jax.Array, jax.Array]:
        del rng
        theta = self.param(
            "theta", nn.initializers.constant(0.2, dtype=jnp.float32), ()
        )
        return theta * obs, jnp.full(obs.shape[0], -self.entropy, jnp.float32)


class _LinearValue(nn.Module):
    @nn.compact
    def __call__(self, obs: jax.Array) -> jax.Array:
        coefficient = self.param(
            "coefficient", nn.initializers.constant(1.0, dtype=jnp.float32), ()
        )
        return coefficient * obs[:, 0]


def _agent(**kwargs: object) -> SAPOAgent:
    config = dict(
        env=_Environment(),
        total_timesteps=4,
        eval_freq=4,
        num_rollouts=2,
        horizon=2,
        hidden_sizes=(4,),
        remat=False,
        eval_n_envs=1,
        critic_epochs=1,
        num_minibatches=1,
        normalize_observations=False,
    )
    config.update(kwargs)
    return SAPOAgent.create(**config)


def _analytic_agent(**kwargs: object) -> SAPOAgent:
    agent = _agent(gamma=1.0, **kwargs)
    object.__setattr__(agent, "actor", _FeedbackActor())
    object.__setattr__(agent, "critic", _LinearValue())
    return agent


def _analytic_state(agent: SAPOAgent) -> SAPOState:
    state = agent.init_state(jax.random.key(0))
    params = {"params": {"coefficient": jnp.asarray([2.0, 4.0], jnp.float32)}}
    return state.replace(critic_ts=state.critic_ts.replace(params=params))


@pytest.mark.parametrize("td_lambda", [0.0, 0.5, 1.0])
def test_soft_lambda_targets_stop_at_episode_and_chunk_boundaries(
    td_lambda: float,
) -> None:
    targets = jax.jit(_soft_lambda_returns)(
        jnp.asarray([[1.0], [2.0], [3.0]]),
        jnp.asarray([[10.0], [0.0], [20.0]]),
        jnp.asarray([[False], [True], [True]]),
        0.9,
        td_lambda,
    )
    expected = [[1.0 + 0.9 * ((1 - td_lambda) * 10 + td_lambda * 2)], [2], [21]]
    np.testing.assert_allclose(targets, expected, rtol=1e-6, atol=1e-6)


def test_actor_feedback_and_terminal_value_gradients_use_mean_critic() -> None:
    agent = _analytic_agent()
    state = _analytic_state(agent)
    (_, (_, _, _, batch)), grads = jax.value_and_grad(agent._actor_loss, has_aux=True)(
        state.actor_ts.params, state
    )
    # x1 = 1 + theta, x2 = (1 + theta)^2, terminal mean V = 3*x2.
    np.testing.assert_allclose(grads["params"]["theta"], -5.3, rtol=1e-6, atol=1e-6)
    # Critic targets retain the smaller head, rather than the actor's mean head.
    np.testing.assert_allclose(
        batch.next_value[:, 0], [2.4, 2.88], rtol=1e-6, atol=1e-6
    )


@pytest.mark.parametrize("num_rollouts", [1, 2])
@pytest.mark.parametrize("remat", [False, True])
def test_conditional_custom_jvp_dynamics_support_batched_actor_gradients(
    num_rollouts: int, remat: bool
) -> None:
    def dynamics(x: jax.Array) -> jax.Array:
        return jax.lax.cond(x > 0, lambda x: x, lambda x: 2 * x, x)

    # Like the physical stepper, this custom rule differentiates a conditional
    # reference implementation. grad(vmap(step)) cannot transpose it in JAX
    # 0.10.1; differentiating each rollout before vmap preserves its gradients.
    differentiable_dynamics = jax.custom_jvp(dynamics)

    @differentiable_dynamics.defjvp
    def dynamics_jvp(primals: tuple, tangents: tuple) -> tuple:
        return jax.jvp(dynamics, primals, tangents)

    class CustomJVPEnvironment(_Environment):
        def step(self, state: _State, action: jax.Array) -> tuple[_State, _Info]:
            state, info = super().step(state, action)
            state = state._replace(x=differentiable_dynamics(state.x))
            return state, self._info(state, state.x, info.truncated)

    agent = _analytic_agent(
        env=CustomJVPEnvironment(), num_rollouts=num_rollouts, remat=remat
    )
    state = _analytic_state(agent)
    (_, _), grads = jax.jit(jax.value_and_grad(agent._actor_loss, has_aux=True))(
        state.actor_ts.params, state
    )
    np.testing.assert_allclose(grads["params"]["theta"], -5.3, rtol=1e-6, atol=1e-6)
    candidate, metrics, safe = jax.jit(agent.update)(state)
    assert bool(safe)
    np.testing.assert_allclose(
        metrics["train/actor_grad_norm"], 5.3, rtol=1e-6, atol=1e-6
    )
    np.testing.assert_array_equal(metrics["train/simulator_steps"], 2 * num_rollouts)
    assert float(candidate.actor_ts.params["params"]["theta"]) > 0.2
    states = jax.tree.map(lambda x: jnp.stack((x, x)), state)
    _, batched_metrics, batched_safe = jax.jit(jax.vmap(agent.update))(states)
    np.testing.assert_array_equal(batched_safe, [True, True])
    np.testing.assert_allclose(
        batched_metrics["train/actor_grad_norm"], 5.3, rtol=1e-6, atol=1e-6
    )


@pytest.mark.parametrize("terminated,expected", [(True, -1.0), (False, -4.0)])
def test_termination_zeroes_value_and_truncation_uses_pre_reset_observation(
    terminated: bool, expected: float
) -> None:
    env = _Environment(terminate=terminated)
    env.max_steps = 1
    agent = _analytic_agent(env=env, horizon=1)
    state = _analytic_state(agent)
    (_, (_, _, obs, _)), grads = jax.value_and_grad(agent._actor_loss, has_aux=True)(
        state.actor_ts.params, state
    )
    np.testing.assert_allclose(grads["params"]["theta"], expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(obs, np.ones((2, 1)))


def test_reset_state_cuts_gradient_but_rollout_continues() -> None:
    env = _Environment(terminate=True, reset_scale=10.0)
    env.max_steps = 1
    agent = _analytic_agent(env=env)
    state = _analytic_state(agent)
    (_, (_, final_env, _, _)), grads = jax.value_and_grad(
        agent._actor_loss, has_aux=True
    )(state.actor_ts.params, state)
    # Second episode starts at stop_gradient(12), yielding -(1 + 12) / 2.
    np.testing.assert_allclose(grads["params"]["theta"], -6.5, rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(final_env.episodes, [2, 2])


@pytest.mark.parametrize("entropy,increases", [(-2.0, True), (1.0, False)])
def test_temperature_uses_raw_entropy_and_moves_towards_target(
    entropy: float, increases: bool
) -> None:
    agent = _analytic_agent()
    object.__setattr__(agent, "actor", _FeedbackActor(entropy=entropy))
    state = _analytic_state(agent)
    candidate, _, safe = agent.update(state)
    assert bool(safe)
    alpha = float(jnp.exp(candidate.alpha_ts.params["log_alpha"]))
    assert (alpha > agent.initial_alpha) == increases


def test_frozen_normalization_and_stochastic_seeded_inference() -> None:
    agent = _agent(normalize_observations=True)
    state = agent.init_state(jax.random.key(1))
    rms = state.obs_rms_state.replace(mean=jnp.array([10.0]), var=jnp.array([4.0]))
    derivative = jax.grad(lambda x: agent._normalize(rms, x).sum())(jnp.array([2.0]))
    np.testing.assert_allclose(derivative, [0.5], rtol=2e-6, atol=1e-6)
    act = jax.jit(agent.make_act(state))
    key = jax.random.key(3)
    np.testing.assert_array_equal(
        act(jnp.array([1.0]), key), act(jnp.array([1.0]), key)
    )
    assert act(jnp.array([1.0]), key).shape == (1,)
    deterministic = agent.make_act(state, deterministic=True)
    np.testing.assert_array_equal(
        deterministic(jnp.array([1.0]), key),
        deterministic(jnp.array([1.0]), jax.random.key(4)),
    )


def test_tiny_training_jit_and_vmap_preserve_interface() -> None:
    agent = _agent(normalize_observations=True, remat=True)
    keys = jax.random.split(jax.random.key(2), 2)
    state, evaluation = jax.jit(agent.train)(keys[0])
    batched, batched_evaluation = jax.jit(jax.vmap(agent.train))(keys)
    assert not bool(state.failed)
    np.testing.assert_array_equal(state.global_step, 4)
    np.testing.assert_array_equal(state.simulator_steps, 4)
    np.testing.assert_allclose(
        state.actor_ts.params["params"]["action_mean"]["bias"],
        batched.actor_ts.params["params"]["action_mean"]["bias"][0],
        rtol=1e-5,
        atol=1e-6,
    )
    np.testing.assert_array_equal(
        evaluation["global_step"], batched_evaluation["global_step"][0]
    )


def test_nonfinite_reward_is_rejected_without_modifying_parameters() -> None:
    class InvalidReward(_Environment):
        def step(self, state: _State, action: jax.Array) -> tuple[_State, _Info]:
            state, info = super().step(state, action)
            return state, info._replace(reward=jnp.float32(jnp.nan))

    agent = _agent(env=InvalidReward())
    state = agent.init_state(jax.random.key(0))
    after, _ = jax.jit(agent._train_step)(state)
    assert bool(after.failed)
    for before_param, after_param in zip(
        jax.tree.leaves(state.actor_ts.params),
        jax.tree.leaves(after.actor_ts.params),
        strict=True,
    ):
        np.testing.assert_array_equal(before_param, after_param)


def test_terminal_reset_halts_collection_and_rejects_update() -> None:
    class InvalidReset(_Environment):
        max_steps = 1

        def reset(self, state: _State, rng: jax.Array) -> tuple[_State, _Info]:
            state, info = super().reset(state, rng)
            return state, info._replace(terminated=jnp.asarray(True))

    agent = _agent(env=InvalidReset(), horizon=3)
    state = agent.init_state(jax.random.key(0))
    candidate, metrics, safe = jax.jit(agent.update)(state)
    assert not bool(safe)
    np.testing.assert_array_equal(candidate.env_state.steps, [0, 0])
    np.testing.assert_array_equal(candidate.env_state.episodes, [1, 1])
    np.testing.assert_array_equal(metrics["train/simulator_steps"], 2)


def test_float64_native_reward_keeps_float32_training_carry() -> None:
    class DoubleReward(_Environment):
        def step(self, state: _State, action: jax.Array) -> tuple[_State, _Info]:
            state, info = super().step(state, action)
            return state, info._replace(reward=info.reward.astype(jnp.float64))

    with jax.enable_x64(True):
        agent = _agent(env=DoubleReward())
        state = agent.init_state(jax.random.key(0))
        (_, metrics, safe) = jax.jit(agent.update)(state)
        assert bool(safe)
        assert metrics["train/actor_loss"].dtype == jnp.float32
