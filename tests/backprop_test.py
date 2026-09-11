"""Pathwise training contracts on a small differentiable Envelope environment."""

import dataclasses
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from envelope import Continuous

from agents.backprop import (
    BackpropOpenLoopAgent,
    BackpropPolicyAgent,
    make_policy_chunk,
    open_loop_action,
)
from agents.direct_gradient import (
    apply_policy_optimizer_update,
    finite_mean_gradients,
    make_knot_chunk,
)
from plasmax.spaces import ObsLayout


class _Info(NamedTuple):
    obs: jax.Array
    reward: jax.Array
    terminated: jax.Array
    truncated: jax.Array

    def update(self, **kwargs):
        return self._replace(**kwargs)


class _State(NamedTuple):
    x: jax.Array
    step: jax.Array
    goal: jax.Array


class _Environment:
    max_steps = 4
    action_space = Continuous(
        low=jnp.asarray([-1.0], jnp.float32), high=jnp.asarray([1.0], jnp.float32)
    )
    observation_space = Continuous(
        low=jnp.full(2, -jnp.inf, jnp.float32), high=jnp.full(2, jnp.inf, jnp.float32)
    )

    def __init__(self, terminal_at: int = 4, fail: bool = False):
        self.terminal_at = terminal_at
        self.fail = fail

    def obs_layout(self):
        return ObsLayout(
            {},
            {"x": slice(0, 1), "elapsed_time": slice(1, 2)},
            (),
            ("x", "elapsed_time"),
        )

    def init(self, rng):
        state = _State(
            jax.random.uniform(rng, (), dtype=jnp.float32) / 10,
            jnp.asarray(0, jnp.int32),
            jax.random.normal(rng, (), dtype=jnp.float32),
        )
        return state, self.info(state, jnp.asarray(0.0, jnp.float32))

    def info(self, state, reward):
        return _Info(
            obs=jnp.stack((state.x, state.step.astype(jnp.float32))),
            reward=reward,
            terminated=state.step >= self.terminal_at,
            truncated=jnp.asarray(False),
        )

    def step(self, state, action):
        state = _State(state.x + action[0], state.step + 1, state.goal)
        reward = -jnp.square(state.x - state.goal)
        if self.fail:
            reward = jnp.where(state.step >= 3, jnp.nan, reward)
        return state, self.info(state, reward)


def _agent(kind="policy", **kwargs):
    args = dict(
        env=_Environment(),
        total_timesteps=16,
        eval_freq=7,
        num_rollouts=2,
        gradient_horizon=2,
        eval_n_envs=2,
        action_setpoint=jnp.zeros(1, jnp.float32),
        remat=False,
    )
    args.update(kwargs)
    if kind == "policy":
        return BackpropPolicyAgent.create(hidden_sizes=(3,), **args)
    return BackpropOpenLoopAgent.create(
        source_times=jnp.arange(4, dtype=jnp.float32), num_knots=2, **args
    )


def _assert_trees_close(actual, expected):
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        np.testing.assert_allclose(a, b, rtol=2e-5, atol=1e-6, equal_nan=True)


@pytest.mark.parametrize("kind", ["policy", "open_loop"])
def test_whole_training_eager_jit_and_seed_vmap_agree(kind):
    agent = _agent(kind)
    keys = jax.random.split(jax.random.key(7), 2)
    eager = [agent.train(key) for key in keys]
    compiled = jax.jit(agent.train)(keys[0])
    batched = jax.jit(jax.vmap(agent.train))(keys)
    _assert_trees_close(eager[0], compiled)
    for index in range(2):
        _assert_trees_close(
            eager[index], jax.tree.map(lambda leaf, i=index: leaf[i], batched)
        )
    state, results = compiled
    np.testing.assert_array_equal(state.global_step, 16)
    np.testing.assert_array_equal(state.alive_steps, 16)
    np.testing.assert_array_equal(results["global_step"], [0, 8, 16])
    np.testing.assert_array_equal(state.failed, False)
    assert any(
        not np.array_equal(a, b)
        for a, b in zip(
            jax.tree.leaves(eager[0][0].params),
            jax.tree.leaves(eager[1][0].params),
            strict=True,
        )
    )


@pytest.mark.parametrize("kind", ["policy", "open_loop"])
def test_evaluation_cadence_does_not_change_training_or_reset_boundaries(kind):
    dense = _agent(kind, env=_Environment(terminal_at=3), eval_freq=1)
    sparse = dataclasses.replace(dense, eval_freq=16)
    left, _ = jax.jit(dense.train)(jax.random.key(1))
    right, _ = jax.jit(sparse.train)(jax.random.key(1))
    _assert_trees_close(left, right)
    np.testing.assert_array_equal(left.alive_steps, 12)


def test_policy_update_matches_existing_single_rollout_gradient_then_adam():
    agent = _agent(total_timesteps=4)
    key = jax.random.key(2)
    initial = agent.init_state(key)
    pass_key = jax.random.fold_in(key, 0)
    keys = jax.vmap(lambda i: jax.random.fold_in(pass_key, i))(jnp.arange(2))
    chunk = make_policy_chunk(
        agent.env, agent.policy, agent.action_setpoint, 2, remat=False
    )
    carries = jax.vmap(chunk.initialize)(keys)
    per_rollout = jax.vmap(
        jax.grad(lambda params, carry: -chunk.run(params, carry)[0]), in_axes=(None, 0)
    )(initial.params, carries)
    grads, _ = finite_mean_gradients(per_rollout)
    expected, expected_opt, _ = apply_policy_optimizer_update(
        agent.optimizer, grads, initial.opt_state, initial.params, agent.grad_clip
    )
    actual, _ = jax.jit(agent.train)(key)
    _assert_trees_close((actual.params, actual.opt_state), (expected, expected_opt))


def test_policy_chunk_keeps_state_to_action_feedback_gradient():
    class LinearPolicy:
        def apply(self, variables, obs):
            return variables["params"] * obs[:1]

    class Accumulator(_Environment):
        def init(self, rng):
            del rng
            state = _State(
                jnp.asarray(0.25, jnp.float32), jnp.asarray(0), jnp.asarray(0.0)
            )
            return state, self.info(state, jnp.asarray(0.0, jnp.float32))

        def step(self, state, action):
            state = state._replace(x=state.x + action[0], step=state.step + 1)
            return state, self.info(state, state.x)

    chunk = make_policy_chunk(
        Accumulator(), LinearPolicy(), jnp.zeros(1, jnp.float32), 2, remat=False
    )
    carry = chunk.initialize(jax.random.key(0))
    grad = jax.grad(lambda p: chunk.run(p, carry)[0])(jnp.asarray([0.5], jnp.float32))
    # x1=.25(1+p), x2=.25(1+p)^2 => derivative=.25+.5(1+p).
    np.testing.assert_allclose(grad, [1.0], rtol=1e-6, atol=0)


@pytest.mark.parametrize("kind", ["policy", "open_loop"])
def test_numerical_failure_retains_last_valid_update_and_masks_later_work(kind):
    agent = _agent(kind, env=_Environment(fail=True), eval_freq=4)
    failed, results = jax.jit(agent.train)(jax.random.key(5))
    one_update = dataclasses.replace(agent, total_timesteps=4)
    valid, _ = jax.jit(one_update.train)(jax.random.key(5))
    np.testing.assert_array_equal(failed.failed, True)
    np.testing.assert_array_equal(failed.failure_step, 8)
    np.testing.assert_array_equal(failed.global_step, 8)
    np.testing.assert_array_equal(failed.update_index, 2)
    np.testing.assert_array_equal(results["global_step"], [0, 4, 8, 8, 8])
    _assert_trees_close(failed.params, valid.params)
    _assert_trees_close(failed.opt_state, valid.opt_state)


def test_open_loop_interpolates_source_time_before_theta_and_holds_endpoints():
    source_times = jnp.asarray([0.0, 0.2, 0.8, 1.0])
    theta = jnp.asarray([[-1.0], [1.0]])
    # Clock is at a non-final observation position, as with observation history.
    action = jax.jit(
        lambda t: open_loop_action(theta, jnp.asarray([99.0, t, -5.0]), source_times, 1)
    )
    np.testing.assert_allclose(
        action(0.2), jnp.tanh(jnp.asarray([-1 / 3])), rtol=1e-6, atol=1e-7
    )
    np.testing.assert_allclose(action(0.5), [0.0], rtol=0, atol=1e-7)
    np.testing.assert_allclose(action(-1.0), jnp.tanh(theta[0]), rtol=0, atol=0)
    np.testing.assert_allclose(action(3.0), jnp.tanh(theta[-1]), rtol=0, atol=0)


def test_open_loop_chunk_update_matches_baseline_gradient_mean_and_clipped_adam():
    agent = _agent("open_loop", total_timesteps=4)
    key = jax.random.key(8)
    state = agent.init_state(key)
    keys = jax.vmap(lambda i: jax.random.fold_in(jax.random.fold_in(key, 0), i))(
        jnp.arange(2)
    )
    chunk = make_knot_chunk(
        agent.env,
        lambda theta: jax.vmap(
            lambda t: open_loop_action(
                theta, jnp.asarray([0, t]), agent.source_times, 1
            )
        )(agent.source_times),
        2,
        remat=False,
    )
    carries = jax.vmap(chunk.initialize)(keys)
    grads = jax.vmap(
        jax.grad(lambda p, carry: -chunk.run(p, carry, jnp.asarray(0))[0]),
        in_axes=(None, 0),
    )(state.params, carries)
    updates, expected_opt = agent.optimizer.update(
        jnp.mean(grads, axis=0), state.opt_state, state.params
    )
    expected = optax.apply_updates(state.params, updates)
    actual, _ = jax.jit(agent.train)(key)
    _assert_trees_close((actual.params, actual.opt_state), (expected, expected_opt))


def test_eval_callback_receives_state_and_interval_diagnostics():
    def callback(agent, state, rng, diagnostics):
        del agent, rng
        return {"step": state.global_step, "alive": diagnostics["train/alive_steps"]}

    state, results = jax.jit(_agent(eval_callback=callback).train)(jax.random.key(0))
    np.testing.assert_array_equal(results["evaluation"]["step"], [0, 8, 16])
    np.testing.assert_array_equal(results["evaluation"]["alive"], [0, 8, 8])
    np.testing.assert_array_equal(state.failed, False)


@pytest.mark.parametrize("kind", ["policy", "open_loop"])
def test_vmapped_seed_failure_does_not_change_healthy_seed(kind):
    class SometimesFails(_Environment):
        def step(self, state, action):
            state, info = super().step(state, action)
            bad = (state.goal > 0) & (state.step >= 3)
            return state, info.update(reward=jnp.where(bad, jnp.nan, info.reward))

    agent = _agent(kind, env=SometimesFails(), total_timesteps=8, eval_freq=4)
    keys = jnp.stack([jax.random.key(0), jax.random.key(1)])
    batched, _ = jax.jit(jax.vmap(agent.train))(keys)
    np.testing.assert_array_equal(batched.failed, [False, True])
    healthy, _ = jax.jit(agent.train)(keys[0])
    _assert_trees_close(jax.tree.map(lambda leaf: leaf[0], batched), healthy)


def test_open_loop_reads_clock_index_from_current_history_layout():
    class HistoryEnvironment(_Environment):
        observation_space = Continuous(
            low=jnp.full(9, -jnp.inf), high=jnp.full(9, jnp.inf)
        )

        def obs_layout(self):
            # Three frames of (x, time), followed by three past actions.
            return ObsLayout(
                {},
                {"x": slice(4, 5), "elapsed_time": slice(5, 6)},
                (),
                ("x", "elapsed_time"),
                vector_size=9,
            )

    agent = _agent("open_loop", env=HistoryEnvironment())
    assert agent.time_index == 5
    state = agent.init_state(jax.random.key(0)).replace(
        params=jnp.asarray([[-1.0], [1.0]])
    )
    obs = jnp.asarray([0.0, 0.0, 0.0, 1.0, 0.0, 2.0, 0.9, 0.8, 0.7])
    np.testing.assert_allclose(
        agent.make_act(state)(obs, jax.random.key(0)),
        jnp.tanh(jnp.asarray([1 / 3])),
        rtol=1e-6,
        atol=1e-7,
    )
