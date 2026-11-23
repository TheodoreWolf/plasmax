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
    policy_action,
)
from agents.direct_gradient import (
    apply_policy_optimizer_update,
    finite_mean_gradients,
    gradient_statistics,
    make_knot_chunk,
    make_parameterization,
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
def test_controller_initialization_is_independent_of_reset_defaults(kind: str) -> None:
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

        def init(self, rng: jax.Array) -> tuple[ResetState, _Info]:
            self.reset_calls += 1
            _, info = super().init(rng)
            return ResetState(jnp.asarray([self.initial_action], jnp.float32)), info

    envs = [ResetDefaultsEnv(value) for value in (-1.0, 0.8)]
    agents = [_agent(kind, env=env) for env in envs]
    key = jax.random.key(3)
    states = [agent.init_state(key) for agent in agents]
    _assert_trees_close(states[0].params, states[1].params)
    assert [env.reset_calls for env in envs] == [0, 0]
    observations = jnp.asarray([[0.0, 0.0], [0.5, 1.5]], jnp.float32)
    actions = [
        jax.vmap(agent.make_act(state), in_axes=(0, None))(observations, key)
        for agent, state in zip(agents, states, strict=True)
    ]
    np.testing.assert_array_equal(actions[0], actions[1])
    np.testing.assert_array_equal(actions[0][0], jnp.zeros(1, jnp.float32))
    if kind == "open_loop":
        np.testing.assert_array_equal(actions[0], jnp.zeros((2, 1), jnp.float32))
        np.testing.assert_array_equal(states[0].params, np.zeros((2, 1), np.float32))
        assert states[0].params.dtype == jnp.float32
    else:
        assert np.any(np.asarray(actions[0][1]) != 0)


def test_compiled_policy_chunk_keeps_float32_action_boundary_for_float64_obs() -> None:
    class Float64Observations(_Environment):
        def info(self, state: _State, reward: jax.Array) -> _Info:
            info = super().info(state, reward)
            return info.update(obs=info.obs.astype(jnp.float64))

    with jax.enable_x64():
        env = Float64Observations(terminal_at=1)
        env.action_space = Continuous(
            low=jnp.asarray([-1.0], jnp.float64),
            high=jnp.asarray([1.0], jnp.float64),
        )
        agent = _agent(env=env)
        key = jax.random.key(2)
        state = agent.init_state(key)
        carry = agent.chunk.initialize(key)
        expected_action = agent.make_act(state)(carry.obs, key)
        reward, (next_carry, trajectory) = jax.jit(agent.chunk.run)(state.params, carry)
    assert next_carry.obs.dtype == jnp.float64
    assert reward.dtype == jnp.float32
    assert trajectory.action.dtype == jnp.float32
    np.testing.assert_array_equal(trajectory.alive, [True, False])
    np.testing.assert_allclose(
        trajectory.action[0], expected_action, rtol=1e-6, atol=1e-7
    )
    np.testing.assert_array_equal(trajectory.action[1], np.zeros(1, np.float32))


def test_policy_mean_has_unit_bias_sensitivity_and_clipped_actions() -> None:
    agent = _agent()
    state = agent.init_state(jax.random.key(3))
    obs = jnp.asarray([0.5, 1.5], jnp.float32)

    def action(bias: jax.Array) -> jax.Array:
        params = {
            **state.params,
            "action_mean": {
                "kernel": jnp.zeros_like(state.params["action_mean"]["kernel"]),
                "bias": bias,
            },
        }
        return policy_action(agent.policy, params, obs)

    derivative = jax.jacrev(action)(jnp.zeros(1, jnp.float32))
    np.testing.assert_allclose(derivative, [[1.0]], rtol=1e-6, atol=0)
    np.testing.assert_allclose(action(jnp.asarray([0.5])), [0.5], rtol=1e-6, atol=0)
    for bias in (-1e6, 1e6):
        value = action(jnp.asarray([bias], jnp.float32))
        assert np.all(np.isfinite(value))
        assert np.all(value >= -1.0) and np.all(value <= 1.0)


def test_benchmark_knots_start_at_zero_without_environment_reset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_reset(self: _Environment, rng: jax.Array) -> None:
        raise AssertionError("knot initialization must not reset the environment")

    monkeypatch.setattr(_Environment, "init", unexpected_reset)
    to_actions, params, _ = make_parameterization(_Environment(), 4, n_knots=2)
    np.testing.assert_array_equal(params, np.zeros((2, 1), np.float32))
    assert params.dtype == jnp.float32
    np.testing.assert_array_equal(to_actions(params), np.zeros((4, 1), np.float32))
    derivative = jax.grad(lambda theta: to_actions(theta).sum())(params)
    np.testing.assert_allclose(derivative, [[2.0], [2.0]], rtol=1e-6, atol=0)


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
    chunk = make_policy_chunk(agent.env, agent.policy, 2, remat=False)
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

    chunk = make_policy_chunk(Accumulator(), LinearPolicy(), 2, remat=False)
    carry = chunk.initialize(jax.random.key(0))
    grad = jax.grad(lambda p: chunk.run(p, carry)[0])(jnp.asarray([0.5], jnp.float32))
    # x1=x0+tanh(p*x0), x2=x1+tanh(p*x1); differentiate both rewards.
    x0, p = 0.25, 0.5
    x1 = x0 + np.tanh(p * x0)
    dx1 = x0 * (1 - np.tanh(p * x0) ** 2)
    expected = 2 * dx1 + (1 - np.tanh(p * x1) ** 2) * (x1 + p * dx1)
    np.testing.assert_allclose(grad, [expected], rtol=1e-6, atol=0)


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


def test_raw_gradient_statistics_distinguish_zero_and_missing_signal():
    raw = {
        "weights": jnp.asarray([[jnp.nan, jnp.inf], [0.0, 0.0], [2.0, 0.0]]),
        "bias": jnp.asarray([[-jnp.inf], [0.0], [-2.0]]),
    }
    aggregate, stats = jax.jit(finite_mean_gradients)(raw)
    np.testing.assert_array_equal(stats.nonfinite_elements, 3)
    np.testing.assert_array_equal(stats.total_elements, 9)
    np.testing.assert_array_equal(stats.zero_elements, 4)
    np.testing.assert_array_equal(stats.zero_rollouts, 1)
    np.testing.assert_array_equal(stats.all_nonfinite_rollouts, 1)
    np.testing.assert_array_equal(stats.aggregate_zero, False)
    np.testing.assert_array_equal(aggregate["weights"], [1.0, 0.0])
    np.testing.assert_array_equal(aggregate["bias"], [-1.0])

    invalid = jax.tree.map(lambda leaf: jnp.full_like(leaf, jnp.nan), raw)
    filtered, filtered_stats = jax.jit(finite_mean_gradients)(invalid)
    unfiltered_stats = jax.jit(gradient_statistics)(
        invalid, jax.tree.map(lambda leaf: jnp.mean(leaf, axis=0), invalid)
    )
    np.testing.assert_array_equal(filtered_stats.zero_elements, 0)
    np.testing.assert_array_equal(filtered_stats.zero_rollouts, 0)
    np.testing.assert_array_equal(filtered_stats.all_nonfinite_rollouts, 3)
    np.testing.assert_array_equal(filtered_stats.all_missing_elements, 3)
    np.testing.assert_array_equal(filtered_stats.aggregate_zero, True)
    np.testing.assert_array_equal(unfiltered_stats.aggregate_zero, False)
    np.testing.assert_array_equal(unfiltered_stats.aggregate_finite, False)
    for leaf in jax.tree.leaves(filtered):
        np.testing.assert_array_equal(leaf, jnp.zeros_like(leaf))


def test_both_parameterizations_share_filtered_adam_and_keep_momentum() -> None:
    class Carry(NamedTuple):
        gradients: jax.Array
        step: jax.Array

    class Trajectory(NamedTuple):
        alive: jax.Array

    @jax.custom_vjp
    def loss(params: jax.Array, gradients: jax.Array) -> jax.Array:
        return jnp.asarray(0.0, jnp.float32)

    def forward(params: jax.Array, gradients: jax.Array) -> tuple:
        return loss(params, gradients), gradients

    def backward(gradients: jax.Array, cotangent: jax.Array) -> tuple:
        return cotangent * gradients, jnp.zeros_like(gradients)

    loss.defvjp(forward, backward)

    def run(params: jax.Array, carry: Carry, start_step: jax.Array) -> tuple:
        del start_step
        return -loss(params, carry.gradients), (
            carry._replace(step=carry.step + 1),
            Trajectory(jnp.asarray([True])),
        )

    agents = [_agent(kind) for kind in ("policy", "open_loop")]
    params = jnp.asarray([1.0, -1.0], jnp.float32)
    states = []
    for agent in agents:
        assert agent.learning_rate == 1e-4
        object.__setattr__(agent, "chunk", dataclasses.replace(agent.chunk, run=run))
        states.append(
            agent.init_state(jax.random.key(0)).replace(
                params=params, opt_state=agent.optimizer.init(params)
            )
        )
    update = [jax.jit(agent.update) for agent in agents]
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-4))
    expected_params = params
    expected_opt = optimizer.init(params)
    batches = (
        ([[2.0, 4.0], [4.0, 2.0]], [3.0, 3.0]),
        ([[jnp.nan, 6.0], [2.0, jnp.inf]], [2.0, 6.0]),
        ([[jnp.nan, jnp.inf], [-jnp.inf, jnp.nan]], [0.0, 0.0]),
    )
    for index, (raw, expected_gradient) in enumerate(batches):
        carries = Carry(jnp.asarray(raw, jnp.float32), jnp.full(2, index, jnp.int32))
        expected_updates, expected_opt = optimizer.update(
            jnp.asarray(expected_gradient, jnp.float32), expected_opt, expected_params
        )
        expected_params = optax.apply_updates(expected_params, expected_updates)
        results = [
            apply(state, carries, jnp.asarray(0))
            for apply, state in zip(update, states, strict=True)
        ]
        _assert_trees_close(results[0], results[1])
        for state, next_carries, metrics, safe in results:
            _assert_trees_close(
                (state.params, state.opt_state), (expected_params, expected_opt)
            )
            np.testing.assert_array_equal(next_carries.step, [index + 1, index + 1])
            np.testing.assert_array_equal(safe, True)
            np.testing.assert_array_equal(metrics["train/accepted_updates"], 1)
            np.testing.assert_array_equal(metrics["train/alive_steps"], 2)
        states = [result[0] for result in results]
    metrics = results[0][2]
    np.testing.assert_array_equal(metrics["train/raw_grad_finite_rate"], 0)
    np.testing.assert_array_equal(metrics["train/raw_zero_grad_elements"], 0)
    np.testing.assert_array_equal(metrics["train/raw_zero_grad_rollouts"], 0)
    np.testing.assert_array_equal(metrics["train/raw_all_nonfinite_grad_rollouts"], 2)
    np.testing.assert_array_equal(metrics["train/zero_grad_update_rate"], 1)
    assert float(metrics["train/parameter_change_norm"]) > 0


@pytest.mark.parametrize("kind", ["policy", "open_loop"])
def test_update_diagnostic_counts_sum_while_rates_average_over_intervals(kind):
    agent = _agent(kind)
    _, results = jax.jit(agent.train)(jax.random.key(3))
    metrics = results["diagnostics"]
    np.testing.assert_array_equal(metrics["train/attempted_updates"], [0, 2, 2])
    np.testing.assert_array_equal(metrics["train/accepted_updates"], [0, 2, 2])
    np.testing.assert_array_equal(metrics["train/update_accepted_rate"], [0, 1, 1])
    np.testing.assert_array_equal(metrics["train/raw_grad_finite_rate"], [0, 1, 1])
    np.testing.assert_array_equal(metrics["train/zero_grad_update_rate"], [0, 0, 0])
    assert np.all(np.asarray(metrics["train/parameter_change_norm"])[1:] > 0)


@pytest.mark.parametrize("kind", ["policy", "open_loop"])
def test_unsafe_update_diagnostics_report_no_committed_parameter_change(kind):
    agent = _agent(kind, env=_Environment(fail=True), eval_freq=4)
    _, results = jax.jit(agent.train)(jax.random.key(5))
    metrics = results["diagnostics"]
    np.testing.assert_array_equal(metrics["train/attempted_updates"], [0, 1, 1, 0, 0])
    np.testing.assert_array_equal(metrics["train/accepted_updates"], [0, 1, 0, 0, 0])
    np.testing.assert_array_equal(metrics["train/parameter_change_norm"][2:], [0, 0, 0])
    np.testing.assert_array_equal(
        metrics["train/parameter_change_fraction"][2:], [0, 0, 0]
    )
