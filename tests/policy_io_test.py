"""Policy artifacts restore inference without training templates or resets."""

from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from envelope import Discrete, TruncationWrapper
from flax import linen as nn
from flax import serialization
from helpers import CheapBoundaryEnv

from agents.backprop import BackpropOpenLoopAgent, BackpropPolicyAgent
from agents.mpc import MPCAgent
from agents.policy_io import load_policy, save_policy
from agents.ppo import PPOAdapter, ResidualGaussianPolicy
from agents.sac import SACAdapter
from plasmax import rewards
from plasmax.spaces import ObsLayout
from plasmax.wrappers import ObsHistoryWrapper, QuantizeActionWrapper
from training.envelope_gymnax import EnvelopeGymnax


class ScalarDiscreteEnv(CheapBoundaryEnv):
    @property
    def action_space(self):
        return Discrete(n=3)


class NamedObservationEnv(CheapBoundaryEnv):
    def obs_layout(self):
        return ObsLayout(
            {},
            {"P_fusion": slice(0, 1), "elapsed_time": slice(1, 2)},
            (),
            ("P_fusion", "elapsed_time"),
        )


class ConfiguredEnv(NamedObservationEnv):
    @property
    def plasmax_config(self):
        return {
            "environment_key": "mock/circular/smoke",
            "task": {
                "reward": "lh_transition",
                "terminal_penalty": -100.0,
            },
        }

    @property
    def _dynamics(self):
        return SimpleNamespace(_reward_fn=rewards.Q_fusion, _disruption_penalty=0.0)


def _rejax_agent(algorithm: str, action_kind: str):
    env_type = ScalarDiscreteEnv if action_kind == "discrete" else CheapBoundaryEnv
    env = env_type(obs_dim=2, action_low=(-2.0,), action_high=(3.0,))
    if action_kind == "multidiscrete":
        env = QuantizeActionWrapper(env=env, bin_counts=(3,))
    env = EnvelopeGymnax(TruncationWrapper(env=env, max_steps=2))
    common = dict(
        env=env,
        env_params=env.default_params,
        num_envs=1,
        total_timesteps=2,
        eval_freq=2,
        normalize_observations=True,
    )
    if algorithm == "ppo":
        agent = PPOAdapter.create(
            **common,
            num_steps=2,
            num_epochs=1,
            num_minibatches=1,
            agent_kwargs={"hidden_layer_sizes": (4,), "activation": "tanh"},
        )
        if action_kind == "residual":
            agent = agent.replace(
                actor=ResidualGaussianPolicy(
                    action_dim=1,
                    action_range=(-2.0, 3.0),
                    action_setpoint=(0.7,),
                    hidden_layer_sizes=(4,),
                    activation=nn.tanh,
                    initial_log_std=-0.7,
                )
            )
    else:
        agent = SACAdapter.create(
            **common,
            buffer_size=4,
            batch_size=1,
            fill_buffer=0,
            hidden_layer_sizes=(4,),
        )
    state = agent.init_state(jax.random.PRNGKey(0))
    return agent, state.replace(
        obs_rms_state=state.obs_rms_state.replace(
            mean=jnp.asarray([0.5, -0.7]),
            var=jnp.asarray([2.0, 0.3]),
        )
    )


@pytest.mark.parametrize(
    "algorithm,action_kind",
    [
        ("ppo", "continuous"),
        ("ppo", "residual"),
        ("ppo", "discrete"),
        ("ppo", "multidiscrete"),
        ("sac", "continuous"),
        ("sac", "discrete"),
    ],
)
@pytest.mark.parametrize("deterministic", [False, True])
def test_rejax_roundtrip_matches_actual_actor_and_normalization(
    tmp_path,
    monkeypatch,
    algorithm,
    action_kind,
    deterministic,
):
    agent, state = _rejax_agent(algorithm, action_kind)
    path = save_policy(
        agent,
        state,
        tmp_path / "policy.msgpack",
        metadata={"seed": 7},
        deterministic=deterministic,
    )
    expected = agent.make_act(state, deterministic=deterministic)

    def forbid_init(*args, **kwargs):
        raise AssertionError("loading must not initialize a model or environment")

    monkeypatch.setattr(nn.Module, "init", forbid_init)
    monkeypatch.setattr(CheapBoundaryEnv, "init", forbid_init)
    loaded = load_policy(path)
    restored = jax.jit(loaded.make_act())
    obs = jnp.asarray([0.2, 1.3], dtype=jnp.float32)
    for key in jax.random.split(jax.random.PRNGKey(1), 3):
        np.testing.assert_allclose(
            restored(obs, key), expected(obs, key), rtol=1e-6, atol=1e-6
        )
    assert restored(obs, key).shape == agent.action_space.shape
    assert loaded.metadata["seed"] == 7
    assert loaded.inference["model"]["hidden_layer_sizes"] == [4]
    assert algorithm in loaded.summary()
    payload = serialization.msgpack_restore(path.read_bytes())
    assert set(payload["inference"]) == {"model", "params", "observation_rms"}
    if not deterministic:
        np.testing.assert_allclose(
            loaded.make_act(True)(obs, key),
            agent.make_act(state, True)(obs, key),
            rtol=1e-6,
            atol=1e-6,
        )


def test_unique_default_paths_and_optional_results(tmp_path, monkeypatch):
    agent, state = _rejax_agent("ppo", "continuous")
    monkeypatch.chdir(tmp_path)
    first = save_policy(agent, state, results={"returns": jnp.asarray([3.0])})
    second = save_policy(agent, state)
    assert first != second
    assert first.parent == Path("outputs/policies")
    np.testing.assert_array_equal(load_policy(first).results["returns"], [3.0])
    assert not list(first.parent.glob("*.tmp"))


def test_rejects_failed_state_and_unknown_format(tmp_path):
    agent, state = _rejax_agent("ppo", "continuous")
    with pytest.raises(ValueError, match="failed training"):
        save_policy(agent, SimpleNamespace(failed=True), tmp_path / "bad.msgpack")
    with pytest.raises(ValueError, match=".msgpack"):
        save_policy(agent, state, tmp_path / "old.npz")
    invalid = tmp_path / "invalid.msgpack"
    invalid.write_bytes(serialization.msgpack_serialize({"format_version": 0}))
    with pytest.raises(ValueError, match="unsupported policy artifact"):
        load_policy(invalid)


def test_effective_task_settings_are_distinct_from_configured_defaults(tmp_path):
    env = TruncationWrapper(
        env=ConfiguredEnv(
            obs_dim=2,
            action_low=(-1.0,),
            action_high=(1.0,),
        ),
        max_steps=2,
    )
    agent = BackpropPolicyAgent.create(
        env,
        total_timesteps=2,
        num_rollouts=1,
        gradient_horizon=2,
        hidden_sizes=(4,),
        action_setpoint=jnp.asarray([0.25]),
    )
    path = save_policy(
        agent, agent.init_state(jax.random.key(0)), tmp_path / "overrides.msgpack"
    )
    metadata = load_policy(path).metadata
    assert metadata["source_config"]["task"]["terminal_penalty"] == -100.0
    assert metadata["effective_task"] == {"reward": "Q_fusion", "terminal_penalty": 0.0}
    assert metadata["source_max_steps"] == 2


@pytest.mark.parametrize("kind", ["policy", "open_loop"])
def test_backprop_roundtrip_preserves_parameters_and_source_clock(tmp_path, kind):
    env = TruncationWrapper(
        env=ObsHistoryWrapper(
            env=NamedObservationEnv(obs_dim=2, action_low=(-1.0,), action_high=(1.0,)),
            k=2,
        ),
        max_steps=4,
    )
    kwargs = dict(
        total_timesteps=4,
        eval_freq=4,
        num_rollouts=1,
        gradient_horizon=2,
        action_setpoint=jnp.asarray([0.25]),
    )
    if kind == "policy":
        agent = BackpropPolicyAgent.create(env, hidden_sizes=(4,), **kwargs)
    else:
        agent = BackpropOpenLoopAgent.create(
            env,
            num_knots=2,
            source_times=jnp.asarray([0.0, 0.2, 0.7, 1.0]),
            **kwargs,
        )
    state = agent.init_state(jax.random.key(4))
    if kind == "open_loop":
        state = state.replace(params=jnp.asarray([[-0.8], [0.9]]))
    path = save_policy(agent, state, tmp_path / f"{kind}.msgpack")
    loaded = load_policy(path)
    act = jax.jit(loaded.make_act())
    obs = jnp.zeros(env.observation_space.shape)
    clock = env.obs_layout().slice_of("elapsed_time").start
    assert clock == 3
    for time in (-1.0, 0.1, 0.5, 1.0, 5.0):
        observation = obs.at[clock].set(time)
        np.testing.assert_allclose(
            act(observation, jax.random.key(1)),
            agent.make_act(state)(observation, jax.random.key(2)),
            rtol=1e-6,
            atol=1e-6,
        )
    if kind == "open_loop":
        np.testing.assert_allclose(
            act(obs.at[clock].set(-1), jax.random.key(0)),
            jnp.tanh(state.params[0]),
            rtol=1e-6,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            act(obs.at[clock].set(5), jax.random.key(0)),
            jnp.tanh(state.params[-1]),
            rtol=1e-6,
            atol=1e-6,
        )


@pytest.mark.parametrize("deterministic", [False, True])
def test_mpc_roundtrip_restores_frozen_planner_without_buffer(tmp_path, deterministic):
    env = TruncationWrapper(
        env=NamedObservationEnv(
            obs_dim=2,
            action_low=(-1.0,),
            action_high=(1.0,),
        ),
        max_steps=3,
    )
    agent = MPCAgent.create(
        env,
        reward_scalar="P_fusion",
        hidden=4,
        horizon=2,
        num_samples=4,
        buffer_size=4,
        train_batch_size=2,
        deterministic=deterministic,
        planning_seed=123,
    )
    state = agent.init_state(jax.random.key(1))
    path = save_policy(agent, state, tmp_path / "mpc.msgpack")
    loaded = load_policy(path)
    act = jax.jit(loaded.make_act())
    obs = jnp.asarray([0.2, 0.1], dtype=jnp.float32)
    for key in jax.random.split(jax.random.key(2), 3):
        np.testing.assert_allclose(
            act(obs, key), agent.make_act(state)(obs, key), rtol=1e-6, atol=1e-6
        )
    assert "buffer" not in loaded.inference
    assert "opt_state" not in loaded.inference
    if deterministic:
        np.testing.assert_array_equal(
            act(obs, jax.random.key(1)), act(obs, jax.random.key(2))
        )
    with pytest.raises(ValueError, match="arbitrary reward callables"):
        save_policy(
            agent.replace(reward_scalar=None, reward_slice=None),
            state,
            tmp_path / "callable.msgpack",
        )
