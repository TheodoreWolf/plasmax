"""Host orchestration uses real cheap agents and mocked external tracking."""

import dataclasses
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from envelope import TruncationWrapper
from flax import struct
from helpers import CheapBoundaryEnv

from agents.backprop import BackpropOpenLoopAgent, BackpropPolicyAgent
from agents.mpc import MPCAgent
from agents.policy_io import LoadedPolicy, environment_interface, load_policy
from agents.ppo import PPOAdapter
from agents.sac import SACAdapter
from plasmax.spaces import ObsLayout
from training import runs
from training.envelope_gymnax import EnvelopeGymnax


class NamedEnv(CheapBoundaryEnv):
    def obs_layout(self):
        return ObsLayout(
            {},
            {"P_fusion": slice(0, 1), "elapsed_time": slice(1, 2)},
            (),
            ("P_fusion", "elapsed_time"),
        )


def _env():
    return TruncationWrapper(
        env=NamedEnv(
            obs_dim=2,
            action_low=(-1.0,),
            action_high=(1.0,),
        ),
        max_steps=2,
    )


@dataclasses.dataclass
class RunConfig:
    env: runs.EnvConfig = dataclasses.field(default_factory=runs.EnvConfig)
    wandb: runs.WandbConfig = dataclasses.field(default_factory=runs.WandbConfig)
    seed: int = 3
    num_seeds: int = 2
    checkpoint_dir: str | None = None
    history_dir: str | None = None
    algorithm: str = "backprop_policy"
    study: str = "test"


@pytest.fixture
def tracking(monkeypatch):
    records = []

    class Logger:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.logs = []
            self.artifacts = []
            self.finished = False
            records.append(self)

        def log(self, step, index, metrics):
            self.logs.append((int(step), int(index), metrics))

        def log_once(self, summary):
            self.summary = summary

        def log_artifact(self, artifact):
            self.artifacts.append(artifact)

        def finish(self):
            self.finished = True

    class Artifact:
        def __init__(self, name, type):
            self.files = []

        def add_file(self, path):
            self.files.append(path)

    monkeypatch.setattr(runs, "SeedBufferLogger", Logger)
    monkeypatch.setattr(runs.wandb, "Artifact", Artifact)
    return records


@pytest.mark.parametrize("kind", ["policy", "open_loop", "mpc"])
def test_native_host_runs_compile_log_and_save_each_seed(tmp_path, tracking, kind):
    env = _env()
    if kind == "mpc":
        agent = MPCAgent.create(
            env,
            total_timesteps=4,
            eval_freq=2,
            hidden=4,
            horizon=1,
            num_samples=2,
            buffer_size=4,
            train_batch_size=2,
            eval_num_episodes=1,
        )
    else:
        kwargs = dict(
            total_timesteps=4,
            eval_freq=2,
            gradient_horizon=2,
            num_rollouts=1,
            eval_n_envs=1,
            action_setpoint=jnp.asarray([0.25]),
        )
        if kind == "policy":
            agent = BackpropPolicyAgent.create(env, hidden_sizes=(4,), **kwargs)
        else:
            agent = BackpropOpenLoopAgent.create(
                env, num_knots=2, source_times=jnp.asarray([0.0, 1.0]), **kwargs
            )
    config = RunConfig(
        checkpoint_dir=str(tmp_path), env=runs.EnvConfig(eval_n_envs=1), algorithm=kind
    )
    runs.run_native(agent, config, "cheap")
    assert agent.eval_callback is None
    logger = tracking[0]
    assert logger.finished
    assert logger.kwargs["mode"] == "online"
    assert {index for _, index, _ in logger.logs} == {0, 1}
    assert logger.summary["run/actual_train_steps"] == 4
    paths = sorted(tmp_path.glob("*.msgpack"))
    assert len(paths) == 2
    for seed, path in zip((3, 4), paths, strict=True):
        policy = load_policy(path)
        assert policy.metadata["seed"] == seed
        assert policy.metadata["actual_timesteps"] == 4
        assert policy.metadata["config"]["env"]["eval_n_envs"] == 1
        assert len(policy.results["global_step"]) >= 2
    assert len(logger.artifacts[0].files) == 2


@struct.dataclass
class FailedState:
    global_step: Any
    failed: Any
    failure_step: Any


def test_failed_seed_blocks_the_whole_batch_before_any_export(tmp_path, monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("failed batch must be checked before the first policy export")

    monkeypatch.setattr(runs, "save_policy", unexpected)
    states = FailedState(
        jnp.asarray([4, 2]), jnp.asarray([False, True]), jnp.asarray([-1, 2])
    )
    with pytest.raises(FloatingPointError, match="Training failed"):
        runs.save_run_policies(
            None,
            states,
            RunConfig(checkpoint_dir=str(tmp_path)),
            "failed",
            batched=True,
        )
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("algorithm", ["ppo", "sac"])
def test_rejax_seed_exports_have_scalar_progress_and_nested_configuration(
    tmp_path, algorithm
):
    env = EnvelopeGymnax(_env())
    kwargs = dict(
        env=env,
        env_params=env.default_params,
        num_envs=1,
        total_timesteps=2,
        eval_freq=2,
        normalize_observations=False,
    )
    if algorithm == "ppo":
        agent = PPOAdapter.create(
            **kwargs, num_steps=2, num_epochs=1, num_minibatches=1
        )
    else:
        agent = SACAdapter.create(
            **kwargs,
            buffer_size=4,
            batch_size=1,
            fill_buffer=0,
            hidden_layer_sizes=(4,),
        )
    states = jax.vmap(agent.init_state)(jax.random.split(jax.random.PRNGKey(0), 2))
    paths = runs.save_run_policies(
        agent,
        states,
        RunConfig(checkpoint_dir=str(tmp_path)),
        algorithm,
        batched=True,
        results={"returns": jnp.asarray([[2.0], [3.0]])},
    )
    for index, path in enumerate(paths):
        policy = load_policy(path)
        assert policy.metadata["seed_index"] == index
        assert np.ndim(policy.metadata["actual_train_steps"]) == 0
        np.testing.assert_array_equal(policy.results["returns"], [2.0 + index])


@pytest.mark.parametrize(
    "effective,explicit,kappa,expected",
    [
        (-75.0, None, 0.75, -75.0),
        (None, 0.0, 2.0, 0.0),
        (None, None, 0.75, -75.0),
        (None, None, None, -100.0),
    ],
)
def test_environment_reconstruction_preserves_penalty_and_source_clock(
    monkeypatch,
    effective,
    explicit,
    kappa,
    expected,
):
    env = _env()
    calls = []

    def make(*args, **kwargs):
        calls.append((args, kwargs))
        return env

    def wrappers(env, **kwargs):
        calls.append(kwargs)
        return env

    monkeypatch.setattr(runs, "make", make)
    monkeypatch.setattr(runs, "RealisticWrappers", wrappers)
    metadata = {
        "config": {
            "env": {
                "env_setup": "mock/circular/smoke",
                "backend": "mock",
                "disruption_penalty": explicit,
                "disruption_kappa": kappa,
            }
        },
        "source_config": {
            "task": {"terminal_penalty": -100, "reward": "lh_transition"}
        },
        "source_max_steps": 2,
    }
    if effective is not None:
        metadata["effective_task"] = {"terminal_penalty": effective}
    policy = LoadedPolicy("ppo", {}, environment_interface(env), metadata, True)
    assert runs.load_policy_env(policy) is env
    assert calls[0][1]["disruption_penalty"] == expected
    assert calls[0][1]["reward"] == "lh_transition"
    assert calls[1]["time_aware"] is True
    assert calls[1]["max_steps"] == 2


def test_reconstruction_rejects_changed_history_or_observation_layout(monkeypatch):
    env = _env()
    monkeypatch.setattr(runs, "make", lambda *args, **kwargs: env)
    monkeypatch.setattr(runs, "RealisticWrappers", lambda env, **kwargs: env)
    interface = environment_interface(env)
    interface["history"] = [2]
    policy = LoadedPolicy(
        "ppo",
        {},
        interface,
        {"config": {"env": {"env_setup": "mock/circular/smoke", "backend": "mock"}}},
        True,
    )
    with pytest.raises(ValueError, match="history"):
        runs.load_policy_env(policy)
