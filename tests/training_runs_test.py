"""Host orchestration uses real cheap agents and mocked external tracking."""

import dataclasses
import sys
from types import SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from envelope import TruncationWrapper
from flax import struct
from helpers import CheapBoundaryEnv, CheapBoundaryState

from agents.backprop import BackpropOpenLoopAgent, BackpropPolicyAgent
from agents.es import ESAgent
from agents.mpc import MPCAgent
from agents.policy_io import LoadedPolicy, environment_interface, load_policy
from agents.ppo import PPOAdapter
from agents.sac import SACAdapter
from plasmax.spaces import ObsLayout
from training import runs, train_ppo, train_sac, vmap_logging
from training.envelope_gymnax import EnvelopeGymnax


class NamedEnv(CheapBoundaryEnv):
    def obs_layout(self):
        return ObsLayout(
            {},
            {"P_fusion": slice(0, 1), "elapsed_time": slice(1, 2)},
            (),
            ("P_fusion", "elapsed_time"),
        )


class PhysicsBoundaryState(CheapBoundaryState):
    """Synthetic plasma scalars for testing logging without TORAX compilation."""

    @property
    def plasma(self) -> SimpleNamespace:
        value = self.steps.astype(jnp.float32)
        return SimpleNamespace(
            **{
                name: value
                for name in (
                    "Q_fusion",
                    "W_thermal_total",
                    "P_fusion",
                    "tau_E",
                    "H98",
                    "beta_N",
                    "q_min",
                    "q95",
                    "f_non_inductive",
                    "fgw_n_e_line_avg",
                    "fgw_n_e_volume_avg",
                    "P_SOL_total",
                )
            },
            P_LH=jnp.ones_like(value),
        )


class PhysicsNamedEnv(NamedEnv):
    def init(self, key: jax.Array) -> tuple[PhysicsBoundaryState, Any]:
        state, info = super().init(key)
        return PhysicsBoundaryState(obs=state.obs, steps=state.steps), info


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
            self.finish_calls = 0
            records.append(self)

        def log(self, step, index, metrics):
            self.logs.append((int(step), int(index), metrics))

        def log_once(self, summary):
            self.summary = summary

        def log_artifact(self, artifact):
            self.artifacts.append(artifact)

        def finish(self):
            self.finished = True
            self.finish_calls += 1
            self.finish_error = sys.exc_info()[0]

    class Artifact:
        def __init__(self, name, type):
            self.files = []

        def add_file(self, path):
            self.files.append(path)

    monkeypatch.setattr(runs, "SeedBufferLogger", Logger)
    monkeypatch.setattr(runs.wandb, "Artifact", Artifact)
    return records


@pytest.mark.parametrize(
    "kind", ["policy", "open_loop", "mpc", "es_policy", "es_open_loop"]
)
@pytest.mark.parametrize("num_seeds", [1, 2])
def test_native_host_runs_compile_log_and_save_each_seed(
    tmp_path, tracking, kind, num_seeds, monkeypatch
):
    env = TruncationWrapper(
        env=PhysicsNamedEnv(obs_dim=2, action_low=(-1.0,), action_high=(1.0,)),
        max_steps=2,
    )
    is_es = kind.startswith("es_")
    expected_steps = 16 if is_es else 4
    if is_es:
        agent = ESAgent.create(
            env,
            parameterization=kind.removeprefix("es_"),
            total_timesteps=19,
            eval_freq=5,
            population_size=4,
            num_rollouts=1,
            eval_n_envs=1,
            hidden_sizes=(4,),
            num_knots=2,
            source_times=jnp.asarray([0.0, 1.0]),
            action_setpoint=jnp.asarray([0.25]),
        )
    elif kind == "mpc":
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
    training_keys = []
    vmapped_functions = []
    original_train = type(agent).train
    original_vmap = jax.vmap

    def record_train(current: Any, key: jax.Array) -> Any:
        jax.debug.callback(lambda value: training_keys.append(np.asarray(value)), key)
        return original_train(current, key)

    def record_vmap(function: Any, *args: Any, **kwargs: Any) -> Any:
        vmapped_functions.append(getattr(function, "__name__", None))
        return original_vmap(function, *args, **kwargs)

    monkeypatch.setattr(type(agent), "train", record_train)
    monkeypatch.setattr(jax, "vmap", record_vmap)
    config = RunConfig(
        checkpoint_dir=str(tmp_path),
        env=runs.EnvConfig(eval_n_envs=1),
        algorithm="es" if is_es else kind,
        num_seeds=num_seeds,
    )
    runs.run_native(agent, config, "cheap")
    assert ("train_one" in vmapped_functions) == (num_seeds > 1)
    np.testing.assert_array_equal(
        sorted(tuple(key) for key in training_keys),
        [jax.random.PRNGKey(seed) for seed in range(3, 3 + num_seeds)],
    )
    assert agent.eval_callback is None
    logger = tracking[0]
    assert logger.finished
    assert logger.kwargs["mode"] == "online"
    assert {index for _, index, _ in logger.logs} == set(range(num_seeds))
    assert logger.kwargs["seed_ids"] == tuple(range(3, 3 + num_seeds))
    assert logger.summary["run/actual_train_steps"] == expected_steps
    for _, _, metrics in logger.logs:
        np.testing.assert_array_equal(metrics["obs/beta_N"], 2.0)
        np.testing.assert_array_equal(metrics["termination/completion_rate"], 1.0)
        assert "returns" not in metrics
        assert "lengths" not in metrics
        assert all(not name.startswith("eval/") for name in metrics)
    if is_es:
        assert {step for step, _, _ in logger.logs} == {0, 8, 16}
        initial_keys = tuple(logger.logs[0][2])
        assert all(tuple(metrics) == initial_keys for _, _, metrics in logger.logs)
    paths = sorted(tmp_path.glob("*.msgpack"))
    assert len(paths) == num_seeds
    for seed, path in zip(range(3, 3 + num_seeds), paths, strict=True):
        assert path.name == f"cheap-seed{seed}.msgpack"
        policy = load_policy(path)
        assert policy.metadata["seed"] == seed
        assert policy.metadata["actual_timesteps"] == expected_steps
        assert policy.metadata["config"]["env"]["eval_n_envs"] == 1
        assert len(policy.results["global_step"]) >= 2
        if is_es:
            assert policy.algorithm == "es"
            assert policy.inference["parameterization"] == kind.removeprefix("es_")
            np.testing.assert_array_equal(policy.results["global_step"], [0, 8, 16])
    assert len(logger.artifacts[0].files) == num_seeds


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


@pytest.mark.parametrize("launcher", [train_ppo, train_sac], ids=["ppo", "sac"])
def test_rejax_launcher_defaults_export_to_default_destination(
    tmp_path, monkeypatch, launcher
):
    config = launcher.Config()
    state = SimpleNamespace(global_step=jnp.asarray(12))
    calls = []
    saved_path = tmp_path / "policy.msgpack"

    def save(agent, state, path, **kwargs):
        calls.append((path, kwargs))
        return saved_path

    monkeypatch.setattr(runs, "save_policy", save)
    paths = runs.save_run_policies(None, state, config, "defaults", batched=False)

    assert paths == (saved_path,)
    path, options = calls[0]
    assert path is None
    assert options["metadata"]["config"] == dataclasses.asdict(config)
    assert options["metadata"]["actual_train_steps"] == 12
    assert options["deterministic"] == config.env.deterministic_eval


@pytest.mark.parametrize("diagnose_numerics", [False, True])
def test_sac_launcher_trains_native_world_model_and_saves_each_seed(
    tmp_path, monkeypatch, tracking, diagnose_numerics
):
    """Native backends need neither a TORAX name nor TORAX evaluation state."""
    loaded = []

    def load_envelope(config, backend):
        loaded.append((config.env.env_setup, backend))
        return _env()

    monkeypatch.setattr(train_sac, "_load_envelope", load_envelope)
    monkeypatch.setattr(train_sac, "SeedBufferLogger", runs.SeedBufferLogger)
    config = train_sac.Config(
        env=train_sac.EnvConfig(
            env_setup="kstar_worldmodel", backend=None, eval_n_envs=1
        ),
        sac=train_sac.SACConfig(
            total_timesteps=4,
            eval_freq=4,
            num_envs=2,
            num_epochs=1,
            buffer_size=8,
            fill_buffer=0,
            batch_size=2,
            hidden_sizes=(4,),
            diagnose_numerics=diagnose_numerics,
        ),
        num_seeds=1 if diagnose_numerics else 2,
        seed=3,
        checkpoint_dir=str(tmp_path),
    )

    train_sac.main(config)

    assert loaded == [("kstar_worldmodel", None)]
    logger = tracking[0]
    assert logger.finished
    assert logger.finish_calls == 1
    assert "kstar_worldmodel-native-realistic" in logger.kwargs["run_name"]
    assert {index for _, index, _ in logger.logs} == set(range(config.num_seeds))
    assert logger.kwargs["seed_ids"] == tuple(range(3, 3 + config.num_seeds))
    assert logger.summary["run/actual_train_steps"] == 4
    paths = sorted(tmp_path.glob("*.msgpack"))
    assert len(paths) == config.num_seeds
    assert len(logger.artifacts[0].files) == config.num_seeds
    for seed, path in enumerate(paths):
        policy = load_policy(path)
        assert policy.metadata["seed"] == seed + 3
        assert policy.metadata["config"]["env"]["backend"] is None


def test_sac_numerical_diagnostic_rejects_vmapped_launch_before_tracking(tracking):
    config = train_sac.Config(
        sac=train_sac.SACConfig(diagnose_numerics=True), num_seeds=2
    )
    with pytest.raises(ValueError, match="diagnose_numerics requires num_seeds=1"):
        train_sac.main(config)
    assert tracking == []


def test_sac_export_failure_finishes_tracking(tmp_path, monkeypatch, tracking):
    monkeypatch.setattr(train_sac, "_load_envelope", lambda config, backend: _env())
    monkeypatch.setattr(train_sac, "SeedBufferLogger", runs.SeedBufferLogger)

    def fail_export(*args, **kwargs):
        raise ValueError("cannot export non-finite policy parameters")

    monkeypatch.setattr(train_sac, "save_run_policies", fail_export)
    config = train_sac.Config(
        env=train_sac.EnvConfig(
            env_setup="kstar_worldmodel", backend=None, eval_n_envs=1
        ),
        sac=train_sac.SACConfig(
            total_timesteps=4,
            eval_freq=4,
            num_envs=2,
            num_epochs=1,
            buffer_size=8,
            fill_buffer=0,
            batch_size=2,
            hidden_sizes=(4,),
        ),
        num_seeds=1,
        checkpoint_dir=str(tmp_path),
    )
    with pytest.raises(ValueError, match="cannot export non-finite policy parameters"):
        train_sac.main(config)
    assert len(tracking) == 1
    assert tracking[0].finish_calls == 1
    assert tracking[0].finish_error is ValueError


def test_native_artifact_upload_uses_the_logger_owned_run(tmp_path, monkeypatch):
    uploaded = []
    run = SimpleNamespace(log_artifact=uploaded.append)
    monkeypatch.setattr(vmap_logging, "wandb_dir", lambda: str(tmp_path))
    monkeypatch.setattr(vmap_logging.wandb, "init", lambda **kwargs: run)
    logger = vmap_logging.SeedBufferLogger(num_seeds=1, run_name="native")
    artifact = object()

    logger.log_artifact(artifact)

    assert uploaded == [artifact]


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
    "effective,explicit,expected",
    [
        ("Q_fusion", None, "Q_fusion"),
        (None, "Q_fusion", "Q_fusion"),
        (None, None, "lh_transition"),
    ],
)
def test_environment_reconstruction_preserves_reward_and_source_clock(
    monkeypatch,
    effective,
    explicit,
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
                "reward": explicit,
            }
        },
        "source_config": {"task": {"reward": "lh_transition"}},
        "source_max_steps": 2,
    }
    if effective is not None:
        metadata["effective_task"] = {"reward": effective}
    policy = LoadedPolicy("ppo", {}, environment_interface(env), metadata, True)
    assert runs.load_policy_env(policy) is env
    assert calls[0][1] == {"reward": expected}
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
