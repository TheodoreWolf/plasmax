"""Generic launchers wire the agent API and evaluate paired saved policies."""

import inspect
import json
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from envelope import TruncationWrapper, static_field
from helpers import CheapBoundaryEnv

from agents.backprop import BackpropPolicyAgent
from agents.policy_io import save_policy
from experiments.studies import train_direct_baseline
from scripts import evaluate_policy
from training import train_backprop, train_mpc
from training.evaluation import check_interfaces
from training.runs import EnvConfig


def _capture_launcher(monkeypatch, module):
    calls = SimpleNamespace(created=[], loaded=[], runs=[])
    env = object()

    def load(config, backend):
        calls.loaded.append((config, backend))
        return env

    monkeypatch.setattr(module, "load_env", load)
    monkeypatch.setattr(module, "run_native", lambda *args: calls.runs.append(args))
    for name in ("BackpropPolicyAgent", "BackpropOpenLoopAgent", "MPCAgent"):
        if not hasattr(module, name):
            continue
        original = getattr(module, name)
        signature = inspect.signature(
            original.create if name == "MPCAgent" else original
        )

        def create(environment, _name=name, _signature=signature, **kwargs):
            _signature.bind(environment, **kwargs)
            agent = SimpleNamespace(kind=_name, env=environment, options=kwargs)
            calls.created.append(agent)
            return agent

        monkeypatch.setattr(module, name, SimpleNamespace(create=create))
    return calls


@pytest.mark.parametrize("mode,rate", [("policy", 1e-4), ("open_loop", 5e-2)])
def test_backprop_launcher_passes_supported_constructor_options_and_clock_metadata(
    monkeypatch, mode, rate
):
    calls = _capture_launcher(monkeypatch, train_backprop)
    cfg = train_backprop.Config(
        mode=mode,
        seed=37,
        num_seeds=2,
        env=EnvConfig(reward=None, disruption_penalty=0, eval_seed=42, eval_n_envs=3),
        backprop=train_backprop.BackpropConfig(
            total_timesteps=80, eval_freq=12, num_rollouts=4
        ),
    )
    train_backprop.main(cfg)
    created = calls.created[0]
    _, effective, name = calls.runs[0]
    assert created.options["learning_rate"] == rate
    assert created.options["init_seed"] == 37
    assert created.options["total_timesteps"] == 80
    assert created.options["eval_freq"] == 12
    assert created.options["num_rollouts"] == 4
    assert created.options["eval_seed"] == 42
    assert created.options["eval_n_envs"] == 3
    assert effective.env.time_aware == (mode == "open_loop")
    assert calls.loaded[0][0] is effective.env
    assert effective.env.reward is None
    assert effective.env.disruption_penalty == 0
    assert effective.algorithm == f"backprop_{mode}"
    assert effective.algorithm in name
    assert cfg.env.time_aware is False
    assert ("hidden_sizes" in created.options) == (mode == "policy")
    assert ("num_knots" in created.options) == (mode == "open_loop")


def test_backprop_explicit_zero_learning_rate_and_run_name_are_preserved(monkeypatch):
    calls = _capture_launcher(monkeypatch, train_backprop)
    train_backprop.main(
        train_backprop.Config(
            run_name="given", backprop=train_backprop.BackpropConfig(learning_rate=0.0)
        )
    )
    assert calls.created[0].options["learning_rate"] == 0.0
    assert calls.runs[0][2] == "given"


@pytest.mark.parametrize(
    "algorithm,knots",
    [
        ("direct_policy", None),
        ("direct_knots_1", 1),
        ("direct_knots_10", 10),
        ("direct_knots_100", 100),
    ],
)
def test_direct_study_keeps_labels_and_budget_while_using_new_agents(
    monkeypatch, algorithm, knots
):
    calls = _capture_launcher(monkeypatch, train_direct_baseline)
    validated = []
    monkeypatch.setattr(
        train_direct_baseline, "validate_reward", lambda *args: validated.append(args)
    )
    cfg = train_direct_baseline.Config(
        algorithm=algorithm,
        seed=12,
        study="selected",
        direct=train_direct_baseline.DirectConfig(
            total_timesteps=640, eval_freq=64, num_rollouts=0
        ),
    )
    train_direct_baseline.main(cfg)
    agent, effective, name = calls.runs[0]
    assert agent.options["num_rollouts"] == 64
    assert agent.options["total_timesteps"] == 640
    assert agent.options["init_seed"] == 12
    assert effective.algorithm == algorithm
    assert effective.study == "selected"
    assert algorithm in name
    assert effective.env.time_aware == (knots is not None)
    assert validated == [(cfg.env.env_setup, cfg.env.reward, cfg.env.backend)]
    if knots is not None:
        assert agent.options["num_knots"] == knots
        assert agent.options["learning_rate"] == cfg.direct.knot_learning_rate
    else:
        assert agent.options["learning_rate"] == cfg.direct.policy_learning_rate


def test_mpc_launcher_passes_planner_and_evaluation_settings(monkeypatch):
    calls = _capture_launcher(monkeypatch, train_mpc)
    cfg = train_mpc.Config(
        seed=18,
        env=EnvConfig(eval_n_envs=7, deterministic_eval=False),
        mpc=train_mpc.MPCConfig(
            horizon=2,
            num_samples=3,
            planning_seed=91,
            reward_scalar="q_min",
            total_timesteps=73,
            eval_freq=8,
        ),
    )
    train_mpc.main(cfg)
    options = calls.created[0].options
    assert options["eval_num_episodes"] == 7
    assert options["deterministic"] is False
    assert options["planning_seed"] == 91
    assert options["reward_scalar"] == "q_min"
    assert options["total_timesteps"] == 73
    assert options["eval_freq"] == 8
    assert options["horizon"] == 2
    assert options["num_samples"] == 3


@pytest.mark.parametrize("module", [train_backprop, train_mpc, train_direct_baseline])
def test_tglfnn_vmapped_launch_is_rejected_before_constructing_environment(
    monkeypatch, module
):
    calls = _capture_launcher(monkeypatch, module)
    cfg = module.Config(env=EnvConfig(backend="tglfnn_nr"), num_seeds=2)
    with pytest.raises(ValueError, match="independent processes"):
        module.main(cfg)
    assert not calls.loaded
    assert not calls.runs


class _RandomEnvironment(CheapBoundaryEnv):
    reward_offset: float = static_field(default=0.0)

    def init(self, key):
        state, info = super().init(key)
        state = state.replace(
            obs=jax.random.uniform(key, state.obs.shape, dtype=jnp.float32)
        )
        return state, info.update(obs=state.obs)

    def step(self, state, action):
        state, info = super().step(state, action)
        return state, info.update(reward=state.obs[0] + self.reward_offset)


def _policy_paths(tmp_path):
    def environment(offset=0.0):
        return TruncationWrapper(
            env=_RandomEnvironment(
                obs_dim=2, action_low=(-1.0,), action_high=(1.0,), reward_offset=offset
            ),
            max_steps=3,
        )

    source = environment()
    agent = BackpropPolicyAgent.create(
        source,
        total_timesteps=4,
        num_rollouts=1,
        gradient_horizon=2,
        hidden_sizes=(2,),
        action_setpoint=jnp.zeros(1),
    )
    state = agent.init_state(jax.random.key(1))
    paths = tuple(
        save_policy(
            agent,
            state,
            tmp_path / f"policy{index}.msgpack",
            metadata={
                "config": {"env": {"backend": "source", "env_setup": "fake/task"}}
            },
        )
        for index in range(2)
    )
    return paths, source, environment(2.0)


@pytest.mark.parametrize("trajectories", [False, True])
@pytest.mark.parametrize("source_backend", [None, "source_override"])
def test_evaluator_pairs_policies_and_backends_and_exports_optional_npz(
    tmp_path, monkeypatch, trajectories, source_backend
):
    paths, source, target = _policy_paths(tmp_path)
    loads, logged, initializations = [], [], []
    finished = []

    def load_env(policy, **kwargs):
        loads.append(kwargs)
        env = target if kwargs.get("backend") == "target" else source
        # Match the real host loader, including validation before the JIT call.
        check_interfaces(policy, env)
        return env

    run = SimpleNamespace(log=logged.append, finish=lambda: finished.append(True))

    def init(**kwargs):
        initializations.append(kwargs)
        return run

    monkeypatch.setattr(evaluate_policy, "load_policy_env", load_env)
    monkeypatch.setattr(evaluate_policy.wandb, "init", init)
    output = tmp_path / "evaluation"
    evaluate_policy.main(
        evaluate_policy.Config(
            policies=paths,
            backend=source_backend,
            target_backend="target",
            num_episodes=3,
            eval_seed=27,
            trajectories=trajectories,
            output_dir=output,
        )
    )
    reports = [
        json.loads((output / f"{i}-{path.stem}.json").read_text())
        for i, path in enumerate(paths)
    ]
    assert len(logged) == 2
    assert finished == [True]
    assert initializations[0]["mode"] == "online"
    assert len(loads) == 4
    assert loads[0]["backend"] == source_backend
    np.testing.assert_array_equal(reports[0]["returns"], reports[1]["returns"])
    assert len(set(reports[0]["returns"])) > 1
    for report in reports:
        assert report["eval_seed"] == 27
        assert report["transfer/source_backend"] == (source_backend or "source")
        assert report["transfer/target_backend"] == "target"
        np.testing.assert_allclose(
            report["transfer/return_gap"], 6.0, rtol=1e-6, atol=1e-6
        )
        np.testing.assert_array_equal(report["lengths"], [3, 3, 3])
    files = sorted(output.glob("*.npz"))
    assert len(files) == (4 if trajectories else 0)
    if trajectories:
        with (
            np.load(output / "0-policy0-source.npz", allow_pickle=False) as left,
            np.load(output / "0-policy0-target.npz", allow_pickle=False) as right,
        ):
            np.testing.assert_array_equal(left["obs"], right["obs"])
            np.testing.assert_allclose(
                right["reward"] - left["reward"], 2.0, rtol=1e-6, atol=1e-6
            )
            assert {
                "obs",
                "next_obs",
                "reward",
                "valid",
                "terminated",
                "truncated",
                "requested_action",
                "time_steps",
            } <= set(left.files)


def test_evaluator_stops_when_online_logging_cannot_initialize(tmp_path, monkeypatch):
    def reject(**kwargs):
        raise RuntimeError("authorization failed")

    monkeypatch.setattr(evaluate_policy.wandb, "init", reject)
    output = tmp_path / "absent"
    with pytest.raises(RuntimeError, match="authorization failed"):
        evaluate_policy.main(
            evaluate_policy.Config(
                policies=(tmp_path / "policy.msgpack",), output_dir=output
            )
        )
    assert not output.exists()


def test_launcher_defaults_keep_online_logging_and_realistic_wrapper_labels():
    for config in (
        train_backprop.Config(),
        train_mpc.Config(),
        train_direct_baseline.Config(),
    ):
        assert config.wandb.mode == "online"
        assert config.env.variant == "realistic"
        assert config.env.reward is None
        assert config.env.disruption_penalty is None
