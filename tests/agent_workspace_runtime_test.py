"""Stateful policy execution contracts without running a plasma simulator."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from experiments.agent_policy_search.artifacts import read_json
from experiments.agent_policy_search.config import EvaluationConfig, TaskConfig
from experiments.agent_policy_workspace import runtime

POLICY = """import jax
import jax.numpy as jnp
from custom_primitives import increment

POLICY_API_VERSION = 1

def init_memory():
    return {'count': jnp.asarray(0, jnp.int32)}

def act(obs, reset, memory, rng):
    count = jnp.where(reset, 0, memory['count'])
    action = jax.random.uniform(rng, (1,), dtype=jnp.float32)
    return action, {'count': increment(count)}, {'count': count, 'reset': reset}
"""


@pytest.fixture
def task(monkeypatch: pytest.MonkeyPatch) -> EvaluationConfig:
    from envelope import TruncationWrapper
    from helpers import CheapBoundaryEnv

    from training import runs

    env = TruncationWrapper(
        env=CheapBoundaryEnv(
            obs_dim=2, action_low=(-1.0,), action_high=(1.0,), terminate_after=2
        ),
        max_steps=4,
    )
    monkeypatch.setattr(runs, "load_env", lambda *_args: env)
    return EvaluationConfig(TaskConfig("mock/circular/smoke", "constant"), (7, 17))


def _source(tmp_path: Path, text: str = POLICY) -> Path:
    source = tmp_path / "policy.py"
    source.write_text(text)
    (tmp_path / "custom_primitives.py").write_text(
        "def increment(value):\n    return value + 1\n"
    )
    return source


def test_memory_reset_traces_padding_and_reproducible_keys(
    tmp_path: Path, task: EvaluationConfig
) -> None:
    source = _source(tmp_path)
    first, repeated = tmp_path / "first", tmp_path / "repeated"
    first.mkdir()
    repeated.mkdir()
    report = runtime._evaluate_in_worker(source, first, task)
    runtime._evaluate_in_worker(source, repeated, task)
    assert report["status"] == "ok"
    assert report["finite"] is True
    assert report["mean_return"] == 3.0
    assert "next_memory" in report["interface"]["policy_contract"]
    assert [episode["seed"] for episode in report["feedback"]["episodes"]] == [7, 17]
    with (
        np.load(first / "trajectories.npz", allow_pickle=False) as data,
        np.load(repeated / "trajectories.npz", allow_pickle=False) as other,
    ):
        np.testing.assert_array_equal(data["valid"], [[True, True, False, False]] * 2)
        np.testing.assert_array_equal(data["trace_count"], [[0, 1, 0, 0]] * 2)
        np.testing.assert_array_equal(
            data["trace_reset"], [[True, False, False, False]] * 2
        )
        np.testing.assert_array_equal(data["reward"], [[1, 2, 0, 0]] * 2)
        np.testing.assert_array_equal(
            data["requested_action"], other["requested_action"]
        )
        assert json.loads(str(data["metadata_json"]))["seeds"] == [7, 17]
    assert (first / "episodes" / "0000_seed_7.npz").exists()
    assert read_json(first / "trace_schema.json")["count"] == {
        "shape": [],
        "dtype": "int32",
    }


@pytest.mark.parametrize("backend", ["constant", "bohm_gyrobohm", "tglfnn_nr"])
def test_stateless_policy_matches_existing_evaluator_and_merges_traces(
    tmp_path: Path,
    task: EvaluationConfig,
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
) -> None:
    import jax
    import jax.numpy as jnp

    from training.evaluation import evaluate_policy, trajectory_arrays
    from training.runs import load_env

    config = replace(task, task=replace(task.task, backend=backend))
    source = _source(tmp_path)
    env = load_env(None, None)

    def stateless(obs: Any, key: Any) -> Any:
        del obs
        return jax.random.uniform(key, (1,), dtype=jnp.float32)

    expected, returns = [], []
    for seed in config.seeds:
        metrics, trajectory = evaluate_policy(
            stateless, env, jax.random.key(seed), num_episodes=1
        )
        expected.append(trajectory_arrays(trajectory, env))
        returns.append(float(metrics["returns"][0]))

    def subprocess(output: Path, _timeout: float) -> dict[str, Any]:
        request = read_json(output / "request.json")
        worker_config = replace(config, seeds=tuple(request["config"]["seeds"]))
        return runtime._evaluate_in_worker(source, output, worker_config) | {
            "evaluation_seconds": 1.0
        }

    monkeypatch.setattr(runtime, "_subprocess", subprocess)
    output = tmp_path / "evaluation"
    report = runtime.evaluate_bundle(source, output, config)
    assert report["mean_return"] == np.mean(returns)
    assert [item["return"] for item in report["feedback"]["episodes"]] == returns
    with np.load(output / "trajectories.npz", allow_pickle=False) as data:
        for field in (
            "requested_action",
            "obs",
            "next_obs",
            "valid",
            "reward",
            "terminated",
            "truncated",
        ):
            np.testing.assert_array_equal(
                data[field], np.concatenate([episode[field] for episode in expected])
            )
        np.testing.assert_array_equal(data["trace_count"], [[0, 1, 0, 0]] * 2)
        assert (
            json.loads(str(data["policy_trace_schema_json"]))["count"]["dtype"]
            == "int32"
        )


def test_batched_rollouts_match_sequential_and_keep_partial_batch(
    tmp_path: Path, task: EvaluationConfig
) -> None:
    source = _source(tmp_path)
    seeds = tuple(range(66))
    sequential, batched = tmp_path / "sequential", tmp_path / "batched"
    sequential.mkdir()
    batched.mkdir()
    expected = runtime._evaluate_in_worker(
        source, sequential, replace(task, seeds=seeds)
    )
    actual = runtime._evaluate_in_worker(
        source,
        batched,
        replace(task, seeds=seeds, task=replace(task.task, backend="bohm_gyrobohm")),
    )
    assert actual["feedback"] == expected["feedback"]
    with (
        np.load(sequential / "trajectories.npz", allow_pickle=False) as reference,
        np.load(batched / "trajectories.npz", allow_pickle=False) as data,
    ):
        assert set(data.files) == set(reference.files)
        for name in data.files:
            if not name.endswith("_json"):
                np.testing.assert_array_equal(data[name], reference[name])
        assert json.loads(str(data["metadata_json"]))["evaluation_batch_size"] == 64
        assert data["valid"].shape == (66, 4)
        with np.load(batched / "episodes" / "0065_seed_65.npz") as episode:
            for name in episode.files:
                np.testing.assert_array_equal(episode[name], data[name][65:66])


def test_compact_device_arrays_preserve_all_plasma_export_fields() -> None:
    import jax
    import jax.numpy as jnp

    from plasmax.rollout import TrajectoryStep
    from training.evaluation import trajectory_arrays

    values = jnp.arange(6, dtype=jnp.float32).reshape(2, 3)
    plasma = SimpleNamespace(
        t=values,
        **{
            name: values + index
            for index, name in enumerate(
                (
                    "Q_fusion",
                    "P_fusion",
                    "W_thermal_total",
                    "q_min",
                    "q95",
                    "beta_N",
                    "P_aux_total",
                    "fgw_n_e_line_avg",
                )
            )
        },
        sim=SimpleNamespace(
            core_profiles=SimpleNamespace(
                Ip_profile_face=jnp.stack((values, values + 1), axis=-1)
            )
        ),
    )
    env = SimpleNamespace(
        action_space=SimpleNamespace(dtype=jnp.float32),
        from_physical=lambda action: action / 2,
        actuator_specs=[SimpleNamespace(name="power")],
    )
    trajectory = TrajectoryStep(
        obs=values[..., None],
        action=values[..., None] + 2,
        reward=values,
        next_obs=values[..., None] + 1,
        terminated=values == 2,
        truncated=values == 5,
        valid=values != 3,
        env_state=SimpleNamespace(prev_action=values[..., None], plasma=plasma),
        info=SimpleNamespace(termination_code=values.astype(jnp.int32)),
    )
    expected = trajectory_arrays(trajectory, env)
    actual = jax.jit(
        lambda action: runtime._numeric_trajectory_arrays(
            trajectory._replace(action=action), env
        )
    )(trajectory.action)
    assert set(actual) == set(expected) - {"actuator_names"}
    for name, value in actual.items():
        np.testing.assert_array_equal(value, expected[name])


@pytest.mark.parametrize(
    "change,error",
    [
        ("action = jnp.zeros((2,), dtype=jnp.float32)", "action shape/dtype"),
        (
            "return action, {'count': jnp.zeros((2,), jnp.int32)}, {'count': count}",
            "memory structure",
        ),
        (
            "return action, {'count': increment(count)}, {'../bad': count}",
            "trace must map",
        ),
        (
            "return action, {'count': increment(count)}, {'bad': {'nested': count}}",
            "trace must map",
        ),
    ],
)
def test_validation_rejects_invalid_contract(
    tmp_path: Path, task: EvaluationConfig, change: str, error: str
) -> None:
    old = (
        "action = jax.random.uniform(rng, (1,), dtype=jnp.float32)"
        if change.startswith("action")
        else (
            "return action, {'count': increment(count)}, "
            "{'count': count, 'reset': reset}"
        )
    )
    source = _source(tmp_path, POLICY.replace(old, change))
    output = tmp_path / "validation"
    output.mkdir()
    with pytest.raises(ValueError, match=error):
        runtime._evaluate_in_worker(source, output, task, validate_only=True)
    assert (output / "interface.json").exists()
    assert not (output / "trajectories.npz").exists()


def test_validation_and_description_do_not_initialize_episode(
    tmp_path: Path, task: EvaluationConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    from helpers import CheapBoundaryEnv

    def unexpected_init(*_args: Any) -> Any:
        raise AssertionError("validation must not initialize an episode")

    monkeypatch.setattr(CheapBoundaryEnv, "init", unexpected_init)
    source = _source(tmp_path)
    for name, policy in (("describe", None), ("validate", source)):
        output = tmp_path / name
        output.mkdir()
        report = runtime._evaluate_in_worker(policy, output, task, validate_only=True)
        assert report["status"] == "ok"
        assert report["interface"]["action_shape"] == [1]
        assert not (output / "trajectories.npz").exists()


def test_subprocess_syntax_failure_is_archived(tmp_path: Path) -> None:
    source = _source(tmp_path, "def act(:\n")
    output = tmp_path / "evaluation"
    report = runtime.evaluate_bundle(
        source, output, EvaluationConfig(TaskConfig("unused"), (1,), 15)
    )
    assert report["status"] == "error"
    assert "SyntaxError" in report["error"]
    assert read_json(output / "result.json") == report
    assert (output / "stderr.txt").is_file()


@pytest.mark.parametrize(
    "ticks,status,expected",
    [
        ([100.0, 101.0, 102.0, 104.0, 105.0], "ok", [([5], 9.0), ([6], 6.0)]),
        ([100.0, 101.0, 102.0, 111.0], "timeout", [([5], 9.0)]),
    ],
)
def test_tglfnn_workers_share_bank_deadline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ticks: list[float],
    status: str,
    expected: list[tuple[list[int], float]],
) -> None:
    calls = []
    clock = iter(ticks)
    monkeypatch.setattr(runtime, "time", SimpleNamespace(monotonic=lambda: next(clock)))

    def subprocess(output: Path, _timeout: float) -> dict[str, Any]:
        request = read_json(output / "request.json")
        calls.append((request["config"]["seeds"], request["config"]["timeout_seconds"]))
        return {"status": "ok"}

    monkeypatch.setattr(runtime, "_subprocess", subprocess)
    monkeypatch.setattr(
        runtime,
        "_merge_seed_results",
        lambda *_args: {"status": "ok", "mean_return": 2.0},
    )
    result = runtime.evaluate_bundle(
        _source(tmp_path),
        tmp_path / "bank",
        EvaluationConfig(TaskConfig("unused", "tglfnn_nr"), (5, 6), 10.0),
    )
    assert calls == expected
    assert result["status"] == status
