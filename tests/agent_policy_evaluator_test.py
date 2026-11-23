"""Generated policy workers and feedback, using cheap synthetic trajectories."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from experiments.agent_policy_search.config import EvaluationConfig, TaskConfig
from experiments.agent_policy_search.evaluator import (
    _descriptor,
    _evaluate_in_worker,
    _report,
    evaluate_candidate,
)
from experiments.agent_policy_search.feedback import build_feedback


def _arrays() -> dict[str, np.ndarray]:
    return {
        "valid": np.asarray([[True, True, False], [True, False, False]]),
        "reward": np.asarray([[1.0, 2.0, np.nan], [4.0, 900.0, 900.0]]),
        "terminated": np.asarray([[False, True, False], [False, False, False]]),
        "truncated": np.asarray([[False, False, False], [True, False, False]]),
        "termination_code": np.asarray([[-1, 2, -1], [-1, -1, -1]]),
        "obs": np.asarray([[[1.0], [3.0], [900.0]], [[5.0], [900.0], [900.0]]]),
        "next_obs": np.asarray([[[2.0], [4.0], [900.0]], [[6.0], [900.0], [900.0]]]),
        "requested_action": np.asarray(
            [[[0.1], [0.3], [900.0]], [[0.5], [900.0], [900.0]]]
        ),
        "Q_fusion": np.full((2, 3), 999.0),
    }


def test_feedback_masks_padding_and_excludes_privileged_values() -> None:
    feedback = build_feedback(
        _arrays(), {"observations": {"density": [0, 1]}}, (10, 20)
    )
    assert [episode["return"] for episode in feedback["episodes"]] == [3.0, 4.0]
    assert [episode["length"] for episode in feedback["episodes"]] == [2, 1]
    assert [episode["termination_reason"] for episode in feedback["episodes"]] == [
        "greenwald_disruption",
        "truncated",
    ]
    assert feedback["nonfinite_reward_count"] == 0
    stats = feedback["trajectory_summary"]["obs"][0]
    assert stats == {
        "index": 0,
        "name": "density",
        "mean": 3.0,
        "min": 1.0,
        "max": 5.0,
        "nonfinite_count": 0,
        "segment_means": [3.0, 3.0, None, None],
    }
    assert "Q_fusion" not in json.dumps(feedback)


def test_feedback_retains_nonfinite_returns_as_labels_without_repair() -> None:
    arrays = _arrays()
    arrays["reward"][0, 0] = np.nan
    arrays["obs"][0, 0] = np.inf
    feedback = build_feedback(arrays, {}, (10, 20))
    assert feedback["episodes"][0]["return"] == "NaN"
    assert feedback["nonfinite_reward_count"] == 1
    assert feedback["trajectory_summary"]["obs"][0]["nonfinite_count"] == 1
    assert np.isnan(arrays["reward"][0, 0])
    assert np.isinf(arrays["obs"][0, 0, 0])
    json.dumps(feedback, allow_nan=False)


def test_nonfinite_requested_actions_prevent_selection_with_finite_returns() -> None:
    arrays = _arrays()
    arrays["requested_action"][0, 0, 0] = np.inf
    report = _report(arrays, {}, (10, 20), np.asarray([3.0, 4.0]))
    assert report["finite"] is False
    assert report["finite_returns"] is True
    assert report["finite_requested_actions"] is False
    assert report["mean_return"] is None
    assert report["feedback"]["nonfinite_requested_action_count"] == 1


def test_descriptor_resolves_default_reward_implementation() -> None:
    from helpers import CheapBoundaryEnv

    from plasmax import rewards

    cheap = CheapBoundaryEnv(obs_dim=2, action_low=(-1.0,), action_high=(1.0,))
    env = SimpleNamespace(
        observation_space=cheap.observation_space,
        action_space=cheap.action_space,
        unwrapped=SimpleNamespace(_dynamics=SimpleNamespace(_reward_fn=rewards.P_diff)),
        plasmax_config=SimpleNamespace(task=SimpleNamespace(reward="P_diff")),
    )
    descriptor = _descriptor(env, TaskConfig("mock/circular/smoke", "constant"))
    assert descriptor["task"]["resolved_reward"] == "P_diff"
    assert "def P_diff(" in descriptor["reward_definition"]["source"]
    assert (
        "def _terminal_reward("
        in descriptor["reward_definition"]["supporting_module_source"]
    )


def test_worker_evaluates_explicit_seeds_and_saves_numeric_trajectories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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
    source = tmp_path / "policy.py"
    source.write_text(
        "import jax\ndef act(obs, rng):\n"
        "    return jax.random.uniform(rng, (1,), dtype='float32')\n"
    )
    config = EvaluationConfig(TaskConfig("mock/circular/smoke", "constant"), (7, 17))
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    report = _evaluate_in_worker(source, first, config)
    _evaluate_in_worker(source, second, config)
    assert report["status"] == "ok"
    assert report["finite"] is True
    assert report["mean_return"] == 3.0
    assert report["interface"]["action_dtype"] == "float32"
    assert report["interface"]["task"]["horizon"] == 4
    assert (first / "episodes" / "0000_seed_7.npz").exists()
    assert [item["seed"] for item in report["feedback"]["episodes"]] == [7, 17]
    with (
        np.load(first / "trajectories.npz", allow_pickle=False) as saved,
        np.load(second / "trajectories.npz", allow_pickle=False) as repeated,
    ):
        np.testing.assert_array_equal(
            saved["requested_action"], repeated["requested_action"]
        )
        np.testing.assert_array_equal(saved["valid"], [[True, True, False, False]] * 2)
        assert json.loads(str(saved["metadata_json"]))["seeds"] == [7, 17]
    source.write_text(
        "import jax.numpy as jnp\ndef act(obs, rng):\n"
        "    return jnp.zeros((2,), dtype=jnp.float32)\n"
    )
    invalid = tmp_path / "invalid"
    invalid.mkdir()
    with pytest.raises(ValueError, match="Policy action shape"):
        _evaluate_in_worker(source, invalid, config)


@pytest.mark.parametrize(
    "source_text,timeout,status",
    [("def act(:\n", 15.0, "error"), ("import time\ntime.sleep(60)\n", 0.2, "timeout")],
)
def test_subprocess_failure_artifacts(
    tmp_path: Path, source_text: str, timeout: float, status: str
) -> None:
    source = tmp_path / "policy.py"
    source.write_text(source_text)
    output = tmp_path / "evaluation"
    report = evaluate_candidate(
        source, output, EvaluationConfig(TaskConfig("unused"), (1,), timeout)
    )
    assert report["status"] == status
    assert report["finite"] is False
    assert report["mean_return"] is None
    assert {"request.json", "result.json", "stdout.txt", "stderr.txt"} <= {
        path.name for path in output.iterdir()
    }
    if status == "error":
        assert "SyntaxError" in report["error"]


def test_tglfnn_seed_bank_uses_separate_workers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from experiments.agent_policy_search import evaluator

    calls = []

    def worker(
        source: Path, output_dir: Path, config: EvaluationConfig
    ) -> dict[str, Any]:
        del source
        calls.append(config.seeds)
        output_dir.mkdir(parents=True)
        arrays = {
            name: value[len(calls) - 1 : len(calls)]
            for name, value in _arrays().items()
        }
        np.savez_compressed(
            output_dir / "trajectories.npz", **arrays, metadata_json=np.asarray("{}")
        )
        return {
            "status": "ok",
            "interface": {},
            "feedback": build_feedback(arrays, {}, config.seeds),
            "evaluation_seconds": 1.0,
        }

    monkeypatch.setattr(evaluator, "_run_worker", worker)
    report = evaluate_candidate(
        tmp_path / "policy.py",
        tmp_path / "out",
        EvaluationConfig(TaskConfig("mock/circular/smoke", "tglfnn_nr"), (3, 4)),
    )
    assert calls == [(3,), (4,)]
    assert report["mean_return"] == 3.5
    assert report["evaluation_seconds"] == 2.0
    with np.load(tmp_path / "out" / "trajectories.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["valid"], _arrays()["valid"])
