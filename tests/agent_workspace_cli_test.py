"""Coding-agent workspace commands without API calls or simulator rollouts."""

from __future__ import annotations

import ast
import copy
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest

from experiments.agent_policy_search.artifacts import read_json, write_json
from experiments.agent_policy_search.config import EvaluationConfig, TaskConfig
from experiments.agent_policy_workspace import cli
from experiments.agent_policy_workspace.ledger import Ledger, bundle_sha256

INTERFACE = {
    "action_shape": [2],
    "action_dtype": "float32",
    "action_low": [-1.0, -2.0],
    "action_high": [1.0, 2.0],
    "observation_shape": [3],
}


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(cli, "provenance", lambda: {"git_commit": "test-checkout"})
    monkeypatch.setattr(
        cli, "describe_task", lambda *_args: {"status": "ok", "interface": INTERFACE}
    )
    root = tmp_path / "workspace"
    cli.init(
        root,
        TaskConfig("mock/circular/smoke", "constant"),
        development_episodes=2,
        heldout_episodes=2,
        development_seed=11,
        heldout_seed=101,
    )
    return root


@dataclass(frozen=True)
class EvaluationCall:
    source: Path
    output: Path
    seeds: tuple[int, ...]
    label: str
    final: bool


def _report(seeds: tuple[int, ...], value: float = 1.0) -> dict[str, Any]:
    return {
        "status": "ok",
        "finite": True,
        "mean_return": value,
        "interface": INTERFACE,
        "evaluation_seconds": 0.1,
        "feedback": {
            "episodes": [
                {"seed": seed, "return": value, "length": 2} for seed in seeds
            ],
            "nonfinite_reward_count": 0,
        },
    }


def _save_trace(output: Path, seeds: tuple[int, ...], value: float) -> None:
    output.mkdir(parents=True, exist_ok=True)
    count = len(seeds)
    np.savez_compressed(
        output / "trajectories.npz",
        valid=np.tile([True, True, False, False], (count, 1)),
        obs=np.tile(np.arange(12).reshape(1, 4, 3), (count, 1, 1)),
        requested_action=np.zeros((count, 4, 2)),
        reward=np.tile([value / 2, value / 2, 999.0, 999.0], (count, 1)),
        trace_error=np.tile([3.0, 2.0, 777.0, 777.0], (count, 1)),
        Q_fusion=np.full((count, 4), 999.0),
    )


@pytest.fixture
def evaluations(monkeypatch: pytest.MonkeyPatch) -> list[EvaluationCall]:
    calls = []

    def evaluate(
        root: Path,
        source: Path,
        output: Path,
        seeds: tuple[int, ...],
        label: str,
        *,
        final: bool = False,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        if final:
            assert (root / ".policy_eval/finalizing.json").exists()
        calls.append(EvaluationCall(source, output, seeds, label, final))
        value = 3.0 if "# improved" in source.read_text() else 1.0
        if label == "exploratory":
            value = 1000.0
        _save_trace(output, seeds, value)
        result = _report(seeds, value)
        write_json(output / "result.json", result)
        return result

    monkeypatch.setattr(cli, "_evaluate", evaluate)
    return calls


def test_init_renders_task_contract_and_initial_attempt(workspace: Path) -> None:
    config = cli._config(workspace)
    assert config.development_seeds == (11, 12)
    assert config.heldout_seeds == (101, 102)
    assert config.task.variant == "realistic"
    source = (workspace / "policy.py").read_text()
    ast.parse(source)
    assert "__ACTION_" not in source
    assert "__INITIAL_ACTION__" not in source
    assert "jnp.asarray([0.0, 0.0], dtype=jnp.float32)" in source
    assert (workspace / "context/starting_policy.py").read_text() == source
    assert read_json(workspace / "context/interface.json") == INTERFACE
    assert (workspace / "AGENTS.md").is_file()
    assert (workspace / "rollouts.py").is_file()
    assert Ledger(workspace).current_attempt() == "A00"
    assert cli.lifetime(workspace)["remaining_seconds"] > 0


def test_official_scoring_and_exploration_have_distinct_selection_rules(
    workspace: Path, evaluations: list[EvaluationCall]
) -> None:
    first = cli.score(workspace)
    assert evaluations[0].seeds == (11, 12)
    assert Ledger(workspace).best_candidate() == first["candidate_sha256"]
    assert cli.score(workspace) == first
    assert len(evaluations) == 1
    with (workspace / "policy.py").open("a") as stream:
        stream.write("\n# improved\n")
    (workspace / "custom_primitives.py").write_text("GAIN = 2\n")
    better = cli.score(workspace)
    assert better["mean_return"] == 3.0
    assert Ledger(workspace).best_candidate() == better["candidate_sha256"]
    assert evaluations[1].seeds == (11, 12)
    archive = workspace / ".policy_eval/candidates" / better["candidate_sha256"]
    assert (archive / "custom_primitives.py").read_text() == "GAIN = 2\n"
    with pytest.raises(ValueError, match="highest mean"):
        cli.select_best(workspace, first["candidate_sha256"], "older is preferable")
    with (workspace / "policy.py").open("a") as stream:
        stream.write("\n# exploratory change\n")
    exploratory = cli.explore(workspace, seeds=(20, 21))
    assert exploratory["mean_return"] == 1000.0
    assert evaluations[-1].seeds == (20, 21)
    assert Ledger(workspace).best_candidate() == better["candidate_sha256"]
    with pytest.raises((ValueError, FileNotFoundError)):
        cli.select_best(workspace, exploratory["candidate_sha256"], "exploratory win")
    count = len(evaluations)
    with pytest.raises(ValueError, match="heldout"):
        cli.explore(workspace, seeds=(101,))
    assert len(evaluations) == count


def test_trace_limits_context_to_valid_observable_windows(
    workspace: Path, evaluations: list[EvaluationCall]
) -> None:
    cli.score(workspace)
    result = cli.trace(
        workspace, episode=1, start=1, stop=4, fields=("reward", "trace_error")
    )
    assert result == {
        "episode": 1,
        "steps": [1],
        "fields": {"reward": [0.5], "trace_error": [2.0]},
    }
    assert len(evaluations) == 1
    with pytest.raises(ValueError, match="observable"):
        cli.trace(workspace, start=0, stop=2, fields=("Q_fusion",))
    with pytest.raises(ValueError, match="1 to 200"):
        cli.trace(workspace, stop=201)
    with pytest.raises(ValueError, match="outside"):
        cli.trace(workspace, episode=2, stop=2)
    with pytest.raises(ValueError, match="exploration identifier"):
        cli.trace(workspace, exploration="../../config.json", stop=2)


def test_failed_score_preserves_attempt_but_allows_same_bundle_retry(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []

    def evaluate(*args: Any, **kwargs: Any) -> dict[str, Any]:
        del kwargs
        calls.append(args)
        if len(calls) == 1:
            return {
                "status": "error",
                "finite": False,
                "mean_return": None,
                "error": "temporary worker failure",
            }
        return _report((11, 12))

    monkeypatch.setattr(cli, "_evaluate", evaluate)
    first = cli.score(workspace)
    archive = workspace / ".policy_eval/candidates" / first["candidate_sha256"]
    assert first["status"] == "error"
    assert not (archive / "score.json").exists()
    assert Ledger(workspace).best_candidate() is None
    second = cli.score(workspace)
    assert second["status"] == "ok"
    assert second["candidate_sha256"] == first["candidate_sha256"]
    assert (archive / "score.json").is_file()
    assert cli.score(workspace) == second
    assert len(calls) == 2
    events = Ledger(workspace).events()
    assert [event["status"] for event in events if event["event"] == "scored"] == [
        "error",
        "ok",
    ]


def test_finish_restores_exact_best_and_rescores_before_heldout(
    workspace: Path, evaluations: list[EvaluationCall]
) -> None:
    scored = cli.score(workspace)
    source = (workspace / "policy.py").read_bytes()
    (workspace / "policy.py").write_text("# displaced unscored policy\n")
    (workspace / "custom_primitives.py").write_text("GAIN = 10\n")
    result = cli.finish(workspace)
    assert result["candidate_sha256"] == scored["candidate_sha256"]
    assert [call.label for call in evaluations] == [
        "development",
        "final-development",
        "final-heldout",
    ]
    assert [call.seeds for call in evaluations] == [(11, 12), (11, 12), (101, 102)]
    assert all(call.final for call in evaluations[1:])
    assert (workspace / "policy.py").read_bytes() == source
    assert not (workspace / "custom_primitives.py").exists()
    backups = list((workspace / ".policy_backups").iterdir())
    assert len(backups) == 1
    assert (backups[0] / "policy.py").read_text() == "# displaced unscored policy\n"
    assert (backups[0] / "custom_primitives.py").read_text() == "GAIN = 10\n"
    assert bundle_sha256(workspace) == scored["candidate_sha256"]
    assert cli.finish(workspace) == result
    assert len(evaluations) == 3
    with pytest.raises(ValueError, match="finalized or finalizing"):
        cli.score(workspace)
    with pytest.raises(ValueError, match="finalized or finalizing"):
        cli.explore(workspace)


def test_failed_independent_rescore_never_runs_heldout(
    workspace: Path, evaluations: list[EvaluationCall], monkeypatch: pytest.MonkeyPatch
) -> None:
    cli.score(workspace)
    final_calls = []

    def disagree(*args: Any, **kwargs: Any) -> dict[str, Any]:
        final_calls.append((args, kwargs))
        return _report((11, 12), 2.0)

    monkeypatch.setattr(cli, "_evaluate", disagree)
    with pytest.raises(AssertionError, match="independent development"):
        cli.finish(workspace)
    assert len(final_calls) == 1
    assert final_calls[0][0][4] == "final-development"
    assert not (workspace / ".policy_eval/final.json").exists()
    assert cli.lifetime(workspace)["finalizing"] is True
    with pytest.raises(ValueError, match="finalized or finalizing"):
        cli.score(workspace)


def test_heldout_failure_keeps_selection_frozen_and_can_retry(
    workspace: Path, evaluations: list[EvaluationCall], monkeypatch: pytest.MonkeyPatch
) -> None:
    scored = cli.score(workspace)
    labels = []
    fail = True

    def evaluate(*args: Any, **kwargs: Any) -> dict[str, Any]:
        assert kwargs["final"] is True
        seeds, label = args[3], args[4]
        labels.append(label)
        if label == "final-heldout" and fail:
            return {"status": "error", "finite": False, "error": "worker failure"}
        return _report(seeds)

    monkeypatch.setattr(cli, "_evaluate", evaluate)
    with pytest.raises(RuntimeError, match="heldout"):
        cli.finish(workspace)
    assert not (workspace / ".policy_eval/final.json").exists()
    assert cli.lifetime(workspace)["finalizing"] is True
    assert Ledger(workspace).best_candidate() == scored["candidate_sha256"]
    with pytest.raises(ValueError, match="finalized or finalizing"):
        cli.select_best(workspace, scored["candidate_sha256"], "try again")
    fail = False
    result = cli.finish(workspace)
    assert result["candidate_sha256"] == scored["candidate_sha256"]
    assert result["heldout"]["status"] == "ok"
    assert labels == [
        "final-development",
        "final-heldout",
        "final-development",
        "final-heldout",
    ]


@dataclass
class FakeArtifact:
    name: str
    type: str
    metadata: dict[str, Any]
    files: dict[str, Path] = field(default_factory=dict)

    def add_file(self, path: str, *, name: str) -> None:
        assert Path(path).is_file()
        self.files[name] = Path(path)


@dataclass
class FakeTracking:
    id: str = "test-run"
    url: str = "https://wandb.example/test-run"
    settings: SimpleNamespace = field(
        default_factory=lambda: SimpleNamespace(mode="online")
    )
    summary: dict[str, Any] = field(default_factory=dict)
    logs: list[dict[str, Any]] = field(default_factory=list)
    artifacts: list[FakeArtifact] = field(default_factory=list)
    finishes: list[int] = field(default_factory=list)

    def log(self, values: dict[str, Any]) -> None:
        self.logs.append(copy.deepcopy(values))

    def log_artifact(self, artifact: FakeArtifact) -> None:
        self.artifacts.append(artifact)

    def finish(self, exit_code: int = 0) -> None:
        self.finishes.append(exit_code)


def _fake_wandb(monkeypatch: pytest.MonkeyPatch, init: Any) -> None:
    module = ModuleType("wandb")
    module.init = init
    module.Artifact = FakeArtifact
    monkeypatch.setitem(sys.modules, "wandb", module)


def test_evaluate_logs_online_and_preserves_source_artifact(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tracker = FakeTracking()
    init_calls = []

    def init(**kwargs: Any) -> FakeTracking:
        init_calls.append(kwargs)
        return tracker

    _fake_wandb(monkeypatch, init)
    configs = []

    def evaluate_bundle(
        source: Path, output: Path, config: EvaluationConfig
    ) -> dict[str, Any]:
        assert source.read_text() == (workspace / "policy.py").read_text()
        configs.append(config)
        _save_trace(output, config.seeds, 1.0)
        report = _report(config.seeds)
        report["feedback"]["episodes"][0]["length"] = 5
        report["feedback"]["nonfinite_reward_count"] = 2
        return report

    monkeypatch.setattr(cli, "evaluate_bundle", evaluate_bundle)
    output = workspace / ".policy_eval/test-evaluation"
    result = cli._evaluate(
        workspace, workspace / "policy.py", output, (11, 12), "development"
    )
    assert result["status"] == "ok"
    assert init_calls[0]["mode"] == "online"
    assert configs[0].seeds == (11, 12)
    assert configs[0].task == cli._config(workspace).task
    assert tracker.logs[0]["evaluation/nonfinite_reward_rate"] == pytest.approx(2 / 7)
    assert tracker.finishes == [0]
    assert "policy.py" in tracker.artifacts[0].files
    assert "config.json" in tracker.artifacts[0].files
    assert read_json(output / "tracking.json")["id"] == tracker.id


@pytest.mark.parametrize("failure", ["authentication", "offline"])
def test_tracking_failure_stops_before_evaluation_without_fallback(
    workspace: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    tracker = FakeTracking(settings=SimpleNamespace(mode="offline"))
    calls = []

    def init(**kwargs: Any) -> FakeTracking:
        calls.append(kwargs)
        if failure == "authentication":
            raise RuntimeError("authentication failed")
        return tracker

    def unexpected(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail("simulator must not run when tracking initialization fails")

    _fake_wandb(monkeypatch, init)
    monkeypatch.setattr(cli, "evaluate_bundle", unexpected)
    output = workspace / ".policy_eval/failed-evaluation"
    with pytest.raises(RuntimeError, match="authentication|online initialization"):
        cli._evaluate(
            workspace, workspace / "policy.py", output, (11, 12), "development"
        )
    assert len(calls) == 1
    assert calls[0]["mode"] == "online"
    assert (output / "command_error.json").is_file()
    assert tracker.finishes == ([] if failure == "authentication" else [1])


@pytest.mark.parametrize("init_seconds", [70.0, 101.0])
def test_final_evaluation_rechecks_deadline_after_tracking_initialization(
    workspace: Path, monkeypatch: pytest.MonkeyPatch, init_seconds: float
) -> None:
    config = cli._config(workspace)
    clock = [config.created_at + config.lifetime_seconds - 100.0]
    monkeypatch.setattr(cli, "time", SimpleNamespace(time=lambda: clock[0]))
    tracker = FakeTracking()
    configs = []

    def init(**_kwargs: Any) -> FakeTracking:
        clock[0] += init_seconds
        return tracker

    def evaluate_bundle(
        source: Path, output: Path, evaluation: EvaluationConfig
    ) -> dict[str, Any]:
        configs.append(evaluation)
        _save_trace(output, evaluation.seeds, 1.0)
        return _report(evaluation.seeds)

    _fake_wandb(monkeypatch, init)
    monkeypatch.setattr(cli, "evaluate_bundle", evaluate_bundle)
    arguments = (
        workspace,
        workspace / "policy.py",
        workspace / ".policy_eval/final-clock",
        (11, 12),
        "final-development",
    )
    if init_seconds < 100:
        assert cli._evaluate(*arguments, final=True)["status"] == "ok"
        assert len(configs) == 1
        assert configs[0].timeout_seconds == pytest.approx(30.0)
        assert tracker.finishes == [0]
    else:
        with pytest.raises(TimeoutError):
            cli._evaluate(*arguments, final=True)
        assert configs == []
        assert tracker.finishes == [1]


def test_final_development_timeout_preserves_best_without_launching_heldout(
    workspace: Path, evaluations: list[EvaluationCall], monkeypatch: pytest.MonkeyPatch
) -> None:
    scored = cli.score(workspace)
    source = (workspace / "policy.py").read_bytes()
    (workspace / "policy.py").write_text("# displaced unscored policy\n")
    labels = []

    def timed_out(*args: Any, **_kwargs: Any) -> dict[str, Any]:
        labels.append(args[4])
        return {"status": "timeout", "finite": False, "error": "deadline"}

    monkeypatch.setattr(cli, "_evaluate", timed_out)
    with pytest.raises(TimeoutError):
        cli.finish(workspace)
    assert labels == ["final-development"]
    assert Ledger(workspace).best_candidate() == scored["candidate_sha256"]
    assert (workspace / "policy.py").read_bytes() == source
    assert not (workspace / ".policy_eval/final.json").exists()
