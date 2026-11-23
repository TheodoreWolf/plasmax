"""Search orchestration preserves context, selection, and reproducible artifacts."""

from __future__ import annotations

import copy
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from experiments.agent_policy_search import evaluate, run
from experiments.agent_policy_search.artifacts import read_json, source_hash, write_json
from experiments.agent_policy_search.config import (
    EvaluationConfig,
    ModelConfig,
    SearchConfig,
    TaskConfig,
    TrackingConfig,
)

SEED_SOURCE = "# seed\ndef act(obs, rng):\n    return obs[:1]\n"
BETTER_SOURCE = "# improved\ndef act(obs, rng):\n    return obs[:1] * 0.5\n"
LOWER_SOURCE = "# lower\ndef act(obs, rng):\n    return obs[:1] * 0.2\n"
BROKEN_SOURCE = (
    "# fails at evaluation\ndef act(obs, rng):\n    raise ValueError('bad')\n"
)
NONFINITE_SOURCE = "# nonfinite\ndef act(obs, rng):\n    return obs[:1] / 0\n"


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
    id: str = "test-tracking-run"
    url: str = "https://wandb.example/test-tracking-run"
    settings: SimpleNamespace = field(
        default_factory=lambda: SimpleNamespace(mode="online")
    )
    summary: dict[str, Any] = field(default_factory=dict)
    logs: list[dict[str, Any]] = field(default_factory=list)
    artifacts: list[FakeArtifact] = field(default_factory=list)
    finish_codes: list[int] = field(default_factory=list)
    init_calls: list[dict[str, Any]] = field(default_factory=list)

    def log(self, metrics: dict[str, Any]) -> None:
        self.logs.append(copy.deepcopy(metrics))

    def log_artifact(self, artifact: FakeArtifact) -> None:
        self.artifacts.append(artifact)

    def finish(self, exit_code: int = 0) -> None:
        self.finish_codes.append(exit_code)


@dataclass(frozen=True)
class EvaluationCall:
    source: Path
    output: Path
    config: EvaluationConfig
    text: str


@pytest.fixture
def tracking(monkeypatch: pytest.MonkeyPatch) -> FakeTracking:
    tracker = FakeTracking()

    def init(**kwargs: Any) -> FakeTracking:
        tracker.init_calls.append(copy.deepcopy(kwargs))
        return tracker

    module = ModuleType("wandb")
    module.init = init
    module.Artifact = FakeArtifact
    monkeypatch.setitem(sys.modules, "wandb", module)
    monkeypatch.setattr(run, "provenance", lambda: {"git_commit": "test-checkout"})
    return tracker


def _config(tmp_path: Path, *, proposals: int = 3) -> SearchConfig:
    policy = tmp_path / "seed.py"
    policy.write_text(SEED_SOURCE)
    return SearchConfig(
        task=TaskConfig(
            env_setup="mock/circular/smoke",
            backend="bohm_gyrobohm",
            variant="realistic",
            max_steps=12,
        ),
        model=ModelConfig(model="test/provider"),
        seed_policy=policy,
        proposals=proposals,
        development_seed=41,
        development_episodes=3,
        heldout_seed=9001,
        heldout_episodes=2,
        evaluation_timeout_seconds=25,
        output_root=tmp_path / "searches",
        tracking=TrackingConfig(project="test-project", entity="test-entity"),
    )


def _response(source: str) -> dict[str, Any]:
    return {
        "id": f"response-{source_hash(source)[:8]}",
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {"source": source, "description": source.splitlines()[0]}
                    )
                }
            }
        ],
        "usage": {"prompt_tokens": 120, "completion_tokens": 30, "total_tokens": 150},
    }


def _mock_responses(
    monkeypatch: pytest.MonkeyPatch, responses: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    requests: list[dict[str, Any]] = []

    def complete(request: dict[str, Any]) -> dict[str, Any]:
        requests.append(copy.deepcopy(request))
        assert len(requests) <= len(responses), "search exceeded its proposal budget"
        return copy.deepcopy(responses[len(requests) - 1])

    monkeypatch.setattr(run.model, "complete", complete)
    return requests


def _mock_evaluator(monkeypatch: pytest.MonkeyPatch) -> list[EvaluationCall]:
    calls: list[EvaluationCall] = []

    def evaluate_candidate(
        source: Path, output: Path, config: EvaluationConfig
    ) -> dict[str, Any]:
        text = source.read_text()
        calls.append(EvaluationCall(source, output, config, text))
        output.mkdir(parents=True)
        if text == BROKEN_SOURCE:
            report = {
                "status": "error",
                "finite": False,
                "error": "policy runtime failure: bad",
            }
        else:
            heldout = config.seeds[0] == 9001
            reward = {
                SEED_SOURCE: 1.0,
                BETTER_SOURCE: 5.0,
                LOWER_SOURCE: 0.5,
                NONFINITE_SOURCE: float("inf"),
            }[text]
            if heldout:
                reward = 918.25
            report = {
                "status": "ok",
                "finite": text != NONFINITE_SOURCE,
                "mean_return": reward,
                "interface": {"action_shape": [1], "dtype": "float32"},
                "feedback": {
                    "returns": [reward] * len(config.seeds)
                    if text != NONFINITE_SOURCE
                    else ["Infinity"] * len(config.seeds),
                    "seeds": list(config.seeds),
                    "marker": "HELDOUT_ONLY_DIAGNOSTIC" if heldout else "development",
                    "episodes": [
                        {"length": length} for length in (2, 5, 0)[: len(config.seeds)]
                    ],
                    "nonfinite_reward_count": 2 if text == NONFINITE_SOURCE else 0,
                },
            }
        write_json(output / "report.json", report)
        (output / "trajectories.npz").write_bytes(b"mock numeric trajectories")
        return report

    monkeypatch.setattr(run, "evaluate_candidate", evaluate_candidate)
    monkeypatch.setattr(evaluate, "evaluate_candidate", evaluate_candidate)
    return calls


def _events(run_dir: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in (run_dir / "events.jsonl").read_text().splitlines()
    ]


@pytest.mark.parametrize("failure", ["parse", "runtime"])
def test_failed_then_better_then_lower_proposals_preserve_search_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tracking: FakeTracking,
    failure: str,
) -> None:
    config = _config(tmp_path)
    first = (
        {"choices": [{"message": {"content": "not valid JSON"}}]}
        if failure == "parse"
        else _response(BROKEN_SOURCE)
    )
    responses = [first, _response(BETTER_SOURCE), _response(LOWER_SOURCE)]
    requests = _mock_responses(monkeypatch, responses)
    calls = _mock_evaluator(monkeypatch)

    result = run.run_search(config)

    assert len(requests) == 3
    assert read_json(result / "best.json") == {"candidate_id": "0002"}
    assert read_json(result / "seeds.json") == {
        "development": [41, 42, 43],
        "heldout": [9001, 9002],
    }
    assert all(call.config.seeds == (41, 42, 43) for call in calls[:-1])
    assert all(call.config.task == config.task for call in calls)
    assert all(call.config.timeout_seconds == 25 for call in calls)
    assert calls[-1].config.seeds == (9001, 9002)
    assert calls[-1].source == result / "candidates" / "0002" / "policy.py"
    assert sum(call.output.name == "heldout" for call in calls) == 1
    assert len(calls) == (4 if failure == "parse" else 5)

    contexts = [json.loads(request["messages"][1]["content"]) for request in requests]
    assert contexts[0]["latest_attempt"] is None
    assert contexts[0]["incumbent"]["source"] == SEED_SOURCE
    assert contexts[1]["incumbent"]["source"] == SEED_SOURCE
    assert contexts[1]["latest_attempt"]["status"] == "error"
    assert contexts[1]["latest_attempt"]["candidate_id"] == "0001"
    assert contexts[1]["latest_attempt"]["error"]
    assert contexts[2]["incumbent"]["source"] == BETTER_SOURCE
    assert contexts[2]["latest_attempt"]["candidate_id"] == "0002"
    assert "HELDOUT_ONLY_DIAGNOSTIC" not in json.dumps(requests)
    assert "9001" not in json.dumps(requests)
    assert "918.25" not in json.dumps(requests)
    for index, (request, response) in enumerate(
        zip(requests, responses, strict=True), 1
    ):
        directory = result / "candidates" / f"{index:04d}"
        assert read_json(directory / "request.json") == request
        assert read_json(directory / "response.json") == response
        expected_parent = "0002" if index == 3 else "0000"
        assert read_json(directory / "parent.json") == {"parent_id": expected_parent}

    for candidate_id, source, parent in (
        ("0000", SEED_SOURCE, None),
        ("0002", BETTER_SOURCE, "0000"),
        ("0003", LOWER_SOURCE, "0002"),
    ):
        directory = result / "candidates" / candidate_id
        assert (directory / "policy.py").read_text() == source
        metadata = read_json(directory / "candidate.json")
        assert metadata["source_sha256"] == source_hash(source)
        assert metadata["parent_id"] == parent
    accepted = [
        read_json(result / "candidates" / candidate_id / "outcome.json")["accepted"]
        for candidate_id in ("0001", "0002", "0003")
    ]
    assert accepted == [False, True, False]
    summary = read_json(result / "summary.json")
    assert summary["best_candidate_id"] == "0002"
    assert summary["development_mean_return"] == 5.0
    assert summary["heldout"]["mean_return"] == 918.25
    assert tracking.init_calls[0]["mode"] == "online"
    assert tracking.finish_codes == [0]
    assert len(tracking.artifacts) == 4
    assert all(
        "search_config.json" in artifact.files for artifact in tracking.artifacts
    )
    assert all(
        path.suffix != ".npz"
        for artifact in tracking.artifacts
        for path in artifact.files.values()
    )
    assert _events(result)[-1]["event"] == "completed"


def test_nonfinite_candidate_never_replaces_finite_incumbent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tracking: FakeTracking
) -> None:
    config = _config(tmp_path, proposals=2)
    requests = _mock_responses(
        monkeypatch, [_response(NONFINITE_SOURCE), _response(LOWER_SOURCE)]
    )
    calls = _mock_evaluator(monkeypatch)

    result = run.run_search(config)

    assert read_json(result / "best.json") == {"candidate_id": "0000"}
    assert calls[-1].text == SEED_SOURCE
    outcome = read_json(result / "candidates" / "0001" / "outcome.json")
    assert not outcome["accepted"]
    assert outcome["mean_return"] == "Infinity"
    context = json.loads(requests[1]["messages"][1]["content"])
    assert context["incumbent"]["source"] == SEED_SOURCE
    assert context["latest_attempt"]["feedback"]["returns"] == ["Infinity"] * 3
    nonfinite_log = next(
        log for log in tracking.logs if log.get("search/nonfinite") == 1
    )
    assert nonfinite_log["evaluation/nonfinite_reward_rate"] == pytest.approx(2 / 7)


def test_api_failure_stops_and_preserves_exact_request_and_previous_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tracking: FakeTracking
) -> None:
    config = _config(tmp_path)
    calls = _mock_evaluator(monkeypatch)
    requests: list[dict[str, Any]] = []

    def fail(request: dict[str, Any]) -> dict[str, Any]:
        requests.append(copy.deepcopy(request))
        raise RuntimeError("API authentication failed")

    monkeypatch.setattr(run.model, "complete", fail)
    with pytest.raises(RuntimeError, match="API authentication failed"):
        run.run_search(config)

    [result] = config.output_root.iterdir()
    assert len(calls) == 1
    assert len(requests) == 1
    assert read_json(result / "best.json") == {"candidate_id": "0000"}
    candidate = result / "candidates" / "0001"
    assert read_json(candidate / "request.json") == requests[0]
    assert (
        "API authentication failed" in read_json(candidate / "api_error.json")["error"]
    )
    assert not (candidate / "response.json").exists()
    assert not (result / "heldout").exists()
    assert not (result / "candidates" / "0002").exists()
    assert (result / "candidates" / "0000" / "policy.py").read_text() == SEED_SOURCE
    assert _events(result)[-1]["event"] == "failed"
    assert tracking.finish_codes == [1]


@pytest.mark.parametrize("failure", ["exception", "offline"])
def test_tracking_initialization_failure_stops_before_evaluation_or_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tracking: FakeTracking,
    failure: str,
) -> None:
    config = _config(tmp_path)

    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("must stop before evaluator or model")

    monkeypatch.setattr(run, "evaluate_candidate", forbidden)
    monkeypatch.setattr(run.model, "complete", forbidden)
    if failure == "exception":

        def failed_init(**kwargs: Any) -> None:
            assert kwargs["mode"] == "online"
            raise RuntimeError("tracking authorization failed")

        monkeypatch.setattr(sys.modules["wandb"], "init", failed_init)
        expected = "tracking authorization failed"
    else:
        tracking.settings.mode = "offline"
        expected = "online initialization is required"

    with pytest.raises(RuntimeError, match=expected):
        run.run_search(config)

    [result] = config.output_root.iterdir()
    assert read_json(result / "config.json")["model"]["model"] == "test/provider"
    assert read_json(result / "seeds.json")["development"] == [41, 42, 43]
    assert read_json(result / "provenance.json") == {"git_commit": "test-checkout"}
    assert _events(result)[-1]["event"] == "failed"
    assert not (result / "candidates").exists()
    assert tracking.finish_codes == ([] if failure == "exception" else [1])


@pytest.mark.parametrize("split", ["development", "heldout"])
def test_evaluation_only_reuses_saved_source_config_and_seeds_without_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tracking: FakeTracking, split: str
) -> None:
    config = _config(tmp_path, proposals=1)
    _mock_responses(monkeypatch, [_response(BETTER_SOURCE)])
    calls = _mock_evaluator(monkeypatch)
    result = run.run_search(config)
    before = {
        path.relative_to(result): path.read_bytes()
        for path in result.rglob("*")
        if path.is_file()
    }
    config.seed_policy.write_text("changed original seed file")

    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("evaluation-only must not invoke a model")

    monkeypatch.setattr(run.model, "complete", forbidden)
    monkeypatch.setattr(run.model, "build_request", forbidden)
    calls.clear()
    replay = evaluate.EvaluateConfig(
        run_dir=result, output_dir=tmp_path / "replay", split=split
    )
    report = evaluate.evaluate_saved(replay)

    assert len(calls) == 1
    call = calls[0]
    assert call.source == result / "candidates" / "0001" / "policy.py"
    assert call.text == BETTER_SOURCE
    assert asdict(call.config.task) == read_json(result / "config.json")["task"]
    assert call.config.seeds == tuple(read_json(result / "seeds.json")[split])
    assert call.config.timeout_seconds == config.evaluation_timeout_seconds
    assert report["status"] == "ok"
    assert read_json(replay.output_dir / "replay.json")["split"] == split
    after = {
        path.relative_to(result): path.read_bytes()
        for path in result.rglob("*")
        if path.is_file()
    }
    assert before == after
    assert tracking.init_calls[-1]["mode"] == "online"
    assert tracking.finish_codes == [0, 0]


def test_evaluation_only_rejects_tampered_archived_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tracking: FakeTracking
) -> None:
    config = _config(tmp_path, proposals=0)
    _mock_evaluator(monkeypatch)
    result = run.run_search(config)
    (result / "candidates" / "0000" / "policy.py").write_text("tampered")
    with pytest.raises(ValueError, match="source hash mismatch"):
        evaluate.evaluate_saved(
            evaluate.EvaluateConfig(run_dir=result, output_dir=tmp_path / "replay")
        )
    assert len(tracking.init_calls) == 1
    assert not (tmp_path / "replay").exists()


def test_crlf_proposal_preserves_exact_source_bytes_and_replays(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tracking: FakeTracking
) -> None:
    config = _config(tmp_path, proposals=1)
    source = BETTER_SOURCE.replace("\n", "\r\n")
    _mock_responses(monkeypatch, [_response(source)])
    calls = _mock_evaluator(monkeypatch)

    result = run.run_search(config)

    candidate = result / "candidates" / "0001"
    assert (candidate / "policy.py").read_bytes() == source.encode("utf-8")
    assert read_json(candidate / "candidate.json")["source_sha256"] == source_hash(
        source
    )
    evaluate.evaluate_saved(
        evaluate.EvaluateConfig(run_dir=result, output_dir=tmp_path / "replay")
    )
    assert calls[-1].source == candidate / "policy.py"
    assert tracking.finish_codes == [0, 0]
