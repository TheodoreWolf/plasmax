"""A bounded policy-writing agent records evidence without running policy code."""

from __future__ import annotations

import copy
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from experiments.agent_policy_search.artifacts import read_json, write_json
from experiments.agent_policy_search.config import TaskConfig
from experiments.agent_policy_workspace import agent, cli, runtime
from experiments.agent_policy_workspace.ledger import Ledger

HELDOUT_RETURN = 979713.25


@dataclass
class Harness:
    root: Path
    timeline: list[str] = field(default_factory=list)


@pytest.fixture
def harness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Harness:
    harness = Harness(tmp_path / "workspace")
    interface = {
        "action_shape": [2],
        "action_dtype": "float32",
        "action_low": [-1.0, -1.0],
        "action_high": [1.0, 1.0],
        "observation_shape": [3],
    }
    monkeypatch.setattr(cli, "provenance", lambda: {})
    monkeypatch.setattr(
        cli, "describe_task", lambda *_args: {"status": "ok", "interface": interface}
    )
    cli.init(
        harness.root,
        TaskConfig("mock/circular/smoke", "constant", max_steps=2),
        development_episodes=2,
        heldout_episodes=2,
        development_seed=11,
        heldout_seed=270001,
    )

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
        harness.timeline.append(label)
        if final:
            assert (root / ".policy_eval/finalizing.json").exists()
        text = source.read_text()
        if "# failed" in text:
            result = {
                "status": "error",
                "finite": False,
                "error": "worker rejected policy",
            }
        else:
            value = 3.0 if "# improved" in text else 1.0
            if label == "final-heldout":
                value = HELDOUT_RETURN
            output.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                output / "trajectories.npz",
                valid=np.ones((len(seeds), 2), dtype=bool),
                obs=np.zeros((len(seeds), 2, 3)),
                requested_action=np.zeros((len(seeds), 2, 2)),
                reward=np.full((len(seeds), 2), value / 2),
            )
            result = {
                "status": "ok",
                "finite": True,
                "mean_return": value,
                "feedback": {
                    "episodes": [
                        {"seed": seed, "return": value, "length": 2} for seed in seeds
                    ]
                },
            }
        write_json(output / "result.json", result)
        return result

    monkeypatch.setattr(cli, "_evaluate", evaluate)
    return harness


def _call(name: str, call_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function_call",
        "name": name,
        "call_id": call_id,
        "arguments": json.dumps(arguments),
    }


def _proposal(marker: str) -> dict[str, Any]:
    return {
        "title": "Test a policy",
        "hypothesis": "Changed gain improves returns",
        "planned_change": "Change the gain",
        "helper": None,
        # The evaluator is mocked; importing this source in the host must fail.
        "source": f'raise AssertionError("policy executed in host")\n# {marker}\n',
    }


def _response(*output: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": "completed",
        "output": list(output),
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }


def _final_message() -> dict[str, Any]:
    return {
        "type": "message",
        "role": "assistant",
        "content": [{"type": "output_text", "text": "Keep the measured best policy."}],
    }


def _model(
    monkeypatch: pytest.MonkeyPatch, harness: Harness, responses: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    requests = []

    def complete(request: dict[str, Any]) -> dict[str, Any]:
        harness.timeline.append("model")
        requests.append(copy.deepcopy(request))
        return responses.pop(0)

    monkeypatch.setattr(agent, "complete", complete)
    return requests


def _events(directory: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in (directory / "events.jsonl").read_text().splitlines()
    ]


def test_codex_host_dispatches_without_api_and_logs_native_artifacts(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    from experiments.agent_policy_workspace import codex_provider

    requests = []

    def codex_complete(request: dict[str, Any], directory: Path) -> dict[str, Any]:
        requests.append(copy.deepcopy(request))
        harness.timeline.append("codex")
        assert directory == harness.root / "agent_session/codex/turn_000"
        directory.mkdir(parents=True)
        (directory / "events.jsonl").write_text('{"type":"turn.completed"}\n')
        return {
            **_response(_final_message()),
            "adapter": "codex_cli",
            "native_artifacts": str(directory),
        }

    def unexpected_api(_request: dict[str, Any]) -> dict[str, Any]:
        pytest.fail("Codex host invoked the API provider")

    monkeypatch.setattr(codex_provider, "complete", codex_complete)
    monkeypatch.setattr(agent, "complete", unexpected_api)
    directory = agent.run(agent.Config(harness.root, host="codex", model=None))
    assert len(requests) == 1 and requests[0]["model"] == "gpt-6.2-sol"
    events = _events(directory)
    response_event = next(e for e in events if e["event"] == "model_response")
    response = read_json(directory / response_event["payload_file"])
    assert response["adapter"] == "codex_cli"
    assert Path(response["native_artifacts"]) == directory / "codex/turn_000"
    assert (Path(response["native_artifacts"]) / "events.jsonl").exists()
    assert harness.timeline == [
        "development",
        "codex",
        "final-development",
        "final-heldout",
    ]
    assert read_json(directory / "summary.json")["status"] == "completed"


def test_agent_logs_replay_failures_tools_and_finalizes_after_search(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    reasoning = {
        "id": "reasoning-1",
        "type": "reasoning",
        "encrypted_content": "opaque-replay",
        "summary": [
            {"type": "summary_text", "text": "Inspect measured behavior first."}
        ],
    }
    requests = _model(
        monkeypatch,
        harness,
        [
            _response(
                reasoning,
                _call(
                    "inspect_trajectory",
                    "trace-1",
                    {
                        "candidate": None,
                        "episode": 0,
                        "start": 0,
                        "stop": 2,
                        "fields": ["obs", "reward"],
                    },
                ),
            ),
            _response(_call("evaluate_policy", "policy-failed", _proposal("failed"))),
            _response(
                _call("evaluate_policy", "policy-improved", _proposal("improved"))
            ),
            _response(_final_message()),
        ],
    )
    directory = agent.run(agent.Config(harness.root, proposals=2, max_model_calls=4))
    assert len(requests) == 4
    assert all(request["reasoning"]["summary"] == "auto" for request in requests)
    assert requests[1]["input"][1] == reasoning
    assert requests[-1]["tool_choice"] == "none"
    assert all(str(HELDOUT_RETURN) not in json.dumps(request) for request in requests)
    assert all("270001" not in json.dumps(request) for request in requests)
    assert harness.timeline[-2:] == ["final-development", "final-heldout"]
    events = _events(directory)
    for call_id in ("trace-1", "policy-failed", "policy-improved", "host-final"):
        call = next(
            e for e in events if e["event"] == "tool_call" and e["call_id"] == call_id
        )
        result = next(
            e for e in events if e["event"] == "tool_result" and e["call_id"] == call_id
        )
        assert call["sequence"] < result["sequence"]
    failed = next(
        e
        for e in events
        if e.get("call_id") == "policy-failed" and e["event"] == "tool_result"
    )
    assert failed["result"]["evaluation"]["status"] == "error"
    assert failed["result"]["proposals_remaining"] == 1
    summary = read_json(directory / "summary.json")
    assert summary["proposals_used"] == 2
    assert summary["best"]["mean_return"] == 3.0
    assert summary["heldout_mean_return"] == HELDOUT_RETURN
    assert summary["usage"] == {"input_tokens": 40, "output_tokens": 20}
    assert "# improved" in (harness.root / "policy.py").read_text()
    assert Ledger(harness.root).current_attempt() is None
    model_events = [e for e in events if e["event"] == "model_request"]
    assert [read_json(directory / e["payload_file"]) for e in model_events] == requests


def test_agent_enforces_proposal_budget_even_if_model_requests_more(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    _model(
        monkeypatch,
        harness,
        [
            _response(_call("evaluate_policy", "first", _proposal("improved"))),
            _response(_call("evaluate_policy", "extra", _proposal("failed"))),
            _response(_final_message()),
        ],
    )
    directory = agent.run(agent.Config(harness.root, proposals=1, max_model_calls=3))
    assert harness.timeline.count("development") == 2  # Baseline and one proposal.
    extra = next(
        e
        for e in _events(directory)
        if e.get("call_id") == "extra" and e["event"] == "tool_result"
    )
    assert "budget exhausted" in extra["result"]["error"]
    assert read_json(directory / "summary.json")["proposals_used"] == 1


def test_agent_stops_at_model_call_cap(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    requests = _model(
        monkeypatch,
        harness,
        [
            _response(
                _call(
                    "inspect_trajectory",
                    "trace-1",
                    {
                        "candidate": None,
                        "episode": 0,
                        "start": 0,
                        "stop": 2,
                        "fields": ["reward"],
                    },
                )
            ),
        ],
    )
    directory = agent.run(agent.Config(harness.root, max_model_calls=1))
    assert len(requests) == 1
    finished = next(e for e in _events(directory) if e["event"] == "search_finished")
    assert finished["reason"] == "model_call_budget"
    assert read_json(directory / "summary.json")["model_calls"] == 1


@pytest.mark.parametrize("stage", ["model", "tracking"])
def test_provider_and_tracking_errors_stop_and_remain_in_transcript(
    harness: Harness, monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    if stage == "model":

        def fail_model(_request: dict[str, Any]) -> dict[str, Any]:
            raise RuntimeError("provider authentication failed")

        monkeypatch.setattr(agent, "complete", fail_model)
        expected = "provider authentication"
    else:
        _model(
            monkeypatch,
            harness,
            [
                _response(_call("evaluate_policy", "candidate", _proposal("improved"))),
            ],
        )
        original = cli._evaluate

        def fail_tracking(*args: Any, **kwargs: Any) -> dict[str, Any]:
            if "# improved" in args[1].read_text():
                raise ValueError("W&B authorization failed")
            return original(*args, **kwargs)

        monkeypatch.setattr(cli, "_evaluate", fail_tracking)
        expected = "W&B authorization"
    with pytest.raises((ValueError, RuntimeError), match=expected):
        agent.run(agent.Config(harness.root))
    events = _events(harness.root / "agent_session")
    assert events[-1]["event"] == "error"
    assert expected in events[-1]["error"]
    assert not (harness.root / ".policy_eval/final.json").exists()
    assert not any(e.get("call_id") == "host-final" for e in events)


def test_worker_command_is_explicit_and_does_not_inherit_secrets_or_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    command = ["remote-worker", "--fixed-container", "theo_policy_worker"]
    monkeypatch.setenv("PLASMAX_POLICY_WORKER_COMMAND", json.dumps(command))
    monkeypatch.setenv("OPENAI_API_KEY", "test-secret")
    monkeypatch.setenv("OTHER_API_TOKEN", "test-token")
    monkeypatch.setenv("KEEP_THIS_FLAG", "yes")
    write_json(tmp_path / "request.json", {})

    def run(arguments: list[str], **kwargs: Any) -> SimpleNamespace:
        assert arguments == [*command, "--request", str(tmp_path / "request.json")]
        assert kwargs["timeout"] == 23.0
        assert kwargs["check"] is False
        assert "OPENAI_API_KEY" not in kwargs["env"]
        assert "OTHER_API_TOKEN" not in kwargs["env"]
        assert "PLASMAX_POLICY_WORKER_COMMAND" not in kwargs["env"]
        assert kwargs["env"]["KEEP_THIS_FLAG"] == "yes"
        write_json(tmp_path / "worker_result.json", {"status": "ok", "finite": True})
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(subprocess, "run", run)
    report = runtime._subprocess(tmp_path, 23.0)
    assert report["status"] == "ok" and report["finite"]
    assert report["evaluation_seconds"] >= 0


@pytest.mark.parametrize("command", ['"shell command"', "[]", '["worker", 3]'])
def test_invalid_worker_command_never_starts_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, command: str
) -> None:
    monkeypatch.setenv("PLASMAX_POLICY_WORKER_COMMAND", command)

    def unexpected(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail("invalid worker command started a subprocess")

    monkeypatch.setattr(subprocess, "run", unexpected)
    with pytest.raises(ValueError, match="JSON string array"):
        runtime._subprocess(tmp_path, 23.0)


@pytest.fixture
def deadline_clock(harness: Harness, monkeypatch: pytest.MonkeyPatch) -> list[float]:
    clock = [1000.0]
    config_path = harness.root / ".policy_eval/config.json"
    config = read_json(config_path)
    config.update(created_at=clock[0], lifetime_seconds=1000.0)
    config_path.write_text(json.dumps(config))
    fake_time = SimpleNamespace(time=lambda: clock[0], monotonic=lambda: clock[0])
    monkeypatch.setattr(cli, "time", fake_time)
    monkeypatch.setattr(agent, "time", fake_time)
    return clock


def test_agent_reserves_final_banks_before_starting_model(
    harness: Harness,
    monkeypatch: pytest.MonkeyPatch,
    deadline_clock: list[float],
) -> None:
    original = cli._evaluate

    def evaluate(*args: Any, **kwargs: Any) -> dict[str, Any]:
        deadline_clock[0] += 200.0
        return original(*args, **kwargs)

    monkeypatch.setattr(cli, "_evaluate", evaluate)
    requests = _model(monkeypatch, harness, [])
    directory = agent.run(agent.Config(harness.root))
    assert requests == []
    assert harness.timeline == ["development", "final-development", "final-heldout"]
    summary = read_json(directory / "summary.json")
    assert summary["status"] == "completed"
    assert summary["proposals_used"] == 0
    assert deadline_clock[0] < 2000.0


def test_agent_rechecks_reserve_after_model_before_editing_or_evaluating(
    harness: Harness,
    monkeypatch: pytest.MonkeyPatch,
    deadline_clock: list[float],
) -> None:
    original = cli._evaluate
    starting_source = (harness.root / "policy.py").read_bytes()

    def evaluate(*args: Any, **kwargs: Any) -> dict[str, Any]:
        deadline_clock[0] += 100.0
        return original(*args, **kwargs)

    def complete(_request: dict[str, Any]) -> dict[str, Any]:
        harness.timeline.append("model")
        deadline_clock[0] += 500.0
        return _response(_call("evaluate_policy", "too-late", _proposal("improved")))

    monkeypatch.setattr(cli, "_evaluate", evaluate)
    monkeypatch.setattr(agent, "complete", complete)
    directory = agent.run(agent.Config(harness.root))
    assert harness.timeline == [
        "development",
        "model",
        "final-development",
        "final-heldout",
    ]
    summary = read_json(directory / "summary.json")
    assert summary["status"] == "completed"
    assert summary["model_calls"] == 1
    assert summary["proposals_used"] == 0
    assert (harness.root / "policy.py").read_bytes() == starting_source
    assert deadline_clock[0] < 2000.0


def test_agent_final_timeout_saves_partial_summary_and_restores_best(
    harness: Harness,
    monkeypatch: pytest.MonkeyPatch,
    deadline_clock: list[float],
) -> None:
    original = cli._evaluate
    starting_source = (harness.root / "policy.py").read_bytes()

    def evaluate(*args: Any, **kwargs: Any) -> dict[str, Any]:
        if kwargs.get("final"):
            harness.timeline.append(args[4])
            deadline_clock[0] = 2000.0
            return {"status": "timeout", "finite": False, "error": "deadline"}
        deadline_clock[0] += 10.0
        return original(*args, **kwargs)

    monkeypatch.setattr(cli, "_evaluate", evaluate)
    _model(
        monkeypatch,
        harness,
        [
            _response(_call("evaluate_policy", "rejected", _proposal("failed"))),
            _response(_final_message()),
        ],
    )
    directory = agent.run(agent.Config(harness.root))
    summary = read_json(directory / "summary.json")
    assert summary["status"] == "time_limit"
    assert summary["best"]["mean_return"] == 1.0
    assert summary.get("heldout_mean_return") is None
    assert "final-heldout" not in harness.timeline
    assert not (harness.root / ".policy_eval/final.json").exists()
    assert (harness.root / "policy.py").read_bytes() == starting_source


def test_proposal_timeout_leaves_finalization_reserve(
    harness: Harness,
    monkeypatch: pytest.MonkeyPatch,
    deadline_clock: list[float],
) -> None:
    original = cli._evaluate
    proposal_timeouts = []

    def evaluate(*args: Any, **kwargs: Any) -> dict[str, Any]:
        if "# improved" in args[1].read_text() and not kwargs.get("final"):
            proposal_timeouts.append(kwargs["timeout_seconds"])
        deadline_clock[0] += 100.0
        return original(*args, **kwargs)

    monkeypatch.setattr(cli, "_evaluate", evaluate)
    _model(
        monkeypatch,
        harness,
        [
            _response(_call("evaluate_policy", "candidate", _proposal("improved"))),
            _response(_final_message()),
        ],
    )
    directory = agent.run(agent.Config(harness.root))
    # After a 100s baseline, 900s remain; two 125s banks plus 120s are reserved.
    assert proposal_timeouts == [pytest.approx(530.0)]
    assert read_json(directory / "summary.json")["status"] == "completed"
