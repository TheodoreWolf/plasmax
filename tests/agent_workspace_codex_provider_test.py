"""Codex turns are constrained and retain their native transcript."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from experiments.agent_policy_search.artifacts import read_json
from experiments.agent_policy_workspace import codex_provider


@pytest.fixture
def model_request() -> dict[str, Any]:
    return {
        "model": "gpt-5.4-mini",
        "instructions": "Improve the controller.",
        "input": [{"role": "user", "content": "There is one proposal left."}],
        "tools": [{"name": "inspect_trajectory"}],
        "tool_choice": "auto",
        "reasoning": {"effort": "medium", "summary": "auto"},
        "timeout": 30.0,
        "max_output_tokens": 8192,
    }


def _mock_run(
    monkeypatch: pytest.MonkeyPatch,
    *,
    action: dict[str, Any] | None = None,
    events: list[dict[str, Any]] | None = None,
    returncode: int = 0,
) -> list[tuple[list[str], dict[str, Any]]]:
    calls = []
    if action is None:
        action = {
            "tool": "inspect_trajectory",
            "arguments": '{"episode": 0}',
            "message": "Inspect the baseline.",
        }
    if events is None:
        events = [
            {
                "type": "item.completed",
                "item": {"type": "reasoning", "text": "Check measured behavior."},
            },
            {
                "type": "turn.completed",
                "usage": {
                    "input_tokens": 100,
                    "cached_input_tokens": 20,
                    "output_tokens": 30,
                },
            },
        ]

    def run(command: list[str], **kwargs: Any) -> SimpleNamespace:
        calls.append((command, kwargs))
        kwargs["stdout"].write("".join(json.dumps(event) + "\n" for event in events))
        kwargs["stderr"].write("retained CLI diagnostics\n")
        output = Path(command[command.index("--output-last-message") + 1])
        output.write_text(json.dumps(action))
        return SimpleNamespace(returncode=returncode)

    monkeypatch.setattr(subprocess, "run", run)
    return calls


def test_codex_native_capture_summary_and_normalized_tool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, model_request: dict[str, Any]
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "secret-openai")
    monkeypatch.setenv("CODEX_API_KEY", "secret-codex")
    monkeypatch.setenv("OTHER_API_TOKEN", "secret-token")
    monkeypatch.setenv("KEEP_VALUE", "kept")
    calls = _mock_run(monkeypatch)
    directory = tmp_path / "turn"
    response = codex_provider.complete(model_request, directory)
    command, kwargs = calls[0]
    assert command[:2] == ["codex", "exec"] and command[-1] == "-"
    assert command[command.index("--sandbox") + 1] == "read-only"
    for flag in (
        "--ignore-user-config",
        "--ephemeral",
        "--json",
        "--skip-git-repo-check",
    ):
        assert flag in command
    assert not any("bypass" in part or "dangerously" in part for part in command)
    for feature in codex_provider._DISABLED_FEATURES:
        assert f"features.{feature}=false" in command
    assert 'web_search="disabled"' in command
    assert 'model_reasoning_summary="detailed"' in command
    assert kwargs["cwd"] == directory and kwargs["timeout"] == 30.0
    assert kwargs["input"] == (directory / "prompt.txt").read_text()
    assert kwargs["text"] is True and kwargs["check"] is False
    assert kwargs["env"]["KEEP_VALUE"] == "kept"
    assert not any(
        name in kwargs["env"]
        for name in ("OPENAI_API_KEY", "CODEX_API_KEY", "OTHER_API_TOKEN")
    )
    assert read_json(directory / "command.json") == command
    assert "There is one proposal left" in (directory / "prompt.txt").read_text()
    assert (directory / "stderr.txt").read_text() == "retained CLI diagnostics\n"
    assert all("secret-" not in path.read_text() for path in directory.iterdir())
    assert response["status"] == "completed" and response["adapter"] == "codex_cli"
    assert response["native_artifacts"] == str(directory)
    assert response["output"][0]["summary"][0]["text"] == "Check measured behavior."
    assert response["output"][1]["content"][0]["text"] == "Inspect the baseline."
    tool = response["output"][2]
    assert tool["type"] == "function_call" and tool["name"] == "inspect_trajectory"
    assert tool["call_id"].startswith("codex-")
    assert json.loads(tool["arguments"]) == {"episode": 0}
    assert response["usage"] == {
        "input_tokens": 100,
        "cached_input_tokens": 20,
        "output_tokens": 30,
    }
    with pytest.raises(FileExistsError):
        codex_provider.complete(model_request, directory)


def test_finish_only_turn_has_no_normalized_tool_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, model_request: dict[str, Any]
) -> None:
    model_request["tool_choice"] = "none"
    _mock_run(
        monkeypatch,
        action={"tool": "finish", "arguments": "{}", "message": "Keep best."},
    )
    response = codex_provider.complete(model_request, tmp_path / "turn")
    schema = read_json(tmp_path / "turn/output_schema.json")
    assert schema["properties"]["tool"]["enum"] == ["finish"]
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {"tool", "arguments", "message"}
    assert not any(item["type"] == "function_call" for item in response["output"])


@pytest.mark.parametrize(
    "failure",
    ["exit", "incomplete", "malformed", "finish_arguments", "tool_disallowed", "extra"],
)
def test_codex_failures_preserve_native_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    model_request: dict[str, Any],
    failure: str,
) -> None:
    action = {"tool": "finish", "arguments": "{}", "message": "Done."}
    kwargs: dict[str, Any] = {}
    if failure == "exit":
        kwargs["returncode"] = 1
    elif failure == "incomplete":
        kwargs["events"] = [{"type": "turn.failed", "error": "failed"}]
    elif failure == "malformed":
        action["arguments"] = "not JSON"
    elif failure == "finish_arguments":
        action["arguments"] = '{"unexpected": true}'
    elif failure == "tool_disallowed":
        model_request["tool_choice"] = "none"
        action["tool"] = "evaluate_policy"
    else:
        action["extra"] = "not allowed"
    _mock_run(monkeypatch, action=action, **kwargs)
    directory = tmp_path / "turn"
    with pytest.raises(RuntimeError, match="Codex"):
        codex_provider.complete(model_request, directory)
    assert {path.name for path in directory.iterdir()} == {
        "prompt.txt",
        "command.json",
        "output_schema.json",
        "events.jsonl",
        "stderr.txt",
        "final.json",
    }


def test_codex_timeout_preserves_partial_native_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, model_request: dict[str, Any]
) -> None:
    def timeout(command: list[str], **kwargs: Any) -> None:
        kwargs["stdout"].write('{"type":"turn.started"}\n')
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(subprocess, "run", timeout)
    with pytest.raises(subprocess.TimeoutExpired):
        codex_provider.complete(model_request, tmp_path / "turn")
    assert (tmp_path / "turn/events.jsonl").read_text() == '{"type":"turn.started"}\n'
