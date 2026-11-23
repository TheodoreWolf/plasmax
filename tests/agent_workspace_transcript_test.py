"""Model payloads, visible summaries, and tool evidence survive logging."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.agent_policy_workspace.transcript import Transcript


def test_transcript_preserves_model_payloads_and_tool_evidence(tmp_path: Path) -> None:
    log = Transcript(tmp_path / "agent")
    request = {
        "model": "test-model",
        "instructions": "Improve the controller.\nInspect measurements first.",
        "input": [{"role": "user", "content": "Score and inspect the baseline."}],
        "reasoning": {"summary": "auto"},
    }
    response = {
        "id": "response-1",
        "output": [
            {
                "type": "reasoning",
                "encrypted_content": "opaque-ciphertext",
                "summary": [{"type": "summary_text", "text": "Inspect the trace."}],
            },
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Testing lower gain."}],
            },
            {
                "type": "function_call",
                "call_id": "call-1",
                "name": "score",
                "arguments": "{}",
            },
        ],
        "usage": {"input_tokens": 40, "output_tokens": 20},
    }
    log.record("model_request", request=request)
    log.record("model_response", response=response)
    prefix = log.events.read_bytes()
    log.record("tool_call", call_id="call-1", name="score", arguments={})
    log.record("tool_result", call_id="call-1", name="score", result={"return": 2.5})
    log.record("error", error="example provider timeout")
    entries = [json.loads(line) for line in log.events.read_text().splitlines()]
    assert [entry["sequence"] for entry in entries] == [1, 2, 3, 4, 5]
    assert all(entry["time"] for entry in entries)
    assert log.events.read_bytes().startswith(prefix)
    assert (
        json.loads((log.directory / entries[0]["payload_file"]).read_text()) == request
    )
    assert (
        json.loads((log.directory / entries[1]["payload_file"]).read_text()) == response
    )
    assert entries[3]["result"] == {"return": 2.5}
    readable = log.markdown.read_text()
    assert "Improve the controller" in readable
    assert "    Inspect measurements first.\n" in readable
    assert "Reasoning summary" in readable and "Inspect the trace." in readable
    assert "Testing lower gain." in readable and "call-1" in readable
    assert "example provider timeout" in readable
    assert "opaque-ciphertext" not in readable


def test_replayed_opaque_reasoning_is_not_displayed_as_text(tmp_path: Path) -> None:
    log = Transcript(tmp_path / "agent")
    request = {"input": [{"type": "reasoning", "encrypted_content": "opaque-replay"}]}
    log.record("model_request", request=request)
    log.record("model_response", response={"output": [{"type": "reasoning"}]})
    assert "opaque-replay" not in log.markdown.read_text()
    assert "No reasoning summary returned" in log.markdown.read_text()
    assert "opaque-replay" in (log.directory / "000001-model_request.json").read_text()


def test_existing_transcript_is_never_overwritten(tmp_path: Path) -> None:
    log = Transcript(tmp_path / "agent")
    log.record("error", error="preserve this evidence")
    original = log.events.read_bytes()
    with pytest.raises(FileExistsError):
        Transcript(log.directory)
    assert log.events.read_bytes() == original
    with pytest.raises(ValueError, match="lowercase"):
        log.record("../unsafe", response={})
    assert log.events.read_bytes() == original
