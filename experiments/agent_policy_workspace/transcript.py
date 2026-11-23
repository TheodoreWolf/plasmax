"""Persist visible agent activity and the exact serialized model exchange."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from experiments.agent_policy_search.artifacts import plain, timestamp, write_json

__all__ = ["Transcript"]


def _display(value: Any) -> Any:
    """Keep opaque reasoning out of the readable transcript, including replay."""
    if isinstance(value, dict):
        return {
            key: "[opaque reasoning; see JSON]"
            if key == "encrypted_content"
            else _display(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_display(item) for item in value]
    return value


def _block(value: Any) -> str:
    text = (
        value
        if isinstance(value, str)
        else json.dumps(_display(value), indent=2, ensure_ascii=False, allow_nan=False)
    )
    return "\n".join(f"    {line}" for line in text.splitlines()) + "\n"


class Transcript:
    """An append-only transcript owned by one sequential agent loop.

    Only supplied data is logged; callers must never pass credentials or headers.
    Each run gets a new directory so a restart cannot overwrite old evidence.
    """

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=False)
        self.events = directory / "events.jsonl"
        self.markdown = directory / "transcript.md"
        self.events.touch(exist_ok=False)
        self.markdown.write_text(
            "# Agent transcript\n\n"
            "Reasoning summaries are model-provided summaries, not raw "
            "internal reasoning. Exact model payloads are saved separately.\n",
            encoding="utf-8",
        )
        self.sequence = 0

    def record(self, event: str, **data: Any) -> None:
        """Append a timestamped event; preserve full model payloads exactly once."""
        if re.fullmatch(r"[a-z][a-z_]*", event) is None:
            raise ValueError("event must contain lowercase letters and underscores")
        values = plain(data)
        # Validate before creating any artifact, including for ordinary tool events.
        json.dumps(values, allow_nan=False)
        sequence = self.sequence + 1
        entry = {"sequence": sequence, "time": timestamp(), "event": event}
        sections = [f"\n## {sequence:04d} · {event} · {entry['time']}\n"]
        payload_key = {"model_request": "request", "model_response": "response"}.get(
            event
        )
        if payload_key is not None:
            payload = values.pop(payload_key)
            filename = f"{sequence:06d}-{event}.json"
            write_json(self.directory / filename, payload)
            entry["payload_file"] = filename
            sections.append(f"[Exact JSON]({filename})\n")
            if event == "model_response":
                if payload.get("adapter") == "codex_cli":
                    sections.append(
                        "This JSON is the harness's normalized CLI response. "
                        f"[Native CLI artifacts]({payload['native_artifacts']}) "
                        "contain the supplied prompt and emitted events.\n"
                    )
                for item in payload.get("output", []):
                    kind = item.get("type")
                    if kind == "reasoning":
                        sections.append("### Reasoning summary\n")
                        summaries = item.get("summary") or []
                        sections.append(
                            _block(
                                "\n".join(part.get("text", "") for part in summaries)
                                or "No reasoning summary returned."
                            )
                        )
                    elif kind == "message":
                        sections.append("### Assistant response\n")
                        for part in item.get("content", []):
                            sections.append(
                                _block(part.get("text", part.get("refusal", part)))
                            )
                    else:
                        sections.append(_block(item))
            else:
                sections.append(
                    _block(
                        {
                            key: value
                            for key, value in payload.items()
                            if key not in {"instructions", "input"}
                        }
                    )
                )
                if payload.get("instructions"):
                    sections.extend(
                        ["### Instructions\n", _block(payload["instructions"])]
                    )
                inputs = payload.get("input", [])
                for item in inputs if isinstance(inputs, list) else [inputs]:
                    if isinstance(item, dict) and isinstance(item.get("content"), str):
                        sections.extend(
                            [
                                f"### {item.get('role', 'Input')}\n",
                                _block(item["content"]),
                            ]
                        )
                    else:
                        sections.append(_block(item))
        entry.update(values)
        if values:
            sections.append(_block(values))
        with self.events.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(entry, ensure_ascii=False, allow_nan=False) + "\n")
        with self.markdown.open("a", encoding="utf-8") as stream:
            stream.write("\n".join(sections))
        self.sequence = sequence
