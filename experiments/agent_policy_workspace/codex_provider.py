"""Use a saved ChatGPT login for a tool-free, structured Codex planning turn.

The normalized return value is an internal adapter envelope, not an API response.
Native CLI events remain authoritative. Codex exec cannot enforce the caller's
max_output_tokens; the subprocess timeout and parent budgets still apply.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any
from uuid import uuid4

from experiments.agent_policy_search.artifacts import plain, write_json

__all__ = ["complete"]

_DISABLED_FEATURES = (
    "shell_tool",
    "apps",
    "plugins",
    "multi_agent",
    "browser_use",
    "computer_use",
    "image_generation",
    "view_image",
    "workspace_dependencies",
    "artifact",
)


def complete(request: dict[str, Any], directory: Path) -> dict[str, Any]:
    """Record one Codex CLI turn without exposing credentials to artifacts."""
    directory = directory.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    actions = (
        ["finish"]
        if request.get("tool_choice") == "none"
        else ["evaluate_policy", "inspect_trajectory", "finish"]
    )
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "tool": {"type": "string", "enum": actions},
            "arguments": {"type": "string"},
            "message": {"type": "string"},
        },
        "required": ["tool", "arguments", "message"],
    }
    write_json(directory / "output_schema.json", schema)
    prompt = (
        "Choose exactly one next action for the supplied policy-search conversation.\n"
        "Do not execute tools, inspect files, or perform evaluation yourself. The "
        "trusted parent executes the action and returns its measured result.\n"
        "Use the supplied schemas and obey the remaining proposal and call budgets.\n"
        "Return only the required JSON object: tool names the next action; arguments "
        "is a JSON object encoded as a string; message briefly explains the decision.\n"
        'For finish, arguments must be "{}". If tool_choice is none, finish now.\n\n'
        + json.dumps(plain(request), indent=2, ensure_ascii=False, allow_nan=False)
        + "\n"
    )
    (directory / "prompt.txt").write_text(prompt, encoding="utf-8")
    command = [
        "codex",
        "exec",
        "--cd",
        str(directory),
        "--skip-git-repo-check",
        "--sandbox",
        "read-only",
        "--ignore-user-config",
        "--ephemeral",
        "--json",
        "--output-schema",
        str(directory / "output_schema.json"),
        "--output-last-message",
        str(directory / "final.json"),
        "--model",
        request["model"],
        "-c",
        f"model_reasoning_effort={json.dumps(request['reasoning']['effort'])}",
        "-c",
        'model_reasoning_summary="detailed"',
        "-c",
        'web_search="disabled"',
    ]
    for feature in _DISABLED_FEATURES:
        command.extend(["-c", f"features.{feature}=false"])
    command.append("-")
    write_json(directory / "command.json", command)
    environment = {
        name: value
        for name, value in os.environ.items()
        if not name.endswith(("_API_KEY", "_API_TOKEN"))
    }
    with (
        (directory / "events.jsonl").open("x", encoding="utf-8") as stdout,
        (directory / "stderr.txt").open("x", encoding="utf-8") as stderr,
    ):
        process = subprocess.run(
            command,
            input=prompt,
            text=True,
            stdout=stdout,
            stderr=stderr,
            cwd=directory,
            env=environment,
            timeout=request["timeout"],
            check=False,
        )
    if process.returncode:
        raise RuntimeError(
            f"Codex exited {process.returncode}; see {directory / 'stderr.txt'}"
        )
    events = [
        json.loads(line)
        for line in (directory / "events.jsonl").read_text().splitlines()
        if line.strip()
    ]
    completed = [event for event in events if event.get("type") == "turn.completed"]
    if len(completed) != 1 or any(
        event.get("type") == "turn.failed" for event in events
    ):
        raise RuntimeError(
            f"Codex did not complete one turn; see {directory / 'events.jsonl'}"
        )
    try:
        action = json.loads((directory / "final.json").read_text())
        if not isinstance(action, dict) or set(action) != {
            "tool",
            "arguments",
            "message",
        }:
            raise ValueError("expected tool, arguments and message")
        if any(not isinstance(value, str) for value in action.values()):
            raise ValueError("all action fields must be strings")
        arguments = json.loads(action["arguments"])
        if action["tool"] not in actions or not isinstance(arguments, dict):
            raise ValueError("invalid tool or non-object arguments")
        if action["tool"] == "finish" and arguments:
            raise ValueError("finish arguments must be empty")
    except (ValueError, OSError) as error:
        raise RuntimeError(
            f"Invalid Codex action in {directory / 'final.json'}: {error}"
        ) from error
    output = [
        {
            "type": "reasoning",
            "summary": [{"type": "summary_text", "text": event["item"]["text"]}],
        }
        for event in events
        if event.get("type") == "item.completed"
        and event.get("item", {}).get("type") == "reasoning"
        and isinstance(event["item"].get("text"), str)
    ]
    output.append(
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": action["message"]}],
        }
    )
    if action["tool"] != "finish":
        output.append(
            {
                "type": "function_call",
                "call_id": f"codex-{uuid4().hex}",
                "name": action["tool"],
                "arguments": action["arguments"],
            }
        )
    return {
        "status": "completed",
        "output": output,
        "usage": completed[0].get("usage", {}),
        "adapter": "codex_cli",
        "native_artifacts": str(directory),
    }
