"""Explicit model context and provider-neutral policy proposals."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from experiments.agent_policy_search.config import ModelConfig

__all__ = ["Proposal", "build_request", "complete", "parse_proposal"]


@dataclass(frozen=True)
class Proposal:
    """A complete candidate policy and a brief explanation of its change."""

    source: str
    description: str


_SYSTEM_PROMPT = """You are designing a policy for a plasmax control task.
Improve the incumbent policy using the task/interface specification and the
provided evaluation feedback. Diagnose the compact reports before proposing a
change; returns rank candidates, while diagnostics can explain their behavior.
The latest attempt may be worse than the incumbent or may have failed to execute.
Use its feedback to avoid repeating an ineffective change.

Write a complete, readable Python module defining act(obs, rng) -> action.
The function must be pure, stateless, and compatible with JAX jit, vmap, and
lax.scan. Use jax/jax.numpy operations for array computation and JAX control flow
where conditions depend on observations. Return a finite JAX array with the exact
action shape and dtype specified in the task and respect its action bounds.
The observation and action metadata describe the actual policy interface,
including any normalization; do not mistake normalized values for physical units.
The policy may use only obs, rng, and constants defined in its module. Do not use
hidden simulator state, global mutable state, files, network access, subprocesses,
or additional evaluation data. Import only jax and jax.numpy if needed.

Each request supplies the complete available context: the task, the incumbent
source and feedback, and at most the latest attempt. There is no earlier chat or
held-out evaluation information to retrieve. Do not invent unreported results.

Return only a JSON object with exactly these fields:
{"source": "complete Python module", "description": "short explanation of change"}
"""


def build_request(
    config: ModelConfig,
    *,
    task: dict[str, Any],
    incumbent_source: str,
    incumbent_feedback: dict[str, Any],
    latest: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build the full, loggable LiteLLM request without credentials or history."""
    context = {
        "task": task,
        "incumbent": {
            "source": incumbent_source,
            "feedback": incumbent_feedback,
        },
        "latest_attempt": latest,
    }
    request: dict[str, Any] = {
        "model": config.model,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(context, indent=2, allow_nan=False),
            },
        ],
        "max_tokens": config.max_tokens,
        "timeout": config.timeout_seconds,
    }
    if config.temperature is not None:
        request["temperature"] = config.temperature
    if config.api_base is not None:
        request["api_base"] = config.api_base
    return request


def complete(request: dict[str, Any]) -> dict[str, Any]:
    """Call the configured provider and retain its full JSON response.

    Provider initialization, authentication, and request errors propagate to the
    caller. Importing this module alone never imports or initializes LiteLLM.
    """
    from litellm import completion

    response = completion(**request)
    payload = response.model_dump(mode="json")
    if not isinstance(payload, dict):
        raise ValueError("model response must serialize to a JSON object")
    return payload


def parse_proposal(response: dict[str, Any]) -> Proposal:
    """Read a strict proposal object from ordinary or fenced JSON content."""
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("model response has no message content") from exc
    if not isinstance(content, str) or not content.strip():
        raise ValueError("model response message content must be nonempty text")
    content = content.strip()
    fence = re.fullmatch(r"```(?:json)?\s*\n?(.*?)\s*```", content, re.DOTALL)
    if fence is not None:
        content = fence.group(1)
    try:
        proposal = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError("model proposal must be a JSON object") from exc
    if not isinstance(proposal, dict) or set(proposal) != {"source", "description"}:
        raise ValueError("model proposal must contain only source and description")
    for field in ("source", "description"):
        if not isinstance(proposal[field], str) or not proposal[field].strip():
            raise ValueError(f"model proposal {field} must be nonempty text")
    return Proposal(source=proposal["source"], description=proposal["description"])
