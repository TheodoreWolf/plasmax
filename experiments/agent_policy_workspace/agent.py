"""A bounded agent using Responses or Codex CLI and existing workspace tools."""

from __future__ import annotations

import json
import math
import os
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from experiments.agent_policy_search.artifacts import plain, write_json
from experiments.agent_policy_search.run import eligible
from experiments.agent_policy_workspace import cli
from experiments.agent_policy_workspace.ledger import (
    _replace_bytes,
    snapshot_bundle,
)
from experiments.agent_policy_workspace.transcript import Transcript


@dataclass(frozen=True)
class Config:
    workspace: Path
    host: Literal["responses", "codex"] = "responses"
    model: str | None = None
    proposals: int = 2
    max_model_calls: int = 8
    max_output_tokens: int = 8192
    reasoning_effort: Literal["low", "medium", "high"] = "medium"
    api_timeout_seconds: float = 180.0


class Proposal(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    title: str = Field(min_length=1, max_length=200)
    hypothesis: str = Field(min_length=1, max_length=4000)
    planned_change: str = Field(min_length=1, max_length=4000)
    source: str = Field(min_length=1, max_length=60000)
    helper: str | None = Field(max_length=60000)


class TraceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    candidate: str | None = Field(pattern=r"^[0-9a-f]{64}$")
    episode: int = Field(ge=0)
    start: int = Field(ge=0)
    stop: int = Field(ge=1)
    fields: list[str] = Field(min_length=1, max_length=12)


class ToolInputError(ValueError):
    """An agent can correct this request without retrying infrastructure failures."""


def _tool(name: str, description: str, schema: type[BaseModel]) -> dict[str, Any]:
    return {
        "type": "function",
        "name": name,
        "description": description,
        "parameters": schema.model_json_schema(),
        "strict": True,
    }


TOOLS = [
    _tool(
        "evaluate_policy",
        "Submit a complete policy.py and optional custom_primitives.py, together "
        "with your hypothesis. The host archives, validates, scores, and records "
        "selection on fixed development seeds. Every submission consumes one "
        "proposal, including failures. Set helper=null to omit it.",
        Proposal,
    ),
    _tool(
        "inspect_trajectory",
        "Read an observable trajectory window from an officially scored policy. "
        "candidate=null selects the best. Choose at most 200 steps within the "
        "configured horizon; fields include obs, requested_action, reward, "
        "next_obs, time_s, and trace_NAME diagnostics.",
        TraceRequest,
    ),
]

INSTRUCTIONS = """You are improving a plasmax policy with a small evaluation budget.
Read the supplied task, interface, reward, policy contract and baseline. Use
inspect_trajectory to examine measured behavior before revising the policy.
Use evaluate_policy to test a concrete, falsifiable hypothesis; it automatically
performs the workspace's start/edit/score/conclude sequence. Submit complete source
with POLICY_API_VERSION=1, init_memory(), and act(obs, reset, memory, rng).
Only use jax/jax.numpy, fixed constants and optional custom_primitives. No file,
network or process access, callbacks, mutable globals or hidden simulator state.
Do not confuse normalized observations/actions with physical units. Respect
the exact action dtype and bounds and fixed memory shapes. Failed candidates
still consume budget. Use the measured results, retain unsuccessful evidence,
and compare against the best development return, not a fabricated result.
Tools only expose development data. The host selects the best scored candidate
and performs held-out evaluation AFTER this conversation ends. Do not request
held-out data. Spend the available proposals on distinct justified revisions,
then give a concise evidence-based conclusion and limitations. Reasoning
summaries, visible responses and every tool exchange are recorded.
"""


def complete(request: dict[str, Any]) -> dict[str, Any]:
    """Keep credentials in the SDK environment, outside recorded request data."""
    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    from litellm import responses

    return responses(**request).model_dump(mode="json")


class PolicyTools:
    def __init__(self, root: Path, proposals: int, baseline: dict[str, Any]) -> None:
        self.root = root
        self.remaining = proposals
        self.best = baseline
        self.bank_seconds = 0.0

    def search_seconds(self) -> float:
        # Leave two independent banks plus tracking time inside the overall limit.
        reserve = 2 * 1.25 * self.bank_seconds + 120.0
        return cli.lifetime(self.root)["remaining_seconds"] - reserve

    def dispatch(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name == "inspect_trajectory":
            request = TraceRequest.model_validate(arguments)
            try:
                return cli.trace(self.root, **request.model_dump())
            except (ValueError, FileNotFoundError) as error:
                raise ToolInputError(str(error)) from error
        if name != "evaluate_policy":
            raise ToolInputError(f"Unknown tool: {name}")
        if self.remaining <= 0:
            raise ToolInputError(
                "Proposal budget exhausted; summarize the measured results"
            )
        self.remaining -= 1
        proposal = Proposal.model_validate(arguments)
        cli.start(
            self.root,
            title=proposal.title,
            hypothesis=proposal.hypothesis,
            planned_change=proposal.planned_change,
            success_criterion="Improve mean return on the fixed development seeds.",
        )
        snapshot_bundle(self.root, self.root / ".policy_eval/candidates")
        for name, source in (
            ("policy.py", proposal.source),
            ("custom_primitives.py", proposal.helper),
        ):
            if source is None:
                (self.root / name).unlink(missing_ok=True)
            else:
                _replace_bytes(self.root / name, source.encode())
        started = time.monotonic()
        report = cli.score(self.root, timeout_seconds=self.search_seconds())
        self.bank_seconds = max(self.bank_seconds, time.monotonic() - started)
        if eligible(report) and report["mean_return"] >= self.best["mean_return"]:
            self.best = report
        kept = self.best["candidate_sha256"] == report["candidate_sha256"]
        cli.conclude(
            self.root,
            summary=json.dumps(plain(report)),
            decision="keep" if kept else "reject",
            best_candidate_sha256=self.best["candidate_sha256"],
            best_reason="Highest finite mean return on fixed development seeds.",
            limitation="Selection uses only the configured development seed bank.",
        )
        return {
            "evaluation": report,
            "best": self.best,
            "proposals_remaining": self.remaining,
        }


def run(config: Config) -> Path:
    if min(config.proposals, config.max_model_calls, config.max_output_tokens) <= 0:
        raise ValueError(
            "Proposal, model-call and output-token budgets must be positive"
        )
    if not math.isfinite(config.api_timeout_seconds) or config.api_timeout_seconds <= 0:
        raise ValueError("API timeout must be positive")
    root = config.workspace.expanduser().resolve()
    cli._active(root)
    transcript = Transcript(root / "agent_session")
    write_json(transcript.directory / "config.json", config)
    transcript.record("started", config=asdict(config))
    model_calls = 0
    usage: dict[str, int] = {}
    try:
        transcript.record(
            "tool_call",
            call_id="host-baseline",
            name="score",
            arguments={},
            origin="host",
        )
        started = time.monotonic()
        baseline = cli.score(root)
        baseline_seconds = time.monotonic() - started
        transcript.record(
            "tool_result",
            call_id="host-baseline",
            name="score",
            result=baseline,
            origin="host",
        )
        if not eligible(baseline):
            raise RuntimeError(
                "Starter must have a finite successful development score"
            )
        cli.conclude(
            root,
            summary="Established the unmodified starter baseline.",
            decision="keep",
            best_candidate_sha256=baseline["candidate_sha256"],
            best_reason="Initial finite development baseline.",
        )
        policy_tools = PolicyTools(root, config.proposals, baseline)
        policy_tools.bank_seconds = baseline_seconds
        context = {
            "documents": {
                name: (root / "context" / name).read_text()
                for name in (
                    "task_spec.md",
                    "policy_api.md",
                    "policy_guidance.md",
                    "interface.json",
                    "config.json",
                )
            },
            "policy.py": (root / "policy.py").read_text(),
            "baseline": baseline,
            "proposal_budget": config.proposals,
            "model_call_budget": config.max_model_calls,
            "remaining_seconds": cli.lifetime(root)["remaining_seconds"],
            "baseline_evaluation_seconds": baseline_seconds,
        }
        history: list[dict[str, Any]] = [
            {"role": "user", "content": json.dumps(context)}
        ]
        stop_reason = "model_call_budget"
        for turn in range(config.max_model_calls):
            remaining_time = policy_tools.search_seconds()
            if remaining_time <= 1.25 * policy_tools.bank_seconds:
                stop_reason = "time_budget"
                break
            request = {
                "model": config.model
                or ("gpt-6.2-sol" if config.host == "codex" else "openai/gpt-5.4-mini"),
                "instructions": INSTRUCTIONS,
                "input": history,
                "tools": TOOLS,
                "parallel_tool_calls": False,
                "tool_choice": "none" if turn == config.max_model_calls - 1 else "auto",
                "reasoning": {"effort": config.reasoning_effort, "summary": "auto"},
                "include": ["reasoning.encrypted_content"],
                "store": False,
                "max_output_tokens": config.max_output_tokens,
                "timeout": min(config.api_timeout_seconds, remaining_time),
                "max_retries": 0,
            }
            transcript.record("model_request", request=request, turn=turn)
            if config.host == "codex":
                from experiments.agent_policy_workspace.codex_provider import (
                    complete as codex_complete,
                )

                response = codex_complete(
                    request, transcript.directory / "codex" / f"turn_{turn:03d}"
                )
            else:
                response = complete(request)
            model_calls += 1
            transcript.record("model_response", response=response, turn=turn)
            for key, value in (response.get("usage") or {}).items():
                if isinstance(value, int):
                    usage[key] = usage.get(key, 0) + value
            if response.get("status") != "completed":
                raise RuntimeError(
                    f"Model response incomplete: {response.get('incomplete_details')}"
                )
            output = response.get("output", [])
            history.extend(output)
            calls = [item for item in output if item.get("type") == "function_call"]
            if not calls:
                stop_reason = "agent_finished"
                break
            for call in calls:
                transcript.record(
                    "tool_call",
                    call_id=call["call_id"],
                    name=call["name"],
                    arguments=call["arguments"],
                    origin="agent",
                )
                if policy_tools.search_seconds() <= 1.25 * policy_tools.bank_seconds:
                    stop_reason = "time_budget"
                    transcript.record(
                        "tool_result",
                        call_id=call["call_id"],
                        name=call["name"],
                        result={"error": "Time reserved for final evaluation"},
                        origin="host",
                    )
                    break
                try:
                    arguments = json.loads(call["arguments"])
                    if not isinstance(arguments, dict):
                        raise ToolInputError("Tool arguments must be a JSON object")
                    result = policy_tools.dispatch(call["name"], arguments)
                except (ToolInputError, json.JSONDecodeError, ValidationError) as error:
                    result = {
                        "error": str(error),
                        "proposals_remaining": policy_tools.remaining,
                    }
                except Exception as error:
                    transcript.record(
                        "tool_result",
                        call_id=call["call_id"],
                        name=call["name"],
                        result={"error": f"{type(error).__name__}: {error}"},
                        origin="agent",
                    )
                    raise
                transcript.record(
                    "tool_result",
                    call_id=call["call_id"],
                    name=call["name"],
                    result=result,
                    origin="agent",
                )
                history.append(
                    {
                        "type": "function_call_output",
                        "call_id": call["call_id"],
                        "output": json.dumps(plain(result)),
                    }
                )
            if stop_reason == "time_budget":
                break
        transcript.record(
            "search_finished",
            reason=stop_reason,
            model_calls=model_calls,
            usage=usage,
            proposals_used=config.proposals - policy_tools.remaining,
        )
        # No further model calls: held-out outcomes never enter search context.
        transcript.record(
            "tool_call",
            call_id="host-final",
            name="finish",
            arguments={},
            origin="host",
        )
        final = None
        final_error = None
        try:
            final = cli.finish(root)
        except TimeoutError as error:
            final_error = str(error)
            cli.restore_best(root)
        transcript.record(
            "tool_result",
            call_id="host-final",
            name="finish",
            result=final if final is not None else {"error": final_error},
            origin="host",
        )
        write_json(
            transcript.directory / "summary.json",
            {
                "status": "completed" if final is not None else "time_limit",
                "stop_reason": stop_reason,
                "finalization_error": final_error,
                "model_calls": model_calls,
                "usage": usage,
                "proposals_used": config.proposals - policy_tools.remaining,
                "baseline": baseline,
                "best": policy_tools.best,
                "candidate_sha256": policy_tools.best["candidate_sha256"],
                "heldout_mean_return": (
                    final["heldout"].get("mean_return") if final is not None else None
                ),
            },
        )
    except BaseException:
        transcript.record("error", error=traceback.format_exc())
        raise
    return transcript.directory


if __name__ == "__main__":
    import tyro

    print(run(tyro.cli(Config)))
