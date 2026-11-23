"""Evaluate archived source with recorded configuration and seed banks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from experiments.agent_policy_search.artifacts import (
    evaluation_scalars,
    event,
    read_json,
    source_hash,
    write_json,
)
from experiments.agent_policy_search.config import EvaluationConfig, TaskConfig
from experiments.agent_policy_search.evaluator import evaluate_candidate


@dataclass(frozen=True)
class EvaluateConfig:
    run_dir: Path
    output_dir: Path
    candidate_id: str | None = None
    split: Literal["development", "heldout"] = "development"


def evaluate_saved(config: EvaluateConfig) -> dict:
    """Use a fresh output directory, leaving the original experiment immutable."""
    import wandb

    run_dir = config.run_dir.expanduser().resolve()
    settings = read_json(run_dir / "config.json")
    candidate_id = (
        config.candidate_id or read_json(run_dir / "best.json")["candidate_id"]
    )
    if not candidate_id.isdigit() or len(candidate_id) < 4:
        raise ValueError("candidate_id must be a numeric candidate directory name")
    candidate = run_dir / "candidates" / candidate_id
    source = candidate / "policy.py"
    metadata = read_json(candidate / "candidate.json")
    if source_hash(source.read_bytes()) != metadata["source_sha256"]:
        raise ValueError("archived policy source hash mismatch")
    output = config.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "replay.json", config)
    tracking = None
    try:
        tracking = wandb.init(
            project=settings["tracking"]["project"],
            entity=settings["tracking"]["entity"],
            group=settings["tracking"]["group"],
            name=f"replay-{run_dir.name}-{candidate_id}-{config.split}",
            dir=str(output),
            mode="online",
            config={"original_run": str(run_dir), "candidate_id": candidate_id},
        )
        if tracking is None or tracking.settings.mode != "online":
            raise RuntimeError("W&B online initialization is required")
        seeds = read_json(run_dir / "seeds.json")[config.split]
        result = evaluate_candidate(
            source,
            output / "evaluation",
            EvaluationConfig(
                TaskConfig(**settings["task"]),
                tuple(seeds),
                settings["evaluation_timeout_seconds"],
            ),
        )
        tracking.summary.update(
            {"status": result["status"], "mean_return": result.get("mean_return")}
        )
        tracking.log(evaluation_scalars(result))
        if result["status"] != "ok":
            raise RuntimeError(f"evaluation failed: {result.get('error')}")
    except BaseException as error:
        event(output, "failed", error=f"{type(error).__name__}: {error}")
        if tracking is not None:
            tracking.finish(exit_code=1)
        raise
    else:
        event(output, "completed", candidate_id=candidate_id, split=config.split)
        tracking.finish()
    return result


if __name__ == "__main__":
    import tyro

    evaluate_saved(tyro.cli(EvaluateConfig))
