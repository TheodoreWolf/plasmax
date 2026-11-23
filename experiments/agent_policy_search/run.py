"""Sequential policy-as-code search: propose, archive, evaluate, select."""

from __future__ import annotations

import math
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any

from experiments.agent_policy_search import model
from experiments.agent_policy_search.artifacts import (
    create_run,
    evaluation_scalars,
    event,
    plain,
    provenance,
    set_best,
    source_hash,
    write_json,
)
from experiments.agent_policy_search.config import EvaluationConfig, SearchConfig
from experiments.agent_policy_search.evaluator import evaluate_candidate


def eligible(report: dict[str, Any]) -> bool:
    value = report.get("mean_return")
    return (
        report.get("status") == "ok"
        and report.get("finite") is True
        and isinstance(value, (int, float))
        and math.isfinite(value)
    )


def _save_source(
    directory: Path, source: str, *, parent: str | None, description: str
) -> Path:
    source_path = directory / "policy.py"
    with source_path.open("xb") as stream:
        stream.write(source.encode("utf-8"))
    write_json(
        directory / "candidate.json",
        {
            "candidate_id": directory.name,
            "parent_id": parent,
            "description": description,
            "source_sha256": source_hash(source),
        },
    )
    return source_path


def _record(
    run_dir: Path,
    directory: Path,
    report: dict[str, Any],
    *,
    accepted: bool,
    incumbent_id: str,
    tracking: Any,
) -> None:
    import wandb

    outcome = {
        "candidate_id": directory.name,
        "accepted": accepted,
        "incumbent_id": incumbent_id,
        "status": report["status"],
        "mean_return": report.get("mean_return"),
        "finite": report.get("finite", False),
        "error": report.get("error"),
    }
    write_json(directory / "outcome.json", outcome)
    event(run_dir, "candidate_evaluated", **outcome)
    metrics = {
        "search/candidate": int(directory.name),
        "search/accepted": int(accepted),
        "search/failed": int(report["status"] != "ok"),
        "search/nonfinite": int(report["status"] == "ok" and not eligible(report)),
        "search/incumbent": int(incumbent_id),
        **evaluation_scalars(report),
    }
    if eligible(report):
        metrics["search/return_mean"] = report["mean_return"]
    tracking.log(metrics)
    artifact = wandb.Artifact(
        name=f"policy-search-{tracking.id}-{directory.name}",
        type="policy",
        metadata=plain(outcome),
    )
    # Full numeric traces stay local; source, feedback and exact API context are
    # available both locally and in the experiment's online artifact history.
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.suffix in {".json", ".py"}:
            artifact.add_file(str(path), name=str(path.relative_to(directory)))
    artifact.add_file(str(run_dir / "config.json"), name="search_config.json")
    artifact.add_file(str(run_dir / "seeds.json"), name="seed_banks.json")
    tracking.log_artifact(artifact)
    print(
        f"candidate {directory.name}: {report['status']}; "
        f"return={report.get('mean_return')}; incumbent={incumbent_id}",
        flush=True,
    )


def run_search(config: SearchConfig) -> Path:
    """Run a finite search. API and tracking errors stop with artifacts intact."""
    banks = config.seed_banks()
    seed_source = config.seed_policy.expanduser().resolve().read_text()
    run_dir = create_run(config.output_root)
    write_json(run_dir / "config.json", config)
    write_json(run_dir / "seeds.json", banks)
    write_json(run_dir / "provenance.json", provenance())
    event(run_dir, "started")
    print(f"Run artifacts: {run_dir}", flush=True)
    tracking = None
    try:
        import wandb

        tracking = wandb.init(
            project=config.tracking.project,
            entity=config.tracking.entity,
            group=config.tracking.group,
            tags=list(config.tracking.tags),
            name=run_dir.name,
            config=plain(asdict(config)),
            dir=str(run_dir),
            mode="online",
        )
        if tracking is None or tracking.settings.mode != "online":
            raise RuntimeError("W&B online initialization is required")
        write_json(run_dir / "tracking.json", {"id": tracking.id, "url": tracking.url})
        evaluation = EvaluationConfig(
            config.task, banks["development"], config.evaluation_timeout_seconds
        )
        incumbent_id = "0000"
        incumbent_dir = run_dir / "candidates" / incumbent_id
        incumbent_dir.mkdir(parents=True)
        incumbent_path = _save_source(
            incumbent_dir, seed_source, parent=None, description="Supplied seed policy"
        )
        incumbent_report = evaluate_candidate(
            incumbent_path, incumbent_dir / "evaluation", evaluation
        )
        _record(
            run_dir,
            incumbent_dir,
            incumbent_report,
            accepted=eligible(incumbent_report),
            incumbent_id=incumbent_id,
            tracking=tracking,
        )
        if not eligible(incumbent_report):
            raise RuntimeError(
                "seed policy must produce a finite successful evaluation"
            )
        set_best(run_dir, incumbent_id)
        descriptor = incumbent_report["interface"]
        write_json(run_dir / "interface.json", descriptor)
        latest = None
        for index in range(1, config.proposals + 1):
            candidate_id = f"{index:04d}"
            directory = run_dir / "candidates" / candidate_id
            directory.mkdir()
            write_json(directory / "parent.json", {"parent_id": incumbent_id})
            request = model.build_request(
                config.model,
                task={"config": asdict(config.task), "interface": descriptor},
                incumbent_source=incumbent_path.read_text(),
                incumbent_feedback=incumbent_report["feedback"],
                latest=latest,
            )
            write_json(directory / "request.json", request)
            started = time.monotonic()
            try:
                response = model.complete(request)
            except Exception:
                write_json(
                    directory / "api_error.json", {"error": traceback.format_exc()}
                )
                raise
            write_json(directory / "response.json", response)
            usage = response.get("usage") or {}
            tracking.log(
                {
                    "api/candidate": index,
                    "api/seconds": time.monotonic() - started,
                    **{
                        f"api/{key}": usage[key]
                        for key in (
                            "prompt_tokens",
                            "completion_tokens",
                            "total_tokens",
                        )
                        if isinstance(usage.get(key), (int, float))
                    },
                }
            )
            candidate_source = None
            description = None
            try:
                proposal = model.parse_proposal(response)
                candidate_source, description = proposal.source, proposal.description
                candidate_path = _save_source(
                    directory,
                    candidate_source,
                    parent=incumbent_id,
                    description=description,
                )
            except ValueError as error:
                report = {"status": "error", "finite": False, "error": str(error)}
                write_json(directory / "proposal_error.json", report)
            else:
                report = evaluate_candidate(
                    candidate_path, directory / "evaluation", evaluation
                )
            accepted = eligible(report) and (
                report["mean_return"] > incumbent_report["mean_return"]
            )
            if accepted:
                incumbent_id = candidate_id
                incumbent_path = candidate_path
                incumbent_report = report
                set_best(run_dir, incumbent_id)
            latest = {
                "candidate_id": candidate_id,
                "source": candidate_source,
                "description": description,
                "accepted": accepted,
                "status": report["status"],
                "feedback": report.get("feedback"),
                "error": report.get("error"),
            }
            _record(
                run_dir,
                directory,
                report,
                accepted=accepted,
                incumbent_id=incumbent_id,
                tracking=tracking,
            )
        heldout = evaluate_candidate(
            incumbent_path,
            run_dir / "heldout",
            EvaluationConfig(
                config.task, banks["heldout"], config.evaluation_timeout_seconds
            ),
        )
        write_json(
            run_dir / "summary.json",
            {
                "best_candidate_id": incumbent_id,
                "development_mean_return": incumbent_report["mean_return"],
                "heldout": heldout,
            },
        )
        tracking.summary["best_candidate_id"] = incumbent_id
        tracking.summary["development_mean_return"] = incumbent_report["mean_return"]
        tracking.summary["heldout_status"] = heldout["status"]
        tracking.log(
            {
                "evaluation/split": "heldout",
                **evaluation_scalars(heldout),
            }
        )
        if eligible(heldout):
            tracking.summary["heldout_mean_return"] = heldout["mean_return"]
        if heldout["status"] != "ok":
            raise RuntimeError(f"heldout evaluation failed: {heldout.get('error')}")
        event(run_dir, "completed", best_candidate_id=incumbent_id)
    except BaseException:
        event(run_dir, "failed", error=traceback.format_exc())
        if tracking is not None:
            tracking.finish(exit_code=1)
        raise
    else:
        tracking.finish()
    return run_dir


if __name__ == "__main__":
    import tyro

    run_search(tyro.cli(SearchConfig))
