"""Agent-facing evaluation, inspection, experiment notes, and finalization."""

from __future__ import annotations

import json
import math
import re
import shutil
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from experiments.agent_policy_search.artifacts import (
    evaluation_scalars,
    plain,
    provenance,
    read_json,
    write_json,
)
from experiments.agent_policy_search.config import (
    EvaluationConfig,
    TaskConfig,
    TrackingConfig,
)
from experiments.agent_policy_search.run import eligible
from experiments.agent_policy_workspace.ledger import (
    Ledger,
    bundle_sha256,
    restore_bundle,
    snapshot_bundle,
)
from experiments.agent_policy_workspace.runtime import describe_task, evaluate_bundle

DEFAULT_TRACKING = TrackingConfig(group="agent-policy-workspace")


@dataclass(frozen=True)
class WorkspaceConfig:
    task: TaskConfig
    development_seeds: tuple[int, ...]
    heldout_seeds: tuple[int, ...]
    created_at: float
    lifetime_seconds: float = 7200.0
    evaluation_timeout_seconds: float = 1800.0
    tracking: TrackingConfig = field(default_factory=TrackingConfig)


def _root(workspace: Path) -> Path:
    return workspace.expanduser().resolve()


def _config(workspace: Path) -> WorkspaceConfig:
    values = read_json(_root(workspace) / ".policy_eval/config.json")
    return WorkspaceConfig(
        **{
            **values,
            "task": TaskConfig(**values["task"]),
            "tracking": TrackingConfig(**values["tracking"]),
            "development_seeds": tuple(values["development_seeds"]),
            "heldout_seeds": tuple(values["heldout_seeds"]),
        }
    )


def lifetime(workspace: Path) -> dict[str, Any]:
    """Report the overall time available, including final evaluation."""
    config = _config(workspace)
    elapsed = max(0.0, time.time() - config.created_at)
    return {
        "elapsed_seconds": elapsed,
        "remaining_seconds": max(0.0, config.lifetime_seconds - elapsed),
        "finalizing": (_root(workspace) / ".policy_eval/finalizing.json").exists(),
        "finished": (_root(workspace) / ".policy_eval/final.json").exists(),
    }


def _active(workspace: Path) -> float:
    clock = lifetime(workspace)
    if clock["finalizing"] or clock["finished"]:
        raise ValueError(
            "search is finalized or finalizing; only inspection is allowed"
        )
    if clock["remaining_seconds"] <= 0:
        raise ValueError(
            "search lifetime expired; finalize the existing best candidate"
        )
    return clock["remaining_seconds"]


def _candidate(workspace: Path, candidate: str) -> Path:
    if not re.fullmatch(r"[0-9a-f]{64}", candidate):
        raise ValueError("candidate must be a SHA-256 from an official score")
    path = _root(workspace) / ".policy_eval/candidates" / candidate
    if path.is_symlink() or bundle_sha256(path) != candidate:
        raise ValueError("archived candidate hash mismatch")
    return path


def _official(workspace: Path, candidate: str) -> dict[str, Any]:
    path = _candidate(workspace, candidate) / "score.json"
    if not path.exists():
        raise ValueError("candidate has no completed official development score")
    return read_json(path)


def _best_score(workspace: Path) -> float:
    reports = (
        read_json(path)["report"]
        for path in (_root(workspace) / ".policy_eval/candidates").glob("*/score.json")
    )
    return max(
        (report["mean_return"] for report in reports if eligible(report)),
        default=float("-inf"),
    )


def _check_best(workspace: Path, candidate: str) -> dict[str, Any]:
    record = _official(workspace, candidate)
    if not eligible(record["report"]):
        raise ValueError("best candidate must have a finite successful official score")
    if record["report"]["mean_return"] < _best_score(workspace):
        raise ValueError("select a candidate with the highest mean development return")
    return record


def init(
    workspace: Path,
    env: TaskConfig,
    development_episodes: int = 64,
    heldout_episodes: int = 64,
    development_seed: int = 0,
    heldout_seed: int = 10000,
    lifetime_seconds: float = 7200.0,
    evaluation_timeout_seconds: float = 1800.0,
    tracking: TrackingConfig = DEFAULT_TRACKING,
) -> dict[str, Any]:
    """Create task documents and a stateful starter without running rollouts."""
    import numpy as np

    for seed, count in (
        (development_seed, development_episodes),
        (heldout_seed, heldout_episodes),
    ):
        if count <= 0 or seed < 0 or seed + count > 2**32:
            raise ValueError("episode counts must be positive and seeds uint32")
    development = tuple(
        range(development_seed, development_seed + development_episodes)
    )
    heldout = tuple(range(heldout_seed, heldout_seed + heldout_episodes))
    if set(development) & set(heldout):
        raise ValueError("development and heldout seeds must be disjoint")
    if any(
        not math.isfinite(value) or value <= 0
        for value in (lifetime_seconds, evaluation_timeout_seconds)
    ):
        raise ValueError("timeouts must be positive and finite")
    root = _root(workspace)
    root.mkdir(parents=True, exist_ok=False)
    config = WorkspaceConfig(
        env,
        development,
        heldout,
        time.time(),
        lifetime_seconds,
        evaluation_timeout_seconds,
        tracking,
    )
    write_json(root / ".policy_eval/config.json", config)
    templates = Path(__file__).parent / "templates"
    write_json(
        root / ".policy_eval/provenance.json",
        {
            **provenance(),
            "workspace_sources": {
                str(path.relative_to(Path(__file__).parent)): path.read_text()
                for path in sorted(Path(__file__).parent.rglob("*"))
                if path.is_file() and path.suffix in {".py", ".md"}
            },
        },
    )
    context = root / "context"
    context.mkdir()
    for name in ("task_spec.md", "policy_api.md", "policy_guidance.md"):
        shutil.copyfile(templates / name, context / name)
    for name in ("AGENTS.md", "rollouts.py"):
        shutil.copyfile(templates / name, root / name)
    write_json(context / "config.json", env)
    description = describe_task(
        root / ".policy_eval/description",
        EvaluationConfig(env, (development[0],), evaluation_timeout_seconds),
    )
    if description["status"] != "ok":
        raise RuntimeError(f"could not describe task: {description.get('error')}")
    interface = description["interface"]
    write_json(context / "interface.json", interface)
    shape = tuple(interface["action_shape"])
    initial = (
        np.zeros(shape, dtype=interface["action_dtype"])
        if "action_bins" in interface
        else (
            np.asarray(interface["action_low"]) + np.asarray(interface["action_high"])
        )
        / 2
    )
    if not np.isfinite(initial).all():
        raise ValueError("starter needs finite action bounds")
    source = (templates / "policy.py").read_text()
    for marker, value in {
        "__ACTION_SIZE__": str(int(np.prod(shape))),
        "__ACTION_DTYPE__": interface["action_dtype"],
        "__INITIAL_ACTION__": repr(np.asarray(initial).reshape(shape).tolist()),
    }.items():
        source = source.replace(marker, value)
    (root / "policy.py").write_text(source)
    (context / "starting_policy.py").write_text(source)
    Ledger(root).start(
        title="starting template",
        hypothesis="The midpoint policy establishes the development baseline.",
        planned_change="Score the unmodified starter before changing control logic.",
        success_criterion="Official score with finite returns and actions.",
    )
    return {
        "workspace": str(root),
        "interface_path": str(context / "interface.json"),
        "observation_shape": interface["observation_shape"],
        "action_shape": interface["action_shape"],
        "agent_lifetime": lifetime(root),
    }


def _evaluate(
    root: Path,
    source: Path,
    output: Path,
    seeds: tuple[int, ...],
    label: str,
    *,
    final: bool = False,
    timeout_seconds: float | None = None,
) -> dict[str, Any]:
    import wandb

    config = _config(root)
    timeout = config.evaluation_timeout_seconds
    if timeout_seconds is not None:
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("evaluation timeout must be positive and finite")
        timeout = min(timeout, timeout_seconds)
    if not final:
        timeout = min(timeout, _active(root))
    deadline = min(time.time() + timeout, config.created_at + config.lifetime_seconds)
    if deadline <= time.time():
        raise TimeoutError("Overall workspace deadline reached")
    tracking = None
    try:
        tracking = wandb.init(
            project=config.tracking.project,
            entity=config.tracking.entity,
            group=f"{config.tracking.group}-{root.name}",
            tags=list(config.tracking.tags),
            name=f"{label}-{output.name}",
            config=plain({"task": asdict(config.task), "seeds": seeds, "label": label}),
            dir=str(root / ".policy_eval"),
            mode="online",
        )
        if tracking is None or tracking.settings.mode != "online":
            raise RuntimeError("W&B online initialization is required")
        if not final:
            _active(root)
        timeout = deadline - time.time()
        if timeout <= 0:
            raise TimeoutError("Overall evaluation deadline reached")
        report = evaluate_bundle(
            source, output, EvaluationConfig(config.task, seeds, timeout)
        )
        tracking.log(
            {
                "evaluation/finite": int(report.get("finite", False)),
                **evaluation_scalars(report),
                **(
                    {"eval/return_mean": report["mean_return"]}
                    if eligible(report)
                    else {}
                ),
            }
        )
        tracking.summary["status"] = report["status"]
        artifact = wandb.Artifact(
            f"policy-workspace-{tracking.id}",
            type="policy",
            metadata={"label": label, "candidate_sha256": bundle_sha256(source.parent)},
        )
        for name in ("policy.py", "custom_primitives.py"):
            path = source.parent / name
            if path.exists():
                artifact.add_file(str(path), name=name)
        artifact.add_file(str(root / ".policy_eval/config.json"), name="config.json")
        write_json(output / "tracking.json", {"id": tracking.id, "url": tracking.url})
        tracking.log_artifact(artifact)
    except BaseException:
        write_json(output / "command_error.json", {"error": traceback.format_exc()})
        if tracking is not None:
            tracking.finish(exit_code=1)
        raise
    else:
        tracking.finish()
    return report


def _public_record(record: dict[str, Any]) -> dict[str, Any]:
    report = record["report"]
    return {
        "candidate_sha256": record["candidate_sha256"],
        "evaluation_id": record["evaluation_id"],
        "status": report["status"],
        "finite": report.get("finite", False),
        "mean_return": report.get("mean_return"),
        "error": report.get("error"),
        "episodes": report.get("feedback", {}).get("episodes", []),
    }


def validate(workspace: Path) -> dict[str, Any]:
    """Check JAX action, memory and trace contracts without stepping the simulator."""
    root = _root(workspace)
    remaining = _active(root)
    config = _config(root)
    candidate, source = snapshot_bundle(root, root / ".policy_eval/candidates")
    output = root / ".policy_eval/validations" / uuid4().hex
    report = evaluate_bundle(
        source / "policy.py",
        output,
        EvaluationConfig(
            config.task,
            (config.development_seeds[0],),
            min(remaining, config.evaluation_timeout_seconds),
        ),
        validate_only=True,
    )
    Ledger(root).append(
        "validated",
        candidate_sha256=candidate,
        status=report["status"],
        error=report.get("error"),
    )
    return {
        "candidate_sha256": candidate,
        "status": report["status"],
        "error": report.get("error"),
    }


def score(workspace: Path, timeout_seconds: float | None = None) -> dict[str, Any]:
    """Score the exact bundle on fixed development seeds; reuse an existing score."""
    root = _root(workspace)
    _active(root)
    candidate, source = snapshot_bundle(root, root / ".policy_eval/candidates")
    if (source / "score.json").exists():
        return _public_record(read_json(source / "score.json"))
    evaluation_id = uuid4().hex
    report = _evaluate(
        root,
        source / "policy.py",
        root / ".policy_eval/evaluations" / evaluation_id,
        _config(root).development_seeds,
        "development",
        timeout_seconds=timeout_seconds,
    )
    if bundle_sha256(source) != candidate:
        raise ValueError("candidate source changed during evaluation")
    record = {
        "candidate_sha256": candidate,
        "evaluation_id": evaluation_id,
        "report": report,
    }
    if report["status"] == "ok":
        write_json(source / "score.json", record)
    ledger = Ledger(root)
    ledger.append("scored", **_public_record(record))
    if eligible(report) and report["mean_return"] >= _best_score(root):
        ledger.select_best(candidate, "Highest mean return on fixed development seeds.")
    return _public_record(record)


def explore(workspace: Path, seeds: tuple[int, ...] = (7, 8)) -> dict[str, Any]:
    """Run explicitly exploratory rollouts; these never select the incumbent."""
    root = _root(workspace)
    _active(root)
    config = _config(root)
    if not seeds or any(seed < 0 or seed >= 2**32 for seed in seeds):
        raise ValueError("exploratory seeds must be a nonempty uint32 bank")
    if set(seeds) & set(config.heldout_seeds):
        raise ValueError("exploration cannot use the reserved heldout seeds")
    candidate, source = snapshot_bundle(root, root / ".policy_eval/candidates")
    evaluation_id = uuid4().hex
    output = root / ".policy_eval/explorations" / evaluation_id
    report = _evaluate(root, source / "policy.py", output, seeds, "exploratory")
    record = {
        "candidate_sha256": candidate,
        "evaluation_id": evaluation_id,
        "report": report,
    }
    write_json(output / "exploration.json", record)
    Ledger(root).append("exploratory", **_public_record(record))
    return _public_record(record)


def trace(
    workspace: Path,
    candidate: str | None = None,
    exploration: str | None = None,
    episode: int = 0,
    start: int = 0,
    stop: int = 16,
    fields: tuple[str, ...] = ("obs", "requested_action", "reward"),
) -> dict[str, Any]:
    """Read a bounded valid trajectory window, including chosen policy traces."""
    import numpy as np

    from experiments.agent_policy_search.feedback import json_safe

    root = _root(workspace)
    if candidate is not None and exploration is not None:
        raise ValueError("choose candidate or exploration")
    if start < 0 or stop <= start or stop - start > 200 or episode < 0:
        raise ValueError("choose a nonnegative episode and a window of 1 to 200 steps")
    if exploration is not None:
        if not re.fullmatch(r"[0-9a-f]{32}", exploration):
            raise ValueError("invalid exploration identifier")
        path = root / ".policy_eval/explorations" / exploration / "trajectories.npz"
    else:
        selected = candidate or Ledger(root).best_candidate()
        if selected is None:
            raise ValueError("no officially scored best candidate")
        record = _official(root, selected)
        path = (
            root
            / ".policy_eval/evaluations"
            / record["evaluation_id"]
            / "trajectories.npz"
        )
    allowed = {
        "obs",
        "next_obs",
        "requested_action",
        "applied_normalized",
        "reward",
        "terminated",
        "truncated",
        "valid",
        "done",
        "termination_code",
        "time_s",
        "time_steps",
    }
    with np.load(path, allow_pickle=False) as arrays:
        valid = arrays["valid"]
        if episode >= valid.shape[0] or stop > valid.shape[1]:
            raise ValueError("episode or window is outside the recorded trajectory")
        if not fields or any(
            name not in allowed and not name.startswith("trace_") for name in fields
        ):
            raise ValueError(
                "trace fields must be observable data or policy diagnostics"
            )
        if any(name not in arrays.files for name in fields):
            available = sorted(
                name
                for name in arrays.files
                if name in allowed or name.startswith("trace_")
            )
            raise ValueError(f"unknown field; available: {available}")
        steps = np.arange(start, stop)[valid[episode, start:stop]]
        return json_safe(
            {
                "episode": episode,
                "steps": steps,
                "fields": {name: arrays[name][episode, steps] for name in fields},
            }
        )


def start(
    workspace: Path,
    title: str,
    hypothesis: str,
    planned_change: str,
    success_criterion: str,
    parent: str | None = None,
) -> dict[str, Any]:
    """Open the next hypothesis before editing the policy."""
    _active(workspace)
    attempt = Ledger(_root(workspace)).start(
        title=title,
        hypothesis=hypothesis,
        planned_change=planned_change,
        success_criterion=success_criterion,
        parent=parent,
    )
    return {"attempt": attempt}


def observe(
    workspace: Path,
    summary: str,
    kind: Literal["exploratory", "finding", "failure"] = "finding",
) -> dict[str, Any]:
    """Append an observation without changing official scores."""
    _active(workspace)
    return Ledger(_root(workspace)).observe(summary=summary, kind=kind)


def select_best(workspace: Path, candidate: str, reason: str) -> dict[str, Any]:
    """Declare a top-scoring official candidate; raw exploratory results cannot win."""
    _active(workspace)
    _check_best(workspace, candidate)
    return Ledger(_root(workspace)).select_best(candidate, reason)


def conclude(
    workspace: Path,
    summary: str,
    decision: Literal["keep", "reject", "follow_up"],
    best_candidate_sha256: str,
    best_reason: str,
    limitation: str = "",
) -> dict[str, Any]:
    """Close the current experiment and reaffirm the preferred scored candidate."""
    _active(workspace)
    _check_best(workspace, best_candidate_sha256)
    return Ledger(_root(workspace)).conclude(
        decision=decision,
        summary=summary,
        best_candidate_sha256=best_candidate_sha256,
        best_reason=best_reason,
        limitation=limitation,
    )


def restore_best(workspace: Path) -> dict[str, Any]:
    """Restore both bundle files, preserving displaced working files in a backup."""
    root = _root(workspace)
    candidate = Ledger(root).best_candidate()
    if candidate is None:
        raise ValueError("no best candidate has been selected")
    _check_best(root, candidate)
    backup = restore_bundle(root, _candidate(root, candidate))
    return {"candidate_sha256": candidate, "backup": str(backup)}


def finish(workspace: Path) -> dict[str, Any]:
    """Freeze selection, independently rescore, then evaluate heldout seeds once."""
    root = _root(workspace)
    final_path = root / ".policy_eval/final.json"
    if final_path.exists():
        return read_json(final_path)
    candidate = Ledger(root).best_candidate()
    if candidate is None:
        raise ValueError("score a finite candidate before finishing")
    official = _check_best(root, candidate)
    frozen = root / ".policy_eval/finalizing.json"
    if frozen.exists():
        if read_json(frozen)["candidate_sha256"] != candidate:
            raise ValueError("final candidate changed after selection was frozen")
    else:
        write_json(frozen, {"candidate_sha256": candidate})
    restore_best(root)
    source = _candidate(root, candidate) / "policy.py"
    config = _config(root)
    output = root / ".policy_eval/final_evaluation" / uuid4().hex
    development = _evaluate(
        root,
        source,
        output / "development",
        config.development_seeds,
        "final-development",
        final=True,
    )
    if not eligible(development):
        if development["status"] == "timeout":
            raise TimeoutError("Final development evaluation timed out")
        raise RuntimeError(
            "final development verification failed; see saved evaluation"
        )
    import numpy as np

    np.testing.assert_allclose(
        [item["return"] for item in development["feedback"]["episodes"]],
        [item["return"] for item in official["report"]["feedback"]["episodes"]],
        rtol=1e-5,
        atol=1e-6,
        err_msg="independent development evaluation disagrees with the official score",
    )
    heldout = _evaluate(
        root,
        source,
        output / "heldout",
        config.heldout_seeds,
        "final-heldout",
        final=True,
    )
    if heldout["status"] != "ok":
        if heldout["status"] == "timeout":
            raise TimeoutError("Heldout evaluation timed out")
        raise RuntimeError(
            "heldout evaluation failed; inspect artifacts and retry finish"
        )
    if bundle_sha256(source.parent) != candidate:
        raise ValueError("candidate source changed during final evaluation")
    result = {
        "candidate_sha256": candidate,
        "development": development,
        "heldout": heldout,
    }
    write_json(final_path, result)
    Ledger(root).append(
        "finalized",
        candidate_sha256=candidate,
        heldout_status=heldout["status"],
        heldout_finite=heldout.get("finite", False),
    )
    return result


def main() -> None:
    import tyro

    commands = {
        "init": init,
        "validate": validate,
        "score": score,
        "explore": explore,
        "trace": trace,
        "lifetime": lifetime,
        "start": start,
        "observe": observe,
        "conclude": conclude,
        "select-best": select_best,
        "restore-best": restore_best,
        "finish": finish,
    }

    def display(function: Any) -> Any:
        import functools

        @functools.wraps(function)
        def wrapped(*args: Any, **kwargs: Any) -> None:
            print(
                json.dumps(plain(function(*args, **kwargs)), indent=2, allow_nan=False)
            )

        return wrapped

    tyro.extras.subcommand_cli_from_dict(
        {name: display(function) for name, function in commands.items()}
    )
