"""Append-only experiment artifacts, without importing the simulator runtime."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import subprocess
import sys
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any
from uuid import uuid4


def plain(value: Any) -> Any:
    """Use strict JSON, representing nonfinite numbers explicitly as strings."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return plain(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return (
            "NaN" if math.isnan(value) else ("Infinity" if value > 0 else "-Infinity")
        )
    return value


def write_json(path: Path, value: Any) -> None:
    """Create an artifact exactly once; accidental overwrites are errors."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(plain(value), stream, indent=2, allow_nan=False)
        stream.write("\n")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def timestamp() -> str:
    return datetime.now(UTC).isoformat()


def create_run(root: Path) -> Path:
    name = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    path = root.expanduser().resolve() / f"{name}-{uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def source_hash(source: str | bytes) -> str:
    data = source.encode("utf-8") if isinstance(source, str) else source
    return hashlib.sha256(data).hexdigest()


def evaluation_scalars(report: dict[str, Any]) -> dict[str, float]:
    """W&B evaluation metrics, with padding excluded from the reward denominator."""
    values = {}
    if "evaluation_seconds" in report:
        values["evaluation/seconds"] = report["evaluation_seconds"]
    feedback = report.get("feedback", {})
    if "nonfinite_reward_count" in feedback:
        count = sum(episode["length"] for episode in feedback["episodes"])
        values["evaluation/nonfinite_reward_rate"] = feedback[
            "nonfinite_reward_count"
        ] / max(count, 1)
    return values


def event(run_dir: Path, name: str, **data: Any) -> None:
    with (run_dir / "events.jsonl").open("a") as stream:
        stream.write(json.dumps(plain({"time": timestamp(), "event": name, **data})))
        stream.write("\n")


def set_best(run_dir: Path, candidate_id: str) -> None:
    temporary = run_dir / "best.tmp"
    temporary.write_text(json.dumps({"candidate_id": candidate_id}) + "\n")
    temporary.replace(run_dir / "best.json")


def provenance() -> dict[str, Any]:
    """Record checkout and runtime identity, including pre-existing edits."""
    repo = Path(__file__).resolve().parents[2]
    result: dict[str, Any] = {"python": sys.version}
    for key, command in (
        ("git_commit", ["git", "rev-parse", "HEAD"]),
        ("git_status", ["git", "status", "--porcelain"]),
        ("git_diff", ["git", "diff", "HEAD"]),
    ):
        proc = subprocess.run(command, cwd=repo, capture_output=True, text=True)
        result[key] = proc.stdout if proc.returncode == 0 else None
    packages = {}
    for name in ("plasmax", "jax", "jax-envelope", "torax", "litellm", "wandb"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    result["packages"] = packages
    lock = repo / "uv.lock"
    result["uv_lock_sha256"] = source_hash(lock.read_text())
    result["harness_sources"] = {
        str(path.relative_to(repo)): path.read_text()
        for path in sorted(Path(__file__).parent.glob("*.py"))
    }
    return result
