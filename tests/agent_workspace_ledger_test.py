"""Policy bundle identity, reversible restoration, and experiment history."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.agent_policy_workspace.ledger import (
    Ledger,
    bundle_sha256,
    restore_bundle,
    snapshot_bundle,
)


def test_bundle_hash_and_restore_cover_optional_helper_and_exact_bytes(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    source = b"def act(obs, rng):\r\n    return obs\r\n"
    (workspace / "policy.py").write_bytes(source)
    identity, candidate = snapshot_bundle(workspace, tmp_path / "candidates")
    assert snapshot_bundle(workspace, tmp_path / "candidates") == (identity, candidate)
    assert (candidate / "policy.py").read_bytes() == source
    (workspace / "custom_primitives.py").write_bytes(b"")
    assert bundle_sha256(workspace) != identity
    (workspace / "policy.py").write_bytes(b"changed policy\n")
    backup = restore_bundle(workspace, candidate)
    assert (backup / "policy.py").read_bytes() == b"changed policy\n"
    assert (backup / "custom_primitives.py").read_bytes() == b""
    assert not (workspace / "custom_primitives.py").exists()
    assert bundle_sha256(workspace) == identity
    (workspace / "custom_primitives.py").write_bytes(b"GAIN = 2\n")
    with_helper, helper_candidate = snapshot_bundle(workspace, tmp_path / "candidates")
    (workspace / "custom_primitives.py").unlink()
    restore_bundle(workspace, helper_candidate)
    assert bundle_sha256(workspace) == with_helper
    assert (workspace / "custom_primitives.py").read_bytes() == b"GAIN = 2\n"


def test_tampered_archive_and_symlink_policy_are_rejected(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "policy.py").write_text("original\n")
    _, candidate = snapshot_bundle(workspace, tmp_path / "candidates")
    (candidate / "policy.py").write_text("tampered\n")
    with pytest.raises(ValueError, match="archive does not match"):
        restore_bundle(workspace, candidate)
    with pytest.raises(ValueError, match="archive does not match"):
        snapshot_bundle(workspace, tmp_path / "candidates")
    assert (workspace / "policy.py").read_text() == "original\n"
    (workspace / "policy.py").unlink()
    (workspace / "policy.py").symlink_to(candidate / "policy.py")
    with pytest.raises(ValueError, match="non-symlink"):
        bundle_sha256(workspace)
    with pytest.raises(ValueError, match="candidate hash"):
        restore_bundle(workspace, tmp_path / "not-a-candidate-hash")


def test_ledger_requires_conclusion_preserves_history_and_reaffirms_best(
    tmp_path: Path,
) -> None:
    ledger = Ledger(tmp_path)
    attempt = ledger.start(
        title="baseline",
        hypothesis="zero actions establish a baseline",
        planned_change="none",
        success_criterion="finite rollout",
    )
    assert attempt == "A00"
    prefix = ledger.path.read_bytes()
    with pytest.raises(ValueError, match="conclude the current"):
        ledger.start(
            title="next",
            hypothesis="test",
            planned_change="gain",
            success_criterion="improve",
        )
    identity = "a" * 64
    ledger.observe(summary="all returns finite", candidate_sha256=identity)
    ledger.select_best(identity, "only scored candidate")
    ledger.conclude(
        decision="follow_up",
        summary="baseline measured",
        best_candidate_sha256=identity,
        best_reason="only scored candidate",
        candidate_sha256=identity,
        limitation="small development batch",
    )
    assert ledger.current_attempt() is None
    assert ledger.best_candidate() == identity
    assert ledger.path.read_bytes().startswith(prefix)
    assert (
        ledger.start(
            title="adjust gain",
            hypothesis="lower gain reduces overshoot",
            planned_change="halve gain",
            success_criterion="higher mean return",
            parent="A00",
        )
        == "A01"
    )
    assert ledger.current_attempt() == "A01"
    events = [json.loads(line) for line in ledger.path.read_text().splitlines()]
    assert [event["event"] for event in events] == [
        "start",
        "observe",
        "select_best",
        "conclude",
        "start",
    ]
    assert "lower gain reduces overshoot" in ledger.markdown.read_text()
    assert "small development batch" in ledger.markdown.read_text()
    with pytest.raises(ValueError, match="named ledger method"):
        ledger.append("conclude")
