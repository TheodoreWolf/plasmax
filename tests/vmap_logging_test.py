"""Local artifact and uncertainty contracts for vmapped seed logging."""

import csv
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest

import training.vmap_logging as vmap_logging
from scripts.project_paths import OUTPUTS_DIR, wandb_dir


@dataclass
class _FakeArtifact:
    name: str
    type: str
    metadata: dict
    files: list[Path] = field(default_factory=list)

    def add_file(self, path: str) -> None:
        self.files.append(Path(path))


@pytest.fixture(autouse=True)
def mock_artifact(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(vmap_logging.wandb, "Artifact", _FakeArtifact)


class _FakeRun:
    def __init__(self, name):
        self.name = name
        self.summary = {}
        self.logged = []
        self.committed_steps = []
        self.finished = False
        self.exit_code = None
        self.artifacts = []

    def log(self, values, step, commit=False):
        self.logged.append((step, values))
        if commit:
            self.committed_steps.append(step)

    def log_artifact(self, artifact: _FakeArtifact) -> None:
        self.artifacts.append(artifact)

    def finish(self, exit_code: int = 0) -> None:
        self.finished = True
        self.exit_code = exit_code


def test_wandb_dir_defaults_to_outputs_and_preserves_override(monkeypatch, tmp_path):
    monkeypatch.delenv("WANDB_DIR", raising=False)
    assert wandb_dir() == OUTPUTS_DIR / "wandb"

    override = tmp_path / "cluster-wandb"
    monkeypatch.setenv("WANDB_DIR", str(override))
    assert wandb_dir() == override


@pytest.mark.parametrize("with_sweep_manifest", [False, True])
def test_seed_logger_streams_long_form_rows_and_raw_npz(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, with_sweep_manifest: bool
) -> None:
    fake_run = _FakeRun("wandb-name")
    monkeypatch.setattr(vmap_logging.wandb, "init", lambda **_: fake_run)
    if with_sweep_manifest:
        (tmp_path / "sweep_trial.json").write_text('{"snapshot": "source-sha"}\n')
    logger = vmap_logging.SeedBufferLogger(
        num_seeds=2,
        seed_ids=(10, 11),
        run_name="stable-run-name",
        config={"algorithm": "sac"},
        out_dir=str(tmp_path),
    )

    logger.log(
        100,
        0,
        {
            "evaluation/return_mean": 1.0,
            "evaluation/episode_length_mean": 5.0,
        },
    )
    assert fake_run.logged == []
    logger.log(
        100,
        1,
        {
            "evaluation/return_mean": 3.0,
            "evaluation/episode_length_mean": 7.0,
        },
    )
    assert fake_run.committed_steps == [100]
    logger.finish()

    csv_path = tmp_path / "stable-run-name_metrics.csv"
    with csv_path.open(newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))
    assert [int(row["seed"]) for row in rows] == [10, 11]
    assert [float(row["evaluation/return_mean"]) for row in rows] == [1.0, 3.0]

    history = np.load(tmp_path / "stable-run-name_history.npz")
    np.testing.assert_array_equal(history["steps"], [100])
    np.testing.assert_array_equal(history["seed_ids"], [10, 11])
    np.testing.assert_array_equal(history["evaluation/return_mean"], [[1.0, 3.0]])

    step, logged = fake_run.logged[0]
    assert step == 100
    np.testing.assert_allclose(
        logged["evaluation/return_mean"], 2.0, rtol=1e-7, atol=0.0
    )
    np.testing.assert_allclose(
        logged["evaluation/return_mean_seed_std"],
        np.sqrt(2.0),
        rtol=1e-7,
        atol=0.0,
    )
    assert "evaluation/return_mean_seed_sem" not in logged
    assert "evaluation/return_mean_seed_ci95" not in logged
    assert logged["seeds/10/evaluation/return_mean"] == 1.0
    assert logged["seeds/11/evaluation/return_mean"] == 3.0
    assert len(fake_run.artifacts) == 1
    artifact = fake_run.artifacts[0]
    assert artifact.type == "history"
    assert artifact.metadata == {
        "num_seeds": 2,
        "seed_ids": [10, 11],
        "num_checkpoints": 1,
    }
    expected_files = {
        "stable-run-name_metrics.csv",
        "stable-run-name_history.npz",
        "stable-run-name_config.json",
    }
    if with_sweep_manifest:
        expected_files.add("sweep_trial.json")
    assert {path.name for path in artifact.files} == expected_files
    assert all(path.is_file() for path in artifact.files)
    assert fake_run.finished
    assert fake_run.exit_code == 0


def test_seed_logger_accepts_an_all_nan_diagnostic_metric(monkeypatch):
    fake_run = _FakeRun("wandb-name")
    monkeypatch.setattr(vmap_logging.wandb, "init", lambda **_: fake_run)
    logger = vmap_logging.SeedBufferLogger(
        num_seeds=2,
        run_name="nonfinite-gradient-run",
    )

    logger.log_batch(
        100,
        {
            "evaluation/return_mean": np.asarray([1.0, 3.0]),
            "train/grad_norm": np.asarray([np.nan, np.nan]),
        },
    )
    logger.finish()

    _, logged = fake_run.logged[0]
    assert np.isnan(logged["train/grad_norm"])
    assert np.isnan(logged["train/grad_norm_seed_std"])


def test_wandb_only_batches_do_not_add_local_history_steps(monkeypatch, tmp_path):
    fake_run = _FakeRun("wandb-name")
    monkeypatch.setattr(vmap_logging.wandb, "init", lambda **_: fake_run)
    logger = vmap_logging.SeedBufferLogger(
        num_seeds=2,
        seed_ids=(1, 2),
        run_name="per-update-diagnostics",
        out_dir=str(tmp_path),
    )

    logger.log_wandb_batch(
        10,
        {
            "train/grad_norm": np.asarray([2.0, 4.0]),
            "train/grad_clip_scale": np.asarray([0.5, 0.25]),
        },
    )
    logger.log_batch(
        20,
        {
            "evaluation/return_mean": np.asarray([1.0, 3.0]),
            "train/grad_norm": np.asarray([1.5, 2.5]),
            "train/grad_clip_scale": np.asarray([2.0 / 3.0, 0.4]),
        },
    )
    logger.finish()

    assert [step for step, _ in fake_run.logged] == [10, 20]
    assert fake_run.logged[0][1]["train/grad_norm"] == 3.0
    assert fake_run.logged[0][1]["seeds/1/train/grad_norm"] == 2.0
    assert fake_run.logged[0][1]["seeds/2/train/grad_norm"] == 4.0
    assert fake_run.logged[1][1]["evaluation/return_mean"] == 2.0

    history = np.load(tmp_path / "per-update-diagnostics_history.npz")
    np.testing.assert_array_equal(history["steps"], [20])
    np.testing.assert_array_equal(history["train/grad_norm"], [[1.5, 2.5]])

    with (tmp_path / "per-update-diagnostics_metrics.csv").open(newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))
    assert len(rows) == 2
    assert {int(row["train_steps"]) for row in rows} == {20}


def test_finish_records_failure_when_training_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_run = _FakeRun("wandb-name")
    monkeypatch.setattr(vmap_logging.wandb, "init", lambda **_: fake_run)
    logger = vmap_logging.SeedBufferLogger(num_seeds=1, run_name="failed-run")

    with pytest.raises(RuntimeError, match="training failed"):
        try:
            raise RuntimeError("training failed")
        finally:
            logger.finish()

    assert fake_run.finished
    assert fake_run.exit_code == 1


def test_history_upload_failure_finishes_the_run_as_failed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_run = _FakeRun("wandb-name")
    monkeypatch.setattr(vmap_logging.wandb, "init", lambda **_: fake_run)

    def fail_upload(artifact: _FakeArtifact) -> None:
        raise RuntimeError("history upload failed")

    monkeypatch.setattr(fake_run, "log_artifact", fail_upload)
    logger = vmap_logging.SeedBufferLogger(
        num_seeds=1, run_name="failed-upload", out_dir=str(tmp_path)
    )
    logger.log_batch(100, {"evaluation/return_mean": np.asarray([1.0])})

    with pytest.raises(RuntimeError, match="history upload failed"):
        logger.finish()

    assert fake_run.finished
    assert fake_run.exit_code == 1


@pytest.mark.parametrize("with_sweep_manifest", [False, True])
def test_failed_run_before_first_evaluation_uploads_provenance(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, with_sweep_manifest: bool
) -> None:
    fake_run = _FakeRun("wandb-name")
    monkeypatch.setattr(vmap_logging.wandb, "init", lambda **_: fake_run)
    if with_sweep_manifest:
        (tmp_path / "sweep_trial.json").write_text('{"snapshot": "source-sha"}\n')
    logger = vmap_logging.SeedBufferLogger(
        num_seeds=5,
        run_name="failed-compile",
        config={"algorithm": "direct"},
        out_dir=str(tmp_path),
    )

    with pytest.raises(RuntimeError, match="compile failed"):
        try:
            raise RuntimeError("compile failed")
        finally:
            logger.finish()

    assert fake_run.finished
    assert fake_run.exit_code == 1
    assert len(fake_run.artifacts) == 1
    artifact = fake_run.artifacts[0]
    assert artifact.metadata["num_checkpoints"] == 0
    expected_files = {"failed-compile_config.json"}
    if with_sweep_manifest:
        expected_files.add("sweep_trial.json")
    assert {path.name for path in artifact.files} == expected_files
    assert all(path.is_file() for path in artifact.files)
