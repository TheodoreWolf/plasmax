"""Local artifact and uncertainty contracts for vmapped seed logging."""

import csv

import numpy as np

import training.vmap_logging as vmap_logging
from scripts.project_paths import OUTPUTS_DIR, wandb_dir


class _FakeRun:
    def __init__(self, name):
        self.name = name
        self.summary = {}
        self.logged = []
        self.finished = False

    def log(self, values, step):
        self.logged.append((step, values))

    def finish(self):
        self.finished = True


def test_wandb_dir_defaults_to_outputs_and_preserves_override(monkeypatch, tmp_path):
    monkeypatch.delenv("WANDB_DIR", raising=False)
    assert wandb_dir() == OUTPUTS_DIR / "wandb"

    override = tmp_path / "cluster-wandb"
    monkeypatch.setenv("WANDB_DIR", str(override))
    assert wandb_dir() == override


def test_seed_logger_streams_long_form_rows_and_raw_npz(monkeypatch, tmp_path):
    fake_run = _FakeRun("wandb-name")
    monkeypatch.setattr(vmap_logging.wandb, "init", lambda **_: fake_run)
    logger = vmap_logging.SeedBufferLogger(
        num_seeds=2,
        seed_ids=(10, 11),
        run_name="stable-run-name",
        config={"algorithm": "sac"},
        out_dir=str(tmp_path),
    )

    logger.log_batch(
        100,
        {
            "evaluation/return_mean": np.asarray([1.0, 3.0]),
            "evaluation/episode_length_mean": np.asarray([5.0, 7.0]),
        },
    )
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
    assert fake_run.finished


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
    assert fake_run.logged[1][1]["evaluation/return_mean"] == 2.0

    history = np.load(tmp_path / "per-update-diagnostics_history.npz")
    np.testing.assert_array_equal(history["steps"], [20])
    np.testing.assert_array_equal(history["train/grad_norm"], [[1.5, 2.5]])

    with (tmp_path / "per-update-diagnostics_metrics.csv").open(
        newline=""
    ) as csv_file:
        rows = list(csv.DictReader(csv_file))
    assert len(rows) == 2
    assert {int(row["train_steps"]) for row in rows} == {20}
