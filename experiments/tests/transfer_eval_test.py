"""Contracts for paired source-to-target backend transfer metrics."""

import numpy as np
import pytest

from training.evaluation import transfer_metrics, write_transfer_summary


def test_transfer_metrics_aggregate_paired_seed_gaps():
    metrics = transfer_metrics(
        "bohm_gyrobohm",
        "tglfnn",
        source_returns=np.asarray([[2.0, 4.0], [4.0, 6.0]]),
        source_lengths=np.asarray([[10, 10], [10, 10]]),
        target_returns=np.asarray([[1.0, 3.0], [3.0, 5.0]]),
        target_lengths=np.asarray([[9, 10], [8, 10]]),
        eval_seconds=1.5,
    )

    assert metrics["transfer/source_backend"] == "bohm_gyrobohm"
    assert metrics["transfer/target_backend"] == "tglfnn"
    np.testing.assert_allclose(
        metrics["transfer/source_return_mean"], 4.0, atol=0.0, rtol=0.0
    )
    np.testing.assert_allclose(
        metrics["transfer/target_return_mean"], 3.0, atol=0.0, rtol=0.0
    )
    np.testing.assert_allclose(metrics["transfer/return_gap"], -1.0, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(
        metrics["transfer/return_ratio"],
        ((2.0 / 3.0) + (4.0 / 5.0)) / 2.0,
        atol=1e-12,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        metrics["transfer/target_episode_length_mean"],
        9.25,
        atol=0.0,
        rtol=0.0,
    )
    np.testing.assert_allclose(metrics["transfer/eval_s"], 1.5, atol=0.0, rtol=0.0)


def test_transfer_metrics_require_matching_two_dimensional_arrays():
    with pytest.raises(ValueError, match="must have shape"):
        transfer_metrics(
            "bohm_gyrobohm",
            "tglfnn",
            np.asarray([1.0]),
            np.asarray([1.0]),
            np.asarray([1.0]),
            np.asarray([1.0]),
            0.0,
        )

    with pytest.raises(ValueError, match="target_returns shape"):
        transfer_metrics(
            "bohm_gyrobohm",
            "tglfnn",
            np.ones((2, 3)),
            np.ones((2, 3)),
            np.ones((2, 2)),
            np.ones((2, 3)),
            0.0,
        )


def test_write_transfer_summary(tmp_path):
    path = write_transfer_summary(
        tmp_path,
        "run-name",
        {"transfer/source_backend": "bohm_gyrobohm", "transfer/return_gap": -1.0},
    )

    assert path == tmp_path / "run-name_transfer.json"
    assert path is not None
    assert '"transfer/return_gap": -1.0' in path.read_text()


def test_write_transfer_summary_is_disabled_without_output_dir():
    assert write_transfer_summary(None, "run-name", {}) is None
