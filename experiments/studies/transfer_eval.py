"""Shared metrics for zero-shot simulator-backend transfer evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

__all__ = ["transfer_metrics", "write_transfer_summary"]


def transfer_metrics(
    source_backend: str,
    target_backend: str,
    source_returns: np.ndarray,
    source_lengths: np.ndarray,
    target_returns: np.ndarray,
    target_lengths: np.ndarray,
    eval_seconds: float,
) -> dict[str, float | str]:
    """Summarize paired source/target evaluations over seeds and episodes.

    All arrays have shape ``(num_training_seeds, num_evaluation_episodes)``.
    Source and target evaluations must use the same episode-key bank so their
    per-seed gap is paired rather than confounded by evaluation randomness.
    """
    arrays = {
        "source_returns": np.asarray(source_returns),
        "source_lengths": np.asarray(source_lengths),
        "target_returns": np.asarray(target_returns),
        "target_lengths": np.asarray(target_lengths),
    }
    expected_shape = arrays["source_returns"].shape
    if len(expected_shape) != 2:
        raise ValueError(
            "transfer arrays must have shape (training_seeds, evaluation_episodes)"
        )
    for name, values in arrays.items():
        if values.shape != expected_shape:
            raise ValueError(
                f"{name} shape {values.shape} does not match {expected_shape}"
            )

    per_seed_source = arrays["source_returns"].mean(axis=1)
    per_seed_target = arrays["target_returns"].mean(axis=1)
    per_seed_gap = per_seed_target - per_seed_source
    per_seed_ratio = np.divide(
        per_seed_target,
        per_seed_source,
        out=np.full_like(per_seed_target, np.nan, dtype=np.float64),
        where=per_seed_source != 0.0,
    )

    metrics: dict[str, float | str] = {
        "transfer/num_training_seeds": float(expected_shape[0]),
        "transfer/num_evaluation_episodes": float(expected_shape[1]),
        "transfer/source_backend": source_backend,
        "transfer/target_backend": target_backend,
        "transfer/source_return_mean": float(arrays["source_returns"].mean()),
        "transfer/source_return_std": float(arrays["source_returns"].std()),
        "transfer/target_return_mean": float(arrays["target_returns"].mean()),
        "transfer/target_return_std": float(arrays["target_returns"].std()),
        "transfer/target_return_min": float(arrays["target_returns"].min()),
        "transfer/target_return_max": float(arrays["target_returns"].max()),
        "transfer/return_gap": float(per_seed_gap.mean()),
        "transfer/return_ratio": float(np.nanmean(per_seed_ratio)),
        "transfer/target_episode_length_mean": float(arrays["target_lengths"].mean()),
        "transfer/source_episode_length_mean": float(arrays["source_lengths"].mean()),
        "transfer/eval_s": eval_seconds,
    }
    if expected_shape[0] > 1:
        metrics.update(
            {
                "transfer/source_return_seed_std": float(per_seed_source.std()),
                "transfer/target_return_seed_std": float(per_seed_target.std()),
                "transfer/return_gap_seed_std": float(per_seed_gap.std()),
                "transfer/return_ratio_seed_std": float(np.nanstd(per_seed_ratio)),
            }
        )
    return metrics


def write_transfer_summary(
    out_dir: str | Path | None,
    run_name: str,
    metrics: dict[str, float | str],
) -> Path | None:
    """Persist transfer summary metrics beside a run's training artifacts."""
    if out_dir is None:
        return None
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{run_name}_transfer.json"
    path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    return path
