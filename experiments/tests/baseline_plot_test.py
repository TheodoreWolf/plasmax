"""Contracts for local baseline aggregation and plotting."""

import csv
import dataclasses

import numpy as np
import pytest

from experiments.plotting.plot_baseline_learning_curves import (
    Aggregate,
    Observation,
    _aggregate,
    _plot,
    _write_observations,
)


def _observation(seed: int, value: float) -> Observation:
    return Observation(
        algorithm="ppo",
        env="iter/hybrid/flattop",
        backend="bohm_gyrobohm",
        variant="oracle",
        reward="P_diff",
        train_steps=1_000_000,
        seed=seed,
        return_mean=value,
    )


def test_ci_is_across_training_seed_means_with_student_t():
    values = np.arange(10, dtype=np.float64)
    aggregate = _aggregate(
        [_observation(seed, value) for seed, value in enumerate(values)]
    )[0]

    expected_std = values.std(ddof=1)
    expected_half_width = 2.262 * expected_std / np.sqrt(10)
    np.testing.assert_allclose(
        aggregate.return_mean, values.mean(), rtol=1e-7, atol=0.0
    )
    np.testing.assert_allclose(
        aggregate.return_std, expected_std, rtol=1e-7, atol=0.0
    )
    np.testing.assert_allclose(
        aggregate.ci95_high - aggregate.return_mean,
        expected_half_width,
        rtol=1e-7,
        atol=0.0,
    )
    np.testing.assert_allclose(
        aggregate.return_mean - aggregate.ci95_low,
        expected_half_width,
        rtol=1e-7,
        atol=0.0,
    )
    assert aggregate.env == "iter/hybrid/flattop"
    assert aggregate.backend == "bohm_gyrobohm"
    assert aggregate.variant == "oracle"
    assert aggregate.reward == "P_diff"
    assert aggregate.train_steps == 1_000_000


def test_duplicate_seed_artifacts_are_rejected():
    row = _observation(0, 1.0)

    with pytest.raises(ValueError, match="duplicate seed"):
        _aggregate([row, row])


def test_same_algorithm_from_distinct_runs_is_kept_separate():
    row = _observation(0, 1.0)

    aggregates = _aggregate(
        [
            dataclasses.replace(row, run_id="run-a"),
            dataclasses.replace(row, run_id="run-b"),
        ]
    )

    assert [aggregate.run_id for aggregate in aggregates] == ["run-a", "run-b"]


def test_seed_level_csv_preserves_all_recorded_evaluation_fields(tmp_path):
    row = Observation(
        algorithm="direct_policy",
        env="sparc/prd/rampdown",
        backend="bohm_gyrobohm",
        variant="oracle",
        reward="rampdown",
        train_steps=100,
        seed=7,
        seed_index=2,
        return_mean=-10.0,
        source_metrics="run/metrics.csv",
        evaluation=(
            ("evaluation/best_return_mean", "-9.0"),
            ("evaluation/return_mean", "-10.0"),
            ("evaluation/return_std", "1.5"),
        ),
    )

    path = tmp_path / "baseline_returns.csv"
    _write_observations([row], path)

    with path.open(newline="") as input_file:
        written = list(csv.DictReader(input_file))
    assert written == [
        {
            "algorithm": "direct_policy",
            "study": "",
            "run_id": "",
            "series_label": "",
            "env": "sparc/prd/rampdown",
            "backend": "bohm_gyrobohm",
            "variant": "oracle",
            "reward": "rampdown",
            "train_steps": "100",
            "seed": "7",
            "seed_index": "2",
            "source_metrics": "run/metrics.csv",
            "evaluation/best_return_mean": "-9.0",
            "evaluation/return_mean": "-10.0",
            "evaluation/return_std": "1.5",
        }
    ]


def test_plot_emits_one_figure_per_environment(tmp_path):
    rows = [
        Aggregate(
            algorithm="ppo",
            study="baseline-v1",
            run_id="run-ppo",
            series_label="",
            env="iter/hybrid/flattop",
            backend="bohm_gyrobohm",
            variant=variant,
            reward="P_diff",
            train_steps=100,
            n_seeds=10,
            return_mean=value,
            return_std=1.0,
            return_sem=0.1,
            ci95_low=value - 0.2,
            ci95_high=value + 0.2,
        )
        for variant, value in (("oracle", 1.0), ("realistic", 0.5))
    ]

    _plot(rows, tmp_path, "linear", 72)

    assert sorted(path.name for path in tmp_path.glob("*.png")) == [
        "iter_hybrid_flattop.png"
    ]
