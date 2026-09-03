"""Contracts for the canonical baseline-v1 final-return summary."""

import numpy as np
import pytest

from experiments.studies.summarize_baseline_final_returns import (
    Cell,
    ExpectedCell,
    Observation,
    SummaryRow,
    _bootstrap_mean_ci,
    _markdown_table,
    _select_canonical_runs,
)


def _observation(study: str, run_id: str, step: int, seed: int, value: float):
    return Observation(
        cell=Cell(
            algorithm="direct_policy",
            env="sparc/reduced_field/rampup",
            backend="bohm_gyrobohm",
            variant="oracle",
            reward="lh_transition",
        ),
        study=study,
        run_id=run_id,
        train_steps=step,
        seed=seed,
        seed_index=seed,
        return_mean=value,
    )


def test_corrected_recovery_replaces_original_without_pooling():
    rows = []
    for seed in range(10):
        rows.append(
            _observation(
                "baseline-v1-direct-production",
                "original",
                10_000_000,
                seed,
                -1.0,
            )
        )
        rows.append(
            _observation(
                "baseline-v1-direct-backoff-recovery",
                "corrected",
                10_000_000,
                seed,
                1.0,
            )
        )
    cell = rows[0].cell

    selected = _select_canonical_runs(
        rows,
        {cell: ExpectedCell(total_steps=10_000_000, num_seeds=10)},
    )

    assert len(selected) == 1
    assert selected[0].run_id == "corrected"
    np.testing.assert_array_equal(selected[0].values, np.ones(10))


def test_10m_knot_study_replaces_original_1m_run():
    cell = Cell(
        algorithm="direct_knots_10",
        env="sparc/reduced_field/rampup",
        backend="bohm_gyrobohm",
        variant="oracle",
        reward="lh_transition",
    )
    rows = []
    for seed in range(10):
        for study, run_id, step, value in (
            ("baseline-v1-direct-production", "original", 1_000_000, -1.0),
            (
                "baseline-v1-knots-10m-production",
                "extended",
                10_000_000,
                1.0,
            ),
        ):
            rows.append(
                Observation(
                    cell=cell,
                    study=study,
                    run_id=run_id,
                    train_steps=step,
                    seed=seed,
                    seed_index=seed,
                    return_mean=value,
                )
            )

    selected = _select_canonical_runs(
        rows,
        {cell: ExpectedCell(total_steps=10_000_000, num_seeds=10)},
    )

    assert len(selected) == 1
    assert selected[0].study == "baseline-v1-knots-10m-production"
    assert selected[0].run_id == "extended"
    assert selected[0].train_steps == 10_000_000
    np.testing.assert_array_equal(selected[0].values, np.ones(10))


def test_10m_knot_contract_rejects_partial_priority_run():
    cell = Cell(
        algorithm="direct_knots_10",
        env="step/spp_001_ec_hd/flattop",
        backend="bohm_gyrobohm_step",
        variant="oracle",
        reward="P_diff",
    )
    rows = [
        Observation(
            cell=cell,
            study="baseline-v1-knots-10m-production",
            run_id="partial",
            train_steps=5_000_000,
            seed=seed,
            seed_index=seed,
            return_mean=float(seed),
        )
        for seed in range(10)
    ]

    with pytest.raises(ValueError, match="before required 10000000"):
        _select_canonical_runs(
            rows,
            {cell: ExpectedCell(total_steps=10_000_000, num_seeds=10)},
        )


def test_bootstrap_interval_is_deterministic_and_contains_mean():
    values = np.arange(10, dtype=np.float64)

    first = _bootstrap_mean_ci(values, samples=10_000, seed=7)
    second = _bootstrap_mean_ci(values, samples=10_000, seed=7)

    np.testing.assert_array_equal(first, second)
    assert first[0] < values.mean() < first[1]


def test_markdown_has_environment_rows_and_algorithm_columns():
    rows = [
        SummaryRow(
            environment="step/spp_001_ec_hd/flattop",
            variant="oracle",
            algorithm=algorithm,
            backend="bohm_gyrobohm_step",
            reward="P_diff",
            train_steps=10_000_000,
            n_seeds=10,
            metric="evaluation/return_mean",
            return_mean=1.0,
            ci95_low=0.5,
            ci95_high=1.5,
            ci_method="percentile_bootstrap",
            bootstrap_samples=100,
            bootstrap_seed=0,
            study="study",
            run_id="run",
        )
        for algorithm in (
            "ppo",
            "sac",
            "direct_policy",
            "direct_knots_1",
            "direct_knots_10",
            "direct_knots_100",
        )
    ]

    table = _markdown_table(
        rows,
        variant="oracle",
        env_order=["step/spp_001_ec_hd/flattop"],
        precision=1,
    )

    assert "| Environment | PPO | SAC | Direct policy |" in table
    assert "| step/spp_001_ec_hd/flattop | 1.0 [0.5, 1.5]" in table
