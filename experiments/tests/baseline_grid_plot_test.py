"""Contracts for the canonical baseline learning-curve grid."""

import dataclasses
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from experiments.plotting.plot_baseline_grid import (
    _ALGORITHM_COLORS,
    _ALGORITHM_ORDER,
    CanonicalCurve,
    CurvePoint,
    _bootstrap_curve,
    _filter_manifest,
    _horizon_audit,
    _plot_grid,
)
from experiments.studies.summarize_baseline_final_returns import Cell, ExpectedCell


def _cell(algorithm: str, variant: str = "oracle") -> Cell:
    return Cell(
        algorithm=algorithm,
        env="step",
        backend="bohm_gyrobohm",
        variant=variant,
        reward="P_diff",
    )


def _curve(algorithm: str, final_step: int = 10_000_000) -> CanonicalCurve:
    values = tuple(float(seed) for seed in range(10))
    return CanonicalCurve(
        cell=_cell(algorithm),
        study="study",
        run_id="run",
        train_steps=(0, final_step),
        seed_returns=(values, tuple(value + 1.0 for value in values)),
    )


def _point(algorithm: str, variant: str, step: int) -> CurvePoint:
    value = float(_ALGORITHM_ORDER.index(algorithm))
    return CurvePoint(
        environment="step",
        variant=variant,
        algorithm=algorithm,
        backend="bohm_gyrobohm",
        reward="P_diff",
        train_steps=step,
        n_seeds=10,
        return_mean=value,
        ci95_low=value - 0.2,
        ci95_high=value + 0.2,
        ci_method="paired_seed_percentile_bootstrap",
        bootstrap_samples=100,
        bootstrap_seed=0,
        study="study",
        run_id="run",
    )


def test_paired_seed_bootstrap_is_deterministic():
    curve = _curve("ppo")

    first = _bootstrap_curve(curve, samples=10_000, seed=7)
    second = _bootstrap_curve(curve, samples=10_000, seed=7)

    for first_value, second_value in zip(first, second, strict=True):
        np.testing.assert_array_equal(first_value, second_value)
    np.testing.assert_allclose(first[0], [4.5, 5.5], rtol=1e-7, atol=0.0)


def test_horizon_audit_retains_short_knot_endpoints():
    curves = [
        _curve(algorithm, 1_000_000 if "knots" in algorithm else 10_000_000)
        for algorithm in _ALGORITHM_ORDER
    ]

    audits = _horizon_audit(curves, target_steps=10_000_000)
    by_algorithm = {audit.algorithm: audit for audit in audits}

    assert by_algorithm["ppo"].cells_below_target == 0
    assert by_algorithm["direct_knots_10"].cells_below_target == 1
    assert by_algorithm["direct_knots_10"].max_final_steps == 1_000_000


def test_grid_uses_algorithm_colors_variant_styles_and_bands(tmp_path: Path):
    points = [
        _point(algorithm, variant, step)
        for algorithm in _ALGORITHM_ORDER
        for variant in ("oracle", "realistic")
        for step in (0, 10_000_000)
    ]
    output_png = tmp_path / "grid.png"
    output_pdf = tmp_path / "grid.pdf"

    figure = _plot_grid(
        points,
        env_order=["step"],
        output_png=output_png,
        output_pdf=output_pdf,
        dpi=72,
    )

    axis = figure.axes[0]
    assert axis.get_title() == "STEP"
    assert len(axis.lines) == 12
    assert len(axis.collections) == 12
    ppo_lines = axis.lines[:2]
    assert {line.get_color() for line in ppo_lines} == {_ALGORITHM_COLORS["ppo"]}
    assert {line.get_linestyle() for line in ppo_lines} == {"--", "-"}
    assert len(figure.legends) == 1
    assert output_png.stat().st_size > 0
    assert output_pdf.stat().st_size > 0
    plt.close(figure)


def test_filter_manifest_selects_requested_environments_and_variant():
    environments = [
        "iter/hybrid/flattop",
        "iter/hybrid/rampdown",
        "step",
    ]
    expected = {
        Cell(
            algorithm=algorithm,
            env=environment,
            backend="bohm_gyrobohm",
            variant=variant,
            reward="P_diff",
        ): ExpectedCell(total_steps=10_000_000, num_seeds=10)
        for environment in environments
        for variant in ("oracle", "realistic")
        for algorithm in _ALGORITHM_ORDER
    }

    filtered, selected_envs, selected_variants = _filter_manifest(
        expected,
        environments,
        environments=("iter/hybrid/rampdown", "iter/hybrid/flattop"),
        variants=("realistic",),
    )

    assert selected_envs == ["iter/hybrid/rampdown", "iter/hybrid/flattop"]
    assert selected_variants == ("realistic",)
    assert len(filtered) == 12
    assert {cell.variant for cell in filtered} == {"realistic"}
    assert {cell.env for cell in filtered} == set(selected_envs)


def test_grid_can_plot_realistic_only_without_variant_legend(tmp_path: Path):
    points = [
        dataclasses.replace(
            _point(algorithm, "realistic", step),
            environment="iter/hybrid/flattop",
        )
        for algorithm in _ALGORITHM_ORDER
        for step in (0, 10_000_000)
    ]
    output_png = tmp_path / "grid.png"
    output_pdf = tmp_path / "grid.pdf"

    figure = _plot_grid(
        points,
        env_order=["iter/hybrid/flattop"],
        output_png=output_png,
        output_pdf=output_pdf,
        dpi=72,
        variants=("realistic",),
        short_titles=True,
    )

    axis = figure.axes[0]
    assert axis.get_title() == "Flat-top"
    assert len(axis.lines) == 6
    assert len(axis.collections) == 6
    assert {line.get_linestyle() for line in axis.lines} == {"-"}
    legend_labels = {text.get_text() for text in figure.legends[0].get_texts()}
    assert "Oracle" not in legend_labels
    assert "Realistic" not in legend_labels
    assert "95% bootstrap CI" in legend_labels
    assert output_png.stat().st_size > 0
    assert output_pdf.stat().st_size > 0
    plt.close(figure)
