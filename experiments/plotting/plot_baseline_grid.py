"""Generate a configurable grid of canonical 10M baseline learning curves.

The lossless return table contains pilots and superseded runs. This script
first applies the same canonical run selection as the endpoint summary, then
computes pointwise percentile-bootstrap intervals across the ten independent
training-seed trajectories. Seed resampling is paired across checkpoints so
the confidence bands do not acquire avoidable Monte Carlo jitter.
"""

from __future__ import annotations

import csv
import dataclasses
import hashlib
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np
import tyro

from experiments.studies.summarize_baseline_final_returns import (
    Cell,
    ExpectedCell,
    Observation,
    _read_manifest,
    _read_observations,
    _select_canonical_runs,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

plt.style.use(Path(__file__).resolve().parent / "paper.mplstyle")
plt.rcParams.update(
    {
        "font.size": 13,
        "axes.titlesize": 15,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 13,
    }
)

_ALGORITHM_ORDER = (
    "ppo",
    "sac",
    "direct_policy",
    "direct_knots_1",
    "direct_knots_10",
    "direct_knots_100",
)

_ALGORITHM_LABELS = {
    "ppo": "PPO",
    "sac": "SAC",
    "direct_policy": "Policy backprop",
    "direct_knots_1": "Backprop (1 knot)",
    "direct_knots_10": "Backprop (10 knots)",
    "direct_knots_100": "Backprop (100 knots)",
}

_ALGORITHM_COLORS = {
    "ppo": "#1f77b4",
    "sac": "#ff7f0e",
    "direct_policy": "#2ca02c",
    "direct_knots_1": "#d62728",
    "direct_knots_10": "#9467bd",
    "direct_knots_100": "#8c564b",
}

_VARIANT_STYLES = {
    "oracle": "--",
    "realistic": "-",
}

_PHASE_LABELS = {
    "flattop": "Flat-top",
    "rampdown": "Ramp-down",
    "rampup": "Ramp-up",
}

_SCENARIO_LABELS = {
    "advanced": "advanced",
    "baseline": "baseline",
    "hybrid": "hybrid",
    "prd": "PRD",
    "reduced_field": "reduced field",
}


@dataclasses.dataclass
class Args:
    returns_csv: str = "outputs/baseline_combined/baseline_returns.csv"
    manifest: str = "sweeps/baseline_v1_10m.tsv"
    output_csv: str = (
        "outputs/baseline_combined/baseline_canonical_learning_curves_bootstrap.csv"
    )
    output_horizon_csv: str = (
        "outputs/baseline_combined/baseline_training_horizon_audit.csv"
    )
    output_png: str = "plots/baseline_combined/baseline_learning_curves_grid.png"
    output_pdf: str = "plots/baseline_combined/baseline_learning_curves_grid.pdf"
    bootstrap_samples: int = 10_000
    bootstrap_seed: int = 20_260_719
    target_steps: int = 10_000_000
    dpi: int = 220
    max_columns: int = 4
    environments: tuple[str, ...] = ()
    variants: tuple[str, ...] = ()
    short_titles: bool = False


@dataclasses.dataclass(frozen=True)
class CanonicalCurve:
    cell: Cell
    study: str
    run_id: str
    train_steps: tuple[int, ...]
    seed_returns: tuple[tuple[float, ...], ...]


@dataclasses.dataclass(frozen=True)
class CurvePoint:
    environment: str
    variant: str
    algorithm: str
    backend: str
    reward: str
    train_steps: int
    n_seeds: int
    return_mean: float
    ci95_low: float
    ci95_high: float
    ci_method: str
    bootstrap_samples: int
    bootstrap_seed: int
    study: str
    run_id: str


@dataclasses.dataclass(frozen=True)
class HorizonAudit:
    algorithm: str
    n_cells: int
    min_final_steps: int
    max_final_steps: int
    cells_below_target: int
    target_steps: int


def _filter_manifest(
    expected: dict[Cell, ExpectedCell],
    env_order: list[str],
    *,
    environments: tuple[str, ...],
    variants: tuple[str, ...],
) -> tuple[dict[Cell, ExpectedCell], list[str], tuple[str, ...]]:
    """Select a complete environment/variant slice of the manifest."""

    selected_envs = list(environments) if environments else list(env_order)
    if len(selected_envs) != len(set(selected_envs)):
        raise ValueError("environments must not contain duplicates")
    unknown_envs = sorted(set(selected_envs) - set(env_order))
    if unknown_envs:
        raise ValueError(f"unknown environments: {unknown_envs}")

    selected_variants = variants or tuple(_VARIANT_STYLES)
    if len(selected_variants) != len(set(selected_variants)):
        raise ValueError("variants must not contain duplicates")
    unknown_variants = sorted(set(selected_variants) - set(_VARIANT_STYLES))
    if unknown_variants:
        raise ValueError(f"unknown variants: {unknown_variants}")

    selected_env_set = set(selected_envs)
    selected_variant_set = set(selected_variants)
    filtered = {
        cell: contract
        for cell, contract in expected.items()
        if cell.env in selected_env_set and cell.variant in selected_variant_set
    }
    if not filtered:
        raise ValueError("environment/variant filters selected no manifest cells")
    return filtered, selected_envs, selected_variants


def _select_canonical_curves(
    observations: list[Observation],
    expected: dict[Cell, ExpectedCell],
) -> list[CanonicalCurve]:
    """Select one complete seed-history run for each manifest cell."""

    selected_runs = _select_canonical_runs(observations, expected)
    identity_by_cell = {
        run.cell: (run.study, run.run_id, run.train_steps) for run in selected_runs
    }
    rows_by_cell: dict[Cell, list[Observation]] = defaultdict(list)
    for row in observations:
        identity = identity_by_cell.get(row.cell)
        if identity is not None and (row.study, row.run_id) == identity[:2]:
            rows_by_cell[row.cell].append(row)

    curves: list[CanonicalCurve] = []
    for cell, contract in expected.items():
        study, run_id, final_step = identity_by_cell[cell]
        by_step: dict[int, list[Observation]] = defaultdict(list)
        for row in rows_by_cell[cell]:
            by_step[row.train_steps].append(row)
        if not by_step:
            raise ValueError(f"canonical history is empty for {cell}")

        steps = tuple(sorted(by_step))
        if steps[-1] != final_step:
            raise ValueError(
                f"canonical history for {cell} ends at {steps[-1]}, "
                f"but selected endpoint is {final_step}"
            )
        expected_seeds = set(range(contract.num_seeds))
        values_by_step: list[tuple[float, ...]] = []
        for step in steps:
            checkpoint_rows = by_step[step]
            by_seed = {row.seed: row.return_mean for row in checkpoint_rows}
            if len(by_seed) != len(checkpoint_rows):
                raise ValueError(f"duplicate seed at step {step} in {study}/{run_id}")
            if set(by_seed) != expected_seeds:
                raise ValueError(
                    f"checkpoint {step} in {study}/{run_id} has seeds "
                    f"{sorted(by_seed)}, expected {sorted(expected_seeds)}"
                )
            if (
                any(row.seed_index != row.seed for row in checkpoint_rows)
                or {row.seed_index for row in checkpoint_rows} != expected_seeds
            ):
                raise ValueError(
                    f"checkpoint {step} in {study}/{run_id} has inconsistent "
                    "seed indexes"
                )
            values = tuple(by_seed[seed] for seed in range(contract.num_seeds))
            if not np.isfinite(values).all():
                raise ValueError(
                    f"checkpoint {step} in {study}/{run_id} has non-finite returns"
                )
            values_by_step.append(values)

        curves.append(
            CanonicalCurve(
                cell=cell,
                study=study,
                run_id=run_id,
                train_steps=steps,
                seed_returns=tuple(values_by_step),
            )
        )
    return curves


def _curve_seed(global_seed: int, cell: Cell) -> int:
    identity = "|".join(
        (
            str(global_seed),
            cell.algorithm,
            cell.env,
            cell.backend,
            cell.variant,
            cell.reward,
        )
    )
    digest = hashlib.sha256(identity.encode()).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False)


def _bootstrap_curve(
    curve: CanonicalCurve,
    *,
    samples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bootstrap seed trajectories and return pointwise mean/95% intervals."""

    values = np.asarray(curve.seed_returns, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("curve returns must have shape (checkpoints, seeds>=2)")
    if samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.shape[1], size=(samples, values.shape[1]))
    bootstrap_means = values[:, indices].mean(axis=2)
    low, high = np.quantile(bootstrap_means, (0.025, 0.975), axis=1)
    return values.mean(axis=1), low, high


def _summarize_curves(
    curves: list[CanonicalCurve],
    *,
    samples: int,
    seed: int,
) -> list[CurvePoint]:
    points: list[CurvePoint] = []
    for curve in curves:
        curve_seed = _curve_seed(seed, curve.cell)
        mean, low, high = _bootstrap_curve(
            curve,
            samples=samples,
            seed=curve_seed,
        )
        for index, train_steps in enumerate(curve.train_steps):
            points.append(
                CurvePoint(
                    environment=curve.cell.env,
                    variant=curve.cell.variant,
                    algorithm=curve.cell.algorithm,
                    backend=curve.cell.backend,
                    reward=curve.cell.reward,
                    train_steps=train_steps,
                    n_seeds=len(curve.seed_returns[index]),
                    return_mean=float(mean[index]),
                    ci95_low=float(low[index]),
                    ci95_high=float(high[index]),
                    ci_method="paired_seed_percentile_bootstrap",
                    bootstrap_samples=samples,
                    bootstrap_seed=seed,
                    study=curve.study,
                    run_id=curve.run_id,
                )
            )
    return points


def _horizon_audit(
    curves: list[CanonicalCurve],
    *,
    target_steps: int,
) -> list[HorizonAudit]:
    by_algorithm: dict[str, list[int]] = defaultdict(list)
    for curve in curves:
        by_algorithm[curve.cell.algorithm].append(curve.train_steps[-1])
    audits = []
    for algorithm in _ALGORITHM_ORDER:
        endpoints = by_algorithm.get(algorithm, [])
        if not endpoints:
            raise ValueError(f"no canonical curves for {algorithm}")
        audits.append(
            HorizonAudit(
                algorithm=algorithm,
                n_cells=len(endpoints),
                min_final_steps=min(endpoints),
                max_final_steps=max(endpoints),
                cells_below_target=sum(step < target_steps for step in endpoints),
                target_steps=target_steps,
            )
        )
    return audits


def _write_dataclasses(rows: list[object], path: Path) -> None:
    if not rows:
        raise ValueError(f"cannot write empty table to {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = tuple(field.name for field in dataclasses.fields(rows[0]))
    with path.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(dataclasses.asdict(row) for row in rows)


def _plot_grid(
    points: list[CurvePoint],
    *,
    env_order: list[str],
    output_png: Path,
    output_pdf: Path,
    dpi: int,
    max_columns: int = 4,
    variants: tuple[str, ...] | None = None,
    short_titles: bool = False,
) -> Figure:
    if not points:
        raise ValueError("cannot plot an empty curve table")
    if max_columns <= 0:
        raise ValueError("max_columns must be positive")
    selected_variants = variants or tuple(_VARIANT_STYLES)
    unknown_variants = sorted(set(selected_variants) - set(_VARIANT_STYLES))
    if unknown_variants:
        raise ValueError(f"unknown variants: {unknown_variants}")
    ncols = min(max_columns, len(env_order))
    nrows = math.ceil(len(env_order) / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5.4 * ncols + 0.5, 3.8 * nrows + 0.8),
        sharex=True,
        squeeze=False,
    )
    axes_flat = list(axes.flat)
    by_key: dict[tuple[str, str, str], list[CurvePoint]] = defaultdict(list)
    for point in points:
        by_key[(point.environment, point.algorithm, point.variant)].append(point)

    max_steps = max(point.train_steps for point in points)
    x_max_millions = math.ceil(max_steps / 100_000) / 10
    for axis, environment in zip(axes_flat, env_order, strict=False):
        for algorithm in _ALGORITHM_ORDER:
            for variant in selected_variants:
                linestyle = _VARIANT_STYLES[variant]
                rows = sorted(
                    by_key.get((environment, algorithm, variant), []),
                    key=lambda row: row.train_steps,
                )
                if not rows:
                    raise ValueError(
                        f"missing curve for {environment}/{algorithm}/{variant}"
                    )
                x = np.asarray([row.train_steps for row in rows]) / 1_000_000
                mean = np.asarray([row.return_mean for row in rows])
                low = np.asarray([row.ci95_low for row in rows])
                high = np.asarray([row.ci95_high for row in rows])
                color = _ALGORITHM_COLORS[algorithm]
                axis.fill_between(
                    x,
                    low,
                    high,
                    color=color,
                    alpha=0.09,
                    linewidth=0,
                    zorder=1,
                )
                axis.plot(
                    x,
                    mean,
                    color=color,
                    linestyle=linestyle,
                    linewidth=1.5,
                    zorder=2,
                )
        parts = environment.split("/")
        phase = _PHASE_LABELS.get(parts[-1], parts[-1].replace("_", " ").title())
        if short_titles:
            title = phase
        elif len(parts) == 3:
            device = parts[0].upper()
            scenario = _SCENARIO_LABELS.get(
                parts[1],
                parts[1].replace("_", " "),
            )
            title = f"{device} {scenario} — {phase}"
        else:
            title = environment.upper()
        axis.set_title(title, pad=8)
        axis.set_xlim(0, x_max_millions)
        axis.set_xticks(np.arange(0, 10.1, 2))
        axis.margins(y=0.08)

    handles: list[object] = [
        Line2D(
            [0],
            [0],
            color=_ALGORITHM_COLORS[algorithm],
            linewidth=2,
            label=_ALGORITHM_LABELS[algorithm],
        )
        for algorithm in _ALGORITHM_ORDER
    ]
    if len(selected_variants) > 1:
        handles.extend(
            Line2D(
                [0],
                [0],
                color="0.15",
                linestyle=_VARIANT_STYLES[variant],
                label=variant.title(),
            )
            for variant in selected_variants
        )
    handles.append(Patch(facecolor="0.35", alpha=0.15, label="95% bootstrap CI"))
    unused_axes = axes_flat[len(env_order) :]
    if unused_axes:
        legend_axis = unused_axes[0]
        legend_axis.axis("off")
        legend_axis.legend(
            handles=handles,
            loc="center",
            ncol=1,
            handlelength=2.8,
            labelspacing=0.9,
        )
        for axis in unused_axes[1:]:
            axis.set_visible(False)
    else:
        fig.legend(
            handles=handles,
            loc="lower center",
            ncol=min(4, len(handles)),
            handlelength=2.8,
            columnspacing=1.25,
        )
    fig.supxlabel("Training transitions (millions)", y=0.018)
    fig.supylabel("Episode return", x=0.018)
    fig.subplots_adjust(
        left=0.075,
        right=0.992,
        bottom=0.14 if nrows == 1 else 0.10,
        top=0.94,
        hspace=0.38,
        wspace=0.30,
    )
    output_png.parent.mkdir(parents=True, exist_ok=True)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=dpi, facecolor="white", bbox_inches="tight")
    fig.savefig(output_pdf, facecolor="white", bbox_inches="tight")
    return fig


def main(args: Args) -> None:
    if args.dpi <= 0:
        raise ValueError("dpi must be positive")
    expected, env_order = _read_manifest(Path(args.manifest))
    expected, env_order, variants = _filter_manifest(
        expected,
        env_order,
        environments=args.environments,
        variants=args.variants,
    )
    observations = _read_observations(Path(args.returns_csv))
    curves = _select_canonical_curves(observations, expected)
    points = _summarize_curves(
        curves,
        samples=args.bootstrap_samples,
        seed=args.bootstrap_seed,
    )
    audits = _horizon_audit(curves, target_steps=args.target_steps)
    _write_dataclasses(points, Path(args.output_csv))
    _write_dataclasses(audits, Path(args.output_horizon_csv))
    figure = _plot_grid(
        points,
        env_order=env_order,
        output_png=Path(args.output_png),
        output_pdf=Path(args.output_pdf),
        dpi=args.dpi,
        max_columns=args.max_columns,
        variants=variants,
        short_titles=args.short_titles,
    )
    plt.close(figure)

    below_target = sum(audit.cells_below_target for audit in audits)
    print(
        f"Plotted {len(curves)} canonical curves and {len(points)} checkpoints "
        f"across {len(env_order)} environments."
    )
    if below_target:
        print(
            f"WARNING: {below_target}/{len(curves)} curves end before "
            f"{args.target_steps:,} transitions; see {args.output_horizon_csv}."
        )


if __name__ == "__main__":
    main(tyro.cli(Args))
