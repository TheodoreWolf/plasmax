"""Summarize canonical 10M baseline final returns with bootstrap intervals.

The lossless combined return table intentionally retains pilots, superseded
runs, and corrected reruns as separate provenance. This script selects one
canonical run for every cell in ``sweeps/baseline_v1.tsv`` before computing a
non-parametric percentile interval across independent training seeds.
"""

from __future__ import annotations

import csv
import dataclasses
import hashlib
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import tyro

_ALGORITHM_LABELS = {
    "ppo": "PPO",
    "sac": "SAC",
    "direct_policy": "Direct policy",
    "direct_knots_1": "Actions (1 knot)",
    "direct_knots_10": "Actions (10 knots)",
    "direct_knots_100": "Actions (100 knots)",
}

# Earlier studies remain in the lossless source table. These priorities select
# the publication run for each cell without ever pooling repeated seed IDs.
_STUDY_PRIORITY = {
    "ppo": (
        "baseline-v1-policy-realistic-retry",
        "baseline-v1-policy",
    ),
    "sac": (
        "baseline-v1-policy-realistic-retry",
        "baseline-v1-policy",
    ),
    "direct_policy": (
        "baseline-v1-direct-backoff-recovery",
        "baseline-v1-direct-production",
        "baseline-long-iter-advanced-flattop-direct-policy-lr1e6",
    ),
    "direct_knots_1": (
        "baseline-v1-knots-10m-production",
        "baseline-v1-direct-production",
        "baseline-knot-tbptt-pilot",
    ),
    "direct_knots_10": (
        "baseline-v1-knots-10m-production",
        "baseline-v1-direct-production",
        "baseline-knot-tbptt-pilot",
    ),
    "direct_knots_100": (
        "baseline-v1-knots-10m-production",
        "baseline-v1-direct-production",
        "baseline-knot-tbptt-pilot",
    ),
}


@dataclasses.dataclass
class Args:
    returns_csv: str = "outputs/baseline_combined/baseline_returns.csv"
    manifest: str = "sweeps/baseline_v1_10m.tsv"
    output_csv: str = "outputs/baseline_combined/baseline_final_returns_bootstrap.csv"
    output_markdown: str = (
        "outputs/baseline_combined/baseline_final_returns_bootstrap.md"
    )
    bootstrap_samples: int = 100_000
    bootstrap_seed: int = 20_260_719
    precision: int = 3


@dataclasses.dataclass(frozen=True, order=True)
class Cell:
    algorithm: str
    env: str
    backend: str
    variant: str
    reward: str


@dataclasses.dataclass(frozen=True)
class ExpectedCell:
    total_steps: int
    num_seeds: int


@dataclasses.dataclass(frozen=True)
class Observation:
    cell: Cell
    study: str
    run_id: str
    train_steps: int
    seed: int
    seed_index: int
    return_mean: float


@dataclasses.dataclass(frozen=True)
class SelectedRun:
    cell: Cell
    study: str
    run_id: str
    train_steps: int
    values: tuple[float, ...]


@dataclasses.dataclass(frozen=True)
class SummaryRow:
    environment: str
    variant: str
    algorithm: str
    backend: str
    reward: str
    train_steps: int
    n_seeds: int
    metric: str
    return_mean: float
    ci95_low: float
    ci95_high: float
    ci_method: str
    bootstrap_samples: int
    bootstrap_seed: int
    study: str
    run_id: str


def _read_manifest(path: Path) -> tuple[dict[Cell, ExpectedCell], list[str]]:
    expected: dict[Cell, ExpectedCell] = {}
    env_order: list[str] = []
    with path.open(newline="") as manifest_file:
        for row in csv.DictReader(manifest_file, delimiter="\t"):
            cell = Cell(
                algorithm=row["algorithm"],
                env=row["env"],
                backend=row["backend"],
                variant=row["variant"],
                reward=row["reward"],
            )
            if cell in expected:
                raise ValueError(f"duplicate cell in manifest: {cell}")
            expected[cell] = ExpectedCell(
                total_steps=int(row["total_steps"]),
                num_seeds=int(row["num_seeds"]),
            )
            if cell.env not in env_order:
                env_order.append(cell.env)
    if not expected:
        raise ValueError(f"empty baseline manifest: {path}")
    return expected, env_order


def _read_observations(path: Path) -> list[Observation]:
    observations: list[Observation] = []
    with path.open(newline="") as returns_file:
        for row in csv.DictReader(returns_file):
            observations.append(
                Observation(
                    cell=Cell(
                        algorithm=row["algorithm"],
                        env=row["env"],
                        backend=row["backend"],
                        variant=row["variant"],
                        reward=row["reward"],
                    ),
                    study=row["study"],
                    run_id=row["run_id"],
                    train_steps=int(row["train_steps"]),
                    seed=int(row["seed"]),
                    seed_index=int(row["seed_index"]),
                    return_mean=float(row["evaluation/return_mean"]),
                )
            )
    if not observations:
        raise ValueError(f"empty return table: {path}")
    return observations


def _select_canonical_runs(
    observations: list[Observation],
    expected: dict[Cell, ExpectedCell],
) -> list[SelectedRun]:
    grouped: dict[Cell, dict[tuple[str, str], list[Observation]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in observations:
        if row.cell in expected:
            grouped[row.cell][(row.study, row.run_id)].append(row)

    selected: list[SelectedRun] = []
    for cell, contract in expected.items():
        priorities = _STUDY_PRIORITY.get(cell.algorithm)
        if priorities is None:
            raise ValueError(f"no study priority for {cell.algorithm}")
        candidates = grouped.get(cell, {})
        ranked = [
            (priorities.index(study), study, run_id, rows)
            for (study, run_id), rows in candidates.items()
            if study in priorities
        ]
        if not ranked:
            raise ValueError(f"no canonical run candidate for {cell}")
        best_rank = min(item[0] for item in ranked)
        best = [item for item in ranked if item[0] == best_rank]
        if len(best) != 1:
            identities = [(item[1], item[2]) for item in best]
            raise ValueError(f"ambiguous canonical runs for {cell}: {identities}")
        _, study, run_id, rows = best[0]
        train_steps = max(row.train_steps for row in rows)
        if train_steps < contract.total_steps:
            raise ValueError(
                f"canonical run for {cell} ends at {train_steps}, "
                f"before required {contract.total_steps}"
            )
        final_rows = [row for row in rows if row.train_steps == train_steps]
        by_seed = {row.seed: row.return_mean for row in final_rows}
        if len(by_seed) != len(final_rows):
            raise ValueError(f"duplicate final seed in {study}/{run_id}")
        if len(by_seed) != contract.num_seeds:
            raise ValueError(
                f"canonical run for {cell} has {len(by_seed)} final seeds, "
                f"expected {contract.num_seeds}"
            )
        expected_seeds = set(range(contract.num_seeds))
        if set(by_seed) != expected_seeds:
            raise ValueError(
                f"canonical run for {cell} has seed IDs {sorted(by_seed)}, "
                f"expected {sorted(expected_seeds)}"
            )
        seed_indexes = {row.seed_index for row in final_rows}
        if seed_indexes != expected_seeds or any(
            row.seed != row.seed_index for row in final_rows
        ):
            raise ValueError(f"canonical run for {cell} has inconsistent seed indexes")
        values = tuple(by_seed[seed] for seed in range(contract.num_seeds))
        if not np.isfinite(values).all():
            raise ValueError(f"non-finite final return in {study}/{run_id}")
        selected.append(
            SelectedRun(
                cell=cell,
                study=study,
                run_id=run_id,
                train_steps=train_steps,
                values=values,
            )
        )
    return selected


def _bootstrap_seed(global_seed: int, cell: Cell) -> int:
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


def _bootstrap_mean_ci(
    values: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    if values.ndim != 1 or values.size < 2:
        raise ValueError("bootstrap values must be a 1D array with at least 2 seeds")
    if samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(samples, values.size))
    bootstrap_means = values[indices].mean(axis=1)
    low, high = np.quantile(bootstrap_means, (0.025, 0.975))
    return float(low), float(high)


def _summarize(
    selected: list[SelectedRun],
    *,
    samples: int,
    seed: int,
) -> list[SummaryRow]:
    rows: list[SummaryRow] = []
    for run in selected:
        values = np.asarray(run.values, dtype=np.float64)
        low, high = _bootstrap_mean_ci(
            values,
            samples=samples,
            seed=_bootstrap_seed(seed, run.cell),
        )
        rows.append(
            SummaryRow(
                environment=run.cell.env,
                variant=run.cell.variant,
                algorithm=run.cell.algorithm,
                backend=run.cell.backend,
                reward=run.cell.reward,
                train_steps=run.train_steps,
                n_seeds=values.size,
                metric="evaluation/return_mean",
                return_mean=float(values.mean()),
                ci95_low=low,
                ci95_high=high,
                ci_method="percentile_bootstrap",
                bootstrap_samples=samples,
                bootstrap_seed=seed,
                study=run.study,
                run_id=run.run_id,
            )
        )
    return rows


def _write_csv(rows: list[SummaryRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = tuple(field.name for field in dataclasses.fields(SummaryRow))
    with path.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(dataclasses.asdict(row) for row in rows)


def _format_interval(row: SummaryRow, precision: int) -> str:
    return (
        f"{row.return_mean:.{precision}f} "
        f"[{row.ci95_low:.{precision}f}, {row.ci95_high:.{precision}f}]"
    )


def _markdown_table(
    rows: list[SummaryRow],
    *,
    variant: str,
    env_order: list[str],
    precision: int,
) -> str:
    algorithms = tuple(_ALGORITHM_LABELS)
    by_key = {
        (row.environment, row.algorithm): row for row in rows if row.variant == variant
    }
    header = (
        "| Environment | "
        + " | ".join(_ALGORITHM_LABELS[algorithm] for algorithm in algorithms)
        + " |"
    )
    divider = "|---|" + "|".join("---:" for _ in algorithms) + "|"
    lines = [header, divider]
    for env in env_order:
        cells = []
        for algorithm in algorithms:
            row = by_key.get((env, algorithm))
            if row is None:
                raise ValueError(f"missing {variant} summary for {env}/{algorithm}")
            cells.append(_format_interval(row, precision))
        lines.append(f"| {env} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _write_markdown(
    rows: list[SummaryRow],
    path: Path,
    *,
    env_order: list[str],
    samples: int,
    seed: int,
    precision: int,
) -> None:
    variants = ("oracle", "realistic")
    sections = [
        "# Baseline-v1 final evaluation returns",
        "",
        (
            "Each cell is the final-checkpoint `evaluation/return_mean` as mean "
            "[pointwise 95% percentile bootstrap CI] over 10 independent "
            f"training seeds ({samples:,} resamples, deterministic seed {seed}). "
            "Canonical corrected runs replace superseded runs; repeated seeds are "
            "never pooled. Every algorithm is required to reach at least 10M "
            "counted training simulator transitions."
        ),
    ]
    for variant in variants:
        sections.extend(
            (
                "",
                f"## {variant.capitalize()}",
                "",
                _markdown_table(
                    rows,
                    variant=variant,
                    env_order=env_order,
                    precision=precision,
                ),
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(sections) + "\n")


def main(args: Args) -> None:
    if args.precision < 0:
        raise ValueError("precision must be non-negative")
    expected, env_order = _read_manifest(Path(args.manifest))
    observations = _read_observations(Path(args.returns_csv))
    selected = _select_canonical_runs(observations, expected)
    rows = _summarize(
        selected,
        samples=args.bootstrap_samples,
        seed=args.bootstrap_seed,
    )
    if len(rows) != len(expected):
        raise AssertionError(f"selected {len(rows)} rows for {len(expected)} cells")
    _write_csv(rows, Path(args.output_csv))
    _write_markdown(
        rows,
        Path(args.output_markdown),
        env_order=env_order,
        samples=args.bootstrap_samples,
        seed=args.bootstrap_seed,
        precision=args.precision,
    )
    variants = {row.variant for row in rows}
    algorithms = {row.algorithm for row in rows}
    environments = {row.environment for row in rows}
    expected_count = len(variants) * len(algorithms) * len(environments)
    if len(rows) != expected_count:
        raise ValueError(
            f"summary is not a full rectangular matrix: {len(rows)} != "
            f"{len(variants)}*{len(algorithms)}*{len(environments)}"
        )
    if not all(math.isfinite(row.return_mean) for row in rows):
        raise ValueError("summary contains non-finite means")
    print(
        f"Wrote {len(rows)} canonical cells across {len(environments)} "
        f"environments to {args.output_csv} and {args.output_markdown}"
    )


if __name__ == "__main__":
    main(tyro.cli(Args))
