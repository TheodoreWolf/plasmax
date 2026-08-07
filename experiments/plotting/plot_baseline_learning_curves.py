"""Combine baseline returns and plot return against simulator steps.

The confidence interval is computed across independent training-seed means,
not across evaluation episodes within a seed. For ten seeds this uses a
two-sided 95% Student-t interval with nine degrees of freedom.

Two CSV files are emitted:

* ``baseline_returns.csv`` is the lossless long-form table of recorded
  evaluation statistics, with one row per seed and training checkpoint.
* ``baseline_learning_curves.csv`` aggregates the per-seed mean returns into
  plot-ready means and confidence intervals.
"""

from __future__ import annotations

import csv
import dataclasses
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import tyro

_T975 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}

_LABELS = {
    "ppo": "PPO",
    "sac": "SAC",
    "direct_policy": "Direct policy",
    "direct_knots_1": "Direct actions (1 knot)",
    "direct_knots_10": "Direct actions (10 knots)",
    "direct_knots_100": "Direct actions (100 knots)",
}


@dataclasses.dataclass
class Args:
    root: str
    out_dir: str = "outputs/baseline_curves"
    plot_dir: str = "plots/baseline_curves"
    x_scale: Literal["linear", "log"] = "linear"
    dpi: int = 180


@dataclasses.dataclass(frozen=True)
class Observation:
    algorithm: str
    env: str
    backend: str
    variant: str
    reward: str
    train_steps: int
    seed: int
    return_mean: float
    seed_index: int | None = None
    study: str = ""
    run_id: str = ""
    series_label: str = ""
    source_metrics: str = ""
    evaluation: tuple[tuple[str, str], ...] = ()


@dataclasses.dataclass(frozen=True)
class Aggregate:
    algorithm: str
    study: str
    run_id: str
    series_label: str
    env: str
    backend: str
    variant: str
    reward: str
    train_steps: int
    n_seeds: int
    return_mean: float
    return_std: float
    return_sem: float
    ci95_low: float
    ci95_high: float


def _config_path(metrics_path: Path) -> Path:
    suffix = "_metrics.csv"
    if not metrics_path.name.endswith(suffix):
        raise ValueError(f"not a baseline metrics file: {metrics_path}")
    return metrics_path.with_name(
        metrics_path.name.removesuffix(suffix) + "_config.json"
    )


def _read_observations(root: Path) -> list[Observation]:
    observations: list[Observation] = []
    paths = sorted(root.rglob("*_metrics.csv"))
    if not paths:
        raise ValueError(f"no *_metrics.csv files found under {root}")
    for metrics_path in paths:
        config_path = _config_path(metrics_path)
        if not config_path.exists():
            raise ValueError(
                f"missing config paired with {metrics_path}: {config_path}"
            )
        config = json.loads(config_path.read_text())
        env_config = config["env"]
        algorithm = config["algorithm"]
        direct_config = config.get("direct", {})
        series_label = ""
        if algorithm == "direct_policy" and "policy_learning_rate" in direct_config:
            series_label = f"lr={float(direct_config['policy_learning_rate']):g}"
        with metrics_path.open(newline="") as metrics_file:
            for row in csv.DictReader(metrics_file):
                evaluation = tuple(
                    sorted(
                        (name, value)
                        for name, value in row.items()
                        if name.startswith("evaluation/") and value is not None
                    )
                )
                observations.append(
                    Observation(
                        algorithm=algorithm,
                        env=env_config["env_setup"],
                        backend=env_config["backend"],
                        variant=env_config["variant"],
                        reward=env_config["reward"],
                        train_steps=int(row["train_steps"]),
                        seed=int(row["seed"]),
                        return_mean=float(row["evaluation/return_mean"]),
                        seed_index=int(row.get("seed_index", row["seed"])),
                        study=config.get("study", ""),
                        run_id=metrics_path.parent.name,
                        series_label=series_label,
                        source_metrics=metrics_path.relative_to(root).as_posix(),
                        evaluation=evaluation,
                    )
                )
    return observations


def _aggregate(observations: list[Observation]) -> list[Aggregate]:
    grouped: dict[tuple, dict[int, float]] = defaultdict(dict)
    for row in observations:
        key = (
            row.algorithm,
            row.study,
            row.run_id,
            row.series_label,
            row.env,
            row.backend,
            row.variant,
            row.reward,
            row.train_steps,
        )
        if row.seed in grouped[key]:
            raise ValueError(
                f"duplicate seed {row.seed} for {key}; "
                "remove duplicate/restarted artifacts"
            )
        grouped[key][row.seed] = row.return_mean

    output: list[Aggregate] = []
    for key, by_seed in grouped.items():
        values = np.asarray(list(by_seed.values()), dtype=np.float64)
        n = values.size
        mean = float(values.mean())
        if n > 1:
            std = float(values.std(ddof=1))
            sem = std / math.sqrt(n)
            critical = _T975.get(n - 1, 1.96)
            half_width = critical * sem
        else:
            std = sem = half_width = float("nan")
        output.append(
            Aggregate(
                algorithm=key[0],
                study=key[1],
                run_id=key[2],
                series_label=key[3],
                env=key[4],
                backend=key[5],
                variant=key[6],
                reward=key[7],
                train_steps=key[8],
                n_seeds=n,
                return_mean=mean,
                return_std=std,
                return_sem=sem,
                ci95_low=mean - half_width,
                ci95_high=mean + half_width,
            )
        )
    return sorted(
        output,
        key=lambda row: (
            row.env,
            row.variant,
            row.algorithm,
            row.study,
            row.run_id,
            row.train_steps,
        ),
    )


def _write_aggregate(rows: list[Aggregate], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = tuple(field.name for field in dataclasses.fields(Aggregate))
    with path.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(dataclasses.asdict(row) for row in rows)


def _write_observations(rows: list[Observation], path: Path) -> None:
    """Write every recorded evaluation value without re-aggregating it."""

    path.parent.mkdir(parents=True, exist_ok=True)
    metadata_fields = (
        "algorithm",
        "study",
        "run_id",
        "series_label",
        "env",
        "backend",
        "variant",
        "reward",
        "train_steps",
        "seed",
        "seed_index",
        "source_metrics",
    )
    evaluation_fields = sorted(
        {name for row in rows for name, _ in row.evaluation}
        | {"evaluation/return_mean"}
    )
    with path.open("w", newline="") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=metadata_fields + tuple(evaluation_fields),
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            evaluation = dict(row.evaluation)
            evaluation.setdefault("evaluation/return_mean", row.return_mean)
            writer.writerow(
                {
                    "algorithm": row.algorithm,
                    "study": row.study,
                    "run_id": row.run_id,
                    "series_label": row.series_label,
                    "env": row.env,
                    "backend": row.backend,
                    "variant": row.variant,
                    "reward": row.reward,
                    "train_steps": row.train_steps,
                    "seed": row.seed,
                    "seed_index": (
                        row.seed if row.seed_index is None else row.seed_index
                    ),
                    "source_metrics": row.source_metrics,
                    **evaluation,
                }
            )


def _plot(rows: list[Aggregate], out_dir: Path, x_scale: str, dpi: int) -> None:
    by_env: dict[str, list[Aggregate]] = defaultdict(list)
    for row in rows:
        by_env[row.env].append(row)

    variant_order = {"oracle": 0, "realistic": 1}
    for env, env_rows in by_env.items():
        variants = sorted(
            {row.variant for row in env_rows},
            key=lambda variant: (variant_order.get(variant, 2), variant),
        )
        series = sorted(
            {
                (row.algorithm, row.study, row.run_id, row.series_label)
                for row in env_rows
            },
            key=lambda item: (
                list(_LABELS).index(item[0]) if item[0] in _LABELS else len(_LABELS),
                item[1],
                item[2],
            ),
        )
        algorithm_counts: dict[str, int] = defaultdict(int)
        for algorithm, _, _, _ in series:
            algorithm_counts[algorithm] += 1
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        series_colors = {
            item: colors[index % len(colors)] for index, item in enumerate(series)
        }
        fig, axes = plt.subplots(
            1,
            len(variants),
            figsize=(7.0 * len(variants), 5.0),
            sharey=True,
            squeeze=False,
            constrained_layout=True,
        )
        legend_handles: dict[str, object] = {}
        for axis, variant in zip(axes[0], variants, strict=True):
            task_rows = [row for row in env_rows if row.variant == variant]
            for series_key in series:
                algorithm, study, run_id, series_label = series_key
                method_rows = sorted(
                    (
                        row
                        for row in task_rows
                        if (
                            row.algorithm,
                            row.study,
                            row.run_id,
                            row.series_label,
                        )
                        == series_key
                    ),
                    key=lambda row: row.train_steps,
                )
                if not method_rows:
                    continue
                x = np.asarray([row.train_steps for row in method_rows])
                mean = np.asarray([row.return_mean for row in method_rows])
                low = np.asarray([row.ci95_low for row in method_rows])
                high = np.asarray([row.ci95_high for row in method_rows])
                label = _LABELS.get(algorithm, algorithm)
                if algorithm_counts[algorithm] > 1:
                    qualifier = series_label or study or run_id
                    label = f"{label} ({qualifier})"
                line = axis.plot(
                    x,
                    mean,
                    label=label,
                    color=series_colors[series_key],
                )[0]
                legend_handles[label] = line
                finite = np.isfinite(low) & np.isfinite(high)
                if np.any(finite):
                    axis.fill_between(
                        x,
                        low,
                        high,
                        where=finite,
                        color=line.get_color(),
                        alpha=0.18,
                    )
            if x_scale == "log":
                axis.set_xscale("symlog", linthresh=1.0)
            axis.set_xlabel("Training simulator transitions")
            axis.set_title(variant.capitalize())
            axis.grid(alpha=0.25)
        axes[0, 0].set_ylabel("Frozen-policy episode return")
        fig.suptitle(env)
        fig.legend(
            legend_handles.values(),
            legend_handles.keys(),
            loc="outside lower center",
            ncol=min(3, len(legend_handles)),
            fontsize=8,
        )
        name = f"{env.replace('/', '_')}.png"
        fig.savefig(out_dir / name, dpi=dpi, facecolor="white")
        plt.close(fig)


def main(args: Args) -> None:
    root = Path(args.root)
    out_dir = Path(args.out_dir)
    plot_dir = Path(args.plot_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    observations = _read_observations(root)
    aggregates = _aggregate(observations)
    _write_observations(observations, out_dir / "baseline_returns.csv")
    _write_aggregate(aggregates, out_dir / "baseline_learning_curves.csv")
    _plot(aggregates, plot_dir, args.x_scale, args.dpi)
    print(
        f"Aggregated {len(observations)} seed checkpoints into "
        f"{len(aggregates)} curve points under {out_dir}; plots under {plot_dir}"
    )


if __name__ == "__main__":
    main(tyro.cli(Args))
