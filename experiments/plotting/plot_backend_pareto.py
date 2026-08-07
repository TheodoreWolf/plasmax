"""CLI for plotting absolute CPU throughput against TORAX rollout error."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Literal

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import tyro
from matplotlib.ticker import PercentFormatter

plt.style.use(Path(__file__).resolve().parent / "paper.mplstyle")


@dataclasses.dataclass(frozen=True)
class Config:
    input: Path = Path("outputs/backend_agreement_cpu.json")
    output: Path = Path("plots/backend_agreement_mse.pdf")
    error_metric: Literal["input", "mse", "mre"] = "input"
    x_scale: Literal["log", "linear"] = "log"


_LABELS = {
    "cgm": "CGM",
    "bohm_gyrobohm": "Bohm–GyroBohm",
    "qlknn": "QLKNN",
    "tglfnn": "TGLFNN–linear",
    "tglfnn_nr": "TGLFNN-NR",
    "tglfnn_newton": "TGLFNN-NR",
}

_LABEL_OFFSETS = {
    "tglfnn": (6, 6),
    "tglfnn_nr": (6, 18),
    "tglfnn_newton": (6, 18),
    "qlknn": (6, -6),
    "cgm": (-6, 6),
    "bohm_gyrobohm": (-6, 6),
}

_MRE_LABEL_OFFSETS = {
    **_LABEL_OFFSETS,
}

_LABEL_HORIZONTAL_ALIGNMENT = {
    "cgm": "right",
    "bohm_gyrobohm": "right",
}


def _pareto_frontier(points: dict[str, dict[str, float]]) -> list[str]:
    frontier = []
    for name, point in points.items():
        dominated = any(
            other["error"] <= point["error"]
            and other["sps"] >= point["sps"]
            and (other["error"] < point["error"] or other["sps"] > point["sps"])
            for other_name, other in points.items()
            if other_name != name
        )
        if not dominated:
            frontier.append(name)
    return sorted(frontier, key=lambda name: points[name]["error"])


def main(cfg: Config) -> None:
    data = json.loads(cfg.input.read_text())
    error_metric = (
        data["error_metric"] if cfg.error_metric == "input" else cfg.error_metric
    )
    error_key = (
        "error"
        if error_metric == data["error_metric"]
        else f"sensor_balanced_{error_metric}"
    )
    points = {
        name: {
            "sps": point["sps"],
            "error": point[error_key],
            "error_std": point.get(f"{error_key}_std", point.get("error_std", 0.0)),
        }
        for name, point in data["metrics"].items()
        if point.get("agreement_available", True)
    }
    frontier = _pareto_frontier(points)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))

    frontier_points = sorted(
        ((points[name]["sps"], points[name]["error"]) for name in frontier),
        key=lambda point: point[0],
    )
    ax.plot(
        [point[0] for point in frontier_points],
        [point[1] for point in frontier_points],
        color="#5316d6",
        zorder=1,
    )

    for name, point in points.items():
        is_frontier = name in frontier
        if point["error_std"] > 0.0:
            ax.errorbar(
                point["sps"],
                point["error"],
                yerr=point["error_std"],
                fmt="none",
                ecolor="#5316d6" if is_frontier else "#999999",
                elinewidth=1.2,
                capsize=3,
                capthick=1.2,
                zorder=1,
            )
        ax.scatter(
            point["sps"],
            point["error"],
            s=48,
            marker="o" if is_frontier else "X",
            color="#5316d6" if is_frontier else "#999999",
            edgecolor="white",
            linewidth=0.8,
            zorder=2,
        )
        label_offsets = _MRE_LABEL_OFFSETS if error_metric == "mre" else _LABEL_OFFSETS
        x_offset, y_offset = label_offsets.get(name, (8, 8))
        ax.annotate(
            _LABELS[name],
            (point["sps"], point["error"]),
            xytext=(x_offset, y_offset),
            textcoords="offset points",
            ha=_LABEL_HORIZONTAL_ALIGNMENT.get(name, "left"),
            fontsize=10,
            path_effects=[path_effects.withStroke(linewidth=2.5, foreground="white")],
        )

    ax.set_xscale(cfg.x_scale)
    min_sps = min(point["sps"] for point in points.values())
    max_sps = max(point["sps"] for point in points.values())
    max_error = max(point["error"] + point["error_std"] for point in points.values())
    if cfg.x_scale == "log":
        ax.set_xlim(min_sps * 0.75, max_sps * 1.6)
    else:
        ax.set_xlim(-0.02 * max_sps, max_sps * 1.05)
    ax.set_ylim(-0.04 * max_error, max(0.01, max_error * 1.12))
    if error_metric == "mre":
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.set_xlabel("Steps/second")
    metric_label = error_metric.upper()
    ax.set_ylabel(f"Profile {metric_label}")

    cfg.output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(cfg.output)
    plt.close(fig)
    print(f"Saved {cfg.output}")


if __name__ == "__main__":
    main(tyro.cli(Config))
