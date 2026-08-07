"""CLI for plotting CPU and GPU backend Pareto measurements on shared axes."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, Literal

import matplotlib.lines as mlines
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import tyro
from matplotlib.ticker import PercentFormatter

plt.style.use(Path(__file__).resolve().parent / "paper.mplstyle")
plt.rcParams.update(
    {
        "font.size": 12,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
    }
)


@dataclasses.dataclass(frozen=True)
class Config:
    cpu_input: Path = Path("outputs/backend_agreement_flattop_cpu.json")
    gpu_input: Path = Path("outputs/backend_agreement_flattop_a100.json")
    output: Path = Path("plots/backend_agreement_mse.pdf")
    error_metric: Literal["mse", "mre"] = "mse"


@dataclasses.dataclass(frozen=True)
class Point:
    sps: float
    error: float


_BACKEND_ALIASES = {"tglfnn_newton": "tglfnn_nr"}
_LABELS = {
    "cgm": "CGM",
    "bohm_gyrobohm": "Bohm-GyroBohm",
    "qlknn": "QLKNN",
    "tglfnn": "TGLFNN-linear",
    "tglfnn_nr": "TGLFNN-NR",
}
_STYLE_COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]
_HARDWARE = {
    "CPU": {"color": _STYLE_COLORS[0], "marker": "o"},
    "GPU": {"color": _STYLE_COLORS[1], "marker": "s"},
}
_LABEL_OFFSETS = {
    "cgm": (-6, 8),
    "bohm_gyrobohm": (-6, 8),
    "qlknn": (6, 6),
    "tglfnn": (6, 8),
    "tglfnn_nr": (6, 6),
}
_LABEL_HORIZONTAL_ALIGNMENT = {
    "cgm": "right",
    "bohm_gyrobohm": "right",
}


def _canonical_backend(name: str) -> str:
    return _BACKEND_ALIASES.get(name, name)


def _load_points(data: dict[str, Any], error_metric: str) -> dict[str, Point]:
    error_key = f"sensor_balanced_{error_metric}"
    return {
        _canonical_backend(name): Point(
            sps=float(values["sps"]),
            error=float(values[error_key]),
        )
        for name, values in data["metrics"].items()
        if values.get("agreement_available", True)
    }


def _pareto_frontier(points: dict[str, Point]) -> list[str]:
    frontier = []
    for name, point in points.items():
        dominated = any(
            other.error <= point.error
            and other.sps >= point.sps
            and (other.error < point.error or other.sps > point.sps)
            for other_name, other in points.items()
            if other_name != name
        )
        if not dominated:
            frontier.append(name)
    return sorted(frontier, key=lambda name: points[name].sps)


def _validate_inputs(cpu: dict[str, Any], gpu: dict[str, Any]) -> None:
    required_throughput = "completed physical control intervals"
    for hardware, data in (("CPU", cpu), ("GPU", gpu)):
        definition = data.get("throughput_definition", "")
        if not definition.startswith(required_throughput):
            raise ValueError(
                f"{hardware} input predates fixed-duration macro-steps; "
                "rerun the throughput comparison"
            )
    for field in ("environment", "n_steps", "requested_seeds", "metric_sensors"):
        if cpu.get(field) != gpu.get(field):
            raise ValueError(
                f"CPU and GPU results differ in {field}: "
                f"{cpu.get(field)!r} != {gpu.get(field)!r}"
            )


def main(cfg: Config) -> None:
    cpu_data = json.loads(cfg.cpu_input.read_text())
    gpu_data = json.loads(cfg.gpu_input.read_text())
    _validate_inputs(cpu_data, gpu_data)

    hardware_points = {
        "CPU": _load_points(cpu_data, cfg.error_metric),
        "GPU": _load_points(gpu_data, cfg.error_metric),
    }
    if hardware_points["CPU"].keys() != hardware_points["GPU"].keys():
        raise ValueError("CPU and GPU results contain different backends")

    fig, ax = plt.subplots(figsize=(6.8, 4.4))

    for backend in hardware_points["CPU"]:
        cpu_point = hardware_points["CPU"][backend]
        gpu_point = hardware_points["GPU"][backend]
        ax.plot(
            [cpu_point.sps, gpu_point.sps],
            [cpu_point.error, gpu_point.error],
            color="#c7c7c7",
            linewidth=0.8,
            zorder=0,
        )

    for hardware, points in hardware_points.items():
        style = _HARDWARE[hardware]
        frontier = _pareto_frontier(points)
        frontier_points = [points[name] for name in frontier]
        ax.plot(
            [point.sps for point in frontier_points],
            [point.error for point in frontier_points],
            color=style["color"],
            linewidth=1.8,
            zorder=1,
        )
        for backend, point in points.items():
            on_frontier = backend in frontier
            ax.scatter(
                point.sps,
                point.error,
                s=44,
                marker=style["marker"],
                facecolor=style["color"] if on_frontier else "white",
                edgecolor=style["color"],
                linewidth=1.2,
                zorder=2,
            )

    for backend in hardware_points["CPU"]:
        cpu_point = hardware_points["CPU"][backend]
        ax.annotate(
            _LABELS[backend],
            (cpu_point.sps, cpu_point.error),
            xytext=_LABEL_OFFSETS[backend],
            textcoords="offset points",
            ha=_LABEL_HORIZONTAL_ALIGNMENT.get(backend, "left"),
            fontsize=11,
            path_effects=[path_effects.withStroke(linewidth=2.5, foreground="white")],
        )

    legend_handles = [
        mlines.Line2D(
            [],
            [],
            color=style["color"],
            marker=style["marker"],
            markersize=6,
            linewidth=1.8,
            label=hardware,
        )
        for hardware, style in _HARDWARE.items()
    ]
    ax.legend(handles=legend_handles, loc="upper left", frameon=False)

    all_points = [
        point for points in hardware_points.values() for point in points.values()
    ]
    min_sps = min(point.sps for point in all_points)
    max_sps = max(point.sps for point in all_points)
    max_error = max(point.error for point in all_points)
    ax.set_xscale("log")
    ax.set_xlim(min_sps * 0.65, max_sps * 1.35)
    ax.set_ylim(-0.04 * max_error, max(0.01, max_error * 1.12))
    if cfg.error_metric == "mre":
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.set_xlabel("Steps/second")
    ax.set_ylabel(f"Profile {cfg.error_metric.upper()}")

    cfg.output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(cfg.output)
    plt.close(fig)
    print(f"Saved {cfg.output}")


if __name__ == "__main__":
    main(tyro.cli(Config))
