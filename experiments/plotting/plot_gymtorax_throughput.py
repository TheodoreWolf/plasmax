"""Generate the plasmax versus Gym-TORAX environment-throughput plot.

Data source: outputs/gymtorax_throughput_comparison.csv
"""

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator

from scripts.project_paths import OUTPUTS_DIR, PLOTS_DIR

plt.style.use(Path(__file__).resolve().parent / "paper.mplstyle")

LABELS = {
    ("plasmax - cpu", "apple_m_series"): "plasmax – CPU (M3 Pro)",
    ("plasmax - cpu", "apple_m3_pro"): "plasmax – CPU (M3 Pro)",
    ("plasmax - gpu", "a100_orchid"): "plasmax – GPU (A100)",
    ("plasmax - gpu", "gh200_isambard"): "plasmax – GPU (GH200)",
}

series: dict[str, list[tuple[int, float]]] = defaultdict(list)
gymtorax_sps = None
with (OUTPUTS_DIR / "gymtorax_throughput_comparison.csv").open() as f:
    for row in csv.DictReader(f):
        if row["status"] != "ok" or row["n_envs"] == "8192":
            continue
        series_name = row["series"].lower()
        if series_name.startswith("gym-torax"):
            gymtorax_sps = float(row["sps"])
            continue
        label = LABELS[(series_name, row["hardware"])]
        series[label].append((int(row["n_envs"]), float(row["sps"])))

fig, ax = plt.subplots(figsize=(5, 3.5))

markers = {"CPU": "o", "A100": "s", "GH200": "^"}
for label, points in series.items():
    points.sort()
    marker = next(m for key, m in markers.items() if key in label)
    ax.plot(*zip(*points, strict=True), marker=marker, label=label)

ax.axhline(gymtorax_sps, linestyle="--", color="#999999", label="Gym-TORAX – CPU")

ax.set_xscale("log")
ax.set_yscale("log")
xticks = [1, 4, 16, 64, 256, 1024, 4096, 16384]
ax.xaxis.set_major_locator(FixedLocator(xticks))
ax.xaxis.set_minor_locator(NullLocator())


def fmt(value: float) -> str:
    return f"{value / 1000:.0f}k" if value >= 1000 else f"{value:g}"


ax.set_xticklabels([fmt(t) for t in xticks])
ax.yaxis.set_major_formatter(lambda v, _: fmt(v))
ax.yaxis.set_minor_locator(NullLocator())
ax.set_xlabel("Parallel environments")
ax.set_ylabel("Steps/second")
ax.grid(True, which="major")
ax.legend(loc="upper left", fontsize=8)

out = PLOTS_DIR / "gymtorax_throughput_comparison.pdf"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out)
fig.savefig(out.with_suffix(".png"))
print(f"saved {out}")
