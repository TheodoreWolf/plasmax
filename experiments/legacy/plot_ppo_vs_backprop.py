"""Export the PPO-vs-backprop-control head-to-head (wandb group
``ppo_vs_backprop_iter_hybrid_flattop``) to a CSV and a matplotlib figure.

Three arms, all on iter_hybrid_flattop + CGM + P_diff + disruption_penalty=-5:
  * ppo         -- PPO training curve (evaluation/return_mean vs. env steps,
                    full episode)
  * backprop    -- gradient-based open-loop optimization curve (raw_cum vs. Adam
                    iteration over the full 4400-step episode; plotted separately
                    because it is not on the same x-axis as PPO)
  * steadystate -- a single constant actuator setpoint, optimized cheaply then held
                    forward-only over the full 4400-step episode (directly
                    comparable to ppo)

Usage::

    uv run python experiments/legacy/plot_ppo_vs_backprop.py
    uv run python experiments/legacy/plot_ppo_vs_backprop.py \
        --out-csv outputs/foo.csv --out-png plots/foo.png
"""

from __future__ import annotations

import csv
import dataclasses
from pathlib import Path

import matplotlib.pyplot as plt
import tyro
import wandb

GROUP = "ppo_vs_backprop_iter_hybrid_flattop"
PROJECT = "flair/plasmax"


@dataclasses.dataclass
class Args:
    out_csv: str = "outputs/ppo_vs_backprop.csv"
    out_png: str = "plots/ppo_vs_backprop.png"


def fetch():
    api = wandb.Api()
    runs = api.runs(PROJECT, filters={"group": GROUP})

    ppo, backprop, steadystate = {}, {}, {}
    for run in runs:
        if run.state != "finished":
            continue
        if run.name.startswith("iter_hybrid_flattop"):
            seed = run.config.get("seed")
            hist = run.scan_history(keys=["_step", "evaluation/return_mean"])
            ppo[seed] = [(h["_step"], h["evaluation/return_mean"]) for h in hist]
        elif run.name.startswith("backprop-"):
            seed = int(run.name.rsplit("seed", 1)[1])
            hist = run.scan_history(keys=["iter", "raw_cum"])
            backprop[seed] = [(h["iter"], h["raw_cum"]) for h in hist]
        elif run.name.startswith("steadystate-eval"):
            seed = len(steadystate)
            steadystate[seed] = (
                run.summary.get("final/cum_reward"),
                run.summary.get("final/alive_steps"),
            )
    return ppo, backprop, steadystate


def write_csv(path: str, ppo, backprop, steadystate) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "seed", "x", "x_units", "y", "y_units"])
        for seed, points in ppo.items():
            for x, y in points:
                w.writerow(["ppo", seed, x, "env_step", y, "return_mean"])
        for seed, points in backprop.items():
            for x, y in points:
                w.writerow(
                    ["backprop_open_loop", seed, x, "adam_iter", y, "raw_cum_reward"]
                )
        for seed, (cum_reward, alive) in steadystate.items():
            w.writerow(
                [
                    "steadystate_hold",
                    seed,
                    alive,
                    "episode_step",
                    cum_reward,
                    "cum_reward",
                ]
            )
    print(f"wrote {out}")


def make_plot(path: str, ppo, backprop, steadystate) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    ppo_color = "#2a78d6"
    backprop_color = "#d95f02"
    steadystate_color = "#1baf7a"

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 8))

    for _seed, points in sorted(ppo.items()):
        xs, ys = zip(*sorted(points), strict=True)
        ax1.plot(xs, ys, color=ppo_color, alpha=0.85, lw=1.5, label=None)
    ax1.plot([], [], color=ppo_color, lw=1.5, label=f"PPO ({len(ppo)} seeds)")

    if steadystate:
        ss_mean = sum(v[0] for v in steadystate.values()) / len(steadystate)
        ax1.axhline(
            ss_mean,
            color=steadystate_color,
            lw=2,
            ls="--",
            label=f"steady-state hold: {ss_mean:.0f}",
        )

    ax1.set_xlabel("environment steps")
    ax1.set_ylabel("evaluation return_mean")
    ax1.legend(frameon=False, fontsize=9)
    ax1.xaxis.set_major_formatter(lambda x, _: f"{x / 1e6:.0f}M")
    ax1.set_title("Closed-loop PPO evaluation")

    for i, (_seed, points) in enumerate(sorted(backprop.items())):
        xs, ys = zip(*sorted(points), strict=True)
        ax2.plot(
            xs,
            ys,
            color=backprop_color,
            alpha=0.85,
            lw=1.5,
            label=f"open-loop backprop ({len(backprop)} seeds)" if i == 0 else None,
        )
    if backprop:
        ax2.legend(frameon=False, fontsize=9)
    else:
        ax2.text(
            0.5,
            0.5,
            "No finished backprop runs found",
            ha="center",
            va="center",
            transform=ax2.transAxes,
        )
    ax2.set_xlabel("Adam iteration")
    ax2.set_ylabel("raw cumulative reward")
    ax2.set_title("Open-loop trajectory optimization")

    fig.suptitle(
        "PPO vs. backprop control -- ITER hybrid flattop"
        " (CGM, P_diff, disruption_penalty=-5)",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"wrote {out}")


def main() -> None:
    args = tyro.cli(Args)
    ppo, backprop, steadystate = fetch()
    print(
        f"ppo seeds: {len(ppo)}  backprop seeds: {len(backprop)}  "
        f"steadystate seeds: {len(steadystate)}"
    )
    write_csv(args.out_csv, ppo, backprop, steadystate)
    make_plot(args.out_png, ppo, backprop, steadystate)


if __name__ == "__main__":
    main()
