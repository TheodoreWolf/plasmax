"""Visualize an optimized open-loop discharge against the setpoint-hold baseline.

Loads the ``.npz`` written by ``experiments/studies/optimize_open_loop.py`` and
replays the optimized actuator schedule and constant setpoint-hold schedule
through the env. It plots scalar traces (with q_min / Greenwald limits
drawn in), the four actuator ramps, and the optimization learning curve.

Usage::

    uv run python experiments/plotting/plot_open_loop.py
    uv run python experiments/plotting/plot_open_loop.py \
        --npz outputs/open_loop_baseline.npz \
        --out plots/open_loop_discharge.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from plasmax.environment import factory as sc
from plasmax.wrappers import RealisticWrappers, unwrap_to_env_state


def _replay(env, key, actions_norm):
    """Roll a normalized schedule and retain only its first episode."""

    def body(carry, a):
        state, active = carry

        def step_active(_):
            next_state, info = env.step(state, a)
            plasma = unwrap_to_env_state(next_state).plasma
            boundary = info.terminated | info.truncated
            return (next_state, ~boundary), (
                plasma.sim,
                plasma,
                jnp.asarray(True),
            )

        def pad(_):
            plasma = unwrap_to_env_state(state).plasma
            return (state, jnp.asarray(False)), (
                plasma.sim,
                plasma,
                jnp.asarray(False),
            )

        return jax.lax.cond(active, step_active, pad, operand=None)

    state, _ = env.init(key)
    _, (ss, po, valid) = jax.lax.scan(body, (state, jnp.asarray(True)), actions_norm)
    return ss, po, np.asarray(valid)


def _scalars(ss, po, valid):
    """Extract scalar traces, excluding fixed-scan padding."""
    end = int(valid.sum())
    sl = slice(0, end)
    return {
        "t": np.asarray(ss.t)[sl],
        "Ip": np.asarray(ss.core_profiles.Ip_profile_face[:, -1])[sl] / 1e6,
        "P_fus": np.asarray(po.P_fusion)[sl] / 1e6,
        "P_aux": np.asarray(po.P_aux_total)[sl] / 1e6,
        "q_min": np.asarray(po.q_min)[sl],
        "fgw": np.asarray(po.fgw_n_e_line_avg)[sl],
        "beta_N": np.asarray(po.beta_N)[sl],
        "end": end,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--npz", default="outputs/open_loop_baseline.npz")
    p.add_argument("--out", default="plots/open_loop_discharge.png")
    args = p.parse_args()

    data = np.load(args.npz, allow_pickle=True)
    env_path = str(data["env"])
    backend_path = str(data["backend"])
    reward = str(data["reward"])
    opt_actions = jnp.asarray(data["actions_norm"])  # (T, A) normalized in [-1,1]
    num_steps = opt_actions.shape[0]

    env = RealisticWrappers(
        sc.make(env_path, backend_path, reward=reward), max_steps=num_steps
    )
    key = jax.random.key(0)

    # Setpoint-hold schedule: the env reset setpoint, normalized, held for the run.
    s0, _ = env.init(key)
    s0 = unwrap_to_env_state(s0)
    base = env.unwrapped
    low = jnp.asarray(base.action_space.low)
    high = jnp.asarray(base.action_space.high)
    setp_norm = 2.0 * (s0.prev_action - low) / (high - low) - 1.0
    setp_actions = jnp.broadcast_to(setp_norm, (num_steps, setp_norm.shape[0]))

    actuators = [spec.name for spec in base.actuator_specs]

    print(f"Replaying optimized + setpoint-hold ({num_steps} steps each)...")
    opt = _scalars(*_replay(env, key, opt_actions))
    base_run = _scalars(*_replay(env, key, setp_actions))

    opt_phys = np.asarray(env.to_physical(opt_actions))
    setp_phys = np.asarray(env.to_physical(setp_actions))

    base_cum = float(data["baseline_cum_reward"])
    opt_cum = float(data["optimized_cum_reward"])
    best_iter = int(data["best_iter"]) if "best_iter" in data else -1

    optimized_color, baseline_color = "tab:red", "tab:gray"
    fig, axes = plt.subplots(3, 4, figsize=(20, 12))
    fig.suptitle(
        f"Optimized open-loop discharge vs setpoint-hold — {Path(env_path).stem} / "
        f"{Path(backend_path).stem} / reward={reward}\n"
        f"cum_reward: optimized {opt_cum:+.2f}  vs  baseline {base_cum:+.2f}  "
        f"(Δ {opt_cum - base_cum:+.2f}, "
        f"{100 * (opt_cum - base_cum) / abs(base_cum):+.1f}%)"
        + (f"   [best iterate #{best_iter}]" if best_iter >= 0 else ""),
        fontsize=14,
    )

    def _trace(ax, ky, ylabel, title, hline=None, hlabel=None):
        ax.plot(
            base_run["t"],
            base_run[ky],
            color=baseline_color,
            ls="--",
            lw=1.5,
            label="setpoint-hold",
        )
        ax.plot(opt["t"], opt[ky], color=optimized_color, lw=2, label="optimized")
        if hline is not None:
            ax.axhline(hline, color="k", ls=":", lw=1.2, label=hlabel)
        ax.set(xlabel="t [s]", ylabel=ylabel, title=title)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="best")

    # Row 0: stability-relevant scalars, with disruption limits drawn in.
    _trace(axes[0, 0], "Ip", "Ip [MA]", "Plasma current")
    _trace(axes[0, 1], "P_fus", "P_fusion [MW]", "Fusion power")
    _trace(
        axes[0, 2],
        "q_min",
        "q_min",
        "Safety factor (min)",
        hline=0.8,
        hlabel="disruption (0.8)",
    )
    _trace(
        axes[0, 3],
        "fgw",
        "n / n_GW",
        "Greenwald fraction",
        hline=1.1,
        hlabel="disruption (1.1)",
    )

    # Row 1: actuator schedules (optimized ramp vs constant setpoint).
    t_full = np.asarray(jnp.arange(num_steps))  # actuator index axis (pre-truncation)
    for j, name in enumerate(actuators[:4]):
        ax = axes[1, j]
        ax.plot(
            t_full,
            setp_phys[:, j],
            color=baseline_color,
            ls="--",
            lw=1.5,
            label="setpoint-hold",
        )
        ax.plot(t_full, opt_phys[:, j], color=optimized_color, lw=2, label="optimized")
        ax.set(xlabel="step", ylabel=name, title=f"Actuator: {name}")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="best")

    # Row 2: net gain, beta_N, learning curve; 4th panel off.
    ax = axes[2, 0]
    ax.plot(
        base_run["t"],
        base_run["P_fus"] - base_run["P_aux"],
        color=baseline_color,
        ls="--",
        lw=1.5,
        label="setpoint-hold",
    )
    ax.plot(
        opt["t"],
        opt["P_fus"] - opt["P_aux"],
        color=optimized_color,
        lw=2,
        label="optimized",
    )
    ax.axhline(0.0, color="k", ls=":", lw=1)
    ax.set(
        xlabel="t [s]",
        ylabel="P_fus − P_aux [MW]",
        title="Net power (reward integrand)",
    )
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="best")

    _trace(axes[2, 1], "beta_N", "β_N", "Normalized beta")

    ax = axes[2, 2]
    if "history_iter" in data:
        ax.plot(
            np.asarray(data["history_iter"]),
            np.asarray(data["history_raw_cum"]),
            color=optimized_color,
        )
        if best_iter >= 0:
            ax.axvline(best_iter, color="k", ls=":", lw=1, label=f"best #{best_iter}")
            ax.legend(fontsize=8)
    ax.set(xlabel="iteration", ylabel="cum_reward", title="Optimization learning curve")
    ax.grid(alpha=0.3)

    axes[2, 3].axis("off")

    fig.tight_layout(rect=(0, 0, 1, 0.95))  # leave room for the suptitle
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"Saved {out}")
    print(
        f"optimized final: Ip={opt['Ip'][-1]:.2f}MA  P_fus={opt['P_fus'][-1]:.1f}MW  "
        f"q_min={opt['q_min'][-1]:.2f}  n/n_GW={opt['fgw'][-1]:.3f}  "
        f"alive={opt['end']}/{num_steps}"
    )


if __name__ == "__main__":
    main()
