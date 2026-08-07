"""Roll out a discharge scenario under a constant-actuator policy and plot it.

Rolls a plasmax phase environment forward under constant physical actuators,
then plots the current and scalar traces alongside profile snapshots at several
times.

Usage::

    uv run python scripts/rollout_discharge.py
    uv run python scripts/rollout_discharge.py \
        --env iter/hybrid/flattop \
        --backend cgm \
        --num-steps 4400 --out plots/discharge_rollout.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize

from plasmax import rollout as collect_lib
from plasmax.environment import factory as sc
from plasmax.wrappers import unwrap_to_env_state


def _constant_policy(action_norm: jax.Array):
    """Returns an ``act(obs, rng)`` that always emits ``action_norm``."""

    def act(obs, rng):  # noqa: ANN001 - collector policy signature
        del obs, rng
        return action_norm

    return act


def _hold_action_and_grids(env, key, phys_action):
    """Returns (normalised constant action, rho_cell, rho_face).

    ``phys_action`` is the physical actuator vector to hold for the whole
    rollout, in the env's actuator order. The rho grids come from this single
    (unstacked) reset state — the stacked rollout returns an inconsistently
    shaped cell grid.
    """
    state, _ = env.init(key)
    phys = env.unwrapped.action_space
    low = jnp.asarray(phys.low)
    high = jnp.asarray(phys.high)
    action_norm = 2.0 * (jnp.asarray(phys_action) - low) / (high - low) - 1.0
    geom = unwrap_to_env_state(state).plasma.geo
    return action_norm, np.asarray(geom.rho_norm), np.asarray(geom.rho_face_norm)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--env", default="iter/hybrid/flattop")
    p.add_argument("--backend", default="cgm")
    p.add_argument("--num-steps", type=int, default=4400)
    p.add_argument(
        "--reward",
        default=None,
        help="Reward override; omitted values inherit task YAML metadata.",
    )
    p.add_argument("--out", default="plots/discharge_rollout.png")
    p.add_argument(
        "--action",
        type=float,
        nargs=4,
        default=[15.0e6, 2.0e6, 0.45, 4.0e21],
        metavar=("P_nbi", "P_eccd", "rho_eccd", "gas_puff"),
        help="Constant physical actuator vector held for the whole rollout.",
    )
    p.add_argument(
        "--snapshots",
        type=float,
        nargs="+",
        default=[0.0, 50.0, 100.0, 200.0, 300.0, 440.0],
        help="Times [s] at which to draw profile snapshots.",
    )
    args = p.parse_args()

    env = sc.load_env(
        args.env, args.backend, reward=args.reward, max_steps=args.num_steps
    )
    key = jax.random.key(0)

    action_norm, rho, rho_face = _hold_action_and_grids(env, key, args.action)
    print(f"Rolling out {args.num_steps} steps (compile + scan)...")
    traj = collect_lib.collect_episode(
        _constant_policy(action_norm), env, key, args.num_steps
    )

    # Retain the ending transition and trim only fixed-scan padding.
    end = int(np.asarray(traj.valid).sum())
    traj = jax.tree_util.tree_map(lambda x: x[:end], traj)
    if end < args.num_steps:
        print(
            f"Episode ended at step {end} (disruption or t_final); "
            f"plotting first {end} steps."
        )

    es = unwrap_to_env_state(traj.env_state)
    ss, po = es.plasma.sim, es.plasma

    t = np.asarray(ss.t)  # (num_steps,), post-step times 0.1 .. t_final
    Ip = np.asarray(ss.core_profiles.Ip_profile_face[:, -1]) / 1e6  # edge = total [MA]
    P_fus = np.asarray(po.P_fusion) / 1e6  # [MW]
    q_min = np.asarray(po.q_min)
    fgw = np.asarray(po.fgw_n_e_line_avg)
    beta_N = np.asarray(po.beta_N)

    T_e = np.asarray(ss.core_profiles.T_e.value)  # (num_steps, n_rho)
    T_i = np.asarray(ss.core_profiles.T_i.value)
    n_e = np.asarray(ss.core_profiles.n_e.value)
    q = np.asarray(ss.core_profiles.q_face)  # (num_steps, n_rho+1)

    snap_idx = [int(np.argmin(np.abs(t - s))) for s in args.snapshots]
    norm = Normalize(vmin=t[0], vmax=t[-1])
    cmap = cm.viridis

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    pn, pe, re, gp = args.action
    fig.suptitle(
        f"Discharge-phase rollout — {Path(args.env).stem} / "
        f"{Path(args.backend).stem}\n"
        f"constant actuators: P_nbi={pn / 1e6:.0f} MW, P_eccd={pe / 1e6:.0f} MW, "
        f"rho_eccd={re:.2f}, gas_puff={gp:.1e} /s",
        fontsize=12,
    )

    # ---- Row 0: scalar time traces -----------------------------------------
    ax = axes[0, 0]
    ax.plot(t, Ip, color="tab:blue")
    ax.set(xlabel="t [s]", ylabel="Ip [MA]", title="Plasma current")
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(t, P_fus, color="tab:red")
    ax.set(xlabel="t [s]", ylabel="P_fusion [MW]", title="Fusion power")
    ax.grid(alpha=0.3)

    ax = axes[0, 2]
    ax.plot(t, q_min, color="tab:green", label="q_min")
    ax.plot(t, beta_N, color="tab:purple", label="β_N")
    ax.plot(t, fgw, color="tab:orange", label="n/n_GW")
    ax.set(xlabel="t [s]", title="Stability scalars")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)

    # ---- Row 1: profile snapshots ------------------------------------------
    def _plot_snaps(ax, xgrid, data, ylabel, title):
        for i in snap_idx:
            ax.plot(xgrid, data[i], color=cmap(norm(t[i])))
        ax.set(xlabel="ρ_norm", ylabel=ylabel, title=title)
        ax.grid(alpha=0.3)

    _plot_snaps(axes[1, 0], rho, T_e, "T_e [keV]", "Electron temperature")
    axes[1, 0].set_prop_cycle(None)
    for i in snap_idx:  # overlay T_i dashed
        axes[1, 0].plot(rho, T_i[i], color=cmap(norm(t[i])), ls="--", alpha=0.7)
    axes[1, 0].plot([], [], color="k", label="T_e (solid)")
    axes[1, 0].plot([], [], color="k", ls="--", label="T_i (dashed)")
    axes[1, 0].legend(loc="upper right", fontsize=8)

    _plot_snaps(axes[1, 1], rho, n_e / 1e19, "n_e [10¹⁹ m⁻³]", "Electron density")
    _plot_snaps(axes[1, 2], rho_face, q, "q", "Safety factor")

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes[1, :].tolist(), location="right", shrink=0.9)
    cbar.set_label("t [s]")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"Saved {out}")
    print(
        f"Final: t={t[-1]:.1f}s  Ip={Ip[-1]:.2f}MA  P_fus={P_fus[-1]:.1f}MW  "
        f"q_min={q_min[-1]:.2f}  n/n_GW={fgw[-1]:.3f}"
    )


if __name__ == "__main__":
    main()
