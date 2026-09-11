"""Plot a saved MessagePack policy against a paired setpoint-hold episode.

    uv run python experiments/plotting/plot_open_loop.py \
        --policy outputs/policies/run.msgpack --out plots/open_loop_discharge.png
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import jax
import matplotlib.pyplot as plt
import numpy as np
import tyro

from agents.policy_io import load_policy
from plasmax.wrappers import unwrap_to_env_state
from training.evaluation import evaluate_policy, trajectory_arrays
from training.runs import load_policy_env


@dataclasses.dataclass(frozen=True)
class Config:
    policy: Path
    out: Path = Path("plots/open_loop_discharge.png")
    seed: int = 0
    backend: str | None = None
    max_steps: int | None = None


def _scalars(arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    """Extract physical scalar traces from the common trajectory arrays."""
    valid = arrays["valid"]
    return {
        "t": arrays["time_s"][valid],
        "Ip": arrays["Ip"][valid] / 1e6,
        "P_fus": arrays["P_fusion"][valid] / 1e6,
        "P_aux": arrays["P_aux_total"][valid] / 1e6,
        "q_min": arrays["q_min"][valid],
        "fgw": arrays["fgw_n_e_line_avg"][valid],
        "beta_N": arrays["beta_N"][valid],
        "end": int(valid.sum()),
    }


def _learning_curve(results: Any) -> tuple[np.ndarray, np.ndarray] | None:
    if not isinstance(results, dict) or "global_step" not in results:
        return None
    evaluation = results["evaluation"]
    if isinstance(evaluation, dict):
        returns = evaluation.get(
            "evaluation/return_mean", evaluation.get("eval/return_mean")
        )
        if returns is None:
            return None
    else:
        returns = np.asarray(evaluation[1]).mean(axis=-1)
    return np.asarray(results["global_step"]), np.asarray(returns)


def _compare(policy: Any, env: Any, key: jax.Array) -> tuple[dict, dict, dict, dict]:
    """Collect both controllers with identical environment and policy key banks."""
    # collect_episodes splits the episode key, then collect_episode splits its
    # environment and policy streams. Match that reset to choose the hold value.
    initial_key = jax.random.split(jax.random.split(key, 1)[0])[0]
    state, _ = env.init(initial_key)
    initial_action = unwrap_to_env_state(state).prev_action
    setpoint = env.from_physical(initial_action)

    def hold(obs: jax.Array, rng: jax.Array) -> jax.Array:
        del obs, rng
        return setpoint

    optimized_metrics, optimized = evaluate_policy(policy, env, key, num_episodes=1)
    baseline_metrics, baseline = evaluate_policy(hold, env, key, num_episodes=1)
    return (
        optimized_metrics,
        trajectory_arrays(optimized, env),
        baseline_metrics,
        trajectory_arrays(baseline, env),
    )


def main(config: Config) -> None:
    policy = load_policy(config.policy)
    env = load_policy_env(policy, backend=config.backend, max_steps=config.max_steps)
    env_config = policy.metadata.get("config", {}).get("env", {})
    env_path = env_config.get("env_setup", "saved task")
    backend_path = config.backend or env_config.get("backend") or "native"
    reward = env_config.get("reward") or "task"
    actuators = [spec.name for spec in env.unwrapped.actuator_specs]
    opt_metrics, opt_arrays, base_metrics, base_arrays = _compare(
        policy,
        env,
        jax.random.key(config.seed),
    )
    opt, base_run = _scalars(opt_arrays), _scalars(base_arrays)
    opt_phys = opt_arrays["command_physical"][opt_arrays["valid"]]
    setp_phys = base_arrays["command_physical"][base_arrays["valid"]]
    num_steps = opt_arrays["valid"].shape[-1]
    base_cum = float(np.asarray(base_metrics["returns"])[0])
    opt_cum = float(np.asarray(opt_metrics["returns"])[0])
    gain = f"Δ {opt_cum - base_cum:+.2f}"
    if base_cum != 0:
        gain += f", {100 * (opt_cum - base_cum) / abs(base_cum):+.1f}%"

    optimized_color, baseline_color = "tab:red", "tab:gray"
    fig, axes = plt.subplots(3, 4, figsize=(20, 12))
    fig.suptitle(
        f"Optimized open-loop discharge vs setpoint-hold — {Path(env_path).stem} / "
        f"{Path(backend_path).stem} / reward={reward}\n"
        f"cum_reward: optimized {opt_cum:+.2f}  vs  baseline {base_cum:+.2f}  "
        f"({gain})",
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
    for j, name in enumerate(actuators[:4]):
        ax = axes[1, j]
        ax.plot(
            base_run["t"],
            setp_phys[:, j],
            color=baseline_color,
            ls="--",
            lw=1.5,
            label="setpoint-hold",
        )
        ax.plot(
            opt["t"], opt_phys[:, j], color=optimized_color, lw=2, label="optimized"
        )
        ax.set(xlabel="t [s]", ylabel=name, title=f"Actuator: {name}")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="best")

    for ax in axes[1, len(actuators[:4]) :]:
        ax.axis("off")

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
    curve = _learning_curve(policy.results)
    if curve is not None:
        ax.plot(*curve, color=optimized_color)
    ax.set(
        xlabel="training transitions",
        ylabel="episode return",
        title="Optimization learning curve",
    )
    ax.grid(alpha=0.3)

    axes[2, 3].axis("off")

    fig.tight_layout(rect=(0, 0, 1, 0.95))  # leave room for the suptitle
    out = config.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"Saved {out}")
    if opt["end"]:
        print(
            f"optimized final: Ip={opt['Ip'][-1]:.2f}MA  "
            f"P_fus={opt['P_fus'][-1]:.1f}MW  "
            f"q_min={opt['q_min'][-1]:.2f}  n/n_GW={opt['fgw'][-1]:.3f}  "
            f"alive={opt['end']}/{num_steps}"
        )


if __name__ == "__main__":
    main(tyro.cli(Config))
