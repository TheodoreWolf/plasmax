"""Collect constant-actuator discharge trajectories and numeric diagnostics.

    uv run python scripts/rollout_discharge.py --env iter/hybrid/flattop \
        --out plotting/data/discharge_rollout.npz

The NPZ contains complete profile arrays, cell/face grids, scalar traces, and
validity masks for offline rendering. Initial states and JSON diagnostics are
saved beside it. Single and multiple episodes preserve the paired key protocol.
Use the local ``plotting/scripts/plot_discharge_rollout.py`` to redraw saved data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from agents.policy_io import environment_interface
from plasmax import rollout as collect_lib
from plasmax.environment import factory as sc
from plasmax.wrappers import OracleWrappers, RealisticWrappers, unwrap_to_env_state
from training.evaluation import trajectory_arrays


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


def _json_finite(value: Any) -> Any:
    """Represent nonfinite diagnostics as null in portable JSON reports."""
    if isinstance(value, dict):
        return {key: _json_finite(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_finite(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _save_discharge_trajectories(
    path: Path,
    trajectory: Any,
    env: Any,
    rho: np.ndarray,
    rho_face: np.ndarray,
    *,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save every numeric input required to redraw profiles without simulation."""
    arrays = trajectory_arrays(trajectory, env)
    core = unwrap_to_env_state(trajectory.env_state).plasma.core
    arrays.update(rho_norm=rho, rho_face_norm=rho_face)
    for name in ("T_e", "T_i", "n_e"):
        arrays[name] = np.asarray(getattr(core, name).value)
    arrays["q_face"] = np.asarray(core.q_face)
    np.savez_compressed(
        path,
        **arrays,
        interface_json=np.asarray(json.dumps(environment_interface(env))),
        metadata_json=np.asarray(json.dumps({"format_version": 1, **(metadata or {})})),
    )


def _collect(
    args: argparse.Namespace,
) -> tuple[Any, np.ndarray, np.ndarray, dict[str, Any]]:
    wrappers = OracleWrappers if args.variant == "oracle" else RealisticWrappers
    env = wrappers(
        sc.make(args.env, args.backend, reward=args.reward), max_steps=args.num_steps
    )
    _ = env.action_space, env.observation_space
    key = jax.random.key(args.seed)
    episode_keys = (
        jax.random.split(key, args.num_episodes) if args.num_episodes > 1 else key[None]
    )
    init_keys = jax.vmap(lambda rng: jax.random.split(rng)[0])(episode_keys)
    action_norm, rho, rho_face = _hold_action_and_grids(env, init_keys[0], args.action)
    initial_states, initial_info = jax.jit(jax.vmap(env.init))(init_keys)
    print(
        f"Rolling out {args.num_episodes} episodes of {args.num_steps} steps "
        f"with {args.variant} wrappers, seed bank {args.seed} (compile + scan)...",
        flush=True,
    )
    act = _constant_policy(action_norm)
    if args.num_episodes > 1:
        trajectory = collect_lib.collect_episodes(
            act, env, key, args.num_steps, args.num_episodes
        )
    else:
        trajectory = jax.tree.map(
            lambda array: array[None],
            collect_lib.collect_episode(act, env, key, args.num_steps),
        )
    jax.block_until_ready(trajectory)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    _save_discharge_trajectories(
        out.with_suffix(".npz"),
        trajectory,
        env,
        rho,
        rho_face,
        metadata={
            "configuration": vars(args),
            "episode_keys": np.asarray(jax.random.key_data(episode_keys)).tolist(),
            "units": {"time_s": "s", "T_e": "keV", "T_i": "keV", "n_e": "m^-3"},
        },
    )
    initial = unwrap_to_env_state(initial_states)
    final = unwrap_to_env_state(trajectory.env_state)
    initial_arrays = {
        "episode_keys": np.asarray(jax.random.key_data(episode_keys)),
        "init_keys": np.asarray(jax.random.key_data(init_keys)),
        "obs": np.asarray(initial_info.obs),
        "q_min": np.asarray(initial.plasma.q_min),
        "fgw_n_e_line_avg": np.asarray(initial.plasma.fgw_n_e_line_avg),
    }
    for name in ("T_e", "T_i", "n_e"):
        initial_arrays[name] = np.asarray(getattr(initial.plasma.core, name).value)
    for name, value in initial.phys_params.items():
        initial_arrays[f"physics/{name}"] = np.asarray(value)
        initial_arrays[f"first_step_physics/{name}"] = np.asarray(
            final.phys_params[name][:, 0]
        )
    np.savez_compressed(out.with_name(f"{out.stem}-initial.npz"), **initial_arrays)
    valid = np.asarray(trajectory.valid)
    terminated = np.asarray(trajectory.terminated) & valid
    codes = np.asarray(trajectory.info.termination_code)
    lengths = valid.sum(axis=1)
    report = {
        "configuration": vars(args),
        "state_noise": dict(env.plasmax_config.state_noise),
        "physics_randomization_active": args.variant == "realistic"
        and bool(env.plasmax_config.physics_randomization),
        "first_step_failure_count": int(terminated[:, 0].sum()),
        "completed_horizon_count": int(
            np.sum((lengths == args.num_steps) & ~terminated.any(axis=1))
        ),
        "episodes": [
            {
                "index": index,
                "episode_key": initial_arrays["episode_keys"][index].tolist(),
                "init_key": initial_arrays["init_keys"][index].tolist(),
                "valid_steps": int(lengths[index]),
                "first_step_terminated": bool(terminated[index, 0]),
                "first_step_termination_code": int(codes[index, 0]),
                "first_termination_step": (
                    int(np.flatnonzero(terminated[index])[0]) + 1
                    if terminated[index].any()
                    else None
                ),
                "termination_code": (
                    int(codes[index, np.flatnonzero(terminated[index])[0]])
                    if terminated[index].any()
                    else -1
                ),
                "initial_profiles_finite": all(
                    np.isfinite(initial_arrays[name][index]).all().item()
                    for name in ("T_e", "T_i", "n_e")
                ),
                "initial_obs_finite": bool(
                    np.isfinite(initial_arrays["obs"][index]).all()
                ),
                "first_applied_action": np.asarray(
                    final.prev_action[index, 0]
                ).tolist(),
                "return": float(np.asarray(trajectory.reward)[index].sum()),
            }
            for index in range(args.num_episodes)
        ],
    }
    out.with_suffix(".json").write_text(
        json.dumps(_json_finite(report), indent=2, allow_nan=False) + "\n"
    )
    print(
        f"First-step failures: {report['first_step_failure_count']}/"
        f"{args.num_episodes}; completed horizon: "
        f"{report['completed_horizon_count']}/{args.num_episodes}",
        flush=True,
    )
    return trajectory, rho, rho_face, report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--env", default="iter/hybrid/flattop")
    p.add_argument("--backend", default="cgm")
    p.add_argument("--num-steps", type=int, default=4400)
    p.add_argument("--variant", choices=("oracle", "realistic"), default="realistic")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-episodes", type=int, default=1)
    p.add_argument("--wandb-project", default=None)
    p.add_argument("--wandb-entity", default="flair")
    p.add_argument("--wandb-group", default="discharge-rollouts")
    p.add_argument(
        "--reward",
        default=None,
        help="Reward override; omitted values inherit task YAML metadata.",
    )
    p.add_argument("--out", default="outputs/discharge_rollout.npz")
    p.add_argument(
        "--action",
        type=float,
        nargs=4,
        default=[15.0e6, 2.0e6, 0.45, 4.0e21],
        metavar=("P_nbi", "P_eccd", "rho_eccd", "gas_puff"),
        help="Constant physical actuator vector held for the whole rollout.",
    )
    args = p.parse_args()
    if args.num_steps <= 0 or args.num_episodes <= 0:
        p.error("num-steps and num-episodes must be positive")
    run = None
    if args.wandb_project is not None:
        import wandb

        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            group=args.wandb_group,
            mode="online",
            job_type="discharge_rollout",
            config=vars(args),
        )
    try:
        _, _, _, report = _collect(args)
        if run is not None:
            run.log(
                {
                    "first_step_failure_count": report["first_step_failure_count"],
                    "completed_horizon_count": report["completed_horizon_count"],
                }
            )
    finally:
        if run is not None:
            run.finish()


if __name__ == "__main__":
    main()
