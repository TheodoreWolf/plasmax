"""Shared collection, trajectory exports, and paired transfer metrics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from agents.policy_io import LoadedPolicy, environment_interface
from plasmax.rollout import TrajectoryStep, collect_episodes
from plasmax.wrappers import find_max_steps, unwrap_to_env_state

__all__ = [
    "check_interfaces",
    "evaluate_policy",
    "evaluate_returns",
    "save_trajectories",
    "trajectory_arrays",
    "transfer_metrics",
    "write_transfer_summary",
]


def check_interfaces(source: Any, target: Any) -> None:
    """Reject changes to policy inputs/outputs while permitting backend changes."""
    expected = (
        source.interface
        if isinstance(source, LoadedPolicy)
        else environment_interface(source)
    )
    actual = environment_interface(target)
    mismatched = [
        key
        for key in expected.keys() | actual.keys()
        if expected.get(key) != actual.get(key)
    ]
    if mismatched:
        raise ValueError(
            "incompatible policy/environment interface: "
            f"{', '.join(sorted(mismatched))}"
        )


def _horizon(env: Any, num_steps: int | None) -> int:
    # Envelope spaces may be cached properties. Materialize constants before
    # collection traces them, so later artifact exports do not see leaked tracers.
    with jax.ensure_compile_time_eval():
        _ = env.action_space, env.observation_space
    horizon = find_max_steps(env) if num_steps is None else num_steps
    if horizon is None or horizon <= 0:
        raise ValueError(
            "evaluation requires a positive num_steps or a truncation wrapper"
        )
    return horizon


def evaluate_returns(
    act: Any,
    env: Any,
    rng: jax.Array,
    *,
    num_episodes: int,
    num_steps: int | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Return episode lengths and returns with JIT/vmap-compatible collection."""
    trajectory = collect_episodes(
        act, env, rng, num_steps=_horizon(env, num_steps), n_seeds=num_episodes
    )
    return trajectory.valid.sum(axis=-1), jnp.where(
        trajectory.valid, trajectory.reward, 0
    ).sum(axis=-1)


def evaluate_policy(
    policy: LoadedPolicy | Any,
    env: Any,
    rng: jax.Array,
    *,
    num_episodes: int = 1,
    num_steps: int | None = None,
    deterministic: bool | None = None,
) -> tuple[dict[str, jax.Array], TrajectoryStep]:
    """Collect comparable episodes; reusing ``rng`` pairs policies/backends."""
    if isinstance(policy, LoadedPolicy):
        check_interfaces(policy, env)
        act = policy.make_act(deterministic)
    else:
        act = policy
    trajectory = collect_episodes(
        act, env, rng, num_steps=_horizon(env, num_steps), n_seeds=num_episodes
    )
    returns = jnp.where(trajectory.valid, trajectory.reward, 0).sum(axis=-1)
    lengths = trajectory.valid.sum(axis=-1)
    return {
        "returns": returns,
        "lengths": lengths,
        "eval/return_mean": returns.mean(),
        "eval/return_std": returns.std(),
        "eval/episode_length_mean": lengths.mean(),
        "eval/termination_rate": jnp.any(
            trajectory.valid & trajectory.terminated, axis=-1
        ).mean(),
        "eval/truncation_rate": jnp.any(
            trajectory.valid & trajectory.truncated, axis=-1
        ).mean(),
    }, trajectory


def trajectory_arrays(trajectory: TrajectoryStep, env: Any) -> dict[str, np.ndarray]:
    """Flatten collected outputs for plotting without serializing simulator state."""
    fields = (
        "obs",
        "action",
        "reward",
        "next_obs",
        "terminated",
        "truncated",
        "valid",
        "done",
    )
    arrays = {name: np.asarray(getattr(trajectory, name)) for name in fields}
    arrays["command_physical"] = arrays["action"]
    base = unwrap_to_env_state(trajectory.env_state)
    if hasattr(base, "prev_action"):
        arrays["applied_physical"] = np.asarray(base.prev_action)
    if hasattr(env, "from_physical"):
        arrays["requested_action"] = np.asarray(env.from_physical(trajectory.action))
        if np.issubdtype(env.action_space.dtype, np.floating):
            arrays["command_normalized"] = arrays["requested_action"]
        if "applied_physical" in arrays:
            applied = np.asarray(
                env.from_physical(jnp.asarray(arrays["applied_physical"]))
            )
            if np.issubdtype(env.action_space.dtype, np.floating):
                arrays["applied_normalized"] = applied
    else:
        arrays["requested_action"] = arrays["action"]
    plasma = getattr(base, "plasma", None)
    if plasma is not None:
        arrays["time_s"] = np.asarray(plasma.t)
        for name in (
            "Q_fusion",
            "P_fusion",
            "W_thermal_total",
            "q_min",
            "q95",
            "beta_N",
            "P_aux_total",
            "fgw_n_e_line_avg",
        ):
            if hasattr(plasma, name):
                arrays[name] = np.asarray(getattr(plasma, name))
        if getattr(plasma.sim.core_profiles, "Ip_profile_face", None) is not None:
            arrays["Ip"] = np.asarray(plasma.sim.core_profiles.Ip_profile_face)[..., -1]
    elif (
        hasattr(env, "obs_layout") and "elapsed_time" in env.obs_layout().scalar_slices
    ):
        section = env.obs_layout().slice_of("elapsed_time")
        arrays["time_s"] = arrays["next_obs"][..., section.start]
    else:
        arrays["time_steps"] = np.cumsum(arrays["valid"], axis=-1)
    if hasattr(trajectory.info, "termination_code"):
        arrays["termination_code"] = np.asarray(trajectory.info.termination_code)
    if hasattr(env, "actuator_specs"):
        arrays["actuator_names"] = np.asarray(
            [spec.name for spec in env.actuator_specs]
        )
    return arrays


def save_trajectories(
    path: str | Path,
    trajectory: TrajectoryStep,
    env: Any,
    *,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Export the common numeric NPZ trajectory format with interface metadata."""
    output = Path(path)
    if output.suffix != ".npz":
        raise ValueError("trajectory exports must use the .npz extension")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        **trajectory_arrays(trajectory, env),
        interface_json=np.asarray(json.dumps(environment_interface(env))),
        metadata_json=np.asarray(json.dumps(metadata or {})),
    )
    return output


def transfer_metrics(
    source_backend: str,
    target_backend: str,
    source_returns: np.ndarray,
    source_lengths: np.ndarray,
    target_returns: np.ndarray,
    target_lengths: np.ndarray,
    eval_seconds: float,
) -> dict[str, float | str]:
    """Summarize paired source/target evaluations over seeds and episodes.

    All arrays have shape ``(num_training_seeds, num_evaluation_episodes)``.
    Source and target evaluations must use the same episode-key bank so their
    per-seed gap is paired rather than confounded by evaluation randomness.
    """
    arrays = {
        "source_returns": np.asarray(source_returns),
        "source_lengths": np.asarray(source_lengths),
        "target_returns": np.asarray(target_returns),
        "target_lengths": np.asarray(target_lengths),
    }
    expected_shape = arrays["source_returns"].shape
    if len(expected_shape) != 2:
        raise ValueError(
            "transfer arrays must have shape (training_seeds, evaluation_episodes)"
        )
    for name, values in arrays.items():
        if values.shape != expected_shape:
            raise ValueError(
                f"{name} shape {values.shape} does not match {expected_shape}"
            )

    per_seed_source = arrays["source_returns"].mean(axis=1)
    per_seed_target = arrays["target_returns"].mean(axis=1)
    per_seed_gap = per_seed_target - per_seed_source
    per_seed_ratio = np.divide(
        per_seed_target,
        per_seed_source,
        out=np.full_like(per_seed_target, np.nan, dtype=np.float64),
        where=per_seed_source != 0.0,
    )

    metrics: dict[str, float | str] = {
        "transfer/num_training_seeds": float(expected_shape[0]),
        "transfer/num_evaluation_episodes": float(expected_shape[1]),
        "transfer/source_backend": source_backend,
        "transfer/target_backend": target_backend,
        "transfer/source_return_mean": float(arrays["source_returns"].mean()),
        "transfer/source_return_std": float(arrays["source_returns"].std()),
        "transfer/target_return_mean": float(arrays["target_returns"].mean()),
        "transfer/target_return_std": float(arrays["target_returns"].std()),
        "transfer/target_return_min": float(arrays["target_returns"].min()),
        "transfer/target_return_max": float(arrays["target_returns"].max()),
        "transfer/return_gap": float(per_seed_gap.mean()),
        "transfer/return_ratio": float(np.nanmean(per_seed_ratio)),
        "transfer/target_episode_length_mean": float(arrays["target_lengths"].mean()),
        "transfer/source_episode_length_mean": float(arrays["source_lengths"].mean()),
        "transfer/eval_s": eval_seconds,
    }
    if expected_shape[0] > 1:
        metrics.update(
            {
                "transfer/source_return_seed_std": float(per_seed_source.std()),
                "transfer/target_return_seed_std": float(per_seed_target.std()),
                "transfer/return_gap_seed_std": float(per_seed_gap.std()),
                "transfer/return_ratio_seed_std": float(np.nanstd(per_seed_ratio)),
            }
        )
    return metrics


def write_transfer_summary(
    out_dir: str | Path | None,
    run_name: str,
    metrics: dict[str, float | str],
) -> Path | None:
    """Persist transfer summary metrics beside a run's training artifacts."""
    if out_dir is None:
        return None
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{run_name}_transfer.json"
    path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    return path
