"""Compact, observation-only feedback for generated policy search."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["build_feedback", "json_safe"]


def json_safe(value: Any) -> Any:
    """Encode nonfinite values explicitly without changing numeric artifacts."""
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, (tuple, list)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return "NaN" if math.isnan(value) else "Infinity" if value > 0 else "-Infinity"
    return value


def _dimension_names(
    interface: Mapping[str, Any], size: int, *, observations: bool
) -> list[str]:
    names = [str(index) for index in range(size)]
    if observations:
        for name, (start, stop) in interface.get("observations", {}).items():
            for index in range(start, min(stop, size)):
                names[index] = name if stop - start == 1 else f"{name}[{index - start}]"
    else:
        for index, spec in enumerate(interface.get("actuator_specs", [])[:size]):
            names[index] = spec["name"]
    return names


def _summarize_dimensions(
    values: np.ndarray, valid: np.ndarray, names: list[str]
) -> list[dict[str, Any]]:
    values = np.asarray(values).reshape(*valid.shape, -1)
    flat = values[valid]
    segments: list[list[np.ndarray]] = [[], [], [], []]
    for episode_values, episode_mask in zip(values, valid, strict=True):
        for index, segment in enumerate(
            np.array_split(episode_values[episode_mask], 4)
        ):
            segments[index].append(segment)
    bins = [np.concatenate(segment, axis=0) for segment in segments]
    summaries = []
    for index in range(flat.shape[-1]):
        column = flat[:, index]
        finite = column[np.isfinite(column)]
        temporal = []
        for segment in bins:
            finite_segment = segment[:, index][np.isfinite(segment[:, index])]
            temporal.append(
                float(finite_segment.mean()) if finite_segment.size else None
            )
        summaries.append(
            {
                "index": index,
                "name": names[index],
                "mean": float(finite.mean()) if finite.size else None,
                "min": float(finite.min()) if finite.size else None,
                "max": float(finite.max()) if finite.size else None,
                "nonfinite_count": int(column.size - finite.size),
                "segment_means": temporal,
            }
        )
    return summaries


def build_feedback(
    arrays: Mapping[str, np.ndarray],
    interface: Mapping[str, Any],
    seeds: Sequence[int],
    *,
    episode_returns: np.ndarray | None = None,
) -> dict[str, Any]:
    """Summarize real transitions; never expose privileged simulator fields.

    Statistics describe finite measurements and explicitly count nonfinite ones.
    Episode returns retain nonfinite values as JSON string labels. No reward or
    trajectory value is repaired, and padding never contributes to statistics.
    """
    valid = np.asarray(arrays["valid"], dtype=bool)
    if valid.ndim != 2 or valid.shape[0] != len(seeds):
        raise ValueError("trajectory valid mask must have shape (seeds, steps)")
    rewards = np.asarray(arrays["reward"])
    with np.errstate(invalid="ignore", over="ignore"):
        returns = (
            np.where(valid, rewards, 0).sum(axis=-1)
            if episode_returns is None
            else np.asarray(episode_returns)
        )
    if returns.shape != (len(seeds),):
        raise ValueError("episode returns must have shape (seeds,)")
    reasons = {
        1: "q_min_disruption",
        2: "greenwald_disruption",
        3: "solver_failure",
        4: "invalid_state",
    }
    episodes = []
    for index, seed in enumerate(seeds):
        active = valid[index]
        terminated = np.flatnonzero(active & arrays["terminated"][index])
        truncated = np.flatnonzero(active & arrays["truncated"][index])
        code = None
        if terminated.size:
            if "termination_code" in arrays:
                code = int(arrays["termination_code"][index, terminated[0]])
            reason = reasons.get(code, "terminated")
        elif truncated.size:
            reason = "truncated"
        else:
            reason = "no_valid_transitions" if not active.any() else "rollout_horizon"
        episodes.append(
            {
                "seed": int(seed),
                "return": float(returns[index]),
                "length": int(active.sum()),
                "termination_reason": reason,
                "termination_code": code,
            }
        )
    summaries = {}
    for field in ("obs", "next_obs", "requested_action", "applied_normalized"):
        if field not in arrays:
            continue
        values = np.asarray(arrays[field])
        size = int(np.prod(values.shape[2:]))
        names = _dimension_names(
            interface, size, observations=field in ("obs", "next_obs")
        )
        summaries[field] = _summarize_dimensions(values, valid, names)
    finite_returns = returns[np.isfinite(returns)]
    return json_safe(
        {
            "episodes": episodes,
            "return_summary": {
                "mean": float(returns.mean()),
                "min": float(returns.min()),
                "max": float(returns.max()),
                "finite_count": int(finite_returns.size),
            },
            "nonfinite_reward_count": int((valid & ~np.isfinite(rewards)).sum()),
            "nonfinite_requested_action_count": int(
                (~np.isfinite(arrays["requested_action"][valid])).sum()
            ),
            "trajectory_summary": summaries,
            "summary_scope": (
                "Valid transitions only; statistics use finite values. Segment "
                "means summarize four chronological, equal-count parts of each "
                "episode, pooled across seeds."
            ),
        }
    )
