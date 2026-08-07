"""Measure rollout agreement between transport backends over matched seeds.

The comparison uses the same initial state, configured initial actuator values,
and number of transitions for every backend. Agreement defaults to equal-weight
mean squared error (MSE) on the normalized full profiles T_e, T_i, n_e, and q,
with TGLFNN-linear as the reference. Each profile is averaged over its radial
dimensions and time before the four profile scores are averaged equally. MRE
remains available as an optional diagnostic.

Run on CPU:

    JAX_PLATFORMS=cpu uv run python benchmarks/backend_agreement.py
"""

from __future__ import annotations

import dataclasses
import json
import math
import platform
import statistics
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Literal

import jax
import numpy as np
import tyro

from plasmax.environment.factory import load_env

BACKENDS = (
    "cgm",
    "bohm_gyrobohm",
    "qlknn",
    "tglfnn",
)
ErrorMetric = Literal["mse", "mre"]
METRIC_SENSORS = (
    "T_e",
    "T_i",
    "n_e",
    "q",
)


@dataclasses.dataclass(frozen=True)
class Config:
    env_setup: str = "iter/hybrid/flattop"
    backends: tuple[str, ...] = BACKENDS
    backend_configs: tuple[str, ...] = ()
    validate_backend_pairs: bool = True
    reference_backend: str = "tglfnn"
    n_steps: int = 100
    seeds: tuple[int, ...] = tuple(range(11))
    error_metric: ErrorMetric = "mse"
    metric_sensors: tuple[str, ...] = METRIC_SENSORS
    relative_floor_fraction: float = 1.0e-3
    cpu_scalar_sps: tuple[float, ...] = ()
    timing_repeats: int = 5
    fail_on_boundary: bool = True
    output: Path = Path("outputs/backend_agreement_cpu.json")
    require_cpu: bool = True


@dataclasses.dataclass(frozen=True)
class Rollout:
    observations: np.ndarray
    boundaries: np.ndarray
    control_step_complete: np.ndarray
    solver_iterations: np.ndarray
    solver_error_states: np.ndarray
    elapsed_s: float
    timing_durations_s: tuple[float, ...] = ()


RolloutRunner = Callable[
    [jax.Array],
    tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
]


def _validate_solver_error_states(error_states: np.ndarray) -> None:
    """Reject failed solves while accepting TORAX coarse convergence.

    TORAX uses state 0 for fine-tolerance convergence, state 1 for a failed
    solve, and state 2 for convergence within its accepted coarse tolerance.
    """
    invalid_mask = ~np.isin(error_states, (0, 2))
    if not np.any(invalid_mask):
        return

    details = {
        int(state): np.flatnonzero(error_states == state).tolist()
        for state in np.unique(error_states[invalid_mask])
    }
    raise RuntimeError(
        f"solver reported failed or unknown states at transition indices {details}"
    )


def _make_rollout_runner(env: Any, n_steps: int) -> RolloutRunner:
    """Build one compiled rollout callable that can be reused across seeds."""

    @jax.jit
    def run(key: jax.Array):
        state, _ = env.init(key)
        action = state.prev_action

        def _step(state, _):
            next_state, info = env.step(state, action)
            boundary = info.terminated | info.truncated
            solver_outputs = next_state.plasma.sim.solver_numeric_outputs
            return next_state, (
                info.obs,
                boundary,
                info.control_step_complete,
                solver_outputs.inner_solver_iterations,
                solver_outputs.solver_error_state,
            )

        _, outputs = jax.lax.scan(_step, state, xs=None, length=n_steps)
        return outputs

    return run


def _collect_rollout(
    env: Any,
    seed: int,
    n_steps: int,
    fail_on_boundary: bool,
    timing_repeats: int = 0,
    runner: RolloutRunner | None = None,
) -> Rollout:
    if timing_repeats < 0:
        raise ValueError("timing_repeats must be non-negative")
    key = jax.random.key(seed)
    run = runner or _make_rollout_runner(env, n_steps)

    start = time.perf_counter()
    observations, boundaries, complete, iterations, error_states = run(key)
    jax.block_until_ready(
        (observations, boundaries, complete, iterations, error_states)
    )
    elapsed_s = time.perf_counter() - start

    timing_durations = []
    for _ in range(timing_repeats):
        start = time.perf_counter()
        timed_outputs = run(key)
        jax.block_until_ready(timed_outputs)
        timing_durations.append(time.perf_counter() - start)

    observations_np = np.asarray(observations)
    boundaries_np = np.asarray(boundaries)
    complete_np = np.asarray(complete)
    iterations_np = np.asarray(iterations)
    error_states_np = np.asarray(error_states)
    if fail_on_boundary and np.any(boundaries_np):
        indices = np.flatnonzero(boundaries_np).tolist()
        raise RuntimeError(f"rollout terminated at transition indices {indices}")
    if not np.all(np.isfinite(observations_np)):
        raise RuntimeError("rollout contains non-finite observations")
    if not np.all(complete_np):
        indices = np.flatnonzero(~complete_np).tolist()
        raise RuntimeError(
            f"rollout did not complete the physical control interval at "
            f"transition indices {indices}"
        )
    _validate_solver_error_states(error_states_np)
    return Rollout(
        observations=observations_np,
        boundaries=boundaries_np,
        control_step_complete=complete_np,
        solver_iterations=iterations_np,
        solver_error_states=error_states_np,
        elapsed_s=elapsed_s,
        timing_durations_s=tuple(timing_durations),
    )


def _elementwise_error(
    observations: np.ndarray,
    reference: np.ndarray,
    metric: ErrorMetric,
    floor_fraction: float,
) -> np.ndarray:
    """Return squared or relative elementwise error on normalized values."""
    if metric == "mse":
        return np.square(observations - reference)

    reference_magnitude = np.abs(reference)
    dimension_scale = np.max(reference_magnitude, axis=0, keepdims=True)
    denominator_floor = floor_fraction * np.maximum(dimension_scale, 1.0e-12)
    denominator = np.maximum(reference_magnitude, denominator_floor)
    return np.abs(observations - reference) / denominator


def _sensor_metrics(relative_error: np.ndarray, layout: Any) -> dict[str, float]:
    names = (*layout.profile_names, *layout.scalar_names)
    return {
        name: float(np.mean(relative_error[:, layout.slice_of(name)])) for name in names
    }


def _selected_sensor_mean(
    sensor_metrics: dict[str, float], sensor_names: Sequence[str]
) -> float:
    """Average selected sensor metrics with equal weight."""
    if not sensor_names:
        raise ValueError("metric_sensors must not be empty")
    if len(set(sensor_names)) != len(sensor_names):
        raise ValueError("metric_sensors must not contain duplicates")
    missing = set(sensor_names) - sensor_metrics.keys()
    if missing:
        raise ValueError(f"unknown metric sensors: {sorted(missing)}")
    return float(np.mean([sensor_metrics[name] for name in sensor_names]))


def _sensor_balanced_mean(
    elementwise_error: np.ndarray,
    layout: Any,
    sensor_names: Sequence[str] | None = None,
) -> float:
    """Average profile radii first, then weight selected sensors equally."""
    names = sensor_names or (*layout.profile_names, *layout.scalar_names)
    per_sensor_per_step = np.stack(
        [
            np.mean(elementwise_error[:, layout.slice_of(name)], axis=1)
            for name in names
        ],
        axis=1,
    )
    return float(np.mean(per_sensor_per_step))


def _scalar_ranges(
    observations: np.ndarray,
    layout: Any,
    scalar_scales: dict[str, float],
) -> dict[str, dict[str, float]]:
    """Return raw-unit min/max values for every scalar observation."""
    ranges = {}
    for name in layout.scalar_names:
        values = observations[:, layout.slice_of(name)] * scalar_scales[name]
        ranges[name] = {
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
    return ranges


def _pareto_backends(metrics: dict[str, dict[str, Any]]) -> list[str]:
    available = {
        name: point for name, point in metrics.items() if point.get("error") is not None
    }
    frontier = []
    for name, point in available.items():
        dominated = any(
            other["error"] <= point["error"]
            and other["speedup"] >= point["speedup"]
            and (other["error"] < point["error"] or other["speedup"] > point["speedup"])
            for other_name, other in available.items()
            if other_name != name
        )
        if not dominated:
            frontier.append(name)
    return sorted(frontier, key=lambda name: available[name]["error"])


def _throughput_sps(
    cfg: Config,
    rollouts: dict[str, Rollout] | None = None,
) -> dict[str, float]:
    if cfg.cpu_scalar_sps:
        return dict(zip(cfg.backends, cfg.cpu_scalar_sps, strict=True))
    if cfg.timing_repeats:
        if rollouts is None:
            raise ValueError("rollouts are required when timing_repeats is positive")
        return {
            backend: np.count_nonzero(rollouts[backend].control_step_complete)
            / statistics.median(rollouts[backend].timing_durations_s)
            for backend in cfg.backends
        }
    raise ValueError(
        "throughput must be measured with timing_repeats > 0 or supplied via "
        "cpu_scalar_sps; pre-macro-step timings are not comparable"
    )


def _backend_configs(cfg: Config) -> dict[str, str]:
    configs = cfg.backend_configs or cfg.backends
    return dict(zip(cfg.backends, configs, strict=True))


def _validate(cfg: Config) -> None:
    if not cfg.seeds:
        raise ValueError("seeds must not be empty")
    if len(set(cfg.seeds)) != len(cfg.seeds):
        raise ValueError("seeds must not contain duplicates")
    if cfg.n_steps < 1:
        raise ValueError("n_steps must be positive")
    if cfg.timing_repeats < 0:
        raise ValueError("timing_repeats must be non-negative")
    if not 0.0 < cfg.relative_floor_fraction < 1.0:
        raise ValueError("relative_floor_fraction must lie in (0, 1)")
    if cfg.reference_backend not in cfg.backends:
        raise ValueError("reference_backend must be included in backends")
    if not cfg.metric_sensors:
        raise ValueError("metric_sensors must not be empty")
    if len(set(cfg.metric_sensors)) != len(cfg.metric_sensors):
        raise ValueError("metric_sensors must not contain duplicates")
    if cfg.backend_configs and len(cfg.backend_configs) != len(cfg.backends):
        raise ValueError("backend_configs must have one value per backend")
    if cfg.cpu_scalar_sps and len(cfg.cpu_scalar_sps) != len(cfg.backends):
        raise ValueError("cpu_scalar_sps must have one value per backend")
    if cfg.cpu_scalar_sps and min(cfg.cpu_scalar_sps) <= 0.0:
        raise ValueError("cpu_scalar_sps values must be positive")
    if not cfg.cpu_scalar_sps and not cfg.timing_repeats:
        raise ValueError(
            "set timing_repeats > 0 or provide cpu_scalar_sps; historical "
            "pre-macro-step throughput is invalid"
        )
    if cfg.require_cpu and any(device.platform != "cpu" for device in jax.devices()):
        raise RuntimeError("run with JAX_PLATFORMS=cpu")


def _sample_stats(values: Sequence[float]) -> dict[str, Any]:
    """Return descriptive statistics without hiding the per-seed values."""
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        raise ValueError("cannot summarize an empty sample")
    std = float(np.std(array, ddof=1)) if array.size > 1 else 0.0
    return {
        "mean": float(np.mean(array)),
        "std": std,
        "sem": std / math.sqrt(array.size),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "values": array.tolist(),
    }


def _measured_throughput(
    cfg: Config,
    timed_rollouts: dict[str, Rollout],
) -> dict[str, float]:
    if cfg.cpu_scalar_sps:
        return dict(zip(cfg.backends, cfg.cpu_scalar_sps, strict=True))
    missing = set(cfg.backends) - timed_rollouts.keys()
    if missing:
        raise RuntimeError(
            "cannot measure throughput because every seed failed for backends: "
            f"{sorted(missing)}"
        )
    return _throughput_sps(cfg, timed_rollouts)


def main(cfg: Config) -> None:
    _validate(cfg)
    backend_configs = _backend_configs(cfg)
    rollouts: dict[str, dict[int, Rollout]] = {}
    timed_rollouts: dict[str, Rollout] = {}
    failures: dict[str, dict[str, str]] = {}
    layouts: dict[str, Any] = {}
    scalar_scales: dict[str, dict[str, float]] = {}

    for backend in cfg.backends:
        print(f"Collecting {backend}...", flush=True)
        env = load_env(
            cfg.env_setup,
            backend_configs[backend],
            validate=cfg.validate_backend_pairs,
        ).unwrapped
        layouts[backend] = env.obs_layout()
        scalar_scales[backend] = {
            spec.name: spec.scale for spec in env.scalar_obs_specs
        }
        runner = _make_rollout_runner(env, cfg.n_steps)
        rollouts[backend] = {}
        failures[backend] = {}
        for seed in cfg.seeds:
            timing_repeats = cfg.timing_repeats if backend not in timed_rollouts else 0
            try:
                rollout = _collect_rollout(
                    env,
                    seed,
                    cfg.n_steps,
                    cfg.fail_on_boundary,
                    timing_repeats,
                    runner,
                )
            except RuntimeError as error:
                failures[backend][str(seed)] = str(error)
                print(f"  seed={seed} failed: {error}", flush=True)
                continue
            rollouts[backend][seed] = rollout
            if timing_repeats:
                timed_rollouts[backend] = rollout
            print(
                f"  seed={seed} shape={rollout.observations.shape} "
                f"cold/warm={rollout.elapsed_s:.2f}s",
                flush=True,
            )

    throughput_sps = _measured_throughput(cfg, timed_rollouts)
    reference_rollouts = rollouts[cfg.reference_backend]
    if not reference_rollouts:
        raise RuntimeError("every reference-backend rollout failed")
    reference_layout = layouts[cfg.reference_backend]
    reference_sps = throughput_sps[cfg.reference_backend]
    metrics: dict[str, dict[str, Any]] = {}

    for backend in cfg.backends:
        if layouts[backend] != reference_layout:
            raise ValueError(f"{backend} observation layout differs from reference")
        if scalar_scales[backend] != scalar_scales[cfg.reference_backend]:
            raise ValueError(f"{backend} scalar scales differ from reference")
        paired_seeds = tuple(
            seed
            for seed in cfg.seeds
            if seed in reference_rollouts and seed in rollouts[backend]
        )
        if not paired_seeds:
            metrics[backend] = {
                "error": None,
                "agreement_available": False,
                "paired_seeds": [],
                "sps": throughput_sps[backend],
                "speedup": throughput_sps[backend] / reference_sps,
            }
            continue

        aggregate_mse: list[float] = []
        aggregate_mre: list[float] = []
        dimension_mse: list[float] = []
        dimension_mre: list[float] = []
        sensor_names = (*reference_layout.profile_names, *reference_layout.scalar_names)
        sensor_mse_values = {name: [] for name in sensor_names}
        sensor_mre_values = {name: [] for name in sensor_names}
        observations_by_seed = []
        for seed in paired_seeds:
            observations = rollouts[backend][seed].observations
            reference = reference_rollouts[seed].observations
            if observations.shape != reference.shape:
                raise ValueError(
                    f"{backend} seed {seed} shape {observations.shape} does not "
                    f"match reference shape {reference.shape}"
                )
            squared_error = _elementwise_error(
                observations, reference, "mse", cfg.relative_floor_fraction
            )
            relative_error = _elementwise_error(
                observations, reference, "mre", cfg.relative_floor_fraction
            )
            sensor_mse = _sensor_metrics(squared_error, reference_layout)
            sensor_mre = _sensor_metrics(relative_error, reference_layout)
            aggregate_mse.append(_selected_sensor_mean(sensor_mse, cfg.metric_sensors))
            aggregate_mre.append(_selected_sensor_mean(sensor_mre, cfg.metric_sensors))
            dimension_mse.append(float(np.mean(squared_error)))
            dimension_mre.append(float(np.mean(relative_error)))
            for name in sensor_names:
                sensor_mse_values[name].append(sensor_mse[name])
                sensor_mre_values[name].append(sensor_mre[name])
            observations_by_seed.append(observations)

        mse_stats = _sample_stats(aggregate_mse)
        mre_stats = _sample_stats(aggregate_mre)
        paired_rollouts = [rollouts[backend][seed] for seed in paired_seeds]
        all_observations = np.concatenate(observations_by_seed, axis=0)
        all_iterations = np.concatenate(
            [rollout.solver_iterations.reshape(-1) for rollout in paired_rollouts]
        )
        all_error_states = np.concatenate(
            [rollout.solver_error_states.reshape(-1) for rollout in paired_rollouts]
        )
        selected_stats = mse_stats if cfg.error_metric == "mse" else mre_stats
        metrics[backend] = {
            "error": selected_stats["mean"],
            "error_std": selected_stats["std"],
            "agreement_available": True,
            "paired_seeds": list(paired_seeds),
            "sensor_balanced_mse": mse_stats["mean"],
            "sensor_balanced_mse_std": mse_stats["std"],
            "sensor_balanced_mse_sem": mse_stats["sem"],
            "sensor_balanced_mse_values": mse_stats["values"],
            "sensor_balanced_mre": mre_stats["mean"],
            "sensor_balanced_mre_std": mre_stats["std"],
            "sensor_balanced_mre_sem": mre_stats["sem"],
            "sensor_balanced_mre_values": mre_stats["values"],
            "dimension_weighted_mse": _sample_stats(dimension_mse),
            "dimension_weighted_mre": _sample_stats(dimension_mre),
            "sps": throughput_sps[backend],
            "speedup": throughput_sps[backend] / reference_sps,
            "sensor_mse": {
                name: _sample_stats(values)
                for name, values in sensor_mse_values.items()
            },
            "sensor_mre": {
                name: _sample_stats(values)
                for name, values in sensor_mre_values.items()
            },
            "scalar_ranges": _scalar_ranges(
                all_observations, reference_layout, scalar_scales[backend]
            ),
            "solver_iterations_median": float(np.median(all_iterations)),
            "solver_iterations_max": int(np.max(all_iterations)),
            "solver_error_count": int(np.count_nonzero(all_error_states)),
            "solver_failure_count": int(np.count_nonzero(all_error_states == 1)),
            "solver_coarse_convergence_count": int(
                np.count_nonzero(all_error_states == 2)
            ),
            "boundary_count": int(
                sum(np.count_nonzero(run.boundaries) for run in paired_rollouts)
            ),
            "completed_control_intervals": int(
                sum(
                    np.count_nonzero(run.control_step_complete)
                    for run in paired_rollouts
                )
            ),
        }

    first_reference = reference_rollouts[next(iter(reference_rollouts))]
    result = {
        "environment": cfg.env_setup,
        "reference_backend": cfg.reference_backend,
        "backend_configs": backend_configs,
        "n_steps": cfg.n_steps,
        "requested_seeds": list(cfg.seeds),
        "successful_seeds": {
            backend: sorted(backend_rollouts)
            for backend, backend_rollouts in rollouts.items()
        },
        "failed_runs": failures,
        "host": platform.node(),
        "devices": [str(device) for device in jax.devices()],
        "timing_repeats": cfg.timing_repeats,
        "throughput_definition": (
            "completed physical control intervals per second; every retained "
            "rollout transition passed control_step_complete"
        ),
        "observation_dimensions": int(first_reference.observations.shape[1]),
        "profile_sensors": len(reference_layout.profile_names),
        "scalar_sensors": len(reference_layout.scalar_names),
        "metric_sensors": list(cfg.metric_sensors),
        "metric_sensor_count": len(cfg.metric_sensors),
        "error_metric": cfg.error_metric,
        "relative_floor_fraction": cfg.relative_floor_fraction,
        "error_definition": (
            "elementwise squared error on normalized observations"
            if cfg.error_metric == "mse"
            else "elementwise relative error is abs(x-reference) / "
            "max(abs(reference), relative_floor_fraction * max_t(abs(reference)))"
        ),
        "aggregation": (
            "each backend is compared on its own matched seed intersection with "
            "the reference; failures are reported per backend and do not remove "
            "successful seeds from unrelated backend comparisons"
        ),
        "boundary_policy": (
            "fail on any termination or truncation"
            if cfg.fail_on_boundary
            else "record termination and truncation flags but continue the "
            "uninterrupted simulator trajectory"
        ),
        "metrics": metrics,
        "pareto_frontier": _pareto_backends(metrics),
    }

    cfg.output.parent.mkdir(parents=True, exist_ok=True)
    cfg.output.write_text(json.dumps(result, indent=2) + "\n")

    print(
        f"\nbackend                    {cfg.error_metric.upper():>8}"
        " +/- std       sps  paired  failures",
        flush=True,
    )
    for backend in cfg.backends:
        point = metrics[backend]
        if point["error"] is None:
            print(
                f"{backend:<30} unavailable  {point['sps']:>8.4g}  "
                f"{0:>6}  {len(failures[backend]):>8}",
                flush=True,
            )
            continue
        print(
            f"{backend:<30} {point['error']:>8.4f} +/- "
            f"{point['error_std']:<8.4f} {point['sps']:>8.4g}  "
            f"{len(point['paired_seeds']):>6}  {len(failures[backend]):>8}",
            flush=True,
        )
    print(f"Pareto frontier: {result['pareto_frontier']}", flush=True)
    print(f"Saved {cfg.output}", flush=True)


if __name__ == "__main__":
    main(tyro.cli(Config))
