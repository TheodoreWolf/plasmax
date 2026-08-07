"""Aggregate backend profile MSE over multiple matched random seeds."""

from __future__ import annotations

import dataclasses
import json
import math
import platform
from pathlib import Path
from typing import Any

import jax
import numpy as np
import tyro

from benchmarks.backend_agreement import (
    BACKENDS,
    METRIC_SENSORS,
    _backend_configs,
    _collect_rollout,
    _elementwise_error,
    _make_rollout_runner,
    _pareto_backends,
    _selected_sensor_mean,
    _sensor_metrics,
)
from benchmarks.backend_agreement import (
    Config as SingleSeedConfig,
)
from plasmax.environment.factory import load_env


@dataclasses.dataclass(frozen=True)
class Config:
    env_setup: str = "iter/hybrid/flattop"
    backends: tuple[str, ...] = BACKENDS
    backend_configs: tuple[str, ...] = ()
    validate_backend_pairs: bool = True
    reference_backend: str = "tglfnn"
    seeds: tuple[int, ...] = tuple(range(11))
    n_steps: int = 100
    metric_sensors: tuple[str, ...] = METRIC_SENSORS
    speed_input: Path = Path("outputs/backend_pareto_cpu.json")
    output: Path = Path("outputs/backend_pareto_cpu_seeds.json")
    fail_on_boundary: bool = True
    require_cpu: bool = True


def _validate(cfg: Config, speed_data: dict[str, Any]) -> None:
    if not cfg.seeds:
        raise ValueError("seeds must not be empty")
    if len(set(cfg.seeds)) != len(cfg.seeds):
        raise ValueError("seeds must not contain duplicates")
    if cfg.reference_backend not in cfg.backends:
        raise ValueError("reference_backend must be included in backends")
    if not cfg.metric_sensors:
        raise ValueError("metric_sensors must not be empty")
    if cfg.backend_configs and len(cfg.backend_configs) != len(cfg.backends):
        raise ValueError("backend_configs must have one value per backend")
    if cfg.require_cpu and any(device.platform != "cpu" for device in jax.devices()):
        raise RuntimeError("run with JAX_PLATFORMS=cpu")

    expected = {
        "environment": cfg.env_setup,
        "reference_backend": cfg.reference_backend,
        "n_steps": cfg.n_steps,
        "metric_sensors": list(cfg.metric_sensors),
    }
    for field, value in expected.items():
        if speed_data.get(field) != value:
            raise ValueError(
                f"speed input {field} is {speed_data.get(field)!r}, expected {value!r}"
            )
    if tuple(speed_data["metrics"]) != cfg.backends:
        raise ValueError("speed input backends do not match configured backends")
    throughput_definition = speed_data.get("throughput_definition", "")
    if not throughput_definition.startswith("completed physical control intervals"):
        raise ValueError(
            "speed input predates fixed-duration macro-steps; rerun the "
            "throughput comparison"
        )


def _sample_stats(values: list[float]) -> dict[str, float | list[float]]:
    array = np.asarray(values, dtype=float)
    std = float(np.std(array, ddof=1)) if len(array) > 1 else 0.0
    return {
        "mean": float(np.mean(array)),
        "std": std,
        "sem": std / math.sqrt(len(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "values": values,
    }


def main(cfg: Config) -> None:
    speed_data = json.loads(cfg.speed_input.read_text())
    _validate(cfg, speed_data)
    backend_configs = _backend_configs(
        dataclasses.replace(
            SingleSeedConfig(),
            backends=cfg.backends,
            backend_configs=cfg.backend_configs,
        )
    )

    rollouts = {}
    layouts = {}
    profile_scales = {}
    failed_runs = {}
    for backend in cfg.backends:
        print(f"Collecting {backend}...", flush=True)
        env = load_env(
            cfg.env_setup,
            backend_configs[backend],
            validate=cfg.validate_backend_pairs,
        ).unwrapped
        layouts[backend] = env.obs_layout()
        profile_scales[backend] = {
            spec.name: spec.scale for spec in env.profile_obs_specs
        }
        runner = _make_rollout_runner(env, cfg.n_steps)
        rollouts[backend] = {}
        failed_runs[backend] = {}
        for seed in cfg.seeds:
            try:
                rollout = _collect_rollout(
                    env,
                    seed,
                    cfg.n_steps,
                    cfg.fail_on_boundary,
                    runner=runner,
                )
            except RuntimeError as error:
                failed_runs[backend][str(seed)] = str(error)
                print(f"  seed={seed:>2} failed: {error}", flush=True)
                continue
            rollouts[backend][seed] = rollout
            print(f"  seed={seed:>2} cold/warm={rollout.elapsed_s:.2f}s", flush=True)

    reference_layout = layouts[cfg.reference_backend]
    reference_scales = profile_scales[cfg.reference_backend]
    for backend in cfg.backends:
        if layouts[backend] != reference_layout:
            raise ValueError(f"{backend} observation layout differs from reference")
        if profile_scales[backend] != reference_scales:
            raise ValueError(f"{backend} profile scales differ from reference")

    boundary_details = {}
    excluded_seeds = set()
    for backend in cfg.backends:
        backend_boundaries = {}
        for seed in cfg.seeds:
            if seed not in rollouts[backend]:
                excluded_seeds.add(seed)
                continue
            indices = np.flatnonzero(rollouts[backend][seed].boundaries).tolist()
            if indices:
                excluded_seeds.add(seed)
                backend_boundaries[str(seed)] = {
                    "count": len(indices),
                    "first_transition_index": indices[0],
                }
        boundary_details[backend] = backend_boundaries
    included_seeds = tuple(seed for seed in cfg.seeds if seed not in excluded_seeds)
    if not included_seeds:
        raise RuntimeError("every seed crossed a termination or truncation boundary")

    per_backend = {
        backend: {sensor: [] for sensor in cfg.metric_sensors}
        for backend in cfg.backends
    }
    aggregate_values = {backend: [] for backend in cfg.backends}
    for seed in included_seeds:
        reference = rollouts[cfg.reference_backend][seed].observations
        for backend in cfg.backends:
            squared_error = _elementwise_error(
                rollouts[backend][seed].observations,
                reference,
                metric="mse",
                floor_fraction=1.0e-3,
            )
            sensor_mse = _sensor_metrics(squared_error, reference_layout)
            aggregate_values[backend].append(
                _selected_sensor_mean(sensor_mse, cfg.metric_sensors)
            )
            for sensor in cfg.metric_sensors:
                per_backend[backend][sensor].append(sensor_mse[sensor])

    reference_sps = float(speed_data["metrics"][cfg.reference_backend]["sps"])
    metrics = {}
    for backend in cfg.backends:
        stats = _sample_stats(aggregate_values[backend])
        sps = float(speed_data["metrics"][backend]["sps"])
        metrics[backend] = {
            "error": stats["mean"],
            "error_std": stats["std"],
            "sensor_balanced_mse": stats["mean"],
            "sensor_balanced_mse_std": stats["std"],
            "sensor_balanced_mse_sem": stats["sem"],
            "sensor_balanced_mse_min": stats["min"],
            "sensor_balanced_mse_max": stats["max"],
            "sensor_balanced_mse_values": stats["values"],
            "sensor_mse": {
                sensor: _sample_stats(values)
                for sensor, values in per_backend[backend].items()
            },
            "sps": sps,
            "speedup": sps / reference_sps,
        }

    first_profile = cfg.metric_sensors[0]
    first_profile_slice = reference_layout.slice_of(first_profile)
    profile_points = first_profile_slice.stop - first_profile_slice.start
    result = {
        "environment": cfg.env_setup,
        "reference_backend": cfg.reference_backend,
        "backend_configs": backend_configs,
        "requested_seeds": list(cfg.seeds),
        "seeds": list(included_seeds),
        "excluded_seeds": sorted(excluded_seeds),
        "n_requested_seeds": len(cfg.seeds),
        "n_seeds": len(included_seeds),
        "failed_runs": failed_runs,
        "boundary_details": boundary_details,
        "n_steps": cfg.n_steps,
        "host": platform.node(),
        "devices": [str(device) for device in jax.devices()],
        "metric_sensors": list(cfg.metric_sensors),
        "metric_sensor_count": len(cfg.metric_sensors),
        "profile_points": profile_points,
        "normalization_scales": {
            sensor: reference_scales[sensor] for sensor in cfg.metric_sensors
        },
        "error_metric": "mse",
        "error_definition": "elementwise squared error on normalized observations",
        "aggregation": (
            "within each seed, average over all profile radii and timesteps, "
            "then average T_e, T_i, n_e, and q equally; plotted values are the "
            "arithmetic mean across paired seeds where every backend completed, "
            "with sample-standard-deviation error bars"
        ),
        "throughput_source": str(cfg.speed_input),
        "metrics": metrics,
        "pareto_frontier": _pareto_backends(metrics),
    }

    cfg.output.parent.mkdir(parents=True, exist_ok=True)
    cfg.output.write_text(json.dumps(result, indent=2) + "\n")

    print("\nbackend                         MSE mean +/- std       sps", flush=True)
    for backend in cfg.backends:
        point = metrics[backend]
        print(
            f"{backend:<30} {point['sensor_balanced_mse']:.4f} +/- "
            f"{point['sensor_balanced_mse_std']:.4f}  {point['sps']:8.1f}",
            flush=True,
        )
    print(f"Pareto frontier: {result['pareto_frontier']}", flush=True)
    print(f"Included seeds: {list(included_seeds)}", flush=True)
    print(f"Excluded seeds: {sorted(excluded_seeds)}", flush=True)
    print(f"Saved {cfg.output}", flush=True)


if __name__ == "__main__":
    main(tyro.cli(Config))
