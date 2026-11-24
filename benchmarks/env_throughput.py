"""Benchmark steady-state environment-step throughput.

The timed region is a JIT-compiled ``lax.scan`` over the core environment with
``PhysicsRandomizationWrapper``. Distinct dynamic PRNG keys make the batched
physics trajectories independent; no sensor/action or horizon wrappers are used.
Every scan iteration requests one physical control transition. Throughput is
reported only for transitions that complete their full configured duration. If
a transition terminates, the next iteration starts from a reset state; terminal
slots are never replaced by cheap padding.
Tracing/lowering, backend compilation, and the first execution are timed
separately; cold throughput includes all three stages.

Run on CPU:

    JAX_PLATFORMS=cpu uv run python benchmarks/env_throughput.py \
        --env-setup iter/hybrid/flattop \
        --backend cgm \
        --n-envs 1 4 16
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.metadata
import json
import os
import platform
import statistics
import subprocess
import time
from datetime import UTC, datetime
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import tyro

from plasmax.environment.factory import make
from plasmax.wrappers import PhysicsRandomizationWrapper, unwrap_to_env_state

SEED = 0


@dataclasses.dataclass(frozen=True)
class Config:
    env_setup: str = "iter/hybrid/flattop"
    backend: str = "cgm"
    n_steps: int = 100
    n_envs: tuple[int, ...] = (1, 4, 16)
    repeats: int = 5
    reset_on_boundary: bool = True
    vmap_scalar: bool = True
    require_cpu: bool = True
    output_json: Path | None = None
    hardware_label: str | None = None


@dataclasses.dataclass(frozen=True)
class Timing:
    n_envs: int
    compile_and_first_run_s: float
    trace_lower_s: float
    compile_s: float
    first_execution_s: float
    first_completed_steps: int
    durations_s: tuple[float, ...]
    boundaries: int
    completed_steps: int


def _rollout(
    env: Any,
    key: jax.Array,
    n_steps: int,
    reset_on_boundary: bool,
) -> tuple[Any, jax.Array, jax.Array]:
    """Run requested transitions and count boundaries and completed intervals."""
    state, _ = env.init(key)
    action = unwrap_to_env_state(state).prev_action

    def _step(carry, step_index):
        state, action = carry
        next_state, info = env.step(state, action)
        boundary = info.terminated | info.truncated
        reset_key = jax.random.fold_in(key, step_index + 1)

        def _reset(_):
            reset_state, _ = env.reset(next_state, reset_key)
            return reset_state, unwrap_to_env_state(reset_state).prev_action

        def _continue(_):
            return next_state, action

        if reset_on_boundary:
            next_carry = jax.lax.cond(boundary, _reset, _continue, operand=None)
        else:
            next_carry = _continue(None)
        return next_carry, (boundary, info.control_step_complete)

    (final_state, _), (boundaries, completed_steps) = jax.lax.scan(
        _step,
        (state, action),
        jnp.arange(n_steps, dtype=jnp.uint32),
    )
    return (
        final_state,
        jnp.count_nonzero(boundaries),
        jnp.count_nonzero(completed_steps),
    )


def _make_run(
    env: Any,
    n_steps: int,
    n_envs: int,
    reset_on_boundary: bool,
    vmap_scalar: bool,
) -> partial:
    keys = jax.random.split(jax.random.key(SEED), n_envs)

    def rollout(key: jax.Array):
        return _rollout(env, key, n_steps, reset_on_boundary)

    scalar = n_envs == 1 and not vmap_scalar
    return partial(
        jax.jit(rollout if scalar else jax.vmap(rollout)), keys[0] if scalar else keys
    )


def _time_run(env: Any, cfg: Config, n_envs: int) -> Timing:
    run = _make_run(
        env,
        cfg.n_steps,
        n_envs,
        cfg.reset_on_boundary,
        cfg.vmap_scalar,
    )
    jax.block_until_ready(run.args)

    start = time.perf_counter()
    lowered = run.func.lower(*run.args, **run.keywords)
    trace_lower_s = time.perf_counter() - start
    start = time.perf_counter()
    compiled = lowered.compile()
    compile_s = time.perf_counter() - start
    start = time.perf_counter()
    out = compiled(*run.args, **run.keywords)
    jax.block_until_ready(out)
    first_execution_s = time.perf_counter() - start
    first_completed_steps = int(jax.device_get(out[2]).sum())

    durations = []
    for _ in range(cfg.repeats):
        start = time.perf_counter()
        out = compiled(*run.args, **run.keywords)
        jax.block_until_ready(out)
        durations.append(time.perf_counter() - start)

    boundary_counts = jax.device_get(out[1])
    completed_counts = jax.device_get(out[2])
    return Timing(
        n_envs=n_envs,
        compile_and_first_run_s=trace_lower_s + compile_s + first_execution_s,
        trace_lower_s=trace_lower_s,
        compile_s=compile_s,
        first_execution_s=first_execution_s,
        first_completed_steps=first_completed_steps,
        durations_s=tuple(durations),
        boundaries=int(boundary_counts.sum()),
        completed_steps=int(completed_counts.sum()),
    )


def _validate(cfg: Config) -> None:
    if cfg.n_steps < 1:
        raise ValueError("n_steps must be positive")
    if cfg.repeats < 1:
        raise ValueError("repeats must be positive")
    if not cfg.n_envs or min(cfg.n_envs) < 1:
        raise ValueError("n_envs must contain positive integers")
    if cfg.require_cpu and any(device.platform != "cpu" for device in jax.devices()):
        raise RuntimeError(
            "CPU benchmark requested, but JAX selected non-CPU devices. "
            "Run with JAX_PLATFORMS=cpu."
        )


def _command_output(command: list[str], *, cwd: Path | None = None) -> str | None:
    """Read small optional host metadata without making it a run dependency."""
    try:
        return subprocess.check_output(
            command, cwd=cwd, text=True, stderr=subprocess.DEVNULL, timeout=5
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _provenance() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    versions = {}
    for package in (
        "plasmax",
        "jax",
        "jaxlib",
        "torax",
        "jax-envelope",
        "fusion-surrogates",
        "tglfnn-ukaea",
        "flax",
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    source_paths = (
        "benchmarks/env_throughput.py",
        "src/plasmax/environment/core.py",
        "src/plasmax/environment/stepping.py",
        "src/plasmax/wrappers.py",
        "uv.lock",
    )
    cpu_model = platform.processor()
    if platform.system() == "Darwin":
        cpu_model = _command_output(["sysctl", "-n", "machdep.cpu.brand_string"])
    snapshot_path = os.environ.get("PLASMAX_SOURCE_SNAPSHOT")
    return {
        "system": {
            "platform": platform.platform(),
            "hostname": platform.node(),
            "machine": platform.machine(),
            "cpu_model": cpu_model,
            "logical_cpu_count": os.cpu_count(),
            "python_version": platform.python_version(),
        },
        "devices": [
            {
                "id": device.id,
                "platform": device.platform,
                "device_kind": device.device_kind,
                "process_index": device.process_index,
            }
            for device in jax.devices()
        ],
        "software_versions": versions,
        "git": {
            "commit": _command_output(["git", "rev-parse", "HEAD"], cwd=root),
            "status": _command_output(["git", "status", "--short"], cwd=root),
        },
        "source_sha256": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in source_paths
            if (root / name).is_file()
        },
        "source_snapshot": (
            json.loads(Path(snapshot_path).read_text()) if snapshot_path else None
        ),
        "seed": SEED,
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "jax_enable_compilation_cache": bool(jax.config.jax_enable_compilation_cache),
        "runtime_environment": {
            name: os.environ.get(name)
            for name in (
                "JAX_PLATFORMS",
                "JAX_ENABLE_X64",
                "JAX_ENABLE_COMPILATION_CACHE",
                "XLA_FLAGS",
                "XLA_PYTHON_CLIENT_PREALLOCATE",
                "XLA_PYTHON_CLIENT_MEM_FRACTION",
                "SLURM_JOB_ID",
            )
        },
    }


def _environment_metadata(env: Any) -> dict[str, Any]:
    cfg = env.plasmax_config
    torax = cfg.torax
    initial_time = float(torax.numerics.t_initial)
    return {
        "wrapper": "PhysicsRandomizationWrapper",
        "action_policy": "hold initial prev_action; restore after boundary reset",
        "timed_work": "initialization and scan of full control transitions plus resets",
        "throughput_numerator": "completed physical control intervals",
        "control_interval_s": float(torax.numerics.fixed_dt.get_value(initial_time)),
        "control_interval_reference_time_s": initial_time,
        "torax_numerics": torax.numerics.model_dump(),
        "torax_solver": torax.solver.model_dump(),
        "torax_transport": torax.transport.model_dump(),
        "stepping": cfg.stepping.model_dump(),
        "physics_randomization": {
            name: spec.model_dump() for name, spec in cfg.physics_randomization.items()
        },
        "observation_shape": env.observation_space.shape,
        "action_shape": env.action_space.shape,
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value).__name__} as benchmark metadata")


def _timing_payload(cfg: Config, timing: Timing) -> dict[str, Any]:
    median_s = statistics.median(timing.durations_s)
    return {
        **dataclasses.asdict(timing),
        "requested_steps": cfg.n_steps * timing.n_envs,
        "median_s": median_s,
        "steps_per_second": timing.completed_steps / median_s,
        "cold_steps_per_second": (
            timing.first_completed_steps / timing.compile_and_first_run_s
        ),
        "min_steps_per_second": timing.completed_steps / max(timing.durations_s),
        "max_steps_per_second": timing.completed_steps / min(timing.durations_s),
    }


def _write_report(cfg: Config, report: dict[str, Any]) -> None:
    """Atomically retain all completed cases, even if a later case is interrupted."""
    if cfg.output_json is None:
        return
    cfg.output_json.parent.mkdir(parents=True, exist_ok=True)
    temporary = cfg.output_json.with_suffix(cfg.output_json.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, default=_json_default, allow_nan=False) + "\n"
    )
    temporary.replace(cfg.output_json)


def main(cfg: Config) -> None:
    _validate(cfg)
    devices = jax.devices()
    print("=" * 72, flush=True)
    print(f"env={cfg.env_setup}  backend={cfg.backend}", flush=True)
    print(f"host={platform.platform()}  devices={devices}", flush=True)
    print(
        f"steps/env={cfg.n_steps}  repeats={cfg.repeats}  "
        f"batches={list(cfg.n_envs)}  reset_on_boundary={cfg.reset_on_boundary}  "
        f"vmap_scalar={cfg.vmap_scalar}",
        flush=True,
    )

    start = time.perf_counter()
    env = PhysicsRandomizationWrapper(make(cfg.env_setup, cfg.backend))
    report = {
        "schema_version": 1,
        "status": "running",
        "started_at": datetime.now(UTC).isoformat(),
        "config": dataclasses.asdict(cfg),
        "provenance": _provenance(),
        "environment": _environment_metadata(env),
        "results": [],
    }
    _write_report(cfg, report)
    print(
        f"env ready in {time.perf_counter() - start:.2f}s  "
        f"obs={env.observation_space.shape}  action={env.action_space.shape}",
        flush=True,
    )
    print(flush=True)
    print(
        "n_envs  compile+first  median run    steps/s  completed  boundaries  range",
        flush=True,
    )

    try:
        for n_envs in cfg.n_envs:
            timing = _time_run(env, cfg, n_envs)
            row = _timing_payload(cfg, timing)
            report["results"].append(row)
            _write_report(cfg, report)
            low_s = min(timing.durations_s)
            high_s = max(timing.durations_s)
            print(
                f"{n_envs:>6}  {timing.compile_and_first_run_s:>13.3f}s  "
                f"{row['median_s']:>10.4f}s  {row['steps_per_second']:>9.1f}  "
                f"{timing.completed_steps:>9}  {timing.boundaries:>10}  "
                f"{low_s:.4f}-{high_s:.4f}s",
                flush=True,
            )
    except BaseException as error:
        report["status"] = "error"
        report["error"] = f"{type(error).__name__}: {error}"
        _write_report(cfg, report)
        raise
    report["status"] = "complete"
    report["finished_at"] = datetime.now(UTC).isoformat()
    _write_report(cfg, report)


if __name__ == "__main__":
    main(tyro.cli(Config))
