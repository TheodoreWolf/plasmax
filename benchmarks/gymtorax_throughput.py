"""Measure the optional upstream Gym-TORAX single-environment CPU baseline.

Install Gym-TORAX and Tyro in a separate uv environment, then run this script
with ``uv run --no-project --python /path/to/environment/bin/python``. This
benchmark does not add Gym-TORAX to plasmax's installed dependencies.

The default workload replays Gym-TORAX's own ITER hybrid controller. One warm
rollout precedes repeated timed rollouts. Environment construction, resets,
action generation, and validation are outside the timed region. Each timed
``env.step`` includes the upstream action handling, simulator, observations,
reward, and synchronization. Count actual completed physical control intervals.
The cold measurement includes the first reset and rollout. JAX monitoring
records backend compilation during that work, separately from construction.
"""

from __future__ import annotations

import copy
import dataclasses
import importlib.metadata
import json
import os
import platform
import statistics
import subprocess
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import jax
import numpy as np
import tyro
from jax import monitoring


@dataclasses.dataclass(frozen=True)
class Config:
    """A single CPU environment, with the unmodified upstream configuration."""

    n_steps: int = 150
    repeats: int = 5
    seed: int = 0
    output_json: Path = Path("outputs/gymtorax_throughput.json")
    source_commit: str | None = None


@dataclasses.dataclass(frozen=True)
class RolloutTiming:
    reset_duration_s: float
    duration_s: float
    step_durations_s: tuple[float, ...]
    completed_steps: int
    attempted_steps: int
    final_simulation_time_s: float
    terminated: bool
    truncated: bool
    rewards_finite: bool


@dataclasses.dataclass
class CompilationTiming:
    duration_s: float = 0.0
    events: int = 0


_COMPILE_EVENT = "/jax/core/compile/backend_compile_duration"


@contextmanager
def _compilation_timer() -> Iterator[CompilationTiming]:
    timing = CompilationTiming()

    def record(event: str, duration: float, **_: Any) -> None:
        if event == _COMPILE_EVENT:
            timing.duration_s += duration
            timing.events += 1

    monitoring.register_event_duration_secs_listener(record)
    try:
        yield timing
    finally:
        monitoring.unregister_event_duration_listener(record)


def _cpu_model() -> str:
    if platform.system() == "Darwin":
        return subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip()
    return platform.processor() or platform.machine()


def _actions(env: Any, n_steps: int) -> list[dict[str, Any]]:
    from gymtorax import IterHybridAgent

    agent = IterHybridAgent(env.action_space)
    return [copy.deepcopy(agent.act(None)) for _ in range(n_steps)]


def _rollout(env: Any, actions: list[dict[str, Any]], seed: int) -> RolloutTiming:
    reset_start = time.perf_counter()
    env.reset(seed=seed)
    jax.block_until_ready(env.torax_app.current_sim_state)
    reset_duration_s = time.perf_counter() - reset_start
    duration_s = 0.0
    step_durations_s: list[float] = []
    completed_steps = 0
    attempted_steps = 0
    terminated = truncated = False
    rewards_finite = True
    previous_time = float(env.torax_app.current_sim_state.t)

    for template in actions:
        action = copy.deepcopy(template)
        start = time.perf_counter()
        _, reward, terminated, truncated, _ = env.step(action)
        jax.block_until_ready(
            (env.torax_app.current_sim_state, env.torax_app.current_sim_output)
        )
        step_s = time.perf_counter() - start
        step_durations_s.append(step_s)
        duration_s += step_s

        attempted_steps += 1
        current_time = float(env.torax_app.current_sim_state.t)
        complete = bool(
            np.isclose(current_time - previous_time, env.delta_t_a, rtol=0, atol=1e-8)
        )
        completed_steps += int(complete)
        previous_time = current_time
        rewards_finite = rewards_finite and bool(np.isfinite(reward))
        if not complete or terminated or truncated:
            break

    return RolloutTiming(
        reset_duration_s=reset_duration_s,
        duration_s=duration_s,
        step_durations_s=tuple(step_durations_s),
        completed_steps=completed_steps,
        attempted_steps=attempted_steps,
        final_simulation_time_s=previous_time,
        terminated=bool(terminated),
        truncated=bool(truncated),
        rewards_finite=rewards_finite,
    )


def main(cfg: Config) -> None:
    if not 1 <= cfg.n_steps <= 150:
        raise ValueError("n_steps must lie within the upstream 150-step episode")
    if cfg.repeats < 1:
        raise ValueError("repeats must be positive")
    if any(device.platform != "cpu" for device in jax.devices()):
        raise RuntimeError("Run the CPU baseline with JAX_PLATFORMS=cpu")

    from gymtorax import IterHybridEnv
    from gymtorax.envs.iter_hybrid_env import CONFIG

    cfg.output_json.parent.mkdir(parents=True, exist_ok=True)
    with _compilation_timer() as construction_compilation:
        start = time.perf_counter()
        env = IterHybridEnv(render_mode=None, store_history=False, log_level="warning")
        construction_s = time.perf_counter() - start
    actions = _actions(env, cfg.n_steps)
    print(
        f"Gym-TORAX {importlib.metadata.version('gymtorax')} on {_cpu_model()}; "
        f"{cfg.n_steps} steps, {cfg.repeats} repeats, dt={env.delta_t_a}s",
        flush=True,
    )
    timings: list[RolloutTiming] = []
    try:
        with _compilation_timer() as warmup_compilation:
            warmup = _rollout(env, actions, cfg.seed)
        print(
            f"warmup: {warmup.completed_steps} steps in {warmup.duration_s:.4f}s; "
            f"first step {warmup.step_durations_s[0]:.4f}s",
            flush=True,
        )
        if warmup.completed_steps != cfg.n_steps:
            raise RuntimeError(
                f"Warmup completed only {warmup.completed_steps}/{cfg.n_steps} steps"
            )
        for repeat in range(cfg.repeats):
            timing = _rollout(env, actions, cfg.seed)
            timings.append(timing)
            print(
                f"repeat {repeat + 1}: {timing.completed_steps} steps in "
                f"{timing.duration_s:.4f}s; first step "
                f"{timing.step_durations_s[0]:.4f}s",
                flush=True,
            )
            if timing.completed_steps != cfg.n_steps:
                raise RuntimeError(
                    f"Repeat completed only {timing.completed_steps}/"
                    f"{cfg.n_steps} steps"
                )
    finally:
        env.close()

    durations = [timing.duration_s for timing in timings]
    median_s = statistics.median(durations)
    cold_s = warmup.reset_duration_s + warmup.duration_s
    result = {
        "schema_version": 1,
        "measured_at_utc": datetime.now(UTC).isoformat(),
        "status": "ok",
        "series": "Gym-TORAX - CPU",
        "n_envs": 1,
        "sps": cfg.n_steps / median_s,
        "sps_min": cfg.n_steps / max(durations),
        "sps_max": cfg.n_steps / min(durations),
        "median_duration_s": median_s,
        "construction_s": construction_s,
        "construction_compile_s": construction_compilation.duration_s,
        "construction_compile_events": construction_compilation.events,
        "compile_and_first_run_s": cold_s,
        "compile_s": warmup_compilation.duration_s,
        "compile_events": warmup_compilation.events,
        "first_completed_steps": warmup.completed_steps,
        "cold_steps_per_second": warmup.completed_steps / cold_s,
        "warmup": dataclasses.asdict(warmup),
        "timings": [dataclasses.asdict(timing) for timing in timings],
        "config": {**dataclasses.asdict(cfg), "output_json": str(cfg.output_json)},
        "upstream_env": "gymtorax/IterHybrid-v0",
        "control_dt_s": float(env.delta_t_a),
        "fixed_solver_dt_s": CONFIG["numerics"]["fixed_dt"],
        "transport": CONFIG["transport"],
        "solver": CONFIG["solver"],
        "geometry": CONFIG["geometry"],
        "policy": "unmodified gymtorax.IterHybridAgent, precomputed actions",
        "timing_definition": (
            "Sum of synchronized env.step wall times. Construction, reset, policy, "
            "validation, and one warm rollout excluded. Single environment only."
        ),
        "cold_timing_definition": (
            "First synchronized reset plus the sum of first-rollout env.step wall "
            "times, including lazy JAX compilation. Construction, action generation, "
            "and host validation excluded."
        ),
        "compile_timing_definition": (
            f"Sum of JAX monitoring {_COMPILE_EVENT} event durations during the first "
            "reset and rollout. Measures backend compilation/cache loading, excludes "
            "Python tracing and lowering; construction compilation is separate."
        ),
        "platform": platform.platform(),
        "cpu_model": _cpu_model(),
        "devices": [str(device) for device in jax.devices()],
        "jax_enable_x64": jax.config.jax_enable_x64,
        "jax_enable_compilation_cache": jax.config.jax_enable_compilation_cache,
        "runtime_environment": {
            name: os.environ.get(name)
            for name in (
                "JAX_PLATFORMS",
                "JAX_ENABLE_COMPILATION_CACHE",
                "XLA_FLAGS",
            )
        },
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("gymtorax", "torax", "jax", "jaxlib", "numpy", "gymnasium")
        },
    }
    cfg.output_json.write_text(json.dumps(result, indent=2) + "\n")
    print(f"{result['sps']:.3f} completed steps/s; saved {cfg.output_json}", flush=True)


if __name__ == "__main__":
    main(tyro.cli(Config))
