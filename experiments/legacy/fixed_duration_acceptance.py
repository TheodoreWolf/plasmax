"""Historical fixed-duration TGLFNN-NR rollout acceptance study.

This superseded acceptance run is deliberately stricter than the Pareto
collector: every requested control interval must land on the 0.1 s grid,
complete without a solver/event budget overflow, keep observations finite, and
use fewer than 64 internal TORAX calls. The configured backend cap is 128,
leaving at least 2x headroom over an accepted run.
"""

from __future__ import annotations

import dataclasses
import gc
import json
import platform
import statistics
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import tyro

from plasmax.environment.factory import load_env


@dataclasses.dataclass(frozen=True)
class Config:
    env_setup: str = "iter/hybrid/flattop"
    backend: str = "tglfnn_nr"
    variant: str = "realistic"
    seeds: tuple[int, ...] = tuple(range(10))
    n_steps: int = 100
    control_dt: float = 0.1
    time_tolerance: float = 1.0e-6
    accepted_internal_step_limit: int = 64
    check_gradients: bool = True
    require_gpu: bool = True
    output: Path = Path("outputs/tglfnn_nr_fixed_duration_gpu.json")


def _validate(cfg: Config) -> None:
    if not cfg.seeds or len(set(cfg.seeds)) != len(cfg.seeds):
        raise ValueError("seeds must be non-empty and unique")
    if cfg.n_steps < 1:
        raise ValueError("n_steps must be positive")
    if cfg.control_dt <= 0.0:
        raise ValueError("control_dt must be positive")
    if cfg.time_tolerance <= 0.0:
        raise ValueError("time_tolerance must be positive")
    if cfg.accepted_internal_step_limit < 1:
        raise ValueError("accepted_internal_step_limit must be positive")
    if cfg.require_gpu and all(device.platform == "cpu" for device in jax.devices()):
        raise RuntimeError("a GPU device is required for this acceptance run")


def _make_transition(env: Any):
    @jax.jit
    def transition(state, action):
        next_state, info = env.step(state, action)
        solver = next_state.plasma.sim.solver_numeric_outputs
        outputs = (
            next_state.plasma.t,
            info.obs,
            info.internal_steps,
            info.sawtooth_crashes,
            info.control_step_complete,
            info.step_limit_reached,
            solver.solver_error_state,
            info.terminated,
            info.termination_code,
        )
        return next_state, outputs

    return transition


def _run_rollout(env: Any, transition: Any, key: jax.Array, n_steps: int):
    """Reuse one compiled transition instead of staging a huge 100-step graph."""
    state, _ = env.init(key)
    action = state.prev_action
    initial_t = state.plasma.t
    emissions = []
    for _ in range(n_steps):
        state, outputs = transition(state, action)
        emissions.append(outputs)
    stacked_outputs = tuple(
        jnp.stack(values) for values in zip(*emissions, strict=True)
    )
    return initial_t, stacked_outputs


def _make_gradient_diagnostic(env: Any):
    @jax.jit
    def gradient_diagnostic(key: jax.Array):
        state, _ = env.init(key)

        def objective(action: jax.Array) -> jax.Array:
            _, info = env.step(state, action)
            return jnp.mean(jnp.square(info.obs))

        return jax.value_and_grad(objective)(state.prev_action)

    return gradient_diagnostic


def _seed_result(
    cfg: Config,
    seed: int,
    initial_t: float,
    outputs: tuple[np.ndarray, ...],
    elapsed_s: float,
) -> dict[str, Any]:
    (
        times,
        observations,
        internal_steps,
        sawtooth_crashes,
        complete,
        limit_reached,
        solver_states,
        terminated,
        termination_codes,
    ) = outputs
    expected_times = initial_t + cfg.control_dt * np.arange(1, cfg.n_steps + 1)
    time_errors = np.abs(times - expected_times)
    finite_observations = np.all(np.isfinite(observations), axis=1)
    accepted_solver_states = np.isin(solver_states, (0, 2))
    passed = bool(
        np.all(time_errors <= cfg.time_tolerance)
        and np.all(finite_observations)
        and np.all(complete)
        and not np.any(limit_reached)
        and np.all(accepted_solver_states)
        and not np.any(terminated)
        and np.max(internal_steps) < cfg.accepted_internal_step_limit
    )
    return {
        "seed": seed,
        "elapsed_s": elapsed_s,
        "completed_steps_per_second": (
            float(np.count_nonzero(complete) / elapsed_s)
            if elapsed_s > 0.0
            else float("inf")
        ),
        "max_time_error": float(np.max(time_errors)),
        "internal_steps_min": int(np.min(internal_steps)),
        "internal_steps_median": float(np.median(internal_steps)),
        "internal_steps_max": int(np.max(internal_steps)),
        "total_sawtooth_crashes": int(np.sum(sawtooth_crashes)),
        "fine_convergence_count": int(np.count_nonzero(solver_states == 0)),
        "coarse_convergence_count": int(np.count_nonzero(solver_states == 2)),
        "failure_count": int(np.count_nonzero(solver_states == 1)),
        "incomplete_count": int(np.count_nonzero(~complete)),
        "step_limit_count": int(np.count_nonzero(limit_reached)),
        "nonfinite_observation_count": int(np.count_nonzero(~finite_observations)),
        "termination_count": int(np.count_nonzero(terminated)),
        "termination_codes": np.unique(termination_codes).astype(int).tolist(),
        "gradient": None,
        "passed": passed,
    }


def _build_payload(cfg: Config, results: list[dict[str, Any]]) -> dict[str, Any]:
    warm_durations = [result["elapsed_s"] for result in results[1:]]
    gradients_complete = not cfg.check_gradients or all(
        result["gradient"] is not None and result["gradient"]["finite"]
        for result in results
        if result["passed"]
    )
    return {
        "environment": cfg.env_setup,
        "backend": cfg.backend,
        "variant": cfg.variant,
        "seeds": list(cfg.seeds),
        "n_steps": cfg.n_steps,
        "control_dt": cfg.control_dt,
        "time_tolerance": cfg.time_tolerance,
        "accepted_internal_step_limit": cfg.accepted_internal_step_limit,
        "host": platform.node(),
        "devices": [str(device) for device in jax.devices()],
        "versions": {"jax": jax.__version__, "torax": version("torax")},
        "all_passed": all(result["passed"] for result in results)
        and gradients_complete,
        "gradients_complete": gradients_complete,
        "max_observed_internal_steps": max(
            result["internal_steps_max"] for result in results
        ),
        "median_warm_completed_steps_per_second": (
            cfg.n_steps / statistics.median(warm_durations) if warm_durations else None
        ),
        "results": results,
    }


def _write_payload(cfg: Config, results: list[dict[str, Any]]) -> dict[str, Any]:
    payload = _build_payload(cfg, results)
    cfg.output.parent.mkdir(parents=True, exist_ok=True)
    cfg.output.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def main(cfg: Config) -> None:
    _validate(cfg)
    env = load_env(
        cfg.env_setup,
        cfg.backend,
        variant=cfg.variant,
        max_steps=cfg.n_steps,
    ).unwrapped
    transition = _make_transition(env)
    results = []

    for seed in cfg.seeds:
        key = jax.random.key(seed)
        start = time.perf_counter()
        initial_t, device_outputs = _run_rollout(
            env,
            transition,
            key,
            cfg.n_steps,
        )
        jax.block_until_ready(device_outputs)
        elapsed_s = time.perf_counter() - start
        outputs = tuple(np.asarray(value) for value in device_outputs)
        result = _seed_result(
            cfg,
            seed,
            float(initial_t),
            outputs,
            elapsed_s,
        )
        results.append(result)
        print(json.dumps({"phase": "rollout", **result}, indent=2), flush=True)

    # Persist the expensive primal acceptance before compiling reverse mode.
    _write_payload(cfg, results)

    # The differentiated TGLFNN-NR executable is large. Release the primal
    # transition executable before compiling it so XLA can load both its CUBIN
    # and reverse-mode buffers without depending on process-lifetime cache
    # residency. This does not affect the separately recorded warm throughput.
    del transition, device_outputs
    jax.clear_caches()
    gc.collect()

    if cfg.check_gradients:
        gradient_diagnostic = _make_gradient_diagnostic(env)
        for seed, result in zip(cfg.seeds, results, strict=True):
            if not result["passed"]:
                continue
            key = jax.random.key(seed)
            start = time.perf_counter()
            objective, gradient = gradient_diagnostic(key)
            jax.block_until_ready((objective, gradient))
            gradient_np = np.asarray(gradient)
            result["gradient"] = {
                "objective": float(objective),
                "values": gradient_np.tolist(),
                "norm": float(np.linalg.norm(gradient_np)),
                "finite": bool(np.all(np.isfinite(gradient_np))),
                "elapsed_s": time.perf_counter() - start,
            }
            result["passed"] = result["passed"] and result["gradient"]["finite"]
            print(json.dumps({"phase": "gradient", **result}, indent=2), flush=True)
            _write_payload(cfg, results)

    payload = _write_payload(cfg, results)
    print(f"Saved {cfg.output}", flush=True)
    if not payload["all_passed"]:
        raise RuntimeError("TGLFNN-NR fixed-duration acceptance failed")


if __name__ == "__main__":
    main(tyro.cli(Config))
