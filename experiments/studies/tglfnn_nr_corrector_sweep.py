"""Sweep TGLFNN-NR predictor-corrector warm-start iterations.

This diagnostic keeps the environment step at 0.1 s and disables TORAX's
adaptive time-step fallback. It tests whether additional linear corrector
passes produce a converged Newton-Raphson solve without changing the physical
time represented by one RL transition.
"""

from __future__ import annotations

import dataclasses
import json
import platform
import tempfile
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import tyro
import yaml

from plasmax.environment.factory import make
from plasmax.environment.registry import resolve_backend


@dataclasses.dataclass(frozen=True)
class Config:
    env_setup: str = "iter/hybrid/flattop"
    backend: str = "tglfnn_nr"
    corrector_steps: tuple[int, ...] = (10, 20, 40)
    n_steps: int = 5
    seed: int = 0
    fixed_dt: float = 0.1
    n_max_iterations: int = 100
    log_iterations: bool = False
    require_gpu: bool = True
    output: Path = Path("outputs/tglfnn_nr_corrector_sweep.json")


def _validate(cfg: Config) -> None:
    if not cfg.corrector_steps:
        raise ValueError("corrector_steps must not be empty")
    if min(cfg.corrector_steps) < 1:
        raise ValueError("corrector_steps must contain positive integers")
    if cfg.n_steps < 1:
        raise ValueError("n_steps must be positive")
    if cfg.fixed_dt <= 0.0:
        raise ValueError("fixed_dt must be positive")
    if cfg.n_max_iterations < 1:
        raise ValueError("n_max_iterations must be positive")
    if cfg.require_gpu and all(device.platform == "cpu" for device in jax.devices()):
        raise RuntimeError("a GPU device is required for this run")


def _write_backend(
    source: Path,
    destination: Path,
    *,
    corrector_steps: int,
    cfg: Config,
) -> None:
    raw = yaml.safe_load(source.read_text())
    torax = raw.setdefault("torax", {})
    numerics = torax.setdefault("numerics", {})
    numerics["adaptive_dt"] = False
    numerics["fixed_dt"] = cfg.fixed_dt
    solver = torax.setdefault("solver", {})
    solver["n_corrector_steps"] = corrector_steps
    solver["n_max_iterations"] = cfg.n_max_iterations
    solver["log_iterations"] = cfg.log_iterations
    destination.write_text(yaml.safe_dump(raw, sort_keys=False))


def _make_rollout(env: Any, n_steps: int):
    @jax.jit
    def rollout(key: jax.Array):
        state, _ = env.init(key)
        action = state.prev_action
        initial_t = state.plasma.t

        def step_fn(state, _):
            next_state, info = env.step(state, action)
            solver = next_state.plasma.sim.solver_numeric_outputs
            outputs = (
                next_state.plasma.t,
                solver.inner_solver_iterations,
                solver.solver_error_state,
                info.termination_code != jnp.int32(3),
                info.terminated,
                info.termination_code,
            )
            return next_state, outputs

        _, outputs = jax.lax.scan(step_fn, state, xs=None, length=n_steps)
        return initial_t, action, outputs

    return rollout


def _gradient_diagnostic(env: Any, key: jax.Array) -> dict[str, Any]:
    state, _ = env.init(key)
    action = state.prev_action

    @jax.jit
    def value_and_grad(action: jax.Array):
        def objective(candidate_action: jax.Array) -> jax.Array:
            _, info = env.step(state, candidate_action)
            return jnp.mean(jnp.square(info.obs))

        return jax.value_and_grad(objective)(action)

    start = time.perf_counter()
    value, gradient = value_and_grad(action)
    jax.block_until_ready((value, gradient))
    elapsed_s = time.perf_counter() - start
    gradient_np = np.asarray(gradient)
    return {
        "objective": float(value),
        "gradient": gradient_np.tolist(),
        "gradient_norm": float(np.linalg.norm(gradient_np)),
        "gradient_finite": bool(np.all(np.isfinite(gradient_np))),
        "elapsed_s": elapsed_s,
    }


def _run_case(cfg: Config, backend_path: Path, corrector_steps: int) -> dict[str, Any]:
    env = make(
        cfg.env_setup,
        str(backend_path),
        variant="realistic",
        max_steps=cfg.n_steps,
        validate=False,
    ).unwrapped
    key = jax.random.key(cfg.seed)
    rollout = _make_rollout(env, cfg.n_steps)

    start = time.perf_counter()
    initial_t, _, outputs = rollout(key)
    jax.block_until_ready(outputs)
    elapsed_s = time.perf_counter() - start
    times, iterations, error_states, finite, terminated, termination_codes = (
        np.asarray(value) for value in outputs
    )
    initial_t_float = float(initial_t)
    expected_times = initial_t_float + cfg.fixed_dt * np.arange(1, cfg.n_steps + 1)
    accepted = np.isin(error_states, (0, 2))
    forward_valid = bool(np.all(accepted) and np.all(finite))

    result: dict[str, Any] = {
        "corrector_steps": corrector_steps,
        "elapsed_s": elapsed_s,
        "initial_t": initial_t_float,
        "times": times.tolist(),
        "time_steps": np.diff(np.concatenate(([initial_t_float], times))).tolist(),
        "max_time_error": float(np.max(np.abs(times - expected_times))),
        "solver_iterations": iterations.tolist(),
        "solver_error_states": error_states.tolist(),
        "finite": finite.tolist(),
        "terminated": terminated.tolist(),
        "termination_codes": termination_codes.tolist(),
        "fine_convergence_count": int(np.count_nonzero(error_states == 0)),
        "coarse_convergence_count": int(np.count_nonzero(error_states == 2)),
        "failure_count": int(np.count_nonzero(error_states == 1)),
        "forward_valid": forward_valid,
        "gradient": None,
    }
    if forward_valid:
        result["gradient"] = _gradient_diagnostic(env, key)
    return result


def main(cfg: Config) -> None:
    _validate(cfg)
    source_backend = Path(resolve_backend(cfg.backend))
    results = []

    with tempfile.TemporaryDirectory(prefix="tglfnn-nr-sweep-") as tmp_dir:
        tmp_path = Path(tmp_dir)
        for corrector_steps in cfg.corrector_steps:
            print(f"Running n_corrector_steps={corrector_steps}...", flush=True)
            backend_path = tmp_path / f"tglfnn_nr_correctors_{corrector_steps}.yaml"
            _write_backend(
                source_backend,
                backend_path,
                corrector_steps=corrector_steps,
                cfg=cfg,
            )
            try:
                result = _run_case(cfg, backend_path, corrector_steps)
            except Exception as error:  # keep the sweep running after one bad case
                result = {
                    "corrector_steps": corrector_steps,
                    "exception": f"{type(error).__name__}: {error}",
                }
            results.append(result)
            print(json.dumps(result, indent=2), flush=True)

    payload = {
        "environment": cfg.env_setup,
        "backend": cfg.backend,
        "variant": "realistic",
        "seed": cfg.seed,
        "n_steps": cfg.n_steps,
        "fixed_dt": cfg.fixed_dt,
        "adaptive_dt": False,
        "n_max_iterations": cfg.n_max_iterations,
        "host": platform.node(),
        "devices": [str(device) for device in jax.devices()],
        "results": results,
    }
    cfg.output.parent.mkdir(parents=True, exist_ok=True)
    cfg.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Saved {cfg.output}", flush=True)


if __name__ == "__main__":
    main(tyro.cli(Config))
