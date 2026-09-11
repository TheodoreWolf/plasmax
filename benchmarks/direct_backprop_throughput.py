"""Benchmark pathwise-gradient throughput over rollout batch and horizon grids.

This uses a dense per-step actuator schedule for one optimizer seed and measures
the transform order used by the direct baselines::

    vmap(value_and_grad(single_rollout))

The reported SPS is the number of requested environment transitions divided by
steady-state forward-plus-backward wall time. It excludes the Adam update and
evaluation rollouts; compilation is timed separately.
Run a representative GPU grid with::

    uv run python benchmarks/direct_backprop_throughput.py \
        --env-setup sparc/prd/rampup \
        --backend bohm_gyrobohm \
        --num-envs 64 128 256 512 1024 \
        --num-steps 25 50 100 \
        --repeats 3 \
        --output-json outputs/backprop-throughput.json
"""

from __future__ import annotations

import dataclasses
import gc
import json
import platform
import statistics
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import optax
import tyro

from agents.direct_gradient import make_knot_chunk, make_parameterization
from plasmax.environment.factory import make
from plasmax.wrappers import OracleWrappers, RealisticWrappers, find_max_steps


@dataclasses.dataclass(frozen=True)
class Config:
    """Backpropagation-through-environment benchmark configuration."""

    env_setup: str = "sparc/prd/rampup"
    backend: str = "bohm_gyrobohm"
    variant: str = "realistic"
    reward: str | None = None
    validate_backend_pair: bool = True
    num_envs: tuple[int, ...] = (64, 256, 1024)
    num_steps: tuple[int, ...] = (25, 50, 100)
    repeats: int = 3
    seed: int = 0
    remat: bool = True
    require_gpu: bool = True
    continue_on_error: bool = True
    clear_caches_between_cases: bool = True
    output_json: Path | None = None


@dataclasses.dataclass(frozen=True)
class CaseResult:
    """Timing and validity measurements for one static JAX shape."""

    num_envs: int
    num_steps: int
    status: str
    compile_and_first_run_s: float | None = None
    durations_s: tuple[float, ...] = ()
    median_s: float | None = None
    requested_steps: int | None = None
    alive_steps: int | None = None
    backprop_sps: float | None = None
    alive_backprop_sps: float | None = None
    loss: float | None = None
    grad_norm: float | None = None
    grads_finite: bool | None = None
    peak_bytes_in_use: int | None = None
    error: str | None = None


BackwardPass = Callable[[], tuple[jax.Array, jax.Array, jax.Array, jax.Array]]


def _validate_config(cfg: Config) -> None:
    if not cfg.num_envs or min(cfg.num_envs) < 1:
        raise ValueError("num_envs must contain positive integers")
    if not cfg.num_steps or min(cfg.num_steps) < 1:
        raise ValueError("num_steps must contain positive integers")
    if cfg.repeats < 1:
        raise ValueError("repeats must be positive")


def _make_backward_pass(
    env: Any,
    *,
    num_envs: int,
    num_steps: int,
    seed: int,
    remat: bool,
) -> BackwardPass:
    """Create one compiled forward-plus-backward call for a static grid cell."""
    parameter_key = jax.random.fold_in(jax.random.key(seed), 0x5E7)
    to_actions, theta, _ = make_parameterization(
        env,
        parameter_key,
        num_steps,
        n_knots=num_steps,
    )
    keys = jax.random.split(jax.random.key(seed), num_envs)
    chunk = make_knot_chunk(env, to_actions, num_steps, remat=remat)
    start_step = jnp.asarray(0, dtype=jnp.int32)

    def single_loss(current_theta: jax.Array, key: jax.Array):
        rollout_return, (_, trajectory) = chunk.run(
            current_theta,
            chunk.initialize(key),
            start_step,
        )
        return -rollout_return, jnp.sum(trajectory.alive)

    # Match production exactly: TORAX's adaptive-loop transpose supports
    # vmap(grad(single rollout)), rather than grad(vmap(rollout)).
    per_env_value_and_grad = jax.vmap(
        jax.value_and_grad(single_loss, has_aux=True),
        in_axes=(None, 0),
    )

    @jax.jit
    def backward(current_theta: jax.Array, rollout_keys: jax.Array):
        ((losses, alive_steps), per_env_grads) = per_env_value_and_grad(
            current_theta,
            rollout_keys,
        )
        mean_grads = jax.tree.map(
            lambda value: jnp.mean(value, axis=0),
            per_env_grads,
        )
        finite = jnp.all(
            jnp.stack(
                [jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree.leaves(mean_grads)]
            )
        )
        return (
            jnp.mean(losses),
            optax.global_norm(mean_grads),
            jnp.sum(alive_steps),
            finite,
        )

    return lambda: backward(theta, keys)


def _peak_bytes_in_use() -> int | None:
    stats = jax.local_devices()[0].memory_stats()
    if not stats:
        return None
    value = stats.get("peak_bytes_in_use")
    return None if value is None else int(value)


def _time_case(env: Any, cfg: Config, num_envs: int, num_steps: int) -> CaseResult:
    run = _make_backward_pass(
        env,
        num_envs=num_envs,
        num_steps=num_steps,
        seed=cfg.seed,
        remat=cfg.remat,
    )

    start = time.perf_counter()
    output = run()
    jax.block_until_ready(output)
    compile_and_first_run_s = time.perf_counter() - start

    durations = []
    for _ in range(cfg.repeats):
        start = time.perf_counter()
        output = run()
        jax.block_until_ready(output)
        durations.append(time.perf_counter() - start)

    loss, grad_norm, alive_steps, grads_finite = jax.device_get(output)
    median_s = statistics.median(durations)
    requested_steps = num_envs * num_steps
    alive_steps_int = int(alive_steps)
    return CaseResult(
        num_envs=num_envs,
        num_steps=num_steps,
        status="ok",
        compile_and_first_run_s=compile_and_first_run_s,
        durations_s=tuple(durations),
        median_s=median_s,
        requested_steps=requested_steps,
        alive_steps=alive_steps_int,
        backprop_sps=requested_steps / median_s,
        alive_backprop_sps=alive_steps_int / median_s,
        loss=float(loss),
        grad_norm=float(grad_norm),
        grads_finite=bool(grads_finite),
        peak_bytes_in_use=_peak_bytes_in_use(),
    )


def _report_payload(cfg: Config, results: list[CaseResult]) -> dict[str, Any]:
    return {
        "config": dataclasses.asdict(cfg),
        "host": platform.platform(),
        "jax_version": jax.__version__,
        "devices": [str(device) for device in jax.devices()],
        "results": [dataclasses.asdict(result) for result in results],
    }


def _write_report(cfg: Config, results: list[CaseResult]) -> None:
    if cfg.output_json is None:
        return
    cfg.output_json.parent.mkdir(parents=True, exist_ok=True)
    cfg.output_json.write_text(
        json.dumps(_report_payload(cfg, results), indent=2, default=str) + "\n"
    )


def main(cfg: Config) -> None:
    _validate_config(cfg)
    devices = jax.devices()
    if cfg.require_gpu and any(device.platform != "gpu" for device in devices):
        raise RuntimeError(f"GPU benchmark requested, but JAX selected {devices}")

    env = (RealisticWrappers if cfg.variant == "realistic" else OracleWrappers)(
        make(cfg.env_setup, cfg.backend, reward=cfg.reward)
    )
    max_steps = find_max_steps(env)
    if max_steps is not None and max(cfg.num_steps) > max_steps:
        raise ValueError(
            f"num_steps cannot exceed the environment horizon {max_steps}; "
            f"got {max(cfg.num_steps)}"
        )

    print(f"env={cfg.env_setup} backend={cfg.backend} variant={cfg.variant}")
    print(f"host={platform.platform()} devices={devices}")
    print(
        f"remat={cfg.remat} repeats={cfg.repeats} "
        f"num_envs={list(cfg.num_envs)} num_steps={list(cfg.num_steps)}"
    )
    print(
        "num_envs  num_steps  compile+first  median backward  backprop SPS  "
        "alive SPS  finite  peak GiB"
    )

    results: list[CaseResult] = []
    for num_steps in cfg.num_steps:
        for num_envs in cfg.num_envs:
            try:
                result = _time_case(env, cfg, num_envs, num_steps)
            except Exception as exc:
                if not cfg.continue_on_error:
                    raise
                result = CaseResult(
                    num_envs=num_envs,
                    num_steps=num_steps,
                    status="error",
                    error=f"{type(exc).__name__}: {exc}",
                )
            results.append(result)
            _write_report(cfg, results)

            if result.status == "ok":
                peak_gib = (
                    "n/a"
                    if result.peak_bytes_in_use is None
                    else f"{result.peak_bytes_in_use / 2**30:.2f}"
                )
                print(
                    f"{num_envs:>8}  {num_steps:>9}  "
                    f"{result.compile_and_first_run_s:>13.3f}s  "
                    f"{result.median_s:>15.4f}s  "
                    f"{result.backprop_sps:>12.1f}  "
                    f"{result.alive_backprop_sps:>9.1f}  "
                    f"{str(result.grads_finite):>6}  {peak_gib:>8}"
                )
            else:
                print(f"{num_envs:>8}  {num_steps:>9}  ERROR  {result.error}")

            if cfg.clear_caches_between_cases:
                jax.clear_caches()
                gc.collect()


if __name__ == "__main__":
    main(tyro.cli(Config))
