"""Benchmark steady-state environment-step throughput.

The timed region is a JIT-compiled ``lax.scan`` over the unwrapped environment.
Every scan iteration requests one physical control transition. Throughput is
reported only for transitions that complete their full configured duration. If
a transition terminates, the next iteration starts from a reset state; terminal
slots are never replaced by cheap padding.

Run on CPU:

    JAX_PLATFORMS=cpu uv run python benchmarks/env_throughput.py \
        --env-setup iter/hybrid/flattop \
        --backend cgm \
        --n-envs 1 4 16
"""

from __future__ import annotations

import dataclasses
import platform
import statistics
import time
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import tyro

from plasmax.environment.factory import load_env

SEED = 0


@dataclasses.dataclass(frozen=True)
class Config:
    env_setup: str = "iter/hybrid/flattop"
    backend: str = "cgm"
    validate_backend_pair: bool = True
    n_steps: int = 100
    n_envs: tuple[int, ...] = (1, 4, 16)
    repeats: int = 5
    reset_on_boundary: bool = True
    vmap_scalar: bool = True
    require_cpu: bool = True


@dataclasses.dataclass(frozen=True)
class Timing:
    n_envs: int
    compile_and_first_run_s: float
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
    action = state.prev_action

    def _step(carry, step_index):
        state, action = carry
        next_state, info = env.step(state, action)
        boundary = info.terminated | info.truncated
        reset_key = jax.random.fold_in(key, step_index + 1)

        def _reset(_):
            reset_state, _ = env.reset(next_state, reset_key)
            return reset_state, reset_state.prev_action

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
) -> Callable[[], Any]:
    keys = jax.random.split(jax.random.key(SEED), n_envs)

    @jax.jit
    def run(keys):
        if n_envs == 1 and not vmap_scalar:
            return _rollout(env, keys[0], n_steps, reset_on_boundary)
        return jax.vmap(lambda key: _rollout(env, key, n_steps, reset_on_boundary))(
            keys
        )

    return lambda: run(keys)


def _time_run(env: Any, cfg: Config, n_envs: int) -> Timing:
    run = _make_run(
        env,
        cfg.n_steps,
        n_envs,
        cfg.reset_on_boundary,
        cfg.vmap_scalar,
    )

    start = time.perf_counter()
    out = run()
    jax.block_until_ready(out)
    compile_and_first_run_s = time.perf_counter() - start

    durations = []
    for _ in range(cfg.repeats):
        start = time.perf_counter()
        out = run()
        jax.block_until_ready(out)
        durations.append(time.perf_counter() - start)

    boundary_counts = jax.device_get(out[1])
    completed_counts = jax.device_get(out[2])
    return Timing(
        n_envs=n_envs,
        compile_and_first_run_s=compile_and_first_run_s,
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
    wrapped_env = load_env(
        cfg.env_setup,
        cfg.backend,
        validate=cfg.validate_backend_pair,
    )
    env = wrapped_env.unwrapped
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

    for n_envs in cfg.n_envs:
        timing = _time_run(env, cfg, n_envs)
        median_s = statistics.median(timing.durations_s)
        throughput = timing.completed_steps / median_s
        low_s = min(timing.durations_s)
        high_s = max(timing.durations_s)
        print(
            f"{n_envs:>6}  {timing.compile_and_first_run_s:>13.3f}s  "
            f"{median_s:>10.4f}s  {throughput:>9.1f}  "
            f"{timing.completed_steps:>9}  {timing.boundaries:>10}  "
            f"{low_s:.4f}-{high_s:.4f}s",
            flush=True,
        )


if __name__ == "__main__":
    main(tyro.cli(Config))
