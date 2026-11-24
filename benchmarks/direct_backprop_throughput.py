"""Benchmark pathwise-gradient throughput over rollout batch and horizon grids.

This uses either a dense per-step actuator schedule or the production feedback
policy at initialization, and measures the direct baselines' transform order::

    vmap(value_and_grad(single_rollout))

The reported SPS is the number of requested environment transitions divided by
steady-state forward-plus-backward wall time. It excludes the Adam update and
evaluation rollouts. Environment initialization happens before timing. Identical
parameters and starting states are passed as dynamic arguments on every repeat;
this benchmarks the first gradient chunk, rather than a full learning curve.
Compilation plus first execution is timed separately. ``remat`` controls the
outer rollout-step checkpoint only; TORAX's internal checkpoints are unchanged.
Run each case in a fresh process for comparable peak device-memory measurements.
Run a representative GPU grid with::

    uv run python benchmarks/direct_backprop_throughput.py \
        --mode policy --env-setup iter/advanced/rampup --variant oracle \
        --backend bohm_gyrobohm \
        --num-envs 64 128 256 512 1024 \
        --num-steps 25 --repeats 5 \
        --output-json outputs/backprop-throughput.json
"""

from __future__ import annotations

import dataclasses
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import statistics
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any, Literal, NamedTuple

import jax
import jax.numpy as jnp
import optax
import tyro

from agents.backprop import BackpropPolicyAgent
from agents.direct_gradient import (
    finite_mean_gradients,
    make_knot_chunk,
    make_parameterization,
    tree_is_finite,
)
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
    mode: Literal["dense_schedule", "policy"] = "dense_schedule"
    hidden_sizes: tuple[int, ...] = (64, 64)
    num_envs: tuple[int, ...] = (64, 256, 1024)
    num_steps: tuple[int, ...] = (25, 50, 100)
    repeats: int = 5
    seed: int = 0
    remat: bool = True
    require_gpu: bool = True
    continue_on_error: bool = True
    clear_caches_between_cases: bool = True
    output_json: Path | None = None
    wandb_project: str | None = None
    wandb_entity: str | None = None
    wandb_group: str | None = None


@dataclasses.dataclass(frozen=True)
class CaseResult:
    """Timing and validity measurements for one static JAX shape."""

    num_envs: int
    num_steps: int
    status: str
    initialization_s: float | None = None
    compile_and_first_run_s: float | None = None
    durations_s: tuple[float, ...] = ()
    median_s: float | None = None
    requested_steps: int | None = None
    alive_steps: int | None = None
    alive_fraction: float | None = None
    backprop_sps: float | None = None
    alive_backprop_sps: float | None = None
    loss: float | None = None
    loss_finite: bool | None = None
    grad_norm: float | None = None
    grads_finite: bool | None = None
    aggregate_grads_finite: bool | None = None
    nonfinite_grad_elements: int | None = None
    peak_bytes_in_use: int | None = None
    error: str | None = None


class BackwardOutput(NamedTuple):
    loss: jax.Array
    grad_norm: jax.Array
    alive_steps: jax.Array
    grads_finite: jax.Array
    aggregate_grads_finite: jax.Array
    nonfinite_grad_elements: jax.Array
    next_carries: Any


BackwardPass = Callable[[], BackwardOutput]


def _validate_config(cfg: Config) -> None:
    if not cfg.num_envs or min(cfg.num_envs) < 1:
        raise ValueError("num_envs must contain positive integers")
    if not cfg.num_steps or min(cfg.num_steps) < 1:
        raise ValueError("num_steps must contain positive integers")
    if cfg.repeats < 1:
        raise ValueError("repeats must be positive")
    if cfg.variant not in ("realistic", "oracle"):
        raise ValueError("variant must be realistic or oracle")
    if not cfg.hidden_sizes or min(cfg.hidden_sizes) < 1:
        raise ValueError("hidden_sizes must contain positive integers")


def _make_backward_pass(
    env: Any,
    *,
    num_envs: int,
    num_steps: int,
    seed: int,
    remat: bool,
    mode: Literal["dense_schedule", "policy"] = "dense_schedule",
    hidden_sizes: tuple[int, ...] = (64, 64),
) -> BackwardPass:
    """Initialize fixed dynamic inputs, then return the compiled timed call."""
    rng = jax.random.key(seed)
    if mode == "policy":
        agent = BackpropPolicyAgent.create(
            env,
            num_rollouts=num_envs,
            episode_steps=num_steps,
            gradient_horizon=num_steps,
            total_timesteps=num_envs * num_steps,
            hidden_sizes=hidden_sizes,
            remat=remat,
        )
        parameters = agent.init_state(rng).params
        chunk = agent.chunk
        run_chunk = chunk.run
        # Production first-pass RNGs are prefix-stable when batch size changes.
        pass_key = jax.random.fold_in(rng, 0)
        keys = jax.vmap(lambda index: jax.random.fold_in(pass_key, index))(
            jnp.arange(num_envs)
        )
    else:
        to_actions, parameters, _ = make_parameterization(
            env, num_steps, n_knots=num_steps
        )
        keys = jax.random.split(rng, num_envs)
        chunk = make_knot_chunk(env, to_actions, num_steps, remat=remat)
        run_chunk = partial(chunk.run, start_step=jnp.asarray(0, dtype=jnp.int32))

    carries = jax.jit(jax.vmap(chunk.initialize))(keys)
    jax.block_until_ready((parameters, carries))

    def single_loss(current_parameters: Any, carry: Any) -> tuple:
        rollout_return, (next_carry, trajectory) = run_chunk(current_parameters, carry)
        return -rollout_return, (next_carry, jnp.sum(trajectory.alive))

    # Match production exactly: TORAX's adaptive-loop transpose supports
    # vmap(grad(single rollout)), rather than grad(vmap(rollout)).
    per_env_value_and_grad = jax.vmap(
        jax.value_and_grad(single_loss, has_aux=True),
        in_axes=(None, 0),
    )

    def backward(current_parameters: Any, initial_carries: Any) -> BackwardOutput:
        (losses, (next_carries, alive_steps)), per_env_grads = per_env_value_and_grad(
            current_parameters,
            initial_carries,
        )
        mean_grads, stats = finite_mean_gradients(per_env_grads)
        nonfinite_elements = stats.nonfinite_elements
        return BackwardOutput(
            jnp.mean(losses),
            optax.global_norm(mean_grads),
            jnp.sum(alive_steps),
            nonfinite_elements == 0,
            tree_is_finite(mean_grads),
            nonfinite_elements,
            next_carries,
        )

    return partial(jax.jit(backward), parameters, carries)


def _peak_bytes_in_use() -> int | None:
    stats = jax.local_devices()[0].memory_stats()
    if not stats:
        return None
    value = stats.get("peak_bytes_in_use")
    return None if value is None else int(value)


def _time_case(env: Any, cfg: Config, num_envs: int, num_steps: int) -> CaseResult:
    start = time.perf_counter()
    run = _make_backward_pass(
        env,
        num_envs=num_envs,
        num_steps=num_steps,
        seed=cfg.seed,
        remat=cfg.remat,
        mode=cfg.mode,
        hidden_sizes=cfg.hidden_sizes,
    )
    initialization_s = time.perf_counter() - start

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

    loss, grad_norm, alive_steps, grads_finite, aggregate_finite, nonfinite = (
        jax.device_get(output[:-1])
    )
    median_s = statistics.median(durations)
    requested_steps = num_envs * num_steps
    alive_steps_int = int(alive_steps)
    return CaseResult(
        num_envs=num_envs,
        num_steps=num_steps,
        status="ok",
        initialization_s=initialization_s,
        compile_and_first_run_s=compile_and_first_run_s,
        durations_s=tuple(durations),
        median_s=median_s,
        requested_steps=requested_steps,
        alive_steps=alive_steps_int,
        alive_fraction=alive_steps_int / requested_steps,
        backprop_sps=requested_steps / median_s,
        alive_backprop_sps=alive_steps_int / median_s,
        loss=float(loss),
        loss_finite=math.isfinite(float(loss)),
        grad_norm=float(grad_norm),
        grads_finite=bool(grads_finite),
        aggregate_grads_finite=bool(aggregate_finite),
        nonfinite_grad_elements=int(nonfinite),
        peak_bytes_in_use=_peak_bytes_in_use(),
    )


def _report_payload(cfg: Config, results: list[CaseResult]) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    source_paths = (
        "benchmarks/direct_backprop_throughput.py",
        "agents/backprop.py",
        "agents/direct_gradient.py",
        "uv.lock",
    )
    versions = {}
    for package in ("jax", "jaxlib", "torax", "flax", "optax", "jax-envelope"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    snapshot_path = os.environ.get("PLASMAX_SOURCE_SNAPSHOT")
    snapshot = None
    if snapshot_path:
        snapshot = json.loads(Path(snapshot_path).read_text())
    return {
        "config": dataclasses.asdict(cfg),
        "host": platform.platform(),
        "python_version": platform.python_version(),
        "jax_version": jax.__version__,
        "software_versions": versions,
        "devices": [str(device) for device in jax.devices()],
        "device_models": [device.device_kind for device in jax.devices()],
        "source_sha256": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in source_paths
        },
        "source_snapshot": snapshot,
        "runtime_environment": {
            name: os.environ.get(name)
            for name in (
                "XLA_FLAGS",
                "JAX_ENABLE_X64",
                "XLA_PYTHON_CLIENT_PREALLOCATE",
                "XLA_PYTHON_CLIENT_MEM_FRACTION",
            )
        },
        "semantics": {
            "timed_work": "rollout, per-rollout gradients, gradient aggregation",
            "transform": "vmap(value_and_grad(single_rollout))",
            "inputs": "fixed initial parameters and first-chunk carries as JIT args",
            "excluded_work": "environment initialization, optimizer, evaluation",
            "remat": "outer rollout-step checkpoint; inner TORAX checkpoints unchanged",
            "peak_memory": "device allocator lifetime peak; use fresh process per case",
            "policy_aggregation": "production finite_mean_gradients",
        },
        "results": [dataclasses.asdict(result) for result in results],
    }


def _write_report(cfg: Config, results: list[CaseResult]) -> None:
    if cfg.output_json is None:
        return
    cfg.output_json.parent.mkdir(parents=True, exist_ok=True)
    cfg.output_json.write_text(
        json.dumps(_report_payload(cfg, results), indent=2, default=str) + "\n"
    )


def _start_wandb(cfg: Config, num_envs: int, num_steps: int) -> Any:
    if cfg.wandb_project is None:
        return None
    import wandb

    payload = json.loads(json.dumps(_report_payload(cfg, []), default=str))
    config = dict(payload["config"])
    config.update(num_envs=num_envs, num_steps=num_steps)
    run = wandb.init(
        project=cfg.wandb_project,
        entity=cfg.wandb_entity,
        group=cfg.wandb_group,
        job_type="backprop-throughput",
        name=f"{cfg.mode}-batch{num_envs}-steps{num_steps}-remat{int(cfg.remat)}",
        config=config,
        mode="online",
    )
    if run is None or run.settings.mode != "online":
        raise RuntimeError("W&B online initialization failed")
    run.summary.update(
        {key: value for key, value in payload.items() if key != "results"}
    )
    return run


def _finish_wandb(run: Any, result: CaseResult) -> None:
    if run is None:
        return
    for index, duration in enumerate(result.durations_s):
        run.log({"benchmark/repeat": index, "benchmark/repeat_s": duration})
    metrics = {
        f"benchmark/{key}": value
        for key, value in dataclasses.asdict(result).items()
        if key != "durations_s" and value is not None
    }
    run.log(metrics)
    run.summary.update(metrics)
    run.finish(exit_code=0 if result.status == "ok" else 1)


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
        f"mode={cfg.mode} remat={cfg.remat} repeats={cfg.repeats} "
        f"num_envs={list(cfg.num_envs)} num_steps={list(cfg.num_steps)}"
    )
    print(
        "num_envs  num_steps  compile+first  median backward  backprop SPS  "
        "alive SPS  finite  peak GiB"
    )

    results: list[CaseResult] = []
    for num_steps in cfg.num_steps:
        for num_envs in cfg.num_envs:
            # Authorization/init failures are fatal: never fall back to offline.
            run = _start_wandb(cfg, num_envs, num_steps)
            try:
                result = _time_case(env, cfg, num_envs, num_steps)
            except Exception as exc:
                result = CaseResult(
                    num_envs=num_envs,
                    num_steps=num_steps,
                    status="error",
                    error=f"{type(exc).__name__}: {exc}",
                )
                results.append(result)
                _write_report(cfg, results)
                _finish_wandb(run, result)
                if not cfg.continue_on_error:
                    raise
            else:
                results.append(result)
                _write_report(cfg, results)
                _finish_wandb(run, result)

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
