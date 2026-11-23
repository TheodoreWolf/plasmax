"""Measure full KSTAR world-model environment steps with independent inputs.

Each step runs the native 19 LSTM ensemble predictions and the beta-power head,
observation, and reward. Sample one distinct action per environment before
timing, then hold that action throughout the rollout. Reset keys alone would
only vary targets and would leave the learned dynamics identical across a batch.

Run on one allocated GPU with::

    uv run python -m benchmarks.world_model_throughput \
        --n-envs 1 4 16 64 256 1024 4096 16384 \
        --n-steps 10 --repeats 5 --no-require-cpu \
        --hardware-label H100 --output-json outputs/world_model.json

The JIT-compiled vmap of scans includes initialization. All outputs are
synchronized before stopping the timer. Observations and rewards at every step
remain observable outputs, so their model heads cannot be pruned. Host validation
and input generation occur outside timing; any invalid repeat fails the case.
Cold throughput includes separately timed tracing/lowering, backend compilation,
and first execution.
This benchmark assigns no physical duration to a learned environment step.
"""

from __future__ import annotations

import dataclasses
import hashlib
import math
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import tyro

from benchmarks import env_throughput as common
from plasmax import make
from plasmax.models import world_model, world_model_env


@dataclasses.dataclass(frozen=True)
class Config:
    """Static batch shapes and repeated steady-state timing protocol."""

    env_setup: Literal["kstar_worldmodel"] = "kstar_worldmodel"
    backend: None = dataclasses.field(default=None, init=False)
    n_steps: int = 10
    n_envs: tuple[int, ...] = (1, 4, 16, 64, 256, 1024, 4096, 16384)
    repeats: int = 5
    seed: int = 0
    require_cpu: bool = True
    output_json: Path | None = None
    hardware_label: str | None = None


class RolloutOutput(NamedTuple):
    final_state: Any
    observations: jax.Array
    rewards: jax.Array
    history_finite: jax.Array
    step_counters: jax.Array
    boundaries: jax.Array
    initial_observation: jax.Array
    initial_reward: jax.Array
    initial_history_finite: jax.Array


@dataclasses.dataclass(frozen=True)
class InputMetadata:
    actions_sha256: str
    reset_keys_sha256: str
    actions_shape: tuple[int, ...]
    actions_dtype: str
    unique_action_rows: int
    generator: str = "NumPy PCG64(seed).uniform(-1, 1), cast to float32"
    hash_encoding: str = "C-order little-endian float32 actions / uint32 key data"


@dataclasses.dataclass(frozen=True)
class PreparedRun:
    run: Callable[[jax.Array, jax.Array], RolloutOutput]
    keys: jax.Array
    actions: jax.Array
    metadata: InputMetadata


@dataclasses.dataclass(frozen=True)
class Validation:
    completed_steps: int
    boundaries: int
    history_finite: bool
    observations_finite: bool
    rewards_finite: bool
    step_counters_match: bool
    final_t_min: int
    final_t_max: int
    observation_checksum: float
    reward_checksum: float
    final_history_checksum: float
    distinct_final_prediction_rows: int


@dataclasses.dataclass(frozen=True)
class CaseResult:
    timing: common.Timing
    inputs: InputMetadata
    warmup_validation: Validation
    repeat_validations: tuple[Validation, ...]


def _rollout(
    env: Any, key: jax.Array, action: jax.Array, n_steps: int
) -> RolloutOutput:
    state, initial_info = env.init(key)
    initial_history_finite = jnp.all(jnp.isfinite(state.x))

    def step(state: Any, _: None) -> tuple[Any, tuple[jax.Array, ...]]:
        next_state, info = env.step(state, action)
        return next_state, (
            info.obs,
            info.reward,
            jnp.all(jnp.isfinite(next_state.x)),
            next_state.t,
            info.terminated | info.truncated,
        )

    final_state, outputs = jax.lax.scan(step, state, None, length=n_steps)
    return RolloutOutput(
        final_state,
        *outputs,
        initial_info.obs,
        initial_info.reward,
        initial_history_finite,
    )


def _make_run(env: Any, cfg: Config, n_envs: int) -> PreparedRun:
    """Prepare independent actions and pass them as explicit dynamic JIT args."""
    action_shape = (n_envs, *env.action_space.shape)
    action_values = np.random.default_rng(cfg.seed).uniform(-1.0, 1.0, action_shape)
    action_values = np.asarray(action_values, dtype="<f4")
    unique_actions = np.unique(action_values, axis=0).shape[0]
    if unique_actions != n_envs:
        raise ValueError("Every environment must receive a distinct action row")
    actions = jax.device_put(action_values)
    keys = jax.random.split(jax.random.key(cfg.seed), n_envs)
    jax.block_until_ready((keys, actions))
    key_values = np.asarray(jax.device_get(jax.random.key_data(keys)), dtype="<u4")

    def rollout(key: jax.Array, action: jax.Array) -> RolloutOutput:
        return _rollout(env, key, action, cfg.n_steps)

    metadata = InputMetadata(
        actions_sha256=hashlib.sha256(action_values.tobytes(order="C")).hexdigest(),
        reset_keys_sha256=hashlib.sha256(key_values.tobytes(order="C")).hexdigest(),
        actions_shape=action_shape,
        actions_dtype="float32",
        unique_action_rows=int(unique_actions),
    )
    return PreparedRun(jax.jit(jax.vmap(rollout)), keys, actions, metadata)


def _validate_output(
    output: RolloutOutput,
    *,
    n_envs: int,
    n_steps: int,
    observation_shape: tuple[int, ...],
    label: str,
) -> Validation:
    """Validate synchronized outputs on the host, outside measured wall time."""
    output = jax.device_get(output)
    expected_step_shape = (n_envs, n_steps)
    expected_obs_shape = (*expected_step_shape, *observation_shape)
    expected_initial_obs_shape = (n_envs, *observation_shape)
    shapes_match = (
        output.observations.shape == expected_obs_shape
        and output.initial_observation.shape == expected_initial_obs_shape
        and output.initial_reward.shape == (n_envs,)
        and output.initial_history_finite.shape == (n_envs,)
        and output.final_state.t.shape == (n_envs,)
        and output.final_state.x.shape[0] == n_envs
        and all(
            value.shape == expected_step_shape
            for value in (
                output.rewards,
                output.history_finite,
                output.step_counters,
                output.boundaries,
            )
        )
    )
    if not shapes_match:
        raise ValueError(f"{label}: rollout output shapes do not match requested work")
    history_finite = bool(
        np.all(output.initial_history_finite)
        and np.all(output.history_finite)
        and np.all(np.isfinite(output.final_state.x))
    )
    observations_finite = bool(
        np.all(np.isfinite(output.initial_observation))
        and np.all(np.isfinite(output.observations))
    )
    rewards_finite = bool(
        np.all(np.isfinite(output.initial_reward))
        and np.all(np.isfinite(output.rewards))
    )
    if not (history_finite and observations_finite and rewards_finite):
        raise ValueError(
            f"{label}: nonfinite rollout outputs "
            f"(history={history_finite}, observations={observations_finite}, "
            f"rewards={rewards_finite})"
        )
    expected_counters = np.broadcast_to(
        np.arange(1, n_steps + 1, dtype=np.int32), expected_step_shape
    )
    counters_match = bool(
        np.array_equal(output.step_counters, expected_counters)
        and np.all(output.final_state.t == n_steps)
    )
    boundaries = int(np.count_nonzero(output.boundaries))
    if not counters_match or boundaries:
        raise ValueError(
            f"{label}: rollout did not complete every requested transition "
            f"(step_counters_match={counters_match}, boundaries={boundaries})"
        )
    return Validation(
        completed_steps=n_envs * n_steps,
        boundaries=boundaries,
        history_finite=history_finite,
        observations_finite=observations_finite,
        rewards_finite=rewards_finite,
        step_counters_match=counters_match,
        final_t_min=int(np.min(output.final_state.t)),
        final_t_max=int(np.max(output.final_state.t)),
        observation_checksum=float(np.sum(output.observations, dtype=np.float64)),
        reward_checksum=float(np.sum(output.rewards, dtype=np.float64)),
        final_history_checksum=float(np.sum(output.final_state.x, dtype=np.float64)),
        distinct_final_prediction_rows=int(
            np.unique(output.final_state.x[:, -1, :4], axis=0).shape[0]
        ),
    )


def _time_run(env: Any, cfg: Config, n_envs: int) -> CaseResult:
    prepared = _make_run(env, cfg, n_envs)
    start = time.perf_counter()
    lowered = prepared.run.lower(prepared.keys, prepared.actions)
    trace_lower_s = time.perf_counter() - start
    start = time.perf_counter()
    compiled = lowered.compile()
    compile_s = time.perf_counter() - start

    def execute(label: str) -> tuple[float, Validation]:
        start = time.perf_counter()
        output = compiled(prepared.keys, prepared.actions)
        jax.block_until_ready(output)
        elapsed = time.perf_counter() - start
        if not math.isfinite(elapsed) or elapsed <= 0:
            raise ValueError(f"{label}: expected positive finite elapsed time")
        validation = _validate_output(
            output,
            n_envs=n_envs,
            n_steps=cfg.n_steps,
            observation_shape=env.observation_space.shape,
            label=label,
        )
        return elapsed, validation

    first_execution_s, warmup = execute("first execution")
    durations = []
    validations = []
    for repeat in range(cfg.repeats):
        elapsed, validation = execute(f"timed repeat {repeat + 1}")
        durations.append(elapsed)
        validations.append(validation)
    timing = common.Timing(
        n_envs=n_envs,
        compile_and_first_run_s=trace_lower_s + compile_s + first_execution_s,
        trace_lower_s=trace_lower_s,
        compile_s=compile_s,
        first_execution_s=first_execution_s,
        first_completed_steps=warmup.completed_steps,
        durations_s=tuple(durations),
        boundaries=0,
        completed_steps=n_envs * cfg.n_steps,
    )
    return CaseResult(timing, prepared.metadata, warmup, tuple(validations))


def _file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _provenance(env: Any, cfg: Config) -> dict[str, Any]:
    provenance = common._provenance()
    root = Path(__file__).resolve().parents[1]
    weights = env.plasmax_config.world_model.weights_path
    sources = (
        Path(__file__).resolve(),
        Path(world_model.__file__).resolve(),
        Path(world_model_env.__file__).resolve(),
        env.plasmax_config.initialization,
        weights,
    )
    for path in sources:
        name = str(path.relative_to(root)) if path.is_relative_to(root) else str(path)
        provenance["source_sha256"][name] = _file_sha256(path)
    provenance["weights"] = {
        "path": str(weights),
        "sha256": _file_sha256(weights),
        "size_bytes": weights.stat().st_size,
    }
    provenance["seed"] = cfg.seed
    provenance["jax_default_matmul_precision"] = jax.config.jax_default_matmul_precision
    provenance["software_versions"]["numpy"] = np.__version__
    return provenance


def _environment_metadata(env: Any) -> dict[str, Any]:
    bundle = env._dynamics._bundle
    return {
        "wrapper": None,
        "action_policy": (
            "one independently sampled uniform[-1,1] float32 action per environment, "
            "held across the rollout; sampled outside timing, "
            "passed as dynamic JIT args"
        ),
        "target_policy": (
            "independent seeded reset keys; targets vary, initial history fixed"
        ),
        "timed_work": (
            "initialization plus JIT(vmap(scan(env.step))), retaining every step's "
            "observation and reward, final state, "
            "and intermediate-history finite checks"
        ),
        "excluded_work": (
            "input generation/transfer, compilation warmup, host validation"
        ),
        "throughput_numerator": "validated completed native environment transitions",
        "control_interval_s": None,
        "physical_duration": "not assigned to learned environment transitions",
        "num_lstm_calls_per_step": world_model_env.INTERVAL - 1,
        "num_bpw_calls_per_step": 1,
        "ensemble_members": int(bundle["lstm"]["W0"].shape[0]),
        "bpw_ensemble_members": int(bundle["bpw"]["W1"].shape[0]),
        "model_dtype": str(bundle["lstm"]["W0"].dtype),
        "target_sampling_dtype": "float64",
        "retained_outputs": (
            "initial observation/reward; every step observation/reward, history-finite "
            "flag, step counter and boundary; final environment state"
        ),
        "horizon_wrapper": None,
        "observation_shape": env.observation_space.shape,
        "action_shape": env.action_space.shape,
        "world_model_config": env.plasmax_config.model_dump(),
    }


def _validate_config(cfg: Config) -> None:
    if cfg.env_setup != "kstar_worldmodel":
        raise ValueError("env_setup must be kstar_worldmodel")
    if cfg.n_steps < 1 or cfg.repeats < 1:
        raise ValueError("n_steps and repeats must be positive")
    if not cfg.n_envs or min(cfg.n_envs) < 1 or len(set(cfg.n_envs)) != len(cfg.n_envs):
        raise ValueError("n_envs must contain unique positive integers")
    if cfg.seed < 0:
        raise ValueError("seed must be nonnegative")
    if cfg.require_cpu and any(device.platform != "cpu" for device in jax.devices()):
        raise RuntimeError("CPU benchmark requested; run with JAX_PLATFORMS=cpu")
    if not jax.config.jax_enable_x64:
        raise RuntimeError("World-model target sampling requires JAX float64 support")


def _case_payload(cfg: Config, case: CaseResult) -> dict[str, Any]:
    return {
        **common._timing_payload(cfg, case.timing),
        "inputs": dataclasses.asdict(case.inputs),
        "validation": {
            "warmup": dataclasses.asdict(case.warmup_validation),
            "repeats": [dataclasses.asdict(item) for item in case.repeat_validations],
        },
    }


def main(cfg: Config) -> None:
    _validate_config(cfg)
    env = make(cfg.env_setup)
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "started_at": datetime.now(UTC).isoformat(),
        "config": dataclasses.asdict(cfg),
        "provenance": _provenance(env, cfg),
        "environment": _environment_metadata(env),
        "results": [],
    }
    common._write_report(cfg, report)
    print(
        f"env={cfg.env_setup} devices={jax.devices()} "
        f"steps/env={cfg.n_steps} repeats={cfg.repeats}",
        flush=True,
    )
    print("n_envs  compile+first  median run    steps/s  completed", flush=True)
    try:
        for n_envs in cfg.n_envs:
            case = _time_run(env, cfg, n_envs)
            row = _case_payload(cfg, case)
            report["results"].append(row)
            common._write_report(cfg, report)
            print(
                f"{n_envs:>6}  {row['compile_and_first_run_s']:>13.3f}s  "
                f"{row['median_s']:>10.4f}s  {row['steps_per_second']:>9.1f}  "
                f"{row['completed_steps']:>9}",
                flush=True,
            )
    except BaseException as error:
        report["status"] = "error"
        report["error"] = f"{type(error).__name__}: {error}"
        common._write_report(cfg, report)
        raise
    report["status"] = "complete"
    report["finished_at"] = datetime.now(UTC).isoformat()
    common._write_report(cfg, report)


if __name__ == "__main__":
    main(tyro.cli(Config))
