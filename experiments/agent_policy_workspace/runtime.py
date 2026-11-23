"""Stateful JAX policies evaluated in fault-isolated subprocesses."""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import math
import os
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from experiments.agent_policy_search.artifacts import read_json, write_json
from experiments.agent_policy_search.config import EvaluationConfig, TaskConfig
from experiments.agent_policy_search.evaluator import (
    _concatenate_banks,
    _descriptor,
    _failure,
    _merge_seed_results,
    _report,
)
from experiments.agent_policy_search.feedback import json_safe

__all__ = ["describe_task", "evaluate_bundle"]


def describe_task(output_dir: Path, config: EvaluationConfig) -> dict[str, Any]:
    """Inspect the configured interface without loading a policy or stepping."""
    if not math.isfinite(config.timeout_seconds) or config.timeout_seconds <= 0:
        raise ValueError("evaluation timeout must be positive and finite")
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        output_dir / "request.json",
        {
            "source": None,
            "output_dir": output_dir,
            "config": config,
            "validate_only": True,
        },
    )
    report = _subprocess(output_dir, config.timeout_seconds)
    write_json(output_dir / "result.json", report)
    return report


def evaluate_bundle(
    source: Path,
    output_dir: Path,
    config: EvaluationConfig,
    *,
    validate_only: bool = False,
) -> dict[str, Any]:
    """Validate or score a policy bundle without importing it in the caller."""
    source, output_dir = source.resolve(), output_dir.resolve()
    if not config.seeds or any(seed < 0 or seed >= 2**32 for seed in config.seeds):
        raise ValueError("evaluation requires uint32 seeds")
    if not math.isfinite(config.timeout_seconds) or config.timeout_seconds <= 0:
        raise ValueError("evaluation timeout must be positive and finite")
    deadline = time.monotonic() + config.timeout_seconds
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        output_dir / "request.json",
        {
            "source": source,
            "output_dir": output_dir,
            "config": config,
            "validate_only": validate_only,
        },
    )
    if (
        not validate_only
        and "tglfnn" in (config.task.backend or "").lower()
        and len(config.seeds) > 1
    ):
        reports = []
        for index, seed in enumerate(config.seeds):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                report = _failure("timeout", "Evaluation exhausted its bank timeout")
                break
            report = evaluate_bundle(
                source,
                output_dir / f"seed_{index:04d}_{seed}",
                dataclasses.replace(config, seeds=(seed,), timeout_seconds=remaining),
            )
            reports.append(report)
            if report["status"] != "ok":
                break
        else:
            report = _merge_seed_results(output_dir, config, reports)
    else:
        report = _subprocess(output_dir, config.timeout_seconds)
    write_json(output_dir / "result.json", report)
    return report


def _subprocess(output_dir: Path, timeout: float) -> dict[str, Any]:
    # Credential filtering is not a filesystem or network security boundary.
    environment = {
        name: value
        for name, value in os.environ.items()
        if not name.endswith(("_API_KEY", "_API_TOKEN"))
    }
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    command = json.loads(environment.pop("PLASMAX_POLICY_WORKER_COMMAND", "null"))
    if command is None:
        command = [sys.executable, "-m", "experiments.agent_policy_workspace.runtime"]
    if (
        not isinstance(command, list)
        or not command
        or any(not isinstance(part, str) or not part for part in command)
    ):
        raise ValueError("PLASMAX_POLICY_WORKER_COMMAND must be a JSON string array")
    started = time.monotonic()
    request_path = output_dir / "request.json"
    request = read_json(request_path)
    request["deadline_unix_seconds"] = time.time() + timeout
    request_path.write_text(json.dumps(request, indent=2) + "\n")
    with (
        (output_dir / "stdout.txt").open("x") as stdout,
        (output_dir / "stderr.txt").open("x") as stderr,
    ):
        try:
            process = subprocess.run(
                [*command, "--request", str(output_dir / "request.json")],
                cwd=Path(__file__).resolve().parents[2],
                env=environment,
                stdout=stdout,
                stderr=stderr,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            report = _failure("timeout", f"Evaluation exceeded {timeout:g} seconds")
        else:
            result = output_dir / "worker_result.json"
            report = (
                read_json(result)
                if result.exists()
                else _failure(
                    "error", f"Worker exited {process.returncode}; see stderr.txt"
                )
            )
    descriptor = output_dir / "interface.json"
    if not report.get("interface") and descriptor.exists():
        report["interface"] = read_json(descriptor)
    report["evaluation_seconds"] = time.monotonic() - started
    return report


def _load_policy(source: Path) -> Any:
    spec = importlib.util.spec_from_file_location("_workspace_policy", source)
    if spec is None or spec.loader is None:
        raise ValueError("Cannot load policy.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    previous_helper = sys.modules.pop("custom_primitives", None)
    sys.path.insert(0, str(source.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
        sys.modules.pop("custom_primitives", None)
        if previous_helper is not None:
            sys.modules["custom_primitives"] = previous_helper
    if getattr(module, "POLICY_API_VERSION", None) != 1:
        raise ValueError("policy.py must declare POLICY_API_VERSION = 1")
    if not all(
        callable(getattr(module, name, None)) for name in ("init_memory", "act")
    ):
        raise ValueError(
            "policy.py must define init_memory() and act(obs, reset, memory, rng)"
        )
    return module


def _validate(policy: Any, env: Any) -> tuple[Any, dict[str, Any]]:
    import jax
    import jax.numpy as jnp
    import numpy as np

    memory = jax.eval_shape(policy.init_memory)
    action, next_memory, trace = jax.eval_shape(
        policy.act,
        jax.ShapeDtypeStruct(env.observation_space.shape, env.observation_space.dtype),
        jnp.asarray(False),
        memory,
        jax.random.key(0),
    )
    if getattr(action, "shape", None) != env.action_space.shape or np.dtype(
        action.dtype
    ) != np.dtype(env.action_space.dtype):
        raise ValueError("policy action shape/dtype must match the environment")
    before, before_tree = jax.tree.flatten(memory)
    after, after_tree = jax.tree.flatten(next_memory)
    if before_tree != after_tree or any(
        (old.shape, old.dtype) != (new.shape, new.dtype)
        for old, new in zip(before, after, strict=True)
    ):
        raise ValueError("policy memory structure, shapes and dtypes must stay fixed")
    if any(np.dtype(leaf.dtype).kind not in "biuf" for leaf in before):
        raise ValueError("policy memory must contain numeric or boolean arrays")
    if not isinstance(trace, dict) or any(
        not isinstance(name, str)
        or not re.fullmatch(r"[A-Za-z_]\w*", name)
        or not hasattr(value, "dtype")
        or np.dtype(value.dtype).kind not in "biuf"
        for name, value in trace.items()
    ):
        raise ValueError("trace must map identifier names to numeric or boolean arrays")
    return memory, trace


def _numeric_trajectory_arrays(trajectory: Any, env: Any) -> dict[str, Any]:
    """Match trajectory_arrays without host transfers or string metadata."""
    import jax.numpy as jnp

    from plasmax.wrappers import unwrap_to_env_state

    arrays = {
        name: getattr(trajectory, name)
        for name in (
            "obs",
            "action",
            "reward",
            "next_obs",
            "terminated",
            "truncated",
            "valid",
            "done",
        )
    }
    arrays["command_physical"] = arrays["action"]
    base = unwrap_to_env_state(trajectory.env_state)
    if hasattr(base, "prev_action"):
        arrays["applied_physical"] = base.prev_action
    if hasattr(env, "from_physical"):
        arrays["requested_action"] = env.from_physical(trajectory.action)
        if jnp.issubdtype(env.action_space.dtype, jnp.floating):
            arrays["command_normalized"] = arrays["requested_action"]
        if "applied_physical" in arrays and jnp.issubdtype(
            env.action_space.dtype, jnp.floating
        ):
            arrays["applied_normalized"] = env.from_physical(base.prev_action)
    else:
        arrays["requested_action"] = arrays["action"]
    plasma = getattr(base, "plasma", None)
    if plasma is not None:
        arrays["time_s"] = plasma.t
        for name in (
            "Q_fusion",
            "P_fusion",
            "W_thermal_total",
            "q_min",
            "q95",
            "beta_N",
            "P_aux_total",
            "fgw_n_e_line_avg",
        ):
            if hasattr(plasma, name):
                arrays[name] = getattr(plasma, name)
        current = getattr(plasma.sim.core_profiles, "Ip_profile_face", None)
        if current is not None:
            arrays["Ip"] = current[..., -1]
    elif (
        hasattr(env, "obs_layout") and "elapsed_time" in env.obs_layout().scalar_slices
    ):
        arrays["time_s"] = arrays["next_obs"][
            ..., env.obs_layout().slice_of("elapsed_time").start
        ]
    else:
        arrays["time_steps"] = jnp.cumsum(arrays["valid"], axis=-1)
    if hasattr(trajectory.info, "termination_code"):
        arrays["termination_code"] = trajectory.info.termination_code
    return arrays


def _collector(policy: Any, env: Any, horizon: int, trace_shape: dict[str, Any]) -> Any:
    import jax
    import jax.numpy as jnp

    from plasmax.rollout import TrajectoryStep, _padding_info, _reject_autoreset

    _reject_autoreset(env)
    zero_action = jnp.zeros(env.action_space.shape, env.action_space.dtype)
    to_physical = getattr(env, "to_physical", lambda value: value)
    zero_trace = jax.tree.map(
        lambda value: jnp.zeros(value.shape, value.dtype), trace_shape
    )

    @jax.jit
    def collect(key: Any) -> tuple[Any, dict[str, Any]]:
        environment_key, policy_key = jax.random.split(key)
        state, info = env.init(environment_key)
        carry = (
            state,
            info.obs,
            policy.init_memory(),
            policy_key,
            ~(info.terminated | info.truncated),
            jnp.asarray(True),
            info,
        )

        def real_step(carry: Any) -> tuple[Any, Any]:
            state, obs, memory, key, _, reset, _ = carry
            key, action_key = jax.random.split(key)
            action, memory, trace = policy.act(obs, reset, memory, action_key)
            next_state, info = env.step(state, action)
            active = ~(info.terminated | info.truncated)
            step = TrajectoryStep(
                obs,
                to_physical(action),
                info.reward,
                info.obs,
                info.terminated,
                info.truncated,
                jnp.asarray(True),
                next_state,
                info,
            )
            return (
                next_state,
                info.obs,
                memory,
                key,
                active,
                jnp.asarray(False),
                info,
            ), (step, trace)

        def padding_step(carry: Any) -> tuple[Any, Any]:
            state, obs, _, _, _, _, last_info = carry
            info = _padding_info(last_info, obs)
            step = TrajectoryStep(
                obs,
                to_physical(zero_action),
                info.reward,
                obs,
                info.terminated,
                info.truncated,
                jnp.asarray(False),
                state,
                info,
            )
            return carry, (step, zero_trace)

        def step(carry: Any, _: Any) -> tuple[Any, Any]:
            return jax.lax.cond(carry[4], real_step, padding_step, carry)

        _, (trajectory, traces) = jax.lax.scan(step, carry, None, length=horizon)
        # Keep full simulator state in the carry, not in the returned history.
        return _numeric_trajectory_arrays(trajectory, env), traces

    return collect


def _evaluate_in_worker(
    source: Path | None,
    output_dir: Path,
    config: EvaluationConfig,
    validate_only: bool = False,
) -> dict[str, Any]:
    import jax
    import numpy as np

    from training.evaluation import _horizon
    from training.runs import EnvConfig, load_env

    env = load_env(EnvConfig(**dataclasses.asdict(config.task)), config.task.backend)
    interface = _descriptor(env, config.task)
    interface["policy_contract"] = (
        "init_memory(); act(obs, reset, memory, rng) -> (action, next_memory, trace)"
    )
    write_json(output_dir / "interface.json", interface)
    if source is None:
        return {"status": "ok", "interface": interface}
    policy = _load_policy(source)
    _, trace_shape = _validate(policy, env)
    trace_schema = {
        name: {"shape": list(value.shape), "dtype": str(value.dtype)}
        for name, value in trace_shape.items()
    }
    write_json(output_dir / "trace_schema.json", trace_schema)
    if validate_only:
        return {
            "status": "ok",
            "finite": True,
            "mean_return": None,
            "feedback": {},
            "interface": interface,
            "trace_schema": trace_schema,
        }
    resolved = getattr(env, "plasmax_config", None)
    if resolved is not None:
        write_json(
            output_dir / "resolved_config.json", resolved.model_dump(mode="json")
        )
    collect = _collector(policy, env, _horizon(env, None), trace_shape)
    batch_size = (
        min(64, len(config.seeds)) if config.task.backend == "bohm_gyrobohm" else 1
    )
    if batch_size > 1:
        collect = jax.jit(jax.vmap(collect))
    banks, returns = [], []
    for start in range(0, len(config.seeds), batch_size):
        seeds = config.seeds[start : start + batch_size]
        print(
            f"Collecting episodes {start + 1}-{start + len(seeds)}/{len(config.seeds)}",
            flush=True,
        )
        # Match evaluate_policy(..., num_episodes=1)'s episode-key derivation.
        keys = jax.numpy.stack(
            [jax.random.split(jax.random.key(seed), 1)[0] for seed in seeds]
        )
        arrays, traces = collect(keys if batch_size > 1 else keys[0])
        arrays.update({f"trace_{name}": value for name, value in traces.items()})
        if batch_size == 1:
            arrays = jax.tree.map(lambda value: value[None], arrays)
        arrays = jax.device_get(arrays)
        if hasattr(env, "actuator_specs"):
            arrays["actuator_names"] = np.asarray(
                [spec.name for spec in env.actuator_specs]
            )
        banks.append(arrays)
        for offset, seed in enumerate(seeds):
            episode_arrays = {
                name: value if name == "actuator_names" else value[offset : offset + 1]
                for name, value in arrays.items()
            }
            with np.errstate(invalid="ignore", over="ignore"):
                episode_return = np.where(
                    episode_arrays["valid"], episode_arrays["reward"], 0
                ).sum()
            returns.append(episode_return)
            episode = output_dir / "episodes" / f"{start + offset:04d}_seed_{seed}"
            episode.parent.mkdir(exist_ok=True)
            np.savez_compressed(episode.with_suffix(".npz"), **episode_arrays)
            write_json(
                episode.with_suffix(".json"),
                json_safe({"seed": seed, "return": episode_return}),
            )
        print(f"Saved {start + len(seeds)}/{len(config.seeds)} episodes", flush=True)
        del traces
    arrays = _concatenate_banks(banks)
    np.savez_compressed(
        output_dir / "trajectories.npz",
        **arrays,
        interface_json=np.asarray(json.dumps(interface)),
        policy_trace_schema_json=np.asarray(json.dumps(trace_schema)),
        metadata_json=np.asarray(
            json.dumps(
                {
                    "seeds": config.seeds,
                    "task": dataclasses.asdict(config.task),
                    "evaluation_batch_size": batch_size,
                }
            )
        ),
    )
    return _report(arrays, interface, config.seeds, np.asarray(returns))


def _worker_main(request: Path) -> None:
    payload = read_json(request)
    settings = payload["config"]
    config = EvaluationConfig(
        TaskConfig(**settings["task"]),
        tuple(settings["seeds"]),
        settings["timeout_seconds"],
    )
    output = Path(payload["output_dir"])
    try:
        source = Path(payload["source"]) if payload["source"] is not None else None
        if source is not None:
            compile(source.read_bytes(), str(source), "exec")
        report = _evaluate_in_worker(source, output, config, payload["validate_only"])
    except Exception as error:
        traceback.print_exc()
        descriptor = output / "interface.json"
        report = _failure(
            "error",
            f"{type(error).__name__}: {error}",
            read_json(descriptor) if descriptor.exists() else None,
        )
    write_json(output / "worker_result.json", report)


if __name__ == "__main__":
    import tyro

    tyro.cli(_worker_main)
