"""Evaluate generated source in workers, keeping imports out of the master."""

from __future__ import annotations

import dataclasses
import functools
import importlib.util
import inspect
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from experiments.agent_policy_search.artifacts import read_json, write_json
from experiments.agent_policy_search.config import EvaluationConfig, TaskConfig
from experiments.agent_policy_search.feedback import build_feedback, json_safe

__all__ = ["evaluate_candidate"]


def _failure(
    status: str, error: str, interface: dict[str, Any] | None = None
) -> dict[str, Any]:
    return {
        "status": status,
        "finite": False,
        "mean_return": None,
        "feedback": {},
        "interface": interface or {},
        "error": error,
    }


def _run_worker(
    source: Path, output_dir: Path, config: EvaluationConfig
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    request_path = output_dir / "request.json"
    write_json(
        request_path, {"source": source, "output_dir": output_dir, "config": config}
    )
    env = dict(os.environ)
    # This avoids passing common API credentials directly to generated code.
    # It is not a security sandbox: host files and networking remain accessible.
    for name in tuple(env):
        if name.endswith(("_API_KEY", "_API_TOKEN")):
            env.pop(name)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    started = time.monotonic()
    with (
        (output_dir / "stdout.txt").open("x") as stdout,
        (output_dir / "stderr.txt").open("x") as stderr,
    ):
        try:
            process = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "experiments.agent_policy_search.evaluator",
                    "--request",
                    str(request_path),
                ],
                cwd=Path(__file__).resolve().parents[2],
                env=env,
                stdout=stdout,
                stderr=stderr,
                timeout=config.timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired:
            report = _failure(
                "timeout", f"Evaluation exceeded {config.timeout_seconds:g} seconds"
            )
        else:
            result_path = output_dir / "worker_result.json"
            if result_path.exists():
                report = read_json(result_path)
            else:
                report = _failure(
                    "error",
                    f"Worker exited with code {process.returncode} "
                    "without a result; see stderr.txt",
                )
    interface_path = output_dir / "interface.json"
    if not report["interface"] and interface_path.exists():
        report["interface"] = read_json(interface_path)
    report["evaluation_seconds"] = time.monotonic() - started
    write_json(output_dir / "result.json", report)
    return report


def evaluate_candidate(
    source: Path, output_dir: Path, config: EvaluationConfig
) -> dict[str, Any]:
    """Evaluate fixed seeds without importing candidate source in this process.

    TGLFNN seeds run in independent workers. Other backends stream the explicit
    seed bank in one worker. Each worker has its own timeout, request and logs;
    the non-TGLFNN timeout covers the whole bank, TGLFNN timeouts cover one seed.
    """
    source, output_dir = (
        source.expanduser().resolve(),
        output_dir.expanduser().resolve(),
    )
    if not config.seeds or any(seed < 0 or seed >= 2**32 for seed in config.seeds):
        raise ValueError("evaluation needs a nonempty bank of uint32 seeds")
    if config.timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    tglfnn = config.task.backend is not None and "tglfnn" in config.task.backend.lower()
    if not tglfnn or len(config.seeds) == 1:
        return _run_worker(source, output_dir, config)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        output_dir / "request.json",
        {"source": source, "output_dir": output_dir, "config": config},
    )
    reports = []
    for index, seed in enumerate(config.seeds):
        worker_dir = output_dir / f"seed_{index:04d}_{seed}"
        report = _run_worker(
            source, worker_dir, dataclasses.replace(config, seeds=(seed,))
        )
        reports.append(report)
        if report["status"] != "ok":
            report = dict(
                report,
                failed_seed=seed,
                evaluation_seconds=sum(item["evaluation_seconds"] for item in reports),
            )
            write_json(output_dir / "result.json", report)
            return report
    report = _merge_seed_results(output_dir, config, reports)
    write_json(output_dir / "result.json", report)
    return report


def _merge_seed_results(
    output_dir: Path, config: EvaluationConfig, reports: list[dict[str, Any]]
) -> dict[str, Any]:
    import numpy as np

    banks = []
    for index, seed in enumerate(config.seeds):
        with np.load(
            output_dir / f"seed_{index:04d}_{seed}" / "trajectories.npz",
            allow_pickle=False,
        ) as bank:
            banks.append({name: bank[name] for name in bank.files})
    arrays = _concatenate_banks(banks)
    arrays["metadata_json"] = np.asarray(
        json.dumps({"seeds": config.seeds, "task": dataclasses.asdict(config.task)})
    )
    np.savez_compressed(output_dir / "trajectories.npz", **arrays)
    interface = reports[0]["interface"]
    write_json(output_dir / "interface.json", interface)
    returns = np.asarray(
        [float(report["feedback"]["episodes"][0]["return"]) for report in reports]
    )
    return _report(arrays, interface, config.seeds, returns) | {
        "evaluation_seconds": sum(item["evaluation_seconds"] for item in reports)
    }


def _concatenate_banks(banks: list[dict[str, Any]]) -> dict[str, Any]:
    import numpy as np

    return {
        name: (
            np.concatenate([bank[name] for bank in banks], axis=0)
            if value.ndim >= 2 and value.shape[:2] == banks[0]["valid"].shape
            else value
        )
        for name, value in banks[0].items()
    }


def _report(
    arrays: dict[str, Any],
    interface: dict[str, Any],
    seeds: tuple[int, ...],
    returns: Any,
) -> dict[str, Any]:
    import numpy as np

    valid = np.asarray(arrays["valid"], dtype=bool)
    finite_returns = bool(np.isfinite(returns).all())
    finite_actions = bool(np.isfinite(arrays["requested_action"][valid]).all())
    finite = finite_returns and finite_actions
    return {
        "status": "ok",
        "finite": finite,
        "finite_returns": finite_returns,
        "finite_requested_actions": finite_actions,
        "mean_return": float(np.asarray(returns, dtype=np.float64).mean())
        if finite
        else None,
        "feedback": build_feedback(arrays, interface, seeds, episode_returns=returns),
        "interface": interface,
    }


def _load_policy(source: Path) -> Any:
    spec = importlib.util.spec_from_file_location("_generated_policy", source)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot import policy source: {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    act = getattr(module, "act", None)
    if not callable(act):
        raise ValueError("policy.py must define callable act(obs, rng)")
    return act


def _descriptor(env: Any, task: TaskConfig) -> dict[str, Any]:
    import numpy as np

    from agents.policy_io import environment_interface
    from plasmax.wrappers import find_max_steps

    interface = environment_interface(env)
    interface["observation_dtype"] = str(np.dtype(env.observation_space.dtype))
    cfg = getattr(env, "plasmax_config", None)
    resolved = task.reward or getattr(getattr(cfg, "task", None), "reward", None)
    interface["task"] = dataclasses.asdict(task) | {
        "resolved_reward": resolved,
        "horizon": find_max_steps(env),
    }
    interface["policy_contract"] = (
        "Pure JAX-compatible act(obs, rng) returning one array with "
        "action_shape and action_dtype; no persistent state."
    )
    if "action_bins" in interface:
        action_description = (
            "Integer bin indices: normalized = -1 + 2 * index / (bins - 1); "
            "physical = low + (normalized + 1) / 2 * (high - low)."
        )
    elif hasattr(env, "to_physical"):
        action_description = (
            "Normalized actions: physical = low + (action + 1) / 2 * (high - low), "
            "using actuator_specs physical bounds."
        )
    else:
        action_description = "Native actions in action_low/action_high units."
    interface["normalization"] = {
        "observations": (
            "TORAX profile/scalar measurements are divided by each observation "
            "spec's scale; emitted values include the selected observation "
            "wrappers. KSTAR uses its native observations."
        ),
        "actions": action_description,
    }
    if interface["history"]:
        interface["observation_history"] = (
            "Each history layer concatenates k observation frames oldest to newest, "
            "then k action frames in the inner continuous policy space. Named "
            "observation slices point to the newest frame. Time-aware elapsed time "
            "is appended to each observation frame before history stacking."
        )
    core = env.unwrapped
    reward = getattr(getattr(core, "_dynamics", core), "_reward_fn", None)
    if reward is None and resolved == "native":
        from plasmax.models.world_model_env import REWARD_SCALE, target_tracking_reward

        reward = target_tracking_reward
        constants = {"REWARD_SCALE": np.asarray(REWARD_SCALE).tolist()}
    else:
        constants = {}
    if reward is not None:
        bound = reward.keywords if isinstance(reward, functools.partial) else {}
        function = reward.func if isinstance(reward, functools.partial) else reward
        definition = {
            "name": function.__name__,
            "source_file": inspect.getsourcefile(function),
            "source": inspect.getsource(function),
            "bound_parameters": bound,
            "constants": constants,
        }
        if function.__module__ == "plasmax.rewards":
            definition["supporting_module_source"] = inspect.getsource(
                inspect.getmodule(function)
            )
        interface["reward_definition"] = definition
    return json_safe(interface)


def _evaluate_in_worker(
    source: Path, output_dir: Path, config: EvaluationConfig
) -> dict[str, Any]:
    # Import source first so syntax/import failures do not initialize TORAX.
    act = _load_policy(source)
    import jax
    import numpy as np

    from agents.policy_io import environment_interface
    from training.evaluation import (
        evaluate_policy,
        save_trajectories,
        trajectory_arrays,
    )
    from training.runs import EnvConfig, load_env

    env = load_env(EnvConfig(**dataclasses.asdict(config.task)), config.task.backend)
    interface = _descriptor(env, config.task)
    write_json(output_dir / "interface.json", interface)
    resolved = getattr(env, "plasmax_config", None)
    if resolved is not None:
        write_json(
            output_dir / "resolved_config.json",
            json_safe(resolved.model_dump(mode="json")),
        )
    result = jax.eval_shape(
        act,
        jax.ShapeDtypeStruct(env.observation_space.shape, env.observation_space.dtype),
        jax.random.key(config.seeds[0]),
    )
    if getattr(result, "shape", None) != env.action_space.shape:
        raise ValueError(
            f"Policy action shape {getattr(result, 'shape', None)} "
            f"!= {env.action_space.shape}"
        )
    if np.dtype(result.dtype) != np.dtype(env.action_space.dtype):
        raise ValueError(
            f"Policy action dtype {result.dtype} != {env.action_space.dtype}"
        )

    # Retain only numeric exports between seeds. Full simulator-state stacks are
    # much larger and need not be resident for an entire 64-episode bank.
    banks, episode_metrics, episode_returns = [], [], []
    for index, seed in enumerate(config.seeds):
        metrics, trajectory = evaluate_policy(
            act, env, jax.random.key(seed), num_episodes=1
        )
        episode_path = output_dir / "episodes" / f"{index:04d}_seed_{seed}"
        save_trajectories(
            episode_path.with_suffix(".npz"),
            trajectory,
            env,
            metadata={"seeds": [seed], "task": dataclasses.asdict(config.task)},
        )
        episode_returns.append(float(np.asarray(metrics["returns"])[0]))
        banks.append(trajectory_arrays(trajectory, env))
        seed_metrics = {"seed": seed, **json_safe(jax.device_get(metrics))}
        write_json(episode_path.with_suffix(".json"), seed_metrics)
        episode_metrics.append(seed_metrics)
        del trajectory, metrics
    arrays = _concatenate_banks(banks)
    # Match training.evaluation.save_trajectories' numeric export schema after
    # concatenating its trajectory_arrays outputs, without full-state stacking.
    np.savez_compressed(
        output_dir / "trajectories.npz",
        **arrays,
        interface_json=np.asarray(json.dumps(environment_interface(env))),
        metadata_json=np.asarray(
            json.dumps({"seeds": config.seeds, "task": dataclasses.asdict(config.task)})
        ),
    )
    write_json(output_dir / "metrics.json", episode_metrics)
    return _report(arrays, interface, config.seeds, np.asarray(episode_returns))


def _worker_main(request: Path) -> None:
    payload = read_json(request)
    config_data = payload["config"]
    config = EvaluationConfig(
        task=TaskConfig(**config_data["task"]),
        seeds=tuple(config_data["seeds"]),
        timeout_seconds=config_data["timeout_seconds"],
    )
    output_dir = Path(payload["output_dir"])
    try:
        report = _evaluate_in_worker(Path(payload["source"]), output_dir, config)
    except Exception as error:
        traceback.print_exc()
        interface_path = output_dir / "interface.json"
        report = _failure(
            "error",
            f"{type(error).__name__}: {error}",
            read_json(interface_path) if interface_path.exists() else None,
        )
    write_json(output_dir / "worker_result.json", report)


if __name__ == "__main__":
    import tyro

    tyro.cli(_worker_main)
