"""Host-side logging, policy export, and paired evaluation for clone-only runs."""

from __future__ import annotations

import copy
import dataclasses
import time
from pathlib import Path
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np
import wandb

from agents.policy_io import save_policy
from plasmax import make
from plasmax.environment.config import parse_env_and_backend
from plasmax.wrappers import CorruptedActionWrapper, OracleWrappers, RealisticWrappers
from training.evaluation import (
    check_interfaces,
    evaluate_policy,
    evaluate_returns,
    transfer_metrics,
    write_transfer_summary,
)
from training.vmap_logging import SeedBufferLogger


@dataclasses.dataclass
class EnvConfig:
    env_setup: str = "iter/hybrid/flattop"
    backend: str | None = "bohm_gyrobohm"
    transfer_backend: str | None = None
    reward: str | None = None
    variant: Literal["oracle", "realistic"] = "realistic"
    max_steps: int | None = None
    time_aware: bool = False
    action_noise_scale: float | None = None
    eval_n_envs: int = 16
    eval_seed: int = 10_000
    deterministic_eval: bool = True
    transfer_n_envs: int = 128


@dataclasses.dataclass
class WandbConfig:
    project: str = "plasmax"
    entity: str = "flair"
    group: str = "debug"
    mode: Literal["online", "offline", "disabled"] = "online"
    tags: tuple[str, ...] = ()


def load_env(config: EnvConfig, backend: str | None) -> Any:
    if config.variant not in ("realistic", "oracle"):
        raise ValueError(f"unknown variant {config.variant!r}")
    wrappers = RealisticWrappers if config.variant == "realistic" else OracleWrappers
    env = make(config.env_setup, backend, reward=config.reward)
    if config.action_noise_scale is not None:
        env = CorruptedActionWrapper(env, noise_scale=config.action_noise_scale)
    return wrappers(
        env,
        max_steps=config.max_steps,
        time_aware=config.time_aware,
    )


def load_policy_env(
    policy: Any,
    *,
    backend: str | None = None,
    env_setup: str | None = None,
    max_steps: int | None = None,
    variant: str | None = None,
) -> Any:
    """Rebuild an artifact's named task, optionally on a transfer backend."""
    recorded = policy.metadata.get("config", {}).get("env", {})
    source = policy.metadata.get("source_config", {})
    effective = policy.metadata.get("effective_task", {})
    task = env_setup or recorded.get("env_setup") or source.get("environment_key")
    if task is None:
        raise ValueError("policy has no task metadata; provide env_setup explicitly")
    selected_backend = backend if backend is not None else recorded.get("backend")
    selected_variant = variant or recorded.get("variant", "realistic")
    options = {
        "max_steps": max_steps
        if max_steps is not None
        else (recorded.get("max_steps") or policy.metadata.get("source_max_steps")),
        "time_aware": "elapsed_time" in policy.interface.get("observations", {}),
    }
    if selected_variant == "realistic":
        options["quantize_bins"] = recorded.get("quantize_bins")
        wrappers = RealisticWrappers
    elif selected_variant == "oracle":
        wrappers = OracleWrappers
    else:
        raise ValueError(f"unknown variant {selected_variant!r}")
    reward = (
        recorded.get("reward")
        or effective.get("reward")
        or source.get("task", {}).get("reward")
    )
    # KSTAR's native reward is not a make() override.
    if task == "kstar_worldmodel":
        reward = None
    env = make(task, selected_backend, reward=reward)
    action_noise_scale = recorded.get("action_noise_scale")
    if action_noise_scale is not None:
        env = CorruptedActionWrapper(env, noise_scale=action_noise_scale)
    env = wrappers(env, **options)
    check_interfaces(policy, env)
    return env


def validate_seeds(backend: str | None, num_seeds: int) -> None:
    if num_seeds < 1:
        raise ValueError("num_seeds must be positive")
    if backend is not None and "tglfnn" in backend.lower() and num_seeds > 1:
        raise ValueError("TGLFNN training seeds must run as independent processes")


def reward_for_env(env: str, backend: str = "bohm_gyrobohm") -> str:
    """Return the reward stored with an environment's task metadata."""
    return parse_env_and_backend(env, backend).task.reward


def validate_reward(
    env: str,
    reward: str | None,
    backend: str = "bohm_gyrobohm",
) -> None:
    """Validate an explicit override; omitted rewards inherit task metadata."""
    if reward is None:
        return
    expected = reward_for_env(env, backend)
    if reward != expected:
        raise ValueError(
            f"baseline reward mismatch for {env!r}: got {reward!r}, "
            f"expected {expected!r}"
        )


def seed_keys(first_seed: int, num_seeds: int) -> jax.Array:
    """Return consecutive numeric seed keys, invariant to job sharding.

    Seed ``i`` is always ``jax.random.PRNGKey(i)``, matching single-seed runs.
    Consequently, ``seed_keys(0, 5)`` followed by ``seed_keys(5, 5)`` is
    identical to ``seed_keys(0, 10)``.
    """
    if num_seeds <= 0:
        raise ValueError("num_seeds must be positive")
    indices = jnp.arange(
        first_seed,
        first_seed + num_seeds,
        dtype=jnp.uint32,
    )
    return jax.vmap(jax.random.PRNGKey)(indices)


def run_slug(
    algorithm: str,
    env: str,
    backend: str,
    variant: str,
    reward: str | None,
    num_seeds: int,
) -> str:
    """Return a filesystem- and W&B-friendly stable baseline run label."""
    return "-".join(
        (
            algorithm,
            env.replace("/", "_"),
            backend.replace("/", "_"),
            variant,
            reward or "task",
            f"{num_seeds}seeds",
        )
    )


def save_run_policies(
    agent: Any,
    states: Any,
    config: Any,
    run_name: str,
    *,
    batched: bool,
    results: Any = None,
    metrics: dict[str, Any] | None = None,
) -> tuple[Path, ...]:
    """Export one final inference policy per seed, rejecting failed batches first."""
    if bool(np.any(np.asarray(getattr(states, "failed", False)))):
        failures = getattr(
            states, "failure_step", getattr(states, "first_failure_step", -1)
        )
        raise FloatingPointError(f"Training failed at steps {np.asarray(failures)}")
    count = config.num_seeds if batched else 1
    paths = []
    for index in range(count):
        state = (
            jax.tree.map(lambda value, i=index: value[i], states) if batched else states
        )
        evaluation = (
            jax.tree.map(lambda value, i=index: value[i], results)
            if batched and results is not None
            else results
        )
        seed = config.seed + index
        seed_name = f"{run_name}-seed{seed}"
        path = (
            Path(config.checkpoint_dir) / f"{seed_name}.msgpack"
            if config.checkpoint_dir is not None
            else None
        )
        metadata = {
            "run_name": seed_name,
            "parent_run_name": run_name,
            "seed": seed,
            "seed_index": index,
            "config": dataclasses.asdict(config),
            "actual_train_steps": int(np.asarray(state.global_step)),
            **(metrics or {}),
        }
        paths.append(
            save_policy(
                agent,
                state,
                path,
                results=evaluation,
                metadata=metadata,
                deterministic=config.env.deterministic_eval,
            )
        )
    return tuple(paths)


def evaluate_transfer(
    agent: Any,
    states: Any,
    source_env: Any,
    target_env: Any,
    config: Any,
    *,
    batched: bool,
) -> dict[str, float | str]:
    """Evaluate every frozen seed against the same source/target episode keys."""
    check_interfaces(source_env, target_env)

    def evaluate_one(state: Any) -> tuple[jax.Array, ...]:
        act = agent.make_act(state, deterministic=config.env.deterministic_eval)
        key = jax.random.key(config.env.eval_seed)
        lengths, returns = evaluate_returns(
            act,
            source_env,
            key,
            num_episodes=config.env.transfer_n_envs,
        )
        target_lengths, target_returns = evaluate_returns(
            act,
            target_env,
            key,
            num_episodes=config.env.transfer_n_envs,
        )
        return returns, lengths, target_returns, target_lengths

    state_batch = states if batched else jax.tree.map(lambda x: x[None], states)
    start = time.monotonic()
    outputs = jax.jit(jax.vmap(evaluate_one))(state_batch)
    jax.block_until_ready(outputs)
    elapsed = time.monotonic() - start
    return transfer_metrics(
        config.env.backend or "native",
        config.env.transfer_backend,
        *(np.asarray(value) for value in outputs),
        elapsed,
    )


def run_native(
    agent: Any, config: Any, run_name: str, *, export_policy: bool = True
) -> None:
    """Train a native agent, with optional export for supported policy formats.

    APG and SAPO currently log training history without inference artifacts.
    Disabling export retains the same failure checks and W&B lifecycle.
    """
    validate_seeds(config.env.backend, config.num_seeds)
    logger = SeedBufferLogger(
        num_seeds=config.num_seeds,
        seed_ids=tuple(range(config.seed, config.seed + config.num_seeds)),
        run_name=run_name,
        project=config.wandb.project,
        entity=config.wandb.entity,
        group=config.wandb.group,
        mode=config.wandb.mode,
        config=dataclasses.asdict(config),
        out_dir=config.history_dir,
        job_type=config.algorithm,
        tags=(config.study, *config.wandb.tags),
    )

    def train_one(key: jax.Array, index: jax.Array) -> Any:
        def callback(
            current: Any, state: Any, rng: jax.Array, diagnostics: Any
        ) -> dict:
            del rng
            act = current.make_act(state, deterministic=config.env.deterministic_eval)
            evaluation, _ = evaluate_policy(
                act,
                current.env,
                jax.random.key(config.env.eval_seed),
                num_episodes=config.env.eval_n_envs,
            )
            metrics = {
                name: value
                for name, value in evaluation.items()
                if name not in ("returns", "lengths") and not name.startswith("eval/")
            }
            metrics.update(diagnostics or {})
            jax.debug.callback(logger.log, state.global_step, index, metrics)
            return metrics

        current = copy.copy(agent)
        object.__setattr__(current, "eval_callback", callback)
        return current.train(key)

    try:
        keys = jax.vmap(jax.random.PRNGKey)(
            jnp.arange(config.seed, config.seed + config.num_seeds)
        )
        indices = jnp.arange(config.num_seeds, dtype=jnp.int32)
        if config.num_seeds == 1:
            # Keep each independent run's outer control flow unbatched.
            train = jax.jit(train_one)
            train_args = (keys[0], indices[0])
        else:
            train = jax.jit(jax.vmap(train_one))
            train_args = (keys, indices)
        start = time.monotonic()
        lowered = train.lower(*train_args)
        lower_seconds = time.monotonic() - start
        start = time.monotonic()
        lowered.compile()
        compile_seconds = time.monotonic() - start
        logger.start_time = time.time()
        start = time.monotonic()
        # Reuse JIT's cache: direct Compiled calls mishandle TORAX closure
        # constants on the current JAX version (also see train_ppo).
        states, results = train(*train_args)
        if config.num_seeds == 1:
            states, results = jax.tree.map(lambda value: value[None], (states, results))
        jax.block_until_ready((states, results))
        jax.effects_barrier()
        summary = {
            "time/lower_s": lower_seconds,
            "time/compile_s": compile_seconds,
            "time/train_s": time.monotonic() - start,
            "run/actual_train_steps": int(np.asarray(states.global_step[0])),
        }
        failed = np.asarray(getattr(states, "failed", False))
        failure_steps = np.asarray(
            getattr(states, "failure_step", getattr(states, "first_failure_step", -1))
        )
        summary["run/failed"] = bool(np.any(failed))
        summary["run/failed_seeds"] = int(np.count_nonzero(failed))
        for index in range(config.num_seeds):
            prefix = f"seeds/{config.seed + index}/run"
            summary[f"{prefix}/actual_train_steps"] = int(states.global_step[index])
            if failed.ndim:
                summary[f"{prefix}/failed"] = bool(failed[index])
                summary[f"{prefix}/failure_step"] = int(failure_steps[index])
        if hasattr(states, "simulator_steps"):
            simulator_steps = np.asarray(states.simulator_steps)
            summary["run/simulator_steps_total"] = int(simulator_steps.sum())
            for index in range(config.num_seeds):
                summary[f"seeds/{config.seed + index}/run/simulator_steps"] = int(
                    simulator_steps[index]
                )
        if hasattr(states, "alive_steps"):
            summary["run/alive_train_steps_min"] = int(np.min(states.alive_steps))
            summary["run/alive_train_steps_max"] = int(np.max(states.alive_steps))
        if np.any(failed):
            logger.log_once(summary)
            raise FloatingPointError(f"Training failed at steps {failure_steps}")
        paths = (
            save_run_policies(
                agent,
                states,
                config,
                run_name,
                batched=True,
                results=results,
                metrics=summary,
            )
            if export_policy
            else ()
        )
        if config.env.transfer_backend is not None:
            transfer = evaluate_transfer(
                agent,
                states,
                agent.env,
                load_env(config.env, config.env.transfer_backend),
                config,
                batched=True,
            )
            summary.update(transfer)
            write_transfer_summary(config.history_dir, run_name, transfer)
        if paths:
            artifact = wandb.Artifact(run_name, type="model")
            for path in paths:
                artifact.add_file(str(path))
                print(f"Saved policy to {path}", flush=True)
            logger.log_artifact(artifact)
        logger.log_once(summary)
    finally:
        logger.finish()
