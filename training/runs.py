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
from plasmax.wrappers import OracleWrappers, RealisticWrappers
from training.evaluation import (
    check_interfaces,
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
    disruption_penalty: float | None = None
    max_steps: int | None = None
    time_aware: bool = False
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
    return wrappers(
        make(
            config.env_setup,
            backend,
            reward=config.reward,
            disruption_penalty=config.disruption_penalty,
        ),
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
    penalty = effective.get("terminal_penalty", recorded.get("disruption_penalty"))
    # PPO's optional study multiplier is resolved when the source env is made.
    if penalty is None and recorded.get("disruption_kappa") is not None:
        nominal = source.get("task", {}).get("terminal_penalty")
        if nominal is None:
            raise ValueError("policy lacks the source penalty for disruption_kappa")
        penalty = nominal * recorded["disruption_kappa"]
    if penalty is None:
        penalty = source.get("task", {}).get("terminal_penalty")
    reward = (
        recorded.get("reward")
        or effective.get("reward")
        or source.get("task", {}).get("reward")
    )
    # KSTAR's native reward is not a make() override.
    if task == "kstar_worldmodel":
        reward = penalty = None
    env = wrappers(
        make(task, selected_backend, reward=reward, disruption_penalty=penalty),
        **options,
    )
    check_interfaces(policy, env)
    return env


def validate_seeds(backend: str | None, num_seeds: int) -> None:
    if num_seeds < 1:
        raise ValueError("num_seeds must be positive")
    if backend is not None and "tglfnn" in backend.lower() and num_seeds > 1:
        raise ValueError("TGLFNN training seeds must run as independent processes")


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


def run_native(agent: Any, config: Any, run_name: str) -> None:
    """Run the shared host lifecycle for the native backprop and MPC agents."""
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
            lengths, returns = evaluate_returns(
                act,
                current.env,
                jax.random.key(config.env.eval_seed),
                num_episodes=config.env.eval_n_envs,
            )
            metrics = {
                "evaluation/return_mean": returns.mean(),
                "evaluation/return_std": returns.std(),
                "evaluation/return_min": returns.min(),
                "evaluation/return_max": returns.max(),
                "evaluation/episode_length_mean": lengths.mean(),
                **(diagnostics or {}),
            }
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
        train = jax.jit(jax.vmap(train_one))
        start = time.monotonic()
        lowered = train.lower(keys, indices)
        lower_seconds = time.monotonic() - start
        start = time.monotonic()
        lowered.compile()
        compile_seconds = time.monotonic() - start
        logger.start_time = time.time()
        start = time.monotonic()
        # Reuse JIT's cache: direct Compiled calls mishandle TORAX closure
        # constants on the current JAX version (also see train_ppo).
        states, results = train(keys, indices)
        jax.block_until_ready((states, results))
        jax.effects_barrier()
        summary = {
            "time/lower_s": lower_seconds,
            "time/compile_s": compile_seconds,
            "time/train_s": time.monotonic() - start,
            "run/actual_train_steps": int(np.asarray(states.global_step[0])),
        }
        if hasattr(states, "alive_steps"):
            summary["run/alive_train_steps_min"] = int(np.min(states.alive_steps))
            summary["run/alive_train_steps_max"] = int(np.max(states.alive_steps))
        paths = save_run_policies(
            agent,
            states,
            config,
            run_name,
            batched=True,
            results=results,
            metrics=summary,
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
        artifact = wandb.Artifact(run_name, type="model")
        for path in paths:
            artifact.add_file(str(path))
            print(f"Saved policy to {path}", flush=True)
        logger.log_artifact(artifact)
        logger.log_once(summary)
    finally:
        logger.finish()
