"""Evaluate saved policies with paired episode keys and optional NPZ trajectories."""

# ruff: noqa: E402

from __future__ import annotations

import dataclasses
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from scripts._runtime import set_default_xla_flags

set_default_xla_flags("--xla_gpu_enable_command_buffer=")

import jax
import numpy as np
import tyro
import wandb

from agents.policy_io import load_policy
from training.evaluation import evaluate_policy, save_trajectories, transfer_metrics
from training.runs import WandbConfig, load_policy_env


@dataclasses.dataclass
class Config:
    policies: tuple[Path, ...]
    env_setup: str | None = None
    backend: str | None = None
    target_backend: str | None = None
    variant: Literal["oracle", "realistic"] | None = None
    max_steps: int | None = None
    num_episodes: int = 16
    eval_seed: int = 10_000
    deterministic: bool | None = None
    trajectories: bool = False
    output_dir: Path = dataclasses.field(
        default_factory=lambda: (
            Path("outputs/evaluation") / datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        )
    )
    wandb: WandbConfig = dataclasses.field(default_factory=WandbConfig)


def main(config: Config) -> None:
    if not config.policies or config.num_episodes <= 0:
        raise ValueError("provide policies and a positive num_episodes")
    run = wandb.init(
        project=config.wandb.project,
        entity=config.wandb.entity,
        group=config.wandb.group,
        mode=config.wandb.mode,
        job_type="policy_evaluation",
        tags=list(config.wandb.tags),
        config=dataclasses.asdict(config),
    )
    try:
        config.output_dir.mkdir(parents=True, exist_ok=True)
        key = jax.random.key(config.eval_seed)
        for index, path in enumerate(config.policies):
            policy = load_policy(path)
            print(policy.summary())
            source_env = load_policy_env(
                policy,
                env_setup=config.env_setup,
                backend=config.backend,
                max_steps=config.max_steps,
                variant=config.variant,
            )

            def collect(env, current=policy):
                def evaluate(rng):
                    metrics, trajectory = evaluate_policy(
                        current,
                        env,
                        rng,
                        num_episodes=config.num_episodes,
                        deterministic=config.deterministic,
                    )
                    return metrics, trajectory if config.trajectories else None

                return jax.jit(evaluate)(key)

            start = time.monotonic()
            metrics, source = collect(source_env)
            jax.block_until_ready(metrics)
            values = {
                name: np.asarray(value).tolist() for name, value in metrics.items()
            }
            values["eval/seconds"] = time.monotonic() - start
            stem = f"{index}-{path.stem}"
            if config.trajectories:
                save_trajectories(
                    config.output_dir / f"{stem}-source.npz", source, source_env
                )
            if config.target_backend is not None:
                target_env = load_policy_env(
                    policy,
                    backend=config.target_backend,
                    env_setup=config.env_setup,
                    max_steps=config.max_steps,
                    variant=config.variant,
                )
                target_metrics, target = collect(target_env)
                jax.block_until_ready(target_metrics)
                source_backend = config.backend or (
                    policy.metadata.get("config", {})
                    .get("env", {})
                    .get("backend", "native")
                )
                values.update(
                    transfer_metrics(
                        source_backend,
                        config.target_backend,
                        np.asarray(metrics["returns"])[None],
                        np.asarray(metrics["lengths"])[None],
                        np.asarray(target_metrics["returns"])[None],
                        np.asarray(target_metrics["lengths"])[None],
                        time.monotonic() - start,
                    )
                )
                if config.trajectories:
                    save_trajectories(
                        config.output_dir / f"{stem}-target.npz", target, target_env
                    )
            report = {"policy": str(path), "eval_seed": config.eval_seed, **values}
            (config.output_dir / f"{stem}.json").write_text(
                json.dumps(report, indent=2) + "\n"
            )
            run.log(
                {
                    "policy": str(path),
                    **{
                        name: value
                        for name, value in values.items()
                        if isinstance(value, (str, int, float))
                    },
                }
            )
    finally:
        run.finish()


if __name__ == "__main__":
    main(tyro.cli(Config))
