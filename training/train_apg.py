"""Train APG on native Envelope states and log metrics and history to W&B."""

# ruff: noqa: E402

from __future__ import annotations

import dataclasses
from typing import Literal

from scripts._runtime import set_default_xla_flags

set_default_xla_flags("--xla_gpu_enable_command_buffer=")

import tyro

from agents.apg import APGAgent
from experiments.studies.baseline_study import run_slug
from training.runs import EnvConfig, WandbConfig, load_env, run_native, validate_seeds


@dataclasses.dataclass
class APGConfig:
    total_timesteps: int = 10_000_000
    eval_freq: int = 1_000_000
    num_rollouts: int = 8
    horizon: int = 10
    hidden_sizes: tuple[int, ...] = (64, 64)
    learning_rate: float = 1e-5
    momentum: float = 0.9
    grad_clip: float = 0.0
    remat: bool = True


@dataclasses.dataclass
class Config:
    env: EnvConfig = dataclasses.field(default_factory=EnvConfig)
    apg: APGConfig = dataclasses.field(default_factory=APGConfig)
    wandb: WandbConfig = dataclasses.field(default_factory=WandbConfig)
    seed: int = 0
    num_seeds: int = 1
    run_name: str | None = None
    history_dir: str | None = None
    algorithm: Literal["apg"] = "apg"
    study: str = "debug"


def main(config: Config) -> None:
    validate_seeds(config.env.backend, config.num_seeds)
    agent = APGAgent.create(
        load_env(config.env, config.env.backend),
        **dataclasses.asdict(config.apg),
        eval_n_envs=config.env.eval_n_envs,
        eval_seed=config.env.eval_seed,
    )
    name = config.run_name or run_slug(
        config.algorithm,
        config.env.env_setup,
        config.env.backend or "native",
        config.env.variant,
        config.env.reward,
        config.num_seeds,
    )
    run_native(agent, config, name, export_policy=False)


if __name__ == "__main__":
    main(tyro.cli(Config))
