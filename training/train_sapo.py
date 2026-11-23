"""Train SAPO on native Envelope states and log metrics and history to W&B."""

# ruff: noqa: E402

from __future__ import annotations

import dataclasses
from typing import Literal

from scripts._runtime import set_default_xla_flags

set_default_xla_flags("--xla_gpu_enable_command_buffer=")

import tyro

from agents.sapo import SAPOAgent
from experiments.studies.baseline_study import run_slug
from training.runs import EnvConfig, WandbConfig, load_env, run_native, validate_seeds


@dataclasses.dataclass
class SAPOConfig:
    total_timesteps: int = 10_000_000
    eval_freq: int = 1_000_000
    num_rollouts: int = 64
    horizon: int = 32
    hidden_sizes: tuple[int, ...] = (64, 64)
    gamma: float = 0.99
    td_lambda: float = 0.95
    actor_learning_rate: float = 2e-3
    critic_learning_rate: float = 5e-4
    alpha_learning_rate: float = 5e-3
    min_learning_rate: float = 1e-5
    max_grad_norm: float = 0.5
    critic_epochs: int = 16
    num_minibatches: int = 4
    normalize_observations: bool = True
    initial_alpha: float = 1.0
    remat: bool = True


@dataclasses.dataclass
class Config:
    env: EnvConfig = dataclasses.field(default_factory=EnvConfig)
    sapo: SAPOConfig = dataclasses.field(default_factory=SAPOConfig)
    wandb: WandbConfig = dataclasses.field(default_factory=WandbConfig)
    seed: int = 0
    num_seeds: int = 1
    run_name: str | None = None
    history_dir: str | None = None
    algorithm: Literal["sapo"] = "sapo"
    study: str = "debug"


def main(config: Config) -> None:
    validate_seeds(config.env.backend, config.num_seeds)
    agent = SAPOAgent.create(
        load_env(config.env, config.env.backend),
        **dataclasses.asdict(config.sapo),
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
