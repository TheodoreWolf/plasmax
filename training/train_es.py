"""Train a feedback policy or open-loop knots with full-episode ES fitness."""

# ruff: noqa: E402

from __future__ import annotations

import dataclasses
from typing import Literal

from scripts._runtime import set_default_xla_flags

set_default_xla_flags("--xla_gpu_enable_command_buffer=")

import tyro

from agents.es import ESAgent
from experiments.studies.baseline_study import run_slug
from training.runs import EnvConfig, WandbConfig, load_env, run_native, validate_seeds


@dataclasses.dataclass
class ESConfig:
    strategy: Literal["open_es", "cma_es"] = "open_es"
    total_timesteps: int = 10_000_000
    eval_freq: int = 1_000_000
    population_size: int = 64
    num_rollouts: int = 1
    sigma: float = 0.01
    learning_rate: float = 1e-3
    grad_clip: float = 1.0
    hidden_sizes: tuple[int, ...] = (64, 64)
    num_knots: int = 10


@dataclasses.dataclass
class Config:
    parameterization: Literal["policy", "open_loop"] = "policy"
    env: EnvConfig = dataclasses.field(default_factory=EnvConfig)
    es: ESConfig = dataclasses.field(default_factory=ESConfig)
    wandb: WandbConfig = dataclasses.field(default_factory=WandbConfig)
    seed: int = 0
    num_seeds: int = 1
    run_name: str | None = None
    history_dir: str | None = None
    checkpoint_dir: str | None = None
    algorithm: Literal["es"] = "es"
    study: str = "debug"


def main(config: Config) -> None:
    validate_seeds(config.env.backend, config.num_seeds)
    config = dataclasses.replace(
        config,
        env=dataclasses.replace(
            config.env,
            time_aware=config.env.time_aware or config.parameterization == "open_loop",
        ),
    )
    agent = ESAgent.create(
        load_env(config.env, config.env.backend),
        parameterization=config.parameterization,
        **dataclasses.asdict(config.es),
        eval_n_envs=config.env.eval_n_envs,
        eval_seed=config.env.eval_seed,
        init_seed=config.seed,
    )
    name = config.run_name or run_slug(
        f"es_{config.es.strategy}_{config.parameterization}",
        config.env.env_setup,
        config.env.backend or "native",
        config.env.variant,
        config.env.reward,
        config.num_seeds,
    )
    run_native(agent, config, name)


if __name__ == "__main__":
    main(tyro.cli(Config))
