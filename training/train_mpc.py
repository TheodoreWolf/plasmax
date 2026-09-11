"""Train an online MPC dynamics model and export its frozen planner."""

# ruff: noqa: E402

from __future__ import annotations

import dataclasses

from scripts._runtime import set_default_xla_flags

set_default_xla_flags("--xla_gpu_enable_command_buffer=")

import tyro

from agents.mpc import MPCAgent
from experiments.studies.baseline_study import run_slug
from training.runs import EnvConfig, WandbConfig, load_env, run_native, validate_seeds


@dataclasses.dataclass
class MPCConfig:
    reward_scalar: str = "P_fusion"
    horizon: int = 5
    num_samples: int = 64
    buffer_size: int = 20_000
    hidden: int = 128
    lr: float = 1e-3
    train_batch_size: int = 256
    total_timesteps: int = 6_000
    eval_freq: int = 100
    planning_seed: int = 0


@dataclasses.dataclass
class Config:
    env: EnvConfig = dataclasses.field(
        default_factory=lambda: EnvConfig(backend="qlknn")
    )
    mpc: MPCConfig = dataclasses.field(default_factory=MPCConfig)
    wandb: WandbConfig = dataclasses.field(default_factory=WandbConfig)
    seed: int = 0
    num_seeds: int = 1
    run_name: str | None = None
    history_dir: str | None = None
    checkpoint_dir: str | None = None
    algorithm: str = "mpc"
    study: str = "debug"


def main(config: Config) -> None:
    validate_seeds(config.env.backend, config.num_seeds)
    agent = MPCAgent.create(
        load_env(config.env, config.env.backend),
        **dataclasses.asdict(config.mpc),
        eval_num_episodes=config.env.eval_n_envs,
        deterministic=config.env.deterministic_eval,
    )
    name = config.run_name or run_slug(
        config.algorithm,
        config.env.env_setup,
        config.env.backend or "native",
        config.env.variant,
        config.env.reward,
        config.num_seeds,
    )
    run_native(agent, config, name)


if __name__ == "__main__":
    main(tyro.cli(Config))
