"""Train a native Envelope feedback policy or open-loop knot schedule."""

# ruff: noqa: E402

from __future__ import annotations

import dataclasses
from typing import Literal

from scripts._runtime import set_default_xla_flags

set_default_xla_flags("--xla_gpu_enable_command_buffer=")

import tyro

from agents.backprop import BackpropOpenLoopAgent, BackpropPolicyAgent
from training.runs import (
    EnvConfig,
    WandbConfig,
    load_env,
    run_native,
    run_slug,
    validate_seeds,
)


@dataclasses.dataclass
class BackpropConfig:
    total_timesteps: int = 10_000_000
    eval_freq: int = 1_000_000
    num_rollouts: int = 64
    gradient_horizon: int = 32
    learning_rate: float = 1e-4
    hidden_sizes: tuple[int, ...] = (64, 64)
    num_knots: int = 10
    grad_clip: float = 1.0
    remat: bool = True


@dataclasses.dataclass
class Config:
    mode: Literal["policy", "open_loop"] = "policy"
    env: EnvConfig = dataclasses.field(default_factory=EnvConfig)
    backprop: BackpropConfig = dataclasses.field(default_factory=BackpropConfig)
    wandb: WandbConfig = dataclasses.field(default_factory=WandbConfig)
    seed: int = 0
    num_seeds: int = 10
    run_name: str | None = None
    history_dir: str | None = None
    checkpoint_dir: str | None = None
    algorithm: str = "backprop_policy"
    study: str = "debug"


def main(config: Config) -> None:
    validate_seeds(config.env.backend, config.num_seeds)
    is_policy = config.mode == "policy"
    config = dataclasses.replace(
        config,
        algorithm="backprop_policy" if is_policy else "backprop_open_loop",
        env=dataclasses.replace(
            config.env,
            time_aware=config.env.time_aware or not is_policy,
        ),
    )
    env = load_env(config.env, config.env.backend)
    options = dataclasses.asdict(config.backprop)
    if is_policy:
        options.pop("num_knots")
    else:
        options.pop("hidden_sizes")
    agent_type = BackpropPolicyAgent if is_policy else BackpropOpenLoopAgent
    agent = agent_type.create(
        env,
        **options,
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
    run_native(agent, config, name)


if __name__ == "__main__":
    main(tyro.cli(Config))
