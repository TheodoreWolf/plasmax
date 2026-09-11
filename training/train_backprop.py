"""Train a native Envelope feedback policy or open-loop knot schedule."""

# ruff: noqa: E402

from __future__ import annotations

import dataclasses
from typing import Literal

from scripts._runtime import set_default_xla_flags

set_default_xla_flags("--xla_gpu_enable_command_buffer=")

import tyro

from agents.backprop import BackpropOpenLoopAgent, BackpropPolicyAgent
from experiments.studies.baseline_study import run_slug
from training.runs import EnvConfig, WandbConfig, load_env, run_native, validate_seeds


@dataclasses.dataclass
class BackpropConfig:
    total_timesteps: int = 10_000_000
    eval_freq: int = 1_000_000
    num_rollouts: int = 64
    gradient_horizon: int = 32
    learning_rate: float | None = None
    hidden_sizes: tuple[int, ...] = (64, 64)
    num_knots: int = 10
    grad_clip: float = 1.0
    nonfinite_backoff_factor: float = 0.5
    min_update_scale: float = 1e-3
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
    options["learning_rate"] = (
        config.backprop.learning_rate
        if config.backprop.learning_rate is not None
        else (1e-4 if is_policy else 5e-2)
    )
    if is_policy:
        for name in ("num_knots", "nonfinite_backoff_factor", "min_update_scale"):
            options.pop(name)
    else:
        options.pop("hidden_sizes")
    agent_type = BackpropPolicyAgent if is_policy else BackpropOpenLoopAgent
    agent = agent_type.create(
        env,
        **options,
        eval_n_envs=config.env.eval_n_envs,
        eval_seed=config.env.eval_seed,
        init_seed=config.seed,
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
