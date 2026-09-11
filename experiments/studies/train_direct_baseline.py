"""Study settings for native backprop agents; training lives in agents."""

# ruff: noqa: E402

from __future__ import annotations

import dataclasses
from typing import Literal

from scripts._runtime import set_default_xla_flags

set_default_xla_flags("--xla_gpu_enable_command_buffer=")

import tyro

from agents.backprop import BackpropOpenLoopAgent, BackpropPolicyAgent
from experiments.studies.baseline_study import run_slug, validate_reward
from training.runs import EnvConfig, WandbConfig, load_env, run_native, validate_seeds

Algorithm = Literal[
    "direct_policy", "direct_knots_1", "direct_knots_10", "direct_knots_100"
]


@dataclasses.dataclass
class DirectConfig:
    total_timesteps: int = 10_000_000
    eval_freq: int = 1_000_000
    # Zero selects 64 rollouts per optimizer seed for every direct method.
    # This keeps the GH200 saturated and gives knot schedules the same
    # simulator batch size as the feedback policy.
    num_rollouts: int = 0
    gradient_horizon: int = 32
    # SHAC used 2e-3 for smoother rigid-body simulators. TORAX's stiff
    # implicit solve produced non-finite follow-up gradients at both 2e-3 and
    # 5e-4. A ten-seed SPARC pilot remained finite at every checkpoint with
    # 1e-6, so use that conservative package default.
    policy_learning_rate: float = 1e-6
    knot_learning_rate: float = 5e-2
    hidden_sizes: tuple[int, ...] = (64, 64)
    grad_clip: float = 1.0
    nonfinite_backoff_factor: float = 0.5
    min_update_scale: float = 1e-3
    remat: bool = True


@dataclasses.dataclass
class Config:
    algorithm: Algorithm = "direct_policy"
    env: EnvConfig = dataclasses.field(default_factory=EnvConfig)
    direct: DirectConfig = dataclasses.field(default_factory=DirectConfig)
    wandb: WandbConfig = dataclasses.field(default_factory=WandbConfig)
    seed: int = 0
    num_seeds: int = 10
    history_dir: str | None = None
    checkpoint_dir: str | None = None
    study: str = "debug"
    strict_phase_reward: bool = True


def main(cfg: Config) -> None:
    validate_seeds(cfg.env.backend, cfg.num_seeds)
    if cfg.strict_phase_reward:
        validate_reward(cfg.env.env_setup, cfg.env.reward, cfg.env.backend)
    is_policy = cfg.algorithm == "direct_policy"
    cfg = dataclasses.replace(
        cfg,
        env=dataclasses.replace(
            cfg.env, time_aware=cfg.env.time_aware or not is_policy
        ),
    )
    env = load_env(cfg.env, cfg.env.backend)
    options = dict(
        total_timesteps=cfg.direct.total_timesteps,
        eval_freq=cfg.direct.eval_freq,
        num_rollouts=cfg.direct.num_rollouts or 64,
        gradient_horizon=cfg.direct.gradient_horizon,
        grad_clip=cfg.direct.grad_clip,
        remat=cfg.direct.remat,
        eval_n_envs=cfg.env.eval_n_envs,
        eval_seed=cfg.env.eval_seed,
        init_seed=cfg.seed,
    )
    if is_policy:
        agent = BackpropPolicyAgent.create(
            env,
            **options,
            learning_rate=cfg.direct.policy_learning_rate,
            hidden_sizes=cfg.direct.hidden_sizes,
        )
    else:
        agent = BackpropOpenLoopAgent.create(
            env,
            **options,
            num_knots=int(cfg.algorithm.rsplit("_", 1)[1]),
            learning_rate=cfg.direct.knot_learning_rate,
            nonfinite_backoff_factor=cfg.direct.nonfinite_backoff_factor,
            min_update_scale=cfg.direct.min_update_scale,
        )
    name = run_slug(
        cfg.algorithm,
        cfg.env.env_setup,
        cfg.env.backend or "native",
        cfg.env.variant,
        cfg.env.reward,
        cfg.num_seeds,
    )
    run_native(agent, cfg, name)


if __name__ == "__main__":
    main(tyro.cli(Config))
