"""Evaluate a frozen direct-baseline checkpoint on a second backend."""

# ruff: noqa: E402

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Literal

from scripts._runtime import set_default_xla_flags

set_default_xla_flags("--xla_gpu_enable_command_buffer=")

import jax
import jax.numpy as jnp
import numpy as np
import tyro
import wandb
from flax import serialization

from experiments.studies.baseline_study import seed_keys
from experiments.studies.train_backprop_policy import ResidualPolicy
from experiments.studies.train_direct_baseline import (
    Config as TrainConfig,
)
from experiments.studies.train_direct_baseline import (
    DirectConfig,
    EnvConfig,
    WandbConfig,
    _evaluate_transfer,
    _load_envelope,
)
from scripts.project_paths import wandb_dir


@dataclasses.dataclass
class Config:
    checkpoint: str
    target_backend: str = "tglfnn"
    transfer_n_envs: int = 128
    out: str | None = None
    wandb_project: str = "plasmax"
    wandb_entity: str = "flair"
    wandb_group: str = "baseline-bgb2tglfnn-iter-hybrid-flattop"
    wandb_mode: Literal["online", "offline", "disabled"] = "online"


def _train_config(config: dict, args: Config) -> TrainConfig:
    env_values = dict(config["env"])
    env_values.update(
        transfer_backend=args.target_backend,
        transfer_n_envs=args.transfer_n_envs,
    )
    return TrainConfig(
        algorithm=config["algorithm"],
        env=EnvConfig(**env_values),
        direct=DirectConfig(**config["direct"]),
        wandb=WandbConfig(**config["wandb"]),
        seed=config["seed"],
        num_seeds=config["num_seeds"],
        history_dir=config.get("history_dir"),
        study=config.get("study", "unknown"),
        strict_phase_reward=config.get("strict_phase_reward", True),
    )


def _restore_policy_parameters(
    cfg: TrainConfig,
    serialized_parameters: np.ndarray,
):
    env = _load_envelope(cfg, cfg.env.backend)
    init_key = jax.random.fold_in(jax.random.key(cfg.seed), 0x5E7)
    _, info = env.init(init_key)
    policy = ResidualPolicy(
        action_dim=env.action_space.shape[0],
        hidden_sizes=cfg.direct.hidden_sizes,
    )
    dummy_obs = jnp.zeros(info.obs.shape, dtype=jnp.float32)
    optimizer_keys = seed_keys(cfg.seed, cfg.num_seeds)
    policy_keys = jax.vmap(lambda key: jax.random.fold_in(key, 0x1A17))(optimizer_keys)
    template = jax.vmap(lambda key: policy.init(key, dummy_obs)["params"])(policy_keys)
    parameters = serialization.from_bytes(
        template,
        serialized_parameters.tobytes(),
    )
    return env, parameters


def main(args: Config) -> None:
    checkpoint = Path(args.checkpoint)
    with np.load(checkpoint, allow_pickle=False) as payload:
        config_values = json.loads(str(payload["config_json"]))
        summary = json.loads(str(payload["summary_json"]))
        cfg = _train_config(config_values, args)
        if cfg.direct.total_timesteps != 10_000_000:
            raise ValueError(
                f"checkpoint used {cfg.direct.total_timesteps} transitions, not 10M"
            )
        if cfg.algorithm == "direct_policy":
            source_env, parameters = _restore_policy_parameters(
                cfg,
                payload["policy_params"],
            )
        else:
            source_env = _load_envelope(cfg, cfg.env.backend)
            parameters = jnp.asarray(payload["theta"])

    run_name = (
        f"transfer-{cfg.algorithm}-{cfg.env.env_setup.replace('/', '_')}-"
        f"{cfg.env.backend}2{args.target_backend}-{cfg.env.variant}-10m"
    )
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        dir=wandb_dir(),
        name=run_name,
        group=args.wandb_group,
        mode=args.wandb_mode,
        job_type=cfg.algorithm,
        config={
            "checkpoint": str(checkpoint),
            "source_config": config_values,
            "target_backend": args.target_backend,
            "transfer_n_envs": args.transfer_n_envs,
        },
        tags=["backend-transfer", cfg.algorithm, cfg.env.variant],
    )
    metrics = _evaluate_transfer(cfg, source_env, parameters, summary)
    run.summary.update(metrics)
    run.finish()

    out = (
        Path(args.out)
        if args.out is not None
        else checkpoint.with_name(f"{checkpoint.stem}_to_{args.target_backend}.json")
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "checkpoint": str(checkpoint),
                "run_name": run_name,
                "metrics": metrics,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(f"Saved transfer summary: {out}")


if __name__ == "__main__":
    main(tyro.cli(Config))
