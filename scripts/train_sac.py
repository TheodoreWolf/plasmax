"""Train Rejax SAC on a plasmax environment, including vmapped seeds.

The defaults follow the original SAC continuous-control configuration: Adam at
3e-4, gamma=0.99, two 256-unit ReLU layers, a 1e6 transition replay buffer,
batch size 256, and target smoothing tau=0.005. Rejax names the retained target
weight ``polyak``, hence ``polyak=0.995``. ``num_epochs=num_envs`` gives one
gradient update per collected transition (UTD=1) while retaining vectorized
simulation.
"""

# ruff: noqa: E402

from __future__ import annotations

import dataclasses
import math
import time
from pathlib import Path
from typing import Literal

from scripts._runtime import set_default_xla_flags

set_default_xla_flags("--xla_gpu_enable_command_buffer=")

import jax
import jax.numpy as jnp
import numpy as np
import tyro

from agents.sac import SACAdapter
from experiments.plotting.wandb_logging import make_buffered_seed_callback
from experiments.studies.baseline_study import run_slug, seed_keys, validate_reward
from plasmax.environment.factory import make
from plasmax.environment.registry import resolve_backend
from plasmax.wrappers import OracleWrappers, RealisticWrappers
from training.envelope_gymnax import EnvelopeGymnax
from training.evaluation import write_transfer_summary
from training.runs import evaluate_transfer, save_run_policies, validate_seeds
from training.vmap_logging import SeedBufferLogger


@dataclasses.dataclass
class EnvConfig:
    env_setup: str = "iter/hybrid/flattop"
    backend: str = "bohm_gyrobohm"
    transfer_backend: str | None = None
    reward: str | None = None
    variant: Literal["oracle", "realistic"] = "realistic"
    disruption_penalty: float | None = None
    eval_n_envs: int = 16
    eval_seed: int = 10_000
    deterministic_eval: bool = True
    transfer_n_envs: int = 128


@dataclasses.dataclass
class SACConfig:
    total_timesteps: int = 10_000_000
    eval_freq: int = 1_000_000
    learning_rate: float = 3e-4
    gamma: float = 0.99
    num_envs: int = 64
    # Set equal to num_envs for one network update per collected transition.
    num_epochs: int = 64
    buffer_size: int = 1_000_000
    fill_buffer: int = 10_000
    batch_size: int = 256
    hidden_sizes: tuple[int, ...] = (256, 256)
    activation: str = "relu"
    polyak: float = 0.995
    target_update_freq: int = 1
    max_grad_norm: float = 10.0
    # Rejax 0.1.2 RMS state widens under TORAX's global x64 setting. Keep the
    # upstream implementation unchanged and leave both normalizers disabled.
    normalize_observations: bool = False
    normalize_rewards: bool = False


@dataclasses.dataclass
class WandbConfig:
    project: str = "plasmax"
    entity: str = "flair"
    group: str = "debug"
    mode: Literal["online", "offline", "disabled"] = "online"
    tags: tuple[str, ...] = ()


@dataclasses.dataclass
class Config:
    env: EnvConfig = dataclasses.field(default_factory=EnvConfig)
    sac: SACConfig = dataclasses.field(default_factory=SACConfig)
    wandb: WandbConfig = dataclasses.field(default_factory=WandbConfig)
    seed: int = 0
    num_seeds: int = 10
    history_dir: str | None = None
    checkpoint_dir: str | None = None
    algorithm: Literal["sac"] = "sac"
    study: str = "debug"
    # Optional study guard; the generic launcher accepts explicit task overrides.
    strict_phase_reward: bool = False


def _fmt_steps(steps: int) -> str:
    mantissa, exponent = f"{steps:.0e}".split("e")
    return f"{mantissa}e{int(exponent)}"


def _sac_extra_metrics(ts, train_metrics):
    del train_metrics
    return {
        "train/replay_buffer_size": ts.replay_buffer.num_entries.astype(jnp.float32)
    }


def _build_algo(cfg: Config, env: EnvelopeGymnax) -> SACAdapter:
    if cfg.sac.num_epochs <= 0:
        raise ValueError("sac.num_epochs must be positive")
    return SACAdapter.create(
        env=env,
        env_params=env.default_params,
        total_timesteps=cfg.sac.total_timesteps,
        eval_freq=cfg.sac.eval_freq,
        learning_rate=cfg.sac.learning_rate,
        gamma=cfg.sac.gamma,
        num_envs=cfg.sac.num_envs,
        num_epochs=cfg.sac.num_epochs,
        buffer_size=cfg.sac.buffer_size,
        fill_buffer=cfg.sac.fill_buffer,
        batch_size=cfg.sac.batch_size,
        hidden_layer_sizes=cfg.sac.hidden_sizes,
        agent_kwargs={"activation": cfg.sac.activation},
        polyak=cfg.sac.polyak,
        target_update_freq=cfg.sac.target_update_freq,
        max_grad_norm=cfg.sac.max_grad_norm,
        normalize_observations=cfg.sac.normalize_observations,
        normalize_rewards=cfg.sac.normalize_rewards,
    )


def _backend_name(alias_or_path: str) -> str:
    return Path(resolve_backend(alias_or_path)).stem


def _load_envelope(cfg: Config, backend: str):
    return (RealisticWrappers if cfg.env.variant == "realistic" else OracleWrappers)(
        make(
            cfg.env.env_setup,
            backend,
            reward=cfg.env.reward,
            disruption_penalty=cfg.env.disruption_penalty,
        )
    )


def main(cfg: Config) -> None:
    validate_seeds(cfg.env.backend, cfg.num_seeds)
    if cfg.strict_phase_reward:
        validate_reward(cfg.env.env_setup, cfg.env.reward, cfg.env.backend)

    run_name = run_slug(
        cfg.algorithm,
        cfg.env.env_setup,
        cfg.env.backend,
        cfg.env.variant,
        cfg.env.reward,
        cfg.num_seeds,
    )
    if cfg.env.transfer_backend is not None:
        run_name = f"{run_name}-to-{_backend_name(cfg.env.transfer_backend)}"
    run_name = f"{run_name}-{_fmt_steps(cfg.sac.total_timesteps)}"
    logger = SeedBufferLogger(
        num_seeds=cfg.num_seeds,
        seed_ids=tuple(range(cfg.seed, cfg.seed + cfg.num_seeds)),
        run_name=run_name,
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        group=cfg.wandb.group,
        mode=cfg.wandb.mode,
        config=dataclasses.asdict(cfg),
        out_dir=cfg.history_dir,
        job_type=cfg.algorithm,
        tags=(cfg.study, cfg.algorithm, *cfg.wandb.tags),
    )

    envelope_env = _load_envelope(cfg, cfg.env.backend)
    env = EnvelopeGymnax(envelope_env)
    algo = _build_algo(cfg, env)
    for_run = make_buffered_seed_callback(
        logger,
        num_steps=env.default_params.max_steps_in_episode,
        n_seeds=cfg.env.eval_n_envs,
        kind="physics",
        eval_rng=jax.random.PRNGKey(cfg.env.eval_seed),
        deterministic=cfg.env.deterministic_eval,
        extra_metrics=_sac_extra_metrics,
    )

    def train_one(rng, run_idx):
        return algo.with_eval_callback(for_run(run_idx)).train(rng)

    keys = seed_keys(cfg.seed, cfg.num_seeds)
    run_indices = jnp.arange(cfg.num_seeds, dtype=jnp.int32)
    train = jax.jit(jax.vmap(train_one))

    start = time.monotonic()
    lowered = train.lower(keys, run_indices)
    lower_seconds = time.monotonic() - start
    start = time.monotonic()
    lowered.compile()
    compile_seconds = time.monotonic() - start
    logger.start_time = time.time()

    start = time.monotonic()
    train_states, results = train(keys, run_indices)
    jax.block_until_ready((train_states, results))
    jax.effects_barrier()
    train_seconds = time.monotonic() - start
    actual_steps = int(np.asarray(train_states.global_step[0]))
    expected_steps = math.ceil(cfg.sac.total_timesteps / cfg.sac.eval_freq)
    expected_steps *= math.ceil(cfg.sac.eval_freq / cfg.sac.num_envs) * cfg.sac.num_envs
    summary: dict[str, float | str] = {
        "time/lower_s": lower_seconds,
        "time/compile_s": compile_seconds,
        "time/train_s": train_seconds,
        "run/actual_train_steps": actual_steps,
        "run/planned_train_steps": expected_steps,
        "run/update_to_data_ratio": cfg.sac.num_epochs / cfg.sac.num_envs,
    }
    if cfg.env.transfer_backend is not None:
        transfer_summary = evaluate_transfer(
            algo,
            train_states,
            envelope_env,
            _load_envelope(cfg, cfg.env.transfer_backend),
            cfg,
            batched=True,
        )
        summary.update(transfer_summary)
        write_transfer_summary(cfg.history_dir, run_name, transfer_summary)
    save_run_policies(
        algo,
        train_states,
        cfg,
        run_name,
        batched=True,
        results=results,
        metrics=summary,
    )
    logger.log_once(summary)
    logger.finish()


if __name__ == "__main__":
    main(tyro.cli(Config))
