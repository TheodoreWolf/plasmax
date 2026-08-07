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

from _runtime import set_default_xla_flags

set_default_xla_flags("--xla_gpu_enable_command_buffer=")

import jax
import jax.numpy as jnp
import numpy as np
import tyro
from rejax.evaluate import evaluate

from agents.sac import SACAdapter
from experiments.plotting.wandb_logging import make_buffered_seed_callback
from experiments.studies.baseline_study import run_slug, seed_keys, validate_reward
from experiments.studies.transfer_eval import transfer_metrics, write_transfer_summary
from plasmax.environment.factory import load_env
from plasmax.environment.registry import resolve_backend
from training.envelope_gymnax import EnvelopeGymnax
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
    return load_env(
        cfg.env.env_setup,
        backend,
        reward=cfg.env.reward,
        variant=cfg.env.variant,
        disruption_penalty=cfg.env.disruption_penalty,
    )


def _evaluate_transfer(
    cfg: Config,
    algo: SACAdapter,
    train_states,
) -> dict[str, float | str]:
    if cfg.env.transfer_backend is None:
        raise ValueError("transfer_backend is required for transfer evaluation")

    source_backend = _backend_name(cfg.env.backend)
    target_backend = _backend_name(cfg.env.transfer_backend)
    print(f"Loading transfer env ({target_backend})...", flush=True)
    source_env = algo.env
    target_env = EnvelopeGymnax(_load_envelope(cfg, cfg.env.transfer_backend))

    def evaluate_seed(train_state, key):
        act = (
            algo.make_deterministic_act(train_state)
            if cfg.env.deterministic_eval
            else algo.make_act(train_state)
        )
        source_lengths, source_returns = evaluate(
            act,
            key,
            source_env,
            source_env.default_params,
            num_seeds=cfg.env.transfer_n_envs,
        )
        target_lengths, target_returns = evaluate(
            act,
            key,
            target_env,
            target_env.default_params,
            num_seeds=cfg.env.transfer_n_envs,
        )
        return source_returns, source_lengths, target_returns, target_lengths

    eval_key = jax.random.PRNGKey(cfg.env.eval_seed)
    eval_keys = jnp.broadcast_to(eval_key, (cfg.num_seeds, *eval_key.shape))
    evaluate_all = jax.jit(jax.vmap(evaluate_seed))
    start = time.monotonic()
    outputs = evaluate_all(train_states, eval_keys)
    jax.block_until_ready(outputs[2])
    eval_seconds = time.monotonic() - start
    source_returns, source_lengths, target_returns, target_lengths = (
        np.asarray(value) for value in outputs
    )
    metrics = transfer_metrics(
        source_backend,
        target_backend,
        source_returns,
        source_lengths,
        target_returns,
        target_lengths,
        eval_seconds,
    )
    print(
        f"Transfer {source_backend} -> {target_backend}: "
        f"source return {metrics['transfer/source_return_mean']:.3f}, "
        f"target return {metrics['transfer/target_return_mean']:.3f} "
        f"(ratio {metrics['transfer/return_ratio']:.3f})",
        flush=True,
    )
    return metrics


def main(cfg: Config) -> None:
    if cfg.num_seeds <= 0:
        raise ValueError("num_seeds must be positive")
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
    train_states, _ = train(keys, run_indices)
    jax.block_until_ready(train_states.global_step)
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
        transfer_summary = _evaluate_transfer(cfg, algo, train_states)
        summary.update(transfer_summary)
        write_transfer_summary(cfg.history_dir, run_name, transfer_summary)
    logger.log_once(summary)
    logger.finish()


if __name__ == "__main__":
    main(tyro.cli(Config))
