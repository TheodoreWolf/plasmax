"""Evaluate the MPC baseline and log W&B metrics like ``scripts/train_ppo.py``.

Default comparison target:

    uv run python scripts/evaluate_baselines.py

uses ``iter/hybrid/flattop`` + ``qlknn`` and inherits its task metadata.
"""

# ruff: noqa: E402,I001

import dataclasses
import time
from pathlib import Path
from typing import Literal

from _runtime import set_default_xla_flags

set_default_xla_flags("--xla_gpu_enable_command_buffer=")

import jax
import jax.numpy as jnp
import numpy as np
import tyro
import wandb
from project_paths import wandb_dir

from experiments.studies.mpc import MPCAgent
from experiments.studies.mpc import rollout as mpc_rollout
from plasmax.environment.factory import make
from plasmax.environment.registry import resolve_backend, resolve_env
from plasmax.rollout import TrajectoryStep, collect_episodes
from plasmax.wrappers import unwrap_to_env_state


@dataclasses.dataclass
class EnvConfig:
    env_setup: str = "iter/hybrid/flattop"
    backend: str = "qlknn"
    reward: str | None = None
    variant: Literal["oracle", "realistic"] = "realistic"
    eval_n_envs: int = 16
    disruption_penalty: float | None = None


@dataclasses.dataclass
class MPCConfig:
    reward_scalar: str = "P_fusion"
    horizon: int = 5
    num_samples: int = 64
    buffer_size: int = 20_000
    hidden: int = 128
    lr: float = 1e-3
    train_batch_size: int = 256
    total_steps: int = 6_000
    log_every: int = 100


@dataclasses.dataclass
class WandbConfig:
    group: str = "baselines"
    project: str = "plasmax"
    entity: str = "flair"
    mode: Literal["online", "offline", "disabled"] = "online"


@dataclasses.dataclass
class Config:
    env: EnvConfig = dataclasses.field(default_factory=EnvConfig)
    mpc: MPCConfig = dataclasses.field(default_factory=MPCConfig)
    wandb: WandbConfig = dataclasses.field(default_factory=WandbConfig)
    seed: int = 0


def _fmt_steps(n: int) -> str:
    s = f"{n:.0e}"
    mantissa, exp = s.split("e")
    return f"{mantissa}e{int(exp)}"


def _stem(alias_or_path: str, resolve) -> str:
    return Path(resolve(alias_or_path)).stem


def _run_name(cfg: Config, agent: str, steps: int) -> str:
    env_name = _stem(cfg.env.env_setup, resolve_env)
    backend_name = _stem(cfg.env.backend, resolve_backend)
    reward_name = cfg.env.reward or "task"
    return (
        f"{env_name}-{backend_name}-{_fmt_steps(steps)}"
        f"-{cfg.env.variant}-{reward_name}-{agent}"
    )


def _last_valid(values, valid):
    indices = jnp.maximum(valid.sum(axis=1) - 1, 0)
    return jax.vmap(lambda row, i: row[i])(values, indices)


def _masked_mean(values, valid):
    mask = valid
    while mask.ndim < values.ndim:
        mask = mask[..., None]
    return jnp.sum(jnp.where(mask, values, 0.0)) / jnp.maximum(jnp.sum(mask), 1)


def _scalar_metrics(traj: TrajectoryStep, action_names) -> dict:
    returns = jnp.sum(jnp.where(traj.valid, traj.reward, 0.0), axis=1)
    lengths = traj.valid.sum(axis=1).astype(jnp.float32)
    env_state = unwrap_to_env_state(traj.env_state)
    po = env_state.plasma
    metrics = {
        "evaluation/return_mean": returns.mean(),
        "evaluation/return_std": returns.std(),
        "evaluation/return_min": returns.min(),
        "evaluation/return_max": returns.max(),
        "evaluation/episode_length_mean": lengths.mean(),
        "obs/Q_fusion": _last_valid(po.Q_fusion, traj.valid).mean(),
        "obs/W_thermal_MJ": _last_valid(po.W_thermal_total, traj.valid).mean() * 1e-6,
        "obs/P_fusion_MW": _last_valid(po.P_fusion, traj.valid).mean() * 1e-6,
        "obs/tau_E_s": _last_valid(po.tau_E, traj.valid).mean(),
        "obs/beta_N": _last_valid(po.beta_N, traj.valid).mean(),
        "obs/q_min": _last_valid(po.q_min, traj.valid).mean(),
        "obs/q95": _last_valid(po.q95, traj.valid).mean(),
        "obs/f_non_inductive": _last_valid(po.f_non_inductive, traj.valid).mean(),
        "obs/fgw_n_e_line_avg": _last_valid(po.fgw_n_e_line_avg, traj.valid).mean(),
        "obs/P_SOL_over_P_LH": _masked_mean(po.P_SOL_total / po.P_LH, traj.valid),
        "ref/Q_fusion": _masked_mean(po.Q_fusion, traj.valid),
        "ref/W_thermal_MJ": _masked_mean(po.W_thermal_total, traj.valid) * 1e-6,
        "ref/P_fusion_GW": _masked_mean(po.P_fusion, traj.valid) * 1e-9,
    }
    for i, name in enumerate(action_names):
        metrics[f"actions/{name}_mean"] = _masked_mean(traj.action[:, :, i], traj.valid)
        metrics[f"actions/{name}_final"] = _last_valid(
            traj.action[:, :, i], traj.valid
        ).mean()
    return {k: float(np.asarray(v)) for k, v in metrics.items()}


def _make_mpc_reward_fn(env, scalar_name: str):
    obs_slice = env.obs_layout().slice_of(scalar_name)

    def reward_fn(obs, action, next_obs):
        del obs, action
        return next_obs[obs_slice].sum()

    return reward_fn


def _load_env(cfg: Config):
    return make(
        cfg.env.env_setup,
        cfg.env.backend,
        reward=cfg.env.reward,
        variant=cfg.env.variant,
        disruption_penalty=cfg.env.disruption_penalty,
    )


def _wandb_run(cfg: Config, agent_name: str, steps: int):
    run_name = _run_name(cfg, agent_name, steps)
    config = dataclasses.asdict(cfg)
    config["run_agent"] = agent_name
    return wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        dir=wandb_dir(),
        name=run_name,
        group=cfg.wandb.group,
        mode=cfg.wandb.mode,
        config=config,
    )


def _run_mpc(cfg: Config) -> None:
    env = _load_env(cfg)
    action_names = [s.name for s in env.unwrapped.actuator_specs]
    reward_fn = _make_mpc_reward_fn(env, cfg.mpc.reward_scalar)
    agent, state = MPCAgent.create(
        env,
        reward_fn,
        horizon=cfg.mpc.horizon,
        num_samples=cfg.mpc.num_samples,
        buffer_size=cfg.mpc.buffer_size,
        hidden=cfg.mpc.hidden,
        lr=cfg.mpc.lr,
        seed=cfg.seed,
    )
    _wandb_run(cfg, "mpc", cfg.mpc.total_steps)

    rollout_fn = jax.jit(
        lambda k, s: mpc_rollout(
            agent,
            s,
            env,
            k,
            cfg.mpc.log_every,
            train_batch_size=cfg.mpc.train_batch_size,
        )
    )
    key = jax.random.key(cfg.seed)
    key, warm_key = jax.random.split(key)
    print("Lowering MPC train chunk...", flush=True)
    t0 = time.monotonic()
    lowered = rollout_fn.lower(warm_key, state)
    t_lower = time.monotonic() - t0
    print(f"Lowered in {t_lower:.3f}s", flush=True)

    print("Compiling MPC train chunk...", flush=True)
    t0 = time.monotonic()
    compiled_rollout = lowered.compile()
    t_compile = time.monotonic() - t0
    print(f"Compiled in {t_compile:.3f}s", flush=True)
    wandb.log({"time/lower_s": t_lower, "time/compile_s": t_compile}, step=0)

    step = 0
    t_train0 = time.monotonic()
    while step < cfg.mpc.total_steps:
        key, chunk_key = jax.random.split(key)
        state, out = compiled_rollout(chunk_key, state)
        jax.block_until_ready(out.reward)
        valid_steps = int(np.asarray(out.valid.sum()))
        step += valid_steps
        losses = np.asarray(out.model_loss)
        finite = np.isfinite(losses)
        valid = np.asarray(out.valid, dtype=bool)
        metrics = {
            "train/chunk_return": float(np.asarray(out.total_return)),
            "train/reward_mean": float(np.asarray(out.reward)[valid].mean()),
            "train/model_loss_mean": float(losses[finite].mean())
            if finite.any()
            else np.nan,
            "train/termination_rate": float(np.asarray(out.terminated)[valid].mean()),
            "train/truncation_rate": float(np.asarray(out.truncated)[valid].mean()),
            "time/train_s": time.monotonic() - t_train0,
        }
        wandb.log(metrics, step=min(step, cfg.mpc.total_steps))
        print(
            f"MPC step={min(step, cfg.mpc.total_steps):>6d} "
            f"chunk_return={metrics['train/chunk_return']:.3f}",
            flush=True,
        )

    def act(obs, rng):
        return agent.act(state, obs, rng)

    eval_steps = env.max_steps
    print("Evaluating frozen MPC policy...", flush=True)
    t0 = time.monotonic()
    traj = collect_episodes(
        act, env, key, num_steps=eval_steps, n_seeds=cfg.env.eval_n_envs
    )
    jax.block_until_ready(traj.reward)
    t_eval = time.monotonic() - t0
    metrics = {**_scalar_metrics(traj, action_names), "time/eval_s": t_eval}
    wandb.log(metrics, step=cfg.mpc.total_steps)
    print(
        f"MPC return={metrics['evaluation/return_mean']:.3f} "
        f"len={metrics['evaluation/episode_length_mean']:.1f}",
        flush=True,
    )
    wandb.finish()


def main(cfg: Config) -> None:
    _run_mpc(cfg)


if __name__ == "__main__":
    main(tyro.cli(Config))
