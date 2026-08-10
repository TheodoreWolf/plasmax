"""PPO training entrypoint, consolidating four previously-separate scripts.

Behaviour is selected by two flags:

* ``--num-seeds N`` (default 1): ``N == 1`` runs a single training job with
  the full rich-plotting wandb callback (``make_training_callback``, Plotly
  figures + poloidal animation). ``N > 1`` vmaps ``N`` independent seeds with
  a single ``jax.vmap(algo.train)`` call and logs cross-seed mean +/- std via
  :class:`training.vmap_logging.SeedBufferLogger` (scalars only —
  per-seed figures can't be meaningfully averaged).
* ``--env.transfer-backend NAME`` (default unset): after training, zero-shot
  evaluate the frozen policy (with its frozen obs-normalisation stats) on a
  second, typically higher-fidelity backend via rejax's ``evaluate``, logging
  the transfer gap under ``transfer/``. Works with either seed mode.

This subsumes ``train_ppo_wandb.py``, ``train_ppo_vmap.py``,
``train_transfer_ppo.py``, and ``train_transfer_ppo_vmap.py``.

``--env.env_setup`` / ``--env.backend`` / ``--env.transfer_backend`` accept
either a ``plasmax.environment.registry`` alias (e.g. ``iter/hybrid/flattop``,
``cgm``) or a raw YAML path — see ``registry.ENV_ALIASES`` / ``BACKEND_ALIASES``.

Run inside Docker, e.g.:
    python3 scripts/train_ppo.py \\
        --env.env_setup iter/hybrid/flattop \\
        --env.backend   cgm

    # multi-seed:
    python3 scripts/train_ppo.py --num-seeds 3 ...

    # zero-shot transfer eval after training:
    python3 scripts/train_ppo.py --env.transfer-backend qlknn ...
"""

# ruff: noqa: E402

import dataclasses
import time
from pathlib import Path
from typing import Literal

from _runtime import set_default_xla_flags

# Disable command-buffer dispatch: vmapping NUM_SEEDS independent runs
# multiplies the number of "alive" CUDA graphs XLA keeps around for
# command-buffer dispatch, which OOMs a 40GB A100 at 3+ seeds on a
# long TORAX scenario. Must be set before importing jax.
set_default_xla_flags("--xla_gpu_enable_command_buffer=")

import jax
import jax.numpy as jnp
import numpy as np
import tyro
import wandb
from project_paths import wandb_dir
from rejax.evaluate import evaluate

from agents.ppo import PPOAdapter
from experiments.plotting.wandb_logging import (
    make_buffered_seed_callback,
    make_minimal_training_callback,
    make_training_callback,
    make_world_model_training_callback,
)
from experiments.studies.baseline_study import seed_keys
from experiments.studies.disruption_sweep import (
    calibrated_disruption_penalty,
)
from experiments.studies.transfer_eval import (
    transfer_metrics as _transfer_metrics,
)
from experiments.studies.transfer_eval import (
    write_transfer_summary,
)
from plasmax.environment.config import backend_kind
from plasmax.environment.factory import make
from plasmax.environment.merge import env_key
from plasmax.environment.registry import resolve_backend, resolve_env
from training.envelope_gymnax import EnvelopeGymnax
from training.vmap_logging import SeedBufferLogger

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class EnvConfig:
    env_setup: str = "iter/hybrid/flattop"
    backend: str = "bohm_gyrobohm"
    # If set, zero-shot evaluate the trained policy on this second, typically
    # higher-fidelity, backend after training (must share env_setup, and
    # therefore obs/action spaces, with `backend`). registry alias or raw path.
    transfer_backend: str | None = None
    reward: str | None = None
    variant: Literal["oracle", "realistic"] = "realistic"
    eval_n_envs: int = 128
    # Fixed evaluation seed, independent of the training RNG.
    eval_seed: int = 0
    # Evaluate the distribution mode rather than sampling policy actions.
    deterministic_eval: bool = False
    dt: float = 0.1  # must match the env YAML's numerics.fixed_dt
    # Omitted values inherit the task YAML. Explicit zero disables the penalty.
    disruption_penalty: float | None = None
    # Optional sensitivity-study multiplier for the task YAML penalty.
    disruption_kappa: float | None = None
    # Force the returns-only eval callback even for a TORAX backend (auto-on for
    # world-model envs, and always used when num_seeds > 1). Skips the
    # physics/geometry logging graph.
    force_minimal_callback: bool = False
    # Leave-one-out ablation of the realistic obs stack (variant=realistic
    # only): none | noise | resolution | filter | delay. See scenario_config._ABLATABLE.
    ablate: str = "none"
    # Append the environment time coordinate as an extra observation scalar.
    time_aware: bool = False
    # Discretize every actuator into this many evenly spaced bins (MultiDiscrete
    # action space; realistic variant only). None = continuous actions.
    quantize_bins: int | None = None
    # Number of episodes to roll out per seed during the transfer evaluation.
    transfer_n_envs: int = 128


@dataclasses.dataclass
class PPOConfig:
    total_timesteps: int = 10_000_000
    num_envs: int = 1024
    num_steps: int = 100
    num_epochs: int = 4
    num_minibatches: int = 4
    eval_freq: int = 1_000_000
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5
    # Rejax 0.1.2 RMS state widens under TORAX's global x64 setting; keep its
    # upstream implementation unchanged and leave normalization disabled.
    normalize_rewards: bool = False
    normalize_observations: bool = False
    hidden_sizes: tuple[int, ...] = (64, 64)
    activation: str = "swish"
    # Center the Gaussian mean on the environment's reset action setpoint and
    # initialize the final residual layer to zero.
    residual_policy: bool = False
    # Initial log standard deviation for the residual Gaussian actor.
    initial_log_std: float = 0.0


@dataclasses.dataclass
class WandbConfig:
    group: str = "debug"
    project: str = "plasmax"
    entity: str = "flair"
    mode: Literal["online", "offline"] = "online"
    tags: tuple[str, ...] = ()


@dataclasses.dataclass
class Config:
    env: EnvConfig = dataclasses.field(default_factory=EnvConfig)
    ppo: PPOConfig = dataclasses.field(default_factory=PPOConfig)
    wandb: WandbConfig = dataclasses.field(default_factory=WandbConfig)
    seed: int = 0  # base PRNG seed; per-seed keys are split from this
    num_seeds: int = 1  # >1 vmaps that many independent training runs
    algorithm: Literal["ppo"] = "ppo"
    study: str = "debug"
    # Optional dir to dump per-seed metric history as an .npz (numpy only,
    # num_seeds > 1 only).
    history_dir: str | None = None


def _fmt_steps(n: int) -> str:
    """Format timesteps as compact scientific notation, e.g. 500_000 -> '5e5'."""
    s = f"{n:.0e}"
    mantissa, exp = s.split("e")
    return f"{mantissa}e{int(exp)}"


def _stem(alias_or_path: str, resolve) -> str:
    """Short label for a registry alias or raw YAML path, e.g. 'cgm'."""
    return Path(resolve(alias_or_path)).stem


def _env_label(alias_or_path: str) -> str:
    """Run-name label for an env: its address with '/' flattened to '_', e.g.
    'iter/hybrid/flattop' or '/path/to/custom_env.yaml' ->
    'iter_hybrid_flattop'."""
    return env_key(resolve_env(alias_or_path)).replace("/", "_")


def _backend_kind(alias_or_path: str) -> str:
    return backend_kind(resolve_backend(alias_or_path))


def _run_name(cfg: Config) -> str:
    env_name = _env_label(cfg.env.env_setup)
    backend_name = _stem(cfg.env.backend, resolve_backend)
    is_world_model = _backend_kind(cfg.env.backend) == "world_model"
    reward_name = cfg.env.reward or "task"

    if cfg.env.transfer_backend is not None:
        transfer_backend_name = _stem(cfg.env.transfer_backend, resolve_backend)
        name = (
            f"{env_name}-{backend_name}2{transfer_backend_name}-{cfg.env.variant}"
            f"-{_fmt_steps(cfg.ppo.total_timesteps)}-{reward_name}"
        )
    else:
        name = f"{env_name}-{backend_name}-{_fmt_steps(cfg.ppo.total_timesteps)}"
        # World models use their native reward and observation interface.
        if not is_world_model:
            name += f"-{cfg.env.variant}-{reward_name}"

    if cfg.env.time_aware:
        name += "-time_aware"
    if cfg.env.quantize_bins is not None:
        name += f"-q{cfg.env.quantize_bins}"
    if cfg.env.disruption_penalty is not None:
        if cfg.env.disruption_penalty != 0.0:
            name += f"-dp{cfg.env.disruption_penalty:.4g}"
    elif cfg.env.disruption_kappa is not None:
        name += f"-kappa{cfg.env.disruption_kappa:g}"
    if cfg.ppo.residual_policy:
        name += "-residual"
    if cfg.env.deterministic_eval:
        name += "-det_eval"
    if cfg.num_seeds > 1:
        name += f"-{cfg.num_seeds}seeds"
    return name


def _timed(label: str, fn):
    """Prints ``label``, runs ``fn()``, prints elapsed; returns (result, seconds)."""
    print(f"{label}...", flush=True)
    t0 = time.monotonic()
    result = fn()
    dt = time.monotonic() - t0
    print(f"{label} done in {dt:.3f}s", flush=True)
    return result, dt


def _print_transfer_metrics(metrics: dict[str, float | str]) -> None:
    print(
        f"Transfer {metrics['transfer/source_backend']} -> "
        f"{metrics['transfer/target_backend']}: "
        f"source return {metrics['transfer/source_return_mean']:.3f}, "
        f"target return {metrics['transfer/target_return_mean']:.3f} "
        f"(ratio {metrics['transfer/return_ratio']:.3f})",
        flush=True,
    )


def _build_algo(cfg: Config, env):
    gymnax_env = EnvelopeGymnax(env)
    return PPOAdapter.create(
        env=gymnax_env,
        env_params=gymnax_env.default_params,
        total_timesteps=cfg.ppo.total_timesteps,
        num_envs=cfg.ppo.num_envs,
        num_steps=cfg.ppo.num_steps,
        num_epochs=cfg.ppo.num_epochs,
        num_minibatches=cfg.ppo.num_minibatches,
        eval_freq=cfg.ppo.eval_freq,
        learning_rate=cfg.ppo.learning_rate,
        gamma=cfg.ppo.gamma,
        gae_lambda=cfg.ppo.gae_lambda,
        clip_eps=cfg.ppo.clip_eps,
        vf_coef=cfg.ppo.vf_coef,
        ent_coef=cfg.ppo.ent_coef,
        max_grad_norm=cfg.ppo.max_grad_norm,
        normalize_rewards=cfg.ppo.normalize_rewards,
        normalize_observations=cfg.ppo.normalize_observations,
        agent_kwargs={
            "hidden_layer_sizes": cfg.ppo.hidden_sizes,
            "activation": cfg.ppo.activation,
            "residual_policy": cfg.ppo.residual_policy,
            "initial_log_std": cfg.ppo.initial_log_std,
        },
    )


def _load_env(cfg: Config, backend_alias_or_path: str):
    # make resolves registry aliases for both env_setup and backend
    # internally (plasmax.environment.registry.resolve_env/resolve_backend).
    if cfg.env.disruption_penalty is not None:
        disruption_penalty = cfg.env.disruption_penalty
    elif cfg.env.disruption_kappa is not None:
        disruption_penalty = calibrated_disruption_penalty(
            env_key(resolve_env(cfg.env.env_setup)),
            backend_alias_or_path,
            kappa=cfg.env.disruption_kappa,
        )
    else:
        disruption_penalty = None
    return make(
        cfg.env.env_setup,
        backend_alias_or_path,
        reward=cfg.env.reward,
        variant=cfg.env.variant,
        disruption_penalty=disruption_penalty,
        ablate=cfg.env.ablate,
        time_aware=cfg.env.time_aware,
        quantize_bins=cfg.env.quantize_bins,
    )


# ---------------------------------------------------------------------------
# Single-seed training (num_seeds == 1)
# ---------------------------------------------------------------------------


def _train_single(cfg: Config, env, algo):
    del env
    episode_steps = algo.env_params.max_steps_in_episode
    evaluator_options = {
        "eval_rng": jax.random.PRNGKey(cfg.env.eval_seed),
        "deterministic": cfg.env.deterministic_eval,
    }

    if cfg.env.force_minimal_callback:
        eval_cb = make_minimal_training_callback(
            num_steps=episode_steps,
            n_seeds=cfg.env.eval_n_envs,
            **evaluator_options,
        )
    elif _backend_kind(cfg.env.backend) == "world_model":
        eval_cb = make_world_model_training_callback(
            num_steps=episode_steps,
            n_seeds=cfg.env.eval_n_envs,
            **evaluator_options,
        )
    else:
        eval_cb = make_training_callback(
            num_steps=episode_steps,
            n_seeds=cfg.env.eval_n_envs,
            dt=cfg.env.dt,
            **evaluator_options,
        )
    algo = algo.with_eval_callback(eval_cb)

    rng = jax.random.PRNGKey(cfg.seed)
    train_fn = jax.jit(algo.train)

    lowered, t_lower = _timed("Lowering", lambda: train_fn.lower(rng))
    _, t_compile = _timed("Compiling", lowered.compile)  # warms the jit cache
    wandb.log({"time/lower_s": t_lower, "time/compile_s": t_compile}, step=0)

    # Call the jitted function directly (reusing the compilation warmed above)
    # rather than the Lowered.compile() executable: the latter's Compiled.__call__
    # trips a const-arg mismatch on JAX 0.10.x ("compiled for N inputs but called
    # with 1") because algo.train closes over many constant arrays.
    def _run():
        ts, _ = train_fn(rng)
        jax.effects_barrier()
        return ts

    ts, t_train = _timed("Training", _run)
    wandb.log({"time/train_s": t_train})
    return ts


def _transfer_eval_single(cfg: Config, backend_name: str, env, algo, ts):
    del env
    transfer_backend_name = _stem(cfg.env.transfer_backend, resolve_backend)
    act = (
        algo.make_deterministic_act(ts)
        if cfg.env.deterministic_eval
        else algo.make_act(ts)
    )

    print(f"Loading transfer env ({transfer_backend_name})...", flush=True)
    transfer_env = EnvelopeGymnax(_load_env(cfg, cfg.env.transfer_backend))
    source_env = algo.env

    eval_rng = jax.random.PRNGKey(cfg.env.eval_seed)

    def _eval_both():
        src_lengths, src_returns = evaluate(
            act,
            eval_rng,
            source_env,
            source_env.default_params,
            num_seeds=cfg.env.transfer_n_envs,
        )
        tgt_lengths, tgt_returns = evaluate(
            act,
            eval_rng,
            transfer_env,
            transfer_env.default_params,
            num_seeds=cfg.env.transfer_n_envs,
        )
        jax.block_until_ready(tgt_returns)
        return src_returns, src_lengths, tgt_returns, tgt_lengths

    outs, t_eval = _timed("Evaluating source + transfer backends", _eval_both)
    # Single seed: add a leading seed axis so the metrics helper is shared.
    src_returns, src_lengths, tgt_returns, tgt_lengths = (
        np.asarray(x)[None] for x in outs
    )
    metrics = _transfer_metrics(
        backend_name,
        transfer_backend_name,
        src_returns,
        src_lengths,
        tgt_returns,
        tgt_lengths,
        t_eval,
    )
    _print_transfer_metrics(metrics)
    wandb.log(metrics)


def _run_single(cfg: Config, run_name: str) -> None:
    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        dir=wandb_dir(),
        name=run_name,
        group=cfg.wandb.group,
        mode=cfg.wandb.mode,
        config=dataclasses.asdict(cfg),
        tags=list(cfg.wandb.tags),
        job_type=cfg.algorithm,
    )

    env = _load_env(cfg, cfg.env.backend)
    algo = _build_algo(cfg, env)

    ts = _train_single(cfg, env, algo)

    if cfg.env.transfer_backend is not None:
        backend_name = _stem(cfg.env.backend, resolve_backend)
        _transfer_eval_single(cfg, backend_name, env, algo, ts)

    wandb.finish()
    print("Done.")


# ---------------------------------------------------------------------------
# Multi-seed training (num_seeds > 1), vmapped
# ---------------------------------------------------------------------------


def _run_vmap(cfg: Config, run_name: str) -> None:
    logger = SeedBufferLogger(
        num_seeds=cfg.num_seeds,
        run_name=run_name,
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        group=cfg.wandb.group,
        mode=cfg.wandb.mode,
        config=dataclasses.asdict(cfg),
        out_dir=cfg.history_dir,
        seed_ids=tuple(range(cfg.seed, cfg.seed + cfg.num_seeds)),
        job_type=cfg.algorithm,
        tags=cfg.wandb.tags,
    )

    env = _load_env(cfg, cfg.env.backend)
    algo = _build_algo(cfg, env)
    episode_steps = algo.env_params.max_steps_in_episode

    # World-model envs carry no TORAX postout; log returns-only.
    is_world_model = _backend_kind(cfg.env.backend) == "world_model"
    kind = (
        "minimal" if (is_world_model or cfg.env.force_minimal_callback) else "physics"
    )
    for_run = make_buffered_seed_callback(
        logger,
        num_steps=episode_steps,
        n_seeds=cfg.env.eval_n_envs,
        kind=kind,
        eval_rng=jax.random.PRNGKey(cfg.env.eval_seed),
        deterministic=cfg.env.deterministic_eval,
    )

    def train_one(rng, run_idx):
        # Rebuild the callback inside the trace so each vmapped seed's run_idx
        # tracer flows through to logger.log via jax.debug.callback.
        algo_i = algo.with_eval_callback(for_run(run_idx))
        return algo_i.train(rng)

    seeds = seed_keys(cfg.seed, cfg.num_seeds)
    run_idxs = jnp.arange(cfg.num_seeds)
    train_fn = jax.jit(jax.vmap(train_one))

    lowered, t_lower = _timed(
        f"Lowering ({cfg.num_seeds} seeds)", lambda: train_fn.lower(seeds, run_idxs)
    )
    _, t_compile = _timed("Compiling", lowered.compile)

    logger.start_time = time.time()

    def _run():
        ts, _evals = train_fn(seeds, run_idxs)
        jax.effects_barrier()
        return ts

    ts, t_train = _timed("Training", _run)

    log_once = {
        "time/lower_s": t_lower,
        "time/compile_s": t_compile,
        "time/train_s": t_train,
    }

    if cfg.env.transfer_backend is not None:
        backend_name = _stem(cfg.env.backend, resolve_backend)
        transfer_backend_name = _stem(cfg.env.transfer_backend, resolve_backend)
        transfer_summary = _transfer_eval_vmap(
            cfg,
            backend_name,
            transfer_backend_name,
            env,
            algo,
            ts,
        )
        log_once.update(transfer_summary)
        write_transfer_summary(cfg.history_dir, run_name, transfer_summary)

    logger.log_once(log_once)
    logger.finish()
    print("Done.")


def _transfer_eval_vmap(
    cfg: Config, backend_name, transfer_backend_name, env, algo, ts
):
    del env
    # Transfer: evaluate every seed's trained policy (frozen obs-norm stats
    # baked into `act` via make_act) on both backends. `evaluate`'s `act`/`env`
    # args are static, so each seed's `act` closure is built *inside* the
    # vmapped function over the stacked train states.
    print(f"Loading transfer env ({transfer_backend_name})...", flush=True)
    transfer_env = EnvelopeGymnax(_load_env(cfg, cfg.env.transfer_backend))
    source_env = algo.env

    def eval_transfer(seed_ts, rng):
        act = (
            algo.make_deterministic_act(seed_ts)
            if cfg.env.deterministic_eval
            else algo.make_act(seed_ts)
        )
        src_lengths, src_returns = evaluate(
            act,
            rng,
            source_env,
            source_env.default_params,
            num_seeds=cfg.env.transfer_n_envs,
        )
        tgt_lengths, tgt_returns = evaluate(
            act,
            rng,
            transfer_env,
            transfer_env.default_params,
            num_seeds=cfg.env.transfer_n_envs,
        )
        return src_returns, src_lengths, tgt_returns, tgt_lengths

    eval_rng = jax.random.PRNGKey(cfg.env.eval_seed)
    eval_rngs = jnp.broadcast_to(eval_rng, (cfg.num_seeds, *eval_rng.shape))
    eval_fn = jax.jit(jax.vmap(eval_transfer))

    def _eval_all():
        outs = eval_fn(ts, eval_rngs)
        jax.block_until_ready(outs[2])
        return outs

    outs, t_eval = _timed(
        f"Evaluating {cfg.num_seeds} seeds on source + transfer backends", _eval_all
    )
    # Shapes: (num_seeds, transfer_n_envs).
    src_returns, src_lengths, tgt_returns, tgt_lengths = (np.asarray(x) for x in outs)
    metrics = _transfer_metrics(
        backend_name,
        transfer_backend_name,
        src_returns,
        src_lengths,
        tgt_returns,
        tgt_lengths,
        t_eval,
    )
    _print_transfer_metrics(metrics)
    return metrics


def main(cfg: Config) -> None:
    run_name = _run_name(cfg)
    if cfg.num_seeds > 1:
        _run_vmap(cfg, run_name)
    else:
        _run_single(cfg, run_name)


if __name__ == "__main__":
    main(tyro.cli(Config))
