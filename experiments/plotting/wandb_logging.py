"""Wandb logging utilities for plasmax training.

Builds rejax ``eval_callback``s that collect episode trajectories inside JIT
and log physics scalars (and, for the single-seed TORAX callback, interactive
Plotly episode figures) to wandb via ``jax.experimental.io_callback``.
"""

import jax
import jax.numpy as jnp
import numpy as np
import plotly.graph_objects as go
import wandb
from plotly.subplots import make_subplots

from experiments.plotting.viz import (
    PROFILE_LABELS,
    make_profile_rho_figure,
)
from plasmax.rollout import collect_episodes
from plasmax.wrappers import unwrap_to_env_state
from training.envelope_gymnax import to_typed_key

# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

# Display scale + y-axis unit per actuator for the wandb action figures.
# Actuators not listed plot unscaled with a blank unit.
_ACTUATOR_UNITS: dict[str, tuple[float, str]] = {
    "P_nbi": (1e-6, "MW"),
    "P_eccd": (1e-6, "MW"),
    "rho_eccd": (1.0, "rho_norm"),
    "generic_current_A": (1e-6, "MA"),
    "gas_puff_rate": (1e-21, "10²¹/s"),
    "pellet_rate": (1e-21, "10²¹/s"),
}

_PHYSICS_LABELS = [
    "W_thermal",
    "P_fusion",
    "P_aux",
    "Q_fusion",
    "tau_E",
    "H98",
    "q_min",
    "q95",
    "beta_N",
    "f_non_inductive",
]
_PHYSICS_YLABELS = ["MJ", "MW", "MW", "", "s", "", "", "", "", ""]
_PHYSICS_SCALES = [1e-6, 1e-6, 1e-6, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

_BAND_FILL = "rgba(99,110,250,0.2)"
_LINE_COLOR = "rgb(99,110,250)"


def _add_band_traces(fig, row, t, mean, std, label):
    """Add mean ± std band traces to a subplot panel."""
    fig.add_trace(
        go.Scatter(
            x=t,
            y=mean - std,
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        ),
        row=row,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=t,
            y=mean + std,
            fill="tonexty",
            fillcolor=_BAND_FILL,
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        ),
        row=row,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=t,
            y=mean,
            name=label,
            line=dict(color=_LINE_COLOR),
        ),
        row=row,
        col=1,
    )


def _make_band_fig(
    t_s, mean, std, labels, ylabels, scales, title, vertical_spacing
) -> go.Figure:
    """One panel per quantity, each with mean ± std bands.

    ``mean``/``std`` are ``(n_steps, len(labels))``; column ``i`` is scaled by
    ``scales[i]`` and titled ``labels[i]`` with y-axis unit ``ylabels[i]``.
    """
    n = len(labels)
    fig = make_subplots(
        rows=n,
        cols=1,
        shared_xaxes=True,
        subplot_titles=labels,
        vertical_spacing=vertical_spacing,
    )
    for i, (label, ylabel, scale) in enumerate(
        zip(labels, ylabels, scales, strict=True)
    ):
        row = i + 1
        _add_band_traces(fig, row, t_s, mean[:, i] * scale, std[:, i] * scale, label)
        fig.update_yaxes(title_text=ylabel, row=row, col=1)

    fig.update_xaxes(title_text="Time (s)", row=n, col=1)
    fig.update_layout(height=200 * n, showlegend=False, title_text=title)
    return fig


# ---------------------------------------------------------------------------
# Shared metric builders / logging
# ---------------------------------------------------------------------------


def _collect_returns_and_lengths(
    algo,
    ts,
    rng,
    num_steps,
    n_seeds,
    lean=False,
    deterministic=False,
):
    """Rolls out the current policy; returns ``(traj, returns, lengths)``.

    ``returns`` are summed per-episode rewards (shape ``(n_seeds,)``).
    ``lengths`` count valid transitions per seed. ``lean`` nulls heavy per-step
    SimState fields eval never reads (TORAX envs only).
    """
    act = algo.make_deterministic_act(ts) if deterministic else algo.make_act(ts)
    env = getattr(algo.env, "envelope_env", algo.env)
    traj = collect_episodes(
        act,
        env,
        to_typed_key(rng),
        num_steps=num_steps,
        n_seeds=n_seeds,
        lean=lean,
    )
    episode_returns = jnp.sum(jnp.where(traj.valid, traj.reward, 0.0), axis=1)
    episode_lengths = jnp.sum(traj.valid, axis=1).astype(jnp.float32)
    return traj, episode_returns, episode_lengths


def _last_valid(values, valid):
    """Gather each seed's final valid time entry without dynamic shapes."""
    indices = jnp.maximum(jnp.sum(valid, axis=1) - 1, 0)
    return jax.vmap(lambda row, i: row[i])(values, indices)


def _masked_mean(values, valid):
    """Mean over seed/time axes, excluding invalid fixed-scan padding."""
    mask = valid
    while mask.ndim < values.ndim:
        mask = mask[..., None]
    count = jnp.sum(mask, axis=(0, 1))
    total = jnp.sum(jnp.where(mask, values, 0.0), axis=(0, 1))
    return total / jnp.maximum(count, 1)


def _masked_band(values, valid):
    """Per-time mean/std across seeds with NaNs after every episode ended."""
    mask = valid
    while mask.ndim < values.ndim:
        mask = mask[..., None]
    count = jnp.sum(mask, axis=0)
    safe_count = jnp.maximum(count, 1)
    mean = jnp.sum(jnp.where(mask, values, 0.0), axis=0) / safe_count
    variance = jnp.sum(jnp.where(mask, (values - mean) ** 2, 0.0), axis=0)
    variance = variance / safe_count
    present = count > 0
    return jnp.where(present, mean, jnp.nan), jnp.where(
        present, jnp.sqrt(variance), jnp.nan
    )


def _conditional_mean(values, mask):
    count = jnp.sum(mask)
    total = jnp.sum(jnp.where(mask, values, 0.0))
    return jnp.where(count > 0, total / count, jnp.nan)


def _base_metrics(traj, episode_returns, episode_lengths, train_metrics) -> dict:
    """Evaluation metrics, plus diagnostics when an adapter provides them."""
    metrics = {
        "evaluation/return_mean": episode_returns.mean(),
        "evaluation/return_std": episode_returns.std(),
        "evaluation/return_min": episode_returns.min(),
        "evaluation/return_max": episode_returns.max(),
        "evaluation/episode_length_mean": episode_lengths.mean(),
        **_termination_metrics(traj, episode_returns, episode_lengths),
    }
    if train_metrics is not None:
        metrics.update(
            {
                "train/actor_loss": train_metrics.actor_loss,
                "train/critic_loss": train_metrics.critic_loss,
                "train/entropy": train_metrics.entropy,
                "train/actor_grad_norm": train_metrics.actor_grad_norm,
                "train/critic_grad_norm": train_metrics.critic_grad_norm,
            }
        )
    return metrics


def _termination_metrics(traj, episode_returns, episode_lengths) -> dict:
    """Episode completion and failure metrics."""
    terminated = traj.terminated & traj.valid
    truncated = traj.truncated & traj.valid
    has_terminated = jnp.any(terminated, axis=1)
    has_completed = jnp.any(truncated, axis=1)
    metrics = {
        "termination/completion_rate": has_completed.astype(jnp.float32).mean(),
        "termination/termination_rate": has_terminated.astype(jnp.float32).mean(),
        "evaluation/episode_fraction_mean": episode_lengths.mean()
        / traj.valid.shape[1],
        "termination/failure_step_mean": _conditional_mean(
            episode_lengths,
            has_terminated,
        ),
        "evaluation/return_completed_mean": _conditional_mean(
            episode_returns,
            has_completed,
        ),
    }
    if not hasattr(traj.info, "termination_code"):
        return metrics

    first_terminated = jnp.argmax(terminated, axis=1)
    ep_codes = jnp.take_along_axis(
        traj.info.termination_code,
        first_terminated[:, None],
        axis=1,
    )[:, 0]

    def rate(code):
        return ((ep_codes == code) & has_terminated).astype(jnp.float32).mean()

    metrics.update(
        {
            "termination/q_min_disruption_rate": rate(1),
            "termination/greenwald_disruption_rate": rate(2),
            "termination/solver_failure_rate": rate(3),
        }
    )
    return metrics


def _physics_scalar_metrics(traj, episode_returns, episode_lengths, train_metrics):
    """All scalar metrics from a TORAX rollout: base eval + train metrics +
    final-valid and episode-mean physics."""
    env_state = unwrap_to_env_state(traj.env_state)
    plasma = env_state.plasma
    return {
        **_base_metrics(traj, episode_returns, episode_lengths, train_metrics),
        "obs/Q_fusion": _last_valid(plasma.Q_fusion, traj.valid).mean(),
        "obs/W_thermal_MJ": _last_valid(plasma.W_thermal_total, traj.valid).mean()
        * 1e-6,
        "obs/P_fusion_MW": _last_valid(plasma.P_fusion, traj.valid).mean() * 1e-6,
        "obs/tau_E_s": _last_valid(plasma.tau_E, traj.valid).mean(),
        "obs/H98": _last_valid(plasma.H98, traj.valid).mean(),
        "obs/beta_N": _last_valid(plasma.beta_N, traj.valid).mean(),
        "obs/q_min": _last_valid(plasma.q_min, traj.valid).mean(),
        "obs/q95": _last_valid(plasma.q95, traj.valid).mean(),
        "obs/f_non_inductive": _last_valid(plasma.f_non_inductive, traj.valid).mean(),
        "obs/fgw_n_e_line_avg": _last_valid(plasma.fgw_n_e_line_avg, traj.valid).mean(),
        "obs/fgw_n_e_volume_avg": _last_valid(
            plasma.fgw_n_e_volume_avg, traj.valid
        ).mean(),
        "obs/P_SOL_over_P_LH": _masked_mean(
            plasma.P_SOL_total / plasma.P_LH, traj.valid
        ),
        "ref/Q_fusion": _masked_mean(plasma.Q_fusion, traj.valid),
        "ref/W_thermal_MJ": _masked_mean(plasma.W_thermal_total, traj.valid) * 1e-6,
        "ref/P_fusion_GW": _masked_mean(plasma.P_fusion, traj.valid) * 1e-9,
    }


def _log_scalars(step, metrics, summary_keys=()) -> tuple[int, dict]:
    """Host-side: prints a one-line summary, logs the scalars to wandb, and
    returns ``(step, scalars)`` as Python values for further logging."""
    step_int = step.item()
    sm = {k: v.item() for k, v in metrics.items()}
    line = (
        f"step={step_int:>8d}"
        f"  return={sm['evaluation/return_mean']:.3f}"
        f" ± {sm['evaluation/return_std']:.3f}"
    )
    for key in summary_keys:
        line += f"  {key.rsplit('/', 1)[-1]}={sm[key]:.3f}"
    print(line, flush=True)
    wandb.log(sm, step=step_int)
    return step_int, sm


def _io_log_scalars(step, metrics, summary_keys=()):
    """Traced-side wrapper: routes ``_log_scalars`` through ``io_callback``."""

    def _host(s, m):
        _log_scalars(s, m, summary_keys)

    jax.experimental.io_callback(_host, (), step, metrics, ordered=True)


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------


def make_buffered_seed_callback(
    logger,
    *,
    num_steps: int,
    n_seeds: int = 4,
    kind: str = "physics",
    eval_rng: jax.Array | None = None,
    deterministic: bool = False,
    extra_metrics=None,
):
    """Scalars-only eval callback for vmapped multi-seed training.

    Routes metrics to a :class:`training.vmap_logging.SeedBufferLogger`
    via :func:`jax.debug.callback` so they can be buffered and averaged across
    training seeds (per-seed figures can't be averaged, so none are emitted).
    Returns a ``run_idx -> callback`` factory: the training script rebuilds the
    callback inside the trace so each seed's ``run_idx`` tracer flows into
    ``logger.log``.

    Args:
        logger: A :class:`SeedBufferLogger`.
        num_steps: Episode length (concrete int; controls the eval ``lax.scan``).
        n_seeds: Eval rollouts per checkpoint *within* one training seed.
        kind: ``"physics"`` for TORAX envs or ``"minimal"`` for world-model /
            returns-only envs.
        eval_rng: Optional fixed evaluation key. When set, it overrides the
            evolving callback key, including across vmapped training seeds.
        deterministic: Use ``algo.make_deterministic_act(ts)`` instead of the
            stochastic policy action function.
        extra_metrics: Optional ``(train_state, train_metrics) -> dict`` hook.
    """

    def for_run(run_idx):
        def _callback(algo, ts, rng, train_metrics):
            traj, episode_returns, episode_lengths = _collect_returns_and_lengths(
                algo,
                ts,
                eval_rng if eval_rng is not None else rng,
                num_steps,
                n_seeds,
                lean=(kind == "physics"),
                deterministic=deterministic,
            )
            if kind == "physics":
                metrics = _physics_scalar_metrics(
                    traj, episode_returns, episode_lengths, train_metrics
                )
            else:
                metrics = _base_metrics(
                    traj,
                    episode_returns,
                    episode_lengths,
                    train_metrics,
                )
            if extra_metrics is not None:
                metrics.update(extra_metrics(ts, train_metrics))
            jax.debug.callback(logger.log, ts.global_step, run_idx, metrics)
            return episode_returns, episode_lengths

        return _callback

    return for_run


def make_minimal_training_callback(
    num_steps: int,
    n_seeds: int = 4,
    *,
    eval_rng: jax.Array | None = None,
    deterministic: bool = False,
):
    """Env-agnostic eval callback: returns, episode length, PPO train metrics.

    Reads nothing TORAX-specific from the trajectory, so it also works for
    learned-dynamics envs whose state carries no ``plasma``.
    """

    def _callback(algo, ts, rng, train_metrics):
        traj, episode_returns, episode_lengths = _collect_returns_and_lengths(
            algo,
            ts,
            eval_rng if eval_rng is not None else rng,
            num_steps,
            n_seeds,
            deterministic=deterministic,
        )
        metrics = _base_metrics(
            traj,
            episode_returns,
            episode_lengths,
            train_metrics,
        )
        _io_log_scalars(
            ts.global_step, metrics, summary_keys=("evaluation/episode_length_mean",)
        )
        return episode_returns, episode_lengths

    return _callback


def make_world_model_training_callback(
    num_steps: int,
    n_seeds: int = 4,
    *,
    eval_rng: jax.Array | None = None,
    deterministic: bool = False,
):
    """KSTAR/``fusion_lstm`` eval callback: adds target-tracking scalars.

    The world-model env's flat observation already carries the physics of
    interest (``betap``/``q95``/``li`` and their per-episode targets), so it is
    read from the final valid ``traj.next_obs`` entry for each seed.
    """
    from plasmax.models.world_model_env import OBS_NAMES

    _idx = {name: i for i, name in enumerate(OBS_NAMES)}

    def _callback(algo, ts, rng, train_metrics):
        traj, episode_returns, episode_lengths = _collect_returns_and_lengths(
            algo,
            ts,
            eval_rng if eval_rng is not None else rng,
            num_steps,
            n_seeds,
            deterministic=deterministic,
        )
        last = _last_valid(traj.next_obs, traj.valid)

        def _tracking_error(name):
            return jnp.abs(last[:, _idx[name]] - last[:, _idx[f"{name}_target"]])

        metrics = {
            **_base_metrics(
                traj,
                episode_returns,
                episode_lengths,
                train_metrics,
            ),
            "obs/betap": last[:, _idx["betap"]].mean(),
            "obs/q95": last[:, _idx["q95"]].mean(),
            "obs/li": last[:, _idx["li"]].mean(),
            "tracking/betap_error": _tracking_error("betap").mean(),
            "tracking/q95_error": _tracking_error("q95").mean(),
            "tracking/li_error": _tracking_error("li").mean(),
        }
        _io_log_scalars(
            ts.global_step,
            metrics,
            summary_keys=(
                "tracking/betap_error",
                "tracking/q95_error",
                "tracking/li_error",
            ),
        )
        return episode_returns, episode_lengths

    return _callback


def make_training_callback(
    num_steps: int,
    n_seeds: int = 4,
    dt: float = 0.1,
    n_frames: int = 50,
    *,
    eval_rng: jax.Array | None = None,
    deterministic: bool = False,
):
    """Single-seed TORAX eval callback: physics scalars plus Plotly figures.

    Logs the :func:`_physics_scalar_metrics` scalars and three interactive
    figures per eval checkpoint: ``plots/actions`` and ``plots/physics``
    (mean ± std over the ``n_seeds`` rollouts vs time) and ``profiles/rho``
    (T_e/T_i/n_e/q vs ρ overlaid at successive episode times, seed 0).

    Args:
        num_steps: Episode length; a concrete Python int (controls the inner
            ``lax.scan`` length).
        n_seeds: Independent eval rollouts per checkpoint.
        dt: Outer RL step size in seconds; must match the env's
            ``numerics.fixed_dt`` (renders the figures' time axis).
        n_frames: Evenly-spaced time frames in the ``profiles/rho`` figure.
        eval_rng: Optional fixed evaluation key. When set, it overrides the
            evolving callback key.
        deterministic: Use ``algo.make_deterministic_act(ts)`` instead of the
            stochastic policy action function.

    Returns:
        Callable returning ``(returns, lengths)`` so that the
        Upstream Rejax records the returned episode lengths and returns.
    """

    def _callback(algo, ts, rng, train_metrics):
        traj, episode_returns, episode_lengths = _collect_returns_and_lengths(
            algo,
            ts,
            eval_rng if eval_rng is not None else rng,
            num_steps,
            n_seeds,
            lean=True,
            deterministic=deterministic,
        )
        env_state = unwrap_to_env_state(traj.env_state)
        plasma = env_state.plasma
        metrics = _physics_scalar_metrics(
            traj, episode_returns, episode_lengths, train_metrics
        )

        # Actuator labels/scales for the actions figure, in action-vector order.
        env = getattr(algo.env, "envelope_env", algo.env)
        actuator_names = [s.name for s in env.unwrapped.actuator_specs]
        act_scales = [_ACTUATOR_UNITS.get(n, (1.0, ""))[0] for n in actuator_names]
        act_ylabels = [_ACTUATOR_UNITS.get(n, (1.0, ""))[1] for n in actuator_names]

        # --- Fixed-shape trajectory data for plots; invalid tails become NaN. ---
        t_s = jnp.arange(num_steps) * dt
        actions_mean, actions_std = _masked_band(traj.action, traj.valid)

        # Physics scalars stacked: (n_seeds, num_steps, len(_PHYSICS_LABELS))
        ppo_stack = jnp.stack(
            [
                plasma.W_thermal_total,
                plasma.P_fusion,
                plasma.P_aux_total,
                plasma.Q_fusion,
                plasma.tau_E,
                plasma.H98,
                plasma.q_min,
                plasma.q95,
                plasma.beta_N,
                plasma.f_non_inductive,
            ],
            axis=-1,
        )
        ppo_mean, ppo_std = _masked_band(ppo_stack, traj.valid)

        # --- Profile-vs-rho data. Host callback trims/subsamples valid frames. ---
        cp = env_state.plasma.core
        q_face = cp.q_face[0]  # (num_steps, n_rho+1)
        q_cell = (q_face[:, :-1] + q_face[:, 1:]) / 2.0
        # Stack profiles in PROFILE_LABELS order: (num_steps, n_rho, n_profiles)
        profile_stack = jnp.stack(
            [
                cp.T_e.value[0],
                cp.T_i.value[0],
                cp.n_e.value[0],
                q_cell,
            ],
            axis=-1,
        )
        # Geometry is stripped from the lean eval trajectory; rho_norm is static,
        # so re-source it from one fresh init (single, unbatched -> (n_rho,)).
        init_state, _ = env.init(jax.random.key(0))
        rho = unwrap_to_env_state(init_state).plasma.geo.rho_norm

        def _log(
            step,
            metrics,
            t_s,
            actions_mean,
            actions_std,
            ppo_mean,
            ppo_std,
            profile_stack,
            profile_valid,
            rho,
        ):
            step_int, _ = _log_scalars(step, metrics, summary_keys=("obs/Q_fusion",))
            fig_act = _make_band_fig(
                t_s,
                actions_mean,
                actions_std,
                actuator_names,
                act_ylabels,
                act_scales,
                "Actions",
                0.05,
            )
            fig_phy = _make_band_fig(
                t_s,
                ppo_mean,
                ppo_std,
                _PHYSICS_LABELS,
                _PHYSICS_YLABELS,
                _PHYSICS_SCALES,
                "Physics",
                0.04,
            )
            # profiles/rho: dynamically trim and subsample seed 0 on the host.
            valid_idx = np.flatnonzero(np.asarray(profile_valid, dtype=bool))
            n_anim = min(n_frames, valid_idx.size)
            positions = np.unique(
                np.round(np.linspace(0, valid_idx.size - 1, n_anim)).astype(int)
            )
            frame_idx = valid_idx[positions]
            stack = np.asarray(profile_stack)[frame_idx]
            frame_times = np.asarray(t_s)[frame_idx]
            profiles = {name: stack[:, :, i] for i, name in enumerate(PROFILE_LABELS)}
            fig_prof = make_profile_rho_figure(np.asarray(rho), profiles, frame_times)
            wandb.log(
                {
                    "plots/actions": wandb.Plotly(fig_act),
                    "plots/physics": wandb.Plotly(fig_phy),
                    "profiles/rho": wandb.Plotly(fig_prof),
                },
                step=step_int,
            )

        jax.experimental.io_callback(
            _log,
            (),
            ts.global_step,
            metrics,
            t_s,
            actions_mean,
            actions_std,
            ppo_mean,
            ppo_std,
            profile_stack,
            traj.valid[0],
            rho,
        )
        return episode_returns, episode_lengths

    return _callback
