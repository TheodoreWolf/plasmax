""":func:`make` constructs validated plasmax environments from configuration."""

from __future__ import annotations

import functools
import math
import numbers
import operator
from typing import Literal

from envelope import Environment

from plasmax import rewards as rewards_lib
from plasmax import wrappers as wrappers_lib
from plasmax.environment import core as env_lib
from plasmax.environment import initialization as initialization_lib
from plasmax.environment.config import parse_env_and_backend
from plasmax.environment.merge import validate_env_backend
from plasmax.environment.schema import (
    PhaseInitializationConfig,
    PlasmaxConfig,
    WorldModelConfig,
)
from plasmax.models.world_model import load_bundle


def _integral_step_count(value: object, *, name: str) -> int:
    """Return an exact integer step count, rejecting bools and coercions."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    try:
        return operator.index(value)
    except TypeError as error:
        raise ValueError(f"{name} must be an integer, got {value!r}") from error


def _resolve_max_steps(requested: int | None, safe_max_steps: int) -> int:
    """Validate a caller horizon against the backend's configured safe limit."""
    safe_max_steps = _integral_step_count(
        safe_max_steps, name="configured safe horizon"
    )
    if safe_max_steps < 1:
        raise ValueError(
            f"configured safe horizon must be positive, got {safe_max_steps}"
        )
    max_steps = (
        safe_max_steps
        if requested is None
        else _integral_step_count(requested, name="max_steps")
    )
    if max_steps < 1:
        raise ValueError(f"max_steps must be positive, got {max_steps}")
    if max_steps > safe_max_steps:
        raise ValueError(
            "max_steps cannot exceed the backend's configured safe horizon: "
            f"requested {max_steps}, at most {safe_max_steps}"
        )
    return max_steps


def _validate_options(
    env: str,
    backend: str | None,
    reward: str | rewards_lib.RewardFn | None,
    disruption_penalty: float | None,
    variant: str,
    max_steps: int | None,
    time_aware: bool,
    quantize_bins: int | None,
) -> tuple[int | None, int | None]:
    """Reject public-input errors before loading assets or constructing TORAX."""
    validate_env_backend(env, backend)
    if variant not in ("oracle", "realistic"):
        raise ValueError(
            f"Unknown variant {variant!r}; expected 'oracle' or 'realistic'"
        )
    if not isinstance(time_aware, bool):
        raise ValueError(f"time_aware must be a boolean, got {time_aware!r}")
    checked_max_steps = (
        None if max_steps is None else _integral_step_count(max_steps, name="max_steps")
    )
    if checked_max_steps is not None and checked_max_steps < 1:
        raise ValueError(f"max_steps must be positive, got {checked_max_steps}")
    checked_bins = (
        None
        if quantize_bins is None
        else _integral_step_count(quantize_bins, name="quantize_bins")
    )
    if checked_bins is not None and checked_bins < 2:
        raise ValueError(f"quantize_bins must be at least 2, got {checked_bins}")
    if disruption_penalty is not None and (
        isinstance(disruption_penalty, bool)
        or not isinstance(disruption_penalty, numbers.Real)
        or not math.isfinite(float(disruption_penalty))
    ):
        raise ValueError("disruption_penalty must be a finite number or None")

    if env == "kstar_worldmodel":
        if variant != "realistic":
            raise ValueError("kstar_worldmodel only supports the realistic variant")
        if reward is not None:
            raise ValueError("world-model environments use their native reward")
        if disruption_penalty is not None:
            raise ValueError(
                "world-model environments do not support disruption_penalty"
            )
        if quantize_bins is not None:
            raise ValueError("world-model environments do not support quantize_bins")
    else:
        if quantize_bins is not None and variant == "oracle":
            raise ValueError("quantize_bins is a realistic action degradation")
        if reward is not None:
            rewards_lib.resolve_reward_fn(reward)
    return checked_max_steps, checked_bins


def _load_world_model_env(
    cfg: WorldModelConfig,
    *,
    max_steps: int | None,
    time_aware: bool,
) -> Environment:
    """Build a scalar Envelope environment around learned KSTAR dynamics."""
    from plasmax.models import world_model_env as wm_env_lib

    spec = cfg.world_model
    safe_max_steps = spec.max_steps_in_episode
    episode_max_steps = _resolve_max_steps(max_steps, safe_max_steps)
    env: Environment = wm_env_lib.WorldModelEnv.from_bundle(
        load_bundle(spec.weights_path),
        random_target=spec.random_target,
    )
    if time_aware:
        env = wrappers_lib.TimeAwareWrapper(env=env)
    return wrappers_lib.PlasmaxTruncationWrapper(env=env, max_steps=episode_max_steps)


def _resolve_task_settings(
    cfg: PlasmaxConfig,
    reward: str | rewards_lib.RewardFn | None,
    disruption_penalty: float | None,
) -> tuple[str | rewards_lib.RewardFn, float]:
    """Resolve omitted task settings while preserving explicit zero values."""
    resolved_reward = cfg.task.reward if reward is None else reward
    resolved_penalty = (
        cfg.task.terminal_penalty if disruption_penalty is None else disruption_penalty
    )
    if resolved_penalty is None:
        raise ValueError("TORAX tasks require a numeric task.terminal_penalty")
    return resolved_reward, resolved_penalty


def _load_phase_snapshot(
    spec: PhaseInitializationConfig | None,
    *,
    expected_environment: str,
) -> initialization_lib.PhaseSnapshot | None:
    """Resolve and validate one phase-owned NPZ snapshot."""
    if spec is None:
        return None
    return initialization_lib.load_snapshot(
        spec.path,
        expected_sha256=spec.sha256,
        expected_environment=expected_environment,
    )


def _build_env(
    cfg: PlasmaxConfig,
    *,
    reward: str | rewards_lib.RewardFn | None,
    disruption_penalty: float | None,
    variant: Literal["oracle", "realistic"],
    max_steps: int | None,
    time_aware: bool,
    quantize_bins: int | None,
) -> Environment:
    """Assembles the wrapper stack from a validated config and RL settings.

    The realistic variant samples ``cfg.physics_randomization`` independently
    for every transition. ``time_aware`` appends the environment time
    coordinate via ``TimeAwareWrapper``, applied in both variants.
    """
    resolved_reward, resolved_penalty = _resolve_task_settings(
        cfg, reward, disruption_penalty
    )
    reward_fn = rewards_lib.resolve_reward_fn(resolved_reward)
    if resolved_reward == "lh_transition":
        # Ramp-up lengths differ substantially (e.g. 10 s SPARC, 60 s ITER
        # baseline, 100 s ITER hybrid/advanced). Bind the reward's elapsed-time
        # normalization to the loaded scenario instead of its historical 100 s
        # default, which would silently define different tasks.
        reward_fn = functools.partial(
            rewards_lib.lh_transition,
            t_final=float(cfg.torax.numerics.t_final),
        )
    phase_snapshot = _load_phase_snapshot(
        cfg.initialization,
        expected_environment=cfg.environment_key,
    )

    profile_obs_specs = [item.to_spec() for item in cfg.observations.profiles]
    scalar_obs_specs = [item.to_spec() for item in cfg.observations.scalars]
    actuator_specs = [item.to_spec() for item in cfg.actuators]
    realistic = variant == "realistic"
    base_env = env_lib.PlasmaxEnv.from_config(
        cfg.torax,
        actuator_specs,
        reward_fn,
        disruption_penalty=float(resolved_penalty),
        clip_by_max_action_delta=cfg.clip_by_max_action_delta,
        disruption=cfg.disruption,
        state_noise_config=cfg.state_noise,
        physics_randomization=(cfg.physics_randomization if realistic else {}),
        stepping=cfg.stepping,
        profile_obs_specs=profile_obs_specs,
        scalar_obs_specs=scalar_obs_specs,
        _initialization=phase_snapshot,
    )

    # Realistic: degrading sensor effects, applied to the obs from the base env:
    # noise → profile resolution → obs filter → delay.
    real = cfg.observations.realistic
    env: Environment = base_env
    if realistic and real.noise:
        noise_cfg = wrappers_lib.SensorNoiseConfig(relative_std=real.noise)
        noise_scale = noise_cfg.to_noise_scale(base_env.obs_layout())
        env = wrappers_lib.NoiseWrapper(env=env, noise_scale=noise_scale)
    if realistic and real.resolution:
        res_cfg = wrappers_lib.ProfileResolutionConfig(n_obs=real.resolution)
        env = wrappers_lib.ObsFilterWrapper.from_resolution_config(env, res_cfg)
    if realistic and real.filter is not None:
        obs_filter = real.filter
        env = wrappers_lib.ObsFilterWrapper.from_obs_config(
            env,
            wrappers_lib.ObsFilterConfig(
                profiles=list(obs_filter.profiles),
                scalars=list(obs_filter.scalars),
            ),
        )
    if realistic and real.delay:
        # Build hold probabilities against the (possibly filtered/downsampled)
        # layout the delay wrapper actually sees.
        delay_cfg = wrappers_lib.ObsDelayConfig(repeat_prob=real.delay)
        hold_prob = delay_cfg.to_hold_prob(env.obs_layout())
        env = wrappers_lib.ObsDelayWrapper(env=env, hold_prob=hold_prob)

    # ActionRescale (action-only) sits below the history stack so the
    # frame-stacked action history records the policy's [-1, 1] actions.
    # TimeAware and ObsHistory are both advantageous, so they are applied
    # in both variants; TimeAware sits below ObsHistory so the stacked
    # frames each carry their own elapsed-time value. Quantize (if
    # configured) sits above both. The TORAX truncation wrapper is always
    # outermost; loaders never add autoreset or vectorization.
    env = wrappers_lib.ActionRescaleWrapper(env=env)
    if time_aware:
        env = wrappers_lib.TimeAwareWrapper(env=env)
    if cfg.observations.history is not None:
        env = wrappers_lib.ObsHistoryWrapper(env=env, k=cfg.observations.history.length)

    configured_bins = (
        cfg.actions.realistic.quantize if realistic and quantize_bins is None else {}
    )
    if quantize_bins is not None:
        bin_counts = (quantize_bins,) * len(actuator_specs)
    elif configured_bins:
        quantize_cfg = wrappers_lib.ActionQuantizeConfig(bins=configured_bins)
        bin_counts = quantize_cfg.to_bin_counts(
            [actuator.name for actuator in actuator_specs]
        )
    else:
        bin_counts = None
    if bin_counts is not None:
        env = wrappers_lib.QuantizeActionWrapper(env=env, bin_counts=bin_counts)

    episode_max_steps = _resolve_max_steps(max_steps, env.unwrapped.safe_max_steps)
    return wrappers_lib.PlasmaxTruncationWrapper(env=env, max_steps=episode_max_steps)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def make(
    env: str,
    backend: str | None = None,
    *,
    reward: str | rewards_lib.RewardFn | None = None,
    variant: Literal["oracle", "realistic"] = "realistic",
    disruption_penalty: float | None = None,
    max_steps: int | None = None,
    time_aware: bool = False,
    quantize_bins: int | None = None,
) -> Environment:
    """Build an environment from registry aliases.

    Omit ``backend`` for the standalone ``"kstar_worldmodel"`` environment.
    The realistic variant may override all actuator bin counts with
    ``quantize_bins``.
    """
    max_steps, quantize_bins = _validate_options(
        env,
        backend,
        reward,
        disruption_penalty,
        variant,
        max_steps,
        time_aware,
        quantize_bins,
    )
    cfg = parse_env_and_backend(env, backend)
    if isinstance(cfg, WorldModelConfig):
        return _load_world_model_env(
            cfg,
            max_steps=max_steps,
            time_aware=time_aware,
        )
    return _build_env(
        cfg,
        reward=reward,
        disruption_penalty=disruption_penalty,
        variant=variant,
        max_steps=max_steps,
        time_aware=time_aware,
        quantize_bins=quantize_bins,
    )


__all__ = ["make"]
