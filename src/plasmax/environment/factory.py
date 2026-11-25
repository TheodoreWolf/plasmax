""":func:`make` constructs validated plasmax environments from configuration."""

from __future__ import annotations

import functools

from envelope import Environment

from plasmax import rewards as rewards_lib
from plasmax.environment import core as env_lib
from plasmax.environment import initialization as initialization_lib
from plasmax.environment.config import parse_env_and_backend
from plasmax.environment.merge import validate_env_backend
from plasmax.environment.schema import (
    PlasmaxConfig,
    WorldModelConfig,
)
from plasmax.models.world_model import load_bundle


def _validate_options(
    env: str,
    backend: str | None,
    reward: str | rewards_lib.RewardFn | None,
) -> None:
    """Reject public-input errors before loading assets or constructing TORAX."""
    validate_env_backend(env, backend)

    if env == "kstar_worldmodel":
        if reward is not None:
            raise ValueError("world-model environments use their native reward")
    else:
        if reward is not None:
            rewards_lib.resolve_reward_fn(reward)


def _load_world_model_env(cfg: WorldModelConfig) -> Environment:
    """Build a bare Envelope environment around learned KSTAR dynamics."""
    from plasmax.models import world_model_env as wm_env_lib

    spec = cfg.world_model
    actuator_specs = [item.to_spec() for item in cfg.actuators]
    return wm_env_lib.WorldModelEnv.from_bundle(
        load_bundle(spec.weights_path),
        random_target=spec.random_target,
        actuator_specs=actuator_specs,
        physics_randomization=cfg.physics_randomization,
        clip_by_max_action_delta=cfg.clip_by_max_action_delta,
        _plasmax_config=cfg,
        _initialization=cfg._initial_state,
    )


def _build_env(
    cfg: PlasmaxConfig,
    *,
    reward: str | rewards_lib.RewardFn | None,
) -> Environment:
    """Build the bare environment from a validated configuration."""
    resolved_reward = cfg.task.reward if reward is None else reward
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
    phase_snapshot = initialization_lib.snapshot_from_initialization(cfg._initial_state)

    profile_obs_specs = [item.to_spec() for item in cfg.observations.profiles]
    scalar_obs_specs = [item.to_spec() for item in cfg.observations.scalars]
    actuator_specs = [item.to_spec() for item in cfg.actuators]
    return env_lib.PlasmaxEnv.from_config(
        cfg.torax,
        actuator_specs,
        reward_fn,
        clip_by_max_action_delta=cfg.clip_by_max_action_delta,
        disruption=cfg.disruption,
        state_noise_config=cfg.state_noise,
        physics_randomization=cfg.physics_randomization,
        stepping=cfg.stepping,
        profile_obs_specs=profile_obs_specs,
        scalar_obs_specs=scalar_obs_specs,
        _initialization=phase_snapshot,
        _plasmax_config=cfg,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def make(
    env: str,
    backend: str | None = None,
    *,
    reward: str | rewards_lib.RewardFn | None = None,
) -> Environment:
    """Build a bare environment from registry aliases.

    Omit ``backend`` for ``"kstar_worldmodel"``. Apply wrappers explicitly,
    for example ``RealisticWrappers(make(env, backend), max_steps=100)``.
    A custom reward receives ``(state, action, next_state, termination_code)``
    and returns the final scalar reward, which is cast to float32.
    """
    _validate_options(env, backend, reward)
    cfg = parse_env_and_backend(env, backend)
    if isinstance(cfg, WorldModelConfig):
        return _load_world_model_env(cfg)
    return _build_env(cfg, reward=reward)


__all__ = ["make"]
