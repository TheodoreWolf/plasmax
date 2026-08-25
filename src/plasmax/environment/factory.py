""":func:`make` constructs validated plasmax environments from configuration.

Single-file scenarios take no backend; device scenarios take an env + backend
pair, merging backend defaults with env overrides. Both accept registry aliases
or paths.
"""

from __future__ import annotations

import functools
import operator
from typing import Any, Literal

from envelope import Environment
from torax._src.torax_pydantic import model_config

from plasmax import rewards as rewards_lib
from plasmax import spaces as spaces_lib
from plasmax import wrappers as wrappers_lib
from plasmax.environment import core as env_lib
from plasmax.environment import initialization as initialization_lib
from plasmax.environment import registry as registry_lib
from plasmax.environment.config import (
    _parse_world_model_sources,
    backend_kind,
    parse_env_and_backend,
    parse_scenario,
    validate_env_backend,
)
from plasmax.environment.merge import (
    _apply_imas_init,
    _load_extended_yaml,
    _load_phase_initialization,
    _resolve_config_asset,
    _resolve_geometry_dir,
    env_key,
)
from plasmax.environment.schema import (
    ScenarioConfig,
)

__all__ = [
    "make",
]


def _validate_realistic_obs_names(cfg: ScenarioConfig) -> None:
    """Raises ValueError if filter/delay names reference unknown sensors.

    Filter names must be declared profiles/scalars; delay names must be among
    the sensors that survive the filter (the obs the delay wrapper sees).
    """
    profile_names = {p.name for p in cfg.observations.profiles}
    scalar_names = {s.name for s in cfg.observations.scalars}
    f = cfg.observations.realistic.filter
    if f is not None:
        for name in f.profiles or []:
            if name not in profile_names:
                raise ValueError(
                    f"filter profile {name!r} not in observations.profiles: "
                    f"{sorted(profile_names)}"
                )
        for name in f.scalars or []:
            if name not in scalar_names:
                raise ValueError(
                    f"filter scalar {name!r} not in observations.scalars: "
                    f"{sorted(scalar_names)}"
                )
        surviving = set(f.profiles or []) | set(f.scalars or [])
    else:
        surviving = profile_names | scalar_names

    for name in cfg.observations.realistic.delay:
        if name not in surviving:
            qualifier = "filtered " if f is not None else ""
            raise ValueError(
                f"delay sensor {name!r} not in {qualifier}observations: "
                f"{sorted(surviving)}"
            )

    for name in cfg.observations.realistic.resolution:
        if name not in profile_names:
            raise ValueError(
                f"resolution profile {name!r} not in observations.profiles: "
                f"{sorted(profile_names)}"
            )


def _validate_quantize_names(cfg: ScenarioConfig) -> None:
    """Raises ValueError unless quantize is empty or covers every actuator.

    The array-valued Envelope Discrete space needs one categorical dimension
    per actuator, so quantization can't be applied to a subset — it's all
    actuators or none.
    """
    quantize = cfg.actions.realistic.quantize
    if not quantize:
        return
    actuator_names = {a.name for a in cfg.actuators}
    quantize_names = set(quantize)
    if quantize_names != actuator_names:
        missing = actuator_names - quantize_names
        extra = quantize_names - actuator_names
        raise ValueError(
            "actions.realistic.quantize must cover every actuator or none; "
            f"missing: {sorted(missing)}, unknown: {sorted(extra)}"
        )


def _with_quantized_actions(
    cfg: ScenarioConfig,
    variant: Literal["oracle", "realistic"],
    quantize_bins: int | None,
) -> ScenarioConfig:
    if quantize_bins is None:
        return cfg
    if variant == "oracle":
        raise ValueError("quantize_bins requires variant='realistic'")
    realistic = type(cfg.actions.realistic)(
        quantize={a.name: quantize_bins for a in cfg.actuators}
    )
    return cfg.model_copy(
        update={"actions": cfg.actions.model_copy(update={"realistic": realistic})}
    )


def _load_world_model_env(
    env_cfg: dict[str, Any],
    backend_cfg: dict[str, Any],
    *,
    max_steps: int | None,
    time_aware: bool,
) -> Environment:
    """Build a scalar Envelope environment around learned KSTAR dynamics."""
    from plasmax.models import world_model_env as wm_env_lib

    name = backend_cfg.get("name")
    if name != "kstar_lstm":
        raise ValueError(
            f"unknown world_model backend {name!r}; only 'kstar_lstm' is supported"
        )
    if "max_steps_in_episode" not in env_cfg:
        raise ValueError(
            "world_model_env.max_steps_in_episode is required to define the "
            "KSTAR simulator-safe horizon"
        )
    safe_max_steps = env_cfg["max_steps_in_episode"]
    episode_max_steps = _resolve_max_steps(max_steps, safe_max_steps)
    env: Environment = wm_env_lib.WorldModelEnv(
        random_target=env_cfg.get("random_target", True)
    )
    if time_aware:
        env = wrappers_lib.TimeAwareWrapper(env=env)
    return wrappers_lib.PlasmaxTruncationWrapper(env=env, max_steps=episode_max_steps)


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


# Degrading realistic wrappers that can be individually ablated (leave-one-out
# from the realistic stack). Maps the ablation name to the RealisticObsConfig
# field reset that disables it.
_ABLATABLE: dict[str, Any] = {
    "noise": {"noise": {}},
    "resolution": {"resolution": {}},
    "filter": {"filter": None},
    "delay": {"delay": {}},
}


def _resolve_task_settings(
    cfg: ScenarioConfig,
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
    env_path: str,
    spec: dict[str, Any] | None,
) -> initialization_lib.PhaseSnapshot | None:
    """Resolve and validate one phase-owned NPZ snapshot."""
    if spec is None:
        return None
    expected_keys = {"path", "sha256"}
    if set(spec) != expected_keys:
        missing = sorted(expected_keys - set(spec))
        extra = sorted(set(spec) - expected_keys)
        raise ValueError(
            "initialization metadata must contain exactly path and sha256; "
            f"missing={missing}, extra={extra}"
        )
    path = spec["path"]
    sha256 = spec["sha256"]
    if not isinstance(path, str) or not path:
        raise ValueError("initialization.path must be a non-empty string")
    if not isinstance(sha256, str):
        raise ValueError("initialization.sha256 must be a SHA-256 string")
    return initialization_lib.load_snapshot(
        _resolve_config_asset(path, env_path),
        expected_sha256=sha256,
        expected_environment=env_key(env_path),
    )


def _build_env(
    cfg: ScenarioConfig,
    torax_config: model_config.ToraxConfig,
    reward: str | rewards_lib.RewardFn,
    variant: Literal["oracle", "realistic"],
    max_steps: int | None,
    disruption_penalty: float,
    ablate: str | None = None,
    time_aware: bool = False,
    phase_snapshot: initialization_lib.PhaseSnapshot | None = None,
) -> Environment:
    """Assembles the wrapper stack from a validated ScenarioConfig and RL settings.

    ``ablate`` drops a single degrading wrapper from the realistic stack
    (leave-one-out: one of ``noise``, ``resolution``, ``filter``, ``delay``);
    ``None``/``"none"`` keeps the full stack. It only affects the realistic
    variant (oracle has no degrading wrappers). Realistic also samples
    ``cfg.physics_randomization`` independently for every transition.

    ``time_aware`` appends the environment time coordinate via
    ``TimeAwareWrapper``, applied in all variants.
    """
    if variant not in ("oracle", "realistic"):
        raise ValueError(
            f"unknown variant {variant!r}; expected 'oracle' or 'realistic'"
        )
    if ablate not in (None, "none") and ablate not in _ABLATABLE:
        raise ValueError(
            f"unknown ablate target {ablate!r}; valid: {sorted(_ABLATABLE)} (or 'none')"
        )
    if variant == "oracle" and ablate not in (None, "none"):
        raise ValueError("ablate only applies to the realistic variant")
    actuator_specs = [
        spaces_lib.ActuatorSpec(
            a.name,
            low=a.low,
            high=a.high,
            max_delta=a.max_delta,
            init=a.init,
        )
        for a in cfg.actuators
    ]
    profile_obs_specs = tuple(
        spaces_lib.ObsSpec(p.name, scale=p.scale, bounds=p.bounds)
        for p in cfg.observations.profiles
    )
    scalar_obs_specs = tuple(
        spaces_lib.ObsSpec(s.name, scale=s.scale, bounds=s.bounds)
        for s in cfg.observations.scalars
    )
    if reward == "lh_transition":
        # Ramp-up lengths differ substantially (e.g. 10 s SPARC, 60 s ITER
        # baseline, 100 s ITER hybrid/advanced). Bind the reward's elapsed-time
        # normalization to the loaded scenario instead of its historical 100 s
        # default, which would silently define different tasks.
        t_final = float(cfg.torax["numerics"]["t_final"])
        reward_fn = functools.partial(rewards_lib.lh_transition, t_final=t_final)
    else:
        reward_fn = rewards_lib.resolve_reward_fn(reward)

    base_env = env_lib.PlasmaxEnv._from_config(
        config=torax_config,
        actuator_specs=actuator_specs,
        reward_fn=reward_fn,
        disruption_penalty=disruption_penalty,
        clip_by_max_action_delta=cfg.clip_by_max_action_delta,
        disruption=cfg.disruption,
        state_noise_config=cfg.state_noise,
        physics_randomization=(
            cfg.physics_randomization if variant == "realistic" else {}
        ),
        stepping=cfg.stepping,
        initialization=phase_snapshot,
        profile_obs_specs=profile_obs_specs,
        scalar_obs_specs=scalar_obs_specs,
    )
    episode_max_steps = _resolve_max_steps(max_steps, base_env.safe_max_steps)

    history = cfg.observations.history

    def _finalize(
        env: Environment,
        quantize_bins: tuple[int, ...] | None = None,
    ) -> Environment:
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
        if history is not None:
            env = wrappers_lib.ObsHistoryWrapper(env=env, k=history.length)
        if quantize_bins is not None:
            env = wrappers_lib.QuantizeActionWrapper(env=env, bin_counts=quantize_bins)
        return wrappers_lib.PlasmaxTruncationWrapper(
            env=env, max_steps=episode_max_steps
        )

    if variant == "oracle":
        # Oracle: only the advantageous wrappers (history), no degrading effects.
        return _finalize(base_env)

    # Realistic: degrading sensor effects, applied to the obs from the base env:
    # noise → profile resolution → obs filter → delay.
    real = cfg.observations.realistic
    if ablate in _ABLATABLE:
        real = real.model_copy(update=_ABLATABLE[ablate])
    env: Environment = base_env
    if real.noise:
        noise_cfg = wrappers_lib.SensorNoiseConfig(relative_std=real.noise)
        noise_scale = noise_cfg.to_noise_scale(base_env.obs_layout())
        env = wrappers_lib.NoiseWrapper(env=env, noise_scale=noise_scale)

    if real.resolution:
        res_cfg = wrappers_lib.ProfileResolutionConfig(n_obs=real.resolution)
        env = wrappers_lib.ObsFilterWrapper.from_resolution_config(env, res_cfg)

    if real.filter is not None:
        obs_filter = wrappers_lib.ObsFilterConfig(
            profiles=real.filter.profiles or [],
            scalars=real.filter.scalars or [],
        )
        env = wrappers_lib.ObsFilterWrapper.from_obs_config(env, obs_filter)

    if real.delay:
        # Build hold probabilities against the (possibly filtered/downsampled)
        # layout the delay wrapper actually sees.
        delay_cfg = wrappers_lib.ObsDelayConfig(repeat_prob=real.delay)
        hold_prob = delay_cfg.to_hold_prob(env.obs_layout())
        env = wrappers_lib.ObsDelayWrapper(env=env, hold_prob=hold_prob)

    quantize_bins = None
    action_realistic = cfg.actions.realistic
    if action_realistic.quantize:
        quantize_cfg = wrappers_lib.ActionQuantizeConfig(bins=action_realistic.quantize)
        quantize_bins = quantize_cfg.to_bin_counts([a.name for a in cfg.actuators])

    return _finalize(env, quantize_bins)


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
    validate: bool = True,
    ablate: str | None = None,
    time_aware: bool = False,
    quantize_bins: int | None = None,
) -> Environment:
    """Build an environment from registry aliases or YAML paths.

    Omit ``backend`` for single-file scenarios such as ``"test"``; ``validate``
    checks env/backend compatibility and is therefore ignored when no backend
    is given. The realistic variant may override all actuator bin counts with
    ``quantize_bins``.
    """
    if backend is None:
        return _load_scenario(
            env,
            reward=reward,
            variant=variant,
            disruption_penalty=disruption_penalty,
            max_steps=max_steps,
            ablate=ablate,
            time_aware=time_aware,
            quantize_bins=quantize_bins,
        )
    return _load_env(
        env,
        backend,
        reward=reward,
        variant=variant,
        disruption_penalty=disruption_penalty,
        max_steps=max_steps,
        validate=validate,
        ablate=ablate,
        time_aware=time_aware,
        quantize_bins=quantize_bins,
    )


def _load_scenario(
    path: str,
    *,
    reward: str | rewards_lib.RewardFn | None = None,
    variant: Literal["oracle", "realistic"] = "realistic",
    disruption_penalty: float | None = None,
    max_steps: int | None = None,
    ablate: str | None = None,
    time_aware: bool = False,
    quantize_bins: int | None = None,
) -> Environment:
    """Loads a single-file scenario YAML (path or ``registry.SCENARIO_ALIASES``
    alias) into a scalar Envelope environment. See :func:`make` for the
    shared keyword arguments."""
    path = registry_lib.resolve_scenario(path)
    if _load_phase_initialization(path) is not None:
        raise ValueError(
            "phase initialization requires an environment with a TORAX backend"
        )
    cfg = parse_scenario(path)
    cfg = _with_quantized_actions(cfg, variant, quantize_bins)
    _validate_realistic_obs_names(cfg)
    _validate_quantize_names(cfg)
    torax_dict = _apply_imas_init(dict(cfg.torax), path)
    torax_dict = _resolve_geometry_dir(torax_dict, path)
    torax_config = model_config.ToraxConfig.from_dict(torax_dict)
    resolved_reward, resolved_penalty = _resolve_task_settings(
        cfg, reward, disruption_penalty
    )
    return _build_env(
        cfg,
        torax_config,
        resolved_reward,
        variant,
        max_steps,
        resolved_penalty,
        ablate,
        time_aware,
    )


def _load_env(
    env_path: str,
    backend_path: str,
    *,
    reward: str | rewards_lib.RewardFn | None = None,
    variant: Literal["oracle", "realistic"] = "realistic",
    disruption_penalty: float | None = None,
    max_steps: int | None = None,
    validate: bool = True,
    ablate: str | None = None,
    time_aware: bool = False,
    quantize_bins: int | None = None,
) -> Environment:
    """Loads an env + backend YAML pair (paths or registry aliases) into a
    scalar Envelope environment.

    The env YAML defines the RL task (actuators, observations, scenario
    plumbing); the backend YAML the simulator engine (transport, solver).
    They are deep-merged with backend supplying defaults and env winning on
    overlap — see the module docstring for the merge contract. For a
    world-model backend the corresponding learned-dynamics env is returned
    instead. See :func:`make` for the shared keyword arguments."""
    env_path = registry_lib.resolve_env(env_path)
    backend_path = registry_lib.resolve_backend(backend_path)
    phase_initialization = _load_phase_initialization(env_path)
    if variant not in ("oracle", "realistic"):
        raise ValueError(
            f"unknown variant {variant!r}; expected 'oracle' or 'realistic'"
        )
    if backend_kind(backend_path) == "world_model":
        if phase_initialization is not None:
            raise ValueError("world-model environments do not support initialization")
        if _load_extended_yaml(backend_path).get("initialization") is not None:
            raise ValueError("initialization is phase-only metadata")
        if validate:
            validate_env_backend(env_path, backend_path)
        sources = _parse_world_model_sources(env_path, backend_path)
        resolved_reward = sources.task.reward if reward is None else reward
        resolved_penalty = (
            sources.task.terminal_penalty
            if disruption_penalty is None
            else disruption_penalty
        )
        if resolved_reward != "native":
            raise ValueError("world-model environments use their native reward")
        if quantize_bins is not None:
            raise ValueError("world-model environments do not support quantize_bins")
        # Learned experimental world models have one intrinsic observation and
        # action interface. Both variants select it; unlike TORAX environments,
        # ``realistic`` adds no degradation wrappers.
        if resolved_penalty is not None:
            raise ValueError(
                "world-model environments do not support disruption_penalty"
            )
        if ablate not in (None, "none"):
            raise ValueError("world-model environments do not support ablations")
        return _load_world_model_env(
            sources.environment,
            sources.backend,
            max_steps=max_steps,
            time_aware=time_aware,
        )
    cfg = parse_env_and_backend(env_path, backend_path, validate=validate)
    cfg = _with_quantized_actions(cfg, variant, quantize_bins)
    _validate_realistic_obs_names(cfg)
    _validate_quantize_names(cfg)
    # Geometry files in the env YAML are resolved relative to the env file,
    # not the backend (backends shouldn't reference geometry files).
    torax_dict = _apply_imas_init(dict(cfg.torax), env_path)
    torax_dict = _resolve_geometry_dir(torax_dict, env_path)
    torax_config = model_config.ToraxConfig.from_dict(torax_dict)
    phase_snapshot = _load_phase_snapshot(env_path, phase_initialization)
    resolved_reward, resolved_penalty = _resolve_task_settings(
        cfg, reward, disruption_penalty
    )
    return _build_env(
        cfg,
        torax_config,
        resolved_reward,
        variant,
        max_steps,
        resolved_penalty,
        ablate,
        time_aware,
        phase_snapshot,
    )
