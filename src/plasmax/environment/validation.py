"""Construction-time validation for TORAX RL environment configs."""

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from torax._src.config import build_runtime_params
from torax._src.torax_pydantic import interpolated_param_1d, model_config

from plasmax.control import _FIELD_TO_PATH
from plasmax.environment.schema import PhysicsRandomizationSpec

__all__ = [
    "validate_actuator_specs",
    "validate_observation_specs",
    "validate_physics_randomization",
    "validate_state_noise_config",
]


_STATE_NOISE_FIELDS = frozenset({"T_e", "T_i", "n_e"})


def _names(specs: Sequence[Any]) -> list[str]:
    return [s.name for s in specs]


def _duplicates(names: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for name in names:
        if name in seen and name not in duplicates:
            duplicates.append(name)
        seen.add(name)
    return duplicates


def config_n_rho(cfg: model_config.ToraxConfig) -> int:
    """n_rho of a TORAX config's geometry.

    ``geometry_configs`` is either a single ``GeometryConfig`` (with ``.config``)
    or a time-keyed ``dict`` of them when geometry is interpolated in time
    (e.g. per-Ip equilibria through a ramp). Time-keyed geometries share one
    mesh — TORAX enforces equal meshes across the dict — so n_rho is read from
    any slice, matching TORAX's own ``Geometry.get_face_centers`` handling.
    """
    geometry_configs = cfg.geometry.geometry_configs
    if isinstance(geometry_configs, dict):
        return next(iter(geometry_configs.values())).config.n_rho
    return geometry_configs.config.n_rho


def validate_actuator_specs(
    actuator_specs: Sequence[Any],
    provider: build_runtime_params.RuntimeParamsProvider,
) -> None:
    """Validates actuator names, bounds, and provider availability."""
    names = _names(actuator_specs)
    unknown = [name for name in names if name not in _FIELD_TO_PATH]
    if unknown:
        raise ValueError(
            f"Unknown actuator names: {unknown}. "
            f"Valid names are: {list(_FIELD_TO_PATH)}"
        )

    duplicates = _duplicates(names)
    if duplicates:
        raise ValueError(f"Duplicate actuator names are not allowed: {duplicates}")

    for spec in actuator_specs:
        if not spec.low < spec.high:
            raise ValueError(
                f"Actuator {spec.name!r} must have low < high, got "
                f"low={spec.low}, high={spec.high}."
            )
        if spec.max_delta < 0:
            raise ValueError(
                f"Actuator {spec.name!r} must have non-negative max_delta, got "
                f"{spec.max_delta}."
            )

    for name in names:
        path = _FIELD_TO_PATH[name]
        try:
            provider.get_node_from_path(path)
        except ValueError as e:
            raise ValueError(
                f"Actuator {name!r} targets provider path {path!r}, but that "
                "path is not configured."
            ) from e


def validate_observation_specs(
    profile_specs: Sequence[Any],
    scalar_specs: Sequence[Any],
    profile_registry: Mapping[str, Any],
    scalar_registry: Mapping[str, Any],
) -> None:
    """Validates observation names, duplicate entries, scales, and bounds."""
    profile_names = _names(profile_specs)
    scalar_names = _names(scalar_specs)
    bad_profiles = [name for name in profile_names if name not in profile_registry]
    if bad_profiles:
        raise ValueError(
            f"Unknown profile obs names: {bad_profiles}. "
            f"Valid names: {list(profile_registry)}"
        )
    bad_scalars = [name for name in scalar_names if name not in scalar_registry]
    if bad_scalars:
        raise ValueError(
            f"Unknown scalar obs names: {bad_scalars}. "
            f"Valid names: {list(scalar_registry)}"
        )
    duplicate_profiles = _duplicates(profile_names)
    if duplicate_profiles:
        raise ValueError(f"Duplicate profile obs names: {duplicate_profiles}")
    duplicate_scalars = _duplicates(scalar_names)
    if duplicate_scalars:
        raise ValueError(f"Duplicate scalar obs names: {duplicate_scalars}")


def validate_state_noise_config(state_noise_config: Mapping[str, float]) -> None:
    """Validates reset-time state noise targets and scales."""
    unknown = sorted(set(state_noise_config) - _STATE_NOISE_FIELDS)
    if unknown:
        raise ValueError(
            f"Unknown state_noise fields: {unknown}. "
            f"Valid fields are: {sorted(_STATE_NOISE_FIELDS)}"
        )
    for name, scale in state_noise_config.items():
        if not np.isfinite(scale) or scale < 0:
            raise ValueError(
                f"state_noise field {name!r} must have a non-negative finite scale, "
                f"got {scale}."
            )


def validate_physics_randomization(
    physics_randomization: Mapping[str, PhysicsRandomizationSpec],
    provider: build_runtime_params.RuntimeParamsProvider,
) -> None:
    """Validates physics randomization ranges and provider paths."""
    for path, spec in physics_randomization.items():
        lo, hi = spec.bounds
        if not np.isfinite(lo) or not np.isfinite(hi) or not lo < hi:
            raise ValueError(
                f"physics_randomization path {path!r} must have a finite "
                f"(low, high) range with low < high, got {(lo, hi)}."
            )
        try:
            node = provider.get_node_from_path(path)
        except ValueError as e:
            raise ValueError(
                f"physics_randomization path {path!r} is not configured."
            ) from e
        if isinstance(node, interpolated_param_1d.TimeVaryingScalar):
            continue
        node_array = np.asarray(node)
        if node_array.shape or not np.issubdtype(node_array.dtype, np.floating):
            raise ValueError(
                f"physics_randomization path {path!r} must target a floating-point "
                f"scalar or TimeVaryingScalar, got {type(node).__name__}."
            )
