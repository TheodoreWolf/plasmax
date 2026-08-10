"""Parse and serialize validated plasmax scenario configuration."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import yaml

from plasmax.environment import registry as registry_lib
from plasmax.environment.merge import (
    _load_extended_yaml,
    _merge_env_and_backend,
    valid_env_backend_combos,
)
from plasmax.environment.merge import (
    backend_kind as _backend_kind,
)
from plasmax.environment.merge import (
    validate_env_backend as _validate_env_backend,
)
from plasmax.environment.schema import (
    ActuatorConfig,
    DisruptionConfig,
    HistoryConfig,
    ObservationsConfig,
    ObsFilterSpec,
    ObsProfileConfig,
    ObsScalarConfig,
    PhysicsRandomizationSpec,
    RealisticObsConfig,
    ScenarioConfig,
    SteppingConfig,
    TaskConfig,
)


@dataclasses.dataclass(frozen=True)
class _WorldModelSources:
    """Validated task metadata plus raw learned-environment construction data."""

    task: TaskConfig
    environment: dict[str, Any]
    backend: dict[str, Any]


def _load_yaml_mapping(path: str | Path) -> dict[str, Any]:
    """Read one YAML mapping for the parser functions in this module."""
    resolved = Path(path)
    with resolved.open() as stream:
        raw = yaml.safe_load(stream) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"YAML config {str(resolved)!r} must contain a mapping")
    return raw


def backend_kind(backend: str) -> str:
    """Return the kind of a backend path or registered alias."""
    return _backend_kind(registry_lib.resolve_backend(backend))


def validate_env_backend(env: str, backend: str) -> None:
    """Validate an environment/backend pair given paths or aliases."""
    _validate_env_backend(
        registry_lib.resolve_env(env), registry_lib.resolve_backend(backend)
    )


def parse_scenario(path: str) -> ScenarioConfig:
    """Load a single scenario path or alias into a frozen config model."""
    resolved = Path(registry_lib.resolve_scenario(path))
    return ScenarioConfig.model_validate(_load_yaml_mapping(resolved))


def _parse_world_model_sources(env_path: str, backend_path: str) -> _WorldModelSources:
    """Parse the task and learned-model construction blocks for one pair."""
    resolved_env = registry_lib.resolve_env(env_path)
    resolved_backend = registry_lib.resolve_backend(backend_path)
    env_raw = _load_yaml_mapping(resolved_env)
    backend_raw = _load_extended_yaml(resolved_backend)
    return _WorldModelSources(
        task=TaskConfig.model_validate(env_raw.get("task")),
        environment=dict(env_raw.get("world_model_env") or {}),
        backend=dict(backend_raw.get("world_model") or {}),
    )


def parse_env_and_backend(
    env_path: str, backend_path: str, *, validate: bool = True
) -> ScenarioConfig:
    """Load and merge an environment/backend pair into a config model."""
    resolved_env = registry_lib.resolve_env(env_path)
    resolved_backend = registry_lib.resolve_backend(backend_path)
    if validate:
        validate_env_backend(resolved_env, resolved_backend)
    raw = _merge_env_and_backend(resolved_env, resolved_backend)
    return ScenarioConfig.model_validate(raw)


def _tuples_to_lists(obj: Any) -> Any:
    """Convert tuples recursively so PyYAML emits portable sequences."""
    if isinstance(obj, tuple):
        return [_tuples_to_lists(value) for value in obj]
    if isinstance(obj, list):
        return [_tuples_to_lists(value) for value in obj]
    if isinstance(obj, dict):
        return {key: _tuples_to_lists(value) for key, value in obj.items()}
    return obj


def scenario_to_yaml(config: ScenarioConfig, path: str | Path) -> None:
    """Serialize a scenario config to portable YAML."""
    raw = _tuples_to_lists(config.model_dump(mode="python"))
    with Path(path).open("w") as stream:
        yaml.safe_dump(raw, stream, sort_keys=False, allow_unicode=True)


__all__ = [
    "ActuatorConfig",
    "backend_kind",
    "DisruptionConfig",
    "HistoryConfig",
    "ObservationsConfig",
    "ObsFilterSpec",
    "ObsProfileConfig",
    "ObsScalarConfig",
    "parse_env_and_backend",
    "parse_scenario",
    "PhysicsRandomizationSpec",
    "RealisticObsConfig",
    "ScenarioConfig",
    "scenario_to_yaml",
    "SteppingConfig",
    "TaskConfig",
    "validate_env_backend",
    "valid_env_backend_combos",
]
