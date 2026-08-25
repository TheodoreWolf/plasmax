"""Parse and serialize validated plasmax scenario configuration."""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, field_validator

from plasmax.environment import registry as registry_lib
from plasmax.environment.merge import (
    _load_extended_yaml,
    _merge_env_and_backend,
    env_key,
    resolve_config_asset,
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


class PhaseInitializationConfig(BaseModel):
    """Phase-owned reference to one immutable initialization snapshot."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    path: str
    sha256: str

    @field_validator("path")
    @classmethod
    def _check_path(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("initialization.path must be non-empty")
        return value

    @field_validator("sha256")
    @classmethod
    def _check_sha256(cls, value: str) -> str:
        if re.fullmatch(r"[0-9a-fA-F]{64}", value) is None:
            raise ValueError("initialization.sha256 must contain 64 hex characters")
        return value.lower()

    def resolve_path(self, declaring_yaml: str | Path) -> Path:
        """Resolve the configured artifact relative to its owning phase YAML."""
        return resolve_config_asset(self.path, declaring_yaml)


class _PhaseMetadataConfig(BaseModel):
    """Metadata parsed directly from a leaf phase, before inheritance."""

    model_config = ConfigDict(frozen=True, extra="ignore", strict=True)

    initialization: PhaseInitializationConfig | None = None


class WorldModelEnvironmentConfig(BaseModel):
    """Construction settings intrinsic to a learned environment."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    max_steps_in_episode: int
    random_target: bool = True


class WorldModelBackendConfig(BaseModel):
    """Supported learned dynamics backend selector."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    name: Literal["kstar_lstm"]


class _WorldModelEnvironmentYaml(BaseModel):
    """Complete allowed top-level shape of a world-model environment YAML."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    task: TaskConfig
    world_model_env: WorldModelEnvironmentConfig
    initialization: None = None


class _WorldModelBackendYaml(BaseModel):
    """Complete allowed top-level shape of a world-model backend YAML."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    type: Literal["world_model"]
    world_model: WorldModelBackendConfig
    initialization: None = None


@dataclasses.dataclass(frozen=True)
class WorldModelSources:
    """Validated task metadata and learned-environment construction data."""

    task: TaskConfig
    environment: WorldModelEnvironmentConfig
    backend: WorldModelBackendConfig


@dataclasses.dataclass(frozen=True)
class ToraxSources:
    """Validated scenario plus phase-owned reset metadata."""

    scenario: ScenarioConfig
    initialization: PhaseInitializationConfig | None
    environment_key: str


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
    raw = _load_yaml_mapping(resolved)
    metadata = _PhaseMetadataConfig.model_validate(raw)
    if metadata.initialization is not None:
        raise ValueError(
            "phase initialization requires an environment with a TORAX backend"
        )
    return ScenarioConfig.model_validate(raw)


def parse_world_model_sources(
    env_path: str, backend_path: str, *, validate: bool = True
) -> WorldModelSources:
    """Parse the task and learned-model construction blocks for one pair."""
    resolved_env = registry_lib.resolve_env(env_path)
    resolved_backend = registry_lib.resolve_backend(backend_path)
    if validate:
        validate_env_backend(resolved_env, resolved_backend)
    env_yaml = _WorldModelEnvironmentYaml.model_validate(
        _load_yaml_mapping(resolved_env)
    )
    backend_yaml = _WorldModelBackendYaml.model_validate(
        _load_extended_yaml(resolved_backend)
    )
    return WorldModelSources(
        task=env_yaml.task,
        environment=env_yaml.world_model_env,
        backend=backend_yaml.world_model,
    )


def parse_torax_sources(
    env_path: str, backend_path: str, *, validate: bool = True
) -> ToraxSources:
    """Parse one TORAX pair while keeping phase metadata out of its scenario."""
    resolved_env = registry_lib.resolve_env(env_path)
    resolved_backend = registry_lib.resolve_backend(backend_path)
    if validate:
        validate_env_backend(resolved_env, resolved_backend)
    phase_metadata = _PhaseMetadataConfig.model_validate(
        _load_yaml_mapping(resolved_env)
    )
    raw = _merge_env_and_backend(resolved_env, resolved_backend)
    return ToraxSources(
        scenario=ScenarioConfig.model_validate(raw),
        initialization=phase_metadata.initialization,
        environment_key=env_key(resolved_env),
    )


def parse_env_and_backend(
    env_path: str, backend_path: str, *, validate: bool = True
) -> ScenarioConfig:
    """Load and merge an environment/backend pair into a config model."""
    return parse_torax_sources(env_path, backend_path, validate=validate).scenario


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
    "parse_torax_sources",
    "parse_world_model_sources",
    "PhaseInitializationConfig",
    "PhysicsRandomizationSpec",
    "RealisticObsConfig",
    "ScenarioConfig",
    "scenario_to_yaml",
    "SteppingConfig",
    "TaskConfig",
    "ToraxSources",
    "validate_env_backend",
    "valid_env_backend_combos",
    "WorldModelBackendConfig",
    "WorldModelEnvironmentConfig",
    "WorldModelSources",
]
