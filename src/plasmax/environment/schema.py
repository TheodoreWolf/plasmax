"""Pydantic models for YAML scenario configuration."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from plasmax import spaces as spaces_lib


class ActuatorConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    name: str
    low: float
    high: float
    max_delta: float = float("inf")
    init: float | None = None


def _validate_obs_name(v: str, registry: Mapping[str, Any], kind: str) -> str:
    if v not in registry:
        raise ValueError(f"Unknown {kind} name {v!r}. Valid: {list(registry)}")
    return v


def _validate_bounds(v):
    if v is not None and v[0] >= v[1]:
        raise ValueError(f"bounds low must be < high, got {v}")
    return v


class _ObsEntryConfig(BaseModel):
    """Observation entry: registry name + normalisation scale + optional bounds."""

    model_config = ConfigDict(frozen=True)
    name: str
    scale: float
    bounds: tuple[float, float] | None = None

    _REGISTRY: ClassVar[Mapping[str, Any]]
    _KIND: ClassVar[str]

    @field_validator("name")
    @classmethod
    def _check_name(cls, v: str) -> str:
        return _validate_obs_name(v, cls._REGISTRY, cls._KIND)

    @field_validator("bounds")
    @classmethod
    def _check_bounds(cls, v):
        return _validate_bounds(v)

    @field_validator("scale")
    @classmethod
    def _check_scale(cls, value: float) -> float:
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(
                f"observation scale must be positive and finite, got {value}"
            )
        return value


class ObsProfileConfig(_ObsEntryConfig):
    _REGISTRY = spaces_lib.PROFILE_REGISTRY
    _KIND = "profile"


class ObsScalarConfig(_ObsEntryConfig):
    _REGISTRY = spaces_lib.SCALAR_REGISTRY
    _KIND = "scalar"


class ObsFilterSpec(BaseModel):
    """Named filter for oracle -> realistic obs projection."""

    model_config = ConfigDict(frozen=True)
    profiles: list[str] | None = None
    scalars: list[str] | None = None

    @model_validator(mode="after")
    def _check_unique_names(self) -> ObsFilterSpec:
        for kind, names in (("profile", self.profiles), ("scalar", self.scalars)):
            if names is not None and len(set(names)) != len(names):
                raise ValueError(f"duplicate {kind} filter names: {names}")
        return self


class RealisticObsConfig(BaseModel):
    """Disadvantageous sensor effects applied only when variant='realistic'."""

    model_config = ConfigDict(frozen=True)
    noise: dict[str, float] = Field(default_factory=dict)
    resolution: dict[str, int] = Field(default_factory=dict)
    filter: ObsFilterSpec | None = None
    delay: dict[str, float] = Field(default_factory=dict)

    @field_validator("resolution", mode="before")
    @classmethod
    def _check_resolution(cls, values: dict[str, int]) -> dict[str, int]:
        for name, value in values.items():
            if isinstance(value, bool) or value < 1:
                raise ValueError(
                    f"observation resolution for {name!r} must be an integer >= 1"
                )
        return values

    @field_validator("delay")
    @classmethod
    def _check_delay(cls, values: dict[str, float]) -> dict[str, float]:
        for name, value in values.items():
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"delay probability for {name!r} must be finite and in [0, 1]"
                )
        return values


class RealisticActionConfig(BaseModel):
    """Disadvantageous actuator effects applied only when variant='realistic'."""

    model_config = ConfigDict(frozen=True)
    quantize: dict[str, int] = Field(default_factory=dict)

    @field_validator("quantize")
    @classmethod
    def _check_bins(cls, v: dict[str, int]) -> dict[str, int]:
        for name, bins in v.items():
            if bins < 2:
                raise ValueError(f"quantize bins for {name!r} must be >= 2, got {bins}")
        return v


class ActionsConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    realistic: RealisticActionConfig = Field(default_factory=RealisticActionConfig)


class HistoryConfig(BaseModel):
    """Frame-stacking: emit the last ``length`` observations and actions."""

    model_config = ConfigDict(frozen=True)
    length: int

    @field_validator("length")
    @classmethod
    def _check_length(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"history length must be >= 1, got {v}")
        return v


class ObservationsConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    profiles: list[ObsProfileConfig]
    scalars: list[ObsScalarConfig]
    realistic: RealisticObsConfig = Field(default_factory=RealisticObsConfig)
    history: HistoryConfig | None = None

    @field_validator("profiles")
    @classmethod
    def _check_profile_unique(cls, v: list[ObsProfileConfig]) -> list[ObsProfileConfig]:
        names = [p.name for p in v]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate profile names: {names}")
        return v

    @field_validator("scalars")
    @classmethod
    def _check_scalar_unique(cls, v: list[ObsScalarConfig]) -> list[ObsScalarConfig]:
        names = [s.name for s in v]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate scalar names: {names}")
        return v


class DisruptionConfig(BaseModel):
    """Disruption-proxy early-termination thresholds."""

    model_config = ConfigDict(frozen=True)
    q_min_threshold: float = 0.8
    greenwald_threshold: float = 1.1
    # Which Greenwald fraction to threshold on: line-averaged (convention) or
    # volume-averaged n_e. Selects the postout field read in the disruption check.
    greenwald_metric: Literal["line_avg", "volume_avg"] = "line_avg"

    @property
    def greenwald_field(self) -> str:
        return f"fgw_n_e_{self.greenwald_metric}"


class SteppingConfig(BaseModel):
    """Static internal-step limits for one physical RL control interval."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    max_solver_substeps: int = 1
    max_event_substeps: int = 0

    @field_validator("max_solver_substeps", mode="before")
    @classmethod
    def _check_solver_limit(cls, value: int) -> int:
        if isinstance(value, bool) or value < 1:
            raise ValueError("max_solver_substeps must be an integer >= 1")
        return value

    @field_validator("max_event_substeps", mode="before")
    @classmethod
    def _check_event_limit(cls, value: int) -> int:
        if isinstance(value, bool) or value < 0:
            raise ValueError("max_event_substeps must be an integer >= 0")
        return value


class PhysicsRandomizationSpec(BaseModel):
    """Uniform absolute values or multipliers of a parameter's t=0 nominal."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    absolute: tuple[float, float] | None = None
    relative: tuple[float, float] | None = None

    @model_validator(mode="after")
    def _check_one_range(self) -> PhysicsRandomizationSpec:
        ranges = [self.absolute is not None, self.relative is not None]
        if sum(ranges) != 1:
            raise ValueError("set exactly one of absolute or relative")
        bounds = self.absolute if self.absolute is not None else self.relative
        assert bounds is not None
        if not all(math.isfinite(bound) for bound in bounds) or bounds[0] >= bounds[1]:
            raise ValueError(f"range must be finite with low < high, got {bounds}")
        return self

    @property
    def bounds(self) -> tuple[float, float]:
        bounds = self.absolute if self.absolute is not None else self.relative
        assert bounds is not None
        return bounds

    @property
    def is_relative(self) -> bool:
        return self.relative is not None


class TaskConfig(BaseModel):
    """Reward and calibrated terminal penalty defining one control task."""

    model_config = ConfigDict(frozen=True)
    reward: str
    terminal_penalty: float | None


class ScenarioConfig(BaseModel):
    """Scenario configuration: physics problem definition."""

    model_config = ConfigDict(frozen=True)
    torax: dict[str, Any]
    task: TaskConfig
    actuators: list[ActuatorConfig]
    observations: ObservationsConfig
    actions: ActionsConfig = Field(default_factory=ActionsConfig)
    state_noise: dict[str, float] = Field(default_factory=dict)
    physics_randomization: dict[str, PhysicsRandomizationSpec] = Field(
        default_factory=dict
    )
    disruption: DisruptionConfig = Field(default_factory=DisruptionConfig)
    stepping: SteppingConfig = Field(default_factory=SteppingConfig)
    clip_by_max_action_delta: bool = True


__all__ = [
    "ActuatorConfig",
    "DisruptionConfig",
    "HistoryConfig",
    "ObservationsConfig",
    "ObsFilterSpec",
    "ObsProfileConfig",
    "ObsScalarConfig",
    "PhysicsRandomizationSpec",
    "RealisticObsConfig",
    "ScenarioConfig",
    "SteppingConfig",
    "TaskConfig",
]
