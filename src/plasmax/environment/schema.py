"""Pydantic models for YAML environment configuration."""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    StrictInt,
    field_validator,
    model_validator,
)
from torax._src.torax_pydantic import model_config as torax_model_config

from plasmax import spaces as spaces_lib
from plasmax.environment.initialization_data import (
    KstarInitialization,
    ToraxInitialization,
)


class _FrozenModel(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
        frozen=True,
        populate_by_name=True,
    )


class TaskConfig(_FrozenModel):
    """Reward and calibrated terminal penalty defining one control task."""

    reward: str
    terminal_penalty: float | None


class ActuatorConfig(_FrozenModel):
    name: str
    low: float
    high: float
    max_delta: float = float("inf")
    init: float | None = None

    def to_spec(self) -> spaces_lib.ActuatorSpec:
        return spaces_lib.ActuatorSpec(
            name=self.name,
            low=self.low,
            high=self.high,
            max_delta=self.max_delta,
            init=self.init,
        )


def _validate_obs_name(v: str, registry: Mapping[str, Any], kind: str) -> str:
    if v not in registry:
        raise ValueError(f"Unknown {kind} name {v!r}. Valid: {list(registry)}")
    return v


def _validate_bounds(
    v: tuple[float, float] | None,
) -> tuple[float, float] | None:
    if v is not None and v[0] >= v[1]:
        raise ValueError(f"bounds low must be < high, got {v}")
    return v


class _ObsEntryConfig(_FrozenModel):
    """Observation entry: registry name + normalisation scale + optional bounds."""

    name: str
    scale: float
    bounds: tuple[float, float] | None = None

    _REGISTRY: ClassVar[Mapping[str, Any]]
    _KIND: ClassVar[str]

    @field_validator("name")
    @classmethod
    def _check_name(cls, v: str) -> str:
        return _validate_obs_name(v, cls._REGISTRY, cls._KIND)

    @field_validator("scale")
    @classmethod
    def _check_scale(cls, value: float) -> float:
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(
                f"observation scale must be positive and finite, got {value}"
            )
        return value

    @field_validator("bounds")
    @classmethod
    def _check_bounds(cls, v: tuple[float, float] | None) -> tuple[float, float] | None:
        return _validate_bounds(v)

    def to_spec(self) -> spaces_lib.ObsSpec:
        return spaces_lib.ObsSpec(
            name=self.name,
            scale=self.scale,
            bounds=self.bounds,
        )


class ObsProfileConfig(_ObsEntryConfig):
    _REGISTRY = spaces_lib.PROFILE_REGISTRY
    _KIND = "profile"


class ObsScalarConfig(_ObsEntryConfig):
    _REGISTRY = spaces_lib.SCALAR_REGISTRY
    _KIND = "scalar"


class ObsFilterSpec(_FrozenModel):
    """Named filter for oracle -> realistic obs projection."""

    profiles: tuple[str, ...] = ()
    scalars: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _check_unique_names(self) -> Self:
        for kind, names in (("profile", self.profiles), ("scalar", self.scalars)):
            if len(set(names)) != len(names):
                raise ValueError(f"duplicate {kind} filter names: {names}")
        return self


class RealisticObsConfig(_FrozenModel):
    """Disadvantageous sensor effects composed by RealisticWrappers."""

    noise: dict[str, float] = Field(default_factory=dict)
    resolution: dict[str, StrictInt] = Field(default_factory=dict)
    filter: ObsFilterSpec | None = None
    delay: dict[str, float] = Field(default_factory=dict)

    @field_validator("resolution")
    @classmethod
    def _check_resolution(cls, values: dict[str, int]) -> dict[str, int]:
        for name, value in values.items():
            if value < 1:
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


class HistoryConfig(_FrozenModel):
    """Frame-stacking: emit the last ``length`` observations and actions."""

    length: StrictInt = Field(ge=1)


class ObservationsConfig(_FrozenModel):
    profiles: tuple[ObsProfileConfig, ...]
    scalars: tuple[ObsScalarConfig, ...]
    realistic: RealisticObsConfig = Field(default_factory=RealisticObsConfig)
    history: HistoryConfig | None = None

    @model_validator(mode="after")
    def _valid_sensor_sets(self) -> Self:
        profile_names = tuple(item.name for item in self.profiles)
        scalar_names = tuple(item.name for item in self.scalars)
        if len(profile_names) != len(set(profile_names)):
            raise ValueError(f"duplicate profile names: {list(profile_names)}")
        if len(scalar_names) != len(set(scalar_names)):
            raise ValueError(f"duplicate scalar names: {list(scalar_names)}")

        realistic = self.realistic
        base_names = set(profile_names) | set(scalar_names)
        unknown_resolution = sorted(set(realistic.resolution) - set(profile_names))
        if unknown_resolution:
            raise ValueError(
                f"unknown profile resolution sensors: {unknown_resolution}"
            )

        surviving = base_names
        if realistic.filter is not None:
            bad_filter_profiles = sorted(
                set(realistic.filter.profiles) - set(profile_names)
            )
            bad_filter_scalars = sorted(
                set(realistic.filter.scalars) - set(scalar_names)
            )
            if bad_filter_profiles or bad_filter_scalars:
                raise ValueError(
                    "unknown observation filter sensors: "
                    f"{bad_filter_profiles + bad_filter_scalars}"
                )
            surviving = set(realistic.filter.profiles) | set(realistic.filter.scalars)
        unknown_delay = sorted(set(realistic.delay) - surviving)
        if unknown_delay:
            raise ValueError(f"unknown delay sensors: {unknown_delay}")
        return self


class RealisticActionConfig(_FrozenModel):
    """Disadvantageous actuator effects composed by RealisticWrappers."""

    quantize: dict[str, StrictInt] = Field(default_factory=dict)

    @field_validator("quantize")
    @classmethod
    def _check_bins(cls, v: dict[str, int]) -> dict[str, int]:
        for name, bins in v.items():
            if bins < 2:
                raise ValueError(f"quantize bins for {name!r} must be >= 2, got {bins}")
        return v


class ActionsConfig(_FrozenModel):
    realistic: RealisticActionConfig = Field(default_factory=RealisticActionConfig)


class PhysicsRandomizationSpec(_FrozenModel):
    """Uniform absolute values or multipliers of a parameter's t=0 nominal."""

    absolute: tuple[float, float] | None = None
    relative: tuple[float, float] | None = None

    @model_validator(mode="after")
    def _check_one_range(self) -> Self:
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


class DisruptionConfig(_FrozenModel):
    """Disruption-proxy early-termination thresholds."""

    q_min_threshold: float = 0.8
    greenwald_threshold: float = 1.1
    # Which Greenwald fraction to threshold on: line-averaged (convention) or
    # volume-averaged n_e. Selects the postout field read in the disruption check.
    greenwald_metric: Literal["line_avg", "volume_avg"] = "line_avg"

    @property
    def greenwald_field(self) -> str:
        return f"fgw_n_e_{self.greenwald_metric}"


class SteppingConfig(_FrozenModel):
    """Static internal-step limits for one physical RL control interval."""

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


class PlasmaxConfig(_FrozenModel):
    """Fully composed, asset-resolved configuration for one TORAX environment."""

    environment_key: str
    torax: torax_model_config.ToraxConfig
    task: TaskConfig
    initialization: Path
    _initial_state: ToraxInitialization = PrivateAttr()
    actuators: tuple[ActuatorConfig, ...]
    observations: ObservationsConfig
    actions: ActionsConfig = Field(default_factory=ActionsConfig)
    state_noise: dict[str, float] = Field(default_factory=dict)
    physics_randomization: dict[str, PhysicsRandomizationSpec] = Field(
        default_factory=dict
    )
    disruption: DisruptionConfig = Field(default_factory=DisruptionConfig)
    stepping: SteppingConfig = Field(default_factory=SteppingConfig)
    clip_by_max_action_delta: bool = True

    @model_validator(mode="after")
    def _valid_actions(self) -> Self:
        if self.task.terminal_penalty is None:
            raise ValueError("TORAX environment terminal_penalty cannot be null")
        names = tuple(actuator.name for actuator in self.actuators)
        if len(names) != len(set(names)):
            raise ValueError("duplicate actuator names")
        quantized = set(self.actions.realistic.quantize)
        if quantized and quantized != set(names):
            missing = set(names) - quantized
            extra = quantized - set(names)
            raise ValueError(
                "actions.realistic.quantize must cover every actuator or none; "
                f"missing: {sorted(missing)}, unknown: {sorted(extra)}"
            )
        return self


class WorldModelSpec(_FrozenModel):
    name: Literal["kstar_lstm"]
    weights_path: Path
    max_steps_in_episode: StrictInt = Field(ge=1)
    random_target: bool = True


class WorldModelConfig(_FrozenModel):
    """Complete standalone KSTAR learned-environment configuration."""

    environment_key: Literal["kstar_worldmodel"]
    task: TaskConfig
    world_model: WorldModelSpec
    initialization: Path
    _initial_state: KstarInitialization = PrivateAttr()

    @model_validator(mode="after")
    def _native_task(self) -> Self:
        if self.task.reward != "native":
            raise ValueError("world-model reward must be 'native'")
        if self.task.terminal_penalty is not None:
            raise ValueError("world models do not use a terminal penalty")
        return self


__all__ = [
    "ActionsConfig",
    "ActuatorConfig",
    "DisruptionConfig",
    "HistoryConfig",
    "ObservationsConfig",
    "ObsFilterSpec",
    "ObsProfileConfig",
    "ObsScalarConfig",
    "PhysicsRandomizationSpec",
    "PlasmaxConfig",
    "RealisticActionConfig",
    "RealisticObsConfig",
    "SteppingConfig",
    "TaskConfig",
    "WorldModelConfig",
    "WorldModelSpec",
]
