"""Typed provenance records for backend-independent physical reset anchors."""

from __future__ import annotations

import hashlib
import json
import math
import numbers
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class ReferenceSource(BaseModel):
    """One immutable upstream input used to construct a reset reference."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    title: str
    url: str
    sha256: Sha256
    doi: str | None = None
    revision: str | None = None
    license: str | None = None
    redistribution: Literal["vendored", "checksum_only"]
    local_path: str | None = None


class ReferenceExtraction(BaseModel):
    """Deterministic transformation from source material to TORAX inputs."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    method: Literal[
        "imas_import",
        "torax_config",
        "sectioned_profile",
        "figure_digitization",
        "constrained_template",
        "simulation_capture",
        "model_inference",
    ]
    grid: str
    page: int | None = None
    crop: tuple[int, int, int, int] | None = None
    axis_transform: dict[str, tuple[float, float]] = Field(default_factory=dict)
    digitization_tolerance_pixels: float | None = Field(default=None, gt=0.0)
    notes: tuple[str, ...] = ()


class ReferenceProjection(BaseModel):
    """One documented projection from released data to exposed TORAX inputs."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_component: str
    torax_component: str
    matched_quantities: tuple[str, ...]
    source_targets: dict[str, float | str] = Field(default_factory=dict)
    torax_parameters: dict[str, float | str] = Field(default_factory=dict)
    notes: tuple[str, ...] = ()


class PhysicalReference(BaseModel):
    """One cold or hot physical anchor shared by simulator backends."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    machine: Literal["iter", "sparc", "step"]
    scenario: str
    anchor: Literal["cold", "hot"]
    kind: Literal["exact", "upstream", "digitized", "constrained"]
    confidence: Literal["high", "medium"]
    sources: tuple[ReferenceSource, ...]
    extraction: ReferenceExtraction
    profile_sha256: Sha256
    artifacts: dict[str, Sha256] = Field(default_factory=dict)
    locked_paths: tuple[str, ...]
    projections: tuple[ReferenceProjection, ...] = ()
    targets: dict[str, float | str] = Field(default_factory=dict)


class ReferenceManifest(BaseModel):
    """Versioned collection of physical reset references."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1]
    references: dict[str, PhysicalReference]


def reference_manifest_path() -> Path:
    """Return the installed manifest path."""

    return Path(__file__).parents[1] / "configs" / "references.yaml"


def load_reference_manifest(path: str | Path | None = None) -> ReferenceManifest:
    """Load and validate the packaged reset-reference manifest."""

    manifest_path = reference_manifest_path() if path is None else Path(path)
    with manifest_path.open() as stream:
        raw = yaml.safe_load(stream) or {}
    return ReferenceManifest.model_validate(raw)


_TIME_ZERO_PROFILE_FIELDS = frozenset(
    {
        "Ip",
        "T_i",
        "T_i_right_bc",
        "T_e",
        "T_e_right_bc",
        "n_e",
        "n_e_right_bc",
        "nbar",
        "psi",
        "v_loop_lcfs",
    }
)


def _is_numeric_key(value: object) -> bool:
    return isinstance(value, numbers.Real) and not isinstance(value, bool)


def _at_time_zero(value: Any) -> Any:
    """Return the first value from a TORAX time-indexed mapping."""

    if not isinstance(value, Mapping) or not value:
        return value
    if not all(_is_numeric_key(key) for key in value):
        return value
    first_key = min(value, key=float)
    return value[first_key]


def _canonical_json_value(value: Any) -> Any:
    """Normalize YAML values for stable, type-insensitive JSON hashing."""

    if isinstance(value, Mapping):
        return {
            str(key): _canonical_json_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, tuple | list):
        return [_canonical_json_value(item) for item in value]
    if hasattr(value, "tolist"):
        return _canonical_json_value(value.tolist())
    if isinstance(value, numbers.Real) and not isinstance(value, bool):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError("reset-reference payloads must contain finite numbers")
        return result
    return value


def reset_reference_id(env: str) -> str:
    """Read an environment's metadata-only reset-reference identifier."""

    from plasmax.environment.merge import load_env_layers

    raw = load_env_layers(env)
    reference_id = raw.get("reset_reference")
    if not isinstance(reference_id, str) or not reference_id:
        raise ValueError(f"environment {env!r} has no reset_reference ID")
    return reference_id


def reset_reference_payload(
    env: str,
    backend: str | None = None,
) -> dict[str, Any]:
    """Return the resolved physical initialization selected by a task.

    Current/boundary schedules and actuators are task context. The fingerprint
    covers the complete saved state, including history and composition, and is
    independent of the destination transport backend.
    """

    from plasmax.environment.config import _load_initialization, _resolve_assets
    from plasmax.environment.initialization_data import ToraxInitialization
    from plasmax.environment.merge import _merge_env_and_backend, load_env_layers

    raw = (
        load_env_layers(env)
        if backend is None
        else _merge_env_and_backend(env, backend)
    )
    _, initial = _load_initialization(_resolve_assets(raw), env)
    assert isinstance(initial, ToraxInitialization)
    profile_conditions = initial.profile_conditions()
    for field in _TIME_ZERO_PROFILE_FIELDS & profile_conditions.keys():
        profile_conditions[field] = _at_time_zero(profile_conditions[field])
    payload = {
        "profile_conditions": profile_conditions,
        "grid": initial.grid.model_dump(),
        "reset_state": initial.reset_state.model_dump(),
        "composition": (
            None if initial.composition is None else initial.composition.model_dump()
        ),
    }
    return _canonical_json_value(payload)


def reset_reference_sha256(
    env: str,
    backend: str | None = None,
) -> str:
    """Hash the complete rounded physical state selected by an environment."""

    payload = reset_reference_payload(env, backend)
    encoded = json.dumps(
        payload,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "PhysicalReference",
    "ReferenceExtraction",
    "ReferenceManifest",
    "ReferenceProjection",
    "ReferenceSource",
    "load_reference_manifest",
    "reference_manifest_path",
    "reset_reference_id",
    "reset_reference_payload",
    "reset_reference_sha256",
]
