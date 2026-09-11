"""Readable, typed initialization documents shared by TORAX and KSTAR."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Annotated, Any, Literal

import numpy as np
import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
    model_validator,
)

from plasmax.environment.references import ReferenceExtraction

Number = Annotated[float, Field(allow_inf_nan=False)]
Vector = tuple[Number, ...]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Provenance(_Model):
    environment: str
    source_backend: str
    source_step: int
    source_time_s: Number
    seed: int
    source_config_sha256: str
    torax_version: str | None = None
    reference: str | None = None
    sources: tuple[dict[str, Any], ...] = ()
    extraction: ReferenceExtraction | None = None
    artifacts: dict[str, str] = Field(default_factory=dict)
    notes: tuple[str, ...] = ()


class Grid(_Model):
    rho_norm: Vector
    rho_face_norm: Vector


class Profiles(_Model):
    T_i_keV: Vector
    T_e_keV: Vector
    n_e_m3: Vector
    psi_Wb: Vector


class ResetState(_Model):
    confinement_mode: int
    dW_thermal_i_dt_smoothed: Number
    dW_thermal_e_dt_smoothed: Number


class Composition(_Model):
    """Composition sampled on the union of TORAX cell and face coordinates."""

    rho_norm: Vector
    main_ion: dict[str, Number]
    Z_eff: Vector
    impurity_mode: str
    impurity_species: dict[str, Vector | None]

    @model_validator(mode="after")
    def _shapes(self) -> Composition:
        for values in (self.Z_eff, *self.impurity_species.values()):
            if values is not None and len(values) != len(self.rho_norm):
                raise ValueError("composition profile shape differs from rho_norm")
        return self

    def torax_parameters(self) -> dict[str, Any]:
        def profile(values: Vector) -> dict[float, float]:
            return dict(zip(self.rho_norm, values, strict=True))

        return {
            "main_ion": dict(self.main_ion),
            "Z_eff": profile(self.Z_eff),
            "impurity": {
                "impurity_mode": self.impurity_mode,
                "species": {
                    name: None if values is None else profile(values)
                    for name, values in self.impurity_species.items()
                },
            },
        }


class ToraxInitialization(_Model):
    schema_version: Literal[1] = 1
    kind: Literal["torax"] = "torax"
    description: str
    provenance: Provenance
    grid: Grid
    profiles: Profiles
    reset_state: ResetState
    composition: Composition | None = None

    @model_validator(mode="after")
    def _shapes(self) -> ToraxInitialization:
        count = len(self.grid.rho_norm)
        if not count or len(self.grid.rho_face_norm) != count + 1:
            raise ValueError("initialization grid requires n cells and n+1 faces")
        for name in Profiles.model_fields:
            if len(getattr(self.profiles, name)) != count:
                raise ValueError(f"initialization {name} shape differs from cell grid")
        return self

    def profile_conditions(self) -> dict[str, Any]:
        """Complete profile leaves, inserted only after task layer composition."""
        result: dict[str, Any] = {
            "normalize_n_e_to_nbar": False,
            "n_e_nbar_is_fGW": False,
            "initial_psi_mode": "profile_conditions",
        }
        for name, field in (
            ("T_i", "T_i_keV"),
            ("T_e", "T_e_keV"),
            ("n_e", "n_e_m3"),
            ("psi", "psi_Wb"),
        ):
            result[name] = {
                0.0: dict(
                    zip(
                        self.grid.rho_norm,
                        getattr(self.profiles, field),
                        strict=True,
                    )
                )
            }
        return result


class Target(_Model):
    default: Number
    minimum: Number
    maximum: Number


class KstarInitialization(_Model):
    schema_version: Literal[1] = 1
    kind: Literal["kstar"] = "kstar"
    description: str
    provenance: Provenance
    inputs: dict[str, Number]
    input_order: tuple[str, ...]
    history_row: Vector
    history_columns: tuple[str, ...]
    history_length: Literal[10] = 10
    targets: dict[str, Target]

    @model_validator(mode="after")
    def _shapes(self) -> KstarInitialization:
        if (
            len(self.input_order) != 15
            or len(self.inputs) != 15
            or set(self.input_order) != set(self.inputs)
        ):
            raise ValueError("KSTAR initialization requires 15 named inputs")
        if (
            len(self.history_row) != 21
            or len(self.history_columns) != 21
            or len(set(self.history_columns)) != 21
        ):
            raise ValueError("KSTAR initialization history requires 21 columns")
        if set(self.targets) != {"betap", "q95", "li"}:
            raise ValueError("KSTAR initialization requires betap, q95, li targets")
        return self


Initialization = Annotated[
    ToraxInitialization | KstarInitialization, Field(discriminator="kind")
]
_DOCUMENT = TypeAdapter(Initialization)


def load_initialization(
    path: str | Path,
    *,
    kind: Literal["torax", "kstar"] | None = None,
) -> ToraxInitialization | KstarInitialization:
    """Read one complete document; no includes, inheritance, or binary fallback."""
    source = Path(path)
    if source.suffix not in {".yaml", ".yml"}:
        raise ValueError("initialization must be a YAML file")
    with source.open() as stream:
        try:
            document = _DOCUMENT.validate_python(yaml.safe_load(stream))
        except (ValidationError, yaml.YAMLError) as exc:
            raise ValueError(f"Invalid initialization {source}: {exc}") from exc
    if kind is not None and document.kind != kind:
        raise ValueError(f"expected {kind} initialization, got {document.kind}")
    return document


def round_significant(value: Any) -> Any:
    """Convert numerical payloads to plain values with four significant figures."""
    if isinstance(value, Mapping):
        return {key: round_significant(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return round_significant(value.tolist())
    if isinstance(value, tuple | list):
        return [round_significant(item) for item in value]
    if isinstance(value, float | np.floating):
        if not math.isfinite(value):
            raise ValueError("initialization numbers must be finite")
        return 0.0 if value == 0 else float(f"{value:.4g}")
    if isinstance(value, np.integer):
        return int(value)
    return value


class _Dumper(yaml.SafeDumper):
    pass


def _float(dumper: _Dumper, value: float) -> yaml.ScalarNode:
    literal = f"{value:.4g}"
    # A decimal suffix on 2021 would display five significant figures.
    if "." not in literal and "e" not in literal and abs(value) >= 1000:
        literal = f"{value:.3e}"
    if "e" in literal:
        mantissa, exponent = literal.split("e")
        mantissa = mantissa.rstrip("0").rstrip(".") if "." in mantissa else mantissa
        if "." not in mantissa:
            mantissa += ".0"
        literal = f"{mantissa}e{exponent}"
    elif "." not in literal:
        literal += ".0"
    return dumper.represent_scalar("tag:yaml.org,2002:float", literal)


def _sequence(dumper: _Dumper, value: list[Any]) -> yaml.SequenceNode:
    return dumper.represent_sequence(
        "tag:yaml.org,2002:seq",
        value,
        flow_style=bool(value) and all(isinstance(v, int | float) for v in value),
    )


_Dumper.add_representer(float, _float)
_Dumper.add_representer(list, _sequence)


def write_initialization(
    document: ToraxInitialization | KstarInitialization,
    path: str | Path,
) -> str:
    """Write the authoritative rounded payload with stable, wrapped YAML arrays."""
    destination = Path(path)
    if destination.suffix not in {".yaml", ".yml"}:
        raise ValueError("initialization must be a YAML file")
    values = round_significant(document.model_dump(mode="python", exclude_none=True))
    # Preserve explicit null impurity fractions (the species solved from Z_eff).
    if isinstance(document, ToraxInitialization) and document.composition is not None:
        values["composition"]["impurity_species"] = round_significant(
            document.composition.impurity_species
        )
    _DOCUMENT.validate_python(values)
    encoded = yaml.dump(values, Dumper=_Dumper, sort_keys=False, width=88).encode()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(dir=destination.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(encoded)
        temporary.replace(destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "Composition",
    "Grid",
    "KstarInitialization",
    "Profiles",
    "Provenance",
    "ResetState",
    "ToraxInitialization",
    "load_initialization",
    "round_significant",
    "write_initialization",
]
