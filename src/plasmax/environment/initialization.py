"""Resolved TORAX state and adapter for readable YAML initializations."""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.metadata
from pathlib import Path
from typing import Literal

import numpy as np
from jax import numpy as jnp
from pydantic import (
    BaseModel,
    ConfigDict,
    ValidationInfo,
    field_validator,
    model_validator,
)
from torax._src import jax_utils
from torax._src.config import build_runtime_params
from torax._src.core_profiles import profile_conditions as profile_conditions_lib
from torax._src.orchestration import initial_state as initial_state_lib
from torax._src.orchestration import sim_state as sim_state_lib
from torax._src.orchestration.step_function import SimulationStepFn
from torax._src.output_tools import post_processing
from torax._src.transport_model import transport_coefficients_builder

from plasmax.environment.initialization_data import (
    Grid,
    Profiles,
    Provenance,
    ResetState,
    ToraxInitialization,
    load_initialization,
    write_initialization,
)

_GRID_FIELDS = ("rho_norm", "rho_face_norm")
_PROFILE_FIELDS = ("T_i", "T_e", "n_e", "psi")
_ENERGY_HISTORY_FIELDS = (
    "dW_thermal_i_dt_smoothed",
    "dW_thermal_e_dt_smoothed",
)


class SnapshotMetadata(BaseModel):
    """Generation context of a resolved TORAX state."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    environment: str
    source_backend: str
    source_step: int
    source_time_s: float
    seed: int
    source_config_sha256: str
    torax_version: str

    @field_validator("source_time_s")
    @classmethod
    def _validate_finite_source_time(cls, value: float) -> float:
        if not np.isfinite(value):
            raise ValueError("snapshot source_time_s must be finite")
        return value


class PhaseSnapshot(BaseModel):
    """Backend-agnostic physical state reconstructed from a YAML initialization."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
        frozen=True,
        strict=True,
    )

    schema_version: Literal[1]
    rho_norm: np.ndarray
    rho_face_norm: np.ndarray
    T_i: np.ndarray
    T_e: np.ndarray
    n_e: np.ndarray
    psi: np.ndarray
    dW_thermal_i_dt_smoothed: float
    dW_thermal_e_dt_smoothed: float
    confinement_mode: int
    metadata: SnapshotMetadata

    @field_validator("schema_version", "confinement_mode", mode="before")
    @classmethod
    def _normalize_integer_scalar(
        cls,
        value: object,
        info: ValidationInfo,
    ) -> object:
        if not isinstance(value, np.ndarray):
            return value
        if value.shape != ():
            raise ValueError(
                f"snapshot {info.field_name} has shape {value.shape}; expected ()"
            )
        if value.dtype.kind not in "iu":
            raise ValueError(
                f"snapshot {info.field_name} has incompatible dtype {value.dtype}"
            )
        return value.item()

    @field_validator(*_ENERGY_HISTORY_FIELDS, mode="before")
    @classmethod
    def _normalize_float_scalar(
        cls,
        value: object,
        info: ValidationInfo,
    ) -> object:
        if not isinstance(value, np.ndarray):
            return value
        if value.shape != ():
            raise ValueError(
                f"snapshot {info.field_name} has shape {value.shape}; expected ()"
            )
        if value.dtype.kind != "f":
            raise ValueError(
                f"snapshot {info.field_name} has incompatible dtype {value.dtype}"
            )
        return value.item()

    @model_validator(mode="after")
    def _validate_numpy_payload(self) -> PhaseSnapshot:
        arrays = {
            name: getattr(self, name) for name in (*_GRID_FIELDS, *_PROFILE_FIELDS)
        }
        for name, array in arrays.items():
            if array.ndim != 1:
                raise ValueError(
                    f"snapshot {name} must be one-dimensional, got {array.shape}"
                )
            if array.dtype.kind != "f":
                raise ValueError(
                    f"snapshot {name} has incompatible dtype {array.dtype}"
                )
            if not np.all(np.isfinite(array)):
                raise ValueError(f"snapshot {name} must be finite")

        n_rho = self.rho_norm.shape[0]
        expected_shapes = {
            "rho_norm": (n_rho,),
            "rho_face_norm": (n_rho + 1,),
            **{name: (n_rho,) for name in _PROFILE_FIELDS},
        }
        for name, shape in expected_shapes.items():
            actual = getattr(self, name).shape
            if actual != shape:
                raise ValueError(
                    f"snapshot {name} has shape {actual}; expected {shape}"
                )

        for name in _ENERGY_HISTORY_FIELDS:
            if not np.isfinite(getattr(self, name)):
                raise ValueError(f"snapshot {name} must be finite")
        return self


def snapshot_sha256(path: str | Path) -> str:
    """Return the checksum of the saved initialization document."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def snapshot_from_initialization(document: ToraxInitialization) -> PhaseSnapshot:
    """Convert validated YAML values to the numerical reset adapter payload."""
    metadata = document.provenance.model_dump(
        include=set(SnapshotMetadata.model_fields)
    )
    return PhaseSnapshot(
        schema_version=1,
        rho_norm=np.asarray(document.grid.rho_norm, dtype=np.float64),
        rho_face_norm=np.asarray(document.grid.rho_face_norm, dtype=np.float64),
        T_i=np.asarray(document.profiles.T_i_keV, dtype=np.float64),
        T_e=np.asarray(document.profiles.T_e_keV, dtype=np.float64),
        n_e=np.asarray(document.profiles.n_e_m3, dtype=np.float64),
        psi=np.asarray(document.profiles.psi_Wb, dtype=np.float64),
        metadata=SnapshotMetadata.model_validate(metadata),
        **document.reset_state.model_dump(),
    )


def load_snapshot(path: str | Path) -> PhaseSnapshot:
    """Load a TORAX YAML initialization for the numerical reset adapter."""
    document = load_initialization(path, kind="torax")
    assert isinstance(document, ToraxInitialization)
    return snapshot_from_initialization(document)


def initialization_from_snapshot(
    snapshot: PhaseSnapshot,
    *,
    description: str | None = None,
    provenance: Provenance | None = None,
) -> ToraxInitialization:
    """Build a serializable document from a captured nominal state."""
    return ToraxInitialization(
        description=description or f"{snapshot.metadata.environment}: resolved state",
        provenance=provenance or Provenance(**snapshot.metadata.model_dump()),
        grid=Grid(rho_norm=snapshot.rho_norm, rho_face_norm=snapshot.rho_face_norm),
        profiles=Profiles(
            T_i_keV=snapshot.T_i,
            T_e_keV=snapshot.T_e,
            n_e_m3=snapshot.n_e,
            psi_Wb=snapshot.psi,
        ),
        reset_state=ResetState(
            **{name: getattr(snapshot, name) for name in ResetState.model_fields}
        ),
    )


def write_snapshot(snapshot: PhaseSnapshot, path: str | Path) -> str:
    """Write a captured state to YAML at four significant figures."""
    return write_initialization(initialization_from_snapshot(snapshot), path)


def snapshot_from_state(
    sim_state: sim_state_lib.SimState,
    *,
    environment: str,
    source_backend: str,
    source_step: int,
    seed: int,
    source_config_sha256: str,
) -> PhaseSnapshot:
    """Extract the backend-agnostic fields from an accepted TORAX state."""
    energy = sim_state.core_profiles.internal_plasma_energy
    pedestal_state = sim_state.pedestal_transition_state
    if energy is None or pedestal_state is None:
        raise ValueError("TORAX state is missing phase-snapshot fields")
    profiles = {
        name: np.asarray(getattr(sim_state.core_profiles, name).value)
        for name in _PROFILE_FIELDS
    }
    energy_history = {
        name: float(np.asarray(getattr(energy, name)))
        for name in _ENERGY_HISTORY_FIELDS
    }
    return PhaseSnapshot(
        schema_version=1,
        rho_norm=np.asarray(sim_state.geometry.rho_norm),
        rho_face_norm=np.asarray(sim_state.geometry.rho_face_norm),
        confinement_mode=int(np.asarray(pedestal_state.confinement_mode)),
        metadata=SnapshotMetadata(
            environment=environment,
            source_backend=source_backend,
            source_step=source_step,
            source_time_s=float(np.asarray(sim_state.t)),
            seed=seed,
            source_config_sha256=source_config_sha256,
            torax_version=importlib.metadata.version("torax"),
        ),
        **profiles,
        **energy_history,
    )


def rebuild_state_from_snapshot(
    snapshot: PhaseSnapshot,
    *,
    step_fn: SimulationStepFn,
) -> tuple[sim_state_lib.SimState, post_processing.PostProcessedOutputs]:
    """Rebuild a destination-native TORAX state at the phase's initial time."""
    provider = step_fn.runtime_params_provider
    runtime_params, geometry = (
        build_runtime_params.get_consistent_runtime_params_and_geometry(
            t=provider.numerics.t_initial,
            runtime_params_provider=provider,
            geometry_provider=step_fn.geometry_provider,
            is_initialization=True,
        )
    )
    for name in ("rho_norm", "rho_face_norm"):
        expected = getattr(snapshot, name)
        actual = np.asarray(getattr(geometry, name))
        if actual.shape != expected.shape or not np.allclose(
            actual,
            expected,
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise ValueError(
                f"snapshot {name} does not match destination geometry: "
                f"snapshot shape {expected.shape}, destination shape {actual.shape}"
            )

    profile_overrides = {
        name: jnp.asarray(getattr(snapshot, name), dtype=jax_utils.get_dtype())
        for name in _PROFILE_FIELDS
    }
    profile_conditions = dataclasses.replace(
        runtime_params.profile_conditions,
        n_e_nbar_is_fGW=False,
        normalize_n_e_to_nbar=False,
        initial_psi_from_j=False,
        initial_psi_mode=profile_conditions_lib.InitialPsiMode.PROFILE_CONDITIONS,
        **profile_overrides,
    )
    runtime_params = dataclasses.replace(
        runtime_params,
        profile_conditions=profile_conditions,
    )
    sim_state = initial_state_lib._get_initial_state(  # noqa: SLF001
        runtime_params=runtime_params,
        geo=geometry,
        step_fn=step_fn,
    )

    energy = sim_state.core_profiles.internal_plasma_energy
    pedestal_state = sim_state.pedestal_transition_state
    if energy is None or pedestal_state is None:
        raise ValueError("TORAX initialization produced incomplete state")
    energy_history = {
        name: jnp.asarray(
            getattr(snapshot, name),
            dtype=getattr(energy, name).dtype,
        )
        for name in _ENERGY_HISTORY_FIELDS
    }
    core_profiles = dataclasses.replace(
        sim_state.core_profiles,
        internal_plasma_energy=dataclasses.replace(
            energy,
            dW_thermal_i_dt=jnp.zeros_like(energy.dW_thermal_i_dt),
            dW_thermal_e_dt=jnp.zeros_like(energy.dW_thermal_e_dt),
            **energy_history,
        ),
    )
    pedestal_state = dataclasses.replace(
        pedestal_state,
        confinement_mode=jnp.asarray(
            snapshot.confinement_mode,
            dtype=jax_utils.get_int_dtype(),
        ),
    )
    models = step_fn.solver.models
    pedestal_output = models.pedestal_model(
        runtime_params,
        geometry,
        core_profiles,
        sim_state.core_sources,
        pedestal_state,
    )
    pedestal_state = dataclasses.replace(
        pedestal_state,
        pedestal_model_output=pedestal_output,
        previous_pedestal_model_output=pedestal_output,
    )
    core_transport = transport_coefficients_builder.calculate_all_transport_coeffs(
        models.transport_model,
        models.neoclassical_models,
        runtime_params,
        geometry,
        core_profiles,
        pedestal_transition_state=pedestal_state,
    )
    sim_state = dataclasses.replace(
        sim_state,
        core_profiles=core_profiles,
        pedestal_transition_state=pedestal_state,
        core_transport=core_transport,
    )
    post_processed_outputs = post_processing.make_post_processed_outputs(
        sim_state=sim_state,
        runtime_params=runtime_params,
        previous_post_processed_outputs=post_processing.PostProcessedOutputs.zeros(
            geometry
        ),
    )
    return sim_state, post_processed_outputs


__all__ = [
    "PhaseSnapshot",
    "SnapshotMetadata",
    "load_snapshot",
    "snapshot_from_initialization",
    "initialization_from_snapshot",
    "rebuild_state_from_snapshot",
    "snapshot_sha256",
    "snapshot_from_state",
    "write_snapshot",
]
