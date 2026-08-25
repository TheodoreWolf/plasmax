"""Compact phase-snapshot schema and TORAX reset adapter."""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.metadata
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

import numpy as np
from jax import numpy as jnp
from torax._src import jax_utils
from torax._src.config import build_runtime_params
from torax._src.core_profiles import profile_conditions as profile_conditions_lib
from torax._src.orchestration import initial_state as initial_state_lib
from torax._src.orchestration import sim_state as sim_state_lib
from torax._src.orchestration.step_function import SimulationStepFn
from torax._src.output_tools import post_processing
from torax._src.transport_model import transport_coefficients_builder


@dataclasses.dataclass(frozen=True)
class PhaseSnapshot:
    """Backend-agnostic physical state stored by a phase NPZ."""

    schema_version: int
    environment: str
    source_backend: str
    source_step: int
    source_time_s: float
    seed: int
    source_config_sha256: str
    torax_version: str
    rho_norm: np.ndarray
    rho_face_norm: np.ndarray
    T_i: np.ndarray
    T_e: np.ndarray
    n_e: np.ndarray
    psi: np.ndarray
    dW_thermal_i_dt_smoothed: float
    dW_thermal_e_dt_smoothed: float
    confinement_mode: int


@dataclasses.dataclass(frozen=True)
class _ArrayField:
    dtype_kinds: str
    shape: Literal["scalar", "rho", "face"]
    finite: bool = False
    value: object | None = None
    metadata: Mapping[str, type | tuple[type, ...]] | None = None


_NPZ_SCHEMA = {
    "schema_version": _ArrayField("iu", "scalar", value=1),
    "rho_norm": _ArrayField("f", "rho", finite=True),
    "rho_face_norm": _ArrayField("f", "face", finite=True),
    "T_i": _ArrayField("f", "rho", finite=True),
    "T_e": _ArrayField("f", "rho", finite=True),
    "n_e": _ArrayField("f", "rho", finite=True),
    "psi": _ArrayField("f", "rho", finite=True),
    "dW_thermal_i_dt_smoothed": _ArrayField("f", "scalar", finite=True),
    "dW_thermal_e_dt_smoothed": _ArrayField("f", "scalar", finite=True),
    "confinement_mode": _ArrayField("iu", "scalar"),
    "metadata_json": _ArrayField(
        "US",
        "scalar",
        metadata={
            "environment": str,
            "source_backend": str,
            "source_step": int,
            "source_time_s": (int, float),
            "seed": int,
            "source_config_sha256": str,
            "torax_version": str,
        },
    ),
}


def _validate_snapshot(
    arrays: Mapping[str, np.ndarray],
    expected_environment: str | None,
) -> PhaseSnapshot:
    """Apply the complete NPZ schema once and return its typed value."""
    keys = set(arrays)
    expected_keys = set(_NPZ_SCHEMA)
    if len(arrays) != len(_NPZ_SCHEMA) or keys != expected_keys:
        raise ValueError(
            "snapshot fields differ; "
            f"missing={sorted(expected_keys - keys)}, "
            f"extra={sorted(keys - expected_keys)}"
        )

    rho_norm = arrays["rho_norm"]
    if rho_norm.ndim != 1:
        raise ValueError(
            f"snapshot rho_norm must be one-dimensional, got {rho_norm.shape}"
        )
    n_rho = rho_norm.shape[0]
    shapes = {"scalar": (), "rho": (n_rho,), "face": (n_rho + 1,)}
    metadata: dict[str, object] = {}
    for name, field in _NPZ_SCHEMA.items():
        array = arrays[name]
        if array.shape != shapes[field.shape]:
            raise ValueError(
                f"snapshot {name} has shape {array.shape}; "
                f"expected {shapes[field.shape]}"
            )
        if array.dtype.kind not in field.dtype_kinds:
            raise ValueError(f"snapshot {name} has incompatible dtype {array.dtype}")
        if field.finite and not np.all(np.isfinite(array)):
            raise ValueError(f"snapshot {name} must be finite")
        if field.value is not None and array.item() != field.value:
            raise ValueError(
                f"unsupported phase snapshot schema {array.item()}; "
                f"expected {field.value}"
            )
        if field.metadata is not None:
            encoded = array.item()
            if isinstance(encoded, bytes):
                encoded = encoded.decode("utf-8")
            try:
                decoded = json.loads(encoded)
            except (json.JSONDecodeError, TypeError) as error:
                raise ValueError("snapshot metadata_json is not valid JSON") from error
            if not isinstance(decoded, dict):
                raise ValueError("snapshot metadata_json must encode an object")
            if set(decoded) != set(field.metadata) or any(
                not isinstance(decoded[key], expected_type)
                for key, expected_type in field.metadata.items()
            ):
                raise ValueError("snapshot metadata_json does not match its schema")
            metadata = decoded

    if (
        expected_environment is not None
        and metadata["environment"] != expected_environment
    ):
        raise ValueError(
            f"snapshot environment {metadata['environment']!r} does not match "
            f"{expected_environment!r}"
        )
    values = {
        name: array.item() if array.shape == () else array
        for name, array in arrays.items()
        if name != "metadata_json"
    }
    return PhaseSnapshot(**values, **metadata)


def snapshot_sha256(path: str | Path) -> str:
    """Return the checksum of the exact NPZ bytes."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_snapshot(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
    expected_environment: str | None = None,
) -> PhaseSnapshot:
    """Load an NPZ and validate it once against the phase-snapshot schema."""
    source = Path(path)
    if expected_sha256 is not None:
        actual_sha256 = snapshot_sha256(source)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"snapshot checksum differs: expected {expected_sha256}, "
                f"got {actual_sha256}"
            )
    with np.load(source, allow_pickle=False) as archive:
        arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    return _validate_snapshot(arrays, expected_environment)


def write_snapshot(snapshot: PhaseSnapshot, path: str | Path) -> str:
    """Atomically write a compressed snapshot for the capture tool."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        name: np.asarray(getattr(snapshot, name))
        for name in _NPZ_SCHEMA
        if name != "metadata_json"
    }
    payload["schema_version"] = np.asarray(snapshot.schema_version, dtype=np.int32)
    payload["confinement_mode"] = np.asarray(
        snapshot.confinement_mode,
        dtype=np.int32,
    )
    payload["metadata_json"] = np.asarray(
        json.dumps(
            {
                name: getattr(snapshot, name)
                for name in _NPZ_SCHEMA["metadata_json"].metadata or ()
            },
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    _validate_snapshot(payload, snapshot.environment)

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            np.savez_compressed(stream, **payload)
            stream.flush()
            os.fsync(stream.fileno())
        temporary_path.replace(destination)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return snapshot_sha256(destination)


def capture_snapshot(
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
    return PhaseSnapshot(
        schema_version=int(_NPZ_SCHEMA["schema_version"].value),
        environment=environment,
        source_backend=source_backend,
        source_step=source_step,
        source_time_s=float(np.asarray(sim_state.t)),
        seed=seed,
        source_config_sha256=source_config_sha256,
        torax_version=importlib.metadata.version("torax"),
        rho_norm=np.asarray(sim_state.geometry.rho_norm),
        rho_face_norm=np.asarray(sim_state.geometry.rho_face_norm),
        T_i=np.asarray(sim_state.core_profiles.T_i.value),
        T_e=np.asarray(sim_state.core_profiles.T_e.value),
        n_e=np.asarray(sim_state.core_profiles.n_e.value),
        psi=np.asarray(sim_state.core_profiles.psi.value),
        dW_thermal_i_dt_smoothed=float(np.asarray(energy.dW_thermal_i_dt_smoothed)),
        dW_thermal_e_dt_smoothed=float(np.asarray(energy.dW_thermal_e_dt_smoothed)),
        confinement_mode=int(np.asarray(pedestal_state.confinement_mode)),
    )


def materialize_snapshot(
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

    profile_conditions = dataclasses.replace(
        runtime_params.profile_conditions,
        T_i=jnp.asarray(snapshot.T_i, dtype=jax_utils.get_dtype()),
        T_e=jnp.asarray(snapshot.T_e, dtype=jax_utils.get_dtype()),
        n_e=jnp.asarray(snapshot.n_e, dtype=jax_utils.get_dtype()),
        psi=jnp.asarray(snapshot.psi, dtype=jax_utils.get_dtype()),
        n_e_nbar_is_fGW=False,
        normalize_n_e_to_nbar=False,
        initial_psi_from_j=False,
        initial_psi_mode=profile_conditions_lib.InitialPsiMode.PROFILE_CONDITIONS,
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
    core_profiles = dataclasses.replace(
        sim_state.core_profiles,
        internal_plasma_energy=dataclasses.replace(
            energy,
            dW_thermal_i_dt=jnp.zeros_like(energy.dW_thermal_i_dt),
            dW_thermal_e_dt=jnp.zeros_like(energy.dW_thermal_e_dt),
            dW_thermal_i_dt_smoothed=jnp.asarray(
                snapshot.dW_thermal_i_dt_smoothed,
                dtype=energy.dW_thermal_i_dt_smoothed.dtype,
            ),
            dW_thermal_e_dt_smoothed=jnp.asarray(
                snapshot.dW_thermal_e_dt_smoothed,
                dtype=energy.dW_thermal_e_dt_smoothed.dtype,
            ),
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
    "capture_snapshot",
    "load_snapshot",
    "materialize_snapshot",
    "snapshot_sha256",
    "write_snapshot",
]
