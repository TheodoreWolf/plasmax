"""Contracts for phase-owned NPZ initialization snapshots."""

from __future__ import annotations

from pathlib import Path

import jax
import numpy as np
import pytest
import yaml
from helpers import make_test_config, make_test_env

from plasmax import make
from plasmax.environment.config import parse_env_and_backend, parse_torax_sources
from plasmax.environment.factory import _load_env
from plasmax.environment.initialization import (
    PhaseSnapshot,
    load_snapshot,
    snapshot_from_state,
    write_snapshot,
)
from plasmax.environment.merge import (
    _merge_env_and_backend,
    resolve_config_asset,
)
from plasmax.environment.registry import CONFIGS_DIR, resolve_backend, resolve_env
from plasmax.environment.schema import ScenarioConfig
from plasmax.spaces import ActuatorSpec

_SNAPSHOT_PEDESTAL = {
    "model_name": "set_T_ped_n_ped",
    "set_pedestal": False,
    "mode": "ADAPTIVE_TRANSPORT",
}
_ENCODED_STATE_FIELDS = (
    "rho_norm",
    "rho_face_norm",
    "T_i",
    "T_e",
    "n_e",
    "psi",
    "dW_thermal_i_dt_smoothed",
    "dW_thermal_e_dt_smoothed",
    "confinement_mode",
)


def _snapshot_test_config(**overrides):
    return make_test_config(pedestal=_SNAPSHOT_PEDESTAL, **overrides)


@pytest.fixture(scope="module")
def captured_snapshot() -> PhaseSnapshot:
    source_env = make_test_env(config=_snapshot_test_config())
    source_state, _ = source_env.init(jax.random.key(0))
    source_state, info = source_env.step(
        source_state,
        source_state.prev_action,
    )
    assert bool(info.control_step_complete)
    assert not bool(info.terminated)
    return snapshot_from_state(
        source_state.plasma.sim,
        environment="test",
        source_backend="constant",
        source_step=1,
        seed=0,
        source_config_sha256="0" * 64,
    )


def _read_archive(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.array(archive[name], copy=True) for name in archive.files}


def _write_archive(path: Path, arrays: dict[str, np.ndarray]) -> None:
    np.savez_compressed(path, **arrays)


def _valid_archive(
    tmp_path: Path,
    snapshot: PhaseSnapshot,
) -> tuple[Path, dict[str, np.ndarray]]:
    path = tmp_path / "valid.npz"
    write_snapshot(snapshot, path)
    return path, _read_archive(path)


def test_snapshot_write_load_round_trip(
    tmp_path: Path,
    captured_snapshot: PhaseSnapshot,
) -> None:
    path = tmp_path / "phase.npz"
    checksum = write_snapshot(captured_snapshot, path)

    restored = load_snapshot(
        path,
        expected_sha256=checksum,
        expected_environment="test",
    )

    for name in PhaseSnapshot.model_fields:
        expected = getattr(captured_snapshot, name)
        actual = getattr(restored, name)
        if isinstance(expected, np.ndarray):
            np.testing.assert_array_equal(actual, expected)
        else:
            assert actual == expected


def test_load_rejects_checksum_mismatch(
    tmp_path: Path,
    captured_snapshot: PhaseSnapshot,
) -> None:
    path = tmp_path / "phase.npz"
    write_snapshot(captured_snapshot, path)

    with pytest.raises(ValueError, match="snapshot checksum differs"):
        load_snapshot(path, expected_sha256="f" * 64)


def test_load_rejects_unsupported_schema(
    tmp_path: Path,
    captured_snapshot: PhaseSnapshot,
) -> None:
    _, arrays = _valid_archive(tmp_path, captured_snapshot)
    arrays["schema_version"] = np.asarray(2, dtype=np.int32)
    path = tmp_path / "wrong_schema.npz"
    _write_archive(path, arrays)

    with pytest.raises(ValueError, match="schema_version"):
        load_snapshot(path)


def test_load_rejects_missing_key(
    tmp_path: Path,
    captured_snapshot: PhaseSnapshot,
) -> None:
    _, arrays = _valid_archive(tmp_path, captured_snapshot)
    arrays.pop("psi")
    path = tmp_path / "missing_key.npz"
    _write_archive(path, arrays)

    with pytest.raises(ValueError, match="snapshot fields differ"):
        load_snapshot(path)


def test_load_rejects_object_array(
    tmp_path: Path,
    captured_snapshot: PhaseSnapshot,
) -> None:
    _, arrays = _valid_archive(tmp_path, captured_snapshot)
    arrays["T_i"] = np.asarray([object()], dtype=object)
    path = tmp_path / "object_array.npz"
    _write_archive(path, arrays)

    with pytest.raises(ValueError, match="Object arrays cannot be loaded"):
        load_snapshot(path)


def test_load_rejects_non_object_metadata(
    tmp_path: Path,
    captured_snapshot: PhaseSnapshot,
) -> None:
    _, arrays = _valid_archive(tmp_path, captured_snapshot)
    arrays["metadata_json"] = np.asarray("[]")
    path = tmp_path / "metadata_list.npz"
    _write_archive(path, arrays)

    with pytest.raises(ValueError, match="metadata_json must encode an object"):
        load_snapshot(path)


def test_load_rejects_profile_shape_mismatch(
    tmp_path: Path,
    captured_snapshot: PhaseSnapshot,
) -> None:
    _, arrays = _valid_archive(tmp_path, captured_snapshot)
    arrays["T_e"] = arrays["T_e"][:-1]
    path = tmp_path / "wrong_shape.npz"
    _write_archive(path, arrays)

    with pytest.raises(ValueError, match="snapshot T_e has shape"):
        load_snapshot(path)


def test_load_applies_no_physical_admissibility_gate(
    tmp_path: Path,
    captured_snapshot: PhaseSnapshot,
) -> None:
    _, arrays = _valid_archive(tmp_path, captured_snapshot)
    arrays["T_i"][0] = -1.0
    path = tmp_path / "mechanically_valid.npz"
    _write_archive(path, arrays)

    snapshot = load_snapshot(path)

    assert snapshot.T_i[0] == -1.0


def test_load_rejects_wrong_environment(
    tmp_path: Path,
    captured_snapshot: PhaseSnapshot,
) -> None:
    path = tmp_path / "phase.npz"
    write_snapshot(captured_snapshot, path)

    with pytest.raises(ValueError, match="does not match"):
        load_snapshot(path, expected_environment="iter/hybrid/flattop")


def test_phase_reset_rebases_clock_horizon_and_action_then_jits_first_step(
    captured_snapshot: PhaseSnapshot,
) -> None:
    phase_actions = [
        ActuatorSpec("P_nbi", low=1.0e6, high=30.0e6, init=7.0e6),
        ActuatorSpec(
            "gas_puff_rate",
            low=1.0e20,
            high=5.0e21,
            init=2.0e21,
        ),
    ]
    env = make_test_env(
        config=_snapshot_test_config(
            numerics={
                "t_initial": 1.25,
                "t_final": 1.45,
                "fixed_dt": 0.1,
            }
        ),
        initialization=captured_snapshot,
        actuator_specs=phase_actions,
    )
    state, _ = env.init(jax.random.key(1))

    assert captured_snapshot.metadata.source_time_s == pytest.approx(0.1)
    assert float(state.plasma.sim.t) == pytest.approx(1.25)
    assert env.safe_max_steps == 2
    np.testing.assert_array_equal(state.prev_action, [7.0e6, 2.0e21])
    np.testing.assert_array_equal(
        state.plasma.sim.core_profiles.T_i.value,
        captured_snapshot.T_i,
    )
    energy = state.plasma.sim.core_profiles.internal_plasma_energy
    assert energy is not None
    assert float(energy.dW_thermal_i_dt_smoothed) == pytest.approx(
        captured_snapshot.dW_thermal_i_dt_smoothed
    )
    assert float(energy.dW_thermal_e_dt_smoothed) == pytest.approx(
        captured_snapshot.dW_thermal_e_dt_smoothed
    )
    assert (
        int(state.plasma.sim.pedestal_transition_state.confinement_mode)
        == captured_snapshot.confinement_mode
    )
    for name in (
        "E_fusion",
        "E_aux_total",
        "E_ohmic_e",
        "E_external_injected",
        "E_external_total",
    ):
        assert float(getattr(state.plasma.post, name)) == 0.0

    next_state, info = jax.jit(env.step)(state, state.prev_action)
    jax.block_until_ready((next_state, info))
    assert bool(info.control_step_complete)
    assert not bool(info.terminated)
    assert float(next_state.plasma.sim.t) == pytest.approx(1.35)


def test_rebuild_then_projection_preserves_only_encoded_state_payload(
    captured_snapshot: PhaseSnapshot,
) -> None:
    destination_t_initial = 1.25
    env = make_test_env(
        config=_snapshot_test_config(
            numerics={
                "t_initial": destination_t_initial,
                "t_final": 1.45,
                "fixed_dt": 0.1,
            }
        ),
        initialization=captured_snapshot,
    )
    state, _ = env.init(jax.random.key(1))
    projected = snapshot_from_state(
        state.plasma.sim,
        environment=captured_snapshot.metadata.environment,
        source_backend="destination",
        source_step=0,
        seed=1,
        source_config_sha256="1" * 64,
    )

    for name in _ENCODED_STATE_FIELDS:
        expected = getattr(captured_snapshot, name)
        actual = getattr(projected, name)
        if isinstance(expected, np.ndarray):
            np.testing.assert_array_equal(actual, expected)
        else:
            assert actual == pytest.approx(expected)

    # Time and provenance are deliberately destination-owned, so this is a
    # projection/reconstruction invariant rather than a whole-state bijection.
    assert projected.metadata.source_time_s == pytest.approx(destination_t_initial)
    assert projected.metadata.source_time_s != pytest.approx(
        captured_snapshot.metadata.source_time_s
    )
    assert projected.metadata.source_backend == "destination"


def test_rebuild_rejects_grid_mismatch(
    captured_snapshot: PhaseSnapshot,
) -> None:
    snapshot = captured_snapshot.model_copy(
        update={"rho_norm": captured_snapshot.rho_norm + 0.001}
    )

    with pytest.raises(ValueError, match="does not match destination geometry"):
        make_test_env(
            config=_snapshot_test_config(),
            initialization=snapshot,
        )


def test_missing_snapshot_uses_native_reset(monkeypatch: pytest.MonkeyPatch) -> None:
    sources = parse_torax_sources(
        resolve_env("iter/hybrid/rampup"),
        resolve_backend("cgm"),
    )
    assert sources.initialization is None

    def unexpected_rebuild(*_args, **_kwargs):
        raise AssertionError("native reset attempted a snapshot rebuild")

    monkeypatch.setattr(
        "plasmax.environment.initialization.rebuild_state_from_snapshot",
        unexpected_rebuild,
    )
    env = make_test_env()
    state, _ = env.init(jax.random.key(0))
    assert float(state.plasma.t) == pytest.approx(0.0)


def test_phase_metadata_is_not_part_of_scenario_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    env_path = resolve_env("iter/hybrid/flattop")
    backend_path = resolve_backend("cgm")
    metadata = parse_torax_sources(env_path, backend_path).initialization

    assert metadata is not None
    assert set(metadata.model_dump()) == {"path", "sha256"}
    assert "initialization" not in _merge_env_and_backend(env_path, backend_path)
    assert "initialization" not in ScenarioConfig.model_fields
    assert (
        "initialization"
        not in parse_env_and_backend(
            env_path,
            backend_path,
        ).model_dump()
    )

    monkeypatch.chdir(tmp_path)
    path = resolve_config_asset(metadata.path, env_path)
    assert (
        path
        == (
            CONFIGS_DIR
            / "data"
            / "initializations"
            / "iter"
            / "hybrid"
            / "flattop_bgb_settled.npz"
        ).resolve()
    )
    snapshot = load_snapshot(
        path,
        expected_sha256=metadata.sha256,
        expected_environment="iter/hybrid/flattop",
    )
    assert snapshot.metadata.source_backend == "bohm_gyrobohm"
    assert snapshot.metadata.source_step == 1000
    assert snapshot.metadata.source_time_s == pytest.approx(100.0)


def test_initialization_metadata_is_phase_owned(tmp_path: Path) -> None:
    config_root = tmp_path / "configs"
    phase_path = config_root / "envs" / "device" / "scenario" / "flattop.yaml"
    base_path = phase_path.parent / "base.yaml"
    tokamak_path = config_root / "tokamaks" / "device.yaml"
    backend_path = tmp_path / "backend.yaml"
    initialization = {"path": "phase.npz", "sha256": "0" * 64}
    phase_path.parent.mkdir(parents=True)
    tokamak_path.parent.mkdir(parents=True)
    backend_path.write_text("{}\n")

    phase_path.write_text("{}\n")
    base_path.write_text(yaml.safe_dump({"initialization": initialization}))
    with pytest.raises(ValueError, match="scenario base layer"):
        _merge_env_and_backend(str(phase_path), str(backend_path))

    base_path.unlink()
    phase_path.write_text(yaml.safe_dump({"tokamak": "device"}))
    tokamak_path.write_text(yaml.safe_dump({"initialization": initialization}))
    with pytest.raises(ValueError, match="tokamak layer"):
        _merge_env_and_backend(str(phase_path), str(backend_path))

    tokamak_path.unlink()
    phase_path.write_text("initialization: null\n")
    backend_path.write_text("initialization: null\n")
    merged = _merge_env_and_backend(str(phase_path), str(backend_path))
    assert "initialization" not in merged

    backend_path.write_text(yaml.safe_dump({"initialization": initialization}))
    with pytest.raises(ValueError, match="unexpected top-level keys"):
        _merge_env_and_backend(str(phase_path), str(backend_path))


def test_world_model_yamls_reject_initialization(tmp_path: Path) -> None:
    initialization = {"path": "phase.npz", "sha256": "0" * 64}
    env_path = tmp_path / "world_env.yaml"
    backend_path = tmp_path / "world_backend.yaml"
    valid_env = {
        "task": {"reward": "kstar", "terminal_penalty": None},
        "world_model_env": {"max_steps_in_episode": 1},
    }
    valid_backend = {
        "type": "world_model",
        "world_model": {"name": "kstar_lstm"},
    }
    backend_path.write_text(yaml.safe_dump(valid_backend))
    env_path.write_text(yaml.safe_dump({**valid_env, "initialization": initialization}))

    with pytest.raises(ValueError, match="initialization"):
        _load_env(str(env_path), str(backend_path), validate=False)

    env_path.write_text(yaml.safe_dump(valid_env))
    backend_path.write_text(
        yaml.safe_dump(
            {
                **valid_backend,
                "initialization": initialization,
            }
        )
    )
    with pytest.raises(ValueError, match="initialization"):
        _load_env(str(env_path), str(backend_path), validate=False)

    base_path = tmp_path / "world_base.yaml"
    base_path.write_text(
        yaml.safe_dump(
            {
                **valid_backend,
                "initialization": initialization,
            }
        )
    )
    backend_path.write_text("extends: world_base.yaml\n")
    with pytest.raises(ValueError, match="initialization"):
        _load_env(str(env_path), str(backend_path), validate=False)


@pytest.mark.integration
def test_packaged_snapshot_rebuilds_two_backends_and_jits_first_steps() -> None:
    states = {}
    for backend in ("bohm_gyrobohm", "cgm"):
        env = make(
            "iter/hybrid/flattop",
            backend,
            variant="oracle",
            max_steps=1,
        ).unwrapped
        state = env._dynamics._initial_env_state
        assert float(state.plasma.t) == 0.0
        assert env.safe_max_steps == 4400
        next_state, info = jax.jit(env.step)(state, state.prev_action)
        jax.block_until_ready((next_state, info))
        assert bool(info.control_step_complete)
        states[backend] = state

    for name in ("T_i", "T_e", "n_e", "psi"):
        np.testing.assert_array_equal(
            getattr(states["bohm_gyrobohm"].plasma.core, name).value,
            getattr(states["cgm"].plasma.core, name).value,
        )
    assert not np.allclose(
        states["bohm_gyrobohm"].plasma.sim.core_transport.chi_face_ion,
        states["cgm"].plasma.sim.core_transport.chi_face_ion,
    )
