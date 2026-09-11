"""Contracts for resolved YAML initialization snapshots."""

from __future__ import annotations

from pathlib import Path

import jax
import numpy as np
import pytest
import yaml
from helpers import make_test_config, make_test_env

from plasmax import make
from plasmax.environment.config import parse_env_and_backend
from plasmax.environment.initialization import (
    PhaseSnapshot,
    load_snapshot,
    snapshot_from_state,
    write_snapshot,
)
from plasmax.environment.initialization_data import (
    ToraxInitialization,
    load_initialization,
    round_significant,
    write_initialization,
)
from plasmax.environment.registry import CONFIGS_DIR
from plasmax.environment.schema import PlasmaxConfig
from plasmax.wrappers import OracleWrappers

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


def test_snapshot_write_load_round_trip(
    tmp_path: Path, captured_snapshot: PhaseSnapshot
) -> None:
    path = tmp_path / "state.yaml"
    write_snapshot(captured_snapshot, path)
    restored = load_snapshot(path)
    for name in _ENCODED_STATE_FIELDS:
        np.testing.assert_array_equal(
            getattr(restored, name),
            round_significant(getattr(captured_snapshot, name)),
        )
    document = load_initialization(path)
    assert document.provenance.source_step == captured_snapshot.metadata.source_step
    second = tmp_path / "second.yaml"
    write_initialization(document, second)
    assert path.read_bytes() == second.read_bytes()


def test_numeric_yaml_and_four_significant_figures(
    tmp_path: Path, captured_snapshot: PhaseSnapshot
) -> None:
    values = np.array([0.0, -1.234567, 1.234567e20, 1.234567e-12, 9.99999])
    rounded = round_significant(values)
    np.testing.assert_array_equal(rounded, [0.0, -1.235, 1.235e20, 1.235e-12, 10.0])
    np.testing.assert_allclose(rounded, values, rtol=5e-4, atol=0.0)
    path = tmp_path / "state.yaml"
    write_snapshot(captured_snapshot, path)
    raw = yaml.safe_load(path.read_text())
    for values in raw["profiles"].values():
        assert all(isinstance(value, int | float) for value in values)
    assert raw["provenance"]["source_config_sha256"] == "0" * 64


def test_scientific_literals_zero_and_integer_metadata(tmp_path: Path) -> None:
    from plasmax.environment.initialization import initialization_from_snapshot

    snapshot = load_snapshot(
        CONFIGS_DIR / "data/initializations/mock/circular/nominal.yaml"
    )
    numbers = np.array([1e20, -1e-20, 0.0, -0.0, 2021.0])
    snapshot = snapshot.model_copy(
        update={"T_i": np.resize(numbers, snapshot.T_i.shape)}
    )
    doc = initialization_from_snapshot(snapshot)
    path = tmp_path / "numeric.yaml"
    write_initialization(doc, path)
    raw = yaml.safe_load(path.read_text())
    np.testing.assert_array_equal(raw["profiles"]["T_i_keV"], snapshot.T_i)
    assert isinstance(raw["provenance"]["source_step"], int)
    assert "1.0e+20" in path.read_text()
    assert "2.021e+03" in path.read_text()
    assert "-0.0" not in path.read_text()


def test_rounding_bound_over_magnitudes() -> None:
    values = np.outer(
        np.array([-9.999999, -1.0004999, 1.0004999, 4.56789, 9.999999]),
        10.0 ** np.arange(-25, 26),
    )
    rounded = np.asarray(round_significant(values))
    quantum = 10.0 ** (np.floor(np.log10(np.abs(values))) - 3)
    assert np.all(np.abs(rounded - values) <= 0.500001 * quantum)


@pytest.mark.parametrize(
    "change, message",
    [
        ({"schema_version": 2}, "schema_version"),
        ({"kind": "unknown"}, "kind"),
        ({"profiles": {"T_i_keV": [1.0]}}, "Field required"),
        ({"unknown": 1}, "Extra inputs"),
    ],
)
def test_rejects_malformed_yaml(
    tmp_path: Path, captured_snapshot: PhaseSnapshot, change, message
) -> None:
    path = tmp_path / "state.yaml"
    write_snapshot(captured_snapshot, path)
    data = yaml.safe_load(path.read_text())
    data.update(change)
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match=message):
        load_initialization(path)


def test_rejects_wrong_profile_shape_and_kind(
    tmp_path: Path, captured_snapshot: PhaseSnapshot
) -> None:
    path = tmp_path / "state.yaml"
    write_snapshot(captured_snapshot, path)
    with pytest.raises(ValueError, match="expected kstar"):
        load_initialization(path, kind="kstar")
    data = yaml.safe_load(path.read_text())
    data["profiles"]["T_i_keV"].pop()
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match="T_i_keV shape"):
        load_initialization(path)
    with pytest.raises(FileNotFoundError):
        load_initialization(tmp_path / "missing.yaml")
    with pytest.raises(ValueError, match="YAML"):
        load_initialization(tmp_path / "state.npz")


@pytest.mark.parametrize(
    "field, value, message",
    [
        ("input_order", ["Ip"] * 15, "15 named inputs"),
        ("history_row", [0.0] * 20, "21 columns"),
        ("history_columns", ["Ip"] * 21, "21 columns"),
        ("history_length", 9, "history_length"),
        ("targets", {}, "betap, q95, li targets"),
    ],
)
def test_rejects_malformed_kstar_state(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    source = CONFIGS_DIR / "data/initializations/kstar/nominal.yaml"
    payload = yaml.safe_load(source.read_text())
    payload[field] = value
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(payload))
    with pytest.raises(ValueError, match=message):
        load_initialization(path, kind="kstar")


def test_snapshot_schema_does_not_add_physical_gates(
    tmp_path: Path, captured_snapshot: PhaseSnapshot
) -> None:
    values = captured_snapshot.T_i.copy()
    values[0] = -1.0
    path = tmp_path / "state.yaml"
    write_snapshot(captured_snapshot.model_copy(update={"T_i": values}), path)
    assert load_snapshot(path).T_i[0] == -1.0


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
    config = parse_env_and_backend("iter/hybrid/rampup", "cgm")
    assert isinstance(config, PlasmaxConfig)
    assert config.initialization.suffix == ".yaml"

    def unexpected_rebuild(*_args, **_kwargs):
        raise AssertionError("native reset attempted a snapshot rebuild")

    monkeypatch.setattr(
        "plasmax.environment.initialization.rebuild_state_from_snapshot",
        unexpected_rebuild,
    )
    env = make_test_env()
    state, _ = env.init(jax.random.key(0))
    assert float(state.plasma.t) == pytest.approx(0.0)


def test_phase_snapshot_is_resolved_on_final_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    config = parse_env_and_backend("iter/hybrid/flattop", "cgm")
    assert isinstance(config, PlasmaxConfig)
    path = config.initialization
    assert (
        path
        == (CONFIGS_DIR / "data/initializations/iter/hybrid/settled.yaml").resolve()
    )
    snapshot = load_snapshot(path)
    assert snapshot.metadata.source_backend == "bohm_gyrobohm"
    assert snapshot.metadata.source_step == 1000
    assert snapshot.metadata.source_time_s == pytest.approx(100.0)


@pytest.mark.parametrize(
    "scenario, axis",
    [
        ("iter/baseline", 6.0),
        ("iter/hybrid", 6.0),
        ("iter/advanced", 6.0),
        ("sparc/prd", 1.0),
        ("sparc/reduced_field", 1.0),
    ],
)
def test_rampup_profiles_are_entirely_cold(scenario: str, axis: float) -> None:
    config = parse_env_and_backend(f"{scenario}/rampup", "bohm_gyrobohm")
    state = load_initialization(config.initialization, kind="torax")
    assert isinstance(state, ToraxInitialization)
    rho = np.asarray(state.grid.rho_norm)
    expected = round_significant(axis + (0.1 - axis) * rho)
    for profile in ("T_i", "T_e"):
        np.testing.assert_allclose(
            getattr(state.profiles, f"{profile}_keV"),
            expected,
            rtol=1e-14,
            atol=1e-14,
        )
        np.testing.assert_allclose(
            getattr(config.torax.profile_conditions, profile).get_value(0.0),
            expected,
            rtol=1e-14,
            atol=1e-14,
        )


def test_all_tasks_explicitly_reference_yaml_initializations(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from plasmax.environment.registry import ENV_ALIASES

    monkeypatch.chdir(tmp_path)
    for env, task_path in ENV_ALIASES.items():
        task = yaml.safe_load(task_path.read_text())
        reference = task["initialization"].replace(
            "${DATA_DIR}", str(CONFIGS_DIR / "data")
        )
        doc = load_initialization(
            reference, kind="kstar" if env == "kstar_worldmodel" else "torax"
        )
        assert doc.schema_version == 1
        assert not {"T_i", "T_e", "n_e", "psi", "nbar"}.intersection(
            task.get("torax", {}).get("profile_conditions", {})
        )
        if env == "kstar_worldmodel":
            backend = None
        elif env.startswith("step/"):
            backend = "bohm_gyrobohm_step"
        elif env.startswith("mock/"):
            backend = "mock"
        else:
            backend = "bohm_gyrobohm"
        config = parse_env_and_backend(env, backend)
        assert config.initialization == Path(reference)
        assert config._initial_state == doc


def test_loader_requires_reference_and_resolves_relative_assets(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from plasmax.environment import config as config_lib
    from plasmax.environment.merge import _merge_env_and_backend

    raw = _merge_env_and_backend("mock/circular/smoke", "mock")
    monkeypatch.setattr(config_lib, "_merge_env_and_backend", lambda *_: raw.copy())
    monkeypatch.chdir(tmp_path)
    raw["initialization"] = "data/initializations/mock/circular/nominal.yaml"
    parsed = parse_env_and_backend("mock/circular/smoke", "mock")
    assert parsed.initialization == CONFIGS_DIR / raw["initialization"]
    del raw["initialization"]
    with pytest.raises(ValueError, match="requires an initialization YAML"):
        parse_env_and_backend("mock/circular/smoke", "mock")


def test_packaged_initializations_are_canonical_four_figure_yaml(
    tmp_path: Path,
) -> None:
    for source in (CONFIGS_DIR / "data/initializations").rglob("*.yaml"):
        document = load_initialization(source)
        destination = tmp_path / "roundtrip.yaml"
        write_initialization(document, destination)
        assert source.read_bytes() == destination.read_bytes(), source


@pytest.mark.integration
def test_packaged_snapshot_rebuilds_two_backends_and_jits_first_steps() -> None:
    states = {}
    for backend in ("bohm_gyrobohm", "cgm"):
        env = OracleWrappers(
            make("iter/hybrid/flattop", backend), max_steps=1
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
