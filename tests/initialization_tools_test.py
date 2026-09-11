"""Initialization capture and calibration exports use the canonical YAML writer."""

from pathlib import Path

import numpy as np

from plasmax.environment.initialization import load_snapshot
from plasmax.environment.initialization_data import (
    ToraxInitialization,
    load_initialization,
    round_significant,
    write_initialization,
)
from plasmax.environment.registry import CONFIGS_DIR
from tools.calibration.improve_initial_conditions import write_itpa_initialization
from tools.generate_phase_initialization import Config, capture, main


def test_zero_step_capture_exports_the_selected_nominal_state(tmp_path: Path) -> None:
    path = tmp_path / "state.yaml"
    main(Config(environment="iter/hybrid/flattop", output=path))
    selected = CONFIGS_DIR / "data/initializations/iter/hybrid/settled.yaml"
    assert path.read_bytes() == selected.read_bytes()


def test_historical_npz_conversion_preserves_rounded_profiles_and_history(
    tmp_path: Path,
) -> None:
    source = CONFIGS_DIR / "data/initializations/iter/hybrid/flattop_bgb_settled.npz"
    path = tmp_path / "converted.yaml"
    main(Config(import_npz=source, output=path))
    restored = load_snapshot(path)
    with np.load(source, allow_pickle=False) as archive:
        for name in (
            "T_i",
            "T_e",
            "n_e",
            "psi",
            "dW_thermal_i_dt_smoothed",
            "dW_thermal_e_dt_smoothed",
            "confinement_mode",
        ):
            np.testing.assert_array_equal(
                getattr(restored, name), round_significant(archive[name])
            )
    assert restored.metadata.source_step == 1000
    assert restored.metadata.source_time_s == 100.0


def test_kstar_capture_reproduces_the_packaged_rounded_history(tmp_path: Path) -> None:
    document = capture(Config(environment="kstar_worldmodel"))
    path = tmp_path / "kstar.yaml"
    write_initialization(document, path)
    actual = load_initialization(path, kind="kstar")
    expected = load_initialization(
        CONFIGS_DIR / "data/initializations/kstar/nominal.yaml", kind="kstar"
    )
    assert actual.inputs == expected.inputs
    assert actual.history_row == expected.history_row
    assert actual.targets == expected.targets


def test_calibration_exports_complete_rounded_yaml(tmp_path: Path) -> None:
    source = np.genfromtxt(
        CONFIGS_DIR / "data/references/iter_baseline_450s_profiles_25.csv",
        delimiter=",",
        names=True,
    )
    columns = {
        "rho": "rho",
        "T_i": "T_i_keV",
        "T_e": "T_e_keV",
        "n_e": "n_e_m3",
        "psi": "psi_Wb",
        "q": "q",
    }
    profiles = {name: source[column] for name, column in columns.items()}
    path = tmp_path / "itpa.yaml"
    write_itpa_initialization(profiles, path)
    document = load_initialization(path, kind="torax")
    assert isinstance(document, ToraxInitialization)
    for field, column in (
        ("T_i_keV", "T_i"),
        ("T_e_keV", "T_e"),
        ("n_e_m3", "n_e"),
        ("psi_Wb", "psi"),
    ):
        np.testing.assert_array_equal(
            getattr(document.profiles, field), round_significant(profiles[column])
        )
    assert document.provenance.reference == "iter_baseline_hot_450s"
    assert set(tmp_path.iterdir()) == {path}
