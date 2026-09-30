"""Discharge exports contain every profile/grid needed for offline rendering."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import rollout_discharge as discharge


def test_discharge_export_preserves_full_profiles_and_radial_grids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = np.arange(24, dtype=float).reshape(2, 3, 4)
    q_face = np.arange(30, dtype=float).reshape(2, 3, 5)
    trajectory = SimpleNamespace(
        env_state=SimpleNamespace(
            plasma=SimpleNamespace(
                core=SimpleNamespace(
                    T_e=SimpleNamespace(value=profile),
                    T_i=SimpleNamespace(value=profile + 1),
                    n_e=SimpleNamespace(value=profile + 2),
                    q_face=q_face,
                )
            )
        )
    )
    valid = np.asarray([[True, True, False], [True, False, False]])
    monkeypatch.setattr(discharge, "trajectory_arrays", lambda *args: {"valid": valid})
    monkeypatch.setattr(
        discharge, "environment_interface", lambda env: {"action_shape": [1]}
    )
    rho = np.linspace(0.1, 0.9, 4)
    rho_face = np.linspace(0.0, 1.0, 5)
    path = tmp_path / "discharge.npz"
    discharge._save_discharge_trajectories(path, trajectory, object(), rho, rho_face)
    with np.load(path, allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["T_e"], profile)
        np.testing.assert_array_equal(saved["T_i"], profile + 1)
        np.testing.assert_array_equal(saved["n_e"], profile + 2)
        np.testing.assert_array_equal(saved["q_face"], q_face)
        np.testing.assert_array_equal(saved["rho_norm"], rho)
        np.testing.assert_array_equal(saved["rho_face_norm"], rho_face)
        np.testing.assert_array_equal(saved["valid"], valid)
