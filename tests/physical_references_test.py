"""Structural contract for backend-independent physical reset references."""

from __future__ import annotations

import hashlib
from functools import cache
from pathlib import Path
from typing import Any

import jax
import numpy as np
import pytest

from plasmax.environment.config import parse_env_and_backend
from plasmax.environment.factory import make
from plasmax.environment.merge import (
    _merge_env_and_backend,
    valid_env_backend_combos,
)
from plasmax.environment.references import (
    load_reference_manifest,
    reset_reference_id,
    reset_reference_payload,
    reset_reference_sha256,
)
from plasmax.environment.registry import resolve_backend, resolve_env
from plasmax.wrappers import unwrap_to_env_state
from tools.calibration.improve_initial_conditions import (
    CELL_CENTRES,
    REFERENCE_DATA_DIR,
    fit_gaussian_moments,
    interpolate_profile,
    parse_sectioned_profile,
    reconstruct_psi_from_q,
)

CONFIGS_DIR = Path(__file__).parents[1] / "src" / "plasmax" / "configs"
PHYSICAL_ENVS = tuple(env for env in valid_env_backend_combos() if env != "kstar")
MULTIPHASE_SCENARIOS = (
    "iter/baseline",
    "iter/hybrid",
    "iter/advanced",
    "sparc/prd",
    "sparc/reduced_field",
)


@cache
def _config(env: str, backend: str = "bohm_gyrobohm"):
    return parse_env_and_backend(env, backend)


@cache
def _payload(env: str, backend: str = "bohm_gyrobohm") -> dict[str, Any]:
    return reset_reference_payload(env, backend)


def _profile_values(mapping: dict[str, float]) -> np.ndarray:
    return np.asarray(
        [mapping[key] for key in sorted(mapping, key=float)], dtype=np.float64
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _time_zero_geometry_file(env: str) -> str:
    geometry = _config(env).torax["geometry"]
    configurations = geometry.get("geometry_configs")
    if configurations:
        first = configurations[min(configurations, key=float)]
        return str(first["geometry_file"])
    return str(geometry["geometry_file"])


def test_manifest_covers_every_physical_environment_and_pins_artifacts() -> None:
    manifest = load_reference_manifest()
    used_references = {reset_reference_id(env) for env in PHYSICAL_ENVS}
    assert used_references == set(manifest.references)

    for env in PHYSICAL_ENVS:
        reference = manifest.references[reset_reference_id(env)]
        assert reset_reference_sha256(env) == reference.profile_sha256

    for reference in manifest.references.values():
        assert reference.locked_paths
        for relative_path, expected in reference.artifacts.items():
            assert _sha256(CONFIGS_DIR / relative_path) == expected
        for source in reference.sources:
            assert source.license is not None
            if source.redistribution == "vendored":
                assert source.local_path is not None
                assert _sha256(CONFIGS_DIR / source.local_path) == source.sha256
            else:
                assert source.local_path is None

    with pytest.raises(ValueError, match="no reset_reference ID"):
        reset_reference_id("kstar")


def test_reference_metadata_is_stripped_before_validation() -> None:
    for env in PHYSICAL_ENVS:
        for backend in valid_env_backend_combos()[env]:
            merged = _merge_env_and_backend(resolve_env(env), resolve_backend(backend))
            assert "reset_reference" not in merged
            _config(env, backend)


def test_nominal_profile_and_actuator_reset_is_identical_across_backends() -> None:
    for env in PHYSICAL_ENVS:
        backends = sorted(valid_env_backend_combos()[env])
        expected = _payload(env, backends[0])
        for backend in backends[1:]:
            assert _payload(env, backend) == expected


def test_flattop_and_rampdown_reuse_the_exact_hot_anchor() -> None:
    for scenario in MULTIPHASE_SCENARIOS:
        flattop = f"{scenario}/flattop"
        rampdown = f"{scenario}/rampdown"
        assert reset_reference_id(flattop) == reset_reference_id(rampdown)
        assert _payload(flattop) == _payload(rampdown)
        assert _time_zero_geometry_file(flattop) == _time_zero_geometry_file(rampdown)


def test_exact_profile_artifacts_reproduce_yaml_arrays() -> None:
    itpa = np.genfromtxt(
        REFERENCE_DATA_DIR / "iter_baseline_450s_profiles_25.csv",
        delimiter=",",
        names=True,
    )
    baseline = _payload("iter/baseline/flattop")["profile_conditions"]
    for condition, column in (
        ("T_i", "T_i_keV"),
        ("T_e", "T_e_keV"),
        ("n_e", "n_e_m3"),
        ("psi", "psi_Wb"),
    ):
        np.testing.assert_allclose(
            _profile_values(baseline[condition]), itpa[column], rtol=1e-7
        )

    raw = parse_sectioned_profile(REFERENCE_DATA_DIR / "sparc_prd_transp_20221013.txt")
    prd = _payload("sparc/prd/flattop")["profile_conditions"]
    expected = {
        "T_i": interpolate_profile(raw["rho"], raw["ti"]),
        "T_e": interpolate_profile(raw["rho"], raw["te"]),
        "n_e": interpolate_profile(raw["rho"], raw["ne"] * 1e19),
        "psi": interpolate_profile(raw["rho"], raw["polflux"] * 2.0 * np.pi),
    }
    for name, values in expected.items():
        np.testing.assert_allclose(
            _profile_values(prd[name]), values, rtol=2e-6, atol=0.0
        )


def test_digitized_profiles_and_q_reconstruction_are_deterministic() -> None:
    advanced = np.genfromtxt(
        REFERENCE_DATA_DIR / "iter_advanced_slide25_profiles_25.csv",
        delimiter=",",
        names=True,
    )
    advanced_reference = load_reference_manifest().references[
        "iter_advanced_hot_slide25"
    ]
    assert advanced_reference.extraction.digitization_tolerance_pixels == 2.0
    conditions = _payload("iter/advanced/flattop")["profile_conditions"]
    for condition, column in (
        ("T_i", "T_i_keV"),
        ("T_e", "T_e_keV"),
        ("n_e", "n_e_m3"),
    ):
        np.testing.assert_allclose(
            _profile_values(conditions[condition]), advanced[column], rtol=1e-7
        )
    psi = reconstruct_psi_from_q(
        advanced["rho"],
        advanced["q"],
        float(advanced_reference.targets["edge_poloidal_flux_Wb"]),
    )
    np.testing.assert_allclose(
        _profile_values(conditions["psi"]), psi, rtol=2e-7, atol=1e-8
    )

    h8 = np.genfromtxt(
        REFERENCE_DATA_DIR / "sparc_h8_figure16_profiles_25.csv",
        delimiter=",",
        names=True,
    )
    h8_reference = load_reference_manifest().references["sparc_h8_hot_figure16"]
    assert h8_reference.extraction.digitization_tolerance_pixels == 2.0
    h8_conditions = _payload("sparc/reduced_field/flattop")["profile_conditions"]
    for prefix, unit in (("T_i", "keV"), ("T_e", "keV"), ("n_e", "m3")):
        p10 = h8[f"{prefix}_p10_{unit}"]
        median = h8[f"{prefix}_median_{unit}"]
        p90 = h8[f"{prefix}_p90_{unit}"]
        assert np.all(p10 <= median)
        assert np.all(median <= p90)
        np.testing.assert_allclose(
            _profile_values(h8_conditions[prefix]), median, rtol=1e-7
        )


def test_itpa_gaussian_projection_parameters_match_declared_moments() -> None:
    reference = load_reference_manifest().references["iter_baseline_hot_450s"]
    config = _config("iter/baseline/flattop")
    geometry = config.torax["geometry"]
    assert geometry["geometry_file"] == "references/iter_baseline_450s.eqdsk"

    # A synthetic shaped volume measure proves the deterministic fitter does
    # not confuse Gaussian input parameters with physical dV-weighted moments.
    measure = 1.0 + 3.0 * CELL_CENTRES
    location, width = fit_gaussian_moments(
        CELL_CENTRES, measure, target_centroid=0.55, target_rms_width=0.18
    )
    profile = np.exp(-0.5 * np.square((CELL_CENTRES - location) / width))
    weights = profile * measure
    centroid = np.sum(CELL_CENTRES * weights) / np.sum(weights)
    rms_width = np.sqrt(
        np.sum(np.square(CELL_CENTRES - centroid) * weights) / np.sum(weights)
    )
    np.testing.assert_allclose((centroid, rms_width), (0.55, 0.18), atol=1e-12)

    sources = config.torax["sources"]
    projection_by_target = {
        item.torax_component: item for item in reference.projections
    }
    for component, source_name in (
        ("torax.sources.generic_heat", "generic_heat"),
        ("torax.sources.ecrh", "ecrh"),
        ("torax.sources.generic_particle", "generic_particle"),
        ("torax.sources.pellet", "pellet"),
    ):
        declared = projection_by_target[component].torax_parameters
        for key, expected in declared.items():
            if key in sources[source_name]:
                np.testing.assert_allclose(sources[source_name][key], expected)


def test_backend_context_changes_dynamics_but_not_the_reset() -> None:
    env = "iter/hybrid/flattop"
    signatures = {}
    for backend in sorted(valid_env_backend_combos()[env]):
        torax = _config(env, backend).torax
        signatures[backend] = (
            torax["transport"]["model_name"],
            torax["solver"]["solver_type"],
            torax["transport"].get("machine"),
        )
    assert len(set(signatures.values())) == len(signatures)
    payloads = [_payload(env, backend) for backend in signatures]
    assert all(payload == payloads[0] for payload in payloads[1:])


def test_reference_reset_is_jittable_and_vmappable() -> None:
    env = make("step", "bohm_gyrobohm", variant="oracle")
    keys = jax.random.split(jax.random.key(0), 2)
    states, info = jax.jit(jax.vmap(env.init))(keys)
    physical = unwrap_to_env_state(states)
    assert np.asarray(info.obs).shape == (2, *env.observation_space.shape)
    for values in (
        physical.plasma.T_i,
        physical.plasma.T_e,
        physical.plasma.n_e,
        physical.plasma.psi,
        physical.prev_action,
    ):
        array = np.asarray(values)
        np.testing.assert_array_equal(array[0], array[1])


def test_realistic_and_oracle_use_the_same_physical_reset_perturbation() -> None:
    oracle = make("step", "bohm_gyrobohm", variant="oracle")
    realistic = make("step", "bohm_gyrobohm", variant="realistic")
    key = jax.random.key(17)
    oracle_state = unwrap_to_env_state(oracle.init(key)[0])
    realistic_state = unwrap_to_env_state(realistic.init(key)[0])

    for name in ("T_i", "T_e", "n_e", "psi", "q"):
        np.testing.assert_array_equal(
            np.asarray(getattr(oracle_state.plasma, name)),
            np.asarray(getattr(realistic_state.plasma, name)),
        )
    np.testing.assert_array_equal(
        np.asarray(oracle_state.prev_action),
        np.asarray(realistic_state.prev_action),
    )
