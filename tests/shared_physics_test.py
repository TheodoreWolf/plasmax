"""Shared-physics ownership and resolved backend invariants."""

from __future__ import annotations

import functools
import math
from pathlib import Path
from typing import Any

import pytest
from torax._src.torax_pydantic import model_config

from plasmax.environment.config import parse_env_and_backend
from plasmax.environment.merge import (
    _apply_imas_init,
    _load_extended_yaml,
    _resolve_geometry_dir,
    valid_env_backend_combos,
)
from plasmax.environment.registry import resolve_backend, resolve_env
from plasmax.environment.schema import ScenarioConfig

_CONVENTIONAL_ENVS = tuple(
    env
    for env in sorted(valid_env_backend_combos())
    if env.startswith(("iter/", "sparc/"))
)
_TORAX_BACKENDS = tuple(
    sorted(
        {
            backend
            for env, backends in valid_env_backend_combos().items()
            if env != "kstar"
            for backend in backends
        }
    )
)
_SHARED_CONVENTIONAL_RANDOMIZATION = {
    "neoclassical.bootstrap_current.bootstrap_multiplier": (0.95, 1.05),
    "pedestal.T_e_ped": (0.8, 1.2),
    "pedestal.T_i_ped": (0.8, 1.2),
    "pedestal.formation_model.P_LH_prefactor": (0.8, 1.25),
    "pedestal.n_e_ped": (0.8, 1.2),
    "sources.impurity_radiation.radiation_multiplier": (0.9, 1.1),
}


@functools.cache
def _scenario(env: str, backend: str) -> ScenarioConfig:
    return parse_env_and_backend(env, backend)


@functools.cache
def _effective_torax(env: str, backend: str) -> dict[str, Any]:
    """Build TORAX's validated model so omitted upstream defaults are visible."""
    env_path = resolve_env(env)
    torax = dict(_scenario(env, backend).torax)
    torax = _apply_imas_init(torax, env_path)
    torax = _resolve_geometry_dir(torax, env_path)
    return model_config.ToraxConfig.from_dict(torax).model_dump(mode="json")


def _transport_randomization(config: ScenarioConfig) -> dict[str, Any]:
    return {
        key: spec.model_dump(mode="json")
        for key, spec in config.physics_randomization.items()
        if key.startswith("transport_model.")
    }


def _canonicalize(value: Any) -> Any:
    """Make structural signatures comparable when TORAX carries NaN sentinels."""
    if isinstance(value, dict):
        return {key: _canonicalize(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_canonicalize(child) for child in value]
    if isinstance(value, float) and math.isnan(value):
        return "<nan>"
    return value


def _non_transport_signature(env: str, backend: str) -> dict[str, Any]:
    """Canonical resolved config with only the permitted axes removed."""
    config = _scenario(env, backend)
    signature = config.model_dump(mode="json")

    torax = _effective_torax(env, backend).copy()
    torax.pop("transport")
    torax.pop("solver")
    signature["torax"] = torax

    stepping = dict(signature["stepping"])
    stepping.pop("max_solver_substeps")
    signature["stepping"] = stepping
    signature["physics_randomization"] = {
        key: value
        for key, value in signature["physics_randomization"].items()
        if not key.startswith("transport_model.")
    }
    return _canonicalize(signature)


def _assert_canonical_common_physics(env: str, backend: str) -> None:
    config = _scenario(env, backend)
    torax = _effective_torax(env, backend)

    bootstrap = torax["neoclassical"]["bootstrap_current"]
    assert bootstrap["model_name"] == "redl"
    assert bootstrap["bootstrap_multiplier"] == 1.0
    assert torax["neoclassical"]["conductivity"]["model_name"] == "sauter"
    neo_transport = torax["neoclassical"]["transport"]
    assert neo_transport["model_name"] == "angioni_sauter"
    assert neo_transport["use_shaing_ion_correction"] is True

    pedestal = torax["pedestal"]
    assert pedestal["model_name"] == "set_T_ped_n_ped"
    assert pedestal["mode"] == "ADAPTIVE_TRANSPORT"
    assert pedestal["formation_model"]["model_name"] == "martin_scaling"
    assert torax["numerics"]["max_dt"] == 0.1
    assert torax["numerics"]["chi_timestep_prefactor"] == 50
    assert torax["numerics"]["dt_reduction_factor"] == 3
    assert torax["time_step_calculator"]["calculator_type"] == "fixed"

    common_randomization = {
        key: spec.relative
        for key, spec in config.physics_randomization.items()
        if not key.startswith("transport_model.")
    }
    if env.startswith(("iter/", "sparc/")):
        assert common_randomization == _SHARED_CONVENTIONAL_RANDOMIZATION


def test_packaged_torax_backends_own_only_transport_and_solver() -> None:
    assert _TORAX_BACKENDS == (
        "bohm_gyrobohm",
        "cgm",
        "qlknn",
        "tglfnn",
        "tglfnn_nr",
        "tglfnn_spherical",
    )
    for backend in _TORAX_BACKENDS:
        raw = _load_extended_yaml(resolve_backend(backend))
        assert set(raw) <= {"torax", "physics_randomization", "stepping"}, backend
        assert set(raw.get("torax") or {}) <= {"transport", "solver"}, backend
        assert all(
            key.startswith("transport_model.")
            for key in (raw.get("physics_randomization") or {})
        ), backend
        assert set(raw.get("stepping") or {}) <= {"max_solver_substeps"}, backend


def test_conventional_matrix_differs_only_by_transport_and_solver() -> None:
    assert len(_CONVENTIONAL_ENVS) == 15
    assert sum(len(valid_env_backend_combos()[env]) for env in _CONVENTIONAL_ENVS) == 75

    for env in _CONVENTIONAL_ENVS:
        backends = sorted(valid_env_backend_combos()[env])
        reference_backend = backends[0]
        reference = _non_transport_signature(env, reference_backend)
        for backend in backends:
            assert _non_transport_signature(env, backend) == reference, (
                env,
                reference_backend,
                backend,
            )
            _assert_canonical_common_physics(env, backend)


def test_tglfnn_solver_variants_share_transport_and_uncertainty() -> None:
    for env in _CONVENTIONAL_ENVS:
        linear = _scenario(env, "tglfnn")
        nonlinear = _scenario(env, "tglfnn_nr")
        assert linear.torax["transport"] == nonlinear.torax["transport"], env
        assert _transport_randomization(linear) == _transport_randomization(
            nonlinear
        ), env

        linear_signature = linear.model_dump(mode="json")
        nonlinear_signature = nonlinear.model_dump(mode="json")
        for signature in (linear_signature, nonlinear_signature):
            signature["torax"] = dict(signature["torax"])
            signature["torax"].pop("solver")
            signature["stepping"] = dict(signature["stepping"])
            signature["stepping"].pop("max_solver_substeps")
        assert linear_signature == nonlinear_signature, env
        assert linear.torax["solver"]["solver_type"] == "linear"
        assert nonlinear.torax["solver"]["solver_type"] == "newton_raphson"


def test_step_backends_share_explicit_step_physics() -> None:
    bg_b = _scenario("step", "bohm_gyrobohm")
    tglf = _scenario("step", "tglfnn_spherical")
    assert _non_transport_signature("step", "bohm_gyrobohm") == (
        _non_transport_signature("step", "tglfnn_spherical")
    )
    for backend in ("bohm_gyrobohm", "tglfnn_spherical"):
        _assert_canonical_common_physics("step", backend)

    pedestal = bg_b.torax["pedestal"]
    assert pedestal["rho_norm_ped_top"] == 0.95
    assert pedestal["T_i_ped"] == 4.0
    assert pedestal["T_e_ped"] == 5.0
    assert pedestal["n_e_ped"] == 6.0e19
    assert bg_b.torax["sources"]["impurity_radiation"] == {
        "model_name": "P_in_scaled_flat_profile",
        "fraction_P_heating": 0.7,
    }
    assert "pedestal.formation_model.P_LH_prefactor" not in (bg_b.physics_randomization)
    common_randomization = {
        key: spec.relative
        for key, spec in bg_b.physics_randomization.items()
        if not key.startswith("transport_model.")
    }
    assert common_randomization == {
        "neoclassical.bootstrap_current.bootstrap_multiplier": (0.95, 1.05),
        "pedestal.T_e_ped": (0.8, 1.2),
        "pedestal.T_i_ped": (0.8, 1.2),
        "pedestal.n_e_ped": (0.8, 1.2),
        "sources.impurity_radiation.fraction_P_heating": (0.9, 1.1),
    }
    assert _transport_randomization(bg_b) != _transport_randomization(tglf)


def test_registered_physical_tasks_never_accelerate_resistivity() -> None:
    """Registered physical tasks use and retain the physical resistivity."""
    for env, backends in valid_env_backend_combos().items():
        if env == "kstar":
            continue
        for backend in backends:
            scenario = _scenario(env, backend)
            nominal = float(scenario.torax["numerics"]["resistivity_multiplier"])
            assert nominal == 1.0, (env, backend, nominal)
            assert "numerics.resistivity_multiplier" not in (
                scenario.physics_randomization
            ), (env, backend)


def test_extended_yaml_inherits_and_leaf_overrides(tmp_path: Path) -> None:
    (tmp_path / "base.yaml").write_text(
        "torax:\n"
        "  numerics: {max_dt: 2.0, fixed_dt: 0.1}\n"
        "  neoclassical: {conductivity: {model_name: sauter}}\n"
    )
    child = tmp_path / "child.yaml"
    child.write_text(
        "extends: base.yaml\n"
        "torax:\n"
        "  numerics: {max_dt: 0.5}\n"
        "  geometry: {geometry_type: circular}\n"
    )

    resolved = _load_extended_yaml(child)
    assert resolved["torax"]["numerics"] == {"max_dt": 0.5, "fixed_dt": 0.1}
    assert resolved["torax"]["neoclassical"] == {
        "conductivity": {"model_name": "sauter"}
    }
    assert resolved["torax"]["geometry"] == {"geometry_type": "circular"}


def test_extended_yaml_rejects_cycles(tmp_path: Path) -> None:
    first = tmp_path / "first.yaml"
    second = tmp_path / "second.yaml"
    first.write_text("extends: second.yaml\n")
    second.write_text("extends: first.yaml\n")

    with pytest.raises(ValueError, match="cyclic YAML extends chain"):
        _load_extended_yaml(first)


def test_extended_yaml_rejects_path_escape(tmp_path: Path) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    (tmp_path / "outside.yaml").write_text("torax: {}\n")
    child = allowed / "child.yaml"
    child.write_text("extends: ../outside.yaml\n")

    with pytest.raises(ValueError, match="escapes"):
        _load_extended_yaml(child)
