"""Regressions for the total-plasma-current policy observation."""

import jax
import numpy as np
import pytest
from helpers import make_test_env

from plasmax.environment.merge import _merge_env_and_backend
from plasmax.environment.registry import resolve_backend, resolve_env
from plasmax.spaces import SCALAR_REGISTRY


def _merge(env: str) -> dict:
    return _merge_env_and_backend(
        resolve_env(env), resolve_backend("bohm_gyrobohm")
    )


def test_ip_registry_extracts_total_current_at_lcfs():
    state, _ = make_test_env().init(jax.random.key(0))

    np.testing.assert_array_equal(
        SCALAR_REGISTRY["Ip"](state.plasma),
        state.plasma.core.Ip_profile_face[-1],
    )


@pytest.mark.parametrize(
    "env, upper_bound",
    [
        ("iter/baseline/rampup", 2.0e7),
        ("iter/baseline/rampdown", 2.0e7),
        ("sparc/prd/rampup", 1.0e7),
        ("sparc/prd/rampdown", 1.0e7),
        ("step", 2.5e7),
    ],
)
def test_tokamak_policy_interfaces_include_scaled_ip(env, upper_bound):
    merged = _merge(env)
    scalar_specs = {
        spec["name"]: spec for spec in merged["observations"]["scalars"]
    }

    assert scalar_specs["Ip"] == {
        "name": "Ip",
        "scale": 1.0e7,
        "bounds": [0.0, upper_bound],
    }


def test_realistic_ip_sensor_is_retained_noisy_and_delayed():
    realistic = _merge("iter/hybrid/rampup")["observations"]["realistic"]

    assert "Ip" in realistic["filter"]["scalars"]
    assert realistic["noise"]["Ip"] == 0.01
    assert realistic["delay"]["Ip"] == 0.1
