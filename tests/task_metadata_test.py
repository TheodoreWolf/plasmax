"""Task metadata and loader-default regression tests."""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import pytest
import yaml

import plasmax
from plasmax import rewards
from plasmax.environment import registry
from plasmax.environment.config import parse_env_and_backend
from plasmax.environment.factory import make
from plasmax.environment.schema import TaskConfig

_MOCK_ENV = "mock/circular/smoke"
_MOCK_BACKEND = "mock"

_EXPECTED_TASKS: dict[str, str] = {
    "iter/advanced/rampup": "lh_transition",
    "iter/advanced/flattop": "P_diff",
    "iter/advanced/rampdown": "rampdown",
    "iter/baseline/rampup": "lh_transition",
    "iter/baseline/flattop": "P_diff",
    "iter/baseline/rampdown": "rampdown",
    "iter/hybrid/rampup": "lh_transition",
    "iter/hybrid/flattop": "P_diff",
    "iter/hybrid/rampdown": "rampdown",
    "sparc/prd/rampup": "lh_transition",
    "sparc/prd/flattop": "P_diff",
    "sparc/prd/rampdown": "rampdown",
    "sparc/reduced_field/rampup": "lh_transition",
    "sparc/reduced_field/flattop": "P_diff",
    "sparc/reduced_field/rampdown": "rampdown",
    "step/spp_001_ec_hd/flattop": "P_diff",
    _MOCK_ENV: "P_diff",
    "kstar_worldmodel": "native",
}


def _raw_task(path: str | Path) -> dict[str, object]:
    with Path(path).open() as stream:
        return (yaml.safe_load(stream) or {})["task"]


@pytest.mark.parametrize("alias", sorted(_EXPECTED_TASKS))
def test_every_leaf_environment_yaml_stores_task_metadata(alias):
    assert set(registry.ENV_ALIASES) == set(_EXPECTED_TASKS)
    task = _raw_task(registry.resolve_env(alias))
    assert task == {"reward": _EXPECTED_TASKS[alias]}


def test_public_constructor_returns_bare_environment():
    assert isinstance(plasmax.make(_MOCK_ENV, _MOCK_BACKEND), plasmax.PlasmaxEnv)


def test_omitted_reward_resolves_from_task_metadata():
    env = make(_MOCK_ENV, _MOCK_BACKEND)
    dynamics = env.unwrapped._dynamics
    assert dynamics._reward_fn is rewards.P_diff


def test_megawatt_reward_override_resolves_from_alias() -> None:
    env = make(_MOCK_ENV, _MOCK_BACKEND, reward="P_diff_MW")
    assert env.unwrapped._dynamics._reward_fn is rewards.P_diff_MW


def test_string_and_callable_reward_overrides_are_preserved():
    string_env = make(_MOCK_ENV, _MOCK_BACKEND, reward="Q_fusion")
    assert string_env.unwrapped._dynamics._reward_fn is rewards.Q_fusion

    def custom_reward(state, action, next_state, termination_code):
        del state, action, next_state, termination_code
        return jnp.float32(7.0)

    callable_env = make(
        _MOCK_ENV,
        _MOCK_BACKEND,
        reward=custom_reward,
    )
    assert callable_env.unwrapped._dynamics._reward_fn is custom_reward


def test_phase_defaults_are_available_without_duplicated_reward_maps():
    rampup = parse_env_and_backend("iter/advanced/rampup", "cgm")
    rampdown = parse_env_and_backend("sparc/prd/rampdown", "cgm")
    assert rampup.task == TaskConfig(reward="lh_transition")
    assert rampdown.task == TaskConfig(reward="rampdown")
