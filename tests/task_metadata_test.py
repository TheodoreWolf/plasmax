"""Task metadata and loader-default regression tests."""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
import yaml

import plasmax
from plasmax import rewards
from plasmax.environment import registry
from plasmax.environment.config import parse_env_and_backend
from plasmax.environment.factory import make
from plasmax.environment.schema import TaskConfig, WorldModelConfig

_MOCK_ENV = "mock/circular/smoke"
_MOCK_BACKEND = "mock"

_EXPECTED_TASKS: dict[str, tuple[str, float | None]] = {
    "iter/advanced/rampup": ("lh_transition", -100),
    "iter/advanced/flattop": ("P_diff", 0.0),
    "iter/advanced/rampdown": ("rampdown", -330),
    "iter/baseline/rampup": ("lh_transition", -100),
    "iter/baseline/flattop": ("P_diff", 0.0),
    "iter/baseline/rampdown": ("rampdown", -1325),
    "iter/hybrid/rampup": ("lh_transition", -100),
    "iter/hybrid/flattop": ("P_diff", 0.0),
    "iter/hybrid/rampdown": ("rampdown", -1000),
    "sparc/prd/rampup": ("lh_transition", -10),
    "sparc/prd/flattop": ("P_diff", 0.0),
    "sparc/prd/rampdown": ("rampdown", -380),
    "sparc/reduced_field/rampup": ("lh_transition", -16),
    "sparc/reduced_field/flattop": ("P_diff", 0.0),
    "sparc/reduced_field/rampdown": ("rampdown", -462),
    "step/spp_001_ec_hd/flattop": ("P_diff", 0.0),
    _MOCK_ENV: ("P_diff", 0.0),
    "kstar_worldmodel": ("native", None),
}


def _raw_task(path: str | Path) -> dict[str, object]:
    with Path(path).open() as stream:
        return (yaml.safe_load(stream) or {})["task"]


@pytest.mark.parametrize("alias", sorted(_EXPECTED_TASKS))
def test_every_leaf_environment_yaml_stores_task_metadata(alias):
    assert set(registry.ENV_ALIASES) == set(_EXPECTED_TASKS)
    expected_reward, expected_penalty = _EXPECTED_TASKS[alias]
    task = _raw_task(registry.resolve_env(alias))
    assert task["reward"] == expected_reward
    if expected_penalty is None:
        assert task["terminal_penalty"] is None
    else:
        np.testing.assert_array_equal(task["terminal_penalty"], expected_penalty)


def test_public_constructor_returns_bare_environment():
    assert isinstance(plasmax.make(_MOCK_ENV, _MOCK_BACKEND), plasmax.PlasmaxEnv)


def test_omitted_reward_and_penalty_resolve_from_task_metadata():
    env = make(_MOCK_ENV, _MOCK_BACKEND)
    dynamics = env.unwrapped._dynamics
    assert dynamics._reward_fn is rewards.P_diff
    np.testing.assert_array_equal(dynamics._disruption_penalty, jnp.float32(0.0))
    assert dynamics._disruption_penalty.dtype == jnp.float32


def test_string_and_callable_reward_overrides_are_preserved():
    string_env = make(_MOCK_ENV, _MOCK_BACKEND, reward="Q_fusion")
    assert string_env.unwrapped._dynamics._reward_fn is rewards.Q_fusion

    def custom_reward(last_action, state, action, next_state):
        del last_action, state, action, next_state
        return jnp.float32(7.0)

    callable_env = make(
        _MOCK_ENV,
        _MOCK_BACKEND,
        reward=custom_reward,
    )
    assert callable_env.unwrapped._dynamics._reward_fn is custom_reward


def test_explicit_zero_terminal_penalty_overrides_nonzero_metadata():
    env = make(
        "iter/advanced/rampup",
        "cgm",
        disruption_penalty=0.0,
    )
    np.testing.assert_array_equal(env.unwrapped._dynamics._disruption_penalty, 0.0)


def test_phase_defaults_are_available_without_duplicated_reward_maps():
    rampup = parse_env_and_backend("iter/advanced/rampup", "cgm")
    rampdown = parse_env_and_backend("sparc/prd/rampdown", "cgm")
    assert rampup.task == TaskConfig(reward="lh_transition", terminal_penalty=-100)
    assert rampdown.task == TaskConfig(reward="rampdown", terminal_penalty=-380)


def test_kstar_inherits_native_reward_and_rejects_terminal_penalties():
    config = parse_env_and_backend("kstar_worldmodel")
    assert isinstance(config, WorldModelConfig)
    assert config.task == TaskConfig(reward="native", terminal_penalty=None)

    env = make("kstar_worldmodel")
    assert env is not None
    with pytest.raises(ValueError, match="native"):
        make("kstar_worldmodel", reward="P_diff")
    with pytest.raises(ValueError, match="disruption_penalty"):
        make("kstar_worldmodel", disruption_penalty=0.0)
    with pytest.raises(ValueError, match="standalone|no backend"):
        make("kstar_worldmodel", backend="mock")
