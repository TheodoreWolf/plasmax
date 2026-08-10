"""Task metadata and loader-default regression tests."""

from __future__ import annotations

import inspect
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
import yaml

import plasmax
from plasmax import rewards
from plasmax.environment import registry
from plasmax.environment.config import (
    parse_env_and_backend,
    parse_scenario,
    scenario_to_yaml,
)
from plasmax.environment.factory import make
from plasmax.environment.schema import TaskConfig

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
    "step": ("P_diff", 0.0),
    "kstar": ("native", None),
}


def _raw_task(path: str | Path) -> dict[str, object]:
    with Path(path).open() as stream:
        return (yaml.safe_load(stream) or {})["task"]


@pytest.mark.parametrize("alias", sorted(_EXPECTED_TASKS))
def test_every_leaf_environment_yaml_stores_task_metadata(alias):
    assert set(registry.ENV_ALIASES) == set(_EXPECTED_TASKS)
    expected_reward, expected_penalty = _EXPECTED_TASKS[alias]
    task = _raw_task(registry.ENV_ALIASES[alias])
    assert task["reward"] == expected_reward
    if expected_penalty is None:
        assert task["terminal_penalty"] is None
    else:
        np.testing.assert_array_equal(task["terminal_penalty"], expected_penalty)


def test_single_file_test_fixture_stores_uncalibrated_task_metadata():
    task = _raw_task(registry.SCENARIO_ALIASES["test"])
    assert task == {"reward": "P_diff", "terminal_penalty": 0.0}


@pytest.mark.parametrize(
    ("alias", "expected"),
    [
        (alias, penalty)
        for alias, (_, penalty) in _EXPECTED_TASKS.items()
        if alias.endswith(("rampup", "rampdown"))
    ],
)
def test_ten_calibrated_terminal_penalties_are_exact(alias, expected):
    actual = _raw_task(registry.ENV_ALIASES[alias])["terminal_penalty"]
    np.testing.assert_array_equal(actual, expected)


def test_public_constructor_defaults_to_realistic():
    assert inspect.signature(plasmax.make).parameters["variant"].default == "realistic"


def test_omitted_reward_and_penalty_resolve_from_task_metadata():
    env = make("test")
    dynamics = env.unwrapped._dynamics
    assert dynamics._reward_fn is rewards.P_diff
    np.testing.assert_array_equal(dynamics._disruption_penalty, jnp.float32(0.0))
    assert dynamics._disruption_penalty.dtype == jnp.float32


def test_string_and_callable_reward_overrides_are_preserved():
    string_env = make("test", reward="Q_fusion")
    assert string_env.unwrapped._dynamics._reward_fn is rewards.Q_fusion

    def custom_reward(last_action, state, action, next_state):
        del last_action, state, action, next_state
        return jnp.float32(7.0)

    callable_env = make("test", reward=custom_reward)
    assert callable_env.unwrapped._dynamics._reward_fn is custom_reward


def test_explicit_zero_terminal_penalty_overrides_nonzero_metadata(tmp_path):
    config = parse_scenario("test").model_copy(
        update={"task": TaskConfig(reward="P_diff", terminal_penalty=-123.0)}
    )
    path = tmp_path / "explicit-zero.yaml"
    scenario_to_yaml(config, path)
    env = make(str(path), disruption_penalty=0.0)
    np.testing.assert_array_equal(env.unwrapped._dynamics._disruption_penalty, 0.0)


def test_phase_defaults_are_available_without_duplicated_reward_maps():
    rampup = parse_env_and_backend("iter/advanced/rampup", "cgm")
    rampdown = parse_env_and_backend("sparc/prd/rampdown", "cgm")
    assert rampup.task == TaskConfig(reward="lh_transition", terminal_penalty=-100)
    assert rampdown.task == TaskConfig(reward="rampdown", terminal_penalty=-380)


def test_kstar_inherits_native_reward_and_rejects_terminal_penalties():
    env = make("kstar", "fusion_lstm")
    assert env is not None
    make("kstar", "fusion_lstm", reward="native")
    with pytest.raises(ValueError, match="native reward"):
        make("kstar", "fusion_lstm", reward="P_diff")
    with pytest.raises(ValueError, match="disruption_penalty"):
        make("kstar", "fusion_lstm", disruption_penalty=0.0)
