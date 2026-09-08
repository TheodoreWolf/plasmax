"""Cheap shared-evaluation checks: pairing, masking, transfer, and NPZ outputs."""

import jax
import numpy as np
import pytest
from envelope import TruncationWrapper
from helpers import CheapBoundaryEnv

from agents.policy_io import LoadedPolicy, environment_interface
from training.evaluation import (
    check_interfaces,
    evaluate_policy,
    evaluate_returns,
    save_trajectories,
)


def _env(**kwargs):
    return TruncationWrapper(
        env=CheapBoundaryEnv(
            obs_dim=2, action_low=(-1.0,), action_high=(1.0,), **kwargs
        ),
        max_steps=4,
    )


def _act(obs, rng):
    del obs
    return jax.random.uniform(rng, (1,), minval=-1.0, maxval=1.0)


def test_shared_metrics_keep_terminal_transition_mask_padding_and_pair_keys():
    source = _env(terminate_after=2)
    target = _env(truncate_after=3)
    check_interfaces(source, target)
    key = jax.random.key(2)
    metrics, trajectory = evaluate_policy(_act, source, key, num_episodes=3)
    target_metrics, target_trajectory = evaluate_policy(
        _act, target, key, num_episodes=3
    )
    np.testing.assert_array_equal(metrics["returns"], [3, 3, 3])
    np.testing.assert_array_equal(metrics["lengths"], [2, 2, 2])
    np.testing.assert_array_equal(target_metrics["returns"], [6, 6, 6])
    np.testing.assert_array_equal(
        trajectory.action[:, :2], target_trajectory.action[:, :2]
    )
    np.testing.assert_array_equal(trajectory.valid[0], [True, True, False, False])
    np.testing.assert_array_equal(trajectory.done[0], [False, True, False, False])


def test_return_collection_supports_outer_jit_vmap():
    env = _env(terminate_after=2)

    def evaluate(key):
        return evaluate_returns(_act, env, key, num_episodes=2)

    lengths, returns = jax.jit(jax.vmap(evaluate))(
        jax.random.split(jax.random.key(3), 2)
    )
    np.testing.assert_array_equal(lengths, np.full((2, 2), 2))
    np.testing.assert_array_equal(returns, np.full((2, 2), 3))


def test_loaded_policy_evaluation_materializes_fresh_target_spaces_before_jit():
    policy = LoadedPolicy(
        "backprop_open_loop",
        {"params": np.zeros((2, 1)), "source_times": np.arange(4), "time_index": 1},
        environment_interface(_env()),
        {},
        True,
    )
    target = _env(terminate_after=2)

    def evaluate(key):
        return evaluate_policy(policy, target, key, num_episodes=1)[0]

    metrics = jax.jit(evaluate)(jax.random.key(0))
    np.testing.assert_array_equal(metrics["returns"], [3])
    # The cached target bounds must remain concrete for subsequent exports.
    check_interfaces(policy, target)


def test_interface_checks_bounds_and_shapes():
    other = CheapBoundaryEnv(obs_dim=2, action_low=(-2.0,), action_high=(1.0,))
    with pytest.raises(ValueError, match="action_low"):
        check_interfaces(_env(), other)
    other = CheapBoundaryEnv(obs_dim=3, action_low=(-1.0,), action_high=(1.0,))
    with pytest.raises(ValueError, match="observation_shape"):
        check_interfaces(_env(), other)


def test_npz_contains_common_boundaries_observations_and_actions(tmp_path):
    env = _env(terminate_after=2)
    _, trajectory = evaluate_policy(_act, env, jax.random.key(0), num_episodes=2)
    path = save_trajectories(tmp_path / "trajectory.npz", trajectory, env)
    with np.load(path, allow_pickle=False) as values:
        assert {
            "obs",
            "next_obs",
            "reward",
            "valid",
            "done",
            "terminated",
            "truncated",
            "command_physical",
            "termination_code",
            "time_steps",
            "interface_json",
        } <= set(values.files)
        np.testing.assert_array_equal(values["valid"], trajectory.valid)
        np.testing.assert_array_equal(values["reward"], trajectory.reward)
