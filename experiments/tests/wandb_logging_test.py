"""Pure tests for fixed-shape trajectory logging masks and boundary codes."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from envelope import TruncationWrapper

from experiments.plotting.wandb_logging import (
    _collect_returns_and_lengths,
    _last_valid,
    _masked_band,
    _masked_mean,
    _termination_metrics,
)
from tests.helpers import CheapBoundaryEnv
from training.envelope_gymnax import EnvelopeGymnax


def test_collection_uses_wrapped_envelope_env_and_accepts_rejax_key():
    env = TruncationWrapper(
        env=CheapBoundaryEnv(
            obs_dim=1,
            action_low=(-1.0,),
            action_high=(1.0,),
        ),
        max_steps=2,
    )

    class _Algo:
        def __init__(self):
            self.env = EnvelopeGymnax(env)

        def make_act(self, train_state):
            del train_state
            return lambda obs, key: jnp.zeros((1,), jnp.float32)

    traj, returns, lengths = _collect_returns_and_lengths(
        _Algo(), None, jax.random.PRNGKey(0), num_steps=2, n_seeds=3
    )
    assert traj.valid.shape == (3, 2)
    np.testing.assert_array_equal(lengths, np.full(3, 2.0))
    np.testing.assert_array_equal(returns, np.full(3, 3.0))


def test_mask_helpers_ignore_invalid_padding_and_select_last_valid():
    values = jnp.asarray(
        [
            [1.0, 2.0, 999.0],
            [10.0, 999.0, 999.0],
        ]
    )
    valid = jnp.asarray(
        [
            [True, True, False],
            [True, False, False],
        ]
    )

    np.testing.assert_allclose(
        _last_valid(values, valid), [2.0, 10.0], rtol=1e-7, atol=0.0
    )
    np.testing.assert_allclose(
        _masked_mean(values, valid), 13.0 / 3.0, rtol=1e-7, atol=0.0
    )

    mean, std = _masked_band(values, valid)
    np.testing.assert_allclose(mean[:2], [5.5, 2.0], rtol=1e-7, atol=0.0)
    np.testing.assert_allclose(std[:2], [4.5, 0.0], rtol=1e-7, atol=0.0)
    assert np.isnan(np.asarray(mean[2]))
    assert np.isnan(np.asarray(std[2]))


def test_termination_metrics_separate_codes_truncation_and_invalid_padding():
    valid = jnp.asarray(
        [
            [True, True, False],
            [True, False, False],
            [True, True, True],
            [True, False, False],
        ]
    )
    terminated = jnp.asarray(
        [
            [False, True, False],
            [False, True, False],  # Invalid padding must not count.
            [False, False, True],
            [True, False, False],
        ]
    )
    truncated = jnp.asarray(
        [
            [False, False, False],
            [True, False, False],
            [False, False, False],
            [False, False, False],
        ]
    )
    codes = jnp.asarray(
        [
            [-1, 1, -1],
            [-1, 3, -1],
            [-1, -1, 3],
            [2, -1, -1],
        ],
        dtype=jnp.int32,
    )
    traj = SimpleNamespace(
        valid=valid,
        terminated=terminated,
        truncated=truncated,
        info=SimpleNamespace(termination_code=codes),
    )

    metrics = _termination_metrics(
        traj,
        episode_returns=jnp.asarray([1.0, 2.0, 3.0, 4.0]),
        episode_lengths=jnp.asarray([2.0, 1.0, 3.0, 1.0]),
    )
    np.testing.assert_allclose(
        metrics["termination/completion_rate"], 0.25, rtol=1e-7, atol=0.0
    )
    np.testing.assert_allclose(
        metrics["termination/termination_rate"], 0.75, rtol=1e-7, atol=0.0
    )
    np.testing.assert_allclose(
        metrics["termination/q_min_disruption_rate"], 0.25, rtol=1e-7, atol=0.0
    )
    np.testing.assert_allclose(
        metrics["termination/greenwald_disruption_rate"],
        0.25,
        rtol=1e-7,
        atol=0.0,
    )
    np.testing.assert_allclose(
        metrics["termination/solver_failure_rate"], 0.25, rtol=1e-7, atol=0.0
    )
    np.testing.assert_allclose(
        metrics["evaluation/episode_fraction_mean"],
        7.0 / 12,
        rtol=1e-7,
        atol=0.0,
    )
    np.testing.assert_allclose(
        metrics["termination/failure_step_mean"], 2.0, rtol=1e-7, atol=0.0
    )
    np.testing.assert_allclose(
        metrics["evaluation/return_completed_mean"], 2.0, rtol=1e-7, atol=0.0
    )


def test_termination_metrics_all_completed():
    traj = SimpleNamespace(
        valid=jnp.ones((2, 2), dtype=jnp.bool_),
        terminated=jnp.zeros((2, 2), dtype=jnp.bool_),
        truncated=jnp.asarray([[False, True], [False, True]]),
        info=SimpleNamespace(termination_code=jnp.full((2, 2), -1, dtype=jnp.int32)),
    )

    metrics = _termination_metrics(
        traj,
        episode_returns=jnp.asarray([2.0, 4.0]),
        episode_lengths=jnp.asarray([2.0, 2.0]),
    )
    np.testing.assert_allclose(
        metrics["termination/completion_rate"], 1.0, rtol=1e-7, atol=0.0
    )
    np.testing.assert_allclose(
        metrics["termination/termination_rate"], 0.0, rtol=1e-7, atol=0.0
    )
    np.testing.assert_allclose(
        metrics["evaluation/episode_fraction_mean"], 1.0, rtol=1e-7, atol=0.0
    )
    assert np.isnan(np.asarray(metrics["termination/failure_step_mean"]))
    np.testing.assert_allclose(
        metrics["evaluation/return_completed_mean"], 3.0, rtol=1e-7, atol=0.0
    )


def test_termination_metrics_all_failed():
    traj = SimpleNamespace(
        valid=jnp.asarray([[True, False, False], [True, True, False]]),
        terminated=jnp.asarray([[True, False, False], [False, True, False]]),
        truncated=jnp.zeros((2, 3), dtype=jnp.bool_),
        info=SimpleNamespace(termination_code=jnp.asarray([[1, -1, -1], [-1, 2, -1]])),
    )

    metrics = _termination_metrics(
        traj,
        episode_returns=jnp.asarray([-1.0, -2.0]),
        episode_lengths=jnp.asarray([1.0, 2.0]),
    )
    np.testing.assert_allclose(
        metrics["termination/completion_rate"], 0.0, rtol=1e-7, atol=0.0
    )
    np.testing.assert_allclose(
        metrics["termination/termination_rate"], 1.0, rtol=1e-7, atol=0.0
    )
    np.testing.assert_allclose(
        metrics["evaluation/episode_fraction_mean"], 0.5, rtol=1e-7, atol=0.0
    )
    np.testing.assert_allclose(
        metrics["termination/failure_step_mean"], 1.5, rtol=1e-7, atol=0.0
    )
    assert np.isnan(np.asarray(metrics["evaluation/return_completed_mean"]))
