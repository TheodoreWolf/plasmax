"""Pure tests for fixed-shape trajectory logging masks and boundary codes."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from envelope import TruncationWrapper

from tests.helpers import CheapBoundaryEnv
from training.envelope_gymnax import EnvelopeGymnax
from training.wandb_logging import (
    _base_metrics,
    _collect_returns_and_lengths,
    _last_valid,
    _masked_band,
    _masked_mean,
    _termination_metrics,
    evaluation_scalar_metrics,
)


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


def test_mask_helpers_preserve_nan_from_valid_physics_state():
    values = jnp.asarray(
        [
            [1.0, jnp.nan, 999.0],
            [10.0, 20.0, 999.0],
        ]
    )
    valid = jnp.asarray(
        [
            [True, True, False],
            [True, True, False],
        ]
    )

    last = _last_valid(values, valid)
    assert np.isnan(np.asarray(last[0]))
    np.testing.assert_array_equal(last[1], 20.0)
    assert np.isnan(np.asarray(_masked_mean(values, valid)))

    mean, std = _masked_band(values, valid)
    np.testing.assert_allclose(mean[0], 5.5, rtol=1e-7, atol=0.0)
    assert np.isnan(np.asarray(mean[1]))
    assert np.isnan(np.asarray(std[1]))


def test_shared_physics_scalars_use_final_valid_states_under_jit_and_vmap():
    valid = jnp.asarray([[True, True, False], [True, False, False]])
    beta = jnp.asarray(
        [
            [[1.0, 2.0, 999.0], [10.0, 999.0, 999.0]],
            [[3.0, 4.0, 999.0], [20.0, 999.0, 999.0]],
        ]
    )

    def collect_metrics(beta_values: jax.Array) -> dict[str, jax.Array]:
        plasma = SimpleNamespace(
            **{
                name: beta_values
                for name in (
                    "Q_fusion",
                    "W_thermal_total",
                    "P_fusion",
                    "tau_E",
                    "H98",
                    "beta_N",
                    "q_min",
                    "q95",
                    "f_non_inductive",
                    "fgw_n_e_line_avg",
                    "fgw_n_e_volume_avg",
                    "P_SOL_total",
                )
            },
            P_LH=jnp.ones_like(beta_values),
        )
        traj = SimpleNamespace(
            env_state=SimpleNamespace(plasma=plasma),
            reward=jnp.ones_like(beta_values),
            valid=valid,
            terminated=jnp.zeros_like(valid),
            truncated=jnp.asarray([[False, True, False], [True, False, False]]),
            info=None,
        )
        return evaluation_scalar_metrics(
            traj, jnp.asarray([2.0, 1.0]), jnp.asarray([2.0, 1.0])
        )

    metrics = jax.jit(jax.vmap(collect_metrics))(beta)
    np.testing.assert_allclose(metrics["obs/beta_N"], [6.0, 12.0], rtol=1e-7)
    np.testing.assert_allclose(
        metrics["ref/W_thermal_MJ"], [13.0e-6 / 3, 27.0e-6 / 3], rtol=1e-7
    )
    np.testing.assert_array_equal(metrics["termination/completion_rate"], [1.0, 1.0])


def test_shared_scalars_leave_non_plasma_environments_without_normalized_beta():
    valid = jnp.ones((1, 2), dtype=jnp.bool_)
    traj = SimpleNamespace(
        env_state=SimpleNamespace(obs=jnp.ones((1, 2, 3))),
        reward=jnp.asarray([[1.0, 2.0]]),
        valid=valid,
        terminated=jnp.zeros_like(valid),
        truncated=jnp.asarray([[False, True]]),
        info=None,
    )

    metrics = evaluation_scalar_metrics(traj, jnp.asarray([3.0]), jnp.asarray([2.0]))

    assert not any(name.startswith(("obs/", "ref/")) for name in metrics)
    np.testing.assert_array_equal(metrics["evaluation/return_mean"], 3.0)
    np.testing.assert_array_equal(metrics["termination/completion_rate"], 1.0)


def test_nonfinite_reward_rate_counts_valid_steps_under_jit_and_vmap():
    rewards = jnp.asarray(
        [
            [[jnp.nan, jnp.inf, -jnp.inf], [1.0, jnp.nan, jnp.inf]],
            [[1.0, 2.0, jnp.nan], [3.0, jnp.inf, -jnp.inf]],
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            [[jnp.nan, jnp.inf, -jnp.inf], [jnp.nan, jnp.inf, -jnp.inf]],
        ]
    )
    valid = jnp.asarray(
        [
            [[True, True, True], [True, False, False]],
            [[True, True, False], [True, False, False]],
            [[True, True, True], [True, True, True]],
            [[False, False, False], [False, False, False]],
        ]
    )

    def metrics_for_rewards(reward: jax.Array, valid: jax.Array) -> dict:
        traj = SimpleNamespace(
            reward=reward,
            valid=valid,
            terminated=jnp.zeros_like(valid),
            truncated=jnp.zeros_like(valid),
            info=None,
        )
        returns = jnp.sum(jnp.where(valid, reward, 0.0), axis=1)
        lengths = jnp.sum(valid, axis=1).astype(jnp.float32)
        return _base_metrics(traj, returns, lengths, train_metrics=None)

    metrics = jax.jit(jax.vmap(metrics_for_rewards))(rewards, valid)
    np.testing.assert_allclose(
        metrics["evaluation/nonfinite_reward_rate"],
        [0.75, 0.0, 0.0, 0.0],
        rtol=1e-7,
        atol=0.0,
    )
    np.testing.assert_allclose(
        metrics["evaluation/return_mean"],
        [np.nan, 3.0, 10.5, 0.0],
        rtol=1e-7,
        atol=0.0,
        equal_nan=True,
    )


def test_termination_metrics_separate_codes_truncation_and_invalid_padding():
    valid = jnp.asarray(
        [
            [True, True, False],
            [True, False, False],
            [True, True, True],
            [True, False, False],
            [True, True, False],
        ]
    )
    terminated = jnp.asarray(
        [
            [False, True, False],
            [False, True, False],  # Invalid padding must not count.
            [False, False, True],
            [True, False, False],
            [False, True, False],
        ]
    )
    truncated = jnp.asarray(
        [
            [False, False, False],
            [True, False, False],
            [False, False, False],
            [False, False, False],
            [False, False, False],
        ]
    )
    codes = jnp.asarray(
        [
            [-1, 1, -1],
            [-1, 4, -1],
            [-1, -1, 3],
            [2, -1, -1],
            [-1, 4, -1],
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
        episode_returns=jnp.asarray([1.0, 2.0, 3.0, 4.0, 5.0]),
        episode_lengths=jnp.asarray([2.0, 1.0, 3.0, 1.0, 2.0]),
    )
    np.testing.assert_allclose(
        metrics["termination/completion_rate"], 0.2, rtol=1e-7, atol=0.0
    )
    np.testing.assert_allclose(
        metrics["termination/termination_rate"], 0.8, rtol=1e-7, atol=0.0
    )
    np.testing.assert_allclose(
        metrics["termination/q_min_disruption_rate"], 0.2, rtol=1e-7, atol=0.0
    )
    np.testing.assert_allclose(
        metrics["termination/greenwald_disruption_rate"],
        0.2,
        rtol=1e-7,
        atol=0.0,
    )
    np.testing.assert_allclose(
        metrics["termination/solver_failure_rate"], 0.2, rtol=1e-7, atol=0.0
    )
    np.testing.assert_allclose(
        metrics["termination/invalid_state_rate"], 0.2, rtol=1e-7, atol=0.0
    )
    np.testing.assert_allclose(
        metrics["evaluation/episode_fraction_mean"],
        9.0 / 15,
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
        metrics["termination/invalid_state_rate"], 0.0, rtol=1e-7, atol=0.0
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
