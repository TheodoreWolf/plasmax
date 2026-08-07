"""SAC thin-adapter and end-to-end training smoke tests."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from envelope import TruncationWrapper

pytest.importorskip("rejax")

from helpers import CheapBoundaryEnv
from rejax.algos.sac import SAC

from agents.sac import SACAdapter
from training.envelope_gymnax import EnvelopeGymnax


def _cheap_env() -> EnvelopeGymnax:
    env = CheapBoundaryEnv(
        obs_dim=2,
        action_low=(-1.0,),
        action_high=(1.0,),
    )
    return EnvelopeGymnax(TruncationWrapper(env=env, max_steps=2))


def test_adapter_uses_upstream_sac_optimization():
    assert issubclass(SACAdapter, SAC)
    assert "update" not in SACAdapter.__dict__


def test_short_upstream_training_returns_finite_outputs():
    env = _cheap_env()

    def eval_callback(algo, ts, rng, train_metrics):
        del algo, ts, rng, train_metrics
        returns = jnp.ones((2,), dtype=jnp.float32)
        lengths = jnp.full((2,), 2, dtype=jnp.int32)
        return returns, lengths

    algo = SACAdapter.create(
        env=env,
        env_params=env.default_params,
        total_timesteps=8,
        eval_freq=8,
        num_envs=2,
        num_epochs=1,
        buffer_size=32,
        fill_buffer=0,
        batch_size=2,
        hidden_layer_sizes=(8,),
        normalize_observations=False,
        normalize_rewards=False,
    ).with_eval_callback(eval_callback)

    train_state, (returns, lengths) = jax.jit(algo.train)(jax.random.PRNGKey(0))

    assert returns.shape == (2, 2)
    assert lengths.shape == (2, 2)
    assert all(jnp.all(jnp.isfinite(value)) for value in jax.tree.leaves(train_state))


def test_deterministic_action_ignores_rng_and_respects_bounds():
    env = _cheap_env()
    algo = SACAdapter.create(
        env=env,
        env_params=env.default_params,
        total_timesteps=2,
        eval_freq=2,
        num_envs=1,
        buffer_size=4,
        fill_buffer=0,
        batch_size=1,
        hidden_layer_sizes=(8,),
        normalize_observations=True,
    )
    train_state = algo.init_state(jax.random.PRNGKey(0))
    act = algo.make_deterministic_act(train_state)
    obs = jnp.zeros((2,), dtype=jnp.float32)

    action_a = act(obs, jax.random.PRNGKey(1))
    action_b = act(obs, jax.random.PRNGKey(2))

    np.testing.assert_array_equal(action_a, action_b)
    assert jnp.all(action_a >= env.action_space().low)
    assert jnp.all(action_a <= env.action_space().high)
