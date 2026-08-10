"""PPO thin-adapter and end-to-end training smoke tests."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from envelope import Environment, TruncationWrapper
from flax import linen as nn

pytest.importorskip("rejax")

from helpers import CheapBoundaryEnv
from rejax.algos.ppo import PPO

from agents.ppo import MultiDiscretePolicy, PPOAdapter, ResidualGaussianPolicy
from plasmax.environment.factory import load_scenario
from plasmax.wrappers import QuantizeActionWrapper
from training.envelope_gymnax import EnvelopeGymnax


def _make_algo(env: Environment, **ppo_kwargs) -> PPOAdapter:
    gymnax_env = EnvelopeGymnax(env)
    return PPOAdapter.create(
        env=gymnax_env,
        env_params=gymnax_env.default_params,
        **ppo_kwargs,
    )


def test_multidiscrete_policy_samples_and_scores_each_head():
    policy = MultiDiscretePolicy(
        nvec=(2, 4), hidden_layer_sizes=(8,), activation=nn.tanh
    )
    obs = jnp.zeros((3, 2), jnp.float32)
    params = policy.init(jax.random.PRNGKey(0), obs, jax.random.PRNGKey(1))
    actions, log_prob, entropy = policy.apply(params, obs, jax.random.PRNGKey(2))
    rescored_log_prob, rescored_entropy = policy.apply(
        params, obs, actions, method="log_prob_entropy"
    )

    assert actions.shape == (3, 2)
    assert jnp.all((actions[:, 0] >= 0) & (actions[:, 0] < 2))
    assert jnp.all((actions[:, 1] >= 0) & (actions[:, 1] < 4))
    assert jnp.all(jnp.isfinite(log_prob))
    assert jnp.all(jnp.isfinite(entropy))
    np.testing.assert_allclose(
        log_prob, rescored_log_prob, rtol=1e-5, atol=1e-8
    )
    np.testing.assert_allclose(entropy, rescored_entropy, rtol=1e-5, atol=1e-8)


def test_adapter_uses_upstream_ppo_optimization():
    assert issubclass(PPOAdapter, PPO)
    assert "calculate_gae" not in PPOAdapter.__dict__
    assert "update" not in PPOAdapter.__dict__


def test_residual_gaussian_policy_starts_at_action_setpoint():
    policy = ResidualGaussianPolicy(
        action_dim=2,
        action_range=(
            jnp.full((2,), -1.0, dtype=jnp.float32),
            jnp.full((2,), 1.0, dtype=jnp.float32),
        ),
        action_setpoint=(0.25, -1.0),
        hidden_layer_sizes=(8,),
        activation=nn.swish,
        initial_log_std=-1.0,
    )
    obs = jnp.zeros((1, 3), dtype=jnp.float32)
    variables = policy.init(jax.random.PRNGKey(7), obs, jax.random.PRNGKey(8))

    distribution = policy.apply(variables, obs, method="_action_dist")

    np.testing.assert_array_equal(
        distribution.mode(),
        jnp.asarray([[0.25, -1.0]], dtype=jnp.float32),
    )
    assert jnp.all(jnp.isfinite(distribution.stddev()))


@pytest.mark.parametrize("quantized", [False, True], ids=("continuous", "quantized"))
def test_short_training_on_cheap_envelope_env_has_finite_outputs(quantized):
    env = CheapBoundaryEnv(
        obs_dim=2,
        action_low=(-1.0,),
        action_high=(1.0,),
    )
    if quantized:
        env = QuantizeActionWrapper(env=env, bin_counts=(3,))
    env = TruncationWrapper(env=env, max_steps=2)

    algo = _make_algo(
        env,
        total_timesteps=16,
        num_envs=2,
        num_steps=4,
        num_epochs=1,
        num_minibatches=1,
        eval_freq=16,
        normalize_rewards=False,
        normalize_observations=False,
    )
    _, (lengths, returns) = jax.jit(algo.train)(jax.random.PRNGKey(0))

    assert jnp.all(jnp.isfinite(returns))
    assert jnp.all(lengths > 0)


@pytest.mark.integration
class PPOTrainSmokeTest:
    def test_short_torax_training_run_completes_with_finite_outputs(self):
        env = load_scenario("test")
        algo = _make_algo(
            env,
            total_timesteps=1024,
            num_envs=16,
            num_steps=8,
            num_epochs=1,
            num_minibatches=2,
            eval_freq=1024,
            learning_rate=3e-4,
            normalize_rewards=False,
            normalize_observations=False,
        )
        _, (lengths, returns) = jax.jit(algo.train)(jax.random.PRNGKey(0))
        assert jnp.all(jnp.isfinite(returns))
        assert jnp.all(lengths > 0)
