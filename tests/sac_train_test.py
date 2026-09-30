"""SAC thin-adapter and end-to-end training smoke tests."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import tyro
from envelope import TruncationWrapper, static_field
from flax import linen as nn

pytest.importorskip("rejax")

from helpers import CheapBoundaryEnv, CheapBoundaryState
from rejax.algos.sac import SAC
from rejax.networks import SquashedGaussianPolicy

from agents.sac import ResidualSquashedGaussianPolicy, SACAdapter
from training import train_sac
from training.envelope_gymnax import EnvelopeGymnax


class _SetpointState(CheapBoundaryState):
    prev_action: jax.Array


class _SetpointEnv(CheapBoundaryEnv):
    initial_action: tuple[float, ...] = static_field(default=(0.3,))

    def init(self, key):
        state, _ = super().init(key)
        state = _SetpointState(
            obs=state.obs,
            steps=state.steps,
            prev_action=jnp.asarray(self.initial_action, jnp.float32),
        )
        return state, self._info(state)


def _cheap_env() -> EnvelopeGymnax:
    env = _SetpointEnv(
        obs_dim=2,
        action_low=(-1.0,),
        action_high=(1.0,),
    )
    return EnvelopeGymnax(TruncationWrapper(env=env, max_steps=2))


def test_adapter_uses_upstream_sac_optimization():
    assert issubclass(SACAdapter, SAC)
    assert "update" not in SACAdapter.__dict__


def test_launcher_routes_residual_policy_and_requested_epochs():
    config = tyro.cli(
        train_sac.Config,
        args=["--sac.residual-policy", "--sac.num-epochs", "16"],
    )
    algo = train_sac._build_algo(config, _cheap_env())

    assert train_sac.SACConfig().residual_policy is False
    assert isinstance(algo.actor, ResidualSquashedGaussianPolicy)
    assert algo.num_epochs == 16


@pytest.mark.parametrize("residual_policy", [False, True])
def test_short_upstream_training_returns_finite_outputs(residual_policy):
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
        agent_kwargs={"residual_policy": residual_policy},
        normalize_observations=False,
        normalize_rewards=False,
    ).with_eval_callback(eval_callback)

    train_state, (returns, lengths) = jax.jit(algo.train)(jax.random.PRNGKey(0))

    assert returns.shape == (2, 2)
    assert lengths.shape == (2, 2)
    assert all(jnp.all(jnp.isfinite(value)) for value in jax.tree.leaves(train_state))
    if residual_policy:
        assert isinstance(algo.actor, ResidualSquashedGaussianPolicy)
        assert jnp.any(
            train_state.actor_ts.params["params"]["action_mean"]["bias"] != 0
        )


def test_residual_policy_starts_at_setpoint_and_preserves_upstream_std():
    kwargs = {
        "action_dim": 2,
        "action_range": (
            jnp.asarray([-2.0, 1.0], jnp.float32),
            jnp.asarray([4.0, 5.0], jnp.float32),
        ),
        "hidden_layer_sizes": (8,),
        "activation": nn.relu,
        "log_std_range": (-10, 2),
    }
    policy = ResidualSquashedGaussianPolicy(action_setpoint=(0.25, 1.0), **kwargs)
    upstream = SquashedGaussianPolicy(**kwargs)
    obs = jnp.asarray([[0.0, 0.0], [1.0, -0.5]], jnp.float32)
    init_key, sample_key = jax.random.split(jax.random.PRNGKey(0))
    params = policy.init(init_key, obs, sample_key)
    upstream_params = upstream.init(init_key, obs, sample_key)
    distribution = policy.apply(params, obs, method="_action_dist")
    upstream_dist = upstream.apply(upstream_params, obs, method="_action_dist")
    initial_action = policy.action_loc + policy.action_scale * jnp.tanh(
        distribution.mode()
    )

    np.testing.assert_allclose(
        initial_action, [[0.25, 1.002], [0.25, 1.002]], rtol=1e-6, atol=1e-6
    )
    np.testing.assert_array_equal(distribution.stddev(), upstream_dist.stddev())
    sampled_action, log_prob = policy.apply(params, obs, sample_key)
    rescored_log_prob = policy.apply(params, obs, sampled_action, method="log_prob")
    np.testing.assert_allclose(log_prob, rescored_log_prob, rtol=2e-3, atol=2e-3)
    assert jnp.all(jnp.isfinite(log_prob))


def test_residual_actor_has_finite_nonzero_gradients_at_bound_setpoint():
    policy = ResidualSquashedGaussianPolicy(
        action_dim=1,
        action_range=(jnp.asarray([-1.0]), jnp.asarray([1.0])),
        action_setpoint=(-1.0,),
        hidden_layer_sizes=(8,),
        activation=nn.relu,
        log_std_range=(-10, 2),
    )
    obs = jnp.ones((2, 3), jnp.float32)
    params = policy.init(jax.random.PRNGKey(0), obs, jax.random.PRNGKey(1))

    def objective(parameters):
        action, log_prob = policy.apply(parameters, obs, jax.random.PRNGKey(2))
        return (log_prob + jnp.sum(action**2, axis=-1)).mean()

    value, gradient = jax.jit(jax.value_and_grad(objective))(params)

    assert jnp.isfinite(value)
    assert all(jnp.all(jnp.isfinite(value)) for value in jax.tree.leaves(gradient))
    assert jnp.any(gradient["params"]["action_mean"]["bias"] != 0)


def test_residual_adapter_initializes_at_native_reset_action():
    env = EnvelopeGymnax(
        TruncationWrapper(
            env=_SetpointEnv(
                obs_dim=2,
                action_low=(1.0,),
                action_high=(5.0,),
                initial_action=(3.5,),
            ),
            max_steps=2,
        )
    )
    algo = SACAdapter.create(
        env=env,
        env_params=env.default_params,
        num_envs=1,
        buffer_size=4,
        batch_size=1,
        hidden_layer_sizes=(8,),
        agent_kwargs={"residual_policy": True},
    )
    train_state = algo.init_state(jax.random.PRNGKey(0))
    act = algo.make_deterministic_act(train_state)

    np.testing.assert_allclose(
        act(jnp.zeros((2,), jnp.float32), jax.random.PRNGKey(1)),
        [3.5],
        rtol=1e-6,
        atol=1e-6,
    )


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
