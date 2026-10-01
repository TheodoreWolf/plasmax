"""SAC thin-adapter and end-to-end training smoke tests."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import tyro
from envelope import TruncationWrapper, static_field

pytest.importorskip("rejax")

from helpers import CheapBoundaryEnv, CheapBoundaryState
from rejax.algos.sac import SAC
from rejax.networks import SquashedGaussianPolicy

from agents.sac import SACAdapter
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


def test_launcher_routes_requested_epochs_to_upstream_actor() -> None:
    from training import train_sac

    config = tyro.cli(
        train_sac.Config,
        args=["--sac.num-epochs", "16"],
    )
    algo = train_sac._build_algo(config, _cheap_env())

    assert type(algo.actor) is SquashedGaussianPolicy
    assert algo.num_epochs == 16


def test_launcher_rejects_removed_policy_option() -> None:
    from training import train_sac

    with pytest.raises(SystemExit) as error:
        tyro.cli(train_sac.Config, args=["--sac.residual-policy"])
    assert error.value.code != 0


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
    assert type(algo.actor) is SquashedGaussianPolicy


@pytest.mark.parametrize("deterministic", [False, True])
def test_actions_do_not_depend_on_reset_actuator_defaults(deterministic: bool) -> None:
    actions = []
    observation = jnp.asarray([0.2, -0.4], jnp.float32)
    for default in (-1.5, 2.5):
        env = EnvelopeGymnax(
            TruncationWrapper(
                env=_SetpointEnv(
                    obs_dim=2,
                    action_low=(-2.0,),
                    action_high=(3.0,),
                    initial_action=(default,),
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
        )
        assert type(algo.actor) is SquashedGaussianPolicy
        state = algo.init_state(jax.random.PRNGKey(0))
        act = jax.jit(algo.make_act(state, deterministic=deterministic))
        actions.append(act(observation, jax.random.PRNGKey(1)))
    np.testing.assert_array_equal(actions[0], actions[1])


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
