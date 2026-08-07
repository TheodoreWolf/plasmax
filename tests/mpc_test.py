"""Tests for the learned-dynamics MPC agent."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from envelope import AutoResetWrapper
from helpers import CheapBoundaryEnv, make_test_env

from experiments.studies.mpc import MPCAgent, rollout
from plasmax.wrappers import (
    ActionRescaleWrapper,
    PlasmaxTruncationWrapper,
    QuantizeActionWrapper,
    unwrap_to_env_state,
)


def _reward_fn(obs, action, next_obs):
    del action, next_obs
    return -jnp.sum(obs**2)


class MPCAgentTest:
    @classmethod
    def setup_class(cls):
        cls.env = make_test_env()

    def _make_agent(self, **kwargs):
        defaults = dict(horizon=3, num_samples=8, buffer_size=64)
        defaults.update(kwargs)
        return MPCAgent.create(self.env, _reward_fn, **defaults)

    def _transition(self):
        env_state, info = self.env.init(jax.random.key(0))
        obs = info.obs
        # Keep the small repeated-transition regression used by the original
        # model-loss test; the environment applies its normal action clipping.
        action = jnp.zeros(self.env.action_space.shape, jnp.float32)
        _, next_info = self.env.step(env_state, action)
        return obs, action, next_info.reward, next_info.obs

    def test_create_shapes(self):
        agent, state = self._make_agent()
        action_dim = self.env.action_space.shape[0]
        assert agent.action_low.shape == (action_dim,)
        assert state.buffer.size == 64

    def test_create_rejects_quantized_action_space_clearly(self):
        env = QuantizeActionWrapper(
            env=ActionRescaleWrapper(env=self.env),
            bin_counts=(3, 5),
        )
        with pytest.raises(TypeError, match="Continuous.*quantized"):
            MPCAgent.create(env, _reward_fn)

    def test_act_respects_action_bounds(self):
        agent, state = self._make_agent()
        _, info = self.env.init(jax.random.key(0))
        action = agent.act(state, info.obs, jax.random.key(1))
        assert action.shape == (self.env.action_space.shape[0],)
        assert np.all(np.asarray(action) >= np.asarray(agent.action_low) - 1e-5)
        assert np.all(np.asarray(action) <= np.asarray(agent.action_high) + 1e-5)

    def test_train_step_reduces_loss_on_repeated_transition(self):
        agent, state = self._make_agent()
        obs, action, reward, next_obs = self._transition()
        for _ in range(64):
            state = agent.observe(
                state,
                obs,
                action,
                reward,
                terminated=False,
                truncated=False,
                next_obs=next_obs,
            )

        _, loss0 = agent.train_step(state, jax.random.key(3), batch_size=32)
        for i in range(20):
            state, loss = agent.train_step(
                state, jax.random.fold_in(jax.random.key(4), i), batch_size=32
            )
        assert float(loss) < float(loss0)

    def test_rollout_retains_boundary_then_freezes_and_masks_padding(self):
        env = PlasmaxTruncationWrapper(env=self.env, max_steps=1)
        agent, state = self._make_agent(horizon=2, num_samples=4)
        n = 3
        final_state, out = jax.jit(
            lambda k: rollout(agent, state, env, k, n, train_batch_size=8)
        )(jax.random.key(0))

        assert out.reward.shape == (n,)
        assert out.action.shape == (n, self.env.action_space.shape[0])
        assert out.next_obs.shape == (n, self.env.observation_space.shape[0])
        assert np.isfinite(np.asarray(out.total_return))
        np.testing.assert_array_equal(out.valid, [True, False, False])
        np.testing.assert_array_equal(out.terminated, [False, False, False])
        np.testing.assert_array_equal(out.truncated, [True, False, False])
        np.testing.assert_array_equal(out.done, [True, False, False])
        np.testing.assert_allclose(
            out.total_return, out.reward[0], rtol=1e-7, atol=0.0
        )
        np.testing.assert_allclose(
            out.next_obs[1:],
            jnp.broadcast_to(out.next_obs[0], out.next_obs[1:].shape),
            rtol=1e-7,
            atol=0.0,
        )

        env_state = unwrap_to_env_state(out.env_state)
        np.testing.assert_allclose(
            env_state.plasma.t, env_state.plasma.t[0], rtol=1e-7, atol=0.0
        )
        assert final_state.buffer.index == 1

    def test_rollout_skips_training_until_buffer_has_a_full_batch(self):
        env = PlasmaxTruncationWrapper(env=self.env, max_steps=1)
        agent, state = self._make_agent(horizon=2, num_samples=4)
        _, out = rollout(
            agent,
            state,
            env,
            jax.random.key(0),
            5,
            train_batch_size=8,
        )
        assert np.all(np.isnan(np.asarray(out.model_loss)))

    def test_rollout_retains_termination_then_freezes_fake_state(self):
        agent, state = self._make_agent(horizon=2, num_samples=4)
        env = CheapBoundaryEnv(
            obs_dim=self.env.observation_space.shape[0],
            action_low=tuple(np.asarray(self.env.action_space.low)),
            action_high=tuple(np.asarray(self.env.action_space.high)),
            terminate_after=1,
        )
        final_state, out = rollout(
            agent,
            state,
            env,
            jax.random.key(0),
            3,
            train_batch_size=8,
        )

        np.testing.assert_array_equal(out.valid, [True, False, False])
        np.testing.assert_array_equal(out.terminated, [True, False, False])
        np.testing.assert_array_equal(out.truncated, [False, False, False])
        np.testing.assert_array_equal(out.env_state.steps, [1, 1, 1])
        np.testing.assert_allclose(
            out.total_return, out.reward[0], rtol=1e-7, atol=0.0
        )
        assert final_state.buffer.index == 1

    def test_rollout_rejects_autoreset(self):
        agent, state = self._make_agent(horizon=2, num_samples=4)
        base = CheapBoundaryEnv(
            obs_dim=self.env.observation_space.shape[0],
            action_low=tuple(np.asarray(self.env.action_space.low)),
            action_high=tuple(np.asarray(self.env.action_space.high)),
            terminate_after=1,
        )
        with pytest.raises(ValueError, match="non-autoresetting"):
            rollout(
                agent,
                state,
                AutoResetWrapper(env=base),
                jax.random.key(0),
                2,
            )

    def test_train_step_ignores_terminated_transitions(self):
        agent, state = self._make_agent()
        obs, action, reward, next_obs = self._transition()
        mismatched_next_obs = next_obs + 100.0
        for _ in range(64):
            state = agent.observe(
                state,
                obs,
                action,
                reward,
                terminated=True,
                truncated=False,
                next_obs=mismatched_next_obs,
            )
        _, loss = agent.train_step(state, jax.random.key(3), batch_size=32)
        assert np.isfinite(np.asarray(loss))
        np.testing.assert_allclose(loss, 0.0, atol=1e-6, rtol=0.0)

    def test_train_step_keeps_truncated_transitions(self):
        agent, state = self._make_agent()
        obs, action, reward, next_obs = self._transition()
        mismatched_next_obs = next_obs + 100.0
        for _ in range(64):
            state = agent.observe(
                state,
                obs,
                action,
                reward,
                terminated=False,
                truncated=True,
                next_obs=mismatched_next_obs,
            )
        _, loss = agent.train_step(state, jax.random.key(3), batch_size=32)
        assert np.isfinite(np.asarray(loss))
        assert float(loss) > 1.0
