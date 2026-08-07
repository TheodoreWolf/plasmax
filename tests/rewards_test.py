"""Tests for the reward functions in plasmax.rewards.

Rewards are plain callables with signature
``(last_action, state, action, next_state) -> jax.Array``. The named rewards
(Q_fusion, beta_N, P_diff) are thin functions over ``next_state.plasma`` and
must return exactly those quantities; ``resolve_reward_fn`` dispatches string
aliases to those callables.
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from helpers import make_test_env

from plasmax import rewards as rewards_lib


def _make_env_state():
    env = make_test_env()
    state, _ = env.init(jax.random.key(0))
    return state


class NamedRewardsTest:
    """Each named reward returns the right quantity from next_state.plasma."""

    @classmethod
    def setup_class(cls):
        cls._state = _make_env_state()
        # The reward signature exposes (last_action, state, action, next_state);
        # for these checks state is unused and we pass the same EnvState as
        # both state and next_state.
        cls._last_action = cls._state.prev_action
        cls._action = cls._state.prev_action

    def _call(self, fn):
        return fn(self._last_action, self._state, self._action, self._state)

    def test_Q_fusion_matches_postout(self):
        np.testing.assert_allclose(
            self._call(rewards_lib.Q_fusion),
            self._state.plasma.Q_fusion,
            atol=1e-6,
            rtol=0.0,
        )

    def test_beta_N_matches_postout(self):
        np.testing.assert_allclose(
            self._call(rewards_lib.beta_N),
            self._state.plasma.beta_N,
            atol=1e-6,
            rtol=0.0,
        )

    def test_P_diff_matches_postout_expression(self):
        expected = (self._state.plasma.P_fusion - self._state.plasma.P_aux_total) * 1e-9
        np.testing.assert_allclose(
            self._call(rewards_lib.P_diff), expected, atol=1e-12, rtol=0.0
        )

    def test_P_diff_negative_when_fusion_below_aux(self):
        # Independent semantic check (the expression test above mirrors the
        # implementation, so it cannot catch a wrong sign or unit scale).
        # The small circular test plasma produces far less fusion power than
        # its 10 MW of auxiliary heating, so P_diff must be negative and,
        # in GW units, smaller in magnitude than the total heating power.
        reward = float(self._call(rewards_lib.P_diff))
        assert reward < 0.0
        assert reward > -1.0  # |P_aux| is tens of MW, i.e. < 1 GW.


class SoftBarrierTest:
    """soft_barrier(quantity, limit) penalises approaching an upper limit.

    The barrier reads ``quantity(next_state)`` only, so these tests pass a
    constant quantity and a dummy state to probe its shape independently of any
    postout field.
    """

    @staticmethod
    def _barrier_at(value, limit=1.0, **kw):
        fn = rewards_lib.soft_barrier(lambda _s: jnp.asarray(value), limit, **kw)
        return float(fn(None, None, None, None))

    def test_near_zero_well_below_limit(self):
        # At half the limit (below the 0.9 threshold) the penalty is negligible.
        assert self._barrier_at(0.5) > -0.05

    def test_sharply_negative_at_limit(self):
        # At the limit the log-sigmoid barrier has dropped well below zero.
        assert self._barrier_at(1.0) < -1.0

    def test_monotonic_decreasing(self):
        vals = [self._barrier_at(x) for x in (0.5, 0.8, 0.95, 1.05)]
        assert all(b < a for a, b in zip(vals, vals[1:], strict=False))

    def test_finite_above_limit(self):
        # Input is clipped, so even a large overshoot stays finite (no -inf).
        assert np.isfinite(self._barrier_at(10.0))


def _with_postout(state, **updates):
    """Returns ``state`` with the named postout scalars replaced."""
    new_postout = dataclasses.replace(
        state.plasma.post, **{k: jnp.asarray(v) for k, v in updates.items()}
    )
    return dataclasses.replace(
        state, plasma=dataclasses.replace(state.plasma, post=new_postout)
    )


class RampdownRewardTest:
    """Barrier-only reward: ~0 inside every stability limit, sharply negative
    as a limit is approached. Checked by perturbing the relevant postout
    scalars rather than mirroring the implementation's barrier sum."""

    @classmethod
    def setup_class(cls):
        cls._state = _make_env_state()
        cls._action = cls._state.prev_action

    def _reward(self, state):
        return float(rewards_lib.rampdown(self._action, state, self._action, state))

    def test_resolves_and_returns_finite_scalar(self):
        fn = rewards_lib.resolve_reward_fn("rampdown")
        reward = fn(self._action, self._state, self._action, self._state)
        assert np.shape(reward) == ()
        assert np.isfinite(float(reward))

    def test_near_zero_when_safely_inside_limits(self):
        # Barriers only (no Ip progress term — Ip is prescribed, not an
        # actuator): never positive, and ~0 for the safe nominal test plasma.
        safe = self._reward(self._state)
        assert -0.5 < safe <= 0.0

    def test_penalises_approaching_greenwald_limit(self):
        risky = _with_postout(self._state, fgw_n_e_line_avg=1.05)
        assert self._reward(risky) < self._reward(self._state) - 1.0

    def test_penalises_low_q_min(self):
        risky = _with_postout(self._state, q_min=0.9)
        assert self._reward(risky) < self._reward(self._state) - 1.0

    def test_q_min_barrier_gradient_is_finite_at_zero(self):
        def reward_at(q_min):
            state = _with_postout(self._state, q_min=q_min)
            return rewards_lib.rampdown(
                self._action,
                state,
                self._action,
                state,
            )

        gradient = jax.grad(reward_at)(jnp.asarray(0.0))

        assert np.isfinite(float(gradient))


class ResolveRewardFnTest:
    @classmethod
    def setup_class(cls):
        cls._state = _make_env_state()
        cls._action = cls._state.prev_action

    def test_string_alias_returns_named_function(self):
        assert rewards_lib.resolve_reward_fn("Q_fusion") is rewards_lib.Q_fusion
        assert rewards_lib.resolve_reward_fn("beta_N") is rewards_lib.beta_N
        assert rewards_lib.resolve_reward_fn("P_diff") is rewards_lib.P_diff

    def test_passes_through_callable(self):
        def fn(la, s, a, ns):
            return ns.plasma.Q_fusion

        assert rewards_lib.resolve_reward_fn(fn) is fn

    def test_unknown_alias_raises_with_valid_names(self):
        with pytest.raises(ValueError, match="Unknown reward") as exc_info:
            rewards_lib.resolve_reward_fn("not_a_real_reward_xyz")
        msg = str(exc_info.value)
        assert "Q_fusion" in msg
        assert "beta_N" in msg
        assert "P_diff" in msg

    def test_non_callable_non_string_raises_type_error(self):
        with pytest.raises(TypeError, match="callable or string"):
            rewards_lib.resolve_reward_fn(42)
