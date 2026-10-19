"""Tests for the reward functions in plasmax.rewards.

Rewards are plain callables with signature
``(state, action, next_state, termination_code) -> jax.Array``. Named TORAX
rewards preserve their physical scores, return positive utility during normal
operation, and own their terminal behavior. Custom callables pass through.
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from helpers import make_test_env

from plasmax import rewards as rewards_lib
from plasmax.environment.core import EnvState

_NAMED_REWARDS = (
    rewards_lib.Q_fusion,
    rewards_lib.beta_N,
    rewards_lib.P_diff,
    rewards_lib.P_diff_MW,
    rewards_lib.W_thermal,
    rewards_lib.rampdown,
    rewards_lib.lh_transition,
)


def _positive_reference(score: float | jax.Array) -> np.ndarray:
    value = np.asarray(score, dtype=np.float64)
    return 0.5 * (value + np.hypot(value, 2.0))


def _make_env_state() -> EnvState:
    env = make_test_env()
    state, _ = env.init(jax.random.key(0))
    return state


class NamedRewardsTest:
    """Positive utilities retain the physical score and its units."""

    @classmethod
    def setup_class(cls):
        cls._state = _make_env_state()
        cls._action = cls._state.prev_action

    def _call(self, fn):
        return fn(self._state, self._action, self._state, jnp.int32(-1))

    def test_Q_fusion_matches_postout(self):
        np.testing.assert_allclose(
            self._call(rewards_lib.Q_fusion),
            _positive_reference(self._state.plasma.Q_fusion),
            atol=1e-6,
            rtol=0.0,
        )

    def test_beta_N_matches_postout(self):
        np.testing.assert_allclose(
            self._call(rewards_lib.beta_N),
            _positive_reference(self._state.plasma.beta_N),
            atol=1e-6,
            rtol=0.0,
        )

    def test_P_diff_matches_postout_expression(self):
        expected = (self._state.plasma.P_fusion - self._state.plasma.P_aux_total) * 1e-9
        np.testing.assert_allclose(
            self._call(rewards_lib.P_diff),
            _positive_reference(expected),
            atol=1e-12,
            rtol=0.0,
        )

    def test_P_diff_below_unit_baseline_when_fusion_below_aux(self):
        # Independent semantic check (the expression test above mirrors the
        # implementation, so it cannot catch a wrong sign or unit scale).
        # The small circular test plasma produces far less fusion power than
        # its 10 MW of auxiliary heating, so the negative net-power score
        # must give less utility than the unit baseline at zero net power.
        reward = float(self._call(rewards_lib.P_diff))
        assert 0.0 < reward < 1.0

    def test_P_diff_MW_uses_megawatts_before_utility_transform(self) -> None:
        def reward_at(fusion: jax.Array, auxiliary: jax.Array) -> jax.Array:
            next_state = _with_postout(
                self._state, P_fusion=fusion, P_aux_total=auxiliary
            )
            return rewards_lib.P_diff_MW(
                self._state, self._action, next_state, jnp.int32(-1)
            )

        # Fusion at 8, 10, and 12 MW against 10 MW of heating gives scores
        # of -2, 0, and 2 MW. Convert the score before applying squareplus.
        values, (fusion_gradients, auxiliary_gradients) = jax.jit(
            jax.vmap(jax.value_and_grad(reward_at, argnums=(0, 1)))
        )(jnp.asarray([8e6, 10e6, 12e6]), jnp.full(3, 10e6))

        root_two = np.sqrt(2.0)
        np.testing.assert_allclose(
            values, [root_two - 1.0, 1.0, root_two + 1.0], rtol=1e-6, atol=0.0
        )
        expected_gradients = 1e-6 * np.asarray(
            [0.5 - 0.5 / root_two, 0.5, 0.5 + 0.5 / root_two]
        )
        np.testing.assert_allclose(
            fusion_gradients, expected_gradients, rtol=1e-6, atol=0.0
        )
        np.testing.assert_allclose(
            auxiliary_gradients, -expected_gradients, rtol=1e-6, atol=0.0
        )

    def test_W_thermal_keeps_hundred_megajoule_scale(self):
        np.testing.assert_allclose(
            self._call(rewards_lib.W_thermal),
            _positive_reference(self._state.plasma.W_thermal_total * 1e-8),
            rtol=1e-6,
            atol=0.0,
        )


class TerminalRewardTest:
    @classmethod
    def setup_class(cls) -> None:
        cls._state = _make_env_state()

    @pytest.mark.parametrize("reward", _NAMED_REWARDS)
    def test_physical_and_solver_terminations_take_log_of_positive_utility(
        self, reward: rewards_lib.RewardFn
    ) -> None:
        def at_code(code: jax.Array) -> jax.Array:
            return reward(self._state, self._state.prev_action, self._state, code)

        values = jax.jit(jax.vmap(at_code))(jnp.asarray([-1, 1, 2, 3]))
        assert values[0] > 0.0
        np.testing.assert_allclose(
            values[1:], np.full(3, np.log(values[0])), rtol=1e-6, atol=1e-7
        )
        assert np.all(values[1:] < values[0])

    @pytest.mark.parametrize("reward", _NAMED_REWARDS)
    def test_invalid_state_inputs_have_zero_reward_and_gradient_under_vmap(
        self, reward: rewards_lib.RewardFn
    ) -> None:
        def reward_at(value: jax.Array) -> jax.Array:
            next_state = _with_postout(
                self._state,
                Q_fusion=value,
                beta_N=value,
                P_fusion=value,
                P_aux_total=value,
                W_thermal_total=value,
                fgw_n_e_line_avg=value,
                li3=value,
                P_SOL_total=value,
                P_LH=value,
            )
            next_state = _with_grid_q_min(next_state, value)
            plasma = next_state.plasma
            next_state = dataclasses.replace(
                next_state,
                plasma=dataclasses.replace(
                    plasma, sim=dataclasses.replace(plasma.sim, t=value)
                ),
            )
            return reward(
                self._state, self._state.prev_action, next_state, jnp.int32(4)
            )

        values, gradients = jax.jit(jax.vmap(jax.value_and_grad(reward_at)))(
            jnp.asarray([np.nan, np.inf, -np.inf, 0.0])
        )
        np.testing.assert_array_equal(values, np.zeros(4))
        np.testing.assert_array_equal(gradients, np.zeros(4))

    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
    def test_utility_values_and_gradients_at_zero_and_in_the_tails(
        self, dtype: jnp.dtype
    ) -> None:
        scores = jnp.asarray([-4e7, 0.0, 4e7], dtype=dtype)
        evaluate = jax.jit(
            jax.vmap(
                jax.value_and_grad(rewards_lib._terminal_reward), in_axes=(0, None)
            )
        )
        ordinary, ordinary_grad = evaluate(scores, jnp.int32(-1))
        terminal, terminal_grad = evaluate(scores, jnp.int32(3))
        np.testing.assert_allclose(ordinary, [2.5e-8, 1.0, 4e7], rtol=1e-6, atol=0.0)
        np.testing.assert_allclose(
            ordinary_grad, [6.25e-16, 0.5, 1.0], rtol=1e-6, atol=0.0
        )
        np.testing.assert_allclose(
            terminal, [-np.log(4e7), 0.0, np.log(4e7)], rtol=1e-6, atol=0.0
        )
        np.testing.assert_allclose(
            terminal_grad, [2.5e-8, 0.5, 2.5e-8], rtol=1e-6, atol=0.0
        )
        assert ordinary.dtype == terminal.dtype == dtype

    def test_mixed_batch_masks_only_invalid_states(self) -> None:
        scores = jnp.asarray([-2.0, -2.0, -2.0, -2.0, np.nan], dtype=jnp.float32)
        codes = jnp.asarray([-1, 1, 2, 3, 4])
        values, gradients = jax.jit(
            jax.vmap(jax.value_and_grad(rewards_lib._terminal_reward))
        )(scores, codes)
        root_two = np.sqrt(2.0)
        np.testing.assert_allclose(
            values,
            [root_two - 1.0, *([-np.arcsinh(1.0)] * 3), 0.0],
            rtol=1e-6,
            atol=0.0,
        )
        np.testing.assert_allclose(
            gradients,
            [0.5 - 0.5 / root_two, *([0.5 / root_two] * 3), 0.0],
            rtol=1e-6,
            atol=0.0,
        )

    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
    def test_large_and_small_terminal_scores_keep_representable_gradients(
        self, dtype: jnp.dtype
    ) -> None:
        scores = jnp.asarray([-1e20, -1e-20, 1e-20, 1e20], dtype=dtype)
        evaluate = jax.jit(
            jax.vmap(
                jax.value_and_grad(rewards_lib._terminal_reward), in_axes=(0, None)
            )
        )
        values, gradients = evaluate(scores, jnp.int32(1))
        np.testing.assert_allclose(
            values,
            [-np.log(1e20), -0.5e-20, 0.5e-20, np.log(1e20)],
            rtol=1e-6,
            atol=0.0,
        )
        np.testing.assert_allclose(
            gradients, [1e-20, 0.5, 0.5, 1e-20], rtol=1e-6, atol=0.0
        )
        ordinary, ordinary_gradients = evaluate(scores, jnp.int32(-1))
        assert np.all(np.isfinite(ordinary_gradients))
        np.testing.assert_allclose(
            ordinary, [1e-20, 1.0, 1.0, 1e20], rtol=1e-6, atol=0.0
        )


class SoftBarrierTest:
    """soft_barrier(quantity, limit) penalises approaching an upper limit.

    The barrier reads ``quantity(next_state)`` only, so these tests pass a
    constant quantity and a dummy state to probe its shape independently of any
    postout field.
    """

    @staticmethod
    def _barrier_at(value, limit=1.0, **kw):
        fn = rewards_lib.soft_barrier(lambda _s: jnp.asarray(value), limit, **kw)
        return float(fn(None, None, None, jnp.int32(-1)))

    def test_near_zero_well_below_limit(self):
        # At half the limit (below the 0.9 threshold) the penalty is negligible.
        assert self._barrier_at(0.5) > -0.05

    def test_sharply_negative_at_limit(self):
        # At the limit the log-sigmoid barrier has dropped well below zero.
        assert self._barrier_at(1.0) < -1.0

    def test_monotonic_decreasing(self):
        vals = [self._barrier_at(x) for x in (0.5, 0.8, 0.95, 1.05, 1.2, 10.0)]
        assert all(b < a for a, b in zip(vals, vals[1:], strict=False))

    def test_finite_above_limit(self):
        # Stable log-sigmoid evaluation stays finite even for large overshoots.
        assert np.isfinite(self._barrier_at(10.0))

    def test_negative_quantities_share_the_zero_floor(self) -> None:
        np.testing.assert_allclose(
            self._barrier_at(-1.0), self._barrier_at(0.0), atol=0.0, rtol=0.0
        )

    def test_large_violations_keep_negative_gradients(self) -> None:
        barrier = rewards_lib.soft_barrier(lambda value: value, limit=2.0)

        def reward_at(value: jax.Array) -> jax.Array:
            return barrier(None, None, value, jnp.int32(-1))

        gradients = jax.jit(jax.vmap(jax.grad(reward_at)))(
            jnp.asarray([2.4, 4.0, 20.0])
        )

        assert np.all(np.isfinite(gradients))
        # Far above the limit, the penalty keeps its asymptotic slope / limit.
        np.testing.assert_allclose(gradients, -20.0, atol=2e-4, rtol=0.0)


def _with_postout(state: EnvState, **updates: float | jax.Array) -> EnvState:
    """Returns ``state`` with the named postout scalars replaced."""
    new_postout = dataclasses.replace(
        state.plasma.post, **{k: jnp.asarray(v) for k, v in updates.items()}
    )
    return dataclasses.replace(
        state, plasma=dataclasses.replace(state.plasma, post=new_postout)
    )


def _with_grid_q_min(state: EnvState, q_min: float | jax.Array) -> EnvState:
    """Set one face below a fixed safe grid, retaining the fitted diagnostic."""
    core = dataclasses.replace(
        state.plasma.core,
        q_face=jnp.full_like(state.plasma.core.q_face, 4.0).at[1].set(q_min),
    )
    sim = dataclasses.replace(state.plasma.sim, core_profiles=core)
    return dataclasses.replace(state, plasma=dataclasses.replace(state.plasma, sim=sim))


class RampdownRewardTest:
    """Barrier-only utility: near one inside every stability limit, decreasing
    as a limit is approached. Checked by perturbing the relevant postout
    scalars and face-grid q rather than mirroring the barrier sum."""

    @classmethod
    def setup_class(cls):
        cls._state = _make_env_state()
        cls._action = cls._state.prev_action

    def _reward(self, state):
        return float(rewards_lib.rampdown(state, self._action, state, jnp.int32(-1)))

    def test_resolves_and_returns_finite_scalar(self):
        fn = rewards_lib.resolve_reward_fn("rampdown")
        reward = fn(self._state, self._action, self._state, jnp.int32(-1))
        assert np.shape(reward) == ()
        assert np.isfinite(float(reward))

    def test_near_one_when_safely_inside_limits(self):
        # Barriers only (no Ip progress term — Ip is prescribed, not an
        # actuator): utility is at most one and near one in the safe plasma.
        safe = self._reward(self._state)
        assert 0.75 < safe <= 1.0

    def test_penalises_approaching_greenwald_limit(self):
        risky = _with_postout(self._state, fgw_n_e_line_avg=1.05)
        assert self._reward(risky) < 0.5 * self._reward(self._state)

    def test_penalises_low_q_min(self) -> None:
        risky = _with_grid_q_min(self._state, 0.9)
        assert self._reward(risky) < 0.5 * self._reward(self._state)

    def test_q_min_barrier_gradient_is_finite_at_zero(self) -> None:
        def reward_at(q_min: jax.Array) -> jax.Array:
            state = _with_grid_q_min(self._state, q_min)
            return rewards_lib.rampdown(
                state,
                self._action,
                state,
                jnp.int32(-1),
            )

        gradient = jax.jit(jax.grad(reward_at))(jnp.asarray(0.0))

        assert np.isfinite(float(gradient))


@pytest.mark.parametrize("reward", [rewards_lib.rampdown, rewards_lib.lh_transition])
def test_q_barriers_use_grid_min_instead_of_fitted_min(
    reward: rewards_lib.RewardFn,
) -> None:
    state = _with_grid_q_min(_make_env_state(), 2.0)

    def fitted_reward(q_min: jax.Array) -> jax.Array:
        next_state = _with_postout(state, q_min=q_min)
        return reward(state, state.prev_action, next_state, jnp.int32(-1))

    fitted_values = jax.jit(jax.vmap(fitted_reward))(jnp.asarray([2.0, -50.0]))
    np.testing.assert_allclose(fitted_values[0], fitted_values[1], rtol=0.0, atol=0.0)

    def grid_reward(q_min: jax.Array) -> jax.Array:
        next_state = _with_grid_q_min(state, q_min)
        return reward(state, state.prev_action, next_state, jnp.int32(-1))

    values, gradients = jax.jit(jax.vmap(jax.value_and_grad(grid_reward)))(
        jnp.asarray([2.0, 0.9, 0.0])
    )
    assert np.all(np.isfinite(values))
    assert np.all(np.isfinite(gradients))
    assert values[1] < 0.5 * values[0]
    assert gradients[1] > 0.0


class LHTransitionRewardTest:
    """Power shaping keeps its time-weighted gradient for every positive ratio."""

    @classmethod
    def setup_class(cls) -> None:
        cls._state = _make_env_state()

    @pytest.mark.parametrize("t_final", [10.0, 100.0])
    def test_power_shaping_is_uncapped_with_zero_floor(self, t_final: float) -> None:
        plasma = self._state.plasma
        state = dataclasses.replace(
            self._state,
            plasma=dataclasses.replace(
                plasma,
                sim=dataclasses.replace(plasma.sim, t=jnp.asarray(0.5 * t_final)),
            ),
        )

        def reward_at(ratio: jax.Array) -> jax.Array:
            next_state = _with_postout(state, P_SOL_total=ratio * 2e6, P_LH=2e6)
            return rewards_lib.lh_transition(
                state,
                state.prev_action,
                next_state,
                jnp.int32(-1),
                t_final=t_final,
            )

        values, gradients = jax.jit(jax.vmap(jax.value_and_grad(reward_at)))(
            jnp.asarray([-1.0, 0.0, 0.5, 1.5, 2.0, 10.0])
        )

        # Invert squareplus to recover the physical score. Its power term
        # keeps weight 1/4 at half the ramp duration and remains uncapped.
        scores = values - 1.0 / values
        np.testing.assert_allclose(
            scores - scores[1],
            [0.0, 0.0, 0.125, 0.375, 0.5, 2.5],
            atol=1e-6,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            gradients * (1.0 + 1.0 / values**2),
            [0.0, 0.0, 0.25, 0.25, 0.25, 0.25],
            atol=1e-6,
            rtol=0.0,
        )


class ResolveRewardFnTest:
    @classmethod
    def setup_class(cls):
        cls._state = _make_env_state()
        cls._action = cls._state.prev_action

    def test_string_alias_returns_named_function(self):
        assert rewards_lib.resolve_reward_fn("Q_fusion") is rewards_lib.Q_fusion
        assert rewards_lib.resolve_reward_fn("beta_N") is rewards_lib.beta_N
        assert rewards_lib.resolve_reward_fn("P_diff") is rewards_lib.P_diff
        assert rewards_lib.resolve_reward_fn("P_diff_MW") is rewards_lib.P_diff_MW

    def test_passes_through_callable(self):
        def fn(state, action, next_state, termination_code):
            del state, action, next_state, termination_code
            return jnp.asarray(-7.0)

        assert rewards_lib.resolve_reward_fn(fn) is fn
        np.testing.assert_array_equal(
            fn(self._state, self._action, self._state, jnp.int32(1)), -7.0
        )

    def test_unknown_alias_raises_with_valid_names(self):
        with pytest.raises(ValueError, match="Unknown reward") as exc_info:
            rewards_lib.resolve_reward_fn("not_a_real_reward_xyz")
        msg = str(exc_info.value)
        assert "Q_fusion" in msg
        assert "beta_N" in msg
        assert "P_diff" in msg
        assert "P_diff_MW" in msg

    def test_non_callable_non_string_raises_type_error(self):
        with pytest.raises(TypeError, match="callable or string"):
            rewards_lib.resolve_reward_fn(42)
