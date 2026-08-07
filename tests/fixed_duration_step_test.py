"""Unit tests for bounded differentiable physical control steps."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from torax._src import state as torax_state

from plasmax.environment.stepping import (
    _fixed_duration_step_scan,
    fixed_duration_step,
)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class _SyntheticState:
    t: jax.Array
    dt: jax.Array
    value: jax.Array
    calls: jax.Array
    action_sum: jax.Array
    gain_sum: jax.Array
    solver_numeric_outputs: torax_state.SolverNumericOutputs


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class _SyntheticProvider:
    proposed_dts: jax.Array
    solver_states: jax.Array
    sawtooth_events: jax.Array
    inner_iterations: jax.Array
    outer_iterations: jax.Array
    action: jax.Array
    gain: jax.Array


def _numeric(
    error: int | jax.Array = 0,
    *,
    inner: int | jax.Array = 0,
    outer: int | jax.Array = 0,
    event: bool | jax.Array = False,
) -> torax_state.SolverNumericOutputs:
    return torax_state.SolverNumericOutputs(
        outer_solver_iterations=jnp.asarray(outer, dtype=jnp.int32),
        solver_error_state=jnp.asarray(error, dtype=jnp.int32),
        inner_solver_iterations=jnp.asarray(inner, dtype=jnp.int32),
        sawtooth_crash=jnp.asarray(event, dtype=jnp.bool_),
    )


def _initial_state(value: float | jax.Array = 1.0) -> _SyntheticState:
    value = jnp.asarray(value, dtype=jnp.float64)
    return _SyntheticState(
        t=jnp.asarray(0.0, dtype=value.dtype),
        dt=jnp.asarray(0.0, dtype=value.dtype),
        value=value,
        calls=jnp.asarray(0, dtype=jnp.int32),
        action_sum=jnp.asarray(0.0, dtype=value.dtype),
        gain_sum=jnp.asarray(0.0, dtype=value.dtype),
        solver_numeric_outputs=_numeric(),
    )


def _provider(
    proposed_dts,
    *,
    solver_states=None,
    sawtooth_events=None,
    inner_iterations=None,
    outer_iterations=None,
    action: float | jax.Array = 2.0,
    gain: float | jax.Array = 0.5,
) -> _SyntheticProvider:
    proposed_dts = jnp.asarray(proposed_dts, dtype=jnp.float64)
    size = proposed_dts.shape[0]

    def values_or_default(values, default, dtype):
        if values is None:
            values = jnp.full((size,), default, dtype=dtype)
        return jnp.asarray(values, dtype=dtype)

    return _SyntheticProvider(
        proposed_dts=proposed_dts,
        solver_states=values_or_default(solver_states, 0, jnp.int32),
        sawtooth_events=values_or_default(sawtooth_events, False, jnp.bool_),
        inner_iterations=values_or_default(inner_iterations, 1, jnp.int32),
        outer_iterations=values_or_default(outer_iterations, 1, jnp.int32),
        action=jnp.asarray(action, dtype=jnp.float64),
        gain=jnp.asarray(gain, dtype=jnp.float64),
    )


def _synthetic_step(state, post, max_dt, provider):
    index = jnp.minimum(state.calls, provider.proposed_dts.shape[0] - 1)
    dt = jnp.minimum(provider.proposed_dts[index], max_dt)
    # Sequential dependence is deliberate: the derivative of a macro-step must
    # include every accepted substep, not only the final solve.
    value = state.value * (1.0 + provider.gain * provider.action * dt)
    numeric = _numeric(
        provider.solver_states[index],
        inner=provider.inner_iterations[index],
        outer=provider.outer_iterations[index],
        event=provider.sawtooth_events[index],
    )
    next_state = dataclasses.replace(
        state,
        t=state.t + dt,
        dt=dt,
        value=value,
        calls=state.calls + 1,
        action_sum=state.action_sum + provider.action,
        gain_sum=state.gain_sum + provider.gain,
        solver_numeric_outputs=numeric,
    )
    return next_state, post + value * dt


def _state_is_finite(state, post):
    return jnp.isfinite(state.value) & jnp.isfinite(post)


def _run(
    provider,
    *,
    control_dt=0.1,
    max_solver_substeps=8,
    max_event_substeps=0,
    state=None,
):
    if state is None:
        state = _initial_state()
    return fixed_duration_step(
        _synthetic_step,
        _state_is_finite,
        max_solver_substeps,
        max_event_substeps,
        jnp.asarray(control_dt, dtype=jnp.float64),
        state,
        jnp.asarray(0.0, dtype=jnp.float64),
        provider,
    )


class FixedDurationPrimalTest:
    def test_one_step_completion(self):
        result = _run(_provider([1.0]), max_solver_substeps=1)
        np.testing.assert_allclose(result.sim_state.t, 0.1, atol=1e-12, rtol=0.0)
        np.testing.assert_allclose(result.sim_state.dt, 0.1, atol=1e-12, rtol=0.0)
        assert int(result.internal_steps) == 1
        assert int(result.sawtooth_crashes) == 0
        assert bool(result.control_step_complete)
        assert not bool(result.step_limit_reached)

    def test_shortened_substeps_sum_to_exact_interval(self):
        result = _run(_provider([0.04, 0.03, 0.03]))
        np.testing.assert_allclose(result.sim_state.t, 0.1, atol=1e-12, rtol=0.0)
        np.testing.assert_allclose(result.sim_state.dt, 0.1, atol=1e-12, rtol=0.0)
        assert int(result.internal_steps) == 3
        assert bool(result.control_step_complete)

    def test_coarse_then_fine_convergence_is_accepted_and_aggregated(self):
        result = _run(
            _provider(
                [0.04, 0.06],
                solver_states=[2, 0],
                inner_iterations=[7, 3],
                outer_iterations=[2, 1],
            )
        )
        numeric = result.sim_state.solver_numeric_outputs
        assert int(numeric.solver_error_state) == 2
        assert int(numeric.inner_solver_iterations) == 10
        assert int(numeric.outer_solver_iterations) == 3
        assert bool(result.control_step_complete)

    def test_solver_state_one_stops_at_partial_time(self):
        result = _run(_provider([0.02, 0.08], solver_states=[1, 0]))
        np.testing.assert_allclose(result.sim_state.t, 0.02, atol=1e-12, rtol=0.0)
        np.testing.assert_allclose(result.sim_state.dt, 0.02, atol=1e-12, rtol=0.0)
        assert int(result.internal_steps) == 1
        assert int(result.sim_state.solver_numeric_outputs.solver_error_state) == 1
        assert not bool(result.control_step_complete)
        assert not bool(result.step_limit_reached)

    @pytest.mark.parametrize("invalid_dt", [0.0, -0.01, np.nan])
    def test_invalid_or_nonpositive_dt_fails_without_elapsed_time(self, invalid_dt):
        result = _run(_provider([invalid_dt]))
        np.testing.assert_allclose(result.sim_state.t, 0.0, atol=0.0, rtol=0.0)
        np.testing.assert_allclose(result.sim_state.dt, 0.0, atol=0.0, rtol=0.0)
        assert int(result.internal_steps) == 1
        assert not bool(result.control_step_complete)
        assert int(result.sim_state.solver_numeric_outputs.solver_error_state) == 1

    @pytest.mark.parametrize(
        "provider, solver_limit, event_limit, expected_time, expected_steps",
        [
            (_provider([0.02, 0.02, 0.06]), 2, 0, 0.04, 2),
            (
                _provider([0.01], sawtooth_events=[True]),
                1,
                0,
                0.01,
                1,
            ),
        ],
        ids=("solver-budget", "event-budget"),
    )
    def test_budget_overflow_returns_partial_time_and_failure(
        self,
        provider,
        solver_limit,
        event_limit,
        expected_time,
        expected_steps,
    ):
        result = _run(
            provider,
            max_solver_substeps=solver_limit,
            max_event_substeps=event_limit,
        )
        np.testing.assert_allclose(
            result.sim_state.t, expected_time, atol=1e-12, rtol=0.0
        )
        assert int(result.internal_steps) == expected_steps
        assert not bool(result.control_step_complete)
        assert bool(result.step_limit_reached)
        assert int(result.sim_state.solver_numeric_outputs.solver_error_state) == 1

    def test_sawtooth_event_is_followed_by_remaining_pde_step(self):
        provider = _provider([0.01, 1.0], sawtooth_events=[True, False])
        result = _run(
            provider,
            max_solver_substeps=1,
            max_event_substeps=1,
        )
        np.testing.assert_allclose(result.sim_state.t, 0.1, atol=1e-12, rtol=0.0)
        assert int(result.internal_steps) == 2
        assert int(result.sawtooth_crashes) == 1
        assert bool(result.sim_state.solver_numeric_outputs.sawtooth_crash)
        assert bool(result.control_step_complete)

    def test_action_and_provider_are_constant_across_internal_steps(self):
        provider = _provider([0.04, 0.03, 0.03], action=3.0, gain=0.25)
        result = _run(provider)
        np.testing.assert_allclose(
            result.sim_state.action_sum, 9.0, atol=1e-12, rtol=0.0
        )
        np.testing.assert_allclose(
            result.sim_state.gain_sum, 0.75, atol=1e-12, rtol=0.0
        )


class FixedDurationTransformTest:
    @staticmethod
    def _objective(initial_value, gain):
        provider = _provider([0.04, 0.03, 0.03], gain=gain)
        result = _run(provider, state=_initial_state(initial_value))
        return result.sim_state.value

    def test_dynamic_while_and_masked_scan_primals_agree(self):
        provider = _provider([0.04, 0.03, 0.03])
        state = _initial_state()
        while_result = _run(provider, state=state)
        scan_result = _fixed_duration_step_scan(
            _synthetic_step,
            _state_is_finite,
            8,
            0,
            jnp.asarray(0.1, dtype=jnp.float64),
            state,
            jnp.asarray(0.0, dtype=jnp.float64),
            provider,
        )
        for actual, expected in zip(
            jax.tree.leaves(while_result),
            jax.tree.leaves(scan_result),
            strict=True,
        ):
            np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0.0)

    def test_jit_jvp_and_reverse_mode_match_scan_and_finite_difference(self):
        initial_value = jnp.asarray(1.3, dtype=jnp.float64)
        gain = jnp.asarray(0.4, dtype=jnp.float64)
        jitted_value = jax.jit(self._objective)(initial_value, gain)
        value, jvp = jax.jvp(
            self._objective,
            (initial_value, gain),
            (jnp.asarray(0.0), jnp.asarray(1.0)),
        )
        value_and_grad, reverse_grad = jax.value_and_grad(self._objective, argnums=1)(
            initial_value, gain
        )

        def scan_objective(g):
            return _fixed_duration_step_scan(
                _synthetic_step,
                _state_is_finite,
                8,
                0,
                jnp.asarray(0.1, dtype=jnp.float64),
                _initial_state(initial_value),
                jnp.asarray(0.0, dtype=jnp.float64),
                _provider([0.04, 0.03, 0.03], gain=g),
            ).sim_state.value

        scan_grad = jax.grad(scan_objective)(gain)
        epsilon = 1.0e-5
        finite_difference = (
            self._objective(initial_value, gain + epsilon)
            - self._objective(initial_value, gain - epsilon)
        ) / (2.0 * epsilon)

        for actual in (jitted_value, value, value_and_grad):
            np.testing.assert_allclose(actual, value, atol=1e-12, rtol=0.0)
        for actual in (jvp, reverse_grad, scan_grad, finite_difference):
            np.testing.assert_allclose(actual, scan_grad, atol=1e-6, rtol=1e-4)

    def test_vmap_matches_scalar_results(self):
        initial_values = jnp.asarray([0.8, 1.0, 1.4], dtype=jnp.float64)
        gains = jnp.asarray([0.2, 0.4, 0.6], dtype=jnp.float64)
        vmapped = jax.vmap(self._objective)(initial_values, gains)
        scalar = jnp.stack(
            [
                self._objective(value, gain)
                for value, gain in zip(initial_values, gains, strict=True)
            ]
        )
        np.testing.assert_allclose(vmapped, scalar, atol=1e-12, rtol=0.0)
