"""Differentiable, fixed-duration TORAX control stepping."""

from __future__ import annotations

import dataclasses
import functools
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
from torax._src import constants, jax_utils
from torax._src import state as torax_state


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class FixedDurationStepResult:
    """Result of one bounded physical control interval.

    ``internal_steps`` counts every TORAX call, while ``sawtooth_crashes``
    counts the event-only calls within that total. A transition is successful
    only when the complete requested duration was reached without a solver or
    budget failure.
    """

    sim_state: Any
    post_processed_outputs: Any
    internal_steps: jax.Array
    sawtooth_crashes: jax.Array
    control_step_complete: jax.Array
    step_limit_reached: jax.Array


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class _FixedDurationCarry:
    """Fixed-shape immutable carry shared by the while and scan paths."""

    remaining_dt: jax.Array
    sim_state: Any
    post_processed_outputs: Any
    internal_steps: jax.Array
    solver_substeps: jax.Array
    event_substeps: jax.Array
    inner_solver_iterations: jax.Array
    outer_solver_iterations: jax.Array
    saw_coarse_convergence: jax.Array
    failed: jax.Array
    step_limit_reached: jax.Array


def _fixed_duration_should_continue(carry: _FixedDurationCarry) -> jax.Array:
    """Return whether another accepted TORAX substep is required."""
    return (
        (carry.remaining_dt > constants.CONSTANTS.eps)
        & ~carry.failed
        & ~carry.step_limit_reached
    )


def _fixed_duration_body(
    carry: _FixedDurationCarry,
    step_fn: Callable[[Any, Any, jax.Array, Any], tuple[Any, Any]],
    state_is_finite: Callable[[Any, Any], jax.Array],
    runtime_params_provider: Any,
    max_solver_substeps: int,
    max_event_substeps: int,
) -> _FixedDurationCarry:
    """Take one event or PDE substep and update bounded-loop statistics."""
    next_state, next_post = step_fn(
        carry.sim_state,
        carry.post_processed_outputs,
        carry.remaining_dt,
        runtime_params_provider,
    )
    numeric = next_state.solver_numeric_outputs
    is_event = jnp.asarray(numeric.sawtooth_crash, dtype=jnp.bool_)
    returned_dt = jnp.asarray(next_state.dt)
    valid_dt = jnp.isfinite(returned_dt) & (returned_dt > 0.0)
    finite_state = state_is_finite(next_state, next_post)
    solver_failed = numeric.solver_error_state == 1

    # A failed nonlinear solve still reports how much physical time it tried to
    # advance, matching TORAX's forward-only fixed_time_step(). Invalid dt does
    # not contribute elapsed time.
    elapsed_dt = jnp.where(
        valid_dt,
        jnp.minimum(returned_dt, carry.remaining_dt),
        jnp.zeros_like(returned_dt),
    )
    remaining_dt = jnp.maximum(
        carry.remaining_dt - elapsed_dt,
        jnp.zeros_like(carry.remaining_dt),
    )
    internal_steps = carry.internal_steps + jnp.int32(1)
    solver_substeps = carry.solver_substeps + (~is_event).astype(jnp.int32)
    event_substeps = carry.event_substeps + is_event.astype(jnp.int32)
    category_budget_exceeded = (solver_substeps > max_solver_substeps) | (
        event_substeps > max_event_substeps
    )
    total_budget_exhausted = (
        internal_steps >= max_solver_substeps + max_event_substeps
    ) & (remaining_dt > constants.CONSTANTS.eps)
    step_limit_reached = category_budget_exceeded | total_budget_exhausted

    return _FixedDurationCarry(
        remaining_dt=remaining_dt,
        sim_state=next_state,
        post_processed_outputs=next_post,
        internal_steps=internal_steps,
        solver_substeps=solver_substeps,
        event_substeps=event_substeps,
        inner_solver_iterations=(
            carry.inner_solver_iterations + numeric.inner_solver_iterations
        ),
        outer_solver_iterations=(
            carry.outer_solver_iterations + numeric.outer_solver_iterations
        ),
        saw_coarse_convergence=(
            carry.saw_coarse_convergence | (numeric.solver_error_state == 2)
        ),
        failed=carry.failed | solver_failed | ~valid_dt | ~finite_state,
        step_limit_reached=step_limit_reached,
    )


def _initial_fixed_duration_carry(
    control_dt: jax.Array,
    input_state: Any,
    previous_post_processed_outputs: Any,
) -> _FixedDurationCarry:
    int_dtype = jax_utils.get_int_dtype()
    zero_int = jnp.array(0, dtype=int_dtype)
    invalid_control_dt = ~jnp.isfinite(control_dt) | (control_dt <= 0.0)
    return _FixedDurationCarry(
        remaining_dt=control_dt,
        sim_state=input_state,
        post_processed_outputs=previous_post_processed_outputs,
        internal_steps=zero_int,
        solver_substeps=zero_int,
        event_substeps=zero_int,
        inner_solver_iterations=zero_int,
        outer_solver_iterations=zero_int,
        saw_coarse_convergence=jnp.asarray(False),
        failed=invalid_control_dt,
        step_limit_reached=jnp.asarray(False),
    )


def _finalize_fixed_duration_step(
    carry: _FixedDurationCarry,
    control_dt: jax.Array,
    input_state: Any,
) -> FixedDurationStepResult:
    """Normalize time and aggregate diagnostics over all internal calls."""
    complete = (
        (carry.remaining_dt <= constants.CONSTANTS.eps)
        & ~carry.failed
        & ~carry.step_limit_reached
    )
    valid_control_dt = jnp.isfinite(control_dt) & (control_dt > 0.0)
    elapsed_dt = jnp.where(
        valid_control_dt,
        jnp.where(
            complete,
            control_dt,
            jnp.maximum(control_dt - carry.remaining_dt, 0.0),
        ),
        jnp.zeros_like(control_dt),
    )
    aggregate_error = jnp.where(
        carry.failed | carry.step_limit_reached,
        jnp.array(1, dtype=jax_utils.get_int_dtype()),
        jnp.where(
            carry.saw_coarse_convergence,
            jnp.array(2, dtype=jax_utils.get_int_dtype()),
            jnp.array(0, dtype=jax_utils.get_int_dtype()),
        ),
    )
    aggregate_numeric = torax_state.SolverNumericOutputs(
        outer_solver_iterations=carry.outer_solver_iterations,
        solver_error_state=aggregate_error,
        inner_solver_iterations=carry.inner_solver_iterations,
        sawtooth_crash=carry.event_substeps > 0,
    )
    output_state = dataclasses.replace(
        carry.sim_state,
        t=input_state.t + elapsed_dt,
        dt=elapsed_dt,
        solver_numeric_outputs=aggregate_numeric,
    )
    return FixedDurationStepResult(
        sim_state=output_state,
        post_processed_outputs=carry.post_processed_outputs,
        internal_steps=carry.internal_steps,
        sawtooth_crashes=carry.event_substeps,
        control_step_complete=complete,
        step_limit_reached=carry.step_limit_reached,
    )


def _fixed_duration_step_while(
    step_fn: Callable[[Any, Any, jax.Array, Any], tuple[Any, Any]],
    state_is_finite: Callable[[Any, Any], jax.Array],
    max_solver_substeps: int,
    max_event_substeps: int,
    control_dt: jax.Array,
    input_state: Any,
    previous_post_processed_outputs: Any,
    runtime_params_provider: Any,
) -> FixedDurationStepResult:
    """Efficient primal implementation that executes only required substeps."""
    body = functools.partial(
        _fixed_duration_body,
        step_fn=step_fn,
        state_is_finite=state_is_finite,
        runtime_params_provider=runtime_params_provider,
        max_solver_substeps=max_solver_substeps,
        max_event_substeps=max_event_substeps,
    )
    # Keep the overwhelmingly common first call outside dynamic control flow so
    # XLA can optimize the large linear solve as a straight-line program.
    carry = body(
        _initial_fixed_duration_carry(
            control_dt,
            input_state,
            previous_post_processed_outputs,
        )
    )
    if max_solver_substeps + max_event_substeps > 1:
        carry = jax.lax.while_loop(
            _fixed_duration_should_continue,
            body,
            carry,
        )
    return _finalize_fixed_duration_step(carry, control_dt, input_state)


def _fixed_duration_step_scan(
    step_fn: Callable[[Any, Any, jax.Array, Any], tuple[Any, Any]],
    state_is_finite: Callable[[Any, Any], jax.Array],
    max_solver_substeps: int,
    max_event_substeps: int,
    control_dt: jax.Array,
    input_state: Any,
    previous_post_processed_outputs: Any,
    runtime_params_provider: Any,
) -> FixedDurationStepResult:
    """Fixed-length masked reference used by the custom tangent rule."""
    body = functools.partial(
        _fixed_duration_body,
        step_fn=step_fn,
        state_is_finite=state_is_finite,
        runtime_params_provider=runtime_params_provider,
        max_solver_substeps=max_solver_substeps,
        max_event_substeps=max_event_substeps,
    )
    checkpointed_body = jax.checkpoint(body, prevent_cse=False)

    def scan_body(carry: _FixedDurationCarry, _) -> tuple[_FixedDurationCarry, None]:
        next_carry = jax.lax.cond(
            _fixed_duration_should_continue(carry),
            checkpointed_body,
            lambda current: current,
            carry,
        )
        return next_carry, None

    carry, _ = jax.lax.scan(
        scan_body,
        _initial_fixed_duration_carry(
            control_dt, input_state, previous_post_processed_outputs
        ),
        xs=None,
        length=max_solver_substeps + max_event_substeps,
    )
    return _finalize_fixed_duration_step(carry, control_dt, input_state)


@functools.partial(jax.custom_jvp, nondiff_argnums=(0, 1, 2, 3))
def fixed_duration_step(
    step_fn: Callable[[Any, Any, jax.Array, Any], tuple[Any, Any]],
    state_is_finite: Callable[[Any, Any], jax.Array],
    max_solver_substeps: int,
    max_event_substeps: int,
    control_dt: jax.Array,
    input_state: Any,
    previous_post_processed_outputs: Any,
    runtime_params_provider: Any,
) -> FixedDurationStepResult:
    """Advance one differentiable, statically bounded control interval."""
    return _fixed_duration_step_while(
        step_fn,
        state_is_finite,
        max_solver_substeps,
        max_event_substeps,
        control_dt,
        input_state,
        previous_post_processed_outputs,
        runtime_params_provider,
    )


@fixed_duration_step.defjvp
def _fixed_duration_step_jvp(
    step_fn,
    state_is_finite,
    max_solver_substeps,
    max_event_substeps,
    primals,
    tangents,
):
    """Replay sequential accepted steps through the transposable scan path."""
    primal_out = fixed_duration_step(
        step_fn,
        state_is_finite,
        max_solver_substeps,
        max_event_substeps,
        *primals,
    )
    scan_fn = functools.partial(
        _fixed_duration_step_scan,
        step_fn,
        state_is_finite,
        max_solver_substeps,
        max_event_substeps,
    )
    _, tangent_out = jax.jvp(scan_fn, primals, tangents)
    return primal_out, tangent_out


__all__ = ["FixedDurationStepResult", "fixed_duration_step"]
