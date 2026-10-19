"""TORAX rewards and their termination behavior.

Registered rewards preserve their physical scores, then return squareplus on
ordinary transitions and its logarithm on physical or solver termination.
Invalid-state transitions return zero. Custom reward callables own their full
return value and are not transformed by the environment.
"""

import dataclasses
from collections.abc import Callable

import jax
import jax.numpy as jnp
from torax._src.pedestal_model import (
    pedestal_transition_state as pedestal_transition_state_lib,
)

from plasmax.environment.core import EnvState

RewardFn = Callable[[EnvState, jax.Array, EnvState, jax.Array], jax.Array]
Quantity = Callable[[EnvState], jax.Array]


def _valid_input(
    value: jax.Array, termination_code: jax.Array, *, fallback: float = 0.0
) -> jax.Array:
    """Mask invalid-state lanes before potentially undefined arithmetic."""
    return jnp.where(termination_code == 4, jnp.full_like(value, fallback), value)


def _terminal_reward(score: jax.Array, termination_code: jax.Array) -> jax.Array:
    """Apply positive and terminal utilities without overflow or cancellation."""
    score = _valid_input(score, termination_code)
    half = 0.5 * score
    radius = jnp.hypot(half, jnp.ones_like(half))
    absolute_half = jnp.abs(half)
    magnitude = radius + absolute_half
    positive = jnp.where(half >= 0.0, magnitude, jnp.reciprocal(magnitude))
    # log(squareplus(score)) == asinh(score / 2), including large negative scores.
    # This form avoids the squared-input overflow in JAX's asinh derivative,
    # while log1p preserves small scores whose squareplus rounds to one.
    log_magnitude = jnp.log1p(absolute_half + (radius - 1.0))
    terminal = jnp.where(half >= 0.0, log_magnitude, -log_magnitude)
    reward = jnp.where(termination_code == -1, positive, terminal)
    return jnp.where(termination_code == 4, jnp.zeros_like(reward), reward)


def _safe_barrier_state(next_state: EnvState, termination_code: jax.Array) -> EnvState:
    """Keep q reductions and their VJPs defined on invalid vmapped lanes."""
    core = dataclasses.replace(
        next_state.plasma.core,
        q_face=_valid_input(
            next_state.plasma.core.q_face, termination_code, fallback=1.0
        ),
    )
    sim = dataclasses.replace(next_state.plasma.sim, core_profiles=core)
    return dataclasses.replace(
        next_state, plasma=dataclasses.replace(next_state.plasma, sim=sim)
    )


def Q_fusion(
    state: EnvState,
    action: jax.Array,
    next_state: EnvState,
    termination_code: jax.Array,
) -> jax.Array:
    """Fusion power gain (~10 at Q=10). Prone to reward hacking at low power."""
    del state, action
    return _terminal_reward(next_state.plasma.Q_fusion, termination_code)


def beta_N(
    state: EnvState,
    action: jax.Array,
    next_state: EnvState,
    termination_code: jax.Array,
) -> jax.Array:
    """Normalised toroidal beta (~2)."""
    del state, action
    return _terminal_reward(next_state.plasma.beta_N, termination_code)


def P_diff(
    state: EnvState,
    action: jax.Array,
    next_state: EnvState,
    termination_code: jax.Array,
) -> jax.Array:
    """Fusion power minus auxiliary heating power, in GW."""
    del state, action
    fusion = _valid_input(next_state.plasma.P_fusion, termination_code)
    auxiliary = _valid_input(next_state.plasma.P_aux_total, termination_code)
    return _terminal_reward((fusion - auxiliary) * 1e-9, termination_code)


def P_diff_MW(
    state: EnvState,
    action: jax.Array,
    next_state: EnvState,
    termination_code: jax.Array,
) -> jax.Array:
    """Fusion power minus auxiliary heating power, in MW."""
    del state, action
    fusion = _valid_input(next_state.plasma.P_fusion, termination_code)
    auxiliary = _valid_input(next_state.plasma.P_aux_total, termination_code)
    return _terminal_reward((fusion - auxiliary) * 1e-6, termination_code)


def W_thermal(
    state: EnvState,
    action: jax.Array,
    next_state: EnvState,
    termination_code: jax.Array,
) -> jax.Array:
    """Thermal stored energy, in units of 100 MJ (~1 at the ITER hybrid flat-top).

    Unlike P_diff this has no auxiliary-heating penalty, so heating during the
    cold ramp-up pays off immediately instead of reading as pure cost."""
    del state, action
    thermal = _valid_input(next_state.plasma.W_thermal_total, termination_code)
    return _terminal_reward(thermal * 1e-8, termination_code)


def soft_barrier(
    quantity: Quantity, limit: float, *, slope: float = 40.0, threshold: float = 0.9
) -> RewardFn:
    """Raw soft penalty enforcing ``quantity(next_state) <= limit``:
    ~0 below ``threshold * limit``, increasingly negative above it. The ReLU
    floor keeps the normalized quantity nonnegative without capping penalties
    or their gradients for large violations.
    """

    def reward(
        state: EnvState,
        action: jax.Array,
        next_state: EnvState,
        termination_code: jax.Array,
    ) -> jax.Array:
        del state, action
        value = _valid_input(quantity(next_state), termination_code)
        x = jax.nn.relu(value / limit)
        return jax.nn.log_sigmoid(-slope * (x - threshold))

    return reward


def _safe_inverse(value: jax.Array, epsilon: float = 1e-6) -> jax.Array:
    """Invert a positive stability quantity without a singular VJP at zero."""

    return jnp.reciprocal(jnp.maximum(value, epsilon))


# Ramp-down stability barriers (PopDownGym-style log-sigmoid soft barriers on the
# quantities that go unstable as the current is brought down).
# Use the grid q minimum, as termination does; the fitted minimum can undershoot.
_RAMPDOWN_BARRIERS: tuple[RewardFn, ...] = (
    soft_barrier(lambda s: s.plasma.fgw_n_e_line_avg, 1.0),  # Greenwald fraction
    soft_barrier(lambda s: s.plasma.li3, 1.5),  # internal inductance
    soft_barrier(
        lambda s: _safe_inverse(jnp.min(s.plasma.core.q_face)), 1.0
    ),  # q_min > 1
    soft_barrier(lambda s: s.plasma.beta_N, 3.0),  # beta limit
)


def rampdown(
    state: EnvState,
    action: jax.Array,
    next_state: EnvState,
    termination_code: jax.Array,
) -> jax.Array:
    """Keep the plasma inside stability limits while Ip is ramped down.

    Unlike PopDownGym — where the agent *controls* the current and the reward
    includes an ``ip_reward`` term for driving Ip toward zero — here Ip is
    prescribed by ``profile_conditions.Ip`` (a fixed schedule), not an actuator.
    So there is no current-progress term to reward: the agent can only steer the
    heating/fuelling actuators to hold the plasma safe (Greenwald / li / q_min /
    beta_N barriers) as the current is brought down externally. The physical
    score sums log-sigmoid barriers and is near zero inside every limit. Its
    positive reward is near one; termination returns the logarithm of it.
    """
    next_state = _safe_barrier_state(next_state, termination_code)
    args = (state, action, next_state, termination_code)
    return _terminal_reward(sum(b(*args) for b in _RAMPDOWN_BARRIERS), termination_code)


_RAMPUP_LH_BARRIERS: tuple[RewardFn, ...] = (
    soft_barrier(lambda s: s.plasma.fgw_n_e_line_avg, 1.0),  # Greenwald fraction < 1
    soft_barrier(
        lambda s: _safe_inverse(jnp.min(s.plasma.core.q_face)),
        1.0 / 1.6,
    ),  # q_min > ~1.6
)


def lh_transition(
    state: EnvState,
    action: jax.Array,
    next_state: EnvState,
    termination_code: jax.Array,
    *,
    t_final: float = 100.0,
) -> jax.Array:
    """Stable ramp-up ending in a clean, sustained H-mode.

    ITER's L–H transition is timed at/after Ip flat-top, not throughout the
    ramp, so this does NOT reward crossing early:

    * ReLU P_SOL/P_LH shaping, time-weighted ``(t/t_final)**2`` — an early
      crossing contributes almost nothing; the reward for being above
      threshold only matters near the end of the ramp. Positive ratios remain
      uncapped so their gradients persist above the L–H threshold.
    * Discrete ``H_MODE`` bonus, time-weighted and summed each step, so
      *sustaining* H-mode pays more than a single crossing.
    * Back-transition penalty (``TRANSITIONING_TO_L_MODE``) discourages sitting
      on the L–H margin and dithering in/out — this is what makes it "clean".
    * q_min / Greenwald barriers held across the whole ramp.

    ``t_final=100.0`` matches the ramp-up envs (``iter/hybrid/rampup.yaml`` etc.),
    like ``rampdown``'s scenario-specific constant. The ``-back`` weight is the
    main tuning knob: too high and the agent never approaches the margin; too
    low and it dithers.
    """
    next_state = _safe_barrier_state(next_state, termination_code)
    args = (state, action, next_state, termination_code)
    mode = next_state.plasma.mode
    Mode = pedestal_transition_state_lib.ConfinementMode
    t_frac = _valid_input(next_state.plasma.t, termination_code) / t_final

    p_sol = _valid_input(next_state.plasma.P_SOL_total, termination_code)
    p_lh = _valid_input(next_state.plasma.P_LH, termination_code, fallback=1.0)
    ratio = jax.nn.relu(p_sol / p_lh)
    h_bonus = jnp.where(mode == Mode.H_MODE, 1.0, 0.0)
    back = jnp.where(mode == Mode.TRANSITIONING_TO_L_MODE, 1.0, 0.0)

    score = (
        ratio * t_frac**2
        + h_bonus * t_frac
        - 2.0 * back
        + sum(b(*args) for b in _RAMPUP_LH_BARRIERS)
    )
    return _terminal_reward(score, termination_code)


_REWARD_ALIASES: dict[str, RewardFn] = {
    "Q_fusion": Q_fusion,
    "beta_N": beta_N,
    "P_diff": P_diff,
    "P_diff_MW": P_diff_MW,
    "W_thermal": W_thermal,
    "rampdown": rampdown,
    "lh_transition": lh_transition,
}


def resolve_reward_fn(reward_fn: RewardFn | str) -> RewardFn:
    """Resolves a string alias to a callable; passes callables through."""
    if isinstance(reward_fn, str):
        if reward_fn not in _REWARD_ALIASES:
            raise ValueError(
                f"Unknown reward {reward_fn!r}. Valid: {sorted(_REWARD_ALIASES)}"
            )
        return _REWARD_ALIASES[reward_fn]
    if not callable(reward_fn):
        raise TypeError(
            f"reward_fn must be a callable or string, got {type(reward_fn).__name__!r}"
        )
    return reward_fn
