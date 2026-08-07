from collections.abc import Callable

import jax
import jax.numpy as jnp
from torax._src.pedestal_model import (
    pedestal_transition_state as pedestal_transition_state_lib,
)

from plasmax.environment.core import EnvState

RewardFn = Callable[[jax.Array, EnvState, jax.Array, EnvState], jax.Array]
Quantity = Callable[[EnvState], jax.Array]


def Q_fusion(last_action, state, action, next_state):
    """Fusion power gain (~10 at Q=10). Prone to reward hacking at low power."""
    return next_state.plasma.Q_fusion


def beta_N(last_action, state, action, next_state):
    """Normalised toroidal beta (~2)."""
    return next_state.plasma.beta_N


def P_diff(last_action, state, action, next_state):
    """Fusion power minus auxiliary heating power, in GW."""
    return (next_state.plasma.P_fusion - next_state.plasma.P_aux_total) * 1e-9


def W_thermal(last_action, state, action, next_state):
    """Thermal stored energy, in units of 100 MJ (~1 at the ITER hybrid flat-top).

    Unlike P_diff this has no auxiliary-heating penalty, so heating during the
    cold ramp-up pays off immediately instead of reading as pure cost."""
    return next_state.plasma.W_thermal_total * 1e-8


def soft_barrier(
    quantity: Quantity, limit: float, *, slope: float = 40.0, threshold: float = 0.9
) -> RewardFn:
    """Soft penalty enforcing ``quantity(next_state) <= limit`` (PopDownGym-style):
    ~0 while below ``threshold * limit``, dropping sharply (log-sigmoid) toward 0."""

    def reward(last_action, state, action, next_state):
        x = jnp.clip(quantity(next_state) / limit, 0.0, 1.1)
        return jax.nn.log_sigmoid(-slope * (x - threshold))

    return reward


def _safe_inverse(value: jax.Array, epsilon: float = 1e-6) -> jax.Array:
    """Invert a positive stability quantity without a singular VJP at zero."""

    return jnp.reciprocal(jnp.maximum(value, epsilon))


# Ramp-down stability barriers (PopDownGym-style log-sigmoid soft barriers on the
# quantities that go unstable as the current is brought down).
_RAMPDOWN_BARRIERS: tuple[RewardFn, ...] = (
    soft_barrier(lambda s: s.plasma.fgw_n_e_line_avg, 1.0),  # Greenwald fraction
    soft_barrier(lambda s: s.plasma.li3, 1.5),  # internal inductance
    soft_barrier(lambda s: _safe_inverse(s.plasma.q_min), 1.0),  # q_min > 1
    soft_barrier(lambda s: s.plasma.beta_N, 3.0),  # beta limit
)


def rampdown(last_action, state, action, next_state):
    """Keep the plasma inside stability limits while Ip is ramped down.

    Unlike PopDownGym — where the agent *controls* the current and the reward
    includes an ``ip_reward`` term for driving Ip toward zero — here Ip is
    prescribed by ``profile_conditions.Ip`` (a fixed schedule), not an actuator.
    So there is no current-progress term to reward: the agent can only steer the
    heating/fuelling actuators to hold the plasma safe (Greenwald / li / q_min /
    beta_N barriers) as the current is brought down externally. Pure sum of the
    log-sigmoid stability barriers (0 when comfortably inside every limit).
    """
    args = (last_action, state, action, next_state)
    return sum(b(*args) for b in _RAMPDOWN_BARRIERS)


_RAMPUP_LH_BARRIERS: tuple[RewardFn, ...] = (
    soft_barrier(lambda s: s.plasma.fgw_n_e_line_avg, 1.0),  # Greenwald fraction < 1
    soft_barrier(
        lambda s: _safe_inverse(s.plasma.q_min),
        1.0 / 1.6,
    ),  # q_min > ~1.6
)


def lh_transition(last_action, state, action, next_state, *, t_final: float = 100.0):
    """Stable ramp-up ending in a clean, sustained H-mode.

    ITER's L–H transition is timed at/after Ip flat-top, not throughout the
    ramp, so this does NOT reward crossing early:

    * P_SOL/P_LH shaping, time-weighted ``(t/t_final)**2`` — an early crossing
      contributes almost nothing; the reward for being above threshold only
      matters near the end of the ramp.
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
    args = (last_action, state, action, next_state)
    mode = next_state.plasma.mode
    Mode = pedestal_transition_state_lib.ConfinementMode
    t_frac = next_state.plasma.t / t_final

    ratio = jnp.clip(next_state.plasma.P_SOL_total / next_state.plasma.P_LH, 0.0, 1.5)
    h_bonus = jnp.where(mode == Mode.H_MODE, 1.0, 0.0)
    back = jnp.where(mode == Mode.TRANSITIONING_TO_L_MODE, 1.0, 0.0)

    return (
        ratio * t_frac**2
        + h_bonus * t_frac
        - 2.0 * back
        + sum(b(*args) for b in _RAMPUP_LH_BARRIERS)
    )


_REWARD_ALIASES: dict[str, RewardFn] = {
    "Q_fusion": Q_fusion,
    "beta_N": beta_N,
    "P_diff": P_diff,
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
