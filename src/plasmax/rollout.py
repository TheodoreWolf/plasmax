"""Trajectory collection utilities for Envelope-native TORAX environments."""

import dataclasses
import functools
from collections.abc import Callable
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from envelope import AutoResetWrapper, PooledInitVmapWrapper, WrappedState, Wrapper
from torax._src.output_tools import output
from torax._src.state import SimError
from torax._src.torax_pydantic import model_config

from plasmax.wrappers import unwrap_to_env_state


class TrajectoryStep(NamedTuple):
    """One step of a collected trajectory.

    When returned by :func:`collect_episode` all fields are stacked along axis 0,
    so each has shape ``(num_steps, ...)``. When returned by
    :func:`collect_episodes` they have shape ``(n_seeds, num_steps, ...)``.

    The transition that reaches an episode boundary is retained with
    ``valid=True``. Remaining fixed-length scan slots repeat the terminal state
    with ``valid=False`` and zero reward/boundary flags, so they cannot be
    mistaken for transitions from a second episode.

    Attributes:
      obs: Flat observation vector before the step, shape ``(..., obs_dim)``.
      action: Action requested at this step, in physical units if the environment
        exposes ``to_physical()``, shape ``(..., act_dim)``.
      reward: Scalar reward received, shape ``(...,)``.
      next_obs: Flat observation vector after the step, shape
        ``(..., obs_dim)``.
      terminated: Whether the transition ended because of a terminal condition,
        shape ``(...,)``.
      truncated: Whether the transition reached an external episode horizon,
        shape ``(...,)``.
      valid: Whether this fixed-length slot contains a real transition, shape
        ``(...,)``.
      env_state: Full Envelope state pytree at the post-step time, including its
        wrapper-state nesting.
      info: Envelope ``Info`` pytree returned by the step, with the same leading
        stack dimensions.
      done: Derived ``valid & (terminated | truncated)`` boundary flag.
    """

    obs: jax.Array
    action: jax.Array
    reward: jax.Array
    next_obs: jax.Array
    terminated: jax.Array
    truncated: jax.Array
    valid: jax.Array
    env_state: Any
    info: Any

    @property
    def done(self) -> jax.Array:
        return self.valid & (self.terminated | self.truncated)


# Heavy per-step SimState fields eval never reads (geometry is static; transport/
# source/edge outputs are unused). Plus: keep only the CoreProfiles fields the
# eval figures plot; the other ~27 (currents, psi, charge states, face grids…)
# dominate the stacked trajectory. Nulling them in the *emitted* copy cuts the
# eval buffer ~4-5x (34 -> 7 KB/step/seed). The scan carry keeps the full state.
_LEAN_DROP_SIM_FIELDS = ("geometry", "core_transport", "core_sources", "edge_outputs")
_LEAN_KEEP_PROFILE_FIELDS = frozenset({"T_e", "T_i", "n_e", "q_face"})


def _strip_state(state: Any) -> Any:
    """Return state with heavy, eval-unused TORAX leaves nulled.

    Wrapper-owned state is retained by rebuilding each Envelope ``WrappedState``
    around the stripped base state. Requires a TORAX base ``EnvState``; do not
    use ``lean=True`` on world-model environments.
    """
    if isinstance(state, WrappedState):
        return state.replace(inner_state=_strip_state(state.inner_state))

    es = state
    cp = es.plasma.core
    lean_cp = dataclasses.replace(
        cp,
        **{
            f.name: None
            for f in dataclasses.fields(cp)
            if f.name not in _LEAN_KEEP_PROFILE_FIELDS
        },
    )
    lean_sim = dataclasses.replace(
        es.plasma.sim,
        core_profiles=lean_cp,
        **{f: None for f in _LEAN_DROP_SIM_FIELDS},
    )
    return dataclasses.replace(
        es,
        plasma=dataclasses.replace(es.plasma, sim=lean_sim),
    )


def _iter_envelope_wrappers(env):
    while isinstance(env, Wrapper):
        yield env
        env = env.env


def _reject_autoreset(env) -> None:
    """Reject wrappers that replace the terminal state before collection sees it."""
    autoreset_types = (AutoResetWrapper, PooledInitVmapWrapper)
    if any(
        isinstance(layer, autoreset_types) for layer in _iter_envelope_wrappers(env)
    ):
        raise ValueError(
            "trajectory collection requires a non-autoresetting environment; "
            "collect from the truncation-wrapped scalar environment before adding "
            "Envelope autoreset or pooled-init wrappers"
        )


def _padding_info(info, obs):
    """Return shape-stable inactive-slot info without changing its extras tree."""
    updates = {
        "obs": obs,
        "reward": jnp.zeros_like(info.reward),
        "terminated": jnp.zeros_like(info.terminated, dtype=jnp.bool_),
        "truncated": jnp.zeros_like(info.truncated, dtype=jnp.bool_),
    }
    if hasattr(info, "termination_code"):
        updates["termination_code"] = jnp.full_like(info.termination_code, -1)
    return info.update(**updates)


class _StaticEnvironment:
    """Identity-hash an environment without hashing its array-valued fields.

    JAX requires static arguments to be hashable. Envelope environments are
    frozen dataclasses, so their generated hash recursively visits wrapper
    fields such as realistic sensor-noise and observation-delay arrays. Those
    arrays are deliberately part of the environment pytree and are not Python
    hashable. Collection still needs the environment itself to be static so
    method dispatch and TORAX configuration remain compile-time constants.

    Equality by wrapped-object identity preserves JIT cache hits for repeated
    collection with the same environment while keeping distinct environment
    instances in distinct cache entries.
    """

    __slots__ = ("value",)

    def __init__(self, value: Any):
        self.value = value

    def __hash__(self) -> int:
        return id(self.value)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _StaticEnvironment) and self.value is other.value


def _collect_episode_impl(
    act: Callable[[jax.Array, jax.Array], jax.Array],
    env: Any,
    rng: jax.Array,
    num_steps: int,
    lean: bool,
) -> TrajectoryStep:
    env_key, policy_key = jax.random.split(rng)
    env_state, init_info = env.init(env_key)
    obs = init_info.obs

    zero_action = jnp.zeros(env.action_space.shape, dtype=env.action_space.dtype)
    emitted_zero_action = (
        env.to_physical(zero_action) if hasattr(env, "to_physical") else zero_action
    )

    def _real_step(carry):
        obs, env_state, policy_key, _active, _last_info = carry
        policy_key, act_key = jax.random.split(policy_key)
        # Policies may inherit JAX's process-wide x64 default. Normalize their
        # output to the public action-space dtype so real and padded scan
        # branches have identical structures and dtypes.
        action = jnp.asarray(act(obs, act_key), dtype=env.action_space.dtype)
        next_state, info = env.step(env_state, action)
        boundary = info.terminated | info.truncated
        emitted_action = (
            env.to_physical(action) if hasattr(env, "to_physical") else action
        )
        step = TrajectoryStep(
            obs=obs,
            action=emitted_action,
            reward=info.reward,
            next_obs=info.obs,
            terminated=info.terminated,
            truncated=info.truncated,
            valid=jnp.ones_like(boundary, dtype=jnp.bool_),
            env_state=_strip_state(next_state) if lean else next_state,
            info=info,
        )
        next_carry = (
            info.obs,
            next_state,
            policy_key,
            ~boundary,
            info,
        )
        return next_carry, step

    def _padding_step(carry):
        obs, env_state, _policy_key, active, last_info = carry
        info = _padding_info(last_info, obs)
        step = TrajectoryStep(
            obs=obs,
            action=emitted_zero_action,
            reward=info.reward,
            next_obs=obs,
            terminated=info.terminated,
            truncated=info.truncated,
            valid=jnp.zeros_like(active, dtype=jnp.bool_),
            env_state=_strip_state(env_state) if lean else env_state,
            info=info,
        )
        return carry, step

    def _scan_step(carry, _):
        return jax.lax.cond(carry[3], _real_step, _padding_step, carry)

    active = ~(init_info.terminated | init_info.truncated)
    carry = (obs, env_state, policy_key, active, init_info)
    _, trajectory = jax.lax.scan(_scan_step, carry, None, length=num_steps)
    return trajectory


@functools.partial(jax.jit, static_argnames=("act", "static_env", "num_steps", "lean"))
def _collect_episode_jit(
    act: Callable[[jax.Array, jax.Array], jax.Array],
    static_env: _StaticEnvironment,
    rng: jax.Array,
    num_steps: int,
    lean: bool,
) -> TrajectoryStep:
    return _collect_episode_impl(act, static_env.value, rng, num_steps, lean)


def collect_episode(
    act: Callable[[jax.Array, jax.Array], jax.Array],
    env: Any,
    rng: jax.Array,
    num_steps: int,
    lean: bool = False,
) -> TrajectoryStep:
    """Collect one non-autoresetting episode in a fixed-length compiled scan.

    A boundary transition is retained. Later slots freeze the scan carry and are
    marked invalid, so callers can keep static shapes without starting another
    episode or stepping an already-finished simulator.
    """
    _reject_autoreset(env)
    return _collect_episode_jit(act, _StaticEnvironment(env), rng, num_steps, lean)


@functools.partial(
    jax.jit,
    static_argnames=("act", "static_env", "num_steps", "n_seeds", "lean"),
)
def _collect_episodes_jit(
    act: Callable[[jax.Array, jax.Array], jax.Array],
    static_env: _StaticEnvironment,
    rng: jax.Array,
    num_steps: int,
    n_seeds: int,
    lean: bool,
) -> TrajectoryStep:
    env = static_env.value
    keys = jax.random.split(rng, n_seeds)
    return jax.vmap(
        functools.partial(
            _collect_episode_impl,
            act,
            env,
            num_steps=num_steps,
            lean=lean,
        )
    )(keys)


def collect_episodes(
    act: Callable[[jax.Array, jax.Array], jax.Array],
    env: Any,
    rng: jax.Array,
    num_steps: int,
    n_seeds: int,
    lean: bool = False,
) -> TrajectoryStep:
    """Collect ``n_seeds`` independent fixed-shape episodes with ``jax.vmap``.

    Ended lanes retain frozen observable carry and state. Under JAX batching,
    however, a vmapped ``lax.cond`` may evaluate inactive branch work as part of
    the compiled SIMD computation; callers should not interpret masking as a
    per-lane compute-saving guarantee.
    """
    _reject_autoreset(env)
    return _collect_episodes_jit(
        act, _StaticEnvironment(env), rng, num_steps, n_seeds, lean
    )


def make_collect_callback(num_steps: int, n_seeds: int = 1) -> Callable:
    """Return a callback that collects trajectories from an algorithm policy."""

    def _callback(algo, ts, rng):
        return collect_episodes(
            algo.make_act(ts),
            algo.env,
            rng,
            num_steps=num_steps,
            n_seeds=n_seeds,
        )

    return _callback


def trajectories_to_state_history(
    traj: TrajectoryStep,
    config: model_config.ToraxConfig,
    seed_idx: int = 0,
) -> output.StateHistory:
    """Convert valid full TORAX states, including the terminal state.

    Trajectories collected with ``lean=True`` intentionally omit fields required
    by TORAX output conversion and are rejected with a clear error.
    """
    env_state = unwrap_to_env_state(traj.env_state)
    valid = np.asarray(traj.valid, dtype=bool)

    sim_state = env_state.plasma.sim
    if sim_state.geometry is None or sim_state.core_profiles.psi is None:
        raise ValueError(
            "state-history conversion requires a full trajectory; collect with "
            "lean=False"
        )

    # Slice seed dimension if present (check by time field rank).
    if np.ndim(env_state.plasma.t) == 2:
        env_state = jax.tree_util.tree_map(lambda x: x[seed_idx], env_state)
        valid = valid[seed_idx]

    valid_indices = np.flatnonzero(valid)
    if valid_indices.size == 0:
        raise ValueError("trajectory contains no valid TORAX transitions")

    stacked_sim_state = env_state.plasma.sim
    stacked_ppo = env_state.plasma.post
    sim_states = [
        jax.tree_util.tree_map(lambda x, i=i: np.asarray(x[i]), stacked_sim_state)
        for i in valid_indices
    ]
    ppos = tuple(
        jax.tree_util.tree_map(lambda x, i=i: np.asarray(x[i]), stacked_ppo)
        for i in valid_indices
    )
    return output.StateHistory(sim_states, ppos, SimError.NO_ERROR, config)
