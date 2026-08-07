"""Gymnax compatibility for scalar Envelope environments.

The adapter is intentionally kept with plasmax's optional training tooling:
Gymnax and Rejax are not dependencies of the core environment package.
"""

from __future__ import annotations

import operator
from functools import cached_property
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from envelope import BatchedSpace, Continuous, Discrete, Environment, PyTreeSpace
from gymnax.environments import spaces as gymnax_spaces
from gymnax.environments.environment import EnvParams

from plasmax.wrappers import find_max_steps

__all__ = ["EnvelopeGymnax", "GymnaxMultiDiscrete", "to_typed_key"]


def to_typed_key(key: jax.Array) -> jax.Array:
    """Return a scalar typed JAX key from a typed or legacy Gymnax key."""
    if jnp.issubdtype(key.dtype, jax.dtypes.prng_key):
        if key.shape != ():
            raise ValueError("typed PRNG keys must be scalar")
        return key
    if key.shape != (2,) or key.dtype != jnp.uint32:
        raise ValueError(
            "key must be a scalar typed key or a legacy uint32 key with shape (2,)"
        )
    return jax.random.wrap_key_data(key)


class GymnaxMultiDiscrete(gymnax_spaces.Space):
    """Gymnax-compatible independent categorical action dimensions."""

    def __init__(self, nvec: Any):
        values = np.asarray(nvec)
        if values.ndim != 1:
            raise ValueError("GymnaxMultiDiscrete requires a one-dimensional nvec")
        if not np.issubdtype(values.dtype, np.integer) or np.any(values <= 0):
            raise ValueError("nvec entries must be positive integers")
        self.nvec = tuple(int(value) for value in values.reshape(-1))
        self.n = jnp.asarray(values)
        self.shape = self.n.shape
        self.dtype = self.n.dtype

    def sample(self, key: jax.Array) -> jax.Array:
        return jax.random.randint(
            key,
            shape=self.shape,
            minval=0,
            maxval=self.n,
            dtype=self.dtype,
        )

    def contains(self, value: jax.Array) -> jax.Array:
        value = jnp.asarray(value)
        if value.shape != self.shape:
            return jnp.asarray(False)
        return jnp.all((value >= 0) & (value < self.n) & (value == jnp.floor(value)))


def _convert_space(space):
    if isinstance(space, (BatchedSpace, PyTreeSpace)):
        raise TypeError(
            "EnvelopeGymnax supports scalar Continuous and Discrete spaces only; "
            f"got {type(space).__name__}"
        )
    if isinstance(space, Continuous):
        return gymnax_spaces.Box(
            low=jnp.asarray(space.low),
            high=jnp.asarray(space.high),
            shape=space.shape,
            dtype=space.dtype,
        )
    if isinstance(space, Discrete):
        n = np.asarray(space.n)
        if n.ndim == 0:
            return gymnax_spaces.Discrete(int(n))
        return GymnaxMultiDiscrete(n)
    raise TypeError(
        "EnvelopeGymnax supports Continuous and Discrete spaces only; "
        f"got {type(space).__name__}"
    )


def _validate_max_steps(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError(f"max_steps_in_episode must be an integer, got {value!r}")
    try:
        max_steps = operator.index(value)
    except TypeError as error:
        raise ValueError(
            f"max_steps_in_episode must be an integer, got {value!r}"
        ) from error
    if max_steps <= 0:
        raise ValueError(f"max_steps_in_episode must be positive, got {max_steps}")
    return max_steps


class EnvelopeGymnax:
    """Expose a scalar Envelope environment through the Gymnax API.

    Gymnax has one ``done`` flag, so Envelope termination and truncation are
    combined. The returned ``info`` remains the terminal Envelope emission and
    therefore retains both original flags. On a boundary the observation and
    state are reset for the next training transition, as Gymnax expects.
    """

    def __init__(
        self,
        env: Environment,
        *,
        max_steps_in_episode: int | None = None,
    ) -> None:
        if not isinstance(env, Environment):
            raise TypeError(f"env must be an Envelope Environment, got {type(env)!r}")
        if max_steps_in_episode is None:
            max_steps_in_episode = find_max_steps(env)
        if max_steps_in_episode is None:
            raise ValueError(
                "max_steps_in_episode is required when the Envelope environment "
                "has no TruncationWrapper"
            )
        self.envelope_env = env
        self.max_steps_in_episode = _validate_max_steps(max_steps_in_episode)

    @cached_property
    def default_params(self) -> EnvParams:
        return EnvParams(max_steps_in_episode=self.max_steps_in_episode)

    @cached_property
    def _action_space(self):
        return _convert_space(self.envelope_env.action_space)

    @cached_property
    def _observation_space(self):
        return _convert_space(self.envelope_env.observation_space)

    def action_space(self, params: EnvParams | None = None):
        del params
        return self._action_space

    def observation_space(self, params: EnvParams | None = None):
        del params
        return self._observation_space

    @property
    def unwrapped(self):
        return self.envelope_env.unwrapped

    def reset(
        self, key: jax.Array, params: EnvParams | None = None
    ) -> tuple[jax.Array, Any]:
        del params
        state, info = self.envelope_env.init(to_typed_key(key))
        return info.obs, state

    def step(
        self,
        key: jax.Array,
        state: Any,
        action: Any,
        params: EnvParams | None = None,
    ) -> tuple[jax.Array, Any, jax.Array, jax.Array, Any]:
        del params
        next_state, info = self.envelope_env.step(state, action)
        done = jnp.asarray(info.terminated) | jnp.asarray(info.truncated)
        reset_key = to_typed_key(jax.random.fold_in(key, 0))

        def reset(_):
            reset_state, reset_info = self.envelope_env.reset(next_state, reset_key)
            return reset_state, reset_info.obs

        def keep(_):
            return next_state, info.obs

        output_state, obs = jax.lax.cond(done, reset, keep, operand=None)
        return obs, output_state, info.reward, done, info
