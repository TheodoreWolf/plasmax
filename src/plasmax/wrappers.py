"""Composable Envelope-native wrappers for plasmax environments.

Wrappers follow Envelope's two-value lifecycle: ``init`` and ``reset`` receive
typed PRNG keys, while ``step`` receives only state and action.  Stochastic
wrappers therefore own and advance their own key in a :class:`WrappedState`.
"""

from __future__ import annotations

import dataclasses
import operator
from collections.abc import Callable, Sequence
from functools import cached_property
from typing import Any, override

import jax
import jax.numpy as jnp
import numpy as np
from envelope import (
    Continuous,
    Discrete,
    TruncationWrapper,
    WrappedState,
    Wrapper,
    field,
    static_field,
)
from envelope.environment import Info
from envelope.typing import Key, PyTree, State

from plasmax.spaces import ObsLayout

_NOISE_SAMPLE_STREAM = 0x4E4F4953  # "NOIS"
_NOISE_STATE_STREAM = 0x4E4F4954
_DELAY_STATE_STREAM = 0x44454C59  # "DELY"


def _require_typed_key(key: Key) -> None:
    if (
        not hasattr(key, "dtype")
        or not jnp.issubdtype(key.dtype, jax.dtypes.prng_key)
        or key.shape != ()
    ):
        raise ValueError("key must be a scalar typed, new-style `jax.random.key`.")


@dataclasses.dataclass
class SensorNoiseConfig:
    """Per-sensor relative noise levels (fractional std by sensor name)."""

    relative_std: dict[str, float]

    def to_noise_scale(self, layout: ObsLayout) -> jax.Array:
        """Return a flat array of per-observation relative standard deviations."""
        scale = np.zeros(layout.size, dtype=np.float32)
        for name, std in self.relative_std.items():
            scale[layout.slice_of(name)] = std
        return jnp.asarray(scale)


class NoiseEnvState(WrappedState):
    """Inner environment state plus the sensor-noise PRNG stream."""

    key: jax.Array = field()


class NoiseWrapper(Wrapper):
    """Add multiplicative Gaussian observation noise.

    The emitted observation is ``obs + normal * noise_scale * abs(obs)``.
    The wrapper's key is independent of the inner environment's key and is
    advanced once per step.  Reset starts a fresh wrapper-local stream.
    """

    noise_scale: jax.Array = field()

    def __post_init__(self) -> None:
        noise_scale = jnp.asarray(self.noise_scale)
        if noise_scale.shape != self.env.observation_space.shape:
            raise ValueError(
                "noise_scale shape must match observation space: "
                f"{noise_scale.shape} != {self.env.observation_space.shape}"
            )
        object.__setattr__(self, "noise_scale", noise_scale)
        super().__post_init__()

    def _add_noise(self, info: Info, key: Key) -> Info:
        obs = info.obs
        noisy_obs = obs + jax.random.normal(
            key, obs.shape, dtype=obs.dtype
        ) * self.noise_scale * jnp.abs(obs)
        return info.update(obs=noisy_obs)

    @override
    def init(self, key: Key) -> tuple[NoiseEnvState, Info]:
        _require_typed_key(key)
        # Wrapper-local streams are derived without consuming the root reset
        # key. The physical environment therefore sees exactly the same key
        # regardless of whether this realistic sensor wrapper is present.
        noise_key = jax.random.fold_in(key, _NOISE_SAMPLE_STREAM)
        next_key = jax.random.fold_in(key, _NOISE_STATE_STREAM)
        inner_state, info = self.env.init(key)
        state = NoiseEnvState(inner_state=inner_state, key=next_key)
        return state, self._add_noise(info, noise_key)

    @override
    def reset(self, state: NoiseEnvState, key: Key) -> tuple[NoiseEnvState, Info]:
        _require_typed_key(key)
        noise_key = jax.random.fold_in(key, _NOISE_SAMPLE_STREAM)
        next_key = jax.random.fold_in(key, _NOISE_STATE_STREAM)
        inner_state, info = self.env.reset(state.inner_state, key)
        next_state = NoiseEnvState(inner_state=inner_state, key=next_key)
        return next_state, self._add_noise(info, noise_key)

    @override
    def step(self, state: NoiseEnvState, action: PyTree) -> tuple[NoiseEnvState, Info]:
        next_key, noise_key = jax.random.split(state.key)
        inner_state, info = self.env.step(state.inner_state, action)
        next_state = NoiseEnvState(inner_state=inner_state, key=next_key)
        return next_state, self._add_noise(info, noise_key)


@dataclasses.dataclass
class ProfileResolutionConfig:
    """Per-profile number of observation points.

    Profiles not listed keep full n_rho.
    """

    n_obs: dict[str, int]


@dataclasses.dataclass
class ObsFilterConfig:
    """Named filter specifying which profiles and scalars to retain."""

    profiles: list[str]
    scalars: list[str]


def _subsample_indices(start: int, stop: int, n_obs: int) -> list[int]:
    """Return ``n_obs`` indices linearly spaced over ``[start, stop)``."""
    n = stop - start
    if n_obs == n:
        return list(range(start, stop))
    raw = np.linspace(0, n - 1, n_obs)
    return [start + int(np.round(x)) for x in raw]


def _build_indexed_layout(
    inner_layout: ObsLayout,
    profile_names: Sequence[str],
    scalar_names: Sequence[str],
    profile_indices_fn: Callable[[str, slice], Sequence[int]],
) -> tuple[list[int], ObsLayout]:
    """Build selected observation indices and their compact named layout."""
    indices: list[int] = []
    profile_slices: dict[str, slice] = {}
    scalar_slices: dict[str, slice] = {}
    offset = 0
    for name in profile_names:
        source = inner_layout.profile_slices[name]
        profile_indices = list(profile_indices_fn(name, source))
        indices.extend(profile_indices)
        profile_slices[name] = slice(offset, offset + len(profile_indices))
        offset += len(profile_indices)
    for name in scalar_names:
        source = inner_layout.scalar_slices[name]
        indices.extend(range(source.start, source.stop))
        scalar_slices[name] = slice(offset, offset + source.stop - source.start)
        offset += source.stop - source.start
    return indices, ObsLayout(
        profile_slices=profile_slices,
        scalar_slices=scalar_slices,
        profile_names=tuple(profile_names),
        scalar_names=tuple(scalar_names),
    )


class ObsFilterWrapper(Wrapper):
    """Retain a subset of observation elements by index."""

    indices: Sequence[int] = static_field()
    # ObsLayout is immutable wrapper metadata, but its Mapping fields are unhashable.
    layout: ObsLayout | None = static_field(default=None, unsafe=True)

    def __post_init__(self) -> None:
        indices = tuple(int(index) for index in self.indices)
        object.__setattr__(self, "indices", indices)
        super().__post_init__()

    @classmethod
    def from_obs_config(cls, env, config: ObsFilterConfig) -> ObsFilterWrapper:
        """Construct a filter from named profiles and scalars."""
        inner_layout = env.obs_layout()
        indices, layout = _build_indexed_layout(
            inner_layout,
            config.profiles,
            config.scalars,
            lambda _name, source: list(range(source.start, source.stop)),
        )
        return cls(env=env, indices=indices, layout=layout)

    @classmethod
    def from_resolution_config(
        cls, env, config: ProfileResolutionConfig
    ) -> ObsFilterWrapper:
        """Subsample profiles while preserving their order and all scalars."""
        inner_layout = env.obs_layout()

        def profile_indices(name: str, source: slice) -> list[int]:
            n_obs = config.n_obs.get(name, source.stop - source.start)
            n_available = source.stop - source.start
            if n_obs > n_available:
                raise ValueError(
                    f"profile resolution for {name!r} cannot exceed "
                    f"the {n_available} available points, got {n_obs}"
                )
            return _subsample_indices(source.start, source.stop, n_obs)

        indices, layout = _build_indexed_layout(
            inner_layout,
            inner_layout.profile_names,
            inner_layout.scalar_names,
            profile_indices,
        )
        return cls(env=env, indices=indices, layout=layout)

    def _filter_info(self, info: Info) -> Info:
        return info.update(obs=info.obs[jnp.asarray(self.indices, dtype=jnp.int32)])

    @override
    def init(self, key: Key) -> tuple[State, Info]:
        state, info = self.env.init(key)
        return state, self._filter_info(info)

    @override
    def reset(self, state: State, key: Key) -> tuple[State, Info]:
        state, info = self.env.reset(state, key)
        return state, self._filter_info(info)

    @override
    def step(self, state: State, action: PyTree) -> tuple[State, Info]:
        state, info = self.env.step(state, action)
        return state, self._filter_info(info)

    @override
    @cached_property
    def observation_space(self) -> Continuous:
        inner = self.env.observation_space
        if not isinstance(inner, Continuous):
            raise TypeError("ObsFilterWrapper requires a Continuous observation space")
        indices = jnp.asarray(self.indices, dtype=jnp.int32)
        return Continuous(
            low=jnp.asarray(inner.low)[indices],
            high=jnp.asarray(inner.high)[indices],
        )

    def obs_layout(self) -> ObsLayout:
        if self.layout is None:
            raise RuntimeError(
                "ObsFilterWrapper was constructed without a layout. "
                "Use from_obs_config or pass layout=..."
            )
        return self.layout


class ActionRescaleWrapper(Wrapper):
    """Map policy actions from ``[-1, 1]`` to physical actuator bounds."""

    def __post_init__(self) -> None:
        inner = self.env.action_space
        if not isinstance(inner, Continuous):
            raise TypeError("ActionRescaleWrapper requires a Continuous action space")
        super().__post_init__()

    @override
    def step(self, state: State, action: PyTree) -> tuple[State, Info]:
        return self.env.step(state, self.to_physical(action))

    @override
    @cached_property
    def action_space(self) -> Continuous:
        inner = self.env.action_space
        return Continuous(
            low=jnp.full(inner.shape, -1.0, dtype=inner.dtype),
            high=jnp.full(inner.shape, 1.0, dtype=inner.dtype),
        )

    def to_physical(self, action: jax.Array) -> jax.Array:
        action = jnp.asarray(action)
        low = jnp.asarray(self.env.action_space.low)
        high = jnp.asarray(self.env.action_space.high)
        return low + (action + 1.0) / 2.0 * (high - low)

    def from_physical(self, action: jax.Array) -> jax.Array:
        action = jnp.asarray(action)
        low = jnp.asarray(self.env.action_space.low)
        high = jnp.asarray(self.env.action_space.high)
        return 2.0 * (action - low) / (high - low) - 1.0


@dataclasses.dataclass
class ActionQuantizeConfig:
    """Per-actuator bin counts, keyed by actuator name."""

    bins: dict[str, int]

    def to_bin_counts(self, actuator_names: Sequence[str]) -> tuple[int, ...]:
        """Order bin counts to match the environment's actuator vector."""
        return tuple(self.bins[name] for name in actuator_names)


class QuantizeActionWrapper(Wrapper):
    """Decode one discrete bin index per actuator into normalized actions."""

    bin_counts: Sequence[int] = static_field()

    def __post_init__(self) -> None:
        bin_counts = tuple(int(count) for count in self.bin_counts)
        if len(bin_counts) != self.env.action_space.shape[0]:
            raise ValueError("bin counts must match the action dimensions")
        if not bin_counts or min(bin_counts) < 2:
            raise ValueError("each action dimension needs at least two bins")
        object.__setattr__(self, "bin_counts", bin_counts)
        super().__post_init__()

    @property
    def _bin_counts_array(self) -> jax.Array:
        return jnp.asarray(self.bin_counts, dtype=jnp.float32)

    def _decode(self, action: jax.Array) -> jax.Array:
        return -1.0 + 2.0 * jnp.asarray(action, dtype=jnp.float32) / (
            self._bin_counts_array - 1.0
        )

    @override
    def step(self, state: State, action: PyTree) -> tuple[State, Info]:
        return self.env.step(state, self._decode(action))

    def to_physical(self, action: jax.Array) -> jax.Array:
        return self.env.to_physical(self._decode(action))

    def from_physical(self, action: jax.Array) -> jax.Array:
        normalized = self.env.from_physical(action)
        bins = 0.5 * (normalized + 1.0) * (self._bin_counts_array - 1.0)
        return jnp.rint(bins).astype(jnp.int32)

    @override
    @cached_property
    def action_space(self) -> Discrete:
        return Discrete(n=jnp.asarray(self.bin_counts, dtype=jnp.int32))


class HistoryEnvState(WrappedState):
    """Rolling observation/action buffers around an inner environment state."""

    obs_history: jax.Array = field()
    action_history: jax.Array = field()


class ObsHistoryWrapper(Wrapper):
    """Frame-stack the last ``k`` observations and policy-space actions."""

    k: int = static_field()

    def __post_init__(self) -> None:
        if self.k < 1:
            raise ValueError(f"history length k must be >= 1, got {self.k}")
        # Materialize any cached space properties before this wrapper is traced.
        _ = self.env.observation_space
        _ = self.env.action_space
        super().__post_init__()

    @property
    def _obs_dim(self) -> int:
        return int(self.env.observation_space.shape[0])

    @property
    def _action_dim(self) -> int:
        return int(self.env.action_space.shape[0])

    @staticmethod
    def _flatten(obs_history: jax.Array, action_history: jax.Array) -> jax.Array:
        return jnp.concatenate([obs_history.reshape(-1), action_history.reshape(-1)])

    def _initial_state(
        self, inner_state: State, info: Info
    ) -> tuple[HistoryEnvState, Info]:
        obs_history = jnp.broadcast_to(info.obs, (self.k, self._obs_dim))
        base_state = unwrap_to_env_state(inner_state)
        initial_action = jnp.zeros((self._action_dim,), dtype=info.obs.dtype)
        if hasattr(base_state, "prev_action"):
            initial_action = jnp.asarray(base_state.prev_action, dtype=info.obs.dtype)
            for layer in iter_wrappers(self.env):
                if isinstance(layer, ActionRescaleWrapper):
                    initial_action = layer.from_physical(initial_action)
                    break
        action_history = jnp.broadcast_to(initial_action, (self.k, self._action_dim))
        state = HistoryEnvState(
            inner_state=inner_state,
            obs_history=obs_history,
            action_history=action_history,
        )
        return state, info.update(obs=self._flatten(obs_history, action_history))

    @override
    def init(self, key: Key) -> tuple[HistoryEnvState, Info]:
        inner_state, info = self.env.init(key)
        return self._initial_state(inner_state, info)

    @override
    def reset(self, state: HistoryEnvState, key: Key) -> tuple[HistoryEnvState, Info]:
        inner_state, info = self.env.reset(state.inner_state, key)
        return self._initial_state(inner_state, info)

    @override
    def step(
        self, state: HistoryEnvState, action: PyTree
    ) -> tuple[HistoryEnvState, Info]:
        inner_state, info = self.env.step(state.inner_state, action)
        action = jnp.asarray(action, dtype=state.action_history.dtype)
        obs_history = jnp.concatenate([state.obs_history[1:], info.obs[None]], axis=0)
        action_history = jnp.concatenate(
            [state.action_history[1:], action[None]], axis=0
        )
        next_state = HistoryEnvState(
            inner_state=inner_state,
            obs_history=obs_history,
            action_history=action_history,
        )
        return next_state, info.update(obs=self._flatten(obs_history, action_history))

    @override
    @cached_property
    def observation_space(self) -> Continuous:
        inner = self.env.observation_space
        action = self.env.action_space
        if not isinstance(inner, Continuous) or not isinstance(action, Continuous):
            raise TypeError(
                "ObsHistoryWrapper requires Continuous observation and action spaces"
            )
        return Continuous(
            low=jnp.concatenate(
                [jnp.tile(inner.low, self.k), jnp.tile(action.low, self.k)]
            ),
            high=jnp.concatenate(
                [jnp.tile(inner.high, self.k), jnp.tile(action.high, self.k)]
            ),
        )

    def obs_layout(self) -> ObsLayout:
        inner = self.env.obs_layout()
        offset = (self.k - 1) * self._obs_dim

        def shift(source: slice) -> slice:
            return slice(source.start + offset, source.stop + offset)

        return ObsLayout(
            profile_slices={
                name: shift(source) for name, source in inner.profile_slices.items()
            },
            scalar_slices={
                name: shift(source) for name, source in inner.scalar_slices.items()
            },
            profile_names=inner.profile_names,
            scalar_names=inner.scalar_names,
            vector_size=self.observation_space.shape[0],
        )


@dataclasses.dataclass
class ObsDelayConfig:
    """Per-sensor probability of repeating the previously emitted value."""

    repeat_prob: dict[str, float]

    def to_hold_prob(self, layout: ObsLayout) -> jax.Array:
        """Return a flat array of per-observation hold probabilities."""
        probability = np.zeros(layout.size, dtype=np.float32)
        for name, value in self.repeat_prob.items():
            probability[layout.slice_of(name)] = value
        return jnp.asarray(probability)


class DelayEnvState(WrappedState):
    """Inner state, last emitted observation, and delay PRNG stream."""

    last_emitted_obs: jax.Array = field()
    key: jax.Array = field()


class ObsDelayWrapper(Wrapper):
    """Stochastically hold previous observation values per dimension."""

    hold_prob: jax.Array = field()

    def __post_init__(self) -> None:
        hold_prob = jnp.asarray(self.hold_prob)
        if hold_prob.shape != self.env.observation_space.shape:
            raise ValueError(
                "hold_prob shape must match observation space: "
                f"{hold_prob.shape} != {self.env.observation_space.shape}"
            )
        object.__setattr__(self, "hold_prob", hold_prob)
        super().__post_init__()

    @override
    def init(self, key: Key) -> tuple[DelayEnvState, Info]:
        _require_typed_key(key)
        state_key = jax.random.fold_in(key, _DELAY_STATE_STREAM)
        inner_state, info = self.env.init(key)
        state = DelayEnvState(
            inner_state=inner_state, last_emitted_obs=info.obs, key=state_key
        )
        return state, info

    @override
    def reset(self, state: DelayEnvState, key: Key) -> tuple[DelayEnvState, Info]:
        _require_typed_key(key)
        state_key = jax.random.fold_in(key, _DELAY_STATE_STREAM)
        inner_state, info = self.env.reset(state.inner_state, key)
        next_state = DelayEnvState(
            inner_state=inner_state, last_emitted_obs=info.obs, key=state_key
        )
        return next_state, info

    @override
    def step(self, state: DelayEnvState, action: PyTree) -> tuple[DelayEnvState, Info]:
        next_key, hold_key = jax.random.split(state.key)
        inner_state, info = self.env.step(state.inner_state, action)
        hold_mask = jax.random.bernoulli(hold_key, self.hold_prob)
        emitted = jnp.where(hold_mask, state.last_emitted_obs, info.obs)
        next_state = DelayEnvState(
            inner_state=inner_state,
            last_emitted_obs=emitted,
            key=next_key,
        )
        return next_state, info.update(obs=emitted)


class TimeAwareEnvState(WrappedState):
    """Inner state plus the simulator time at the start of this episode."""

    episode_start: jax.Array = field()


class TimeAwareWrapper(Wrapper):
    """Append simulator time (or a world-model step counter) to observations."""

    @staticmethod
    def _time(state: State) -> jax.Array:
        base_state = unwrap_to_env_state(state)
        return base_state.plasma.t if hasattr(base_state, "plasma") else base_state.t

    @classmethod
    def _append_elapsed(
        cls, obs: jax.Array, state: State, episode_start: jax.Array
    ) -> jax.Array:
        elapsed = cls._time(state) - episode_start
        return jnp.concatenate([obs, jnp.asarray(elapsed, dtype=obs.dtype)[None]])

    @override
    def init(self, key: Key) -> tuple[TimeAwareEnvState, Info]:
        inner_state, info = self.env.init(key)
        episode_start = self._time(inner_state)
        state = TimeAwareEnvState(inner_state=inner_state, episode_start=episode_start)
        return state, info.update(
            obs=self._append_elapsed(info.obs, inner_state, episode_start)
        )

    @override
    def reset(
        self, state: TimeAwareEnvState, key: Key
    ) -> tuple[TimeAwareEnvState, Info]:
        inner_state, info = self.env.reset(state.inner_state, key)
        episode_start = self._time(inner_state)
        next_state = TimeAwareEnvState(
            inner_state=inner_state, episode_start=episode_start
        )
        return next_state, info.update(
            obs=self._append_elapsed(info.obs, inner_state, episode_start)
        )

    @override
    def step(
        self, state: TimeAwareEnvState, action: PyTree
    ) -> tuple[TimeAwareEnvState, Info]:
        inner_state, info = self.env.step(state.inner_state, action)
        next_state = TimeAwareEnvState(
            inner_state=inner_state, episode_start=state.episode_start
        )
        return next_state, info.update(
            obs=self._append_elapsed(info.obs, inner_state, state.episode_start)
        )

    @override
    @cached_property
    def observation_space(self) -> Continuous:
        inner = self.env.observation_space
        if not isinstance(inner, Continuous):
            raise TypeError("TimeAwareWrapper requires a Continuous observation space")
        return Continuous(
            low=jnp.concatenate(
                [jnp.asarray(inner.low), jnp.asarray([0.0], dtype=inner.dtype)]
            ),
            high=jnp.concatenate(
                [jnp.asarray(inner.high), jnp.asarray([jnp.inf], dtype=inner.dtype)]
            ),
        )

    def obs_layout(self) -> ObsLayout:
        inner = self.env.obs_layout()
        scalar_slices = dict(inner.scalar_slices)
        scalar_slices["elapsed_time"] = slice(inner.size, inner.size + 1)
        return ObsLayout(
            profile_slices=inner.profile_slices,
            scalar_slices=scalar_slices,
            profile_names=inner.profile_names,
            scalar_names=(*inner.scalar_names, "elapsed_time"),
        )


class PlasmaxTruncationWrapper(TruncationWrapper):
    """Episode horizon with TORAX termination-over-truncation precedence."""

    def __post_init__(self) -> None:
        if isinstance(self.max_steps, bool):
            raise ValueError(f"max_steps must be an integer, got {self.max_steps!r}")
        try:
            max_steps = operator.index(self.max_steps)
        except TypeError as error:
            raise ValueError(
                f"max_steps must be an integer, got {self.max_steps!r}"
            ) from error
        object.__setattr__(self, "max_steps", max_steps)
        if max_steps <= 0:
            raise ValueError(f"max_steps must be positive, got {self.max_steps}")
        safe_max_steps = getattr(self.env.unwrapped, "safe_max_steps", None)
        if safe_max_steps is not None and max_steps > safe_max_steps:
            raise ValueError(
                f"max_steps={self.max_steps} exceeds the configured safe horizon "
                f"of {safe_max_steps}"
            )
        super().__post_init__()

    @override
    def step(self, state: WrappedState, action: PyTree) -> tuple[WrappedState, Info]:
        state, info = super().step(state, action)
        # A genuine failure on the cutoff transition is a termination only.
        truncated = jnp.logical_and(info.truncated, jnp.logical_not(info.terminated))
        return state, info.update(truncated=truncated)


def iter_wrappers(env):
    """Yield each Envelope wrapper followed by the innermost environment."""
    while True:
        yield env
        if not isinstance(env, Wrapper):
            return
        env = env.env


def find_max_steps(env) -> int | None:
    """Return the first explicit Envelope truncation horizon in a wrapper stack."""
    for layer in iter_wrappers(env):
        if isinstance(layer, TruncationWrapper):
            return int(layer.max_steps)
    return None


def unwrap_to_env_state(state: Any) -> Any:
    """Descend stateful Envelope wrapper layers to the base environment state."""
    while isinstance(state, WrappedState):
        state = state.inner_state
    return state
