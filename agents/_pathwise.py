"""Small native lifecycle shared by the clone-only APG and SAPO agents.

Algorithms own their differentiable rollouts and optimizer updates. Reporting
boundaries never change the update budget or reset the training environments.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any, Self

import jax
import jax.numpy as jnp
import numpy as np
from envelope import AutoResetWrapper, Continuous, PooledInitVmapWrapper, Wrapper
from flax import struct

from training.envelope_gymnax import to_typed_key


@struct.dataclass
class PathwiseState:
    """Dynamic collection state and counters, extended with optimizer states."""

    rng: jax.Array
    env_state: Any
    obs: jax.Array
    global_step: jax.Array
    simulator_steps: jax.Array
    update_index: jax.Array
    failed: jax.Array
    failure_step: jax.Array


@dataclasses.dataclass(frozen=True, eq=False)
class PathwiseAgent:
    """Common configuration, validation and evaluation for native agents.

    ``total_timesteps`` rounds up once to a complete algorithm update. APG
    counts committed transitions; SAPO counts its on-policy rollout transitions.
    ``simulator_steps`` additionally exposes APG's valid lookahead transitions.
    These are logical transition counts, not guarantees about physical work
    elided by a batched JAX conditional.

    ``episode_steps`` limits evaluation only. Training uses the environment's
    termination/truncation flags; callers supply any training truncation wrapper.
    """

    env: Any
    total_timesteps: int = 10_000_000
    eval_freq: int = 1_000_000
    num_rollouts: int = 64
    horizon: int = 32
    hidden_sizes: tuple[int, ...] = (64, 64)
    remat: bool = True
    eval_n_envs: int = 16
    eval_seed: int = 10_000
    episode_steps: int | None = None
    eval_callback: Callable | None = None
    action_low: jax.Array = dataclasses.field(init=False, repr=False)
    action_high: jax.Array = dataclasses.field(init=False, repr=False)
    action_loc: jax.Array = dataclasses.field(init=False, repr=False)
    action_scale: jax.Array = dataclasses.field(init=False, repr=False)
    obs_shape: tuple[int, ...] = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        # Materialize cached spaces before they can capture compiled tracers.
        action_space, obs_space = self.env.action_space, self.env.observation_space
        if not isinstance(action_space, Continuous):
            raise TypeError("pathwise agents require continuous actions")
        if len(action_space.shape) != 1 or len(obs_space.shape) != 1:
            raise ValueError("pathwise agents require flat actions and observations")
        low, high = np.asarray(action_space.low), np.asarray(action_space.high)
        if not (np.all(np.isfinite(low)) and np.all(np.isfinite(high))):
            raise ValueError("pathwise agents require finite action bounds")
        if not np.all(high > low):
            raise ValueError("each action upper bound must exceed its lower bound")
        layer = self.env
        while isinstance(layer, Wrapper):
            if isinstance(layer, (AutoResetWrapper, PooledInitVmapWrapper)):
                raise ValueError(
                    "pathwise agents require a non-autoresetting environment"
                )
            layer = layer.env
        episode_steps = self.episode_steps
        if episode_steps is None:
            episode_steps = getattr(self.env, "max_steps", None)
        if episode_steps is None or episode_steps <= 0:
            raise ValueError("provide positive episode_steps or an env.max_steps")
        sizes = (
            self.total_timesteps,
            self.eval_freq,
            self.num_rollouts,
            self.horizon,
            self.eval_n_envs,
            *self.hidden_sizes,
        )
        if any(size <= 0 for size in sizes):
            raise ValueError("training, network and evaluation sizes must be positive")
        object.__setattr__(self, "episode_steps", int(episode_steps))
        object.__setattr__(self, "obs_shape", tuple(obs_space.shape))
        object.__setattr__(self, "action_low", jnp.asarray(low, jnp.float32))
        object.__setattr__(self, "action_high", jnp.asarray(high, jnp.float32))
        object.__setattr__(self, "action_loc", (self.action_low + self.action_high) / 2)
        object.__setattr__(
            self, "action_scale", (self.action_high - self.action_low) / 2
        )

    @classmethod
    def create(cls, env: Any, **kwargs: Any) -> Self:
        return cls(env=env, **kwargs)

    @property
    def config(self) -> dict[str, Any]:
        """Expose configuration without copying the simulator or its closures."""
        return {
            field.name: getattr(self, field.name)
            for field in dataclasses.fields(self)
            if field.init
        }

    @property
    def steps_per_update(self) -> int:
        raise NotImplementedError

    @property
    def num_updates(self) -> int:
        return (
            self.total_timesteps + self.steps_per_update - 1
        ) // self.steps_per_update

    def initial_state_fields(self, rng: jax.Array) -> dict[str, Any]:
        rng, env_key = jax.random.split(to_typed_key(rng))
        states, info = jax.vmap(self.env.init)(
            jax.random.split(env_key, self.num_rollouts)
        )
        zero = jnp.asarray(0, jnp.int32)
        failed = jnp.any(info.terminated | info.truncated)
        return dict(
            rng=rng,
            env_state=states,
            obs=jnp.asarray(info.obs, jnp.float32),
            global_step=zero,
            simulator_steps=zero,
            update_index=zero,
            failed=failed,
            failure_step=jnp.where(failed, 0, -1).astype(jnp.int32),
        )

    def init_state(self, rng: jax.Array) -> PathwiseState:
        raise NotImplementedError

    def empty_diagnostics(self) -> dict[str, jax.Array]:
        raise NotImplementedError

    def update(
        self, state: PathwiseState
    ) -> tuple[PathwiseState, dict[str, jax.Array], jax.Array]:
        """Return a candidate, scalar diagnostics and whether it is finite."""
        raise NotImplementedError

    def make_act(
        self, state: PathwiseState, deterministic: bool = False
    ) -> Callable[[jax.Array, jax.Array], jax.Array]:
        raise NotImplementedError

    def _train_step(
        self, state: PathwiseState
    ) -> tuple[PathwiseState, dict[str, jax.Array]]:
        def active(
            current: PathwiseState,
        ) -> tuple[PathwiseState, dict[str, jax.Array]]:
            candidate, metrics, safe = self.update(current)
            accepted = jax.lax.cond(safe, lambda: candidate, lambda: current)
            next_step = current.global_step + self.steps_per_update
            accepted = accepted.replace(
                global_step=next_step,
                simulator_steps=current.simulator_steps
                + metrics["train/simulator_steps"].astype(jnp.int32),
                update_index=current.update_index + 1,
                failed=~safe,
                failure_step=jnp.where(safe, -1, next_step).astype(jnp.int32),
            )
            # Full simulator states and optimizer states never link updates.
            return jax.tree.map(jax.lax.stop_gradient, accepted), metrics

        return jax.lax.cond(
            state.failed,
            lambda current: (current, self.empty_diagnostics()),
            active,
            state,
        )

    def _evaluate(
        self, state: PathwiseState, diagnostics: dict[str, jax.Array]
    ) -> tuple[Any, dict[str, jax.Array]]:
        eval_key = jax.random.key(self.eval_seed)
        if self.eval_callback is not None:
            return self.eval_callback(self, state, eval_key, diagnostics), {}
        from plasmax import collect_episode

        act = self.make_act(state)

        def one(key: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
            trajectory = collect_episode(
                act, self.env, key, num_steps=self.episode_steps
            )
            nonfinite = trajectory.valid & ~jnp.isfinite(trajectory.reward)
            return (
                jnp.sum(trajectory.reward),
                jnp.sum(trajectory.valid),
                jnp.sum(nonfinite),
            )

        keys = jax.vmap(lambda i: jax.random.fold_in(eval_key, i))(
            jnp.arange(self.eval_n_envs)
        )
        returns, lengths, nonfinite = jax.vmap(one)(keys)
        rate = nonfinite.sum().astype(jnp.float32) / jnp.maximum(lengths.sum(), 1)
        return (returns, lengths), {"evaluation/nonfinite_reward_rate": rate}

    def train(self, rng: jax.Array) -> tuple[PathwiseState, dict[str, Any]]:
        state = self.init_state(to_typed_key(rng))
        initial_metrics = self.empty_diagnostics()
        initial_evaluation, eval_metrics = self._evaluate(state, initial_metrics)
        initial_metrics = {**initial_metrics, **eval_metrics}
        size = self.steps_per_update
        boundaries = sorted(
            {
                min((step + size - 1) // size, self.num_updates)
                for step in range(self.eval_freq, self.total_timesteps, self.eval_freq)
            }
            | {self.num_updates}
        )

        def interval(current: PathwiseState, end: jax.Array) -> tuple:
            start = current.update_index

            def step(_: int, carry: tuple) -> tuple:
                current, accumulated = carry
                current, metrics = self._train_step(current)
                return current, jax.tree.map(jnp.add, accumulated, metrics)

            current, metrics = jax.lax.fori_loop(
                start, end, step, (current, self.empty_diagnostics())
            )
            count = jnp.maximum(current.update_index - start, 1)
            metrics = {
                name: value if name == "train/simulator_steps" else value / count
                for name, value in metrics.items()
            }
            evaluation, eval_metrics = self._evaluate(current, metrics)
            return current, (
                current.global_step,
                evaluation,
                {**metrics, **eval_metrics},
            )

        state, (steps, evaluation, metrics) = jax.lax.scan(
            interval, state, jnp.asarray(boundaries, jnp.int32)
        )

        def prepend(first: jax.Array, rest: jax.Array) -> jax.Array:
            return jnp.concatenate((jnp.asarray(first)[None], rest), axis=0)

        return state, {
            "global_step": prepend(jnp.asarray(0, jnp.int32), steps),
            "evaluation": jax.tree.map(prepend, initial_evaluation, evaluation),
            "diagnostics": jax.tree.map(prepend, initial_metrics, metrics),
        }


__all__ = ["PathwiseAgent", "PathwiseState"]
