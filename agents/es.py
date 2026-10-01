"""Full-episode evolution strategies for feedback policies and actuator knots."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np
import optax
from envelope import Continuous
from evosax.algorithms import CMA_ES, Open_ES
from flax import struct

from agents.backprop import (
    DeterministicPolicy,
    open_loop_action,
    policy_action,
    source_time_grid,
)
from agents.direct_gradient import make_optimizer, tree_is_finite
from plasmax import collect_episode
from training.envelope_gymnax import to_typed_key


@dataclasses.dataclass(frozen=True)
class _PolicyParameterization:
    policy: DeterministicPolicy
    observation_shape: tuple[int, ...]

    def init_params(self, key: jax.Array) -> Any:
        return self.policy.init(key, jnp.zeros(self.observation_shape, jnp.float32))[
            "params"
        ]

    def action(self, params: Any, observation: jax.Array) -> jax.Array:
        return policy_action(self.policy, params, observation)


@dataclasses.dataclass(frozen=True)
class _KnotParameterization:
    theta0: jax.Array
    source_times: jax.Array
    time_index: int

    def init_params(self, key: jax.Array) -> jax.Array:
        del key
        return self.theta0

    def action(self, params: jax.Array, observation: jax.Array) -> jax.Array:
        return open_loop_action(params, observation, self.source_times, self.time_index)


@struct.dataclass
class ESTrainState:
    params: Any
    es_state: Any
    global_step: jax.Array
    alive_steps: jax.Array
    update_index: jax.Array
    failed: jax.Array
    failure_step: jax.Array


@dataclasses.dataclass(frozen=True, eq=False)
class ESAgent:
    """One population optimizer, independent of the controller parameterization.

    ``population_size`` counts candidates (including both antithetic signs), and
    ``num_rollouts`` counts full episodes per candidate. Each generation shares
    rollout seeds across its population and starts from fresh environment resets.
    """

    env: Any
    parameterization: Literal["policy", "open_loop"] = "policy"
    strategy: Literal["open_es", "cma_es"] = "open_es"
    total_timesteps: int = 10_000_000
    eval_freq: int = 1_000_000
    population_size: int = 64
    num_rollouts: int = 1
    sigma: float = 0.01
    learning_rate: float = 1e-3
    grad_clip: float = 1.0
    hidden_sizes: tuple[int, ...] = (64, 64)
    num_knots: int = 10
    eval_n_envs: int = 16
    eval_seed: int = 10_000
    episode_steps: int | None = None
    source_times: jax.Array | None = None
    init_seed: int = 0
    eval_callback: Callable | None = None
    deterministic: bool = dataclasses.field(default=True, init=False)
    controller: _PolicyParameterization | _KnotParameterization = dataclasses.field(
        init=False, repr=False
    )
    es: Any = dataclasses.field(init=False, repr=False)
    es_params: Any = dataclasses.field(init=False, repr=False)
    _ask: Callable = dataclasses.field(init=False, repr=False)
    _tell: Callable = dataclasses.field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.parameterization not in ("policy", "open_loop"):
            raise ValueError(f"unknown parameterization {self.parameterization!r}")
        if self.strategy not in ("open_es", "cma_es"):
            raise ValueError(f"unknown ES strategy {self.strategy!r}")
        if not isinstance(self.env.action_space, Continuous):
            raise ValueError("ES requires continuous actions")
        if self.population_size < 2:
            raise ValueError("population_size must be at least two")
        if self.strategy == "open_es" and self.population_size % 2:
            raise ValueError("OpenES population_size must be even")
        if not np.isfinite(self.sigma) or self.sigma <= 0:
            raise ValueError("sigma must be finite and positive")
        steps = self.env.max_steps if self.episode_steps is None else self.episode_steps
        object.__setattr__(self, "episode_steps", int(steps))
        if (
            min(self.episode_steps, self.num_rollouts, self.eval_freq, self.eval_n_envs)
            <= 0
        ):
            raise ValueError(
                "episode_steps, num_rollouts and evaluation sizes must be positive"
            )
        if self.total_timesteps < self.generation_steps:
            raise ValueError(
                "total_timesteps must cover at least one complete generation"
            )

        if self.parameterization == "policy":
            controller = _PolicyParameterization(
                DeterministicPolicy(self.env.action_space.shape[0], self.hidden_sizes),
                self.env.observation_space.shape,
            )
        else:
            if self.num_knots <= 0:
                raise ValueError("num_knots must be positive")
            try:
                clock = self.env.obs_layout().slice_of("elapsed_time")
            except (KeyError, ValueError, AttributeError) as error:
                raise ValueError("open-loop ES requires TimeAwareWrapper") from error
            if clock.stop - clock.start != 1:
                raise ValueError("elapsed_time must be a scalar observation")
            times = np.asarray(
                source_time_grid(self.env, self.episode_steps)
                if self.source_times is None
                else self.source_times
            )
            if (
                times.shape != (self.episode_steps,)
                or not np.all(np.isfinite(times))
                or np.any(np.diff(times) <= 0)
            ):
                raise ValueError(
                    "source_times must contain one increasing finite time "
                    "per source step"
                )
            controller = _KnotParameterization(
                jnp.zeros(
                    (
                        min(self.num_knots, self.episode_steps),
                        self.env.action_space.shape[0],
                    ),
                    jnp.float32,
                ),
                jnp.asarray(times),
                clock.start,
            )
        object.__setattr__(self, "controller", controller)

        # TORAX enables x64 globally; confine ES arithmetic to float32 without
        # changing precision inside environment initialization or simulation.
        with jax.enable_x64(False):
            prototype = controller.init_params(jax.random.key(self.init_seed))
            if self.strategy == "open_es":
                es = Open_ES(
                    population_size=self.population_size,
                    solution=prototype,
                    optimizer=make_optimizer(self.learning_rate, self.grad_clip),
                    std_schedule=optax.constant_schedule(self.sigma),
                )
                params = es.default_params
            else:
                es = CMA_ES(population_size=self.population_size, solution=prototype)
                params = es.default_params.replace(std_init=jnp.asarray(self.sigma))
        object.__setattr__(self, "es", es)
        object.__setattr__(self, "es_params", params)

        # Keep the context inside each wrapper too: vmap retraces its body.
        def ask(key: jax.Array, state: Any, es_params: Any) -> tuple[Any, Any]:
            with jax.enable_x64(False):
                return es.ask(key, state, es_params)

        def tell(
            key: jax.Array,
            population: Any,
            fitness: jax.Array,
            state: Any,
            es_params: Any,
        ) -> tuple[Any, Any]:
            with jax.enable_x64(False):
                return es.tell(key, population, fitness, state, es_params)

        # Seed batching can change covariance rounding and rotate degenerate
        # CMA eigenspaces. Keep its updates per seed; rollouts remain batched.
        if self.strategy == "cma_es":
            ask = jax.custom_batching.sequential_vmap(ask)
            tell = jax.custom_batching.sequential_vmap(tell)
        object.__setattr__(self, "_ask", ask)
        object.__setattr__(self, "_tell", tell)

    @classmethod
    def create(cls, env: Any, **kwargs: Any) -> ESAgent:
        return cls(env, **kwargs)

    @property
    def generation_steps(self) -> int:
        return self.population_size * self.num_rollouts * self.episode_steps

    def init_state(self, rng: jax.Array) -> ESTrainState:
        param_key, es_key = jax.random.split(to_typed_key(rng))
        with jax.enable_x64(False):
            params = self.controller.init_params(param_key)
            es_state = self.es.init(es_key, params, self.es_params)
        zero = jnp.asarray(0, jnp.int32)
        return ESTrainState(
            params,
            es_state,
            zero,
            zero,
            zero,
            jnp.asarray(False),
            jnp.asarray(-1, jnp.int32),
        )

    def make_act(
        self, state: ESTrainState, deterministic: bool | None = None
    ) -> Callable:
        del deterministic
        return lambda obs, rng: self.controller.action(state.params, obs)

    def _rollouts(self, params: Any, keys: jax.Array) -> tuple[jax.Array, jax.Array]:
        def one(key: jax.Array) -> tuple[jax.Array, jax.Array]:
            trajectory = collect_episode(
                lambda obs, rng: self.controller.action(params, obs),
                self.env,
                key,
                num_steps=self.episode_steps,
            )
            return trajectory.reward.sum(), trajectory.valid.sum(dtype=jnp.int32)

        return jax.vmap(one)(keys)

    def update(self, state: ESTrainState, rng: jax.Array) -> ESTrainState:
        def active(_: None) -> ESTrainState:
            ask_key, rollout_key, tell_key = jax.random.split(to_typed_key(rng), 3)
            with jax.enable_x64(False):
                population, asked = self._ask(ask_key, state.es_state, self.es_params)
            keys = jax.random.split(rollout_key, self.num_rollouts)
            returns, lengths = jax.vmap(self._rollouts, in_axes=(0, None))(
                population, keys
            )
            fitness = -returns.mean(axis=1).astype(jnp.float32)

            def tell(_: None) -> tuple[Any, jax.Array]:
                with jax.enable_x64(False):
                    proposed, _ = self._tell(
                        tell_key, population, fitness, asked, self.es_params
                    )
                safe = tree_is_finite(proposed)
                return jax.lax.cond(
                    safe, lambda: proposed, lambda: state.es_state
                ), safe

            # Rank shaping must never turn a nonfinite return into a usable rank.
            es_state, safe = jax.lax.cond(
                jnp.all(jnp.isfinite(returns)) & jnp.all(jnp.isfinite(fitness)),
                tell,
                lambda _: (state.es_state, jnp.asarray(False)),
                None,
            )
            with jax.enable_x64(False):
                params = self.es.get_mean(es_state)
            next_step = state.global_step + self.generation_steps
            return state.replace(
                params=params,
                es_state=es_state,
                global_step=next_step,
                alive_steps=state.alive_steps + lengths.sum(dtype=jnp.int32),
                update_index=state.update_index + 1,
                failed=~safe,
                failure_step=jnp.where(safe, -1, next_step),
            )

        return jax.lax.cond(state.failed, lambda _: state, active, None)

    def train(self, rng: jax.Array) -> tuple[ESTrainState, dict[str, Any]]:
        rng = to_typed_key(rng)
        state = self.init_state(rng)
        generations = self.total_timesteps // self.generation_steps
        boundaries = sorted(
            {
                min(
                    (step + self.generation_steps - 1) // self.generation_steps,
                    generations,
                )
                for step in range(self.eval_freq, self.total_timesteps, self.eval_freq)
            }
            | {generations}
        )
        eval_key = jax.random.key(self.eval_seed)
        eval_keys = jax.random.split(eval_key, self.eval_n_envs)
        train_key = jax.random.fold_in(rng, 0xE5)

        def diagnostics(current: ESTrainState) -> dict[str, jax.Array]:
            return {
                "train/generation": current.update_index.astype(jnp.float32),
                "train/search_scale": jnp.asarray(current.es_state.std, jnp.float32),
            }

        def evaluate(current: ESTrainState) -> Any:
            if self.eval_callback is not None:
                return self.eval_callback(self, current, eval_key, diagnostics(current))
            return self._rollouts(current.params, eval_keys)

        initial = evaluate(state)
        initial_diagnostics = diagnostics(state)

        def step(_: jax.Array, current: ESTrainState) -> ESTrainState:
            return self.update(
                current, jax.random.fold_in(train_key, current.update_index)
            )

        def interval(current: ESTrainState, end: jax.Array) -> tuple:
            current = jax.lax.fori_loop(current.update_index, end, step, current)
            return current, (
                current.global_step,
                evaluate(current),
                diagnostics(current),
            )

        state, (steps, evaluation, metrics) = jax.lax.scan(
            interval, state, jnp.asarray(boundaries, jnp.int32)
        )

        def prepend(first: jax.Array, rest: jax.Array) -> jax.Array:
            return jnp.concatenate((jnp.asarray(first)[None], rest), axis=0)

        return state, {
            "global_step": prepend(jnp.asarray(0, jnp.int32), steps),
            "evaluation": jax.tree.map(prepend, initial, evaluation),
            "diagnostics": jax.tree.map(prepend, initial_diagnostics, metrics),
        }


__all__ = ["ESAgent", "ESTrainState"]
