"""Clone-only pathwise agents with Rejax-style training and inference methods.

The optimizer differentiates one native Envelope rollout at a time, then vmaps
those gradients. Whole training can be jitted and vmapped across optimizer seeds.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from envelope import Continuous
from flax import linen as nn
from flax import struct

from agents.direct_gradient import (
    apply_policy_optimizer_update,
    apply_updates_with_backoff,
    finite_mean_gradients,
    gradient_horizon,
    knot_actions,
    make_knot_chunk,
    make_optimizer,
    setpoint_theta_row,
    tree_is_finite,
)
from training.envelope_gymnax import to_typed_key


class ResidualPolicy(nn.Module):
    """The baseline MLP's residual around the normalized reset setpoint."""

    action_dim: int
    hidden_sizes: tuple[int, ...]

    @nn.compact
    def __call__(self, obs: jax.Array) -> jax.Array:
        x = obs
        for width in self.hidden_sizes:
            x = nn.swish(nn.Dense(width)(x))
        return nn.Dense(
            self.action_dim,
            kernel_init=nn.initializers.zeros_init(),
            bias_init=nn.initializers.zeros_init(),
            name="action_residual",
        )(x)


def policy_action(
    policy: ResidualPolicy, params: Any, action_setpoint: jax.Array, obs: jax.Array
) -> jax.Array:
    return jnp.clip(action_setpoint + policy.apply({"params": params}, obs), -1.0, 1.0)


def open_loop_action(
    theta: jax.Array, obs: jax.Array, source_times: jax.Array, time_index: int
) -> jax.Array:
    """Map elapsed source time to a knot index; never retime on transfer."""
    source_index = jnp.interp(
        obs[time_index], source_times, jnp.arange(source_times.size, dtype=theta.dtype)
    )
    knots = jnp.linspace(0.0, source_times.size - 1, theta.shape[0], dtype=theta.dtype)
    return jnp.tanh(
        jax.vmap(lambda values: jnp.interp(source_index, knots, values), in_axes=1)(
            theta
        )
    )


class PolicyCarry(NamedTuple):
    obs: jax.Array
    env_state: Any
    alive: jax.Array


class PolicyStep(NamedTuple):
    reward: jax.Array
    action: jax.Array
    done: jax.Array
    alive: jax.Array


@dataclasses.dataclass(frozen=True)
class PolicyChunk:
    initialize: Callable
    run: Callable


def make_policy_chunk(
    env: Any,
    policy: ResidualPolicy,
    action_setpoint: jax.Array,
    chunk_steps: int,
    *,
    remat: bool,
) -> PolicyChunk:
    """Carry feedback gradients within each chunk and freeze terminal episodes."""

    def step(
        params: Any, carry: PolicyCarry, _: None
    ) -> tuple[PolicyCarry, PolicyStep]:
        obs, env_state, alive = carry

        def active(_: None) -> tuple[PolicyCarry, PolicyStep]:
            action = policy_action(policy, params, action_setpoint, obs)
            state, info = env.step(env_state, action)
            done = info.terminated | info.truncated
            return PolicyCarry(info.obs, state, ~done), PolicyStep(
                info.reward, action, done, jnp.asarray(True)
            )

        def inactive(_: None) -> tuple[PolicyCarry, PolicyStep]:
            return carry, PolicyStep(
                jnp.zeros((), jnp.float32),
                jnp.zeros_like(action_setpoint),
                jnp.asarray(False),
                jnp.asarray(False),
            )

        return jax.lax.cond(alive, active, inactive, None)

    scan_step = jax.checkpoint(step) if remat else step

    def initialize(key: jax.Array) -> PolicyCarry:
        state, info = env.init(key)
        return PolicyCarry(info.obs, state, jnp.asarray(True))

    def run(params: Any, carry: PolicyCarry) -> tuple[jax.Array, Any]:
        carry, trajectory = jax.lax.scan(
            lambda state, _: scan_step(params, state, None),
            carry,
            None,
            length=chunk_steps,
        )
        return jnp.sum(trajectory.reward), (carry, trajectory)

    return PolicyChunk(initialize, run)


@struct.dataclass
class PolicyTrainState:
    params: Any
    opt_state: Any
    global_step: jax.Array
    alive_steps: jax.Array
    update_index: jax.Array
    failed: jax.Array
    failure_step: jax.Array


@struct.dataclass
class OpenLoopTrainState:
    params: jax.Array
    opt_state: Any
    rollback_params: jax.Array
    rollback_opt_state: Any
    update_scale: jax.Array
    global_step: jax.Array
    alive_steps: jax.Array
    update_index: jax.Array
    failed: jax.Array
    failure_step: jax.Array


def _setup(agent: Any) -> None:
    if not isinstance(agent.env.action_space, Continuous):
        raise ValueError("backprop requires continuous actions")
    if agent.eval_freq <= 0 or agent.eval_n_envs <= 0:
        raise ValueError("eval_freq and eval_n_envs must be positive")
    episode_steps = agent.episode_steps
    if episode_steps is None:
        episode_steps = agent.env.max_steps
    object.__setattr__(agent, "episode_steps", int(episode_steps))
    object.__setattr__(
        agent,
        "gradient_horizon",
        gradient_horizon(
            int(episode_steps),
            agent.total_timesteps,
            agent.num_rollouts,
            agent.gradient_horizon,
        ),
    )
    object.__setattr__(
        agent, "optimizer", make_optimizer(agent.learning_rate, agent.grad_clip)
    )
    _ = agent.env.observation_space


def _state_counters() -> dict[str, jax.Array]:
    zero = jnp.asarray(0, jnp.int32)
    return dict(
        global_step=zero,
        alive_steps=zero,
        update_index=zero,
        failed=jnp.asarray(False),
        failure_step=jnp.asarray(-1, jnp.int32),
    )


def _train(agent: Any, rng: jax.Array) -> tuple[Any, dict[str, Any]]:
    """Shared chunk scheduling only; each agent owns its objective and update."""
    rng = to_typed_key(rng)
    state = agent.init_state(rng)
    step_size = agent.gradient_horizon * agent.num_rollouts
    num_updates = agent.total_timesteps // step_size
    chunks_per_pass = agent.episode_steps // agent.gradient_horizon

    def initialize(update_index: jax.Array) -> Any:
        pass_key = jax.random.fold_in(rng, update_index // chunks_per_pass)
        keys = jax.vmap(lambda index: jax.random.fold_in(pass_key, index))(
            jnp.arange(agent.num_rollouts)
        )
        return jax.vmap(agent.chunk.initialize)(keys)

    diagnostics = agent.empty_diagnostics()
    env_carries = initialize(jnp.asarray(0, jnp.int32))

    def update(_: jax.Array, carry: tuple) -> tuple:
        current, carries, accumulated = carry

        def active(_: None) -> tuple:
            fresh = jax.lax.cond(
                current.update_index % chunks_per_pass == 0,
                lambda _: initialize(current.update_index),
                lambda _: carries,
                None,
            )
            start_step = (
                current.update_index % chunks_per_pass
            ) * agent.gradient_horizon
            candidate, next_carries, metrics, safe = agent.update(
                current, fresh, start_step
            )
            next_step = current.global_step + step_size
            accepted = jax.lax.cond(safe, lambda _: candidate, lambda _: current, None)
            accepted = accepted.replace(
                global_step=next_step,
                alive_steps=current.alive_steps
                + metrics["train/alive_steps"].astype(jnp.int32),
                update_index=current.update_index + 1,
                failed=~safe,
                failure_step=jnp.where(safe, -1, next_step),
            )
            return (
                accepted,
                jax.tree.map(jax.lax.stop_gradient, next_carries),
                jax.tree.map(jnp.add, accumulated, metrics),
            )

        return jax.lax.cond(current.failed, lambda _: carry, active, None)

    eval_key = jax.random.key(agent.eval_seed)

    def evaluate(current: Any, metrics: dict[str, jax.Array]) -> Any:
        if agent.eval_callback is not None:
            return agent.eval_callback(agent, current, eval_key, metrics)
        from plasmax import collect_episode

        keys = jax.vmap(lambda index: jax.random.fold_in(eval_key, index))(
            jnp.arange(agent.eval_n_envs)
        )

        def one(key: jax.Array) -> tuple[jax.Array, jax.Array]:
            trajectory = collect_episode(
                agent.make_act(current), agent.env, key, num_steps=agent.episode_steps
            )
            return jnp.sum(trajectory.reward), jnp.sum(trajectory.valid)

        return jax.vmap(one)(keys)

    initial = evaluate(state, diagnostics)
    boundaries = sorted(
        {
            min((step + step_size - 1) // step_size, num_updates)
            for step in range(agent.eval_freq, agent.total_timesteps, agent.eval_freq)
        }
        | {num_updates}
    )

    def interval(carry: tuple, end: jax.Array) -> tuple:
        current, carries = carry
        previous_index = current.update_index
        current, carries, metrics = jax.lax.fori_loop(
            previous_index, end, update, (current, carries, agent.empty_diagnostics())
        )
        count = jnp.maximum(current.update_index - previous_index, 1)
        # Counts remain sums; the other update diagnostics are interval means.
        counts = {
            "train/alive_steps",
            "train/nonfinite_grad_elements",
            "train/total_grad_elements",
            "train/rollouts_with_nonfinite_grads",
            "train/total_rollouts",
            "train/all_missing_grad_elements",
        }
        metrics = {
            name: value if name in counts else value / count
            for name, value in metrics.items()
        }
        evaluation = evaluate(current, metrics)
        return (current, carries), (current.global_step, evaluation, metrics)

    (state, _), (steps, evaluation, metrics) = jax.lax.scan(
        interval, (state, env_carries), jnp.asarray(boundaries, jnp.int32)
    )

    def prepend(first: jax.Array, rest: jax.Array) -> jax.Array:
        return jnp.concatenate((jnp.asarray(first)[None], rest), axis=0)

    return state, {
        "global_step": prepend(jnp.asarray(0, jnp.int32), steps),
        "evaluation": jax.tree.map(prepend, initial, evaluation),
        "diagnostics": jax.tree.map(prepend, diagnostics, metrics),
    }


@dataclasses.dataclass(frozen=True, eq=False)
class BackpropPolicyAgent:
    env: Any
    total_timesteps: int = 10_000_000
    eval_freq: int = 1_000_000
    num_rollouts: int = 64
    gradient_horizon: int = 32
    learning_rate: float = 1e-4
    grad_clip: float = 1.0
    hidden_sizes: tuple[int, ...] = (64, 64)
    remat: bool = True
    eval_n_envs: int = 16
    eval_seed: int = 10_000
    episode_steps: int | None = None
    action_setpoint: jax.Array | None = None
    init_seed: int = 0
    eval_callback: Callable | None = None
    policy: ResidualPolicy = dataclasses.field(init=False, repr=False)
    optimizer: Any = dataclasses.field(init=False, repr=False)
    chunk: PolicyChunk = dataclasses.field(init=False, repr=False)

    def __post_init__(self) -> None:
        _setup(self)
        if self.action_setpoint is None:
            key = jax.random.fold_in(jax.random.key(self.init_seed), 0x5E7)
            object.__setattr__(
                self, "action_setpoint", jnp.tanh(setpoint_theta_row(self.env, key))
            )
        object.__setattr__(
            self,
            "action_setpoint",
            jnp.asarray(self.action_setpoint),
        )
        object.__setattr__(
            self,
            "policy",
            ResidualPolicy(self.env.action_space.shape[0], self.hidden_sizes),
        )
        object.__setattr__(
            self,
            "chunk",
            make_policy_chunk(
                self.env,
                self.policy,
                self.action_setpoint,
                self.gradient_horizon,
                remat=self.remat,
            ),
        )

    @classmethod
    def create(cls, env: Any, **kwargs: Any) -> BackpropPolicyAgent:
        return cls(env, **kwargs)

    def init_state(self, rng: jax.Array) -> PolicyTrainState:
        params = self.policy.init(
            jax.random.fold_in(rng, 0x1A17),
            jnp.zeros(self.env.observation_space.shape, jnp.float32),
        )["params"]
        return PolicyTrainState(
            params, self.optimizer.init(params), **_state_counters()
        )

    def empty_diagnostics(self) -> dict[str, jax.Array]:
        names = (
            "grad_norm",
            "grad_clip_scale",
            "optimizer_update_finite",
            "params_finite",
            "aggregate_grads_finite",
            "nonfinite_grad_elements",
            "total_grad_elements",
            "rollouts_with_nonfinite_grads",
            "total_rollouts",
            "all_missing_grad_elements",
            "alive_steps",
        )
        return {f"train/{name}": jnp.asarray(0.0, jnp.float32) for name in names}

    def update(
        self, state: PolicyTrainState, carries: Any, start_step: jax.Array
    ) -> tuple:
        del start_step

        def loss(params: Any, carry: Any) -> tuple:
            reward, (next_carry, trajectory) = self.chunk.run(params, carry)
            return -reward, (next_carry, jnp.sum(trajectory.alive))

        (losses, (carries, alive)), per_rollout_grads = jax.vmap(
            jax.value_and_grad(loss, has_aux=True), in_axes=(None, 0)
        )(state.params, carries)
        grads, stats = finite_mean_gradients(per_rollout_grads)
        params, opt_state, diagnostics = apply_policy_optimizer_update(
            self.optimizer, grads, state.opt_state, state.params, self.grad_clip
        )
        metrics = dict(
            zip(
                (
                    "train/grad_norm",
                    "train/grad_clip_scale",
                    "train/optimizer_update_finite",
                    "train/params_finite",
                ),
                diagnostics,
                strict=True,
            )
        )
        metrics.update(
            {
                "train/aggregate_grads_finite": stats.aggregate_finite,
                "train/nonfinite_grad_elements": stats.nonfinite_elements,
                "train/total_grad_elements": stats.total_elements,
                "train/rollouts_with_nonfinite_grads": stats.rollouts_with_nonfinite,
                "train/total_rollouts": stats.total_rollouts,
                "train/all_missing_grad_elements": stats.all_missing_elements,
                "train/alive_steps": jnp.sum(alive),
            }
        )
        safe = (
            stats.aggregate_finite
            & diagnostics.optimizer_update_finite
            & diagnostics.params_finite
            & tree_is_finite(opt_state)
            & jnp.all(jnp.isfinite(losses))
        )
        return (
            state.replace(params=params, opt_state=opt_state),
            carries,
            jax.tree.map(lambda value: value.astype(jnp.float32), metrics),
            safe,
        )

    def train(self, rng: jax.Array) -> tuple[PolicyTrainState, dict[str, Any]]:
        return _train(self, rng)

    def make_act(
        self, state: PolicyTrainState, deterministic: bool | None = None
    ) -> Callable:
        del deterministic
        return lambda obs, rng: policy_action(
            self.policy, state.params, self.action_setpoint, obs
        )


def source_time_grid(env: Any, num_steps: int) -> jax.Array:
    """Elapsed action-slot times from the source's fixed control schedule."""
    config = env.unwrapped.config
    numerics = config.numerics
    initial = float(numerics.t_initial)
    current = initial
    times = []
    for _ in range(num_steps):
        times.append(current - initial)
        next_time = current + float(numerics.fixed_dt.get_value(current))
        current = (
            min(next_time, float(numerics.t_final))
            if numerics.exact_t_final
            else next_time
        )
    return jnp.asarray(times)


@dataclasses.dataclass(frozen=True, eq=False)
class BackpropOpenLoopAgent:
    env: Any
    total_timesteps: int = 10_000_000
    eval_freq: int = 1_000_000
    num_rollouts: int = 64
    gradient_horizon: int = 32
    learning_rate: float = 5e-2
    grad_clip: float = 1.0
    num_knots: int = 10
    nonfinite_backoff_factor: float = 0.5
    min_update_scale: float = 1e-3
    remat: bool = True
    eval_n_envs: int = 16
    eval_seed: int = 10_000
    episode_steps: int | None = None
    action_setpoint: jax.Array | None = None
    source_times: jax.Array | None = None
    init_seed: int = 0
    eval_callback: Callable | None = None
    time_index: int = dataclasses.field(init=False)
    theta0: jax.Array = dataclasses.field(init=False, repr=False)
    optimizer: Any = dataclasses.field(init=False, repr=False)
    chunk: Any = dataclasses.field(init=False, repr=False)

    def __post_init__(self) -> None:
        _setup(self)
        if self.num_knots <= 0:
            raise ValueError("num_knots must be positive")
        if (
            not 0 < self.nonfinite_backoff_factor < 1
            or not 0 < self.min_update_scale <= 1
        ):
            raise ValueError(
                "backoff_factor must be in (0, 1) and min_update_scale in (0, 1]"
            )
        try:
            clock_slice = self.env.obs_layout().slice_of("elapsed_time")
        except (KeyError, ValueError, AttributeError) as error:
            raise ValueError(
                "open-loop inference requires elapsed_time in observations; "
                "add TimeAwareWrapper"
            ) from error
        if clock_slice.stop - clock_slice.start != 1:
            raise ValueError("elapsed_time must be a scalar observation")
        object.__setattr__(self, "time_index", clock_slice.start)
        times = (
            source_time_grid(self.env, self.episode_steps)
            if self.source_times is None
            else self.source_times
        )
        times = np.asarray(times)
        if (
            times.shape != (self.episode_steps,)
            or not np.all(np.isfinite(times))
            or np.any(np.diff(times) <= 0)
        ):
            raise ValueError(
                "source_times must contain one increasing finite time per source step"
            )
        object.__setattr__(self, "source_times", jnp.asarray(times))
        key = jax.random.fold_in(jax.random.key(self.init_seed), 0x5E7)
        row = (
            setpoint_theta_row(self.env, key)
            if self.action_setpoint is None
            else jnp.arctanh(
                jnp.clip(jnp.asarray(self.action_setpoint, jnp.float32), -0.999, 0.999)
            )
        )
        object.__setattr__(
            self,
            "theta0",
            jnp.broadcast_to(
                row, (min(self.num_knots, self.episode_steps), row.shape[0])
            ),
        )
        object.__setattr__(
            self,
            "chunk",
            make_knot_chunk(
                self.env,
                lambda theta: knot_actions(theta, self.episode_steps),
                self.gradient_horizon,
                remat=self.remat,
            ),
        )

    @classmethod
    def create(cls, env: Any, **kwargs: Any) -> BackpropOpenLoopAgent:
        return cls(env, **kwargs)

    def init_state(self, rng: jax.Array) -> OpenLoopTrainState:
        del rng
        opt_state = self.optimizer.init(self.theta0)
        return OpenLoopTrainState(
            self.theta0,
            opt_state,
            self.theta0,
            opt_state,
            jnp.asarray(1.0, jnp.float32),
            **_state_counters(),
        )

    def empty_diagnostics(self) -> dict[str, jax.Array]:
        return {
            f"train/{name}": jnp.asarray(0.0, jnp.float32)
            for name in ("grad_norm", "grads_finite", "update_scale", "alive_steps")
        }

    def update(
        self, state: OpenLoopTrainState, carries: Any, start_step: jax.Array
    ) -> tuple:
        def loss(params: Any, carry: Any, start: jax.Array) -> tuple:
            reward, (next_carry, trajectory) = self.chunk.run(params, carry, start)
            return -reward, (next_carry, jnp.sum(trajectory.alive))

        (losses, (carries, alive)), per_rollout_grads = jax.vmap(
            jax.value_and_grad(loss, has_aux=True), in_axes=(None, 0, None)
        )(state.params, carries, start_step)
        grads = jax.tree.map(lambda value: jnp.mean(value, axis=0), per_rollout_grads)
        params, opt_state, rollback_params, rollback_opt_state, scale, finite = (
            apply_updates_with_backoff(
                self.optimizer,
                grads,
                state.opt_state,
                state.params,
                state.rollback_opt_state,
                state.rollback_params,
                state.update_scale,
                backoff_factor=self.nonfinite_backoff_factor,
                min_update_scale=self.min_update_scale,
            )
        )
        candidate = state.replace(
            params=params,
            opt_state=opt_state,
            rollback_params=rollback_params,
            rollback_opt_state=rollback_opt_state,
            update_scale=scale,
        )
        metrics = {
            "train/grad_norm": optax.global_norm(grads),
            "train/grads_finite": finite,
            "train/update_scale": scale,
            "train/alive_steps": jnp.sum(alive),
        }
        return (
            candidate,
            carries,
            jax.tree.map(lambda value: value.astype(jnp.float32), metrics),
            tree_is_finite((params, opt_state)) & jnp.all(jnp.isfinite(losses)),
        )

    def train(self, rng: jax.Array) -> tuple[OpenLoopTrainState, dict[str, Any]]:
        return _train(self, rng)

    def make_act(
        self, state: OpenLoopTrainState, deterministic: bool | None = None
    ) -> Callable:
        del deterministic
        return lambda obs, rng: open_loop_action(
            state.params, obs, self.source_times, self.time_index
        )
