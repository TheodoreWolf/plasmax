"""Online learned-dynamics MPC with the clone-only agent interface."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import optax
from envelope import AutoResetWrapper, Continuous, PooledInitVmapWrapper, Wrapper
from flax import linen as nn
from flax import struct
from flax.training.train_state import TrainState
from rejax.buffers import Minibatch, ReplayBuffer

from plasmax.rollout import collect_episodes

RewardFn = Callable[[jax.Array, jax.Array, jax.Array], jax.Array]


class WorldModel(nn.Module):
    """MLP predicting ``next_obs = obs + delta(obs, action)``."""

    obs_dim: int
    hidden: int = 128

    @nn.compact
    def __call__(self, obs: jax.Array, action: jax.Array) -> jax.Array:
        x = jnp.concatenate([obs, action], axis=-1)
        x = nn.relu(nn.Dense(self.hidden)(x))
        x = nn.relu(nn.Dense(self.hidden)(x))
        return obs + nn.Dense(self.obs_dim)(x)


def plan_action(
    model: WorldModel,
    params: Any,
    action_low: jax.Array,
    action_high: jax.Array,
    reward_fn: RewardFn,
    obs: jax.Array,
    rng: jax.Array,
    *,
    horizon: int,
    num_samples: int,
) -> jax.Array:
    """Plan against a frozen model; also used by loaded policy artifacts."""
    sequences = jax.random.uniform(
        rng,
        (num_samples, horizon, action_low.size),
        dtype=jnp.float32,
        minval=action_low,
        maxval=action_high,
    )

    def rollout_return(actions: jax.Array) -> jax.Array:
        def step(o: jax.Array, a: jax.Array) -> tuple[jax.Array, jax.Array]:
            next_o = model.apply(params, o, a)
            return next_o, reward_fn(o, a, next_o)

        _, rewards = jax.lax.scan(step, jnp.asarray(obs, jnp.float32), actions)
        return rewards.sum()

    returns = jax.vmap(rollout_return)(sequences)
    return sequences[jnp.argmax(returns), 0]


def _all_finite(tree: Any) -> jax.Array:
    return jnp.all(jnp.stack([jnp.all(jnp.isfinite(x)) for x in jax.tree.leaves(tree)]))


class MPCState(struct.PyTreeNode):
    """Training state; only ``model_ts.params`` is needed for inference."""

    buffer: ReplayBuffer
    model_ts: TrainState
    env_state: Any
    obs: jax.Array
    rng: jax.Array
    boundary: jax.Array
    global_step: jax.Array
    failed: jax.Array
    first_failure_step: jax.Array


class MPCAgent(struct.PyTreeNode):
    """Random-shooting MPC trained online over native Envelope transitions."""

    action_low: jax.Array
    action_high: jax.Array
    env: Any = struct.field(pytree_node=False)
    model: WorldModel = struct.field(pytree_node=False)
    reward_fn: RewardFn = struct.field(pytree_node=False)
    reward_scalar: str | None = struct.field(pytree_node=False)
    reward_slice: tuple[int, int] | None = struct.field(pytree_node=False)
    horizon: int = struct.field(pytree_node=False, default=15)
    num_samples: int = struct.field(pytree_node=False, default=256)
    buffer_size: int = struct.field(pytree_node=False, default=20_000)
    lr: float = struct.field(pytree_node=False, default=1e-3)
    train_batch_size: int = struct.field(pytree_node=False, default=256)
    total_timesteps: int = struct.field(pytree_node=False, default=6_000)
    eval_freq: int = struct.field(pytree_node=False, default=100)
    eval_num_episodes: int = struct.field(pytree_node=False, default=16)
    eval_max_steps: int = struct.field(pytree_node=False, default=100)
    eval_callback: Callable | None = struct.field(pytree_node=False, default=None)
    deterministic: bool = struct.field(pytree_node=False, default=False)
    planning_seed: int = struct.field(pytree_node=False, default=0)

    @classmethod
    def create(
        cls,
        env: Any,
        reward_fn: RewardFn | None = None,
        *,
        reward_scalar: str = "P_fusion",
        horizon: int = 15,
        num_samples: int = 256,
        buffer_size: int = 20_000,
        hidden: int = 128,
        lr: float = 1e-3,
        train_batch_size: int = 256,
        total_timesteps: int = 6_000,
        eval_freq: int = 100,
        eval_num_episodes: int = 16,
        eval_max_steps: int | None = None,
        eval_callback: Callable | None = None,
        deterministic: bool = False,
        planning_seed: int = 0,
    ) -> MPCAgent:
        action_space = env.action_space
        if not isinstance(action_space, Continuous):
            raise TypeError(
                "MPCAgent requires an Envelope Continuous action space; "
                "discrete or quantized actions are not supported"
            )
        if len(action_space.shape) != 1 or len(env.observation_space.shape) != 1:
            raise ValueError("MPCAgent requires flat action and observation spaces")
        layer = env
        while isinstance(layer, Wrapper):
            if isinstance(layer, (AutoResetWrapper, PooledInitVmapWrapper)):
                raise ValueError("MPCAgent requires a non-autoresetting environment")
            layer = layer.env
        sizes = (
            horizon,
            num_samples,
            buffer_size,
            hidden,
            train_batch_size,
            total_timesteps,
            eval_freq,
            eval_num_episodes,
        )
        if any(value <= 0 for value in sizes):
            raise ValueError("MPC training and planner sizes must be positive")
        if train_batch_size > buffer_size:
            raise ValueError("train_batch_size must not exceed buffer_size")
        if eval_max_steps is None:
            eval_max_steps = getattr(env, "max_steps", total_timesteps)
        if eval_max_steps <= 0:
            raise ValueError("eval_max_steps must be positive")
        objective_slice = None
        if reward_fn is None:
            obs_slice = env.obs_layout().slice_of(reward_scalar)
            objective_slice = (obs_slice.start, obs_slice.stop)

            def reward_fn(obs: jax.Array, action: jax.Array, next_obs: jax.Array):
                del obs, action
                return next_obs[obs_slice].sum()

        else:
            reward_scalar = None
        return cls(
            action_low=jnp.asarray(action_space.low, jnp.float32),
            action_high=jnp.asarray(action_space.high, jnp.float32),
            env=env,
            model=WorldModel(obs_dim=env.observation_space.shape[0], hidden=hidden),
            reward_fn=reward_fn,
            reward_scalar=reward_scalar,
            reward_slice=objective_slice,
            horizon=horizon,
            num_samples=num_samples,
            buffer_size=buffer_size,
            lr=lr,
            train_batch_size=train_batch_size,
            total_timesteps=total_timesteps,
            eval_freq=eval_freq,
            eval_num_episodes=eval_num_episodes,
            eval_max_steps=eval_max_steps,
            eval_callback=eval_callback,
            deterministic=deterministic,
            planning_seed=planning_seed,
        )

    def init_state(self, rng: jax.Array) -> MPCState:
        rng, model_rng, env_rng = jax.random.split(rng, 3)
        params = self.model.init(
            model_rng,
            jnp.zeros(self.env.observation_space.shape, jnp.float32),
            jnp.zeros(self.env.action_space.shape, jnp.float32),
        )
        env_state, info = self.env.init(env_rng)
        failed = info.terminated | info.truncated | ~_all_finite(info.obs)
        return MPCState(
            buffer=ReplayBuffer.empty(
                self.buffer_size, self.env.observation_space, self.env.action_space
            ),
            model_ts=TrainState.create(
                apply_fn=self.model.apply, params=params, tx=optax.adam(self.lr)
            ),
            env_state=env_state,
            obs=jnp.asarray(info.obs, jnp.float32),
            rng=rng,
            boundary=info.terminated | info.truncated,
            global_step=jnp.int32(0),
            failed=failed,
            first_failure_step=jnp.where(failed, jnp.int32(0), jnp.int32(-1)),
        )

    def observe(
        self,
        state: MPCState,
        obs: jax.Array,
        action: jax.Array,
        reward: jax.Array,
        terminated: jax.Array,
        truncated: jax.Array,
        next_obs: jax.Array,
    ) -> MPCState:
        # A truncation still has a learnable next-state target.
        del truncated
        transition = jax.tree.map(
            lambda value, data: jnp.asarray(value, data.dtype),
            Minibatch(obs, action, reward, terminated, next_obs),
            state.buffer.data,
        )
        return state.replace(buffer=state.buffer.append(transition))

    def train_step(
        self, state: MPCState, rng: jax.Array, batch_size: int | None = None
    ) -> tuple[MPCState, jax.Array]:
        batch = state.buffer.sample(batch_size or self.train_batch_size, rng)

        def loss_fn(params: Any) -> jax.Array:
            pred = self.model.apply(params, batch.obs, batch.action)
            sq_err = jnp.mean((pred - batch.next_obs) ** 2, axis=-1)
            weight = 1.0 - batch.done.astype(sq_err.dtype)
            return jnp.sum(sq_err * weight) / jnp.maximum(jnp.sum(weight), 1.0)

        loss, grads = jax.value_and_grad(loss_fn)(state.model_ts.params)
        candidate = state.model_ts.apply_gradients(grads=grads)
        failed = state.failed | ~_all_finite(
            (loss, grads, candidate.params, candidate.opt_state)
        )
        model_ts = jax.lax.cond(failed, lambda: state.model_ts, lambda: candidate)
        return state.replace(
            model_ts=model_ts,
            failed=failed,
            first_failure_step=jnp.where(
                failed & ~state.failed, state.global_step, state.first_failure_step
            ),
        ), loss

    def make_act(
        self, state: MPCState, deterministic: bool | None = None
    ) -> Callable[[jax.Array, jax.Array], jax.Array]:
        """Bind a frozen model; planning never updates training state."""
        deterministic = self.deterministic if deterministic is None else deterministic
        params = state.model_ts.params

        def act(obs: jax.Array, rng: jax.Array) -> jax.Array:
            if deterministic:
                rng = jax.random.key(self.planning_seed)
            return plan_action(
                self.model,
                params,
                self.action_low,
                self.action_high,
                self.reward_fn,
                obs,
                rng,
                horizon=self.horizon,
                num_samples=self.num_samples,
            )

        return act

    def _step(self, state: MPCState) -> tuple[MPCState, jax.Array]:
        rng, reset_rng, act_rng, train_rng = jax.random.split(state.rng, 4)

        def reset(ms: MPCState) -> MPCState:
            env_state, info = self.env.reset(ms.env_state, reset_rng)
            failed = info.terminated | info.truncated | ~_all_finite(info.obs)
            return ms.replace(
                env_state=env_state,
                obs=jnp.asarray(info.obs, jnp.float32),
                boundary=info.terminated | info.truncated,
                failed=failed,
                first_failure_step=jnp.where(
                    failed, ms.global_step, ms.first_failure_step
                ),
            )

        state = jax.lax.cond(state.boundary, reset, lambda ms: ms, state)

        def transition(ms: MPCState) -> tuple[MPCState, jax.Array]:
            action = self.make_act(ms, deterministic=False)(ms.obs, act_rng)
            env_state, info = self.env.step(
                ms.env_state, action.astype(self.env.action_space.dtype)
            )
            failed = ~_all_finite((action, info.obs, info.reward))
            next_state = ms.replace(
                env_state=env_state,
                obs=jnp.asarray(info.obs, jnp.float32),
                rng=rng,
                boundary=info.terminated | info.truncated,
                global_step=ms.global_step + 1,
                failed=failed,
                first_failure_step=jnp.where(
                    failed, ms.global_step + 1, ms.first_failure_step
                ),
            )

            def learn(ms: MPCState) -> tuple[MPCState, jax.Array]:
                ms = self.observe(
                    ms,
                    state.obs,
                    action,
                    info.reward,
                    info.terminated,
                    info.truncated,
                    info.obs,
                )
                return jax.lax.cond(
                    ms.buffer.num_entries >= self.train_batch_size,
                    lambda s: self.train_step(s, train_rng),
                    lambda s: (s, jnp.float32(jnp.nan)),
                    ms,
                )

            return jax.lax.cond(
                failed, lambda s: (s, jnp.float32(jnp.nan)), learn, next_state
            )

        return jax.lax.cond(
            state.failed, lambda s: (s, jnp.float32(jnp.nan)), transition, state
        )

    def train(self, rng: jax.Array) -> tuple[MPCState, dict[str, Any]]:
        """Run the exact transition budget, evaluating without resetting training."""
        state = self.init_state(rng)
        eval_rng = jax.random.fold_in(rng, 0xE7A1)
        boundaries = jnp.asarray(
            [
                0,
                *range(self.eval_freq, self.total_timesteps, self.eval_freq),
                self.total_timesteps,
            ],
            jnp.int32,
        )

        def evaluate(ms: MPCState, model_loss: jax.Array) -> Any:
            if self.eval_callback is not None:
                return self.eval_callback(
                    self,
                    ms,
                    eval_rng,
                    {"training/model_loss": model_loss, "training/failed": ms.failed},
                )
            trajectories = collect_episodes(
                self.make_act(ms),
                self.env,
                eval_rng,
                self.eval_max_steps,
                self.eval_num_episodes,
            )
            return (
                trajectories.valid.sum(axis=1),
                jnp.where(trajectories.valid, trajectories.reward, 0.0).sum(axis=1),
            )

        def interval(ms: MPCState, end: jax.Array) -> tuple[MPCState, dict[str, Any]]:
            def update(_: int, carry: tuple[MPCState, jax.Array, jax.Array]):
                current, loss_sum, count = carry
                current, loss = jax.lax.cond(
                    current.failed,
                    lambda s: (s, jnp.float32(jnp.nan)),
                    self._step,
                    current,
                )
                finite = jnp.isfinite(loss)
                return current, loss_sum + jnp.where(finite, loss, 0.0), count + finite

            ms, loss_sum, count = jax.lax.fori_loop(
                ms.global_step, end, update, (ms, jnp.float32(0.0), jnp.int32(0))
            )
            model_loss = jnp.where(count > 0, loss_sum / jnp.maximum(count, 1), jnp.nan)
            return ms, {
                "global_step": ms.global_step,
                "model_loss": model_loss,
                "evaluation": evaluate(ms, model_loss),
            }

        return jax.lax.scan(interval, state, boundaries)


__all__ = ["MPCAgent", "MPCState", "WorldModel", "plan_action"]
