"""Learned-dynamics MPC agent.

The agent learns a small obs-space world model online from a replay buffer, then
uses random shooting to choose the first action of the best sampled sequence.
"""

from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax
from envelope import AutoResetWrapper, Continuous, PooledInitVmapWrapper, Wrapper
from flax import linen as nn
from flax import struct
from flax.training.train_state import TrainState
from rejax.buffers import Minibatch, ReplayBuffer


class WorldModel(nn.Module):
    """MLP predicting ``next_obs = obs + delta(obs, action)``."""

    obs_dim: int
    hidden: int = 128

    @nn.compact
    def __call__(self, obs: jax.Array, action: jax.Array) -> jax.Array:
        x = jnp.concatenate([obs, action], axis=-1)
        x = nn.relu(nn.Dense(self.hidden)(x))
        x = nn.relu(nn.Dense(self.hidden)(x))
        delta = nn.Dense(self.obs_dim)(x)
        return obs + delta


class MPCState(struct.PyTreeNode):
    """Mutable state threaded through :class:`MPCAgent` methods."""

    buffer: ReplayBuffer
    model_ts: TrainState


class MPCAgent(struct.PyTreeNode):
    """Random-shooting MPC over an online learned dynamics model."""

    action_low: jax.Array
    action_high: jax.Array
    model: nn.Module = struct.field(pytree_node=False)
    reward_fn: Callable[[jax.Array, jax.Array, jax.Array], jax.Array] = struct.field(
        pytree_node=False
    )
    horizon: int = struct.field(pytree_node=False, default=15)
    num_samples: int = struct.field(pytree_node=False, default=256)

    @classmethod
    def create(
        cls,
        env,
        reward_fn: Callable[[jax.Array, jax.Array, jax.Array], jax.Array],
        *,
        horizon: int = 15,
        num_samples: int = 256,
        buffer_size: int = 20_000,
        hidden: int = 128,
        lr: float = 1e-3,
        seed: int = 0,
    ) -> tuple["MPCAgent", MPCState]:
        obs_space = env.observation_space
        action_space = env.action_space
        if not isinstance(action_space, Continuous):
            raise TypeError(
                "MPCAgent requires an Envelope Continuous action space; "
                "discrete or quantized actions are not supported"
            )
        model = WorldModel(obs_dim=obs_space.shape[0], hidden=hidden)
        rng = jax.random.key(seed)
        params = model.init(
            rng,
            jnp.zeros(obs_space.shape, jnp.float32),
            jnp.zeros(action_space.shape, jnp.float32),
        )
        model_ts = TrainState.create(
            apply_fn=model.apply, params=params, tx=optax.adam(lr)
        )
        buffer = ReplayBuffer.empty(buffer_size, obs_space, action_space)
        agent = cls(
            action_low=jnp.asarray(action_space.low, jnp.float32),
            action_high=jnp.asarray(action_space.high, jnp.float32),
            model=model,
            reward_fn=reward_fn,
            horizon=int(horizon),
            num_samples=int(num_samples),
        )
        return agent, MPCState(buffer=buffer, model_ts=model_ts)

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
        # Rejax calls this field ``done``. For model learning it must encode
        # true termination only: a time-limit truncation still has a valid
        # next-state target and remains learnable.
        del truncated
        transition = Minibatch(
            obs=obs,
            action=action,
            reward=reward,
            done=terminated,
            next_obs=next_obs,
        )
        return state.replace(buffer=state.buffer.append(transition))

    def train_step(
        self, state: MPCState, rng: jax.Array, batch_size: int = 256
    ) -> tuple[MPCState, jax.Array]:
        batch = state.buffer.sample(batch_size, rng)

        def loss_fn(params):
            pred = self.model.apply(params, batch.obs, batch.action)
            sq_err = jnp.mean((pred - batch.next_obs) ** 2, axis=-1)
            weight = 1.0 - batch.done.astype(sq_err.dtype)
            return jnp.sum(sq_err * weight) / jnp.maximum(jnp.sum(weight), 1.0)

        loss, grads = jax.value_and_grad(loss_fn)(state.model_ts.params)
        model_ts = state.model_ts.apply_gradients(grads=grads)
        return state.replace(model_ts=model_ts), loss

    def act(self, state: MPCState, obs: jax.Array, rng: jax.Array) -> jax.Array:
        action_dim = self.action_low.shape[0]
        sequences = jax.random.uniform(
            rng,
            (self.num_samples, self.horizon, action_dim),
            dtype=jnp.float32,
            minval=self.action_low,
            maxval=self.action_high,
        )

        def step(o, a):
            next_o = self.model.apply(state.model_ts.params, o, a)
            return next_o, self.reward_fn(o, a, next_o)

        def rollout_return(actions):
            _, rewards = jax.lax.scan(step, obs, actions)
            return rewards.sum()

        returns = jax.vmap(rollout_return)(sequences)
        return sequences[jnp.argmax(returns), 0]


class MPCRollout(NamedTuple):
    """Stacked per-step outputs from :func:`rollout`."""

    obs: jax.Array
    action: jax.Array
    reward: jax.Array
    next_obs: jax.Array
    terminated: jax.Array
    truncated: jax.Array
    valid: jax.Array
    env_state: object
    model_loss: jax.Array
    total_return: jax.Array

    @property
    def done(self) -> jax.Array:
        return self.valid & (self.terminated | self.truncated)


def _reject_autoreset(env) -> None:
    autoreset_types = (AutoResetWrapper, PooledInitVmapWrapper)
    layer = env
    while isinstance(layer, Wrapper):
        if isinstance(layer, autoreset_types):
            raise ValueError("MPC rollout requires a non-autoresetting environment")
        layer = layer.env


def rollout(
    agent: MPCAgent,
    state: MPCState,
    env,
    key: jax.Array,
    num_steps: int,
    *,
    train_batch_size: int = 256,
) -> tuple[MPCState, MPCRollout]:
    """Run one fixed-shape episode and train the model on valid transitions."""

    _reject_autoreset(env)
    init_key, rollout_key = jax.random.split(key)
    env_state0, init_info = env.init(init_key)
    obs0 = init_info.obs
    zero_action = jnp.zeros(env.action_space.shape, dtype=env.action_space.dtype)
    zero_reward = jnp.zeros_like(init_info.reward)
    nan_loss = jnp.asarray(jnp.nan, dtype=jnp.float32)

    def _real_step(carry):
        obs, env_state, mpc_state, rng, _active = carry
        rng, act_key, train_key = jax.random.split(rng, 3)
        action = agent.act(mpc_state, obs, act_key)
        next_env_state, info = env.step(env_state, action)
        mpc_state = agent.observe(
            mpc_state,
            obs,
            action,
            info.reward,
            info.terminated,
            info.truncated,
            info.obs,
        )

        def _train(ms):
            new_ms, loss = agent.train_step(ms, train_key, train_batch_size)
            return new_ms, loss.astype(jnp.float32)

        def _skip(ms):
            return ms, nan_loss

        do_train = mpc_state.buffer.num_entries >= train_batch_size
        mpc_state, loss = jax.lax.cond(do_train, _train, _skip, mpc_state)
        boundary = info.terminated | info.truncated
        return (info.obs, next_env_state, mpc_state, rng, ~boundary), (
            obs,
            action,
            info.reward,
            info.obs,
            info.terminated,
            info.truncated,
            jnp.ones_like(boundary, dtype=jnp.bool_),
            next_env_state,
            loss,
        )

    def _padding_step(carry):
        obs, env_state, _mpc_state, _rng, active = carry
        return carry, (
            obs,
            zero_action,
            zero_reward,
            obs,
            jnp.zeros_like(active, dtype=jnp.bool_),
            jnp.zeros_like(active, dtype=jnp.bool_),
            jnp.zeros_like(active, dtype=jnp.bool_),
            env_state,
            nan_loss,
        )

    def _step(carry, _):
        return jax.lax.cond(carry[4], _real_step, _padding_step, carry)

    carry = (
        obs0,
        env_state0,
        state,
        rollout_key,
        ~(init_info.terminated | init_info.truncated),
    )
    (_, _, final_state, _, _), trajectory = jax.lax.scan(_step, carry, None, num_steps)
    (
        obs_seq,
        action_seq,
        reward_seq,
        next_obs_seq,
        terminated_seq,
        truncated_seq,
        valid_seq,
        env_state_seq,
        loss_seq,
    ) = trajectory
    return final_state, MPCRollout(
        obs=obs_seq,
        action=action_seq,
        reward=reward_seq,
        next_obs=next_obs_seq,
        terminated=terminated_seq,
        truncated=truncated_seq,
        valid=valid_seq,
        env_state=env_state_seq,
        model_loss=loss_seq,
        total_return=jnp.sum(jnp.where(valid_seq, reward_seq, 0.0)),
    )
