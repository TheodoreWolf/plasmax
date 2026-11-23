"""Clone-only Soft Analytic Policy Optimization on native Envelope environments.

Implements Xing et al., ICLR 2025, https://arxiv.org/abs/2412.12089, using
Rejax's squashed policy and normalization primitives. Reference configuration:
https://github.com/etaoxing/mineral/tree/12a660435e6d71bf05872e5236234d90af02e6fd
This is an independent implementation of the published algorithm.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any, NamedTuple

import distrax
import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax import struct
from flax.training.train_state import TrainState
from rejax.algos.mixins import RMSState, update_rms
from rejax.networks import SquashedGaussianPolicy

from agents._pathwise import PathwiseAgent, PathwiseState
from agents.direct_gradient import tree_is_finite
from training.envelope_gymnax import to_typed_key


class _Features(nn.Module):
    hidden_sizes: tuple[int, ...]

    @nn.compact
    def __call__(self, obs: jax.Array) -> jax.Array:
        x = obs.reshape((obs.shape[0], -1))
        for width in self.hidden_sizes:
            x = nn.Dense(width, kernel_init=nn.initializers.orthogonal(1.0))(x)
            x = nn.silu(nn.LayerNorm(epsilon=1e-5)(x))
        return x


class _Actor(SquashedGaussianPolicy):
    """Keep Rejax's stable sampling; replace only its feature architecture."""

    def setup(self) -> None:
        self.features = _Features(tuple(self.hidden_layer_sizes))
        initializer = nn.initializers.orthogonal(1.0)
        self.action_mean = nn.Dense(self.action_dim, kernel_init=initializer)
        self.action_log_std = nn.Dense(self.action_dim, kernel_init=initializer)
        self.bij = distrax.Tanh()


class _Value(nn.Module):
    hidden_sizes: tuple[int, ...]

    @nn.compact
    def __call__(self, obs: jax.Array) -> jax.Array:
        x = _Features(self.hidden_sizes)(obs)
        return nn.Dense(1, kernel_init=nn.initializers.orthogonal(1.0))(x)[..., 0]


@struct.dataclass
class SAPOState(PathwiseState):
    actor_ts: TrainState
    critic_ts: TrainState
    alpha_ts: TrainState
    obs_rms_state: RMSState


class _Transition(NamedTuple):
    obs: jax.Array
    raw_obs: jax.Array
    reward: jax.Array
    entropy: jax.Array
    next_value: jax.Array
    boundary: jax.Array
    reset_valid: jax.Array
    active: jax.Array


def _soft_lambda_returns(
    rewards: jax.Array,
    next_values: jax.Array,
    boundaries: jax.Array,
    gamma: float,
    td_lambda: float,
) -> jax.Array:
    """Finite-window lambda returns; terminal bootstraps are already zero."""

    def step(carry: jax.Array, data: tuple) -> tuple[jax.Array, jax.Array]:
        reward, value, boundary = data
        continuation = jnp.where(
            boundary, value, (1.0 - td_lambda) * value + td_lambda * carry
        )
        target = reward + gamma * continuation
        return target, target

    _, targets = jax.lax.scan(
        step,
        next_values[-1],
        (rewards, next_values, boundaries),
        reverse=True,
    )
    return targets


@dataclasses.dataclass(frozen=True, eq=False)
class SAPOAgent(PathwiseAgent):
    """On-policy stochastic actor with short simulator gradients and soft values."""

    gamma: float = 0.99
    td_lambda: float = 0.95
    actor_learning_rate: float = 2e-3
    critic_learning_rate: float = 5e-4
    alpha_learning_rate: float = 5e-3
    min_learning_rate: float = 1e-5
    max_grad_norm: float = 0.5
    critic_epochs: int = 16
    num_minibatches: int = 4
    normalize_observations: bool = True
    initial_alpha: float = 1.0
    actor: nn.Module = dataclasses.field(init=False, repr=False)
    critic: nn.Module = dataclasses.field(init=False, repr=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        if not 0 <= self.gamma <= 1 or not 0 <= self.td_lambda <= 1:
            raise ValueError("gamma and td_lambda must be in [0, 1]")
        if self.critic_epochs <= 0 or self.num_minibatches <= 0:
            raise ValueError("critic_epochs and num_minibatches must be positive")
        if self.steps_per_update % self.num_minibatches:
            raise ValueError("num_minibatches must divide num_rollouts * horizon")
        if (
            min(
                self.actor_learning_rate,
                self.critic_learning_rate,
                self.alpha_learning_rate,
                self.min_learning_rate,
                self.max_grad_norm,
                self.initial_alpha,
            )
            <= 0
        ):
            raise ValueError(
                "learning rates, max_grad_norm and initial_alpha must be positive"
            )
        object.__setattr__(
            self,
            "actor",
            _Actor(
                action_dim=self.action_low.size,
                action_range=(self.action_low, self.action_high),
                hidden_layer_sizes=self.hidden_sizes,
                activation=nn.silu,
                log_std_range=(-5.0, 2.0),
            ),
        )
        object.__setattr__(self, "critic", _Value(self.hidden_sizes))

    @property
    def steps_per_update(self) -> int:
        return self.num_rollouts * self.horizon

    @property
    def target_entropy(self) -> float:
        return -float(self.action_low.size) / 2.0

    def _optimizer(
        self,
        learning_rate: float,
        updates: int | None,
        *,
        updates_per_iteration: int = 1,
    ) -> Any:
        rate = learning_rate
        if updates is not None:
            schedule = optax.linear_schedule(
                learning_rate, self.min_learning_rate, max(updates - 1, 1)
            )

            def rate(count: jax.Array) -> jax.Array:
                return schedule(count // updates_per_iteration)

        return optax.chain(
            optax.clip_by_global_norm(self.max_grad_norm),
            optax.adamw(rate, b1=0.7, b2=0.95, weight_decay=0.01),
        )

    def init_state(self, rng: jax.Array) -> SAPOState:
        env_rng, actor_rng, actor_sample_rng, critic_rng = jax.random.split(
            to_typed_key(rng), 4
        )
        fields = self.initial_state_fields(env_rng)
        obs = jnp.zeros((1, *self.obs_shape), jnp.float32)
        actor_params = self.actor.init(actor_rng, obs, actor_sample_rng)
        critic_params = jax.vmap(self.critic.init, in_axes=(0, None))(
            jax.random.split(critic_rng, 2), obs
        )
        return SAPOState(
            **fields,
            actor_ts=TrainState.create(
                apply_fn=self.actor.apply,
                params=actor_params,
                tx=self._optimizer(self.actor_learning_rate, self.num_updates),
            ),
            critic_ts=TrainState.create(
                apply_fn=self.critic.apply,
                params=critic_params,
                tx=self._optimizer(
                    self.critic_learning_rate,
                    self.num_updates,
                    updates_per_iteration=self.critic_epochs * self.num_minibatches,
                ),
            ),
            alpha_ts=TrainState.create(
                apply_fn=(),
                params={
                    "log_alpha": jnp.log(jnp.asarray(self.initial_alpha, jnp.float32))
                },
                tx=self._optimizer(self.alpha_learning_rate, None),
            ),
            obs_rms_state=RMSState.create(self.obs_shape).replace(
                count=jnp.asarray(1e-4, jnp.float32)
            ),
        )

    def _normalize(self, rms: RMSState, obs: jax.Array) -> jax.Array:
        if not self.normalize_observations:
            return obs
        return (obs - rms.mean) / jnp.sqrt(rms.var + 1e-5)

    def _entropy(self, entropy: jax.Array) -> jax.Array:
        target = abs(self.target_entropy)
        return (entropy + target) / (2.0 * target)

    def _values(self, params: Any, obs: jax.Array) -> jax.Array:
        return jax.vmap(self.critic.apply, in_axes=(0, None))(params, obs)

    def _actor_loss_one(
        self,
        params: Any,
        state: SAPOState,
        env_state: Any,
        raw_obs: jax.Array,
        rng: jax.Array,
    ) -> tuple[jax.Array, Any]:
        """One native rollout, retaining feedback and terminal-value gradients."""
        rms = jax.tree.map(jax.lax.stop_gradient, state.obs_rms_state)
        alpha = jax.lax.stop_gradient(jnp.exp(state.alpha_ts.params["log_alpha"]))

        def reset_one(
            env_state: Any, obs: jax.Array, done: jax.Array, key: jax.Array
        ) -> tuple:
            def reset(_: None) -> tuple:
                reset_state, info = self.env.reset(env_state, key)
                return jax.tree.map(
                    jax.lax.stop_gradient,
                    (
                        reset_state,
                        jnp.asarray(info.obs, jnp.float32),
                        ~(info.terminated | info.truncated),
                    ),
                )

            return jax.lax.cond(
                done, reset, lambda _: (env_state, obs, jnp.asarray(True)), None
            )

        def active_step(carry: tuple, index: jax.Array) -> tuple:
            rng, env_state, raw_obs, discount, total, _ = carry
            rng, action_rng, entropy_rng, reset_rng = jax.random.split(rng, 4)
            obs = self._normalize(rms, raw_obs)
            actions, _ = self.actor.apply(params, obs[None], action_rng)
            _, entropy_logprob = self.actor.apply(params, obs[None], entropy_rng)
            entropy = -entropy_logprob[0]
            next_state, info = self.env.step(
                env_state, actions[0].astype(self.env.action_space.dtype)
            )
            terminal_obs = jnp.asarray(info.obs, jnp.float32)
            reward = jnp.asarray(info.reward, jnp.float32)
            done = info.terminated | info.truncated
            # A terminal state has no value. Avoid evaluating the critic on a
            # terminal observation that may be outside the simulator's domain.
            value_obs = jnp.where(info.terminated, 0.0, terminal_obs)
            values = self._values(
                state.critic_ts.params, self._normalize(rms, value_obs)[None]
            )[:, 0]
            values = jnp.where(info.terminated, 0.0, values)
            boundary = done | (index == self.horizon - 1)
            total += discount * (
                reward
                + alpha * self._entropy(entropy)
                + self.gamma * jnp.where(boundary, values.mean(axis=0), 0.0)
            )
            next_state, next_obs, reset_valid = reset_one(
                next_state,
                terminal_obs,
                done,
                reset_rng,
            )
            next_carry = (
                rng,
                next_state,
                next_obs,
                jnp.where(done, 1.0, discount * self.gamma),
                total,
                reset_valid,
            )
            transition = _Transition(
                obs,
                raw_obs,
                reward,
                entropy,
                values.min(axis=0),
                boundary,
                reset_valid,
                jnp.asarray(True),
            )
            return next_carry, transition

        def inactive_step(carry: tuple, _: jax.Array) -> tuple:
            obs = carry[2]
            zero = jnp.asarray(0.0, jnp.float32)
            return carry, _Transition(
                self._normalize(rms, obs),
                obs,
                zero,
                zero,
                zero,
                jnp.asarray(True),
                jnp.asarray(False),
                jnp.asarray(False),
            )

        def step(carry: tuple, index: jax.Array) -> tuple:
            return jax.lax.cond(carry[-1], active_step, inactive_step, carry, index)

        scan_step = jax.checkpoint(step) if self.remat else step
        carry, batch = jax.lax.scan(
            scan_step,
            (
                rng,
                env_state,
                raw_obs,
                jnp.asarray(1.0, jnp.float32),
                jnp.asarray(0.0, jnp.float32),
                jnp.asarray(True),
            ),
            jnp.arange(self.horizon),
        )
        _, env_state, obs, _, total, _ = carry
        return -total / self.horizon, (env_state, obs, batch)

    def _actor_loss(self, params: Any, state: SAPOState) -> tuple[jax.Array, Any]:
        """Batched objective, also usable directly with ``value_and_grad``."""
        rng, rollout_rng = jax.random.split(state.rng)
        losses, (env_state, obs, batch) = jax.lax.map(
            lambda inputs: self._actor_loss_one(params, state, *inputs),
            (
                state.env_state,
                state.obs,
                jax.random.split(rollout_rng, self.num_rollouts),
            ),
        )
        batch = jax.tree.map(lambda leaf: jnp.swapaxes(leaf, 0, 1), batch)
        return losses.mean(), (rng, env_state, obs, batch)

    def update(
        self, state: SAPOState
    ) -> tuple[SAPOState, dict[str, jax.Array], jax.Array]:
        rng, rollout_rng = jax.random.split(state.rng)
        # Differentiate scalar simulator calls before batching. Batching a
        # custom JVP containing conditionals before transposition can produce
        # non-transposable stop_gradient primitives in JAX's conditional rule.
        (actor_losses, data), rollout_grads = jax.vmap(
            jax.value_and_grad(self._actor_loss_one, has_aux=True),
            in_axes=(None, None, 0, 0, 0),
        )(
            state.actor_ts.params,
            state,
            state.env_state,
            state.obs,
            jax.random.split(rollout_rng, self.num_rollouts),
        )
        actor_loss = actor_losses.mean()
        actor_grads = jax.tree.map(lambda leaf: leaf.mean(axis=0), rollout_grads)
        env_state, obs, batch = jax.tree.map(jax.lax.stop_gradient, data)
        batch = jax.tree.map(lambda leaf: jnp.swapaxes(leaf, 0, 1), batch)
        actor_ts = state.actor_ts.apply_gradients(grads=actor_grads)

        def alpha_loss_fn(params: Any) -> jax.Array:
            return (
                jnp.exp(params["log_alpha"])
                * abs(self.target_entropy)
                * (batch.entropy.mean() - self.target_entropy)
            )

        alpha_loss, alpha_grads = jax.value_and_grad(alpha_loss_fn)(
            state.alpha_ts.params
        )
        alpha_ts = state.alpha_ts.apply_gradients(grads=alpha_grads)
        alpha = jnp.exp(alpha_ts.params["log_alpha"])
        targets = _soft_lambda_returns(
            batch.reward + alpha * self._entropy(batch.entropy),
            batch.next_value,
            batch.boundary,
            self.gamma,
            self.td_lambda,
        )
        targets = jax.lax.stop_gradient(targets)
        minibatch_obs = batch.obs.reshape(self.num_minibatches, -1, *self.obs_shape)
        minibatch_targets = targets.reshape(self.num_minibatches, -1)

        def critic_step(carry: tuple, data: tuple) -> tuple:
            critic_ts, all_finite = carry
            observations, target = data

            def loss_fn(params: Any) -> jax.Array:
                return jnp.mean(jnp.square(self._values(params, observations) - target))

            loss, grads = jax.value_and_grad(loss_fn)(critic_ts.params)
            finite = tree_is_finite((loss, grads))
            return (critic_ts.apply_gradients(grads=grads), all_finite & finite), loss

        def critic_epoch(carry: tuple, _: None) -> tuple:
            return jax.lax.scan(critic_step, carry, (minibatch_obs, minibatch_targets))

        (critic_ts, critic_finite), critic_losses = jax.lax.scan(
            critic_epoch,
            (state.critic_ts, jnp.asarray(True)),
            None,
            length=self.critic_epochs,
        )
        rms = state.obs_rms_state
        if self.normalize_observations:
            observations = jnp.concatenate((batch.raw_obs, obs[None]), axis=0)
            rms = update_rms(rms, observations.reshape(-1, *self.obs_shape))
        candidate = state.replace(
            rng=rng,
            env_state=env_state,
            obs=obs,
            actor_ts=actor_ts,
            critic_ts=critic_ts,
            alpha_ts=alpha_ts,
            obs_rms_state=rms,
        )
        safe = (
            jnp.all(batch.reset_valid)
            & critic_finite
            & tree_is_finite(
                (
                    actor_loss,
                    alpha_loss,
                    actor_grads,
                    alpha_grads,
                    targets,
                    obs,
                    rms,
                    actor_ts.params,
                    actor_ts.opt_state,
                    critic_ts.params,
                    critic_ts.opt_state,
                    alpha_ts.params,
                    alpha_ts.opt_state,
                )
            )
        )
        metrics = {
            "train/actor_loss": actor_loss,
            "train/critic_loss": critic_losses.mean(),
            "train/alpha": alpha,
            "train/entropy": batch.entropy.mean(),
            "train/actor_grad_norm": optax.global_norm(actor_grads),
            "train/simulator_steps": batch.active.sum().astype(jnp.float32),
        }
        return candidate, jax.tree.map(lambda x: x.astype(jnp.float32), metrics), safe

    def empty_diagnostics(self) -> dict[str, jax.Array]:
        return {
            f"train/{name}": jnp.asarray(0.0, jnp.float32)
            for name in (
                "actor_loss",
                "critic_loss",
                "alpha",
                "entropy",
                "actor_grad_norm",
                "simulator_steps",
            )
        }

    def make_act(self, state: SAPOState, deterministic: bool = False) -> Callable:
        def act(obs: jax.Array, rng: jax.Array) -> jax.Array:
            obs = self._normalize(state.obs_rms_state, jnp.asarray(obs, jnp.float32))[
                None
            ]
            if deterministic:
                distribution = self.actor.apply(
                    state.actor_ts.params, obs, method="_action_dist"
                )
                action = self.action_loc + self.action_scale * jnp.tanh(
                    distribution.mode()
                )
            else:
                action, _ = self.actor.apply(
                    state.actor_ts.params, obs, to_typed_key(rng)
                )
            return action.reshape(self.action_low.shape)

        return act


__all__ = ["SAPOAgent", "SAPOState"]
