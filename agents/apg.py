"""Concurrent analytic policy gradients for native Envelope environments.

This adapts Wiedemann et al., ``Training Efficient Controllers via Analytic
Policy Gradient`` (ICRA 2023, https://arxiv.org/abs/2209.13052). A policy predicts
an entire action sequence once, differentiates its finite-horizon reward, and
executes only its first action before replanning. Training uses current online
states and native rewards; it does not reproduce the paper's reference-tracking
dataset or curriculum. The implementation is independent of the upstream code.
Reference code: https://github.com/lis-epfl/apg_trajectory_tracking.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import struct
from flax.training.train_state import TrainState

from agents._pathwise import PathwiseAgent, PathwiseState
from agents.backprop import DeterministicPolicy, PolicyCarry
from agents.direct_gradient import (
    apply_policy_optimizer_update,
    tree_is_finite,
)
from training.envelope_gymnax import to_typed_key

__all__ = ["APGAgent", "APGState"]


class _FirstTransition(NamedTuple):
    env_state: Any
    obs: jax.Array
    done: jax.Array
    simulator_steps: jax.Array


@struct.dataclass
class APGState(PathwiseState):
    """Policy optimizer and the current batch of committed simulator states."""

    actor_ts: TrainState


@dataclasses.dataclass(frozen=True, eq=False)
class APGAgent(PathwiseAgent):
    """Train a concurrent action-sequence policy without a critic or replay.

    ``global_step`` counts committed environment transitions. ``simulator_steps``
    additionally accounts for valid speculative lookahead, including the first
    transition committed by each update. The paper-derived optimizer defaults
    are starting values, not tuned plasma hyperparameters. Gradients are averaged
    across starting states and rewards are summed over the horizon.

    The policy predicts absolute commands within the environment's action bounds.
    Its zero-initialized output starts at the midpoint of those bounds.
    """

    num_rollouts: int = 8
    horizon: int = 10
    learning_rate: float = 1e-5
    momentum: float = 0.9
    grad_clip: float = 0.0
    policy: DeterministicPolicy = dataclasses.field(init=False, repr=False)
    optimizer: optax.GradientTransformation = dataclasses.field(init=False, repr=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive and finite")
        if not np.isfinite(self.momentum) or not 0 <= self.momentum < 1:
            raise ValueError("momentum must be finite and in [0, 1)")
        if not np.isfinite(self.grad_clip) or self.grad_clip < 0:
            raise ValueError("grad_clip must be nonnegative and finite")

        object.__setattr__(
            self,
            "policy",
            DeterministicPolicy(self.horizon * self.action_low.size, self.hidden_sizes),
        )
        transforms = (
            [optax.clip_by_global_norm(self.grad_clip)] if self.grad_clip > 0 else []
        )
        object.__setattr__(
            self,
            "optimizer",
            optax.chain(
                *transforms,
                optax.sgd(self.learning_rate, momentum=self.momentum),
            ),
        )

    @property
    def steps_per_update(self) -> int:
        return self.num_rollouts

    def init_state(self, rng: jax.Array) -> APGState:
        env_rng, network_rng = jax.random.split(to_typed_key(rng))
        params = self.policy.init(network_rng, jnp.zeros(self.obs_shape, jnp.float32))[
            "params"
        ]
        return APGState(
            **self.initial_state_fields(env_rng),
            actor_ts=TrainState.create(
                apply_fn=self.policy.apply, params=params, tx=self.optimizer
            ),
        )

    def action_sequence(self, params: Any, obs: jax.Array) -> jax.Array:
        """Predict all horizon actions from this observation in one policy call."""
        logits = self.policy.apply({"params": params}, jnp.asarray(obs, jnp.float32))
        logits = logits.reshape((self.horizon, self.action_low.size))
        actions = self.action_loc + self.action_scale * jnp.tanh(logits)
        return actions.astype(self.env.action_space.dtype)

    def _rollout_loss(
        self, params: Any, env_state: Any, obs: jax.Array
    ) -> tuple[jax.Array, _FirstTransition]:
        actions = self.action_sequence(params, obs)
        first_state, first_info = self.env.step(env_state, actions[0])
        first_done = first_info.terminated | first_info.truncated
        zero_reward = jnp.zeros_like(first_info.reward)

        def step(
            carry: PolicyCarry, action: jax.Array
        ) -> tuple[PolicyCarry, tuple[jax.Array, jax.Array]]:
            def active(_: None) -> tuple[PolicyCarry, tuple[jax.Array, jax.Array]]:
                next_state, info = self.env.step(carry.env_state, action)
                done = info.terminated | info.truncated
                return PolicyCarry(info.obs, next_state, ~done), (
                    info.reward,
                    jnp.int32(1),
                )

            return jax.lax.cond(
                carry.alive,
                active,
                lambda _: (carry, (zero_reward, jnp.int32(0))),
                None,
            )

        scan_step = jax.checkpoint(step) if self.remat else step
        _, (rewards, alive) = jax.lax.scan(
            scan_step,
            PolicyCarry(first_info.obs, first_state, ~first_done),
            actions[1:],
        )
        return -(first_info.reward + jnp.sum(rewards)), _FirstTransition(
            first_state,
            first_info.obs,
            first_done,
            jnp.int32(1) + jnp.sum(alive),
        )

    def empty_diagnostics(self) -> dict[str, jax.Array]:
        names = (
            "loss",
            "grad_norm",
            "grad_clip_scale",
            "optimizer_update_finite",
            "params_finite",
            "grads_finite",
            "simulator_steps",
        )
        return {f"train/{name}": jnp.asarray(0.0, jnp.float32) for name in names}

    def update(
        self, state: APGState
    ) -> tuple[APGState, dict[str, jax.Array], jax.Array]:
        (losses, first), per_rollout_grads = jax.vmap(
            jax.value_and_grad(self._rollout_loss, has_aux=True),
            in_axes=(None, 0, 0),
        )(state.actor_ts.params, state.env_state, state.obs)
        grads = jax.tree.map(lambda value: jnp.mean(value, axis=0), per_rollout_grads)
        grads_finite = tree_is_finite(per_rollout_grads)
        params, opt_state, diagnostics = apply_policy_optimizer_update(
            self.optimizer,
            grads,
            state.actor_ts.opt_state,
            state.actor_ts.params,
            self.grad_clip,
        )
        rng, reset_rng = jax.random.split(state.rng)

        def commit(
            transition: _FirstTransition, key: jax.Array
        ) -> tuple[Any, jax.Array, jax.Array]:
            def reset(_: None) -> tuple[Any, jax.Array, jax.Array]:
                reset_state, info = self.env.reset(transition.env_state, key)
                return reset_state, info.obs, ~(info.terminated | info.truncated)

            return jax.lax.cond(
                transition.done,
                reset,
                lambda _: (transition.env_state, transition.obs, jnp.asarray(True)),
                None,
            )

        env_state, obs, reset_ready = jax.vmap(commit)(
            first, jax.random.split(reset_rng, self.num_rollouts)
        )
        actor_ts = state.actor_ts.replace(
            step=state.actor_ts.step + 1, params=params, opt_state=opt_state
        )
        candidate = state.replace(
            rng=rng,
            actor_ts=actor_ts,
            env_state=jax.tree.map(jax.lax.stop_gradient, env_state),
            obs=jax.lax.stop_gradient(jnp.asarray(obs, jnp.float32)),
        )
        metrics = {
            "train/loss": jnp.mean(losses),
            "train/grad_norm": diagnostics.grad_norm,
            "train/grad_clip_scale": diagnostics.grad_clip_scale,
            "train/optimizer_update_finite": diagnostics.optimizer_update_finite,
            "train/params_finite": diagnostics.params_finite,
            "train/grads_finite": grads_finite,
            "train/simulator_steps": jnp.sum(first.simulator_steps),
        }
        safe = (
            jnp.all(jnp.isfinite(losses))
            & grads_finite
            & diagnostics.optimizer_update_finite
            & diagnostics.params_finite
            & tree_is_finite(opt_state)
            & tree_is_finite(obs)
            & jnp.all(reset_ready)
        )
        return candidate, jax.tree.map(lambda x: x.astype(jnp.float32), metrics), safe

    def make_act(
        self, state: APGState, deterministic: bool | None = False
    ) -> Callable[[jax.Array, jax.Array], jax.Array]:
        del deterministic

        def act(obs: jax.Array, rng: jax.Array) -> jax.Array:
            del rng
            return self.action_sequence(state.actor_ts.params, obs)[0]

        return act
