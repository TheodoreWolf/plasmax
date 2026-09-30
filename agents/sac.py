"""Thin Rejax SAC adapter for clone-only plasmax training."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import fields
from typing import Any

import distrax
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax import struct
from rejax.algos.sac import SAC
from rejax.networks import MLP, SquashedGaussianPolicy

from agents.normalization import EnvelopeNormalizationMixin
from agents.ppo import _reset_action_setpoint
from agents.sac_numerics import check_numerics


class ResidualSquashedGaussianPolicy(SquashedGaussianPolicy):
    """SAC Gaussian with a zero-initialized latent residual at a reset setpoint.

    Setpoints use environment action units. Bound setpoints move inward by
    0.05% of the action span so their inverse tanh and gradients remain finite,
    matching the existing pathwise agents' setpoint convention. Sampling,
    log-probabilities, and the learned standard deviation retain upstream SAC
    behavior.
    """

    action_setpoint: tuple[float, ...]

    def setup(self) -> None:
        self.features = MLP(self.hidden_layer_sizes, self.activation)
        self.action_mean = nn.Dense(
            self.action_dim,
            kernel_init=nn.initializers.zeros_init(),
            bias_init=nn.initializers.zeros_init(),
        )
        self.action_log_std = nn.Dense(self.action_dim)
        self.bij = distrax.Tanh()

    def _action_dist(self, obs: jax.Array) -> distrax.Distribution:
        residual_dist = super()._action_dist(obs)
        setpoint = jnp.asarray(self.action_setpoint, dtype=residual_dist.mean().dtype)
        normalized = (setpoint - self.action_loc) / self.action_scale
        latent_setpoint = jnp.arctanh(jnp.clip(normalized, -0.999, 0.999))
        return distrax.MultivariateNormalDiag(
            loc=latent_setpoint + residual_dist.mean(),
            scale_diag=residual_dist.stddev(),
        )


class SACAdapter(EnvelopeNormalizationMixin, SAC):
    """Upstream Rejax SAC with callback and deterministic-eval adapters."""

    diagnose_numerics: bool = struct.field(pytree_node=False, default=False)

    def collect_transitions(self, ts: Any) -> Any:
        if self.diagnose_numerics:
            check_numerics("before_collection", ts, last_obs=ts.last_obs)
        ts, batch = super().collect_transitions(ts)
        if self.diagnose_numerics:
            check_numerics(
                "after_collection",
                ts,
                **{
                    f"raw_{name}": getattr(batch, name)
                    for name in ("obs", "next_obs", "action", "reward")
                },
            )
        return ts, batch

    def update_actor(self, ts: Any, mb: Any) -> Any:
        if self.diagnose_numerics:
            check_numerics(
                "before_actor_update",
                ts,
                **{
                    f"minibatch_{name}": getattr(mb, name)
                    for name in ("obs", "next_obs", "action", "reward")
                },
            )
        ts, logprob = super().update_actor(ts, mb)
        if self.diagnose_numerics:
            check_numerics("after_actor_update", ts, logprob=logprob)
        return ts, logprob

    def update_critic(self, ts: Any, mb: Any) -> Any:
        ts = super().update_critic(ts, mb)
        if self.diagnose_numerics:
            check_numerics("after_critic_update", ts)
        return ts

    def update_alpha(self, ts: Any, logprob: Any) -> Any:
        ts = super().update_alpha(ts, logprob)
        if self.diagnose_numerics:
            check_numerics("after_alpha_update", ts)
        return ts

    @classmethod
    def create(cls, **config):
        callback = config.pop("eval_callback", None)
        instance = super().create(**config)
        return instance if callback is None else instance.with_eval_callback(callback)

    @classmethod
    def create_agent(
        cls, config: dict[str, Any], env: Any, env_params: Any
    ) -> dict[str, nn.Module]:
        agent_kwargs = dict(config.get("agent_kwargs", {}))
        residual_policy = agent_kwargs.pop("residual_policy", False)
        config["agent_kwargs"] = agent_kwargs
        agents = super().create_agent(config, env, env_params)
        if not residual_policy:
            return agents
        actor = agents["actor"]
        if not isinstance(actor, SquashedGaussianPolicy):
            raise ValueError("residual_policy requires a continuous action space")
        agents["actor"] = ResidualSquashedGaussianPolicy(
            action_dim=actor.action_dim,
            action_range=actor.action_range,
            hidden_layer_sizes=actor.hidden_layer_sizes,
            activation=actor.activation,
            log_std_range=actor.log_std_range,
            action_setpoint=_reset_action_setpoint(env, env_params),
        )
        return agents

    @property
    def config(self) -> dict:
        """Return a trace-safe shallow config for Envelope environments."""
        return {field.name: getattr(self, field.name) for field in fields(self)}

    def with_eval_callback(self, callback: Callable) -> SACAdapter:
        """Attach a repository callback without changing Rejax optimization."""

        def wrapped(algo, train_state, rng):
            return callback(algo, train_state, rng, None)

        return self.replace(eval_callback=wrapped)

    def make_deterministic_act(self, train_state):
        """Return the distribution-mode policy used for evaluation."""

        def act(obs, rng):
            del rng
            if self.normalize_observations:
                obs = self.normalize_obs(train_state.obs_rms_state, obs)
            obs = jnp.expand_dims(obs, 0)
            distribution = self.actor.apply(
                train_state.actor_ts.params,
                obs,
                method="_action_dist",
            )
            action = distribution.mode()
            if not self.discrete:
                action = jnp.tanh(action)
                action = self.actor.action_loc + action * self.actor.action_scale
            return jnp.squeeze(action, axis=0)

        return act

    def make_act(self, train_state, deterministic: bool = False):
        """Bind an inference snapshot while preserving singleton action axes."""
        if deterministic:
            return self.make_deterministic_act(train_state)
        sample = super().make_act(train_state)
        return lambda obs, rng: jnp.reshape(sample(obs, rng), self.action_space.shape)


__all__ = ["ResidualSquashedGaussianPolicy", "SACAdapter"]
