"""Thin Rejax PPO adapter for clone-only plasmax training."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import fields

import jax
import jax.numpy as jnp
from flax import linen as nn
from rejax.algos.ppo import PPO
from rejax.networks import MLP, VNetwork

from agents.normalization import EnvelopeNormalizationMixin
from training.envelope_gymnax import GymnaxMultiDiscrete


def _categorical_log_prob_entropy(
    logits: jax.Array, action: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Score a categorical action without relying on removed Distrax APIs."""
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    selected = jnp.take_along_axis(log_probs, action[..., None], axis=-1).squeeze(-1)
    safe_log_probs = jnp.where(jnp.isfinite(logits), log_probs, 0.0)
    entropy = -jnp.sum(jax.nn.softmax(logits, axis=-1) * safe_log_probs, axis=-1)
    return selected, entropy


class MultiDiscretePolicy(nn.Module):
    """Independent categorical heads over a shared feature trunk."""

    nvec: Sequence[int]
    hidden_layer_sizes: Sequence[int]
    activation: Callable

    def setup(self) -> None:
        self.features = MLP(self.hidden_layer_sizes, self.activation)
        self.heads = nn.vmap(
            nn.Dense,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            in_axes=None,
            out_axes=-2,
            axis_size=len(self.nvec),
        )(features=max(self.nvec))

    def _padded_logits(self, obs: jax.Array) -> jax.Array:
        features = self.features(obs)
        max_categories = max(self.nvec)
        logits = self.heads(features)
        valid = jnp.arange(max_categories) < jnp.asarray(self.nvec)[:, None]
        return jnp.where(valid, logits, -jnp.inf)

    def __call__(
        self, obs: jax.Array, rng: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        logits = self._padded_logits(obs)
        rngs = jax.random.split(rng, len(self.nvec))

        def sample_head(head_logits, head_rng):
            action = jax.random.categorical(head_rng, head_logits, axis=-1)
            log_prob, entropy = _categorical_log_prob_entropy(head_logits, action)
            return action, log_prob, entropy

        actions, log_probs, entropies = jax.vmap(
            sample_head,
            in_axes=(-2, 0),
            out_axes=-1,
        )(logits, rngs)
        return actions, log_probs.sum(-1), entropies.sum(-1)

    def act(self, obs: jax.Array, rng: jax.Array) -> jax.Array:
        action, _, _ = self(obs, rng)
        return action

    def log_prob_entropy(
        self, obs: jax.Array, action: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        logits = self._padded_logits(obs)
        log_probs, entropies = jax.vmap(
            _categorical_log_prob_entropy,
            in_axes=(-2, -1),
            out_axes=-1,
        )(logits, action)
        return log_probs.sum(-1), entropies.sum(-1)

    def action_log_prob(
        self, obs: jax.Array, rng: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        action, log_prob, _ = self(obs, rng)
        return action, log_prob


class PPOAdapter(EnvelopeNormalizationMixin, PPO):
    """Upstream Rejax PPO with plasmax action-space and eval adapters."""

    @classmethod
    def create(cls, **config):
        callback = config.pop("eval_callback", None)
        instance = super().create(**config)
        return instance if callback is None else instance.with_eval_callback(callback)

    @property
    def config(self) -> dict:
        """Return a trace-safe shallow config for Envelope environments."""
        return {field.name: getattr(self, field.name) for field in fields(self)}

    def with_eval_callback(self, callback: Callable) -> PPOAdapter:
        """Attach a repository callback without changing Rejax optimization."""

        def wrapped(algo, train_state, rng):
            return callback(algo, train_state, rng, None)

        return self.replace(eval_callback=wrapped)

    @classmethod
    def create_agent(cls, config, env, env_params):
        action_space = env.action_space(env_params)
        agent_kwargs = dict(config.pop("agent_kwargs", {}))
        activation = agent_kwargs.pop("activation", "swish")
        hidden_layer_sizes = agent_kwargs.pop("hidden_layer_sizes", (64, 64))

        if not isinstance(action_space, GymnaxMultiDiscrete):
            config["agent_kwargs"] = {
                **agent_kwargs,
                "activation": activation,
                "hidden_layer_sizes": tuple(hidden_layer_sizes),
            }
            return super().create_agent(config, env, env_params)

        agent_kwargs["activation"] = getattr(nn, activation)
        agent_kwargs["hidden_layer_sizes"] = tuple(hidden_layer_sizes)
        return {
            "actor": MultiDiscretePolicy(nvec=action_space.nvec, **agent_kwargs),
            "critic": VNetwork(**agent_kwargs),
        }

    def make_deterministic_act(self, train_state):
        """Return the clipped mode policy used by deterministic evaluation."""

        def act(obs, rng):
            del rng
            if getattr(self, "normalize_observations", False):
                obs = self.normalize_obs(train_state.obs_rms_state, obs)
            obs = jnp.expand_dims(obs, 0)
            if isinstance(self.action_space, GymnaxMultiDiscrete):
                logits = self.actor.apply(
                    train_state.actor_ts.params,
                    obs,
                    method="_padded_logits",
                )
                action = jnp.argmax(logits, axis=-1)
            else:
                action_dist = self.actor.apply(
                    train_state.actor_ts.params,
                    obs,
                    method="_action_dist",
                )
                action = action_dist.mode()
                if not self.discrete:
                    action = jnp.clip(
                        action, self.action_space.low, self.action_space.high
                    )
            return jnp.squeeze(action, axis=0)

        return act

    def make_act(self, train_state, deterministic: bool = False):
        """Bind an inference snapshot while preserving singleton action axes."""
        if deterministic:
            return self.make_deterministic_act(train_state)
        sample = super().make_act(train_state)
        return lambda obs, rng: jnp.reshape(sample(obs, rng), self.action_space.shape)

    @property
    def discrete(self):
        if isinstance(self.action_space, GymnaxMultiDiscrete):
            return True
        return super().discrete

    @property
    def action_dim(self):
        if isinstance(self.action_space, GymnaxMultiDiscrete):
            return sum(self.action_space.nvec)
        return super().action_dim


__all__ = ["MultiDiscretePolicy", "PPOAdapter"]
