"""Thin Rejax SAC adapter for clone-only plasmax training."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import fields

import jax.numpy as jnp
from rejax.algos.sac import SAC


class SACAdapter(SAC):
    """Upstream Rejax SAC with callback and deterministic-eval adapters."""

    @classmethod
    def create(cls, **config):
        callback = config.pop("eval_callback", None)
        instance = super().create(**config)
        return instance if callback is None else instance.with_eval_callback(callback)

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


__all__ = ["SACAdapter"]
