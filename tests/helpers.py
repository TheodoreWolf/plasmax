"""Shared scenario fixtures for PlasmaxEnv unit tests.

Centralises the fast circular test scenario (TORAX config, actuator specs,
obs specs) used by the env, reward, wrapper, and controller test files.
Not collected by pytest (no ``*_test`` suffix); import as ``from helpers
import ...`` — pytest puts ``tests/`` on ``sys.path`` during collection.
"""

from functools import cached_property
from typing import Any

import jax
import jax.numpy as jnp
from envelope import (
    Continuous,
    Environment,
    FrozenPyTreeNode,
    InfoContainer,
    static_field,
)
from torax._src.orchestration import run_simulation
from torax._src.test_utils import default_configs
from torax._src.torax_pydantic import model_config

from plasmax import rewards as rewards_lib
from plasmax.environment import PlasmaxEnv
from plasmax.spaces import ActuatorSpec, ObsSpec

# Number of radial cells for fast test episodes.
N_RHO = 4

DEFAULT_ACTUATOR_SPECS: list[ActuatorSpec] = [
    ActuatorSpec("P_nbi", low=1e6, high=30e6),
    ActuatorSpec("gas_puff_rate", low=1e20, high=5e21),
]
NOMINAL_ACTION: jax.Array = jnp.array([10e6, 1e21], dtype=jnp.float32)

# Explicit obs specs (the registry carries no default scales). Profile order is
# part of the obs-layout contract (positional slices), so keep it stable:
# T_e, T_i, n_e, psi, q.
PROFILE_OBS_SPECS = [
    ObsSpec("T_e", 10.0),
    ObsSpec("T_i", 10.0),
    ObsSpec("n_e", 1e20),
    ObsSpec("psi", 10.0),
    ObsSpec("q", 5.0),
]
SCALAR_OBS_SPECS = [
    ObsSpec("W_thermal", 1e8),
    ObsSpec("tau_E", 1.0),
    ObsSpec("P_fusion", 1e8),
    ObsSpec("t", 10.0),
    ObsSpec("q_min", 3.0),
    ObsSpec("q95", 5.0),
    ObsSpec("beta_N", 3.0),
    ObsSpec("f_non_inductive", 1.0),
]


class CheapBoundaryState(FrozenPyTreeNode):
    """Minimal rollout state for controller boundary-contract tests."""

    obs: jax.Array
    steps: jax.Array


class CheapBoundaryEnv(Environment):
    """Cheap flat-array environment with configurable first-episode boundaries."""

    obs_dim: int = static_field()
    action_low: tuple[float, ...] = static_field()
    action_high: tuple[float, ...] = static_field()
    terminate_after: int | None = static_field(default=None)
    truncate_after: int | None = static_field(default=None)

    @cached_property
    def observation_space(self):
        return Continuous(
            low=jnp.full((self.obs_dim,), -jnp.inf, jnp.float32),
            high=jnp.full((self.obs_dim,), jnp.inf, jnp.float32),
        )

    @cached_property
    def action_space(self):
        return Continuous(
            low=jnp.asarray(self.action_low, jnp.float32),
            high=jnp.asarray(self.action_high, jnp.float32),
        )

    @staticmethod
    def _info(state, *, terminated=False, truncated=False):
        return InfoContainer(
            obs=state.obs,
            reward=state.steps.astype(jnp.float32),
            terminated=jnp.asarray(terminated),
            truncated=jnp.asarray(truncated),
        ).update(termination_code=jnp.where(terminated, jnp.int32(1), jnp.int32(-1)))

    def init(self, key):
        del key
        state = CheapBoundaryState(
            obs=jnp.zeros((self.obs_dim,), jnp.float32),
            steps=jnp.asarray(0, jnp.int32),
        )
        return state, self._info(state)

    def reset(self, state, key):
        del state
        return self.init(key)

    def step(self, state, action):
        del action
        next_state = state.replace(
            obs=state.obs + 1.0,
            steps=state.steps + 1,
        )
        terminated = (
            jnp.asarray(False)
            if self.terminate_after is None
            else next_state.steps >= self.terminate_after
        )
        truncated = (
            jnp.asarray(False)
            if self.truncate_after is None
            else next_state.steps >= self.truncate_after
        )
        return next_state, self._info(
            next_state,
            terminated=terminated,
            truncated=truncated,
        )


def make_test_config(
    n_rho: int = N_RHO, **torax_overrides: Any
) -> model_config.ToraxConfig:
    """Builds a minimal circular-geometry ToraxConfig for fast env tests.

    Keyword overrides replace the corresponding top-level TORAX config keys
    wholesale (no deep merge).
    """
    config = default_configs.get_default_config_dict()
    config["geometry"] = {"geometry_type": "circular", "n_rho": n_rho}
    config["numerics"] = {"t_final": 0.2, "fixed_dt": 0.1}
    config["time_step_calculator"] = {"calculator_type": "fixed"}
    config["sources"] = {
        "generic_heat": {"P_total": 10e6, "electron_heat_fraction": 0.5},
        "gas_puff": {"S_total": 1e21},
    }
    # Keep Greenwald fraction below 1.1 so early-termination doesn't fire.
    config["profile_conditions"] = {
        **config.get("profile_conditions", {}),
        "nbar": 0.5,
        "n_e_nbar_is_fGW": True,
        "normalize_n_e_to_nbar": True,
    }
    config.update(torax_overrides)
    return model_config.ToraxConfig.from_dict(config)


def make_default_step_fn(extra_config: dict[str, Any] | None = None):
    """A SimulationStepFn from TORAX's default (non-circular) config.

    Used by control/validation tests, which need the *unmodified* default
    geometry/sources (unlike make_test_config's fast circular scenario) so
    they can exercise actuator paths (e.g. ecrh) that aren't configured there.
    """
    config = default_configs.get_default_config_dict()
    config["numerics"] = {"t_final": 0.2, "fixed_dt": 0.1}
    config["time_step_calculator"] = {"calculator_type": "fixed"}
    if extra_config:
        config.update(extra_config)
    torax_config = model_config.ToraxConfig.from_dict(config)
    return run_simulation.make_step_fn(torax_config)


def make_test_env(config=None, **kwargs) -> PlasmaxEnv:
    """Builds a PlasmaxEnv over the fast test scenario; kwargs override defaults."""
    if config is None:
        config = make_test_config()
    initialization = kwargs.pop("initialization", None)
    return PlasmaxEnv.from_config(
        config=config,
        actuator_specs=kwargs.pop("actuator_specs", DEFAULT_ACTUATOR_SPECS),
        reward_fn=kwargs.pop("reward_fn", rewards_lib.Q_fusion),
        profile_obs_specs=kwargs.pop("profile_obs_specs", PROFILE_OBS_SPECS),
        scalar_obs_specs=kwargs.pop("scalar_obs_specs", SCALAR_OBS_SPECS),
        _initialization=initialization,
        **kwargs,
    )
