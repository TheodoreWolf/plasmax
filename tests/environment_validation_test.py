"""Construction-time validation for actuator and randomization contracts."""

import jax
import jax.numpy as jnp
import pytest
from helpers import make_default_step_fn, make_test_env

from plasmax.environment.schema import PhysicsRandomizationSpec
from plasmax.environment.validation import (
    validate_actuator_specs,
    validate_physics_randomization,
)
from plasmax.spaces import ActuatorSpec


def _provider(extra_config=None):
    return make_default_step_fn(extra_config).runtime_params_provider


def _spec(name: str) -> ActuatorSpec:
    return ActuatorSpec(name=name, low=0.0, high=1e7)


class ValidateActuatorSpecsTest:
    @classmethod
    def setup_class(cls):
        cls._provider = _provider()

    @pytest.mark.parametrize("name", ["Ip_A", "v_loop_lcfs_V"])
    def test_boundary_conditions_are_not_actuators(self, name):
        with pytest.raises(ValueError, match="Unknown actuator"):
            validate_actuator_specs([_spec(name)], self._provider)

    def test_source_not_configured_raises(self):
        # ecrh is absent from the default config.
        with pytest.raises(ValueError, match="not configured"):
            validate_actuator_specs([_spec("P_eccd")], self._provider)

    def test_configured_source_passes(self):
        provider = _provider(extra_config={"sources": {"ecrh": {"P_total": 5e6}}})
        validate_actuator_specs([_spec("P_eccd")], provider)  # no raise


class ValidatePhysicsRandomizationTest:
    @classmethod
    def setup_class(cls):
        cls._provider = _provider()

    def test_unknown_path_raises(self):
        with pytest.raises(ValueError, match="not configured"):
            validate_physics_randomization(
                {
                    "numerics.not_a_real_param": PhysicsRandomizationSpec(
                        absolute=(0.1, 0.2)
                    )
                },
                self._provider,
            )

    def test_bad_range_raises(self):
        with pytest.raises(ValueError, match="low < high"):
            validate_physics_randomization(
                {
                    "numerics.resistivity_multiplier": PhysicsRandomizationSpec(
                        absolute=(1.0, 0.5)
                    )
                },
                self._provider,
            )

    def test_scalar_randomization_survives_environment_step(self):
        env = make_test_env(
            physics_randomization={
                "numerics.resistivity_multiplier": PhysicsRandomizationSpec(
                    absolute=(0.5, 1.5)
                )
            }
        )
        state, _ = env.init(jax.random.key(0))
        action = jnp.asarray((env.action_space.low + env.action_space.high) / 2)
        next_state, info = env.step(state, action)
        np_value = next_state.phys_params["numerics.resistivity_multiplier"]
        assert 0.5 <= float(np_value) <= 1.5
        assert not bool(info.terminated)
        assert not bool(info.truncated)

    def test_static_float_path_passes(self):
        validate_physics_randomization(
            {
                "neoclassical.bootstrap_current.bootstrap_multiplier": (
                    PhysicsRandomizationSpec(absolute=(0.5, 1.5))
                )
            },
            self._provider,
        )

    def test_array_randomization_rejected_at_construction(self):
        with pytest.raises(ValueError, match="floating-point scalar"):
            make_test_env(
                physics_randomization={
                    "plasma_composition.Z_eff": PhysicsRandomizationSpec(
                        absolute=(1.0, 2.0)
                    )
                }
            )
