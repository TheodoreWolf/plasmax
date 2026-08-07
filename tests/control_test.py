"""Tests for torax._src.control."""

import jax
import numpy as np
from helpers import make_default_step_fn
from jax import numpy as jnp

from plasmax.control import (
    ControlInputs,
    _ControlInputsApplier,
    _PhysicsParamsApplier,
)


class ControlInputsTest:
    def test_partial_construction(self):
        ci = ControlInputs(P_nbi=jnp.array(8e6))
        assert ci.P_nbi == jnp.array(8e6)
        assert ci.gas_puff_rate is None

    def test_pytree_leaves_none_gives_empty(self):
        ci = ControlInputs()
        leaves = jax.tree.leaves(ci)
        # generates an empty list because all fields are None
        assert leaves == []

    def test_pytree_leaves_with_values(self):
        ci = ControlInputs(P_nbi=jnp.array(30e6), gas_puff_rate=jnp.array(1e21))
        leaves = jax.tree.leaves(ci)
        assert len(leaves) == 2


class ControlInputsApplierTest:
    """Value application only; name/path validation lives in
    environment_validation (see environment_validation_test.py)."""

    @classmethod
    def setup_class(cls):
        cls._step_fn = make_default_step_fn()
        cls._provider = cls._step_fn.runtime_params_provider

    def test_heating_override_changes_provider_value(self):
        step_fn = make_default_step_fn(
            extra_config={"sources": {"generic_heat": {"P_total": 5e6}}}
        )
        provider = step_fn.runtime_params_provider
        new_power = 12e6
        applier = _ControlInputsApplier(["P_nbi"])
        new_provider = applier(ControlInputs(P_nbi=jnp.array(new_power)), provider)
        result = new_provider.sources.generic_heat.P_total.get_value(t=0.0)
        np.testing.assert_allclose(result, new_power, atol=0.5, rtol=0.0)

    def test_no_override_returns_same_provider(self):
        applier = _ControlInputsApplier(["P_nbi"])
        assert applier(ControlInputs(), self._provider) is self._provider

    def test_ecrh_override_works_when_configured(self):
        step_fn = make_default_step_fn(
            extra_config={"sources": {"ecrh": {"P_total": 5e6}}}
        )
        provider = step_fn.runtime_params_provider
        applier = _ControlInputsApplier(["P_eccd"])
        new_provider = applier(ControlInputs(P_eccd=jnp.array(10e6)), provider)
        result = new_provider.sources.ecrh.P_total.get_value(t=0.0)
        np.testing.assert_allclose(result, 10e6, atol=0.5, rtol=0.0)


class PhysicsParamsApplierTest:
    @classmethod
    def setup_class(cls):
        cls._step_fn = make_default_step_fn()
        cls._provider = cls._step_fn.runtime_params_provider

    def test_call_overrides_scalar_path(self):
        path = "numerics.resistivity_multiplier"
        applier = _PhysicsParamsApplier([path])
        new_provider = applier({path: jnp.array(0.5)}, self._provider)
        result = new_provider.numerics.resistivity_multiplier.get_value(t=0.0)
        np.testing.assert_allclose(result, 0.5, atol=1e-6, rtol=0.0)

    def test_call_overrides_static_float_path(self):
        path = "neoclassical.bootstrap_current.bootstrap_multiplier"
        applier = _PhysicsParamsApplier([path])
        new_provider = applier({path: jnp.array(0.75)}, self._provider)
        np.testing.assert_allclose(
            new_provider.neoclassical.bootstrap_current.bootstrap_multiplier,
            0.75,
            atol=1e-6,
            rtol=0.0,
        )

    def test_empty_paths_returns_same_provider(self):
        applier = _PhysicsParamsApplier([])
        assert applier({}, self._provider) is self._provider
