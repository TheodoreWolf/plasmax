"""Control API for TORAX: named actuator interface over SimulationStepFn."""

import dataclasses
from collections.abc import Iterable, Sequence

import jax
from jax import numpy as jnp
from torax._src import jax_utils
from torax._src.config import build_runtime_params
from torax._src.torax_pydantic import interpolated_param_1d

# Mapping from ControlInputs field name to RuntimeParamsProvider dot-path
# (one canonical name per path).
_FIELD_TO_PATH: dict[str, str] = {
    "P_eccd": "sources.ecrh.P_total",
    "rho_eccd": "sources.ecrh.gaussian_location",
    "P_nbi": "sources.generic_heat.P_total",
    "generic_current_A": "sources.generic_current.I_generic",
    "gas_puff_rate": "sources.gas_puff.S_total",
    "pellet_rate": "sources.pellet.S_total",
}


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class ControlInputs:
    """Typed interface for physical actuators in a TORAX simulation.

    All fields default to None (= 'do not override this actuator').
    Setting a field to a JAX scalar array will override the corresponding
    RuntimeParamsProvider entry for that step.

    Attributes:
      P_eccd: Total ECRH/ECCD power [W]. Requires ecrh source configured.
      rho_eccd: EC deposition location [rho_norm]. Requires ecrh source
        configured.
      P_nbi: Generic heat source power [W] (NBI/ICRF). Requires generic_heat
        source configured.
      generic_current_A: Total generic current [A]. Requires generic_current
        source configured with use_absolute_current=True.
      gas_puff_rate: Gas puff particle source rate [particles/s]. Requires
        gas_puff source configured.
      pellet_rate: Total pellet particle source rate [particles/s]. Requires
        pellet source configured.
    """

    P_eccd: jax.Array | None = None
    rho_eccd: jax.Array | None = None
    P_nbi: jax.Array | None = None
    generic_current_A: jax.Array | None = None
    gas_puff_rate: jax.Array | None = None
    pellet_rate: jax.Array | None = None


def _scalar_to_time_varying_update(
    value: jax.Array,
) -> interpolated_param_1d.TimeVaryingScalarUpdate:
    """Wrap a scalar as a zero-order-held runtime-parameter update."""
    dtype = jax_utils.get_dtype()
    return interpolated_param_1d.TimeVaryingScalarUpdate(
        time=jnp.zeros(1, dtype=dtype),
        value=jnp.reshape(jnp.asarray(value, dtype=dtype), (1,)),
    )


def _apply_scalar_overrides(
    provider: build_runtime_params.RuntimeParamsProvider,
    items: Iterable[tuple[str, jax.Array]],
) -> build_runtime_params.RuntimeParamsProvider:
    """Builds correctly typed scalar replacements and applies them.

    The single override-building path shared by every applier below: build the
    path→update mapping, then return the provider unchanged when there is
    nothing to override. Does no validation — paths are assumed valid (checked
    once at env construction by :mod:`plasmax.environment.validation`).
    """
    overrides = {}
    for path, value in items:
        node = provider.get_node_from_path(path)
        if isinstance(node, interpolated_param_1d.TimeVaryingScalar):
            overrides[path] = _scalar_to_time_varying_update(value)
        else:
            overrides[path] = jnp.asarray(value, dtype=jnp.asarray(node).dtype)
    if not overrides:
        return provider
    return provider.update_provider_from_mapping(overrides)


class _ControlInputsApplier:
    """Applies named actuator values to a ``RuntimeParamsProvider``.

    Provider-independent: built once from the actuator field names, then called
    with any provider sharing that actuator set. Does **no** validation —
    actuator names and source availability are checked once at env construction
    by :func:`plasmax.environment.validation.validate_actuator_specs`.
    """

    def __init__(self, field_names: Sequence[str]):
        """Resolves the ``field_name -> provider dot-path`` map for the actuators."""
        self._field_to_path = {name: _FIELD_TO_PATH[name] for name in field_names}

    def __call__(
        self,
        control_inputs: ControlInputs,
        provider: build_runtime_params.RuntimeParamsProvider,
    ) -> build_runtime_params.RuntimeParamsProvider:
        """Applies the non-None ``control_inputs`` fields to ``provider``."""
        items = (
            (path, getattr(control_inputs, name))
            for name, path in self._field_to_path.items()
            if getattr(control_inputs, name) is not None
        )
        return _apply_scalar_overrides(provider, items)


class _PhysicsParamsApplier:
    """Overrides transition-specific physics params by provider dot-path.

    Unlike actuators (named and rate-limited), these are sampled independently
    for each transition. Provider-independent like
    :class:`_ControlInputsApplier`; paths are validated once at env construction by
    :func:`plasmax.environment.validation.validate_physics_randomization`.
    """

    def __init__(self, paths: Sequence[str]):
        self._paths = tuple(paths)

    def __call__(
        self,
        phys_params: dict[str, jax.Array],
        provider: build_runtime_params.RuntimeParamsProvider,
    ) -> build_runtime_params.RuntimeParamsProvider:
        return _apply_scalar_overrides(
            provider, ((path, phys_params[path]) for path in self._paths)
        )


__all__ = ["ControlInputs"]
