"""Action/observation space descriptors for PlasmaxEnv.

Single source of truth for what observations and actuators exist, how to
extract/normalize them, and how they lay out in the flat obs/action vectors
consumed by the wrapper stack and RL agents.
"""

import dataclasses
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import jax
import numpy as np
from jax import numpy as jnp


@dataclasses.dataclass
class ObsSpec:
    """Normalization spec for a single observation entry.

    Dividing the raw quantity by ``scale`` brings it to O(1), keeping policy
    network inputs at consistent magnitudes regardless of physical units.

    Attributes:
      name: Registry key (must be in PROFILE_REGISTRY or SCALAR_REGISTRY).
      scale: Value to divide by before feeding to the policy.
      bounds: Optional ``(low, high)`` in physical (raw) units. Divided by
        ``scale`` to populate the normalized Envelope ``Continuous`` space.
        Profile bounds are broadcast across all radial points. ``None``
        leaves the entry unbounded (``±inf``).
    """

    name: str
    scale: float
    bounds: tuple[float, float] | None = None


# ---------------------------------------------------------------------------
# Observation registry
# ---------------------------------------------------------------------------
#
# Single source of truth for which named observations exist and how to extract
# them from PlasmaState. Normalisation scale is NOT stored here — it is a
# per-scenario choice supplied separately via ObsSpec (from the YAML's
# observations block, or explicitly by the caller). Add a new obs by adding one
# entry here — scenario_config and the wrappers pick it up automatically.

_ExtractFn = Callable[[Any], jax.Array]


PROFILE_REGISTRY: Mapping[str, _ExtractFn] = {
    "T_e": lambda p: p.T_e,
    "T_i": lambda p: p.T_i,
    "n_e": lambda p: p.n_e,
    "psi": lambda p: p.psi,
    "q": lambda p: p.q,
}

SCALAR_REGISTRY: Mapping[str, _ExtractFn] = {
    "W_thermal": lambda p: p.W_thermal_total,
    "tau_E": lambda p: p.tau_E,
    "P_fusion": lambda p: p.P_fusion,
    "t": lambda p: p.t,
    "Ip": lambda p: p.Ip,
    "q_min": lambda p: p.q_min,
    "q95": lambda p: p.q95,
    "beta_N": lambda p: p.beta_N,
    "f_non_inductive": lambda p: p.f_non_inductive,
}


# ---------------------------------------------------------------------------
# Observation layout
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ObsLayout:
    """Slice map for a flat observation vector.

    Each entry maps a sensor name to the slice it occupies in the obs vector.
    Profile slices have length n_rho (or fewer, after discretization); scalar
    slices have length 1. ``profile_names`` and ``scalar_names`` preserve the
    order the slices appear in.
    """

    profile_slices: Mapping[str, slice]
    scalar_slices: Mapping[str, slice]
    profile_names: tuple[str, ...]
    scalar_names: tuple[str, ...]
    vector_size: int | None = None

    @property
    def size(self) -> int:
        if self.vector_size is not None:
            return self.vector_size
        total = 0
        for s in self.profile_slices.values():
            total += s.stop - s.start
        return total + len(self.scalar_slices)

    def slice_of(self, name: str) -> slice:
        """Returns the slice for a named sensor (profile or scalar)."""
        if name in self.profile_slices:
            return self.profile_slices[name]
        if name in self.scalar_slices:
            return self.scalar_slices[name]
        raise ValueError(
            f"Unknown sensor name {name!r}. Known: "
            f"{list(self.profile_slices)} + {list(self.scalar_slices)}"
        )

    @classmethod
    def from_specs(
        cls,
        profile_specs: Sequence[ObsSpec],
        scalar_specs: Sequence[ObsSpec],
        n_rho: int,
    ) -> "ObsLayout":
        profile_slices, profile_names = {}, []
        off = 0
        for spec in profile_specs:
            profile_slices[spec.name] = slice(off, off + n_rho)
            profile_names.append(spec.name)
            off += n_rho
        scalar_slices, scalar_names = {}, []
        for spec in scalar_specs:
            scalar_slices[spec.name] = slice(off, off + 1)
            scalar_names.append(spec.name)
            off += 1
        return cls(
            profile_slices=profile_slices,
            scalar_slices=scalar_slices,
            profile_names=tuple(profile_names),
            scalar_names=tuple(scalar_names),
        )


def build_obs_bounds(
    obs_size: int,
    layout: ObsLayout,
    profile_specs: Sequence[ObsSpec],
    scalar_specs: Sequence[ObsSpec],
) -> tuple[np.ndarray, np.ndarray]:
    """Builds (low, high) arrays in normalized units, broadcasting profile bounds.

    Falls back to ``±inf`` for any entry without ``bounds``, and silently
    preserves the existing unbounded behaviour when ``layout`` does not match
    ``obs_size`` (e.g. when a custom ``obs_fn`` reshapes the vector).
    """
    low = np.full(obs_size, float("-inf"), dtype=np.float32)
    high = np.full(obs_size, float("inf"), dtype=np.float32)
    if layout.size != obs_size:
        return low, high
    for spec in profile_specs:
        if spec.bounds is None:
            continue
        sl = layout.profile_slices[spec.name]
        lo, hi = spec.bounds
        low[sl] = lo / spec.scale
        high[sl] = hi / spec.scale
    for spec in scalar_specs:
        if spec.bounds is None:
            continue
        sl = layout.scalar_slices[spec.name]
        lo, hi = spec.bounds
        low[sl] = lo / spec.scale
        high[sl] = hi / spec.scale
    return low, high


# ---------------------------------------------------------------------------
# Actuators
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class ActuatorSpec:
    """Specification for a single RL actuator.

    Attributes:
      name: ControlInputs field name (e.g. 'P_nbi').
      low: Physical-unit lower bound.
      high: Physical-unit upper bound.
      max_delta: Maximum change per step. ``inf`` disables rate limiting.
      init: Optional reset-time ``prev_action`` anchor. If ``None``, reset uses
        the scenario source default when available, then falls back to midpoint.
    """

    name: str
    low: float
    high: float
    max_delta: float = float("inf")
    init: float | None = None


# ---------------------------------------------------------------------------
# Obs extraction
# ---------------------------------------------------------------------------


def extract_obs(
    plasma,
    profile_obs_specs: Sequence[ObsSpec],
    scalar_obs_specs: Sequence[ObsSpec],
) -> jax.Array:
    """Returns a flat float32 obs vector built by name-based dispatch.

    Profiles (concatenated) then scalars, each divided by its ``scale``. Names
    must be in PROFILE_REGISTRY / SCALAR_REGISTRY — the YAML order is honoured.
    """
    profiles = (
        jnp.concatenate(
            [PROFILE_REGISTRY[s.name](plasma) / s.scale for s in profile_obs_specs]
        )
        if profile_obs_specs
        else jnp.zeros((0,))
    )
    scalars = (
        jnp.array([SCALAR_REGISTRY[s.name](plasma) / s.scale for s in scalar_obs_specs])
        if scalar_obs_specs
        else jnp.zeros((0,))
    )
    return jnp.concatenate([profiles, scalars]).astype(jnp.float32)
