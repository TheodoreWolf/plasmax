"""Runtime patches to TORAX for plasmax's JAX-transformed usage.

Imported for its side effects at the top of ``plasmax/__init__.py`` so the
patches are in place before any geometry/env is built. Each patch is a minimal
shim around a TORAX behavior that conflicts with plasmax's runtime semantics.

Currently patched:

``Grid1D.cell_widths`` — TORAX declares this as a ``functools.cached_property``
returning ``jnp.diff(self.face_centers)``. ``torax_mesh`` is a traced pytree
argument to the step/reset functions, so ``face_centers`` is a tracer and the
diff is (correctly) a tracer; but ``cached_property`` then *stores* that
per-trace tracer on the mesh object. Across rejax's separate jit traces (the
training ``lax.scan`` vs. the eval-callback rollout) the stale cached tracer
escapes its original trace, raising ``UnexpectedTracerError`` from ``reset``
(via ``Geometry.drho_norm`` → ``calculate_stored_thermal_energy``). This trips
PPO on every EQDSK/CHEASE (``StandardGeometry``) env. Replacing it with a plain
``property`` recomputes the (26-element) diff each trace — trivially cheap — and,
being a data descriptor, also overrides any stale value left in ``__dict__``.
The numerical result is unchanged.

``PedestalModelOutput.modify_core_transport`` — TORAX applies adaptive-pedestal
transport multipliers by matching coefficient names. That also scales the
Pereverzev coefficients, even though they are numerical stabilizers rather than
turbulent transport. Restoring those fields after the upstream modification
keeps the stabilizer independent of the physical pedestal suppression while
leaving every other transport coefficient under TORAX's normal handling.
"""

import dataclasses

import jax.numpy as jnp
from torax._src import state as _state
from torax._src.geometry import geometry as _geometry
from torax._src.pedestal_model import pedestal_model_output as _pedestal_output
from torax._src.pedestal_model import runtime_params as _pedestal_runtime_params
from torax._src.torax_pydantic import interpolated_param_2d as _interpolated_param_2d

_PEREVERZEV_FIELDS = (
    "chi_face_ion_pereverzev",
    "chi_face_el_pereverzev",
    "full_v_heat_face_ion_pereverzev",
    "full_v_heat_face_el_pereverzev",
    "d_face_el_pereverzev",
    "v_face_el_pereverzev",
)

_original_modify_core_transport = (
    _pedestal_output.PedestalModelOutput.modify_core_transport
)


def _modify_core_transport_preserving_pereverzev(
    self: _pedestal_output.PedestalModelOutput,
    core_transport: _state.CoreTransport,
    geo: _geometry.Geometry,
    pedestal_runtime_params: _pedestal_runtime_params.RuntimeParams,
) -> _state.CoreTransport:
    """Apply pedestal transport changes without modifying the stabilizer."""
    modified = _original_modify_core_transport(
        self,
        core_transport,
        geo,
        pedestal_runtime_params,
    )
    return dataclasses.replace(
        modified,
        **{field: getattr(core_transport, field) for field in _PEREVERZEV_FIELDS},
    )


_interpolated_param_2d.Grid1D.cell_widths = property(
    lambda self: jnp.diff(self.face_centers)
)
# TODO: Remove this assignment after TORAX stops applying physical pedestal
# multipliers to the numerical Pereverzev stabilizer.
_pedestal_output.PedestalModelOutput.modify_core_transport = (
    _modify_core_transport_preserving_pereverzev
)
