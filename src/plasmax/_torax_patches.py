"""Runtime patches to TORAX for plasmax's JAX-transformed usage.

Imported for its side effects at the top of ``plasmax/__init__.py`` so the
patches are in place before any geometry/env is built. Each patch is a minimal,
numerically-identical shim around a TORAX rough edge that only manifests under
the kind of multi-trace ``jax.jit``/``lax.scan`` usage RL training does (rejax
PPO, ``collect.py``), not in TORAX's own single-loop simulator.

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
"""

import jax.numpy as jnp
from torax._src.torax_pydantic import interpolated_param_2d as _interpolated_param_2d

_interpolated_param_2d.Grid1D.cell_widths = property(
    lambda self: jnp.diff(self.face_centers)
)
