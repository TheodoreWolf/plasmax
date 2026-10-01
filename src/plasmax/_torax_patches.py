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

``calculate_f_trap`` — TORAX evaluates ``sqrt(epsilon_effective)`` at the
magnetic axis, where its derivative is singular even for a zero tangent.
Using a safe square-root operand at face zero and explicitly returning the
axis value of zero preserves the formula away from the axis while keeping
the constrained axis derivative finite.

``redl.calculate_alpha`` — Redl's bootstrap coefficient takes another square
root of the trapped fraction, which is also zero at the axis. Guarding that
operand preserves the coefficient's dependence on collisionality and charge
while avoiding the singular trapped-fraction derivative.

``calc_heating_and_current`` — the ECCD source evaluates ``log(power_density)``
inside an exponential. At zero power this gives finite zero current but NaN
derivatives. Factoring power and current-drive efficiency outside the logarithm
preserves their linear response, including the nonzero power derivative at zero.
"""

import dataclasses

import jax
import jax.numpy as jnp
from torax._src import state as _state
from torax._src.config import runtime_params as _runtime_params
from torax._src.geometry import geometry as _geometry
from torax._src.neoclassical.conductivity import base as _conductivity_base
from torax._src.neoclassical.formulas import formulas as _neo_formulas
from torax._src.neoclassical.formulas import redl as _redl_formulas
from torax._src.pedestal_model import pedestal_model_output as _pedestal_output
from torax._src.pedestal_model import runtime_params as _pedestal_runtime_params
from torax._src.sources import electron_cyclotron_source as _ec_source
from torax._src.sources import source_profiles as _source_profiles
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
_original_calc_heating_and_current = _ec_source.calc_heating_and_current


def _calc_heating_and_current_zero_safe(
    runtime_params: _runtime_params.RuntimeParams,
    geo: _geometry.Geometry,
    source_name: str,
    core_profiles: _state.CoreProfiles,
    unused_calculated_source_profiles: _source_profiles.SourceProfiles | None,
    unused_conductivity: _conductivity_base.Conductivity | None,
) -> tuple[jax.Array, jax.Array]:
    """Keep TORAX's EC source linear in power and efficiency at zero."""
    del unused_calculated_source_profiles, unused_conductivity
    source_params = runtime_params.sources[source_name]
    assert isinstance(source_params, _ec_source.RuntimeParams)
    ec_power_density = (
        source_params.extra_prescribed_power_density
        + _ec_source.formulas.gaussian_profile(
            center=source_params.gaussian_location,
            width=source_params.gaussian_width,
            total=source_params.P_total,
            geo=geo,
        )
    )
    constants = _ec_source.constants.CONSTANTS
    # Merely masking exp(log(power)) at zero would discard its finite,
    # nonzero power derivative. Factor out both potentially zero multipliers.
    current_per_power_and_efficiency = jnp.exp(
        jnp.log(16.0)
        + jnp.log(jnp.pi)
        + 2 * jnp.log(constants.epsilon_0)
        + jnp.log(core_profiles.T_e.value * 1e3)
        - (
            2 * jnp.log(constants.q_e)
            + jnp.log(
                _ec_source.collisions.calculate_log_lambda_ee(
                    core_profiles.T_e.value, core_profiles.n_e.value
                )
            )
            + jnp.log(core_profiles.n_e.value)
        )
    )
    j_tor_ec = (
        ec_power_density
        * source_params.current_drive_efficiency
        * current_per_power_and_efficiency
    )
    q_cell = _geometry.face_to_cell(core_profiles.q_face)
    fsa_j_dot_B = (
        geo.F
        * geo.gm9
        * (1 + geo.g2 * geo.g3 / (16 * jnp.pi**4 * q_cell**2))
        * j_tor_ec
    )
    return ec_power_density, fsa_j_dot_B


def _calculate_f_trap_axis_safe(geo: _geometry.Geometry) -> jax.Array:
    """Evaluate TORAX's trapped fraction with a fixed magnetic-axis value."""
    epsilon_effective = (
        0.67 * (1.0 - 1.4 * jnp.abs(geo.delta_face) * geo.delta_face) * geo.epsilon_face
    )
    aa = (1.0 - geo.epsilon_face) / (1.0 + geo.epsilon_face)

    # Face zero is the magnetic axis. Guard the operand before taking sqrt:
    # masking only the output would leave a singular derivative in the trace.
    is_axis = jnp.arange(epsilon_effective.shape[-1]) == 0
    sqrt_epsilon = jnp.sqrt(jnp.where(is_axis, 1.0, epsilon_effective))
    f_trap = 1.0 - jnp.sqrt(aa) * (1.0 - epsilon_effective) / (1.0 + 2.0 * sqrt_epsilon)
    return jnp.where(is_axis, 0.0, f_trap)


def _calculate_redl_alpha_axis_safe(
    f_trap: jax.Array, nu_i_star: jax.Array, Z_eff: jax.Array
) -> jax.Array:
    """Evaluate Redl's coefficient without differentiating sqrt(0) on axis."""
    alpha_0 = -((0.62 + 0.055 * (Z_eff - 1)) / (0.53 + 0.17 * (Z_eff - 1))) * (
        (1 - f_trap) / (1 - (0.31 - 0.065 * (Z_eff - 1)) * f_trap - 0.25 * f_trap**2)
    )
    # Only the square-root term has a fixed axis value; alpha itself still
    # depends on collisionality and effective charge at the magnetic axis.
    is_axis = jnp.arange(f_trap.shape[-1]) == 0
    sqrt_f_trap = jnp.where(is_axis, 0.0, jnp.sqrt(jnp.where(is_axis, 1.0, f_trap)))
    return (
        (alpha_0 + 0.7 * Z_eff * sqrt_f_trap * jnp.sqrt(nu_i_star))
        / (1 + 0.18 * jnp.sqrt(nu_i_star))
        - 0.002 * nu_i_star**2 * f_trap**6
    ) / (1 + 0.004 * nu_i_star**2 * f_trap**6)


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
# TODO: Remove this assignment once TORAX guards the trapped-fraction square
# root at the magnetic axis before differentiating it.
_neo_formulas.calculate_f_trap = _calculate_f_trap_axis_safe
# TODO: Remove this assignment once TORAX guards Redl's sqrt(f_trap) on axis.
_redl_formulas.calculate_alpha = _calculate_redl_alpha_axis_safe
# TODO: Remove once TORAX factors zero-capable EC power and efficiency out of
# its logarithmic current calculation. Config-built sources resolve this name.
_ec_source.calc_heating_and_current = _calc_heating_and_current_zero_safe
# Direct source construction also captures the model function as a dataclass
# default; keep that path aligned with ElectronCyclotronSourceConfig.build_source.
_ec_source.ElectronCyclotronSource.model_func = _calc_heating_and_current_zero_safe
_ec_source.ElectronCyclotronSource.__dataclass_fields__[
    "model_func"
].default = _calc_heating_and_current_zero_safe
_ec_source.ElectronCyclotronSource.__init__.__kwdefaults__["model_func"] = (
    _calc_heating_and_current_zero_safe
)
