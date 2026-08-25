"""Regression tests for plasmax's process-wide TORAX workarounds."""

from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from torax._src import state
from torax._src.pedestal_model import pedestal_model_output

import plasmax  # noqa: F401  # Applies the process-wide TORAX patches.

_PEREVERZEV_FIELDS = (
    "chi_face_ion_pereverzev",
    "chi_face_el_pereverzev",
    "full_v_heat_face_ion_pereverzev",
    "full_v_heat_face_el_pereverzev",
    "d_face_el_pereverzev",
    "v_face_el_pereverzev",
)


def test_adaptive_pedestal_does_not_scale_pereverzev_coefficients() -> None:
    geometry = SimpleNamespace(
        rho_face_norm=jnp.asarray([0.0, 0.5, 1.0]),
    )
    runtime_params = SimpleNamespace(
        pedestal_top_smoothing_width=jnp.asarray(0.0),
        chi_max=jnp.asarray(100.0),
        D_e_max=jnp.asarray(100.0),
        V_e_min=jnp.asarray(-100.0),
        V_e_max=jnp.asarray(100.0),
    )
    output = pedestal_model_output.PedestalModelOutput(
        rho_norm_ped_top=jnp.asarray(0.5),
        T_i_ped=jnp.asarray(1.0),
        T_e_ped=jnp.asarray(1.0),
        n_e_ped=jnp.asarray(1.0),
        transport_multipliers=pedestal_model_output.TransportMultipliers(
            chi_e_multiplier=jnp.asarray(0.25),
            chi_i_multiplier=jnp.asarray(0.5),
            D_e_multiplier=jnp.asarray(0.75),
            v_e_multiplier=jnp.asarray(0.5),
        ),
    )
    pereverzev = {
        field: jnp.asarray([index, index + 1, index + 2], dtype=jnp.float32)
        for index, field in enumerate(_PEREVERZEV_FIELDS, start=1)
    }
    transport = state.CoreTransport(
        chi_face_ion=jnp.asarray([2.0, 4.0, 8.0]),
        chi_face_el=jnp.asarray([3.0, 6.0, 9.0]),
        d_face_el=jnp.asarray([4.0, 8.0, 12.0]),
        v_face_el=jnp.asarray([-4.0, -8.0, -12.0]),
        chi_face_ion_bohm=jnp.asarray([2.0, 4.0, 8.0]),
        chi_face_el_bohm=jnp.asarray([3.0, 6.0, 9.0]),
        d_face_el_itg=jnp.asarray([4.0, 8.0, 12.0]),
        v_face_el_tem=jnp.asarray([-4.0, -8.0, -12.0]),
        chi_neo_i=jnp.asarray([5.0, 10.0, 15.0]),
        **pereverzev,
    )

    modified = jax.jit(
        lambda coefficients: output.modify_core_transport(
            coefficients,
            geometry,
            runtime_params,
        )
    )(transport)

    expected_modified = {
        "chi_face_ion": [2.0, 4.0, 4.0],
        "chi_face_el": [3.0, 6.0, 2.25],
        "d_face_el": [4.0, 8.0, 9.0],
        "v_face_el": [-4.0, -8.0, -6.0],
        "chi_face_ion_bohm": [2.0, 4.0, 4.0],
        "chi_face_el_bohm": [3.0, 6.0, 2.25],
        "d_face_el_itg": [4.0, 8.0, 9.0],
        "v_face_el_tem": [-4.0, -8.0, -6.0],
        "chi_neo_i": [5.0, 10.0, 15.0],
    }
    for field, expected in expected_modified.items():
        np.testing.assert_array_equal(getattr(modified, field), expected)
    for field, expected in pereverzev.items():
        np.testing.assert_array_equal(getattr(modified, field), expected)
