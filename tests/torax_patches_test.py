"""Regression tests for plasmax's process-wide TORAX workarounds."""

from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from torax._src import state
from torax._src.neoclassical.formulas import formulas, redl
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


def _trapped_fraction(epsilon: jax.Array, delta: jax.Array) -> jax.Array:
    return formulas.calculate_f_trap(
        SimpleNamespace(epsilon_face=epsilon, delta_face=delta)
    )


def _upstream_trapped_fraction(epsilon: jax.Array, delta: jax.Array) -> jax.Array:
    """Reference expression from TORAX, evaluated away from its singular axis."""
    epsilon_effective = 0.67 * (1.0 - 1.4 * jnp.abs(delta) * delta) * epsilon
    aa = (1.0 - epsilon) / (1.0 + epsilon)
    return 1.0 - jnp.sqrt(aa) * (1.0 - epsilon_effective) / (
        1.0 + 2.0 * jnp.sqrt(epsilon_effective)
    )


@pytest.mark.parametrize("stacked", [False, True])
def test_trapped_fraction_preserves_forward_values(stacked: bool) -> None:
    epsilon = jnp.asarray([0.0, 1e-8, 0.08, 0.25, 0.4])
    delta = jnp.asarray([0.0, -0.3, 0.0, 0.2, 0.4])
    if stacked:
        epsilon = jnp.stack([epsilon, epsilon * 0.8])
        delta = jnp.stack([delta, delta * 0.9])

    actual = jax.jit(_trapped_fraction)(epsilon, delta)
    expected = _upstream_trapped_fraction(epsilon, delta)

    np.testing.assert_array_equal(actual[..., 0], 0.0)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)


def test_trapped_fraction_zero_jvp_and_vjp_are_zero() -> None:
    epsilon = jnp.asarray([0.0, 0.08, 0.25, 0.4])
    delta = jnp.asarray([0.0, -0.3, 0.2, 0.4])
    zeros = jnp.zeros_like(epsilon)

    _, tangent = jax.jit(
        lambda eps, tri: jax.jvp(_trapped_fraction, (eps, tri), (zeros, zeros))
    )(epsilon, delta)

    def zero_pullback(eps: jax.Array, tri: jax.Array) -> tuple[jax.Array, ...]:
        _, pullback = jax.vjp(_trapped_fraction, eps, tri)
        return pullback(zeros)

    cotangents = jax.jit(zero_pullback)(epsilon, delta)

    np.testing.assert_array_equal(tangent, zeros)
    for cotangent in cotangents:
        np.testing.assert_array_equal(cotangent, zeros)


def test_trapped_fraction_derivatives_match_off_axis_and_finite_differences() -> None:
    with jax.enable_x64(True):
        epsilon = jnp.asarray([0.0, 0.08, 0.25, 0.4])
        delta = jnp.asarray([0.0, -0.3, 0.2, 0.4])
        # Only the axis entries vary in the first direction. The second stays
        # on the physical geometry constraint epsilon[0] == 0.
        eps_directions = jnp.asarray([[1.0, 0.0, 0.0, 0.0], [0.0, 0.1, -0.2, 0.3]])
        tri_directions = jnp.asarray([[1.0, 0.0, 0.0, 0.0], [0.0, -0.2, 0.1, 0.2]])
        cotangent = jnp.asarray([1.0, -0.3, 0.5, 0.8])

        def pushforward(eps_dot: jax.Array, tri_dot: jax.Array) -> jax.Array:
            return jax.jvp(_trapped_fraction, (epsilon, delta), (eps_dot, tri_dot))[1]

        tangents = jax.jit(jax.vmap(pushforward))(eps_directions, tri_directions)
        expected_tangent = jax.jvp(
            _upstream_trapped_fraction,
            (epsilon[1:], delta[1:]),
            (eps_directions[1, 1:], tri_directions[1, 1:]),
        )[1]

        def pullback(eps: jax.Array, tri: jax.Array) -> tuple[jax.Array, ...]:
            return jax.vjp(_trapped_fraction, eps, tri)[1](cotangent)

        gradients = jax.jit(pullback)(epsilon, delta)
        expected_gradients = jax.vjp(
            _upstream_trapped_fraction, epsilon[1:], delta[1:]
        )[1](cotangent[1:])

        np.testing.assert_array_equal(tangents[0], jnp.zeros_like(epsilon))
        np.testing.assert_array_equal(tangents[:, 0], 0.0)
        np.testing.assert_allclose(
            tangents[1, 1:], expected_tangent, rtol=1e-12, atol=1e-12
        )
        for gradient, expected in zip(gradients, expected_gradients, strict=True):
            np.testing.assert_array_equal(gradient[0], 0.0)
            np.testing.assert_allclose(gradient[1:], expected, rtol=1e-12, atol=1e-12)

        for step in (1e-3, 1e-4, 1e-5):
            plus = _trapped_fraction(
                epsilon + step * eps_directions[1], delta + step * tri_directions[1]
            )
            minus = _trapped_fraction(
                epsilon - step * eps_directions[1], delta - step * tri_directions[1]
            )
            finite_difference = (plus - minus) / (2.0 * step)
            np.testing.assert_allclose(
                tangents[1], finite_difference, rtol=1e-6, atol=1e-9
            )
            reverse_projection = sum(
                jnp.vdot(gradient, direction[1])
                for gradient, direction in zip(
                    gradients, (eps_directions, tri_directions), strict=True
                )
            )
            np.testing.assert_allclose(
                reverse_projection,
                jnp.vdot(cotangent, finite_difference),
                rtol=1e-6,
                atol=1e-9,
            )


def _upstream_redl_alpha(
    trapped: jax.Array, collisionality: jax.Array, charge: jax.Array
) -> jax.Array:
    alpha_zero = -((0.62 + 0.055 * (charge - 1)) / (0.53 + 0.17 * (charge - 1))) * (
        (1 - trapped)
        / (1 - (0.31 - 0.065 * (charge - 1)) * trapped - 0.25 * trapped**2)
    )
    return (
        (alpha_zero + 0.7 * charge * trapped**0.5 * jnp.sqrt(collisionality))
        / (1 + 0.18 * jnp.sqrt(collisionality))
        - 0.002 * collisionality**2 * trapped**6
    ) / (1 + 0.004 * collisionality**2 * trapped**6)


@pytest.mark.parametrize("stacked", [False, True])
def test_redl_alpha_preserves_forward_values_and_nonzero_axis(stacked: bool) -> None:
    inputs = (
        jnp.asarray([0.0, 1e-8, 0.2, 0.4]),
        jnp.asarray([5.0, 0.1, 1.0, 3.0]),
        jnp.asarray([1.5, 1.2, 1.8, 2.0]),
    )
    if stacked:
        inputs = tuple(jnp.stack([value, value * 1.1]) for value in inputs)
    actual = jax.jit(redl.calculate_alpha)(*inputs)

    np.testing.assert_allclose(
        actual, _upstream_redl_alpha(*inputs), rtol=1e-6, atol=1e-7
    )
    assert np.all(np.asarray(actual[..., 0]) < -0.5)


def test_redl_alpha_zero_jvp_and_vjp_are_zero() -> None:
    inputs = (
        jnp.asarray([0.0, 0.1, 0.3]),
        jnp.asarray([5.0, 0.1, 1.0]),
        jnp.asarray([1.5, 1.2, 1.8]),
    )
    zeros = tuple(jnp.zeros_like(value) for value in inputs)
    _, tangent = jax.jit(lambda *x: jax.jvp(redl.calculate_alpha, x, zeros))(*inputs)
    cotangents = jax.jit(lambda *x: jax.vjp(redl.calculate_alpha, *x)[1](zeros[0]))(
        *inputs
    )

    np.testing.assert_array_equal(tangent, zeros[0])
    for cotangent in cotangents:
        np.testing.assert_array_equal(cotangent, zeros[0])


def test_redl_alpha_retains_axis_collisionality_and_charge_derivatives() -> None:
    with jax.enable_x64(True):
        inputs = (
            jnp.asarray([0.0, 0.1, 0.2, 0.4]),
            jnp.asarray([5.0, 0.1, 1.0, 3.0]),
            jnp.asarray([1.5, 1.2, 1.8, 2.0]),
        )
        directions = (
            jnp.asarray([[0.0, 0.1, -0.2, 0.3], [0.0] * 4, [0.0] * 4]),
            jnp.asarray([[0.2, -0.1, 0.3, -0.2], [1.0, 0, 0, 0], [0.0] * 4]),
            jnp.asarray([[0.1, 0.1, -0.2, 0.3], [0.0] * 4, [1.0, 0, 0, 0]]),
        )
        cotangent = jnp.asarray([1.0, -0.3, 0.5, 0.8])
        tangents = jax.jit(
            jax.vmap(lambda *d: jax.jvp(redl.calculate_alpha, inputs, d)[1])
        )(*directions)
        gradients = jax.jit(lambda *x: jax.vjp(redl.calculate_alpha, *x)[1](cotangent))(
            *inputs
        )
        expected_gradients = jax.vjp(
            _upstream_redl_alpha, *(value[1:] for value in inputs)
        )[1](cotangent[1:])
        for actual, expected in zip(gradients, expected_gradients, strict=True):
            assert np.isfinite(np.asarray(actual)).all()
            np.testing.assert_allclose(actual[1:], expected, rtol=1e-12, atol=1e-12)
        # The axis alpha value depends on nu_i_star and Z_eff, even though its
        # sqrt(f_trap) contribution is exactly zero.
        assert np.all(np.abs(np.asarray(tangents[1:, 0])) > 1e-3)
        for index in range(3):
            direction = tuple(value[index] for value in directions)
            reverse_projection = sum(
                jnp.vdot(gradient, delta)
                for gradient, delta in zip(gradients, direction, strict=True)
            )
            for step in (1e-3, 1e-4, 1e-5):
                plus = redl.calculate_alpha(
                    *(
                        value + step * delta
                        for value, delta in zip(inputs, direction, strict=True)
                    )
                )
                minus = redl.calculate_alpha(
                    *(
                        value - step * delta
                        for value, delta in zip(inputs, direction, strict=True)
                    )
                )
                difference = (plus - minus) / (2 * step)
                np.testing.assert_allclose(
                    tangents[index], difference, rtol=1e-6, atol=1e-9
                )
                np.testing.assert_allclose(
                    reverse_projection,
                    jnp.vdot(cotangent, difference),
                    rtol=1e-6,
                    atol=1e-9,
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
