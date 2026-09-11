"""STEP-specific regressions.

Contract-level smoke (construct, reset, step, time advance, finite obs) lives
in :mod:`tests.env_smoke_test`. This file owns the immutable OpenSTEP asset,
reset physics, source projections, upstream-locked transport, actuator, and reward
regressions for the STEP scenario.

Initial profiles and composition are the rounded OpenSTEP values saved in YAML.
Equilibrium is loaded from the packaged OpenSTEP IMAS data file (~1.3 MB).
"""

import hashlib
from pathlib import Path

import h5py
import jax
import jax.numpy as jnp
import numpy as np
from torax._src.config import build_runtime_params

from plasmax.environment.config import parse_env_and_backend
from plasmax.environment.factory import make
from plasmax.environment.initialization_data import round_significant
from plasmax.environment.merge import load_backend
from plasmax.wrappers import OracleWrappers

_ENV = "step/spp_001_ec_hd/flattop"
_BACKEND = "bohm_gyrobohm_step"
_OPENSTEP_PATH = (
    Path(__file__).parents[1]
    / "src"
    / "plasmax"
    / "configs"
    / "data"
    / "STEP_SPP_001_ECHD_ftop.nc"
)
_OPENSTEP_SHA256 = "64fe9e0d7f634be8ce4b748c08090ecdca2a4a5212473fb5a2c33cb2b6fd4bad"


def _decode(value):
    return value.decode() if isinstance(value, bytes) else value


def test_openstep_asset_identity():
    digest = hashlib.sha256(_OPENSTEP_PATH.read_bytes()).hexdigest()
    assert digest == _OPENSTEP_SHA256


class StepEnvOracleTest:
    """Checks that depend on STEP's specific actuator set and profile shape."""

    @classmethod
    def setup_class(cls):
        cls._env = OracleWrappers(make(_ENV, _BACKEND))
        cls._base_env = cls._env.unwrapped
        cls._reset_state, _ = cls._base_env.init(jax.random.key(0))

    def test_action_space_has_three_actuators(self):
        assert self._base_env.action_space.shape == (3,)

    def test_action_bounds(self):
        space = self._base_env.action_space
        # P_eccd: 50-300 MW, rho_eccd: 0.3-0.7, pellet_rate: 1e20-5e22.
        assert space.low[0] < 1e8
        assert space.high[0] > 2e8
        np.testing.assert_allclose(space.low[1], 0.3, atol=1e-5, rtol=0.0)
        np.testing.assert_allclose(space.high[1], 0.7, atol=1e-5, rtol=0.0)
        assert space.low[2] > 0
        assert space.high[2] > space.low[2]

    def test_obs_shape_has_25_rho_cells(self):
        # STEP runs on 25 radial cells; obs layout is 5 profiles + 9 scalars.
        obs_size = self._base_env.observation_space.shape[0]
        n_rho = (obs_size - 9) // 5
        assert n_rho == 25
        assert obs_size == 5 * 25 + 9

    def test_reset_profiles_and_composition_match_openstep(self):
        sim = self._reset_state.plasma.sim
        core_profiles = sim.core_profiles
        rho = np.asarray(sim.geometry.torax_mesh.cell_centers)

        with h5py.File(_OPENSTEP_PATH) as data:
            source_rho = data["core_profiles/0/profiles_1d.grid.rho_tor_norm"][0]
            targets = {
                "T_e": (
                    data["core_profiles/0/profiles_1d.electrons.temperature"][0]
                    * 1.0e-3
                ),
                "T_i": (data["core_profiles/0/profiles_1d.t_i_average"][0] * 1.0e-3),
                "n_e": data["core_profiles/0/profiles_1d.electrons.density"][0],
                "psi": data["core_profiles/0/profiles_1d.grid.psi"][0],
                "Z_eff": data["core_profiles/0/profiles_1d.zeff"][0],
            }

        actual = {
            "T_e": core_profiles.T_e.value,
            "T_i": core_profiles.T_i.value,
            "n_e": core_profiles.n_e.value,
            "psi": core_profiles.psi.value,
            "Z_eff": core_profiles.Z_eff,
        }
        for name, source_profile in targets.items():
            expected = round_significant(np.interp(rho, source_rho, source_profile))
            np.testing.assert_allclose(
                actual[name],
                expected,
                rtol=1.0e-12,
                atol=1.0e-12,
                err_msg=f"{name} reset differs from the rounded OpenSTEP slice",
            )

        np.testing.assert_allclose(
            core_profiles.Z_eff_face[-1],
            round_significant(targets["Z_eff"][-1]),
            rtol=1.0e-12,
            atol=1.0e-12,
        )
        for species, fraction in core_profiles.impurity_fractions.items():
            assert np.all(np.asarray(fraction) >= 0.0), (
                f"{species} has a negative impurity fraction"
            )

    def test_reset_equilibrium_and_global_quantities(self):
        sim = self._reset_state.plasma.sim
        geometry = sim.geometry
        post = self._reset_state.plasma.post

        with h5py.File(_OPENSTEP_PATH) as data:
            equilibrium = data["equilibrium/0"]
            ip = equilibrium["time_slice.global_quantities.ip"][0]
            volume = equilibrium["time_slice.global_quantities.volume"][0]
            area = equilibrium["time_slice.global_quantities.area"][0]
            elongation = equilibrium["time_slice.boundary.elongation"][0]
            minor_radius = equilibrium["time_slice.boundary.minor_radius"][0]
            li3 = equilibrium["time_slice.global_quantities.li_3"][0]
            beta_n = equilibrium["time_slice.global_quantities.beta_tor_norm"][0]
            r0 = equilibrium["vacuum_toroidal_field.r0"][()]
            b0 = equilibrium["vacuum_toroidal_field.b0"][0]

            profile_pressure = data["core_profiles/0/profiles_1d.pressure_thermal"][0]
            profile_volume = data["core_profiles/0/profiles_1d.grid.volume"][0]
            w_thermal = 1.5 * np.trapezoid(profile_pressure, profile_volume)

        np.testing.assert_allclose(
            sim.core_profiles.Ip_profile_face[-1],
            round_significant(ip),
            rtol=1.0e-12,
            atol=1.0e-6,
        )
        np.testing.assert_allclose(
            geometry.volume_face[-1], volume, rtol=5.0e-5, atol=0.0
        )
        np.testing.assert_allclose(geometry.area_face[-1], area, rtol=5.0e-5, atol=0.0)
        np.testing.assert_allclose(
            geometry.elongation_face[-1], elongation, rtol=1.0e-12, atol=0.0
        )
        np.testing.assert_allclose(
            geometry.a_minor, minor_radius, rtol=1.0e-12, atol=0.0
        )
        np.testing.assert_allclose(
            geometry.R_major * geometry.B_0, r0 * b0, rtol=1.0e-12, atol=0.0
        )
        # Differentiating four-figure psi magnifies quantization near the axis:
        # q_min is 2.45984, versus 2.50155 in the full-precision source.
        np.testing.assert_allclose(post.q_min, 2.459840336, rtol=1e-7, atol=0.0)
        np.testing.assert_allclose(post.li3, li3, rtol=5.0e-3, atol=0.0)
        # OpenSTEP beta_N comes from the equilibrium pressure, whereas TORAX
        # reconstructs thermal pressure from its reduced composition model.
        # Keep the known representation gap explicit instead of silently
        # changing the imported T/n profiles to force this scalar to match.
        w_thermal_gap = 1.0 - float(post.W_thermal_total) / w_thermal
        beta_n_gap = 1.0 - float(post.beta_N) / beta_n
        np.testing.assert_allclose(w_thermal_gap, 0.0393027, rtol=0.0, atol=1.0e-5)
        np.testing.assert_allclose(beta_n_gap, 0.0988306, rtol=0.0, atol=1.0e-5)

    def test_imported_composition_preserves_cell_and_face_samples(self):
        document = parse_env_and_backend(_ENV, _BACKEND)._initial_state
        composition = document.composition
        assert composition is not None
        rho = np.asarray(composition.rho_norm)
        grid = np.sort(
            np.concatenate([document.grid.rho_norm, document.grid.rho_face_norm])
        )
        np.testing.assert_array_equal(rho, grid)
        with h5py.File(_OPENSTEP_PATH) as data:
            source = data["core_profiles/0"]
            source_rho = source["profiles_1d.grid.rho_tor_norm"][0]
            electron_density = source["profiles_1d.electrons.density"][0]
            names = [_decode(name) for name in source["profiles_1d.ion.name"][0]]
            for name in ("Xe", "He"):
                density = source["profiles_1d.ion.density"][0, names.index(name)]
                expected = round_significant(
                    np.interp(rho, source_rho, density / electron_density)
                )
                np.testing.assert_array_equal(
                    composition.impurity_species[name], expected
                )
            expected_zeff = round_significant(
                np.interp(
                    document.grid.rho_face_norm,
                    source_rho,
                    source["profiles_1d.zeff"][0],
                )
            )
        step_fn = self._base_env._dynamics._step_fn
        params, _ = build_runtime_params.get_consistent_runtime_params_and_geometry(
            t=step_fn.runtime_params_provider.numerics.t_initial,
            runtime_params_provider=step_fn.runtime_params_provider,
            geometry_provider=step_fn.geometry_provider,
            is_initialization=True,
        )
        # TORAX reconstructs core Z_eff_face from densities and charge states.
        # The imported cell/face composition samples are its runtime inputs.
        np.testing.assert_allclose(
            params.plasma_composition.Z_eff_face,
            expected_zeff,
            rtol=1e-12,
            atol=1e-12,
        )
        assert composition.impurity_species["Ar"] is None

    def test_integrated_sources_and_current_balance(self):
        post = self._reset_state.plasma.post

        with h5py.File(_OPENSTEP_PATH) as data:
            names = [
                _decode(name)
                for name in data["core_sources/0/source.identifier.name"][:]
            ]

            def source_index(name):
                return names.index(name)

            ec_index = source_index("ec")
            pellet_index = source_index("pellet")
            fusion_index = source_index("fusion")
            radiation_index = source_index("radiation")
            total_index = source_index("total")
            ec_power = data["core_sources/0/source.global_quantities.electrons.power"][
                ec_index, 0
            ]
            pellet_rate = data[
                "core_sources/0/source.global_quantities.total_ion_particles"
            ][pellet_index, 0]
            fusion_power = data["core_sources/0/source.global_quantities.power"][
                fusion_index, 0
            ]
            radiation_power = data[
                "core_sources/0/source.global_quantities.electrons.power"
            ][radiation_index, 0]
            net_power = data["core_sources/0/source.global_quantities.power"][
                total_index, 0
            ]

        np.testing.assert_allclose(post.P_ecrh_e, ec_power, rtol=1.0e-10, atol=1.0)
        # The source shape is moment-matched to OpenSTEP, but the total rate is
        # deliberately the exact TORAX v1.4.2 STEP value (3e21 s^-1).
        np.testing.assert_allclose(post.S_pellet, 3.0e21, rtol=1.0e-12, atol=1.0)
        assert not np.isclose(float(post.S_pellet), pellet_rate, rtol=1.0e-3)
        np.testing.assert_allclose(
            post.P_alpha_total, fusion_power, rtol=2.0e-2, atol=0.0
        )
        np.testing.assert_allclose(
            post.P_radiation_e, radiation_power, rtol=2.0e-2, atol=0.0
        )
        np.testing.assert_allclose(post.P_heat_total, net_power, rtol=2.0e-2, atol=0.0)
        # EC heat moments are projected from OpenSTEP, while current-drive
        # efficiency is the exact TORAX v1.4.2 value (0.14). The released
        # OpenSTEP state is a physical reset reference, not a requirement that
        # TORAX's reduced current models reproduce a stationary balance at
        # reset. Hold acceptance belongs in the calibration report.
        assert float(post.I_ecrh) > 0.0
        assert float(post.I_bootstrap) > 0.0
        np.testing.assert_allclose(
            post.I_non_inductive,
            post.I_ecrh + post.I_bootstrap,
            rtol=1.0e-12,
            atol=1.0,
        )

    def test_simplified_source_deposition_moments(self):
        sim = self._reset_state.plasma.sim
        geometry = sim.geometry
        rho = np.asarray(geometry.torax_mesh.cell_centers)

        def torax_moments(profile, metric):
            weights = np.asarray(profile) * np.asarray(metric)
            mean = np.sum(rho * weights) / np.sum(weights)
            sigma = np.sqrt(np.sum(np.square(rho - mean) * weights) / np.sum(weights))
            return np.asarray([mean, sigma])

        with h5py.File(_OPENSTEP_PATH) as data:
            names = [
                _decode(name)
                for name in data["core_sources/0/source.identifier.name"][:]
            ]

            def released_moments(source_name, profile_path, measure_name):
                index = names.index(source_name)
                source_rho = data[
                    "core_sources/0/source.profiles_1d.grid.rho_tor_norm"
                ][index, 0]
                measure = data[
                    f"core_sources/0/source.profiles_1d.grid.{measure_name}"
                ][index, 0]
                profile = data[profile_path][index, 0]
                if profile.ndim > 1:
                    profile = np.sum(profile, axis=0)
                measure_derivative = np.gradient(measure, source_rho)
                norm = np.trapezoid(profile * measure_derivative, source_rho)
                mean = (
                    np.trapezoid(source_rho * profile * measure_derivative, source_rho)
                    / norm
                )
                sigma = np.sqrt(
                    np.trapezoid(
                        np.square(source_rho - mean) * profile * measure_derivative,
                        source_rho,
                    )
                    / norm
                )
                return np.asarray([mean, sigma])

            released_ec_heat = released_moments(
                "ec",
                "core_sources/0/source.profiles_1d.electrons.energy",
                "volume",
            )
            released_ec_current = released_moments(
                "ec",
                "core_sources/0/source.profiles_1d.j_parallel",
                "area",
            )
            released_pellet = released_moments(
                "pellet",
                "core_sources/0/source.profiles_1d.ion.particles",
                "volume",
            )

        np.testing.assert_allclose(
            torax_moments(sim.core_sources.T_e["ecrh"], geometry.vpr),
            released_ec_heat,
            rtol=0.0,
            atol=1.0e-3,
        )
        np.testing.assert_allclose(
            torax_moments(sim.core_sources.n_e["pellet"], geometry.vpr),
            released_pellet,
            rtol=0.0,
            atol=1.0e-3,
        )
        # One coupled Gaussian cannot match both released EC heat and current
        # shapes exactly. The reduced current profile preserves their first
        # two deposition moments within the stated absolute tolerances.
        torax_ec_current = torax_moments(sim.core_sources.psi["ecrh"], geometry.spr)
        np.testing.assert_allclose(
            torax_ec_current[0],
            released_ec_current[0],
            rtol=0.0,
            atol=1.0e-2,
        )
        np.testing.assert_allclose(
            torax_ec_current[1],
            released_ec_current[1],
            rtol=0.0,
            atol=2.0e-2,
        )

    def test_bohm_gyrobohm_uses_exact_torax_v1_4_2_step_settings(self):
        transport = load_backend(_BACKEND)["torax"]["transport"]
        for name in (
            "chi_e_bohm_multiplier",
            "chi_i_bohm_multiplier",
            "chi_e_gyrobohm_multiplier",
            "chi_i_gyrobohm_multiplier",
        ):
            assert transport[name] == 0.15
        assert transport["D_face_c1"] == 1.0
        assert transport["D_face_c2"] == 0.3
        assert transport["V_face_coeff"] == -0.1
        assert transport["chi_min"] == 0.15

    def test_step_is_pure_given_state_and_action(self):
        # Stochastic state owns its key, so replaying the same state and action
        # reproduces the complete emission without an external step key.
        state, _ = self._env.init(jax.random.key(0))
        action = jnp.zeros(self._env.action_space.shape)
        state_a, info_a = self._env.step(state, action)
        state_b, info_b = self._env.step(state, action)
        for a, b in zip(
            jax.tree.leaves((state_a, info_a)),
            jax.tree.leaves((state_b, info_b)),
            strict=True,
        ):
            if hasattr(a, "dtype") and jnp.issubdtype(a.dtype, jax.dtypes.prng_key):
                a = jax.random.key_data(a)
                b = jax.random.key_data(b)
            np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-10)
