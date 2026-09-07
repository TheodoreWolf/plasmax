"""Forward-reference checks against TORAX's fixed_time_step helper."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from helpers import NOMINAL_ACTION, make_test_config, make_test_env

from plasmax.control import ControlInputs
from plasmax.environment.schema import SteppingConfig
from plasmax.wrappers import TruncationWrapper


def _adaptive_backtracking_config():
    return make_test_config(
        numerics={
            "t_final": 0.2,
            "fixed_dt": 0.1,
            "adaptive_dt": True,
            "min_dt": 1.0e-8,
            "dt_reduction_factor": 3.0,
        },
        solver={
            "solver_type": "newton_raphson",
            "n_max_iterations": 2,
            "residual_tol": 1.0e-17,
            "residual_coarse_tol": 1.0e-16,
            "use_pereverzev": True,
            "chi_pereverzev": 30.0,
            "D_pereverzev": 15.0,
        },
    )


def _runtime_provider_for_action(env, action):
    dynamics = env._dynamics
    kwargs = {name: action[index] for index, name in enumerate(dynamics._actuators)}
    controls = ControlInputs(**kwargs)
    return dynamics._phys_applier(
        {},
        dynamics._applier(controls, dynamics._step_fn.runtime_params_provider),
    )


def _assert_reference_endpoint(actual, expected):
    for name in ("T_e", "T_i", "n_e", "psi"):
        np.testing.assert_allclose(
            getattr(actual.core_profiles, name).value,
            getattr(expected.core_profiles, name).value,
            rtol=1e-6,
            atol=1e-7,
        )
    for name in ("q_face", "s_face"):
        np.testing.assert_allclose(
            getattr(actual.core_profiles, name),
            getattr(expected.core_profiles, name),
            rtol=1e-6,
            atol=1e-7,
        )


def _assert_reference_post(actual, expected):
    for name in (
        "W_thermal_total",
        "tau_E",
        "P_fusion",
        "q_min",
        "q95",
        "fgw_n_e_line_avg",
    ):
        np.testing.assert_allclose(
            getattr(actual, name),
            getattr(expected, name),
            rtol=1e-6,
            atol=1e-7,
        )


@pytest.mark.integration
class ToraxFixedDurationReferenceTest:
    def test_adaptive_backtracking_matches_torax_fixed_time_step(self):
        # A deliberately strict low-iteration NR tolerance forces TORAX to
        # accept shortened steps after dt backtracking in this cheap model.
        env = make_test_env(
            config=_adaptive_backtracking_config(),
            stepping=SteppingConfig(max_solver_substeps=128),
        )
        state, _ = env.init(jax.random.key(0))
        provider = _runtime_provider_for_action(env, NOMINAL_ACTION)
        reference_sim, reference_post = env._dynamics._step_fn.fixed_time_step(
            jnp.asarray(0.1),
            state.plasma.sim,
            state.plasma.post,
            runtime_params_overrides=provider,
        )

        actual_state, info = env.step(state, NOMINAL_ACTION)
        assert int(info.internal_steps) > 1
        assert int(info.internal_steps) < 128
        assert bool(info.control_step_complete)
        assert not bool(info.step_limit_reached)
        np.testing.assert_allclose(
            actual_state.plasma.t, reference_sim.t, rtol=0.0, atol=1e-7
        )
        _assert_reference_endpoint(actual_state.plasma.sim, reference_sim)
        _assert_reference_post(actual_state.plasma.post, reference_post)

    def test_solver_budget_exhaustion_terminates_with_code_three(self):
        env = make_test_env(config=_adaptive_backtracking_config())
        state, _ = env.init(jax.random.key(0))
        next_state, info = env.step(state, NOMINAL_ACTION)

        assert bool(info.terminated)
        assert int(info.termination_code) == 3
        assert int(info.internal_steps) == 1
        assert bool(info.step_limit_reached)
        assert not bool(info.control_step_complete)
        assert 0.0 < float(next_state.plasma.t) < 0.1

    def test_episode_truncation_counts_control_intervals_not_internal_calls(self):
        base_env = make_test_env(
            config=_adaptive_backtracking_config(),
            stepping=SteppingConfig(max_solver_substeps=128),
        )
        env = TruncationWrapper(env=base_env, max_steps=1)
        state, _ = env.init(jax.random.key(0))
        state, info = env.step(state, NOMINAL_ACTION)

        assert int(info.internal_steps) > 1
        assert bool(info.control_step_complete)
        assert int(state.steps) == 1
        assert bool(info.truncated)
        assert not bool(info.terminated)

    def test_ordinary_step_matches_torax_fixed_time_step(self):
        env = make_test_env()
        state, _ = env.init(jax.random.key(0))
        provider = _runtime_provider_for_action(env, NOMINAL_ACTION)
        reference_sim, reference_post = env._dynamics._step_fn.fixed_time_step(
            jnp.asarray(0.1),
            state.plasma.sim,
            state.plasma.post,
            runtime_params_overrides=provider,
        )

        actual_state, info = env.step(state, NOMINAL_ACTION)
        np.testing.assert_allclose(
            actual_state.plasma.t, reference_sim.t, rtol=0.0, atol=1e-7
        )
        np.testing.assert_allclose(
            actual_state.plasma.sim.dt, reference_sim.dt, rtol=0.0, atol=1e-7
        )
        _assert_reference_endpoint(actual_state.plasma.sim, reference_sim)
        _assert_reference_post(actual_state.plasma.post, reference_post)
        assert int(info.internal_steps) == 1
        assert bool(info.control_step_complete)

    def test_forced_sawtooth_matches_torax_and_completes_interval(self):
        config = make_test_config(
            mhd={
                "sawtooth": {
                    "trigger_model": {
                        "model_name": "simple",
                        "s_critical": 0.1,
                        "minimum_radius": 0.05,
                    },
                    "redistribution_model": {
                        "model_name": "simple",
                        "flattening_factor": 1.01,
                        "mixing_radius_multiplier": 1.1,
                    },
                    "crash_step_duration": 1.0e-5,
                }
            }
        )
        env = make_test_env(
            config=config,
            stepping=SteppingConfig(
                max_solver_substeps=1,
                max_event_substeps=1,
            ),
        )
        state, _ = env.init(jax.random.key(0))

        # Force one deterministic q=1 crossing with supercritical shear. The
        # redistribution itself still acts on the physically initialized psi.
        sim = state.plasma.sim
        core = dataclasses.replace(
            sim.core_profiles,
            q_face=jnp.linspace(0.8, 1.4, sim.core_profiles.q_face.shape[0]),
            s_face=jnp.ones_like(sim.core_profiles.s_face),
        )
        sim = dataclasses.replace(sim, core_profiles=core)
        state = dataclasses.replace(
            state,
            plasma=dataclasses.replace(state.plasma, sim=sim),
        )
        provider = _runtime_provider_for_action(env, NOMINAL_ACTION)
        reference_sim, reference_post = env._dynamics._step_fn.fixed_time_step(
            jnp.asarray(0.1),
            state.plasma.sim,
            state.plasma.post,
            runtime_params_overrides=provider,
        )

        actual_state, info = env.step(state, NOMINAL_ACTION)
        np.testing.assert_allclose(actual_state.plasma.t, 0.1, rtol=0.0, atol=1e-7)
        np.testing.assert_allclose(actual_state.plasma.sim.dt, 0.1, rtol=0.0, atol=1e-7)
        _assert_reference_endpoint(actual_state.plasma.sim, reference_sim)
        _assert_reference_post(actual_state.plasma.post, reference_post)
        assert int(info.internal_steps) == 2
        assert int(info.sawtooth_crashes) == 1
        assert bool(info.control_step_complete)
        assert not bool(info.step_limit_reached)
        assert jnp.all(jnp.isfinite(info.obs))
