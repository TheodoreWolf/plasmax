"""Envelope contract and physics regressions for :class:`PlasmaxEnv`."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from envelope import Continuous, Environment, Info, VmapWrapper
from helpers import (
    DEFAULT_ACTUATOR_SPECS,
    N_RHO,
    NOMINAL_ACTION,
    make_test_config,
    make_test_env,
)

from plasmax.environment import core as core_lib
from plasmax.environment.core import _derive_safe_max_steps
from plasmax.environment.schema import DisruptionConfig, PhysicsRandomizationSpec
from plasmax.spaces import ActuatorSpec
from plasmax.wrappers import (
    NoiseWrapper,
    PhysicsRandomizationWrapper,
    TimeAwareWrapper,
    unwrap_to_env_state,
)

# Expected observation size: 5 profiles * n_rho + 8 scalars.
_OBS_SIZE = 5 * N_RHO + 8
_ACTION = NOMINAL_ACTION


def _assert_initial_info(info, obs_shape=(_OBS_SIZE,)):
    """Checks the stable reset/init emission required by Envelope."""
    assert isinstance(info, Info)
    assert info.obs.shape == obs_shape
    assert jnp.all(jnp.isfinite(info.obs))
    np.testing.assert_array_equal(info.reward, 0.0)
    assert not bool(info.terminated)
    assert not bool(info.truncated)
    assert int(info.termination_code) == -1
    assert int(info.internal_steps) == 0
    assert int(info.sawtooth_crashes) == 0
    assert not bool(info.control_step_complete)
    assert not bool(info.step_limit_reached)


def _leaf_signature(tree):
    return tuple(
        (jnp.asarray(leaf).shape, jnp.asarray(leaf).dtype)
        for leaf in jax.tree.leaves(tree)
    )


def _assert_same_pytree_values(left, right):
    assert jax.tree.structure(left) == jax.tree.structure(right)
    for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True):
        if hasattr(a, "dtype") and jnp.issubdtype(a.dtype, jax.dtypes.prng_key):
            a = jax.random.key_data(a)
            b = jax.random.key_data(b)
        np.testing.assert_array_equal(a, b)


class PlasmaxEnvContractTest:
    """Tests the public Envelope lifecycle and unchanged TORAX behaviour."""

    @classmethod
    def setup_class(cls):
        cls._env = make_test_env()
        cls._key = jax.random.key(0)

    def test_is_envelope_environment_with_envelope_spaces(self):
        assert isinstance(self._env, Environment)
        assert isinstance(self._env.action_space, Continuous)
        assert isinstance(self._env.observation_space, Continuous)
        assert self._env.action_space.shape == (2,)
        assert self._env.observation_space.shape == (_OBS_SIZE,)

    def test_init_returns_state_and_stable_info(self):
        state, info = self._env.init(self._key)
        assert state is not None
        _assert_initial_info(info)

    def test_public_lifecycle_rejects_legacy_prng_keys(self):
        legacy_key = jax.random.PRNGKey(0)
        with pytest.raises(ValueError, match="typed|new-style"):
            self._env.init(legacy_key)

        state, _ = self._env.init(self._key)
        with pytest.raises(ValueError, match="typed|new-style"):
            self._env.reset(state, legacy_key)

    def test_reset_accepts_prior_state_and_fresh_typed_key(self):
        state, _ = self._env.init(self._key)
        stepped_state, _ = self._env.step(state, _ACTION)
        reset_state, reset_info = self._env.reset(stepped_state, jax.random.key(1))
        _assert_initial_info(reset_info)
        np.testing.assert_allclose(
            reset_state.plasma.t, state.plasma.t, rtol=1e-7, atol=0.0
        )
        np.testing.assert_array_equal(reset_state.prev_action, state.prev_action)

    def test_init_obs_is_deterministic_without_state_noise(self):
        # Physical initial state and emissions are deterministic without state noise.
        state1, info1 = self._env.init(jax.random.key(0))
        state2, info2 = self._env.init(jax.random.key(42))
        np.testing.assert_array_equal(info1.obs, info2.obs)
        _assert_same_pytree_values(state1.plasma, state2.plasma)
        np.testing.assert_array_equal(state1.prev_action, state2.prev_action)
        _assert_same_pytree_values(state1.phys_params, state2.phys_params)

    def test_step_advances_time_by_dt(self):
        # t_final=0.2, fixed_dt=0.1: the first transition lands at t=0.1.
        state, _ = self._env.init(self._key)
        t_before = state.plasma.t
        new_state, _ = self._env.step(state, _ACTION)
        np.testing.assert_allclose(
            new_state.plasma.t, t_before + 0.1, atol=1e-6, rtol=0.0
        )

    def test_step_emits_observation_reward_and_flags_in_info(self):
        state, _ = self._env.init(self._key)
        _, info = self._env.step(state, _ACTION)
        assert info.obs.shape == (_OBS_SIZE,)
        assert jnp.all(jnp.isfinite(info.obs))
        assert jnp.isfinite(info.reward)
        assert info.reward.dtype == jnp.float32
        assert not bool(info.terminated)
        assert not bool(info.truncated)
        assert int(info.termination_code) == -1
        assert int(info.internal_steps) == 1
        assert int(info.sawtooth_crashes) == 0
        assert bool(info.control_step_complete)
        assert not bool(info.step_limit_reached)

    def test_base_environment_never_truncates_at_t_final(self):
        # The base owns physics failures only. Its configured t_final is a safe
        # stepping horizon that a truncation wrapper enforces, not a done flag.
        state, _ = self._env.init(self._key)
        infos = []
        for _ in range(2):
            state, info = self._env.step(state, _ACTION)
            infos.append(info)
        np.testing.assert_allclose(state.plasma.t, 0.2, atol=1e-5, rtol=0.0)
        assert all(not bool(info.terminated) for info in infos)
        assert all(not bool(info.truncated) for info in infos)
        assert all(int(info.termination_code) == -1 for info in infos)
        assert not hasattr(self._env, "max_steps_in_episode")

    def test_info_structure_and_dtypes_are_stable_across_lifecycle(self):
        state, init_info = self._env.init(self._key)
        state, step_info = self._env.step(state, _ACTION)
        _, reset_info = self._env.reset(state, jax.random.key(8))

        expected_tree = jax.tree.structure(init_info)
        expected_signature = _leaf_signature(init_info)
        for info in (step_info, reset_info):
            assert jax.tree.structure(info) == expected_tree
            assert _leaf_signature(info) == expected_signature

    def test_higher_heating_increases_core_temperature(self):
        layout = self._env.obs_layout()
        core_te_idx = layout.slice_of("T_e").start
        initial_state, _ = self._env.init(self._key)

        _, low_info = self._env.step(initial_state, jnp.array([1e6, 1e21]))
        _, high_info = self._env.step(initial_state, jnp.array([30e6, 1e21]))

        # Threshold is in T_e/10 normalized units (registry scale=10.0).
        delta = float(high_info.obs[core_te_idx]) - float(low_info.obs[core_te_idx])
        assert delta > 1e-3

    def test_custom_reward_fn_returns_postout_value(self):
        def reward_fn(la, s, a, ns):
            del la, s, a
            return ns.plasma.tau_E

        env = make_test_env(reward_fn=reward_fn)
        state, _ = env.init(self._key)
        new_state, info = env.step(state, _ACTION)
        assert info.reward.dtype == jnp.float32
        np.testing.assert_array_equal(
            info.reward,
            np.asarray(new_state.plasma.tau_E, dtype=np.float32),
        )
        assert 0.0 < float(info.reward) < 10.0

    def test_rate_limiting_clips_large_action(self):
        env = make_test_env(
            actuator_specs=[
                ActuatorSpec("P_nbi", low=1e6, high=30e6, max_delta=1e6),
                DEFAULT_ACTUATOR_SPECS[1],
            ]
        )
        state, _ = env.init(self._key)
        prev_power = state.prev_action[0]
        new_state, _ = env.step(state, jnp.array([1e6, 1e21]))
        np.testing.assert_allclose(
            new_state.prev_action[0], prev_power - 1e6, atol=1.0, rtol=0.0
        )

    def test_no_rate_limiting_when_delta_is_none(self):
        env = make_test_env()
        state, _ = env.init(self._key)
        action = jnp.array([30e6, 5e21])
        new_state, _ = env.step(state, action)
        np.testing.assert_array_equal(new_state.prev_action, action)


class DisruptionContractTest:
    """Disruptions terminate; time horizons never do so in the base env."""

    @staticmethod
    def _step_with(disruption, **kwargs):
        env = make_test_env(disruption=disruption, disruption_penalty=-123.0, **kwargs)
        state, init_info = env.init(jax.random.key(0))
        state, info = env.step(state, _ACTION)
        return init_info, state, info

    @pytest.mark.parametrize(
        "disruption, expected_code",
        [
            (
                DisruptionConfig(q_min_threshold=1e9, greenwald_threshold=1e9),
                1,
            ),
            (
                DisruptionConfig(q_min_threshold=-1e9, greenwald_threshold=-1e9),
                2,
            ),
        ],
        ids=("q-min", "greenwald"),
    )
    def test_configured_disruption_terminates(self, disruption, expected_code):
        init_info, _, info = self._step_with(disruption)
        assert bool(info.terminated)
        assert not bool(info.truncated)
        assert int(info.termination_code) == expected_code
        np.testing.assert_array_equal(info.reward, -123.0)
        assert jax.tree.structure(info) == jax.tree.structure(init_info)
        assert _leaf_signature(info) == _leaf_signature(init_info)

    def test_q_min_has_priority_over_greenwald(self):
        both = DisruptionConfig(q_min_threshold=1e9, greenwald_threshold=-1e9)
        _, _, info = self._step_with(both)
        assert bool(info.terminated)
        assert int(info.termination_code) == 1

    def test_solver_failure_has_highest_priority(self):
        both = DisruptionConfig(q_min_threshold=1e9, greenwald_threshold=-1e9)

        def nonfinite_obs(_plasma):
            return jnp.full((_OBS_SIZE,), jnp.nan)

        _, _, info = self._step_with(both, obs_fn=nonfinite_obs)
        assert bool(info.terminated)
        assert not bool(info.truncated)
        assert int(info.termination_code) == 3
        np.testing.assert_array_equal(info.reward, -123.0)

    def test_solver_numeric_failure_masks_nonfinite_reward_with_penalty(
        self, monkeypatch
    ):
        disruption = DisruptionConfig(
            q_min_threshold=-1e9,
            greenwald_threshold=1e9,
        )

        def nonfinite_reward(last_action, state, action, next_state):
            del last_action, state, action, next_state
            return jnp.asarray(jnp.nan)

        env = make_test_env(
            disruption=disruption,
            disruption_penalty=-123.0,
            reward_fn=nonfinite_reward,
        )
        state, _ = env.init(jax.random.key(0))
        original_step = core_lib.fixed_duration_step

        def force_solver_failure(*args, **kwargs):
            result = original_step(*args, **kwargs)
            numeric = dataclasses.replace(
                result.sim_state.solver_numeric_outputs,
                solver_error_state=jnp.int32(1),
            )
            sim_state = dataclasses.replace(
                result.sim_state,
                solver_numeric_outputs=numeric,
            )
            return dataclasses.replace(result, sim_state=sim_state)

        monkeypatch.setattr(core_lib, "fixed_duration_step", force_solver_failure)
        _, info = env.step(state, _ACTION)

        assert jnp.all(jnp.isfinite(info.obs))
        assert bool(info.terminated)
        assert not bool(info.truncated)
        assert int(info.termination_code) == 3
        np.testing.assert_array_equal(info.reward, -123.0)
        assert bool(jnp.isfinite(info.reward))


class EnvStateTest:
    def _make_env_state(self):
        env = make_test_env()
        state, _ = env.init(jax.random.key(0))
        return state

    def test_is_jax_pytree(self):
        assert jax.tree.leaves(self._make_env_state())

    def test_leaves_are_arrays(self):
        for leaf in jax.tree.leaves(self._make_env_state()):
            assert isinstance(leaf, jax.Array)


class StateNoiseResetTest:
    """Reset-time domain randomization follows Envelope reset semantics."""

    def _make_noisy_env(self):
        return make_test_env(state_noise_config={"T_e": 0.05, "n_e": 0.03})

    def test_same_key_gives_same_obs(self):
        env = self._make_noisy_env()
        _, info1 = env.init(jax.random.key(7))
        _, info2 = env.init(jax.random.key(7))
        np.testing.assert_array_equal(info1.obs, info2.obs)

    def test_different_keys_give_different_obs(self):
        env = self._make_noisy_env()
        _, info1 = env.init(jax.random.key(1))
        _, info2 = env.init(jax.random.key(2))
        assert not jnp.allclose(info1.obs, info2.obs)

    def test_reset_uses_fresh_key(self):
        env = self._make_noisy_env()
        state, info1 = env.init(jax.random.key(1))
        _, info2 = env.reset(state, jax.random.key(2))
        assert not jnp.allclose(info1.obs, info2.obs)

    def test_no_state_noise_gives_deterministic_init(self):
        env = make_test_env()
        _, info1 = env.init(jax.random.key(1))
        _, info2 = env.init(jax.random.key(2))
        np.testing.assert_array_equal(info1.obs, info2.obs)

    def test_postout_recomputed_from_noisy_profiles(self):
        noisy = self._make_noisy_env()
        clean = make_test_env()
        noisy_state, _ = noisy.init(jax.random.key(11))
        clean_state, _ = clean.init(jax.random.key(11))
        assert jnp.isfinite(noisy_state.plasma.W_thermal_total)
        assert not jnp.allclose(
            noisy_state.plasma.W_thermal_total,
            clean_state.plasma.W_thermal_total,
        )

    def test_noisy_internal_energy_consistent_with_profiles(self):
        from torax._src.physics import formulas

        env = self._make_noisy_env()
        state, _ = env.init(jax.random.key(5))
        cp = state.plasma.core
        _, _, w_total = formulas.calculate_stored_thermal_energy(
            cp.pressure_thermal_e,
            cp.pressure_thermal_i,
            cp.pressure_thermal_total,
            state.plasma.geo,
        )
        np.testing.assert_allclose(
            cp.internal_plasma_energy.W_thermal_total,
            w_total,
            rtol=1e-6,
            atol=0.0,
        )

    def test_all_supported_noise_fields_are_jittable_together(self):
        env = make_test_env(state_noise_config={"T_i": 0.01, "T_e": 0.02, "n_e": 0.03})
        state, info = jax.jit(env.init)(jax.random.key(7))
        assert jnp.all(jnp.isfinite(info.obs))
        assert jnp.all(jnp.isfinite(state.plasma.core.T_i.value))
        assert jnp.all(jnp.isfinite(state.plasma.core.T_e.value))
        assert jnp.all(jnp.isfinite(state.plasma.core.n_e.value))


class ObsFnTest:
    """Custom observation shape and space stay aligned."""

    def test_none_obs_fn_gives_default_shape(self):
        env = make_test_env(obs_fn=None)
        _, info = env.init(jax.random.key(0))
        assert info.obs.shape == (_OBS_SIZE,)
        assert env.observation_space.shape == (_OBS_SIZE,)

    def test_custom_obs_fn_changes_shape(self):
        def my_obs_fn(plasma):
            return plasma.T_e / 10.0

        env = make_test_env(obs_fn=my_obs_fn)
        state, info = env.init(jax.random.key(0))
        assert info.obs.shape == (N_RHO,)
        assert env.observation_space.shape == (N_RHO,)
        assert env.obs_layout().size == N_RHO
        assert env.obs_layout().profile_names == ()
        assert env.obs_layout().scalar_names == ()
        assert jnp.all(jnp.isneginf(env.observation_space.low))
        assert jnp.all(jnp.isposinf(env.observation_space.high))
        _, info = env.step(state, _ACTION)
        assert info.obs.shape == (N_RHO,)

    def test_custom_obs_fn_must_return_a_flat_array(self):
        def matrix_obs_fn(plasma):
            return plasma.T_e[None, :]

        with pytest.raises(ValueError, match="flat|one-dimensional"):
            make_test_env(obs_fn=matrix_obs_fn)

    def test_custom_obs_fn_must_return_a_floating_array(self):
        def integer_obs_fn(plasma):
            return jnp.zeros(plasma.T_e.shape, dtype=jnp.int32)

        with pytest.raises(ValueError, match="floating array"):
            make_test_env(obs_fn=integer_obs_fn)


class SafeHorizonTest:
    def test_fixed_dt_must_advance_floating_point_time(self):
        config = make_test_config(
            numerics={"t_initial": 1.0, "t_final": 1.1, "fixed_dt": 1e-20}
        )
        with pytest.raises(ValueError, match="does not advance"):
            _derive_safe_max_steps(config)


class PhysicalControlGridTest:
    def test_time_varying_fixed_dt_is_resolved_at_each_transition_start(self):
        config = make_test_config(
            numerics={
                "t_final": 0.16,
                "fixed_dt": {0.0: 0.04, 0.07: 0.06},
                "exact_t_final": True,
            }
        )
        env = make_test_env(config=config)
        state, _ = env.init(jax.random.key(0))
        times = [state.plasma.t]
        for _ in range(env.safe_max_steps):
            state, info = env.step(state, _ACTION)
            assert bool(info.control_step_complete)
            times.append(state.plasma.t)
        np.testing.assert_allclose(
            times,
            [0.0, 0.04, 0.08, 0.14, 0.16],
            atol=1e-6,
            rtol=0.0,
        )

    def test_exact_final_clamps_non_integral_last_control_interval(self):
        config = make_test_config(
            numerics={
                "t_final": 0.25,
                "fixed_dt": 0.1,
                "exact_t_final": True,
            }
        )
        env = make_test_env(config=config)
        assert env.safe_max_steps == 3
        state, _ = env.init(jax.random.key(0))
        dts = []
        for expected_time in (0.1, 0.2, 0.25):
            state, info = env.step(state, _ACTION)
            dts.append(state.plasma.sim.dt)
            np.testing.assert_allclose(
                state.plasma.t, expected_time, atol=1e-6, rtol=0.0
            )
            assert bool(info.control_step_complete)
        np.testing.assert_allclose(dts, [0.1, 0.1, 0.05], atol=1e-6, rtol=0.0)


class PhysicsRandomizationTest:
    """Per-transition randomization consumes the wrapper-owned key."""

    _PATH = "numerics.resistivity_multiplier"
    _RANGE = (0.5, 0.9)

    @classmethod
    def setup_class(cls):
        cls._env = PhysicsRandomizationWrapper(
            make_test_env(
                physics_randomization={
                    cls._PATH: PhysicsRandomizationSpec(absolute=cls._RANGE)
                }
            )
        )

    def test_disabled_by_default_gives_empty_phys_params(self):
        state, _ = make_test_env().init(jax.random.key(0))
        assert unwrap_to_env_state(state).phys_params == {}

    def test_fixed_dt_cannot_be_randomized(self):
        with pytest.raises(ValueError, match="physical control interval"):
            make_test_env(
                physics_randomization={
                    "numerics.fixed_dt": PhysicsRandomizationSpec(absolute=(0.05, 0.15))
                }
            )

    def test_init_seeds_nominal_value_before_first_transition(self):
        state, _ = self._env.init(jax.random.key(0))
        np.testing.assert_array_equal(
            unwrap_to_env_state(state).phys_params[self._PATH], 1.0
        )

    def test_transition_sample_within_range(self):
        state, _ = self._env.init(jax.random.key(0))
        state, _ = self._env.step(state, _ACTION)
        lo, hi = self._RANGE
        assert lo <= float(unwrap_to_env_state(state).phys_params[self._PATH]) <= hi

    def test_relative_range_multiplies_nominal_value(self):
        config = make_test_config(
            numerics={
                "t_final": 0.2,
                "fixed_dt": 0.1,
                "resistivity_multiplier": 0.8,
            }
        )
        env = PhysicsRandomizationWrapper(
            make_test_env(
                config=config,
                physics_randomization={
                    self._PATH: PhysicsRandomizationSpec(relative=(0.5, 1.5))
                },
            )
        )
        state, _ = env.init(jax.random.key(0))
        state, _ = env.step(state, _ACTION)
        assert 2.0 <= float(unwrap_to_env_state(state).phys_params[self._PATH]) <= 6.0

    def test_same_initial_key_is_reproducible(self):
        state1, _ = self._env.init(jax.random.key(3))
        state2, _ = self._env.init(jax.random.key(3))
        state1, info1 = self._env.step(state1, _ACTION)
        state2, info2 = self._env.step(state2, _ACTION)
        np.testing.assert_array_equal(
            unwrap_to_env_state(state1).phys_params[self._PATH],
            unwrap_to_env_state(state2).phys_params[self._PATH],
        )
        np.testing.assert_array_equal(info1.obs, info2.obs)

    def test_different_initial_keys_drive_different_transitions(self):
        state1, _ = self._env.init(jax.random.key(1))
        state2, _ = self._env.init(jax.random.key(2))
        state1, _ = self._env.step(state1, _ACTION)
        state2, _ = self._env.step(state2, _ACTION)
        assert not jnp.allclose(
            unwrap_to_env_state(state1).phys_params[self._PATH],
            unwrap_to_env_state(state2).phys_params[self._PATH],
        )

    def test_repeated_step_from_same_state_is_pure(self):
        initial_state, _ = self._env.init(jax.random.key(3))
        state1, info1 = self._env.step(initial_state, _ACTION)
        state2, info2 = self._env.step(initial_state, _ACTION)
        _assert_same_pytree_values(state1, state2)
        _assert_same_pytree_values(info1, info2)

    def test_key_advances_and_randomization_is_resampled_each_transition(self):
        state, _ = self._env.init(jax.random.key(0))
        state1, _ = self._env.step(state, _ACTION)
        state2, _ = self._env.step(state1, _ACTION)
        assert not jnp.allclose(
            unwrap_to_env_state(state2).phys_params[self._PATH],
            unwrap_to_env_state(state1).phys_params[self._PATH],
        )

    def test_updates_persist_through_nested_state_and_reset_to_nominals(self):
        base = make_test_env(
            physics_randomization={
                self._PATH: PhysicsRandomizationSpec(relative=(0.5, 1.5)),
                "neoclassical.bootstrap_current.bootstrap_multiplier": (
                    PhysicsRandomizationSpec(relative=(0.5, 1.5))
                ),
            }
        )
        env = TimeAwareWrapper(
            NoiseWrapper(base, jnp.zeros(base.observation_space.shape))
        )
        original, _ = env.init(jax.random.key(0))
        state = env.with_physics(original, {self._PATH: jnp.asarray(0.7)})
        state = env.with_physics(state, {self._PATH: jnp.asarray(0.8)})
        np.testing.assert_array_equal(
            unwrap_to_env_state(original).phys_params[self._PATH], 1.0
        )
        for _ in range(2):
            state, _ = env.step(state, _ACTION)
            np.testing.assert_array_equal(
                unwrap_to_env_state(state).phys_params[self._PATH], 0.8
            )
            np.testing.assert_array_equal(
                unwrap_to_env_state(state).phys_params[
                    "neoclassical.bootstrap_current.bootstrap_multiplier"
                ],
                base.physics_nominals[
                    "neoclassical.bootstrap_current.bootstrap_multiplier"
                ],
            )
        reset, _ = env.reset(state, jax.random.key(1))
        _assert_same_pytree_values(
            unwrap_to_env_state(reset).phys_params, base.physics_nominals
        )

    def test_randomization_uses_nominals_and_restarts_its_stream_on_reset(self):
        env = PhysicsRandomizationWrapper(
            make_test_env(
                physics_randomization={
                    self._PATH: PhysicsRandomizationSpec(relative=(0.5, 1.0))
                }
            )
        )
        key = jax.random.key(7)
        initial, _ = env.init(key)
        changed = env.with_physics(initial, {self._PATH: jnp.asarray(100.0)})
        state, info = env.step(changed, _ACTION)
        value = unwrap_to_env_state(state).phys_params[self._PATH]
        assert 0.5 <= value <= 1.0
        # Preserve the sampler stream used before extraction from the core.
        step_key, _ = jax.random.split(key)
        _, physics_key = jax.random.split(step_key)
        sample_key = jax.random.split(physics_key, 1)[0]
        expected = jax.random.uniform(
            sample_key, (), minval=0.5, maxval=1.0, dtype=value.dtype
        )
        np.testing.assert_array_equal(value, expected)
        reset, _ = env.reset(state, key)
        _assert_same_pytree_values(reset, initial)
        repeated, repeated_info = env.step(reset, _ACTION)
        _assert_same_pytree_values(state, repeated)
        _assert_same_pytree_values(info, repeated_info)


@pytest.mark.integration
class PlasmaxEnvTransformContractTest:
    """Expensive transform gates for the fast circular TORAX scenario."""

    def test_jit_lax_scan(self):
        env = PhysicsRandomizationWrapper(
            make_test_env(
                physics_randomization={
                    "numerics.resistivity_multiplier": PhysicsRandomizationSpec(
                        relative=(0.5, 1.5)
                    )
                }
            )
        )
        state, _ = env.init(jax.random.key(0))
        actions = jnp.broadcast_to(_ACTION, (2, 2))

        @jax.jit
        def rollout(initial_state, rollout_actions):
            def transition(carry, action):
                next_state, info = env.step(carry, action)
                emissions = (
                    info.obs,
                    info.reward,
                    info.terminated,
                    info.truncated,
                    info.termination_code,
                )
                return next_state, emissions

            return jax.lax.scan(transition, initial_state, rollout_actions)

        final_state, emissions = rollout(state, actions)
        obs, reward, terminated, truncated, termination_code = emissions
        assert obs.shape == (2, _OBS_SIZE)
        assert reward.shape == (2,)
        assert not jnp.any(terminated)
        assert not jnp.any(truncated)
        assert jnp.all(termination_code == -1)
        np.testing.assert_allclose(
            unwrap_to_env_state(final_state).plasma.t, 0.2, atol=1e-5, rtol=0.0
        )

    def test_manual_vmap_and_envelope_vmap_wrapper(self):
        randomized_path = "numerics.resistivity_multiplier"
        env = PhysicsRandomizationWrapper(
            make_test_env(
                physics_randomization={
                    randomized_path: PhysicsRandomizationSpec(absolute=(0.5, 1.5))
                }
            )
        )
        keys = jax.random.split(jax.random.key(9), 2)
        actions = jnp.broadcast_to(_ACTION, (2, 2))

        manual_state, manual_init_info = jax.vmap(env.init)(keys)
        manual_state, manual_step_info = jax.vmap(env.step)(manual_state, actions)

        vector_env = VmapWrapper(env=env, batch_size=2)
        wrapped_state, wrapped_init_info = vector_env.init(keys)
        wrapped_state, wrapped_step_info = vector_env.step(wrapped_state, actions)

        _assert_same_pytree_values(manual_init_info, wrapped_init_info)
        _assert_same_pytree_values(manual_state, wrapped_state)
        _assert_same_pytree_values(manual_step_info, wrapped_step_info)
        randomized_values = unwrap_to_env_state(manual_state).phys_params[
            randomized_path
        ]
        assert randomized_values.shape == (2,)
        assert not jnp.allclose(randomized_values[0], randomized_values[1])
        assert vector_env.action_space.shape == (2, 2)
        assert vector_env.observation_space.shape == (2, _OBS_SIZE)

    def test_jvp_and_reverse_mode_through_real_torax_step(self):
        path = "numerics.resistivity_multiplier"
        env = make_test_env(
            physics_randomization={path: PhysicsRandomizationSpec(relative=(0.5, 1.5))}
        )
        state, _ = env.init(jax.random.key(0))

        def objective(action, resistance):
            updated = env.with_physics(state, {path: resistance})
            return env.step(updated, action)[1].reward

        value, tangent = jax.jvp(
            objective,
            (_ACTION, jnp.asarray(0.8)),
            (jnp.ones_like(_ACTION), jnp.asarray(1.0)),
        )
        reverse_value, gradients = jax.value_and_grad(objective, argnums=(0, 1))(
            _ACTION, jnp.asarray(0.8)
        )
        np.testing.assert_allclose(reverse_value, value, atol=1e-7, rtol=1e-6)
        np.testing.assert_allclose(
            tangent, gradients[0].sum() + gradients[1], atol=1e-7, rtol=1e-6
        )
        assert jnp.isfinite(tangent)
        assert all(jnp.all(jnp.isfinite(gradient)) for gradient in gradients)
