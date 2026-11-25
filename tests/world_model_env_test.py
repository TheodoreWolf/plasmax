"""Envelope contract, wrapper compatibility, and golden tests for KSTAR."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from envelope import (
    AutoResetWrapper,
    Continuous,
    Discrete,
    Environment,
    Info,
    ObservationNormalizationWrapper,
    VmapWrapper,
)

from plasmax.environment.config import parse_env_and_backend
from plasmax.environment.factory import make
from plasmax.environment.schema import PhysicsRandomizationSpec
from plasmax.models.world_model import load_bundle
from plasmax.models.world_model_env import (
    CTRL_IDX,
    INPUT_MAXS,
    INPUT_MINS,
    WorldModelEnv,
    WorldModelEnvState,
    target_tracking_reward,
)
from plasmax.rollout import collect_episode
from plasmax.wrappers import (
    ActionRescaleWrapper,
    CorruptedActionWrapper,
    NoiseWrapper,
    ObsDelayConfig,
    ObsDelayWrapper,
    ObsFilterConfig,
    ObsFilterWrapper,
    ObsHistoryWrapper,
    OracleWrappers,
    PhysicsRandomizationWrapper,
    QuantizeActionWrapper,
    RealisticWrappers,
    SensorNoiseConfig,
    TimeAwareWrapper,
    TruncationWrapper,
    iter_wrappers,
    unwrap_to_env_state,
)

_ACT_LOW = INPUT_MINS[CTRL_IDX].astype(np.float32)
_ACT_HIGH = INPUT_MAXS[CTRL_IDX].astype(np.float32)
_MIDPOINT_ACTION = (0.5 * (_ACT_LOW + _ACT_HIGH)).astype(np.float32)
_RESET_ACTION = np.array([0.5, 1.7, 0.3, 0.75, 1.32, 2.22], np.float32)

# Resets use the authoritative four-significant-figure YAML history.
_GOLDEN_RESET = np.array(
    [
        0.5,
        1.7,
        0.3,
        0.75,
        1.32,
        2.22,
        1.346700,
        4.841000,
        1.080000,
        1.6,
        5.0,
        0.95,
        1.5,
        1.5,
        0.5,
    ],
    np.float32,
)
# Continuous midpoint physical controls applied to the same packaged reset history.
_GOLDEN_STEP0 = np.array(
    [
        0.55,
        1.8,
        0.3,
        0.7,
        1.3125,
        2.235,
        1.404872,
        4.965048,
        0.867773,
        1.6,
        5.0,
        0.95,
        1.5,
        1.5,
        0.5,
    ],
    np.float32,
)
_GOLDEN_STEP0_REWARD = -0.09960221


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


def _assert_initial_info(info, obs_shape=(15,)):
    assert isinstance(info, Info)
    assert info.obs.shape == obs_shape
    assert jnp.all(jnp.isfinite(info.obs))
    np.testing.assert_array_equal(info.reward, 0.0)
    assert not bool(info.terminated)
    assert not bool(info.truncated)
    assert int(info.termination_code) == -1


class WorldModelEnvContractTest:
    def setup_method(self):
        self.env = WorldModelEnv(random_target=False)

    def test_is_envelope_environment_with_envelope_spaces(self):
        assert isinstance(self.env, Environment)
        assert isinstance(self.env.action_space, Continuous)
        assert isinstance(self.env.observation_space, Continuous)
        assert self.env.action_space.shape == (6,)
        assert self.env.observation_space.shape == (15,)
        np.testing.assert_array_equal(self.env.action_space.low, _ACT_LOW)
        np.testing.assert_array_equal(self.env.action_space.high, _ACT_HIGH)
        assert [spec.name for spec in self.env.actuator_specs] == [
            "Ip",
            "Elon",
            "Up.Tri",
            "Lo.Tri",
            "In.Mid",
            "Out.Mid",
        ]
        np.testing.assert_array_equal(
            self.env.observation_space.low[9:12],
            np.array([1.1, 3.8, 0.84], np.float32),
        )
        np.testing.assert_array_equal(
            self.env.observation_space.high[9:12],
            np.array([2.1, 6.2, 1.06], np.float32),
        )
        assert jnp.all(jnp.isneginf(self.env.observation_space.low[6:9]))
        assert jnp.all(jnp.isposinf(self.env.observation_space.high[6:9]))

    def test_obs_layout_names(self):
        layout = self.env.obs_layout()
        assert layout.scalar_names[6:9] == ("betap", "q95", "li")
        assert layout.profile_names == ()
        assert layout.slice_of("betap") == slice(6, 7)

    def test_init_golden_and_info_contract(self):
        state, info = self.env.init(jax.random.key(0))
        _assert_initial_info(info)
        assert bool(self.env.observation_space.contains(info.obs))
        assert int(state.t) == 0
        np.testing.assert_allclose(
            state.prev_action, _RESET_ACTION, rtol=1e-6, atol=1e-6
        )
        assert state.phys_params == {}
        np.testing.assert_allclose(info.obs, _GOLDEN_RESET, rtol=1e-4, atol=1e-4)

    def test_public_lifecycle_rejects_legacy_prng_keys(self):
        legacy_key = jax.random.PRNGKey(0)
        with pytest.raises(ValueError, match="typed|new-style"):
            self.env.init(legacy_key)

        state, _ = self.env.init(jax.random.key(0))
        with pytest.raises(ValueError, match="typed|new-style"):
            self.env.reset(state, legacy_key)

    def test_reset_accepts_prior_state_and_fresh_key(self):
        state, _ = self.env.init(jax.random.key(0))
        state, _ = self.env.step(state, jnp.asarray(_MIDPOINT_ACTION))
        state, info = self.env.reset(state, jax.random.key(1))
        _assert_initial_info(info)
        assert int(state.t) == 0
        np.testing.assert_allclose(
            state.prev_action, _RESET_ACTION, rtol=1e-6, atol=1e-6
        )
        np.testing.assert_allclose(info.obs, _GOLDEN_RESET, rtol=1e-4, atol=1e-4)

    def test_step_golden(self):
        state, _ = self.env.init(jax.random.key(0))
        new_state, info = self.env.step(state, jnp.asarray(_MIDPOINT_ACTION))
        np.testing.assert_allclose(info.obs, _GOLDEN_STEP0, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(
            info.reward, _GOLDEN_STEP0_REWARD, rtol=1e-4, atol=1e-4
        )
        assert bool(self.env.observation_space.contains(info.obs))
        assert not bool(info.terminated)
        assert not bool(info.truncated)
        assert int(info.termination_code) == -1
        assert int(new_state.t) == 1
        np.testing.assert_allclose(
            new_state.prev_action, _MIDPOINT_ACTION, rtol=1e-6, atol=1e-6
        )

        rescaled = ActionRescaleWrapper(self.env)
        r_state, _ = rescaled.init(jax.random.key(0))
        r_next, r_info = rescaled.step(r_state, jnp.zeros(6, dtype=jnp.float32))
        _assert_same_pytree_values(r_next, new_state)
        _assert_same_pytree_values(r_info, info)

    def test_step_is_pure_given_state_and_action(self):
        state, _ = self.env.init(jax.random.key(0))
        action = jnp.asarray(_ACT_LOW + 0.65 * (_ACT_HIGH - _ACT_LOW))
        state1, info1 = self.env.step(state, action)
        state2, info2 = self.env.step(state, action)
        _assert_same_pytree_values(state1, state2)
        _assert_same_pytree_values(info1, info2)

    def test_base_environment_never_owns_horizon_truncation(self):
        state, _ = self.env.init(jax.random.key(0))
        for _ in range(4):
            state, info = self.env.step(state, jnp.asarray(_MIDPOINT_ACTION))
            assert not bool(info.terminated)
            assert not bool(info.truncated)
            assert int(info.termination_code) == -1
        assert int(state.t) == 4
        assert not hasattr(self.env, "max_steps_in_episode")
        # Direct WorldModelEnv() without plasmax_config still accepts truncation.
        truncated_env = TruncationWrapper(self.env, max_steps=2)
        t_state, _ = truncated_env.init(jax.random.key(0))
        t_state, info1 = truncated_env.step(t_state, jnp.asarray(_MIDPOINT_ACTION))
        _, info2 = truncated_env.step(t_state, jnp.asarray(_MIDPOINT_ACTION))
        assert not bool(info1.truncated)
        assert bool(info2.truncated)

    def test_info_structure_and_dtypes_are_stable(self):
        state, init_info = self.env.init(jax.random.key(0))
        state, step_info = self.env.step(state, jnp.asarray(_MIDPOINT_ACTION))
        _, reset_info = self.env.reset(state, jax.random.key(1))
        expected_tree = jax.tree.structure(init_info)
        expected_signature = _leaf_signature(init_info)
        for info in (step_info, reset_info):
            assert jax.tree.structure(info) == expected_tree
            assert _leaf_signature(info) == expected_signature

    def test_random_target_reproducibility_uses_typed_init_key(self):
        env = WorldModelEnv(random_target=True)
        state1, info1 = env.init(jax.random.key(7))
        state2, info2 = env.init(jax.random.key(7))
        state3, info3 = env.init(jax.random.key(8))
        np.testing.assert_array_equal(state1.targets, state2.targets)
        np.testing.assert_array_equal(info1.obs, info2.obs)
        assert not jnp.allclose(state1.targets, state3.targets)
        assert not jnp.allclose(info1.obs, info3.obs)
        expected = jax.random.uniform(
            jax.random.key(7),
            (3,),
            minval=jnp.array([1.1, 3.8, 0.84], dtype=jnp.float64),
            maxval=jnp.array([2.1, 6.2, 1.06], dtype=jnp.float64),
            dtype=jnp.float64,
        ).astype(jnp.float32)
        np.testing.assert_array_equal(state1.targets, expected)

    def test_reset_repeats_the_saved_history_row(self):
        config = parse_env_and_backend("kstar_worldmodel")
        document = config._initial_state
        state, _ = self.env.init(jax.random.key(0))
        np.testing.assert_array_equal(
            state.x,
            np.tile(np.array(document.history_row, np.float32), (10, 1)),
        )
        np.testing.assert_array_equal(
            state.inputs,
            np.array(
                [document.inputs[name] for name in document.input_order], np.float32
            ),
        )

    def test_reset_needs_only_saved_history_and_dynamics_weights(self):
        # The steady-state NN is used only by the capture tool.
        bundle = {name: value for name, value in load_bundle().items() if name != "nn"}
        env = WorldModelEnv.from_bundle(bundle, random_target=False)
        _, info = jax.jit(env.init)(jax.random.key(0))
        np.testing.assert_allclose(info.obs, _GOLDEN_RESET, rtol=1e-4, atol=1e-4)

    def test_reward_is_max_at_target(self):
        target = jnp.array([1.6, 5.0, 0.95])
        on = target_tracking_reward(target, target)
        off = target_tracking_reward(target + jnp.array([0.5, 1.0, 0.1]), target)
        assert jnp.isfinite(on)
        assert float(on) > float(off)

    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
    @pytest.mark.parametrize("offset", [0.0, 1e-7])
    def test_reward_gradient_is_zero_within_target_floor(
        self, dtype: type[np.floating], offset: float
    ) -> None:
        target = jnp.array([1.6, 5.0, 0.95], dtype=dtype)
        prediction = target.at[0].add(offset)
        value, gradient = jax.jit(jax.value_and_grad(target_tracking_reward))(
            prediction, target
        )
        np.testing.assert_allclose(value, -np.log(1e-6), rtol=1e-7, atol=0.0)
        np.testing.assert_array_equal(gradient, np.zeros(3))


class WorldModelTransformContractTest:
    def setup_method(self):
        self.env = WorldModelEnv(random_target=False)

    def test_jit_lax_scan(self):
        state, _ = jax.jit(self.env.init)(jax.random.key(0))
        actions = jnp.broadcast_to(jnp.asarray(_MIDPOINT_ACTION), (5, 6))

        @jax.jit
        def rollout(initial_state, rollout_actions):
            def transition(carry, action):
                next_state, info = self.env.step(carry, action)
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
        obs, rewards, terminated, truncated, termination_code = emissions
        assert obs.shape == (5, 15)
        assert rewards.shape == (5,)
        assert jnp.all(jnp.isfinite(rewards))
        assert not jnp.any(terminated)
        assert not jnp.any(truncated)
        assert jnp.all(termination_code == -1)
        assert int(final_state.t) == 5

    def test_manual_vmap_and_envelope_vmap_wrapper(self):
        keys = jax.random.split(jax.random.key(0), 8)
        actions = jnp.broadcast_to(jnp.asarray(_MIDPOINT_ACTION), (8, 6))

        manual_states, manual_init_info = jax.vmap(self.env.init)(keys)
        manual_states, manual_step_info = jax.vmap(self.env.step)(
            manual_states, actions
        )

        vector_env = VmapWrapper(env=self.env, batch_size=8)
        wrapped_states, wrapped_init_info = vector_env.init(keys)
        wrapped_states, wrapped_step_info = vector_env.step(wrapped_states, actions)

        _assert_same_pytree_values(manual_init_info, wrapped_init_info)
        _assert_same_pytree_values(manual_states, wrapped_states)
        _assert_same_pytree_values(manual_step_info, wrapped_step_info)
        assert vector_env.action_space.shape == (8, 6)
        assert vector_env.observation_space.shape == (8, 15)

    def test_random_targets_are_independent_under_vmap(self):
        env = WorldModelEnv(random_target=True)
        keys = jax.random.split(jax.random.key(4), 8)

        manual_states, manual_info = jax.vmap(env.init)(keys)
        vector_env = VmapWrapper(env=env, batch_size=8)
        wrapped_states, wrapped_info = vector_env.init(keys)

        _assert_same_pytree_values(manual_states, wrapped_states)
        _assert_same_pytree_values(manual_info, wrapped_info)
        assert not jnp.allclose(manual_states.targets[0], manual_states.targets[1])

    def test_rollout_action_gradients_match_finite_differences(self) -> None:
        rescaled = ActionRescaleWrapper(self.env)
        state, _ = rescaled.init(jax.random.key(0))
        actions = jnp.asarray(
            [[0.1, -0.2, 0.15, -0.1, 0.2, -0.15], [-0.1, 0.15, -0.2, 0.2, -0.15, 0.1]],
            dtype=jnp.float32,
        )

        def rollout_return(rollout_actions: jax.Array) -> jax.Array:
            def transition(
                carry: WorldModelEnvState, action: jax.Array
            ) -> tuple[WorldModelEnvState, jax.Array]:
                next_state, info = rescaled.step(carry, action)
                return next_state, info.reward

            _, rewards = jax.lax.scan(transition, state, rollout_actions)
            return jnp.sum(rewards)

        gradient = jax.jit(jax.grad(rollout_return))(actions)
        assert np.all(np.isfinite(gradient))
        assert np.all(np.any(np.abs(gradient) > 1e-5, axis=0))

        forward = jax.jit(rollout_return)
        epsilon = 2e-3
        differences = np.empty(actions.shape, dtype=np.float32)
        for index in np.ndindex(actions.shape):
            offset = jnp.zeros_like(actions).at[index].set(epsilon)
            differences[index] = (
                forward(actions + offset) - forward(actions - offset)
            ) / (2 * epsilon)
        np.testing.assert_allclose(gradient, differences, rtol=1e-2, atol=1e-3)


class KstarLoaderTest:
    """The unified loader and wrapper stack drive KSTAR identically to TORAX."""

    _ENV = "kstar_worldmodel"

    def test_validate_env_backend(self):
        from plasmax.environment.merge import (
            valid_env_backend_combos,
            validate_env_backend,
        )

        assert valid_env_backend_combos()[self._ENV] == frozenset()
        validate_env_backend(self._ENV, None)
        with pytest.raises(ValueError, match="standalone"):
            validate_env_backend(self._ENV, "qlknn")

    def test_load_env_returns_scalar_envelope_environment(self):
        env = RealisticWrappers(make(self._ENV))
        assert isinstance(env, Environment)
        assert env.action_space.shape == (6,)
        np.testing.assert_array_equal(env.action_space.low, -np.ones(6))
        np.testing.assert_array_equal(env.action_space.high, np.ones(6))
        assert env.observation_space.shape == (15,)
        assert env.max_steps == 100

        state, init_info = env.init(jax.random.key(0))
        _assert_initial_info(init_info)
        _, info = env.step(state, jnp.zeros(6))
        assert jnp.all(jnp.isfinite(info.obs))
        assert jnp.isfinite(info.reward)
        assert not bool(info.terminated)
        assert not bool(info.truncated)

    def test_loader_applies_max_steps_and_time_observation(self):
        env = RealisticWrappers(make(self._ENV), max_steps=7, time_aware=True)
        assert env.max_steps == 7
        assert env.observation_space.shape == (16,)
        assert env.obs_layout().slice_of("elapsed_time") == slice(15, 16)
        state, init_info = env.init(jax.random.key(0))
        _, info = env.step(state, jnp.zeros(6))
        np.testing.assert_array_equal(init_info.obs[-1], 0)
        np.testing.assert_array_equal(info.obs[-1], 1)

    def test_loader_truncates_at_exact_requested_horizon(self):
        env = OracleWrappers(make(self._ENV), max_steps=3)
        state, _ = env.init(jax.random.key(0))
        flags = []
        for _ in range(3):
            state, info = env.step(state, jnp.zeros(6))
            flags.append((bool(info.terminated), bool(info.truncated)))
        assert flags == [(False, False), (False, False), (False, True)]

    def test_oracle_matches_bare_env_with_action_rescaling(self):
        bare = make(self._ENV)
        oracle = OracleWrappers(make(self._ENV))
        key = jax.random.key(0)
        action_norm = jnp.array([0.2, -0.4, 0.1, 0.3, -0.2, 0.5], dtype=jnp.float32)
        action_phys = oracle.to_physical(action_norm)

        bare_state, bare_init = bare.init(key)
        oracle_state, oracle_init = oracle.init(key)
        _assert_same_pytree_values(bare_state, unwrap_to_env_state(oracle_state))
        _assert_same_pytree_values(bare_init, oracle_init)
        assert int(bare_state.phys_params["world_model.ensemble_member"]) == -1

        bare_next, bare_step = bare.step(bare_state, action_phys)
        oracle_next, oracle_step = oracle.step(oracle_state, action_norm)
        _assert_same_pytree_values(bare_next, unwrap_to_env_state(oracle_next))
        _assert_same_pytree_values(bare_step, oracle_step)

    def test_realistic_sensor_defaults_keep_targets_clean(self):
        bare = make(self._ENV)
        layout = bare.obs_layout()
        noise = NoiseWrapper(bare)
        delay = ObsDelayWrapper(bare)
        for target in ("betap_target", "q95_target", "li_target"):
            np.testing.assert_array_equal(
                noise.noise_scale[layout.slice_of(target)], 0.0
            )
            np.testing.assert_array_equal(delay.hold_prob[layout.slice_of(target)], 0.0)
        np.testing.assert_allclose(
            noise.noise_scale[layout.slice_of("Ip")], 0.01, rtol=1e-7, atol=0.0
        )
        np.testing.assert_allclose(
            noise.noise_scale[layout.slice_of("betap")], 0.10, rtol=1e-7, atol=0.0
        )
        np.testing.assert_allclose(
            noise.noise_scale[layout.slice_of("q95")], 0.10, rtol=1e-7, atol=0.0
        )
        np.testing.assert_allclose(
            delay.hold_prob[layout.slice_of("betap")], 0.1, rtol=1e-7, atol=0.0
        )

    def test_realistic_quantize_bins_and_physical_round_trip(self):
        env = RealisticWrappers(make(self._ENV), max_steps=4, quantize_bins=5)
        assert isinstance(env.action_space, Discrete)
        np.testing.assert_array_equal(env.action_space.n, np.full(6, 5))
        np.testing.assert_allclose(
            env.to_physical(jnp.zeros(6, dtype=jnp.int32)),
            _ACT_LOW,
            rtol=1e-6,
            atol=0.0,
        )
        np.testing.assert_allclose(
            env.to_physical(jnp.full(6, 4, dtype=jnp.int32)),
            _ACT_HIGH,
            rtol=1e-6,
            atol=0.0,
        )
        np.testing.assert_array_equal(
            env.from_physical(jnp.asarray(_MIDPOINT_ACTION)),
            np.full(6, 2, dtype=np.int32),
        )
        state, _ = env.init(jax.random.key(0))
        _, info = jax.jit(env.step)(state, jnp.full(6, 2, dtype=jnp.int32))
        assert jnp.all(jnp.isfinite(info.obs))
        assert jnp.isfinite(info.reward)

    @pytest.mark.parametrize("max_steps", [0, -1, 101])
    def test_loader_rejects_invalid_or_unsafe_horizon(self, max_steps):
        with pytest.raises(ValueError, match="max_steps"):
            RealisticWrappers(make(self._ENV), max_steps=max_steps)

    @pytest.mark.parametrize("max_steps", [True, 1.5])
    def test_loader_rejects_nonintegral_requested_horizon(self, max_steps):
        with pytest.raises(ValueError, match="max_steps.*integer"):
            RealisticWrappers(make(self._ENV), max_steps=max_steps)

    @pytest.mark.parametrize(
        "kwargs, message",
        [
            ({"reward": "P_diff"}, "native reward"),
            ({"backend": "qlknn"}, "standalone"),
        ],
    )
    def test_loader_rejects_torax_only_options(self, kwargs, message):
        with pytest.raises(ValueError, match=message):
            RealisticWrappers(make(self._ENV, **kwargs))


class WorldModelWrappersTest:
    """Every Envelope wrapper and physics randomization path works on KSTAR."""

    def test_physics_randomization_samples_ensemble_members_per_step(self):
        bare = make("kstar_worldmodel")
        env = PhysicsRandomizationWrapper(bare)
        key = jax.random.key(3)
        state, init_info = env.init(key)
        assert (
            int(unwrap_to_env_state(state).phys_params["world_model.ensemble_member"])
            == -1
        )

        members = []
        cur = state
        for _ in range(8):
            cur, step_info = env.step(cur, jnp.asarray(_MIDPOINT_ACTION))
            member = int(
                unwrap_to_env_state(cur).phys_params["world_model.ensemble_member"]
            )
            assert 0 <= member < 10
            members.append(member)
            assert jnp.all(jnp.isfinite(step_info.obs))
        assert len(set(members)) > 1

        # First randomized transition matches fixing that member via with_physics.
        first_state, first_info = env.step(state, jnp.asarray(_MIDPOINT_ACTION))
        sampled_member = unwrap_to_env_state(first_state).phys_params[
            "world_model.ensemble_member"
        ]
        bare_state, _ = bare.init(key)
        pinned = bare.with_physics(
            bare_state, {"world_model.ensemble_member": sampled_member}
        )
        pinned_next, pinned_info = bare.step(pinned, jnp.asarray(_MIDPOINT_ACTION))
        _assert_same_pytree_values(unwrap_to_env_state(first_state), pinned_next)
        _assert_same_pytree_values(first_info, pinned_info)

        # Reset restores nominal (-1) and reproduces the exact same first step.
        reset_state, reset_info = env.reset(cur, key)
        assert (
            int(
                unwrap_to_env_state(reset_state).phys_params[
                    "world_model.ensemble_member"
                ]
            )
            == -1
        )
        _assert_same_pytree_values(reset_info, init_info)
        again_state, again_info = env.step(reset_state, jnp.asarray(_MIDPOINT_ACTION))
        _assert_same_pytree_values(again_state, first_state)
        _assert_same_pytree_values(again_info, first_info)

    def test_with_physics_supports_exogenous_inputs_and_rejects_actuated_inputs(self):
        bundle = load_bundle()
        env = WorldModelEnv.from_bundle(
            bundle,
            random_target=False,
            physics_randomization={
                "inputs.Bt_T": PhysicsRandomizationSpec(relative=(0.9, 1.1)),
                "inputs.GW_fraction": PhysicsRandomizationSpec(relative=(0.85, 1.15)),
            },
        )
        wrapped = ActionRescaleWrapper(env)
        state, _ = wrapped.init(jax.random.key(0))
        nom_next, nom_info = wrapped.step(state, jnp.zeros(6))
        perturbed = wrapped.with_physics(state, {"inputs.Bt_T": jnp.float32(1.98)})
        pert_next, pert_info = wrapped.step(perturbed, jnp.zeros(6))
        # Raising toroidal field Bt raises q95 on the same plasma current.
        assert float(pert_info.obs[7]) > float(nom_info.obs[7])
        # Persistence across steps until reset restores nominal Bt_T=1.8.
        np.testing.assert_allclose(
            unwrap_to_env_state(pert_next).phys_params["inputs.Bt_T"],
            1.98,
            rtol=1e-6,
            atol=0.0,
        )
        reset_state, _ = wrapped.reset(pert_next, jax.random.key(0))
        np.testing.assert_allclose(
            unwrap_to_env_state(reset_state).phys_params["inputs.Bt_T"],
            1.8,
            rtol=1e-6,
            atol=0.0,
        )

        # PhysicsRandomizationWrapper mixes continuous and discrete specs cleanly.
        mixed = PhysicsRandomizationWrapper(
            WorldModelEnv.from_bundle(
                bundle,
                random_target=False,
                physics_randomization={
                    "world_model.ensemble_member": PhysicsRandomizationSpec(
                        choices=(0, 1, 2)
                    ),
                    "inputs.Bt_T": PhysicsRandomizationSpec(relative=(0.9, 1.1)),
                },
            )
        )
        m_state, _ = jax.jit(mixed.init)(jax.random.key(0))
        m_next, _ = jax.jit(mixed.step)(m_state, jnp.asarray(_MIDPOINT_ACTION))
        m_phys = unwrap_to_env_state(m_next).phys_params
        assert int(m_phys["world_model.ensemble_member"]) in (0, 1, 2)
        assert 1.8 * 0.9 <= float(m_phys["inputs.Bt_T"]) <= 1.8 * 1.1

        with pytest.raises(ValueError, match="Actuated KSTAR input"):
            WorldModelEnv.from_bundle(
                bundle,
                physics_randomization={
                    "inputs.Ip_MA": PhysicsRandomizationSpec(relative=(0.9, 1.1))
                },
            )
        with pytest.raises(ValueError, match="out of range"):
            WorldModelEnv.from_bundle(
                bundle,
                physics_randomization={
                    "world_model.ensemble_member": PhysicsRandomizationSpec(
                        choices=(10,)
                    )
                },
            )

    def test_full_explicit_wrapper_stack_jit_vmap_and_collection(self):
        base = make("kstar_worldmodel")
        env = PhysicsRandomizationWrapper(base)
        env = NoiseWrapper(
            env,
            SensorNoiseConfig({"betap": 0.05, "q95": 0.05}).to_noise_scale(
                env.obs_layout()
            ),
        )
        env = ObsFilterWrapper.from_resolution_config(env)
        env = ObsFilterWrapper.from_obs_config(
            env,
            ObsFilterConfig(
                profiles=[],
                scalars=[
                    "Ip",
                    "betap",
                    "q95",
                    "li",
                    "betap_target",
                    "q95_target",
                    "li_target",
                ],
            ),
        )
        env = ObsDelayWrapper(
            env,
            ObsDelayConfig({"betap": 0.2}).to_hold_prob(env.obs_layout()),
        )
        env = CorruptedActionWrapper(env, noise_scale=0.02)
        env = ActionRescaleWrapper(env)
        env = TimeAwareWrapper(env)
        env = ObsHistoryWrapper(env, k=2)

        # Action history initializes from the physical reset setpoint in [-1, 1].
        h_state, h_info = env.init(jax.random.key(0))
        expected_init_norm = next(
            layer.from_physical(jnp.asarray(_RESET_ACTION))
            for layer in iter_wrappers(env)
            if isinstance(layer, ActionRescaleWrapper)
        )
        np.testing.assert_allclose(
            h_state.action_history,
            np.broadcast_to(expected_init_norm, (2, 6)),
            rtol=1e-6,
            atol=1e-6,
        )
        assert h_info.obs.shape == env.observation_space.shape == (2 * (8 + 6),)

        quantized = TruncationWrapper(QuantizeActionWrapper(env, (3,) * 6), max_steps=3)
        q_state, q_init = jax.jit(quantized.init)(jax.random.key(0))
        q_next, q_step = jax.jit(quantized.step)(q_state, jnp.ones(6, dtype=jnp.int32))
        assert jax.tree.structure(q_init) == jax.tree.structure(q_step)
        assert jnp.all(jnp.isfinite(q_step.obs))

        # Collection and Envelope outer wrappers (AutoReset, Vmap, ObsNorm) all compose.
        continuous = TruncationWrapper(env, max_steps=3)
        traj = collect_episode(
            lambda obs, rng: jnp.zeros(6, dtype=jnp.float32),
            continuous,
            jax.random.key(0),
            num_steps=4,
        )
        np.testing.assert_array_equal(traj.valid, np.array([True, True, True, False]))
        np.testing.assert_allclose(
            traj.action[:3],
            np.broadcast_to(_MIDPOINT_ACTION, (3, 6)),
            rtol=1e-6,
            atol=1e-6,
        )

        outer = VmapWrapper(
            AutoResetWrapper(ObservationNormalizationWrapper(continuous)),
            batch_size=4,
        )
        v_state, v_info = jax.jit(outer.init)(jax.random.key(1))
        _, v_step = jax.jit(outer.step)(v_state, jnp.zeros((4, 6), dtype=jnp.float32))
        assert v_info.obs.shape == v_step.obs.shape == (4, 2 * (8 + 6))
        assert jnp.all(jnp.isfinite(v_step.obs))

    def test_realistic_rollout_action_gradients_are_finite(self):
        env = RealisticWrappers(make("kstar_worldmodel"), max_steps=3)
        state, _ = env.init(jax.random.key(5))
        actions = jnp.asarray(
            [[0.1, -0.1, 0.05, -0.05, 0.1, -0.1], [-0.1, 0.1, -0.05, 0.05, -0.1, 0.1]],
            dtype=jnp.float32,
        )

        def rollout_return(rollout_actions: jax.Array) -> jax.Array:
            def transition(carry, action):
                next_state, info = env.step(carry, action)
                return next_state, info.reward

            _, rewards = jax.lax.scan(transition, state, rollout_actions)
            return jnp.sum(rewards)

        grad = jax.jit(jax.grad(rollout_return))(actions)
        assert grad.shape == actions.shape
        assert np.all(np.isfinite(grad))
        assert np.all(np.any(np.abs(grad) > 1e-5, axis=0))


def _neorl2_fusion_env():
    try:
        from neorl2.envs.fusion import FusionEnv

        return FusionEnv
    except ImportError:
        return None


@pytest.mark.integration
class WorldModelEnvNeoRL2ParityTest:
    """NeoRL2 learned-dynamics parity at matching physical actuator values."""

    def setup_class(self):
        FusionEnv = _neorl2_fusion_env()
        if FusionEnv is None:
            pytest.skip("neorl2 (+torch/gymnasium) not installed")
        self.ref = FusionEnv(random_target=False, max_episode_steps=100)
        self.env = WorldModelEnv(random_target=False)

    def test_reset_and_rollout_match_at_reference_actuator_values(self):
        ref_obs, _ = self.ref.reset(seed=0)
        state, info = self.env.init(jax.random.key(0))
        np.testing.assert_allclose(np.asarray(info.obs), ref_obs, rtol=1e-4, atol=1e-4)

        actions = np.random.default_rng(123).uniform(-1, 1, (20, 6)).astype(np.float32)
        for action in actions:
            ref_obs, ref_reward, *_ = self.ref.step(action)
            # NeoRL2 quantizes actuators; pass its realized physical controls directly.
            state, info = self.env.step(state, jnp.asarray(ref_obs[:6]))
            np.testing.assert_allclose(
                np.asarray(info.obs), ref_obs, rtol=1e-3, atol=1e-3
            )
            np.testing.assert_allclose(
                float(info.reward), ref_reward, rtol=1e-3, atol=1e-3
            )
