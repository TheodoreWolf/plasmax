"""Envelope-native contracts for TORAX spaces and wrappers.

These tests deliberately exercise the public ``Environment`` lifecycle rather than
the removed Gymnax five-tuple interface.  The real fast circular TORAX fixture is
used for action/observation transformations so the refactor cannot weaken the
existing physics-facing assertions.
"""

from __future__ import annotations

import dataclasses
import inspect

import chex
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
    WrappedState,
    Wrapper,
)
from envelope import (
    TruncationWrapper as EnvelopeTruncationWrapper,
)
from helpers import (
    N_RHO,
    NOMINAL_ACTION,
    PROFILE_OBS_SPECS,
    SCALAR_OBS_SPECS,
    make_test_config,
)
from helpers import make_test_env as _make_base_env

from plasmax.environment.config import parse_env_and_backend
from plasmax.environment.factory import _build_env
from plasmax.environment.schema import (
    ActionsConfig,
    DisruptionConfig,
    HistoryConfig,
    ObsFilterSpec,
    PhysicsRandomizationSpec,
    RealisticActionConfig,
    RealisticObsConfig,
)
from plasmax.spaces import ObsLayout
from plasmax.wrappers import (
    ActionQuantizeConfig,
    ActionRescaleWrapper,
    NoiseWrapper,
    ObsDelayConfig,
    ObsDelayWrapper,
    ObsFilterConfig,
    ObsFilterWrapper,
    ObsHistoryWrapper,
    PhysicsRandomizationWrapper,
    ProfileResolutionConfig,
    QuantizeActionWrapper,
    RealisticWrappers,
    SensorNoiseConfig,
    TimeAwareWrapper,
    TruncationWrapper,
    iter_wrappers,
    unwrap_to_env_state,
)

_DEFAULT_LAYOUT = ObsLayout.from_specs(PROFILE_OBS_SPECS, SCALAR_OBS_SPECS, N_RHO)
_ACTION = NOMINAL_ACTION


class WrapperDefaultsTest:
    @classmethod
    def setup_class(cls):
        cfg = parse_env_and_backend("mock/circular/smoke", "mock")
        cfg = cfg.model_copy(
            update={
                "observations": cfg.observations.model_copy(
                    update={
                        "realistic": RealisticObsConfig(
                            noise={"T_e": 0.1, "n_e": 0.3},
                            resolution={"T_e": 3},
                            filter=ObsFilterSpec(profiles=("T_e",), scalars=("q_min",)),
                            delay={"T_e": 0.2},
                        ),
                        "history": HistoryConfig(length=2),
                    }
                ),
                "actions": ActionsConfig(
                    realistic=RealisticActionConfig(
                        quantize={"P_nbi": 3, "gas_puff_rate": 5}
                    )
                ),
                "physics_randomization": {
                    "numerics.resistivity_multiplier": PhysicsRandomizationSpec(
                        relative=(0.5, 1.5)
                    )
                },
            }
        )
        cls.base = _build_env(cfg, reward=None, disruption_penalty=None)

    def test_defaults_resolve_against_each_inner_layout_and_action_order(self):
        noise = NoiseWrapper(self.base)
        np.testing.assert_array_equal(
            noise.noise_scale,
            SensorNoiseConfig(relative_std={"T_e": 0.1, "n_e": 0.3}).to_noise_scale(
                self.base.obs_layout()
            ),
        )
        resolution = ObsFilterWrapper.from_resolution_config(noise)
        assert resolution.obs_layout().slice_of("T_e") == slice(0, 3)
        selected = ObsFilterWrapper(resolution)
        assert selected.observation_space.shape == (4,)
        np.testing.assert_array_equal(
            NoiseWrapper(selected).noise_scale,
            jnp.asarray([0.1, 0.1, 0.1, 0.0], dtype=jnp.float32),
        )
        delay = ObsDelayWrapper(selected)
        np.testing.assert_array_equal(
            delay.hold_prob, jnp.asarray([0.2, 0.2, 0.2, 0.0], dtype=jnp.float32)
        )
        history = ObsHistoryWrapper(ActionRescaleWrapper(delay))
        assert history.k == 2
        assert QuantizeActionWrapper(history).bin_counts == (3, 5)
        assert TruncationWrapper(history).max_steps == 5

    def test_preset_matches_explicit_composition_under_jit(self):
        explicit = PhysicsRandomizationWrapper(self.base)
        explicit = NoiseWrapper(explicit)
        explicit = ObsFilterWrapper.from_resolution_config(explicit)
        explicit = ObsFilterWrapper(explicit)
        explicit = ObsDelayWrapper(explicit)
        explicit = ActionRescaleWrapper(explicit)
        explicit = TimeAwareWrapper(explicit)
        explicit = ObsHistoryWrapper(explicit)
        explicit = QuantizeActionWrapper(explicit)
        explicit = TruncationWrapper(explicit, max_steps=2)
        preset = RealisticWrappers(self.base, max_steps=2, time_aware=True)

        @jax.jit
        def transition(env, key):
            state, _ = env.init(key)
            return env.step(state, jnp.array([1, 2]))

        _assert_same_step(
            transition(explicit, jax.random.key(0)),
            transition(preset, jax.random.key(0)),
        )


def _assert_info_contract(info: Info, shape: tuple[int, ...]) -> None:
    assert isinstance(info, Info)
    assert info.obs.shape == shape
    assert jnp.asarray(info.reward).shape == ()
    assert jnp.asarray(info.terminated).shape == ()
    assert jnp.asarray(info.truncated).shape == ()
    assert jnp.asarray(info.termination_code).shape == ()


def _assert_sparse_layout_values(values, configured: dict[str, float]) -> None:
    for name, expected in configured.items():
        np.testing.assert_allclose(
            values[_DEFAULT_LAYOUT.slice_of(name)], expected, rtol=1e-7, atol=0.0
        )
    untouched = [
        n
        for n in _DEFAULT_LAYOUT.profile_names + _DEFAULT_LAYOUT.scalar_names
        if n not in configured
    ]
    for name in untouched:
        np.testing.assert_array_equal(values[_DEFAULT_LAYOUT.slice_of(name)], 0.0)


def _assert_same_step(a, b) -> None:
    state_a, info_a = a
    state_b, info_b = b
    chex.assert_trees_all_equal(state_a, state_b)
    chex.assert_trees_all_equal(info_a, info_b)


def _local_prng_key(state):
    """Returns the typed PRNG key owned by one wrapper state layer."""
    assert dataclasses.is_dataclass(state)
    leaves = []
    for field in dataclasses.fields(state):
        if field.name == "inner_state":
            continue
        leaves.extend(jax.tree.leaves(getattr(state, field.name)))
    keys = [
        leaf
        for leaf in leaves
        if hasattr(leaf, "dtype") and jnp.issubdtype(leaf.dtype, jax.dtypes.prng_key)
    ]
    assert len(keys) == 1
    return keys[0]


def _with_key_data(tree):
    """Make typed keys comparable by host-side Chex/NumPy assertions."""
    return jax.tree.map(
        lambda leaf: (
            jax.random.key_data(leaf)
            if hasattr(leaf, "dtype")
            and jnp.issubdtype(leaf.dtype, jax.dtypes.prng_key)
            else leaf
        ),
        tree,
    )


class SpaceContractTest:
    def test_base_spaces_are_envelope_continuous_properties(self):
        env = _make_base_env()
        assert isinstance(env, Environment)
        assert isinstance(env.action_space, Continuous)
        assert isinstance(env.observation_space, Continuous)
        assert env.action_space.shape == (2,)
        assert env.observation_space.shape == (_DEFAULT_LAYOUT.size,)
        assert not callable(env.action_space)
        assert not callable(env.observation_space)

    def test_spaces_contain_real_lifecycle_values(self):
        env = _make_base_env()
        state, info = env.init(jax.random.key(0))
        assert bool(env.observation_space.contains(info.obs))
        action = env.action_space.sample(jax.random.key(1))
        assert bool(env.action_space.contains(action))
        _, next_info = env.step(state, action)
        assert bool(env.observation_space.contains(next_info.obs))

    def test_legacy_local_space_types_are_gone(self):
        import plasmax.spaces as spaces_lib

        assert not hasattr(spaces_lib, "Box")
        assert not hasattr(spaces_lib, "MultiDiscrete")

    def test_quantized_space_is_array_valued_envelope_discrete(self):
        env = QuantizeActionWrapper(ActionRescaleWrapper(_make_base_env()), (2, 4))
        assert isinstance(env.action_space, Discrete)
        np.testing.assert_array_equal(env.action_space.n, jnp.array([2, 4]))
        assert env.action_space.shape == (2,)
        sample = env.action_space.sample(jax.random.key(0))
        assert bool(env.action_space.contains(sample))


class WrapperInterfaceTest:
    @pytest.mark.parametrize(
        "factory",
        [
            lambda env: NoiseWrapper(env, jnp.zeros(env.observation_space.shape)),
            lambda env: ObsFilterWrapper(
                env,
                list(range(env.observation_space.shape[0])),
                layout=env.obs_layout(),
            ),
            ActionRescaleWrapper,
            lambda env: QuantizeActionWrapper(ActionRescaleWrapper(env), (3, 5)),
            lambda env: ObsHistoryWrapper(env, k=2),
            lambda env: ObsDelayWrapper(env, jnp.zeros(env.observation_space.shape)),
            TimeAwareWrapper,
            lambda env: TruncationWrapper(env, max_steps=2),
        ],
        ids=[
            "noise",
            "filter",
            "rescale",
            "quantize",
            "history",
            "delay",
            "time",
            "truncation",
        ],
    )
    def test_all_torax_wrappers_are_envelope_wrappers(self, factory):
        wrapped = factory(_make_base_env())
        assert isinstance(wrapped, Wrapper)
        assert isinstance(wrapped, Environment)
        assert wrapped.unwrapped is wrapped.env.unwrapped

    def test_lifecycle_signatures_have_no_external_step_key(self):
        env = NoiseWrapper(
            _make_base_env(), jnp.zeros(_make_base_env().observation_space.shape)
        )
        assert tuple(inspect.signature(env.init).parameters) == ("key",)
        assert tuple(inspect.signature(env.reset).parameters) == ("state", "key")
        assert tuple(inspect.signature(env.step).parameters) == ("state", "action")

    def test_wrapper_iteration_uses_envelope_env_chain(self):
        base = _make_base_env()
        env = TimeAwareWrapper(ActionRescaleWrapper(base))
        layers = list(iter_wrappers(env))
        assert layers[0] is env
        assert layers[1] is env.env
        assert layers[2] is base


class WrapperConfigTest:
    def test_sensor_noise_config_shape_and_values(self):
        scale = SensorNoiseConfig(
            relative_std={"T_e": 0.05, "n_e": 0.03}
        ).to_noise_scale(_DEFAULT_LAYOUT)
        assert scale.shape == (_DEFAULT_LAYOUT.size,)
        _assert_sparse_layout_values(scale, {"T_e": 0.05, "n_e": 0.03})

    def test_sensor_noise_unknown_name_raises(self):
        with pytest.raises(ValueError):
            SensorNoiseConfig({"bad_sensor": 0.1}).to_noise_scale(_DEFAULT_LAYOUT)

    def test_delay_config_shape_and_values(self):
        prob = ObsDelayConfig(repeat_prob={"T_e": 0.5, "n_e": 0.2}).to_hold_prob(
            _DEFAULT_LAYOUT
        )
        assert prob.shape == (_DEFAULT_LAYOUT.size,)
        _assert_sparse_layout_values(prob, {"T_e": 0.5, "n_e": 0.2})

    def test_delay_unknown_name_raises(self):
        with pytest.raises(ValueError):
            ObsDelayConfig({"bad_sensor": 0.1}).to_hold_prob(_DEFAULT_LAYOUT)

    def test_quantize_config_orders_by_actuator_names(self):
        cfg = ActionQuantizeConfig(bins={"P_nbi": 3, "gas_puff_rate": 5})
        assert cfg.to_bin_counts(["P_nbi", "gas_puff_rate"]) == (3, 5)
        assert cfg.to_bin_counts(["gas_puff_rate", "P_nbi"]) == (5, 3)


class NoiseWrapperTest:
    @classmethod
    def setup_class(cls):
        cls.base = _make_base_env()
        cls.scale = jnp.full(cls.base.observation_space.shape, 0.1)
        cls.env = NoiseWrapper(cls.base, cls.scale)

    def test_state_nests_inner_state_and_owner_rng(self):
        state, info = self.env.init(jax.random.key(0))
        assert isinstance(state, WrappedState)
        assert hasattr(state, "inner_state")
        assert _local_prng_key(state).shape == ()
        _assert_info_contract(info, self.env.observation_space.shape)

    def test_noise_changes_obs_with_expected_multiplicative_scale(self):
        key = jax.random.key(0)
        _, noisy = self.env.init(key)
        inner_state = self.env.init(key)[0].inner_state
        _, clean = self.base.reset(inner_state, key)
        perturbation = jnp.abs(noisy.obs - clean.obs)
        assert not jnp.allclose(noisy.obs, clean.obs)
        assert jnp.all(perturbation <= 6.0 * self.scale * jnp.abs(clean.obs))

    def test_zero_noise_preserves_real_torax_transition(self):
        env = NoiseWrapper(self.base, jnp.zeros(self.base.observation_space.shape))
        key = jax.random.key(0)
        state, _ = env.init(key)
        wrapped = env.step(state, _ACTION)
        expected = self.base.step(state.inner_state, _ACTION)
        _assert_same_step((wrapped[0].inner_state, wrapped[1]), expected)

    def test_same_seed_and_action_sequence_reproduces_noise(self):
        def rollout(seed):
            state, info0 = self.env.init(jax.random.key(seed))
            state, info1 = self.env.step(state, _ACTION)
            state, info2 = self.env.step(state, _ACTION)
            return state, (info0, info1, info2)

        chex.assert_trees_all_equal(rollout(4), rollout(4))
        _, a = rollout(4)
        _, b = rollout(5)
        assert not jnp.allclose(a[1].obs, b[1].obs)

    def test_step_advances_local_key_without_external_key(self):
        state, _ = self.env.init(jax.random.key(2))
        next_state, _ = self.env.step(state, _ACTION)
        assert not jnp.array_equal(_local_prng_key(state), _local_prng_key(next_state))

    def test_reset_renews_local_rng_and_refills_from_inner_reset(self):
        state, _ = self.env.init(jax.random.key(2))
        stepped, _ = self.env.step(state, _ACTION)
        reset, reset_info = self.env.reset(stepped, jax.random.key(7))
        repeat, repeat_info = self.env.reset(stepped, jax.random.key(7))
        chex.assert_trees_all_equal((reset, reset_info), (repeat, repeat_info))
        assert not jnp.array_equal(_local_prng_key(stepped), _local_prng_key(reset))


class ObsFilterWrapperTest:
    @classmethod
    def setup_class(cls):
        cls.base = _make_base_env()
        cls.indices = list(range(cls.base.observation_space.shape[0] // 2))
        cls.layout = ObsLayout(
            profile_slices={},
            scalar_slices={"kept": slice(0, len(cls.indices))},
            profile_names=(),
            scalar_names=("kept",),
            vector_size=len(cls.indices),
        )
        cls.env = ObsFilterWrapper(cls.base, cls.indices, layout=cls.layout)

    def test_init_filters_info_obs_and_space(self):
        key = jax.random.key(0)
        _, full = self.base.init(key)
        state, filtered = self.env.init(key)
        np.testing.assert_array_equal(filtered.obs, full.obs[jnp.array(self.indices)])
        assert self.env.observation_space.shape == (len(self.indices),)
        assert isinstance(self.env.observation_space, Continuous)
        assert state.__class__ is self.base.init(key)[0].__class__

    def test_step_filters_only_observation(self):
        state, _ = self.env.init(jax.random.key(0))
        filtered_state, filtered = self.env.step(state, _ACTION)
        full_state, full = self.base.step(state, _ACTION)
        chex.assert_trees_all_equal(filtered_state, full_state)
        np.testing.assert_array_equal(filtered.obs, full.obs[jnp.array(self.indices)])
        np.testing.assert_array_equal(filtered.reward, full.reward)

    def test_named_filter_rebuilds_compact_layout(self):
        env = ObsFilterWrapper.from_obs_config(
            self.base,
            ObsFilterConfig(profiles=["T_e", "n_e"], scalars=["q95"]),
        )
        assert env.obs_layout().profile_names == ("T_e", "n_e")
        assert env.obs_layout().scalar_names == ("q95",)
        assert env.observation_space.shape == (2 * N_RHO + 1,)

    def test_profile_resolution_subsamples_in_layout_order(self):
        env = ObsFilterWrapper.from_resolution_config(
            self.base, ProfileResolutionConfig(n_obs={"T_e": 2, "n_e": 3})
        )
        state, info = env.init(jax.random.key(0))
        assert info.obs.shape == (2 + N_RHO + 3 + N_RHO + N_RHO + 8,)
        assert env.obs_layout().slice_of("T_e") == slice(0, 2)
        assert env.obs_layout().slice_of("n_e") == slice(2 + N_RHO, 5 + N_RHO)
        _, next_info = jax.jit(env.step)(state, _ACTION)
        assert next_info.obs.shape == info.obs.shape


class ActionRescaleWrapperTest:
    @classmethod
    def setup_class(cls):
        cls.base = _make_base_env()
        cls.env = ActionRescaleWrapper(cls.base)
        cls.low = cls.base.action_space.low
        cls.high = cls.base.action_space.high

    def test_action_space_is_continuous_unit_cube_property(self):
        space = self.env.action_space
        assert isinstance(space, Continuous)
        np.testing.assert_array_equal(space.low, np.full(space.shape, -1.0))
        np.testing.assert_array_equal(space.high, np.full(space.shape, 1.0))
        assert space.shape == self.base.action_space.shape

    @pytest.mark.parametrize(
        "norm_val,low_weight,high_weight",
        [(0.0, 0.5, 0.5), (-1.0, 1.0, 0.0), (1.0, 0.0, 1.0)],
        ids=["midpoint", "low", "high"],
    )
    def test_action_maps_to_same_real_torax_transition(
        self, norm_val, low_weight, high_weight
    ):
        del low_weight, high_weight
        normalized = jnp.full(2, norm_val, dtype=self.env.action_space.dtype)
        expected = self.env.to_physical(normalized)
        state, _ = self.env.init(jax.random.key(0))
        wrapped = self.env.step(state, normalized)
        physical = self.base.step(state, expected)
        _assert_same_step(wrapped, physical)

    @pytest.mark.parametrize(
        "norm_val,low_weight,high_weight",
        [(0.0, 0.5, 0.5), (-1.0, 1.0, 0.0), (1.0, 0.0, 1.0)],
    )
    def test_to_physical_maps_exact_bounds(self, norm_val, low_weight, high_weight):
        expected = low_weight * self.low + high_weight * self.high
        np.testing.assert_allclose(
            self.env.to_physical(
                jnp.full(2, norm_val, dtype=self.env.action_space.dtype)
            ),
            expected,
            rtol=2e-7,
            atol=0.0,
        )

    def test_physical_round_trip(self):
        normalized = jnp.array([-0.7, 0.3])
        physical = self.env.to_physical(normalized)
        recovered = self.env.from_physical(physical)
        np.testing.assert_allclose(recovered, normalized, rtol=1e-12, atol=1e-12)


class QuantizeActionWrapperTest:
    @classmethod
    def setup_class(cls):
        cls.base = _make_base_env()
        cls.inner = ActionRescaleWrapper(cls.base)
        cls.env = QuantizeActionWrapper(cls.inner, (3, 5))

    @pytest.mark.parametrize(
        "indices,normalized",
        [
            ((0, 0), (-1.0, -1.0)),
            ((2, 4), (1.0, 1.0)),
            ((1, 2), (0.0, 0.0)),
        ],
    )
    def test_indices_decode_to_same_real_torax_transition(self, indices, normalized):
        state, _ = self.env.init(jax.random.key(0))
        quantized = self.env.step(state, jnp.array(indices))
        continuous = self.inner.step(
            state, jnp.array(normalized, dtype=self.inner.action_space.dtype)
        )
        _assert_same_step(quantized, continuous)

    def test_physical_endpoints_and_nearest_bin_round_trip(self):
        np.testing.assert_allclose(
            self.env.to_physical(jnp.array([0, 0])),
            self.base.action_space.low,
            rtol=1e-7,
            atol=0.0,
        )
        np.testing.assert_allclose(
            self.env.to_physical(jnp.array([2, 4])),
            self.base.action_space.high,
            rtol=1e-7,
            atol=0.0,
        )
        midpoint = 0.5 * (self.base.action_space.low + self.base.action_space.high)
        bins = self.env.from_physical(midpoint)
        np.testing.assert_array_equal(bins, jnp.array([1, 2]))
        np.testing.assert_allclose(
            self.env.to_physical(bins), midpoint, rtol=2e-7, atol=0.0
        )

    @pytest.mark.parametrize("bin_counts", [(3,), (1, 5)])
    def test_invalid_bins_fail_at_construction(self, bin_counts):
        with pytest.raises(ValueError):
            QuantizeActionWrapper(self.inner, bin_counts)


class ObsHistoryWrapperTest:
    @classmethod
    def setup_class(cls):
        cls.base = _make_base_env()
        cls.k = 3
        cls.obs_dim = cls.base.observation_space.shape[0]
        cls.act_dim = cls.base.action_space.shape[0]
        cls.env = ObsHistoryWrapper(cls.base, k=cls.k)

    @property
    def stacked_size(self):
        return self.k * (self.obs_dim + self.act_dim)

    def test_invalid_length_raises(self):
        with pytest.raises(ValueError, match="k must be|length"):
            ObsHistoryWrapper(self.base, k=0)

    def test_state_is_wrapped_and_reset_refills_buffers(self):
        state, info = self.env.init(jax.random.key(0))
        assert isinstance(state, WrappedState)
        assert hasattr(state, "inner_state") and not hasattr(state, "env_state")
        assert info.obs.shape == (self.stacked_size,)
        assert self.env.observation_space.shape == (self.stacked_size,)
        inner_obs = self.base.init(jax.random.key(0))[1].obs
        for row in range(self.k):
            np.testing.assert_array_equal(state.obs_history[row], inner_obs)
        expected_action = jnp.asarray(
            unwrap_to_env_state(state).prev_action,
            dtype=state.action_history.dtype,
        )
        np.testing.assert_array_equal(
            state.action_history,
            np.broadcast_to(expected_action, (self.k, self.act_dim)),
        )

    def test_initial_action_history_uses_policy_space_setpoint(self):
        env = ObsHistoryWrapper(ActionRescaleWrapper(self.base), k=self.k)
        state, _ = env.init(jax.random.key(0))
        physical = unwrap_to_env_state(state).prev_action
        expected = env.env.from_physical(physical)
        np.testing.assert_allclose(
            state.action_history,
            np.broadcast_to(expected, state.action_history.shape),
            atol=1e-7,
            rtol=0.0,
        )

    def test_step_shifts_and_appends_policy_action(self):
        state, _ = self.env.init(jax.random.key(0))
        expected_inner, expected_info = self.base.step(state.inner_state, _ACTION)
        next_state, info = self.env.step(state, _ACTION)
        np.testing.assert_array_equal(next_state.obs_history[-1], expected_info.obs)
        np.testing.assert_array_equal(next_state.action_history[-1], _ACTION)
        chex.assert_trees_all_equal(next_state.inner_state, expected_inner)
        layout = self.env.obs_layout()
        base_slice = self.base.obs_layout().slice_of("q95")
        np.testing.assert_array_equal(
            info.obs[layout.slice_of("q95")], next_state.obs_history[-1][base_slice]
        )

    def test_explicit_reset_refills_episode_scoped_history(self):
        state, _ = self.env.init(jax.random.key(0))
        state, _ = self.env.step(state, _ACTION)
        reset_state, reset_info = self.env.reset(state, jax.random.key(3))
        for row in range(self.k):
            np.testing.assert_array_equal(
                reset_state.obs_history[row], reset_info.obs[: self.obs_dim]
            )
        expected_action = jnp.asarray(
            unwrap_to_env_state(reset_state).prev_action,
            dtype=reset_state.action_history.dtype,
        )
        np.testing.assert_array_equal(
            reset_state.action_history,
            np.broadcast_to(expected_action, reset_state.action_history.shape),
        )

    def test_jit_matches_eager(self):
        state, _ = self.env.init(jax.random.key(0))
        eager = self.env.step(state, _ACTION)
        compiled = jax.jit(self.env.step)(state, _ACTION)
        chex.assert_trees_all_close(
            _with_key_data(eager), _with_key_data(compiled), rtol=1e-6, atol=1e-9
        )


class ObsDelayWrapperTest:
    @classmethod
    def setup_class(cls):
        cls.base = _make_base_env()
        cls.obs_dim = cls.base.observation_space.shape[0]
        cls.layout = cls.base.obs_layout()

    def test_state_nests_inner_state_buffer_and_owner_rng(self):
        env = ObsDelayWrapper(self.base, jnp.zeros(self.obs_dim))
        state, info = env.init(jax.random.key(0))
        assert isinstance(state, WrappedState)
        assert hasattr(state, "inner_state") and not hasattr(state, "env_state")
        assert hasattr(state, "last_emitted_obs")
        assert _local_prng_key(state).shape == ()
        np.testing.assert_array_equal(state.last_emitted_obs, info.obs)

    def test_zero_probability_emits_fresh_real_torax_observation(self):
        env = ObsDelayWrapper(self.base, jnp.zeros(self.obs_dim))
        state, _ = env.init(jax.random.key(0))
        expected_state, expected_info = self.base.step(state.inner_state, _ACTION)
        next_state, info = env.step(state, _ACTION)
        np.testing.assert_array_equal(info.obs, expected_info.obs)
        chex.assert_trees_all_equal(next_state.inner_state, expected_state)

    def test_full_probability_holds_previous_observation(self):
        env = ObsDelayWrapper(self.base, jnp.ones(self.obs_dim))
        state, initial = env.init(jax.random.key(0))
        next_state, emitted = env.step(state, _ACTION)
        np.testing.assert_array_equal(emitted.obs, initial.obs)
        np.testing.assert_array_equal(next_state.last_emitted_obs, initial.obs)

    def test_only_configured_sensor_is_stale(self):
        hold = ObsDelayConfig({"T_e": 1.0}).to_hold_prob(self.layout)
        env = ObsDelayWrapper(self.base, hold)
        state, initial = env.init(jax.random.key(0))
        _, fresh = self.base.step(state.inner_state, _ACTION)
        _, emitted = env.step(state, _ACTION)
        te = self.layout.slice_of("T_e")
        np.testing.assert_array_equal(emitted.obs[te], initial.obs[te])
        mask = np.ones(self.obs_dim, dtype=bool)
        mask[te] = False
        np.testing.assert_array_equal(emitted.obs[mask], np.asarray(fresh.obs)[mask])

    def test_owner_key_advances_and_reset_renews_buffer_and_key(self):
        env = ObsDelayWrapper(self.base, jnp.full(self.obs_dim, 0.5))
        state, _ = env.init(jax.random.key(0))
        stepped, _ = env.step(state, _ACTION)
        assert not jnp.array_equal(_local_prng_key(state), _local_prng_key(stepped))
        reset, info = env.reset(stepped, jax.random.key(8))
        assert not jnp.array_equal(_local_prng_key(stepped), _local_prng_key(reset))
        np.testing.assert_array_equal(reset.last_emitted_obs, info.obs)

    def test_same_seed_reproduces_and_vmap_keys_are_independent(self):
        env = VmapWrapper(
            ObsDelayWrapper(self.base, jnp.full(self.obs_dim, 0.5)), batch_size=4
        )
        state_a, _ = env.init(jax.random.key(9))
        state_b, _ = env.init(jax.random.key(9))
        np.testing.assert_array_equal(
            jax.random.key_data(_local_prng_key(state_a)),
            jax.random.key_data(_local_prng_key(state_b)),
        )
        assert not jnp.array_equal(
            _local_prng_key(state_a)[0], _local_prng_key(state_a)[1]
        )
        action = jnp.broadcast_to(_ACTION, (4, 2))
        _, info_a = env.step(state_a, action)
        _, info_b = env.step(state_b, action)
        np.testing.assert_array_equal(info_a.obs, info_b.obs)


class TimeAwareWrapperTest:
    @classmethod
    def setup_class(cls):
        cls.base = _make_base_env()
        cls.env = TimeAwareWrapper(cls.base)

    def test_space_and_layout_gain_elapsed_time(self):
        space = self.env.observation_space
        assert isinstance(space, Continuous)
        assert space.shape == (self.base.observation_space.shape[0] + 1,)
        np.testing.assert_array_equal(space.low[:-1], self.base.observation_space.low)
        np.testing.assert_array_equal(space.high[:-1], self.base.observation_space.high)
        assert space.low[-1] == 0.0 and np.isinf(space.high[-1])
        assert self.env.obs_layout().slice_of("elapsed_time") == slice(
            self.base.obs_layout().size, self.base.obs_layout().size + 1
        )

    def test_init_and_step_append_elapsed_time(self):
        state, initial = self.env.init(jax.random.key(0))
        np.testing.assert_array_equal(initial.obs[-1], 0.0)
        _, info = self.env.step(state, _ACTION)
        np.testing.assert_allclose(info.obs[-1], 0.1, atol=1e-6, rtol=0.0)

    def test_elapsed_time_is_relative_to_nonzero_episode_start(self):
        base = _make_base_env(
            config=make_test_config(
                numerics={"t_initial": 5.0, "t_final": 5.2, "fixed_dt": 0.1}
            )
        )
        env = TimeAwareWrapper(base)
        state, initial = env.init(jax.random.key(0))
        np.testing.assert_array_equal(initial.obs[-1], 0.0)
        _, info = env.step(state, _ACTION)
        np.testing.assert_allclose(info.obs[-1], 0.1, atol=1e-6, rtol=0.0)

    def test_time_reads_through_nested_inner_state(self):
        env = TimeAwareWrapper(ObsHistoryWrapper(self.base, k=2))
        state, _ = env.init(jax.random.key(0))
        _, info = env.step(state, _ACTION)
        np.testing.assert_allclose(info.obs[-1], 0.1, atol=1e-6, rtol=0.0)


class TruncationWrapperTest:
    def test_subclasses_envelope_truncation_wrapper(self):
        assert issubclass(TruncationWrapper, EnvelopeTruncationWrapper)

    def test_requires_positive_max_steps(self):
        base = _make_base_env()
        for value in (0, -1):
            with pytest.raises(ValueError, match="max_steps.*positive|>= 1"):
                TruncationWrapper(base, max_steps=value)

    def test_requires_integral_nonboolean_max_steps(self):
        base = _make_base_env()
        for value in (True, 1.5):
            with pytest.raises(ValueError, match="max_steps.*integer"):
                TruncationWrapper(base, max_steps=value)

    def test_exact_cutoff_and_state_nesting(self):
        env = TruncationWrapper(_make_base_env(), max_steps=2)
        state, info = env.init(jax.random.key(0))
        assert isinstance(state, WrappedState)
        assert state.steps == 0
        assert not bool(info.truncated)
        state, info = env.step(state, _ACTION)
        assert state.steps == 1 and not bool(info.truncated)
        state, info = env.step(state, _ACTION)
        assert state.steps == 2 and bool(info.truncated)
        assert not bool(info.terminated)
        assert int(info.termination_code) == -1

    def test_reset_zeroes_only_episode_counter(self):
        env = TruncationWrapper(_make_base_env(), max_steps=2)
        state, _ = env.init(jax.random.key(0))
        state, _ = env.step(state, _ACTION)
        reset, info = env.reset(state, jax.random.key(5))
        assert reset.steps == 0
        assert not bool(info.truncated)
        assert int(info.termination_code) == -1

    def test_termination_wins_at_cutoff(self):
        base = _make_base_env(
            disruption_penalty=7.0,
            disruption=DisruptionConfig(q_min_threshold=1e6, greenwald_threshold=1e9),
        )
        env = TruncationWrapper(base, max_steps=1)
        state, _ = env.init(jax.random.key(0))
        _, info = env.step(state, _ACTION)
        assert bool(info.terminated)
        assert not bool(info.truncated)
        assert int(info.termination_code) == 1

    def test_jit_and_vmap_selective_boundaries(self):
        env = VmapWrapper(
            TruncationWrapper(_make_base_env(), max_steps=2), batch_size=2
        )
        state, _ = env.init(jax.random.key(0))
        # Put only lane 0 one step from the horizon; lane 1 remains at zero.
        state = state.replace(steps=jnp.array([1, 0], dtype=jnp.int32))
        actions = jnp.broadcast_to(_ACTION, (2, 2))
        next_state, info = jax.jit(env.step)(state, actions)
        np.testing.assert_array_equal(next_state.steps, jnp.array([2, 1]))
        np.testing.assert_array_equal(info.truncated, jnp.array([True, False]))
        np.testing.assert_array_equal(info.terminated, jnp.array([False, False]))


class CompositionTest:
    def test_stochastic_wrappers_preserve_the_physical_reset_key(self):
        base = _make_base_env()
        noise = NoiseWrapper(base, jnp.zeros(base.observation_space.shape))
        env = ObsDelayWrapper(noise, jnp.zeros(noise.observation_space.shape))

        key = jax.random.key(12)
        expected, _ = base.init(key)
        state, _ = env.init(key)
        chex.assert_trees_all_equal(unwrap_to_env_state(state), expected)

        reset_key = jax.random.key(13)
        expected_reset, _ = base.reset(expected, reset_key)
        reset, _ = env.reset(state, reset_key)
        chex.assert_trees_all_equal(unwrap_to_env_state(reset), expected_reset)

    def test_transform_stack_is_jittable_and_preserves_info_structure(self):
        base = _make_base_env()
        env = NoiseWrapper(base, jnp.zeros(base.observation_space.shape))
        env = ObsFilterWrapper.from_obs_config(
            env,
            ObsFilterConfig(profiles=["T_e", "n_e"], scalars=["t", "q95"]),
        )
        env = ObsDelayWrapper(env, jnp.zeros(env.observation_space.shape))
        env = ActionRescaleWrapper(env)
        env = TimeAwareWrapper(env)
        env = ObsHistoryWrapper(env, k=2)
        env = QuantizeActionWrapper(env, (3, 5))
        env = TruncationWrapper(env, max_steps=2)
        state, info = jax.jit(env.init)(jax.random.key(0))
        structure = jax.tree.structure(info)
        state, next_info = jax.jit(env.step)(state, jnp.array([1, 2]))
        assert jax.tree.structure(next_info) == structure
        _assert_info_contract(next_info, env.observation_space.shape)

    def test_autoreset_outside_truncation_preserves_terminal_snapshot(self):
        env = AutoResetWrapper(TruncationWrapper(_make_base_env(), max_steps=1))
        state, _ = env.init(jax.random.key(0))
        state, info = env.step(state, _ACTION)
        assert not bool(info.terminated)
        assert bool(info.truncated)
        assert bool(info.final_valid)
        assert bool(info.final.truncated)
        assert not bool(info.final.terminated)
        assert int(info.final.termination_code) == -1
        assert state.inner_state.steps == 0

    def test_normalization_is_explicit_and_outside_fixed_physical_scaling(self):
        loader_stack = TruncationWrapper(
            ActionRescaleWrapper(_make_base_env()), max_steps=2
        )
        normalized = ObservationNormalizationWrapper(loader_stack)
        key = jax.random.key(0)
        base_state, base_info = loader_stack.init(key)
        state, info = normalized.init(key)
        np.testing.assert_array_equal(info.unnormalized_obs, base_info.obs)
        assert state.inner_state.steps == base_state.steps == 0
        count = state.rmv_state.count
        reset, reset_info = normalized.reset(state, jax.random.key(1))
        assert reset.rmv_state.count > count
        assert hasattr(reset_info, "unnormalized_obs")
