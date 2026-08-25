"""Envelope contract and golden tests for the KSTAR world-model backend."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import yaml
from envelope import Continuous, Environment, Info, VmapWrapper

from plasmax.models.world_model_env import WorldModelEnv, target_tracking_reward

_GOLDEN_RESET = np.array(
    [
        0.5,
        1.7,
        0.3,
        0.75,
        1.32,
        2.22,
        1.347041,
        4.840889,
        1.079538,
        1.6,
        5.0,
        0.95,
        1.5,
        1.5,
        0.5,
    ],
    np.float32,
)
_GOLDEN_STEP0 = np.array(
    [
        0.55,
        1.8,
        0.3,
        0.7,
        1.312,
        2.235,
        1.404508,
        4.971919,
        0.869126,
        1.6,
        5.0,
        0.95,
        1.5,
        1.5,
        0.5,
    ],
    np.float32,
)
_GOLDEN_STEP0_REWARD = -0.08764740


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
        np.testing.assert_array_equal(self.env.action_space.low, -np.ones(6))
        np.testing.assert_array_equal(self.env.action_space.high, np.ones(6))
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
        state, _ = self.env.step(state, jnp.zeros(6))
        state, info = self.env.reset(state, jax.random.key(1))
        _assert_initial_info(info)
        assert int(state.t) == 0
        np.testing.assert_allclose(info.obs, _GOLDEN_RESET, rtol=1e-4, atol=1e-4)

    def test_step_golden(self):
        state, _ = self.env.init(jax.random.key(0))
        new_state, info = self.env.step(state, jnp.zeros(6))
        np.testing.assert_allclose(info.obs, _GOLDEN_STEP0, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(
            info.reward, _GOLDEN_STEP0_REWARD, rtol=1e-4, atol=1e-4
        )
        assert bool(self.env.observation_space.contains(info.obs))
        assert not bool(info.terminated)
        assert not bool(info.truncated)
        assert int(info.termination_code) == -1
        assert int(new_state.t) == 1

    def test_step_is_pure_given_state_and_action(self):
        state, _ = self.env.init(jax.random.key(0))
        action = jnp.full(6, 0.3)
        state1, info1 = self.env.step(state, action)
        state2, info2 = self.env.step(state, action)
        _assert_same_pytree_values(state1, state2)
        _assert_same_pytree_values(info1, info2)

    def test_base_environment_never_owns_horizon_truncation(self):
        state, _ = self.env.init(jax.random.key(0))
        for _ in range(4):
            state, info = self.env.step(state, jnp.zeros(6))
            assert not bool(info.terminated)
            assert not bool(info.truncated)
            assert int(info.termination_code) == -1
        assert int(state.t) == 4
        assert not hasattr(self.env, "max_steps_in_episode")

    def test_info_structure_and_dtypes_are_stable(self):
        state, init_info = self.env.init(jax.random.key(0))
        state, step_info = self.env.step(state, jnp.zeros(6))
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

    def test_reward_is_max_at_target(self):
        target = jnp.array([1.6, 5.0, 0.95])
        on = target_tracking_reward(target, target)
        off = target_tracking_reward(target + jnp.array([0.5, 1.0, 0.1]), target)
        assert jnp.isfinite(on)
        assert float(on) > float(off)


class WorldModelTransformContractTest:
    def setup_method(self):
        self.env = WorldModelEnv(random_target=False)

    def test_jit_lax_scan(self):
        state, _ = jax.jit(self.env.init)(jax.random.key(0))
        actions = jnp.zeros((5, 6))

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
        actions = jnp.zeros((8, 6))

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


class KstarLoaderTest:
    """The unified loader owns KSTAR truncation, not the learned model."""

    _ENV = "kstar"
    _BACKEND = "fusion_lstm"

    def test_backend_kind_is_world_model(self):
        from plasmax.environment.config import backend_kind

        assert backend_kind(self._BACKEND) == "world_model"

    def test_validate_env_backend(self):
        from plasmax.environment.config import validate_env_backend

        validate_env_backend(self._ENV, self._BACKEND)
        with pytest.raises(ValueError, match="not compatible"):
            validate_env_backend(self._ENV, "qlknn")

    def test_load_env_returns_scalar_envelope_environment(self):
        from plasmax.environment.factory import make

        env = make(self._ENV, self._BACKEND)
        assert isinstance(env, Environment)
        assert env.action_space.shape == (6,)
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
        from plasmax.environment.factory import make

        env = make(self._ENV, self._BACKEND, max_steps=7, time_aware=True)
        assert env.max_steps == 7
        assert env.observation_space.shape == (16,)
        assert env.obs_layout().slice_of("elapsed_time") == slice(15, 16)
        state, init_info = env.init(jax.random.key(0))
        _, info = env.step(state, jnp.zeros(6))
        np.testing.assert_array_equal(init_info.obs[-1], 0)
        np.testing.assert_array_equal(info.obs[-1], 1)

    def test_loader_truncates_at_exact_requested_horizon(self):
        from plasmax.environment.factory import make

        env = make(self._ENV, self._BACKEND, max_steps=3)
        state, _ = env.init(jax.random.key(0))
        flags = []
        for _ in range(3):
            state, info = env.step(state, jnp.zeros(6))
            flags.append((bool(info.terminated), bool(info.truncated)))
        assert flags == [(False, False), (False, False), (False, True)]

    def test_realistic_variant_uses_native_world_model_interface(self):
        from plasmax.environment.factory import make

        oracle = make(self._ENV, self._BACKEND, variant="oracle")
        realistic = make(self._ENV, self._BACKEND, variant="realistic")
        key = jax.random.key(0)
        action = jnp.zeros(6)

        oracle_state, oracle_init_info = oracle.init(key)
        realistic_state, realistic_init_info = realistic.init(key)
        _assert_same_pytree_values(oracle_state, realistic_state)
        _assert_same_pytree_values(oracle_init_info, realistic_init_info)

        oracle_next_state, oracle_step_info = oracle.step(oracle_state, action)
        realistic_next_state, realistic_step_info = realistic.step(
            realistic_state, action
        )
        _assert_same_pytree_values(oracle_next_state, realistic_next_state)
        _assert_same_pytree_values(oracle_step_info, realistic_step_info)

    @pytest.mark.parametrize("max_steps", [0, -1, 101])
    def test_loader_rejects_invalid_or_unsafe_horizon(self, max_steps):
        from plasmax.environment.factory import make

        with pytest.raises(ValueError, match="max_steps"):
            make(self._ENV, self._BACKEND, max_steps=max_steps)

    @pytest.mark.parametrize("max_steps", [True, 1.5])
    def test_loader_rejects_nonintegral_requested_horizon(self, max_steps):
        from plasmax.environment.factory import make

        with pytest.raises(ValueError, match="max_steps.*integer"):
            make(self._ENV, self._BACKEND, max_steps=max_steps)

    @pytest.mark.parametrize("safe_horizon", [True, 3.5])
    def test_loader_rejects_nonintegral_yaml_horizon(self, tmp_path, safe_horizon):
        from plasmax.environment.factory import make

        env_path = tmp_path / "kstar.yaml"
        env_path.write_text(
            yaml.safe_dump(
                {
                    "task": {"reward": "native", "terminal_penalty": None},
                    "world_model_env": {
                        "max_steps_in_episode": safe_horizon,
                        "random_target": False,
                    },
                }
            )
        )
        with pytest.raises(ValueError, match="max_steps_in_episode"):
            make(str(env_path), self._BACKEND, validate=False)

    def test_loader_requires_yaml_safe_horizon(self, tmp_path):
        from plasmax.environment.factory import make

        env_path = tmp_path / "kstar.yaml"
        env_path.write_text(
            yaml.safe_dump(
                {
                    "task": {"reward": "native", "terminal_penalty": None},
                    "world_model_env": {"random_target": False},
                }
            )
        )
        with pytest.raises(ValueError, match="max_steps_in_episode"):
            make(str(env_path), self._BACKEND, validate=False)

    @pytest.mark.parametrize("owner", ["environment", "backend"])
    def test_world_model_yaml_rejects_initialization(self, tmp_path, owner):
        from plasmax.environment.factory import make

        initialization = {"path": "state.npz", "sha256": "0" * 64}
        env: str = self._ENV
        backend: str = self._BACKEND
        if owner == "environment":
            env_path = tmp_path / "kstar.yaml"
            env_path.write_text(
                yaml.safe_dump(
                    {
                        "task": {"reward": "native", "terminal_penalty": None},
                        "world_model_env": {
                            "max_steps_in_episode": 100,
                            "random_target": False,
                        },
                        "initialization": initialization,
                    }
                )
            )
            env = str(env_path)
        else:
            backend_path = tmp_path / "fusion_lstm.yaml"
            backend_path.write_text(
                yaml.safe_dump(
                    {
                        "type": "world_model",
                        "world_model": {"name": "kstar_lstm"},
                        "initialization": initialization,
                    }
                )
            )
            backend = str(backend_path)

        with pytest.raises(ValueError, match="initialization"):
            make(env, backend, validate=False)

    def test_world_model_yaml_allows_null_initialization(self, tmp_path):
        from plasmax.environment.config import parse_world_model_sources

        env_path = tmp_path / "kstar.yaml"
        env_path.write_text(
            yaml.safe_dump(
                {
                    "task": {"reward": "native", "terminal_penalty": None},
                    "world_model_env": {"max_steps_in_episode": 100},
                    "initialization": None,
                }
            )
        )
        backend_path = tmp_path / "fusion_lstm.yaml"
        backend_path.write_text(
            yaml.safe_dump(
                {
                    "type": "world_model",
                    "world_model": {"name": "kstar_lstm"},
                    "initialization": None,
                }
            )
        )

        sources = parse_world_model_sources(
            str(env_path), str(backend_path), validate=False
        )
        assert sources.backend.name == "kstar_lstm"

    @pytest.mark.parametrize(
        "kwargs, message",
        [
            ({"reward": "P_diff"}, "native reward"),
            ({"disruption_penalty": -1.0}, "disruption_penalty"),
            ({"ablate": "noise"}, "ablations"),
            ({"quantize_bins": 5}, "quantize_bins"),
        ],
    )
    def test_loader_rejects_torax_only_options(self, kwargs, message):
        from plasmax.environment.factory import make

        with pytest.raises(ValueError, match=message):
            make(self._ENV, self._BACKEND, **kwargs)

    def test_loader_rejects_unknown_variant(self):
        from plasmax.environment.factory import make

        with pytest.raises(ValueError, match="unknown variant"):
            make(self._ENV, self._BACKEND, variant="invalid")


def _neorl2_fusion_env():
    try:
        from neorl2.envs.fusion import FusionEnv

        return FusionEnv
    except ImportError:
        return None


@pytest.mark.integration
class WorldModelEnvNeoRL2ParityTest:
    """Bit-faithful parity vs NeoRL2 with Envelope lifecycle plumbing."""

    def setup_class(self):
        FusionEnv = _neorl2_fusion_env()
        if FusionEnv is None:
            pytest.skip("neorl2 (+torch/gymnasium) not installed")
        self.ref = FusionEnv(random_target=False, max_episode_steps=100)
        self.env = WorldModelEnv(random_target=False)

    def test_reset_and_rollout_match(self):
        ref_obs, _ = self.ref.reset(seed=0)
        state, info = self.env.init(jax.random.key(0))
        np.testing.assert_allclose(np.asarray(info.obs), ref_obs, rtol=1e-4, atol=1e-4)

        actions = np.random.default_rng(123).uniform(-1, 1, (20, 6)).astype(np.float32)
        for action in actions:
            ref_obs, ref_reward, *_ = self.ref.step(action)
            state, info = self.env.step(state, jnp.asarray(action))
            np.testing.assert_allclose(
                np.asarray(info.obs), ref_obs, rtol=1e-3, atol=1e-3
            )
            np.testing.assert_allclose(
                float(info.reward), ref_reward, rtol=1e-3, atol=1e-3
            )
