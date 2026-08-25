"""Tests for scenario YAML loading, merging, and validation."""

import contextlib
import inspect
import os
import shutil
import tempfile

import jax
import jax.numpy as jnp
import numpy as np
import pydantic
import pytest
import yaml
from envelope import (
    AutoResetWrapper,
    Continuous,
    Discrete,
    Environment,
    VmapWrapper,
)
from torax._src.test_utils import default_configs

from plasmax.environment import config as config_lib
from plasmax.environment import factory as factory_lib
from plasmax.environment import registry as registry_lib
from plasmax.environment.config import (
    ActuatorConfig,
    HistoryConfig,
    ObservationsConfig,
    ObsFilterSpec,
    ObsProfileConfig,
    ObsScalarConfig,
    PhaseInitializationConfig,
    RealisticObsConfig,
    ScenarioConfig,
    SteppingConfig,
    TaskConfig,
    parse_env_and_backend,
    scenario_to_yaml,
)
from plasmax.environment.factory import make
from plasmax.wrappers import (
    ActionRescaleWrapper,
    NoiseWrapper,
    ObsDelayWrapper,
    ObsFilterWrapper,
    ObsHistoryWrapper,
    PlasmaxTruncationWrapper,
    QuantizeActionWrapper,
    TimeAwareWrapper,
    iter_wrappers,
    unwrap_to_env_state,
)

_N_RHO = 10
_CONFIG_DIR = str(registry_lib.CONFIGS_DIR)

_CIRCULAR_TORAX = {
    **default_configs.get_default_config_dict(),
    "geometry": {"geometry_type": "circular", "n_rho": _N_RHO},
    "numerics": {"t_final": 0.2, "fixed_dt": 0.1},
    "time_step_calculator": {"calculator_type": "fixed"},
    "sources": {"generic_heat": {"P_total": 10e6, "electron_heat_fraction": 0.5}},
}

_DEFAULT_ACTUATORS = [
    ActuatorConfig(name="P_nbi", low=1e6, high=33e6, max_delta=2e6),
]

_DEFAULT_PROFILES = [
    ObsProfileConfig(name="T_e", scale=10.0),
    ObsProfileConfig(name="T_i", scale=10.0),
    ObsProfileConfig(name="n_e", scale=1e20),
    ObsProfileConfig(name="psi", scale=10.0),
    ObsProfileConfig(name="q", scale=5.0),
]

_DEFAULT_SCALARS = [
    ObsScalarConfig(name="W_thermal", scale=1e8),
    ObsScalarConfig(name="tau_E", scale=1.0),
    ObsScalarConfig(name="P_fusion", scale=1e8),
    ObsScalarConfig(name="t", scale=1.0),
    ObsScalarConfig(name="q_min", scale=3.0),
    ObsScalarConfig(name="q95", scale=5.0),
    ObsScalarConfig(name="beta_N", scale=3.0),
    ObsScalarConfig(name="f_non_inductive", scale=1.0),
]


def _make_scenario(**kwargs) -> ScenarioConfig:
    defaults = dict(
        torax=_CIRCULAR_TORAX,
        task=TaskConfig(reward="P_diff", terminal_penalty=0.0),
        actuators=_DEFAULT_ACTUATORS,
        observations=ObservationsConfig(
            profiles=_DEFAULT_PROFILES,
            scalars=_DEFAULT_SCALARS,
        ),
    )
    defaults.update(kwargs)
    return ScenarioConfig(**defaults)


def _dump_temp_yaml(cfg: ScenarioConfig) -> str:
    """Writes cfg to a fresh temp YAML file and returns its path (caller cleans up)."""
    with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
        path = f.name
    scenario_to_yaml(cfg, path)
    return path


@contextlib.contextmanager
def _temp_yaml(cfg: ScenarioConfig):
    """Context-managed version of _dump_temp_yaml: deletes the file on exit."""
    path = _dump_temp_yaml(cfg)
    try:
        yield path
    finally:
        os.unlink(path)


class ScenarioConfigRoundTripTest:
    def test_round_trip_preserves_fields(self):
        # Full model equality: any field silently dropped or coerced by
        # serialization fails here, not just the spot-checked ones.
        cfg = _make_scenario()
        with _temp_yaml(cfg) as path:
            loaded = config_lib.parse_scenario(path)
            assert loaded == cfg

    def test_inf_max_delta_round_trips(self):
        cfg = _make_scenario(
            actuators=[ActuatorConfig(name="P_nbi", low=1e6, high=33e6)]
        )
        assert cfg.actuators[0].max_delta == float("inf")
        with _temp_yaml(cfg) as path:
            with open(path) as fh:
                raw = yaml.safe_load(fh)
            assert raw["actuators"][0]["max_delta"] == float("inf")
            loaded = config_lib.parse_scenario(path)
            assert loaded.actuators[0].max_delta == float("inf")


class PhaseInitializationConfigTest:
    def test_validates_and_freezes_snapshot_reference(self):
        config = PhaseInitializationConfig(
            path="${DATA_DIR}/initializations/state.npz",
            sha256="A" * 64,
        )

        assert config.sha256 == "a" * 64
        with pytest.raises(pydantic.ValidationError, match="frozen"):
            config.path = "other.npz"

    @pytest.mark.parametrize(
        "values",
        [
            {"path": "", "sha256": "0" * 64},
            {"path": 1, "sha256": "0" * 64},
            {"path": "state.npz", "sha256": "not-a-checksum"},
            {"path": "state.npz", "sha256": "0" * 64, "unknown": True},
        ],
    )
    def test_rejects_invalid_snapshot_reference(self, values):
        with pytest.raises(pydantic.ValidationError):
            PhaseInitializationConfig.model_validate(values)


class SteppingConfigTest:
    def test_defaults_are_one_solver_step_and_no_events(self):
        stepping = _make_scenario().stepping
        assert stepping.max_solver_substeps == 1
        assert stepping.max_event_substeps == 0

    @pytest.mark.parametrize("value", [0, -1, True])
    def test_solver_limit_must_be_positive_integer(self, value):
        with pytest.raises(pydantic.ValidationError, match="max_solver_substeps"):
            SteppingConfig(max_solver_substeps=value)

    @pytest.mark.parametrize("value", [-1, True])
    def test_event_limit_must_be_nonnegative_integer(self, value):
        with pytest.raises(pydantic.ValidationError, match="max_event_substeps"):
            SteppingConfig(max_event_substeps=value)


class LoadScenarioOracleTest:
    """One env is built once for the whole class: reset/step take state as an
    argument rather than mutating the env, so it's safe to share across tests."""

    @classmethod
    def setup_class(cls):
        cls._path = _dump_temp_yaml(_make_scenario())
        cls._env = make(cls._path)

    @classmethod
    def teardown_class(cls):
        os.unlink(cls._path)

    def test_obs_shape_oracle(self):
        state, info = self._env.init(jax.random.key(0))
        expected_size = 5 * _N_RHO + 8
        assert info.obs.shape == (expected_size,)
        assert state.steps == 0

    def test_envelope_step_returns_state_and_structured_info(self):
        state, _ = self._env.init(jax.random.key(0))
        action = jnp.zeros(self._env.action_space.shape)
        next_state, info = self._env.step(state, action)
        assert next_state.steps == 1
        assert info.obs.shape == self._env.observation_space.shape
        assert jnp.all(jnp.isfinite(info.obs))
        assert jnp.isfinite(info.reward)
        assert not bool(info.terminated)
        assert not bool(info.truncated)
        assert int(info.termination_code) == -1

    def test_loader_returns_scalar_non_autoresetting_envelope_env(self):
        assert isinstance(self._env, Environment)
        assert isinstance(self._env, PlasmaxTruncationWrapper)
        assert not any(
            isinstance(layer, (AutoResetWrapper, VmapWrapper))
            for layer in iter_wrappers(self._env)
        )
        assert isinstance(self._env.action_space, Continuous)
        assert not callable(self._env.action_space)
        assert not callable(self._env.observation_space)


class LoadScenarioRealisticTest:
    """One env is built once for the whole class (see LoadScenarioOracleTest)."""

    @classmethod
    def setup_class(cls):
        cfg = _make_scenario(
            observations=ObservationsConfig(
                profiles=_DEFAULT_PROFILES,
                scalars=_DEFAULT_SCALARS,
                realistic=RealisticObsConfig(
                    noise={"T_e": 0.05, "n_e": 0.03},
                    resolution={"T_e": 3, "n_e": 4},
                    filter=ObsFilterSpec(
                        profiles=["T_e", "n_e"], scalars=["W_thermal", "t"]
                    ),
                ),
            ),
        )
        cls._path = _dump_temp_yaml(cfg)
        cls._env = make(cls._path, variant="realistic")

    @classmethod
    def teardown_class(cls):
        os.unlink(cls._path)

    def test_obs_shape_realistic_smaller(self):
        _, info = self._env.init(jax.random.key(0))
        assert info.obs.shape == (3 + 4 + 2,)

    def test_realistic_resolution_layout(self):
        layout = self._env.obs_layout()
        assert layout.profile_names == ("T_e", "n_e")
        assert layout.scalar_names == ("W_thermal", "t")
        assert layout.profile_slices["T_e"] == slice(0, 3)
        assert layout.profile_slices["n_e"] == slice(3, 7)
        assert layout.scalar_slices["W_thermal"] == slice(7, 8)
        assert layout.scalar_slices["t"] == slice(8, 9)
        assert self._env.observation_space.shape == (9,)

    def test_step_returns_correct_shape(self):
        state, initial = self._env.init(jax.random.key(0))
        action = jnp.zeros(self._env.action_space.shape)
        _, stepped = self._env.step(state, action)
        assert initial.obs.shape == stepped.obs.shape


class VariantWrapperTest:
    """history (both variants) vs delay (realistic only) wiring in _build_env."""

    _OBS_PER_FRAME = 5 * _N_RHO + 8  # 5 profiles * n_rho + 8 scalars
    _ACT_DIM = 1  # single P_nbi actuator

    def setup_method(self):
        self._paths: list[str] = []

    def teardown_method(self):
        for p in self._paths:
            os.unlink(p)

    def _write(self, physics_randomization=None, **obs_kwargs) -> str:
        cfg = _make_scenario(
            observations=ObservationsConfig(
                profiles=_DEFAULT_PROFILES, scalars=_DEFAULT_SCALARS, **obs_kwargs
            ),
            **(
                {"physics_randomization": physics_randomization}
                if physics_randomization is not None
                else {}
            ),
        )
        path = _dump_temp_yaml(cfg)
        self._paths.append(path)
        return path

    def test_no_history_oracle_unchanged(self):
        env = make(self._write())
        _, info = env.init(jax.random.key(0))
        assert info.obs.shape == (self._OBS_PER_FRAME,)

    def test_history_applies_to_both_variants(self):
        path = self._write(history=HistoryConfig(length=3))
        expected = 3 * (self._OBS_PER_FRAME + self._ACT_DIM)
        for variant in ("oracle", "realistic"):
            env = make(path, variant=variant)
            _, info = env.init(jax.random.key(0))
            assert info.obs.shape == (expected,), variant

    def test_delay_holds_selected_sensor_only_in_realistic(self):
        path = self._write(realistic=RealisticObsConfig(delay={"T_e": 1.0}))
        oracle = make(path, variant="oracle")
        realistic = make(path, variant="realistic")
        key = jax.random.key(0)
        oracle_state, oracle_info = oracle.init(key)
        realistic_state, realistic_info = realistic.init(key)
        action = jnp.zeros(oracle.action_space.shape)
        _, oracle_next = oracle.step(oracle_state, action)
        _, realistic_next = realistic.step(realistic_state, action)
        te = oracle.obs_layout().slice_of("T_e")
        assert not jnp.allclose(oracle_next.obs[te], oracle_info.obs[te])
        np.testing.assert_array_equal(realistic_next.obs[te], realistic_info.obs[te])

    def test_delay_does_not_change_obs_shape(self):
        path = self._write(realistic=RealisticObsConfig(delay={"T_e": 0.5}))
        env = make(path, variant="realistic")
        _, info = env.init(jax.random.key(0))
        assert info.obs.shape == (self._OBS_PER_FRAME,)

    def test_oracle_step_runs_with_history(self):
        env = make(self._write(history=HistoryConfig(length=3)))
        state, initial = env.init(jax.random.key(0))
        _, stepped = env.step(state, jnp.zeros(env.action_space.shape))
        assert stepped.obs.shape == initial.obs.shape
        assert jnp.all(jnp.isfinite(stepped.obs))
        assert jnp.isfinite(stepped.reward)
        assert not bool(stepped.terminated)

    def test_delay_unknown_sensor_raises(self):
        path = self._write(realistic=RealisticObsConfig(delay={"not_a_sensor": 0.5}))
        with pytest.raises(ValueError, match="delay sensor"):
            make(path, variant="realistic")

    def test_delay_must_survive_filter(self):
        # T_e is excluded by the filter, so it cannot be a delay target.
        path = self._write(
            realistic=RealisticObsConfig(
                filter=ObsFilterSpec(profiles=["n_e"], scalars=["t"]),
                delay={"T_e": 0.5},
            )
        )
        with pytest.raises(ValueError, match="delay sensor"):
            make(path, variant="realistic")

    def test_history_length_must_be_positive(self):
        with pytest.raises(pydantic.ValidationError):
            HistoryConfig(length=0)

    def test_ablate_filter_restores_unfiltered_observations(self):
        path = self._write(
            realistic=RealisticObsConfig(
                noise={"T_e": 0.05},
                filter=ObsFilterSpec(profiles=["T_e", "n_e"], scalars=["t"]),
                delay={"T_e": 0.3},
            )
        )
        full = make(path, variant="realistic")
        ablated = make(path, variant="realistic", ablate="filter")
        assert full.observation_space.shape == (2 * _N_RHO + 1,)
        assert ablated.observation_space.shape == (self._OBS_PER_FRAME,)

    def test_ablate_delay_allows_sensor_to_update(self):
        path = self._write(
            realistic=RealisticObsConfig(
                filter=ObsFilterSpec(profiles=["T_e"], scalars=["t"]),
                delay={"T_e": 1.0},
            )
        )
        delayed = make(path, variant="realistic")
        ablated = make(path, variant="realistic", ablate="delay")
        key = jax.random.key(0)
        delayed_state, delayed_info = delayed.init(key)
        ablated_state, ablated_info = ablated.init(key)
        action = jnp.zeros(delayed.action_space.shape)
        _, delayed_next = delayed.step(delayed_state, action)
        _, ablated_next = ablated.step(ablated_state, action)
        te = delayed.obs_layout().slice_of("T_e")
        np.testing.assert_array_equal(delayed_next.obs[te], delayed_info.obs[te])
        assert not jnp.allclose(ablated_next.obs[te], ablated_info.obs[te])

    def test_ablate_unknown_target_raises(self):
        path = self._write()
        with pytest.raises(ValueError, match="unknown ablate target"):
            make(path, variant="realistic", ablate="bogus")

    def test_ablate_rejected_for_oracle(self):
        with pytest.raises(ValueError, match="only applies"):
            make(self._write(), variant="oracle", ablate="noise")

    def test_unknown_variant_raises(self):
        with pytest.raises(ValueError, match="unknown variant"):
            make(self._write(), variant="typo")

    def test_physics_randomization_only_applied_under_realistic(self):
        path = self._write(
            physics_randomization={
                "numerics.resistivity_multiplier": {"absolute": (0.5, 1.5)}
            }
        )
        oracle_state, _ = make(path, variant="oracle").init(jax.random.key(0))
        assert unwrap_to_env_state(oracle_state).phys_params == {}

        realistic = make(path, variant="realistic")
        state, _ = realistic.init(jax.random.key(0))
        state, _ = realistic.step(state, jnp.zeros(realistic.action_space.shape))
        value = float(
            unwrap_to_env_state(state).phys_params["numerics.resistivity_multiplier"]
        )
        assert 0.5 <= value <= 1.5

    def test_realistic_physics_params_reproducible_per_transition(self):
        path = self._write(
            physics_randomization={
                "numerics.resistivity_multiplier": {"absolute": (0.5, 1.5)}
            }
        )
        env = make(path, variant="realistic")
        initial_state, _ = env.init(jax.random.key(3))
        action = jnp.zeros(env.action_space.shape)
        s1, info1 = env.step(initial_state, action)
        s2, info2 = env.step(initial_state, action)
        np.testing.assert_array_equal(
            unwrap_to_env_state(s1).phys_params["numerics.resistivity_multiplier"],
            unwrap_to_env_state(s2).phys_params["numerics.resistivity_multiplier"],
        )
        other_initial_state, _ = env.init(jax.random.key(4))
        s3, _ = env.step(other_initial_state, action)
        assert not jnp.allclose(
            unwrap_to_env_state(s3).phys_params["numerics.resistivity_multiplier"],
            unwrap_to_env_state(s1).phys_params["numerics.resistivity_multiplier"],
        )
        np.testing.assert_array_equal(info1.obs, info2.obs)
        assert jnp.all(jnp.isfinite(info2.obs))
        assert jnp.isfinite(info2.reward)

    def test_full_wrapper_stack_has_contract_order(self):
        path = self._write(
            history=HistoryConfig(length=2),
            realistic=RealisticObsConfig(
                noise={"T_e": 0.05},
                resolution={"T_e": 3, "n_e": 4},
                filter=ObsFilterSpec(
                    profiles=["T_e", "n_e"], scalars=["W_thermal", "t"]
                ),
                delay={"T_e": 0.5},
            ),
        )
        env = make(
            path,
            variant="realistic",
            time_aware=True,
            quantize_bins=3,
        )
        layers = list(iter_wrappers(env))
        assert [type(layer) for layer in layers[:-1]] == [
            PlasmaxTruncationWrapper,
            QuantizeActionWrapper,
            ObsHistoryWrapper,
            TimeAwareWrapper,
            ActionRescaleWrapper,
            ObsDelayWrapper,
            ObsFilterWrapper,
            ObsFilterWrapper,
            NoiseWrapper,
        ]
        assert layers[-1] is env.unwrapped
        assert env.obs_layout().profile_names == ("T_e", "n_e")
        assert env.obs_layout().scalar_names[-1] == "elapsed_time"


class ShippedConfigSmokeTest:
    """Smoke tests for YAML configs shipped with the package."""

    def test_load_scenario_test_yaml(self):
        env = make(os.path.join(_CONFIG_DIR, "test.yaml"))
        _, info = env.init(jax.random.key(0))
        assert env.action_space.shape == (2,)
        assert info.obs.shape == (2 * 10 + 4,)
        assert jnp.all(jnp.isfinite(info.obs))

    def test_load_env_iter_hybrid_cgm_smoke(self):
        # Load a packaged phase task through the conventional CGM backend.
        env = make(
            os.path.join(_CONFIG_DIR, "envs", "iter", "hybrid", "flattop.yaml"),
            os.path.join(_CONFIG_DIR, "backends", "cgm.yaml"),
            max_steps=2,
        )
        _, info = env.init(jax.random.key(0))
        assert isinstance(env, PlasmaxTruncationWrapper)
        assert env.max_steps == 2
        assert env.action_space.shape == (4,)
        assert info.obs.shape == env.observation_space.shape == (24,)
        assert jnp.all(jnp.isfinite(info.obs))

    def test_load_env_iter_hybrid_realistic_randomization_smoke(self):
        # variant="realistic" samples every path the merged
        # physics_randomization block declares, within its declared range
        # (device and backend paths, not a hardcoded two-variable list).
        env_path = os.path.join(_CONFIG_DIR, "envs", "iter", "hybrid", "flattop.yaml")
        backend_path = os.path.join(_CONFIG_DIR, "backends", "cgm.yaml")
        ranges = parse_env_and_backend(env_path, backend_path).physics_randomization
        assert ranges, "iter/hybrid/flattop declares no physics_randomization"

        env = make(env_path, backend_path, variant="realistic", max_steps=2)
        state, initial = env.init(jax.random.key(0))
        assert jnp.all(jnp.isfinite(initial.obs))
        state, info = env.step(state, jnp.zeros(env.action_space.shape))
        assert jnp.all(jnp.isfinite(info.obs))
        inner_state = unwrap_to_env_state(state)
        assert set(inner_state.phys_params) == set(ranges)
        for path, spec in ranges.items():
            assert jnp.isfinite(inner_state.phys_params[path]), path
            assert spec.bounds[0] < spec.bounds[1]

    def test_load_env_quantize_bins(self):
        env_path = os.path.join(_CONFIG_DIR, "envs", "iter", "hybrid", "flattop.yaml")
        backend_path = os.path.join(_CONFIG_DIR, "backends", "cgm.yaml")
        env = factory_lib.make(
            env_path,
            backend=backend_path,
            variant="realistic",
            max_steps=2,
            quantize_bins=5,
        )
        space = env.action_space
        assert isinstance(space, Discrete)
        sample = space.sample(jax.random.key(0))
        assert bool(space.contains(sample))
        low = jnp.array([spec.low for spec in env.unwrapped.actuator_specs])
        high = jnp.array([spec.high for spec in env.unwrapped.actuator_specs])
        np.testing.assert_allclose(
            env.to_physical(jnp.zeros(space.shape)), low, rtol=1e-7, atol=0.0
        )
        np.testing.assert_allclose(
            env.to_physical(jnp.array(space.n) - 1),
            high,
            rtol=1e-7,
            atol=0.0,
        )
        with pytest.raises(ValueError, match="must be >= 2"):
            make(
                env_path,
                backend_path,
                variant="realistic",
                max_steps=2,
                quantize_bins=1,
            )


class LoaderMaxStepsContractTest:
    def test_default_is_derived_from_torax_safe_horizon(self):
        env = make("test")
        # Shipped test.yaml: t_initial=0, t_final=.5, fixed_dt=.1.
        assert isinstance(env, PlasmaxTruncationWrapper)
        assert env.max_steps == 5

    def test_default_horizon_uses_start_and_end_time(self):
        torax = {
            **_CIRCULAR_TORAX,
            "numerics": {"t_initial": 0.1, "t_final": 0.3, "fixed_dt": 0.1},
        }
        with _temp_yaml(_make_scenario(torax=torax)) as path:
            env = make(path)
        assert env.max_steps == 2

    @pytest.mark.parametrize("value", [0, -1])
    def test_max_steps_must_be_positive(self, value):
        with pytest.raises(ValueError, match="max_steps.*positive|>= 1"):
            make("test", max_steps=value)

    @pytest.mark.parametrize("value", [True, 1.5])
    def test_max_steps_must_be_integral(self, value):
        with pytest.raises(ValueError, match="max_steps.*integer"):
            make("test", max_steps=value)

    def test_max_steps_cannot_exceed_backend_safe_horizon(self):
        with pytest.raises(ValueError, match="safe horizon|configured horizon|at most"):
            make("test", max_steps=6)

    def test_shorter_caller_horizon_is_allowed(self):
        env = make("test", max_steps=1)
        state, _ = env.init(jax.random.key(0))
        _, info = env.step(state, jnp.zeros(env.action_space.shape))
        assert bool(info.truncated)
        assert not bool(info.terminated)

    def test_num_steps_keyword_has_no_compatibility_alias(self):
        with pytest.raises(TypeError, match="num_steps"):
            make("test", num_steps=1)

    def test_legacy_untyped_key_is_rejected(self):
        env = make("test")
        with pytest.raises(ValueError, match="typed|new-style|jax.random.key"):
            env.init(jax.random.PRNGKey(0))

    def test_make_uses_max_steps_name(self):
        assert "max_steps" in inspect.signature(factory_lib.make).parameters
        assert "num_steps" not in inspect.signature(factory_lib.make).parameters

    @pytest.mark.parametrize(
        "forbidden", ["autoreset", "num_envs", "normalize_observations"]
    )
    def test_loader_has_no_autoreset_vectorization_or_normalization_flags(
        self, forbidden
    ):
        assert forbidden not in inspect.signature(factory_lib.make).parameters


class LoadEnvMergeTest:
    """Tests env+backend deep merge: backend defaults, env wins on overlap."""

    def setup_method(self):
        self._tmpdir = tempfile.mkdtemp()

        # Start from the same TORAX defaults the other tests use, then split
        # into env-side scenario plumbing (geometry + profile_conditions +
        # plasma_composition + sources + numerics episode length) and
        # backend-side engine (transport + solver + time_step_calculator).
        # Use overlapping numerics.resistivity_multiplier to verify env wins.
        env_torax = {
            **default_configs.get_default_config_dict(),
            "profile_conditions": {"Ip": 10.5e6},
            "geometry": {"geometry_type": "circular", "n_rho": _N_RHO},
            "numerics": {
                "t_final": 0.2,
                "fixed_dt": 0.1,
                "resistivity_multiplier": 200,  # env-side; should win on overlap
            },
            "sources": {
                "generic_heat": {"P_total": 10e6, "electron_heat_fraction": 0.5}
            },
        }
        # Drop backend-side keys so the merge has to put them back from backend.
        for k in ("transport", "solver", "time_step_calculator"):
            env_torax.pop(k, None)

        env_yaml = {
            "task": {"reward": "P_diff", "terminal_penalty": 0.0},
            "torax": env_torax,
            "actuators": [
                {"name": "P_nbi", "low": 1e6, "high": 33e6, "max_delta": 2e6}
            ],
            "observations": {
                "profiles": [
                    {"name": p.name, "scale": p.scale} for p in _DEFAULT_PROFILES
                ],
                "scalars": [
                    {"name": s.name, "scale": s.scale} for s in _DEFAULT_SCALARS
                ],
            },
        }
        backend_yaml = {
            "physics_randomization": {
                "numerics.resistivity_multiplier": {"relative": [0.9, 1.1]}
            },
            "torax": {
                "transport": {
                    "model_name": "constant",
                    "chi_i": 1.0,
                    "chi_e": 1.0,
                    "D_e": 0.1,
                    "V_e": 0.0,
                },
                "solver": {
                    "solver_type": "linear",
                    "use_predictor_corrector": True,
                    "n_corrector_steps": 1,
                    "use_pereverzev": True,
                    "chi_pereverzev": 30,
                    "D_pereverzev": 15,
                },
                "time_step_calculator": {"calculator_type": "fixed"},
                "numerics": {
                    "resistivity_multiplier": 1,  # backend default — env overrides this
                    "max_dt": 0.5,  # backend-only key — should survive
                },
            },
        }

        self._env_path = os.path.join(self._tmpdir, "env.yaml")
        self._backend_path = os.path.join(self._tmpdir, "backend.yaml")
        with open(self._env_path, "w") as f:
            yaml.dump(env_yaml, f)
        with open(self._backend_path, "w") as f:
            yaml.dump(backend_yaml, f)

    def teardown_method(self):
        shutil.rmtree(self._tmpdir)

    def test_env_wins_on_overlap(self):
        cfg = parse_env_and_backend(self._env_path, self._backend_path, validate=False)
        # resistivity_multiplier is set on both sides; env wins.
        assert cfg.torax["numerics"]["resistivity_multiplier"] == 200

    def test_backend_keys_propagate(self):
        cfg = parse_env_and_backend(self._env_path, self._backend_path, validate=False)
        # backend-only keys come through unchanged.
        assert cfg.torax["transport"]["model_name"] == "constant"
        assert cfg.torax["solver"]["solver_type"] == "linear"
        assert cfg.torax["numerics"]["max_dt"] == 0.5

    def test_backend_physics_randomization_propagates(self):
        cfg = parse_env_and_backend(self._env_path, self._backend_path, validate=False)
        spec = cfg.physics_randomization["numerics.resistivity_multiplier"]
        assert spec.relative == (0.9, 1.1)

    def test_env_only_keys_propagate(self):
        cfg = parse_env_and_backend(self._env_path, self._backend_path, validate=False)
        # env-only keys survive the merge.
        assert cfg.torax["profile_conditions"]["Ip"] == 10.5e6
        assert cfg.torax["geometry"]["geometry_type"] == "circular"

    def test_load_env_produces_runnable_wrapper(self):
        env = make(self._env_path, self._backend_path, validate=False)
        _, info = env.init(jax.random.key(0))
        assert isinstance(env, PlasmaxTruncationWrapper)
        assert jnp.all(jnp.isfinite(info.obs))

    def test_backend_with_rl_keys_raises(self):
        bad_backend = os.path.join(self._tmpdir, "bad_backend.yaml")
        with open(bad_backend, "w") as f:
            yaml.dump(
                {"torax": {"solver": {"solver_type": "linear"}}, "actuators": []}, f
            )
        with pytest.raises(ValueError, match="unexpected top-level keys"):
            parse_env_and_backend(self._env_path, bad_backend, validate=False)


class EnvBackendValidatorTest:
    def test_valid_pair_returns_silently(self):
        config_lib.validate_env_backend("iter/hybrid/flattop", "qlknn")
        config_lib.validate_env_backend("step", "tglfnn_spherical")
        config_lib.validate_env_backend("step", "bohm_gyrobohm")

    def test_removed_newton_backend_is_rejected(self):
        with pytest.raises(ValueError, match="not compatible"):
            config_lib.validate_env_backend("iter/hybrid/flattop", "qlknn_nr")

    def test_unknown_env_raises(self):
        with pytest.raises(ValueError, match="not in valid_env_backend_combos"):
            config_lib.validate_env_backend("not_a_real_env", "not_a_backend")

    def test_disallowed_backend_raises_listing_allowed(self):
        with pytest.raises(
            ValueError, match="not compatible with backend 'qlknn'"
        ) as exc_info:
            config_lib.validate_env_backend("step", "qlknn")
        assert "bohm_gyrobohm" in str(exc_info.value)

    def test_generic_bgb_registered_for_all_torax_envs(self):
        combos = config_lib.valid_env_backend_combos()
        for env_name, backends in combos.items():
            if env_name == "kstar":
                continue
            assert "bohm_gyrobohm" in backends, env_name

    def test_compatibility_registry_is_immutable(self):
        combos = config_lib.valid_env_backend_combos()
        with pytest.raises(TypeError):
            combos["new/env"] = frozenset({"cgm"})

    def test_tglfnn_backend_registered_for_iter(self):
        # tglfnn-ukaea (multimachine variant) is valid for the conventional
        # ITER/SPARC env family. STEP uses the explicit tglfnn_spherical backend.
        config_lib.validate_env_backend("iter/hybrid/flattop", "tglfnn")
        config_lib.validate_env_backend("iter/hybrid/flattop", "tglfnn_nr")
        config_lib.validate_env_backend("step", "tglfnn_spherical")


class StepBackendMergeTest:
    """STEP's BgB calibration lives in its packaged env YAML and must
    win the merge over the generic bohm_gyrobohm backend, which supplies the
    solver. Asserted against the raw YAMLs rather than hardcoded values, so
    retuning the calibration doesn't break the test."""

    @classmethod
    def setup_class(cls):
        cls._env_path = registry_lib.resolve_env("step")
        cls._backend_path = registry_lib.resolve_backend("bohm_gyrobohm")
        with open(cls._env_path) as f:
            cls._env_torax = yaml.safe_load(f).get("torax") or {}
        with open(cls._backend_path) as f:
            cls._backend_torax = yaml.safe_load(f).get("torax") or {}
        cls._merged = parse_env_and_backend(
            cls._env_path, cls._backend_path, validate=False
        ).torax

    def test_env_side_calibration_wins_merge(self):
        # Every leaf the env declares under transport/pedestal reaches the
        # merged config verbatim (env wins on overlap with the backend).
        for section in ("transport", "pedestal"):
            env_block = self._env_torax.get(section) or {}
            assert env_block, f"step.yaml declares no {section} calibration"
            for key, value in env_block.items():
                assert self._merged[section][key] == value, (section, key)

    def test_backend_solver_survives_merge(self):
        # Solver settings the env does not override come from the backend.
        env_solver = self._env_torax.get("solver") or {}
        for key, value in (self._backend_torax.get("solver") or {}).items():
            if key not in env_solver:
                assert self._merged["solver"][key] == value, key


class TglfnnMachineConfigTest:
    """The TGLFNN `machine` variant is explicit in each backend YAML."""

    def test_conventional_backend_declares_multimachine(self):
        cfg = parse_env_and_backend("iter/hybrid/flattop", "tglfnn")
        assert cfg.torax["transport"]["machine"] == "multimachine"

    def test_nr_backend_declares_reference_solver(self):
        cfg = parse_env_and_backend("iter/hybrid/flattop", "tglfnn_nr")
        assert cfg.torax["transport"]["machine"] == "multimachine"
        assert cfg.torax["solver"]["solver_type"] == "newton_raphson"
        assert cfg.torax["solver"]["n_max_iterations"] == 100
        assert cfg.torax["solver"]["tau_min"] == 1.0e-6

        nr_randomization = cfg.physics_randomization
        assert "transport_model.collisionality_multiplier" in nr_randomization

    def test_spherical_backend_declares_step(self):
        cfg = parse_env_and_backend("step", "tglfnn_spherical")
        assert cfg.torax["transport"]["model_name"] == "tglfnn-ukaea"
        assert cfg.torax["transport"]["machine"] == "step"
        assert "chi_e_bohm_multiplier" not in cfg.torax["transport"]

    def test_non_tglfnn_backend_gets_no_machine_key(self):
        cfg = parse_env_and_backend("iter/hybrid/flattop", "qlknn")
        assert "machine" not in cfg.torax["transport"]


class ValidationTest:
    def test_unknown_profile_name_raises(self):
        with pytest.raises(pydantic.ValidationError, match="not_a_real_profile"):
            ObsProfileConfig(name="not_a_real_profile", scale=10.0)

    def test_unknown_scalar_name_raises(self):
        with pytest.raises(pydantic.ValidationError, match="not_a_real_scalar"):
            ObsScalarConfig(name="not_a_real_scalar", scale=1.0)

    @pytest.mark.parametrize("scale", [0.0, -1.0, float("inf"), float("nan")])
    def test_observation_scale_must_be_positive_and_finite(self, scale):
        with pytest.raises(pydantic.ValidationError, match="scale.*positive.*finite"):
            ObsScalarConfig(name="q95", scale=scale)

    @pytest.mark.parametrize("probability", [-0.1, 1.1, float("nan")])
    def test_delay_probability_must_be_in_unit_interval(self, probability):
        with pytest.raises(pydantic.ValidationError, match="delay probability"):
            RealisticObsConfig(delay={"q95": probability})

    @pytest.mark.parametrize("resolution", [0, -1, True])
    def test_profile_resolution_must_be_positive_integer(self, resolution):
        with pytest.raises(pydantic.ValidationError, match="resolution"):
            RealisticObsConfig(resolution={"T_e": resolution})

    @pytest.mark.parametrize("field", ["profiles", "scalars"])
    def test_filter_names_must_be_unique(self, field):
        with pytest.raises(pydantic.ValidationError, match="duplicate"):
            ObsFilterSpec(**{field: ["q95", "q95"]})

    def test_duplicate_profile_names_raises(self):
        with pytest.raises(pydantic.ValidationError, match="duplicate"):
            ObservationsConfig(
                profiles=[
                    ObsProfileConfig(name="T_e", scale=10.0),
                    ObsProfileConfig(name="T_e", scale=10.0),
                ],
                scalars=_DEFAULT_SCALARS,
            )

    def test_duplicate_scalar_names_raises(self):
        with pytest.raises(pydantic.ValidationError, match="duplicate"):
            ObservationsConfig(
                profiles=_DEFAULT_PROFILES,
                scalars=[
                    ObsScalarConfig(name="W_thermal", scale=1e8),
                    ObsScalarConfig(name="W_thermal", scale=1e8),
                ],
            )

    def test_reordered_profiles_accepted(self):
        observations = ObservationsConfig(
            profiles=[
                _DEFAULT_PROFILES[1],
                _DEFAULT_PROFILES[0],
                *_DEFAULT_PROFILES[2:],
            ],
            scalars=_DEFAULT_SCALARS,
        )
        with _temp_yaml(_make_scenario(observations=observations)) as path:
            env = make(path)
            state, info = env.init(jax.random.key(0))
        core_state = unwrap_to_env_state(state)
        layout = env.obs_layout()
        np.testing.assert_allclose(
            info.obs[layout.slice_of("T_i")],
            core_state.plasma.core.T_i.value / 10.0,
            rtol=1e-6,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            info.obs[layout.slice_of("T_e")],
            core_state.plasma.core.T_e.value / 10.0,
            rtol=1e-6,
            atol=1e-6,
        )

    def test_unknown_filter_profile_raises(self):
        cfg = _make_scenario(
            observations=ObservationsConfig(
                profiles=_DEFAULT_PROFILES,
                scalars=_DEFAULT_SCALARS,
                realistic=RealisticObsConfig(
                    filter=ObsFilterSpec(profiles=["bad_profile_name"]),
                ),
            ),
        )
        with (
            _temp_yaml(cfg) as path,
            pytest.raises(ValueError, match="bad_profile_name"),
        ):
            make(path)
