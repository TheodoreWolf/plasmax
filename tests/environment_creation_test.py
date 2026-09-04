"""Clean-slate registered environment creation contracts."""

from __future__ import annotations

import inspect
from types import MappingProxyType

import jax
import jax.numpy as jnp
import numpy as np
import pydantic
import pytest
from envelope import AutoResetWrapper, Continuous, Discrete, Environment, VmapWrapper
from torax._src.torax_pydantic import model_config as torax_model_config

import plasmax
from plasmax.environment import schema as schema_lib
from plasmax.environment.config import parse_env_and_backend
from plasmax.environment.factory import make
from plasmax.environment.merge import (
    _deep_merge,
    _load_extended_yaml,
    _merge_env_and_backend,
    _union_no_duplicate,
    load_backend,
    load_env_layers,
    valid_env_backend_combos,
    validate_env_backend,
)
from plasmax.environment.registry import (
    BACKEND_ALIASES,
    CONFIGS_DIR,
    ENV_ALIASES,
    resolve_backend,
    resolve_env,
)
from plasmax.environment.schema import (
    HistoryConfig,
    ObservationsConfig,
    ObsFilterSpec,
    ObsProfileConfig,
    ObsScalarConfig,
    PhaseInitializationConfig,
    PlasmaxConfig,
    RealisticObsConfig,
    SteppingConfig,
    TaskConfig,
    WorldModelConfig,
)
from plasmax.wrappers import (
    ActionRescaleWrapper,
    NoiseWrapper,
    ObsDelayWrapper,
    ObsFilterWrapper,
    PlasmaxTruncationWrapper,
    QuantizeActionWrapper,
    TimeAwareWrapper,
    iter_wrappers,
)

_MOCK_ENV = "mock/circular/smoke"
_MOCK_BACKEND = "mock"
_STEP_ENV = "step/spp_001_ec_hd/flattop"
_CONVENTIONAL_ENVS = frozenset(
    {
        *(
            f"iter/{scenario}/{phase}"
            for scenario in ("advanced", "baseline", "hybrid")
            for phase in ("rampup", "flattop", "rampdown")
        ),
        *(
            f"sparc/{scenario}/{phase}"
            for scenario in ("prd", "reduced_field")
            for phase in ("rampup", "flattop", "rampdown")
        ),
    }
)
_STANDARD_BACKENDS = frozenset({"cgm", "qlknn", "bohm_gyrobohm", "tglfnn", "tglfnn_nr"})


class PublicApiTest:
    def test_make_has_exact_clean_slate_signature(self):
        signature = inspect.signature(plasmax.make)
        assert tuple(signature.parameters) == (
            "env",
            "backend",
            "reward",
            "variant",
            "disruption_penalty",
            "max_steps",
            "time_aware",
            "quantize_bins",
        )
        assert (
            signature.parameters["env"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        )
        assert (
            signature.parameters["backend"].kind
            is inspect.Parameter.POSITIONAL_OR_KEYWORD
        )
        for name in tuple(signature.parameters)[2:]:
            assert signature.parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
        assert signature.parameters["backend"].default is None
        assert signature.parameters["reward"].default is None
        assert signature.parameters["disruption_penalty"].default is None
        assert signature.parameters["variant"].default == "realistic"
        assert signature.parameters["max_steps"].default is None
        assert signature.parameters["time_aware"].default is False
        assert signature.parameters["quantize_bins"].default is None

    def test_backend_can_be_positional(self):
        assert make(_MOCK_ENV, _MOCK_BACKEND) is not None

    def test_removed_keywords_have_no_compatibility_aliases(self):
        for keyword in (
            "validate",
            "ablate",
            "num_steps",
            "autoreset",
            "num_envs",
            "normalize_observations",
        ):
            assert keyword not in inspect.signature(make).parameters

    def test_public_config_surface_has_one_torax_schema(self):
        assert plasmax.PlasmaxConfig is PlasmaxConfig
        assert "PlasmaxConfig" in schema_lib.__all__
        assert schema_lib.TaskConfig is TaskConfig
        assert "TaskConfig" in schema_lib.__all__
        for removed in ("ScenarioConfig", "BackendConfig"):
            assert not hasattr(plasmax, removed)
            assert not hasattr(schema_lib, removed)

    def test_raw_paths_are_not_constructor_inputs(self):
        raw_path = str(resolve_env(_MOCK_ENV))
        with pytest.raises(ValueError, match="Unknown environment"):
            make(raw_path, backend=_MOCK_BACKEND)


class EnvBackendValidatorTest:
    def test_exact_env_and_backend_matrix(self):
        matrix = valid_env_backend_combos()
        assert len(_CONVENTIONAL_ENVS) == 15
        assert set(ENV_ALIASES) == {
            *_CONVENTIONAL_ENVS,
            _STEP_ENV,
            _MOCK_ENV,
            "kstar_worldmodel",
        }
        assert set(BACKEND_ALIASES) == {
            *_STANDARD_BACKENDS,
            "bohm_gyrobohm_step",
            "tglfnn_spherical",
            _MOCK_BACKEND,
        }
        for env in _CONVENTIONAL_ENVS:
            assert matrix[env] == _STANDARD_BACKENDS
        assert matrix[_STEP_ENV] == frozenset(
            {"bohm_gyrobohm_step", "tglfnn_spherical"}
        )
        assert matrix[_MOCK_ENV] == frozenset({_MOCK_BACKEND})
        assert matrix["kstar_worldmodel"] == frozenset()
        assert sum(map(len, matrix.values())) == 78

    def test_compatibility_registry_is_immutable(self):
        assert isinstance(ENV_ALIASES, MappingProxyType)
        assert isinstance(BACKEND_ALIASES, MappingProxyType)
        assert isinstance(valid_env_backend_combos(), MappingProxyType)
        with pytest.raises(TypeError):
            ENV_ALIASES["new/env"] = resolve_env(_MOCK_ENV)
        with pytest.raises(TypeError):
            valid_env_backend_combos()[_MOCK_ENV] = frozenset()
        assert all(path.is_file() for path in ENV_ALIASES.values())
        assert all(path.is_file() for path in BACKEND_ALIASES.values())

    def test_registry_helpers_return_namespaced_values(self):
        assert resolve_env(_MOCK_ENV).name == "smoke.yaml"
        assert resolve_backend(_MOCK_BACKEND).name == "mock.yaml"

    def test_valid_pair_returns_silently(self):
        validate_env_backend(_MOCK_ENV, _MOCK_BACKEND)
        validate_env_backend("kstar_worldmodel", None)

    def test_pair_validation_has_no_inferred_backend(self):
        with pytest.raises(ValueError, match="requires a backend"):
            validate_env_backend(_MOCK_ENV, None)
        with pytest.raises(ValueError, match="standalone"):
            validate_env_backend("kstar_worldmodel", _MOCK_BACKEND)

    def test_disallowed_backend_raises_listing_allowed(self):
        with pytest.raises(ValueError, match="not compatible"):
            validate_env_backend(_STEP_ENV, "cgm")

    def test_unknown_env_raises(self):
        with pytest.raises(ValueError, match="Unknown environment"):
            validate_env_backend("not/an/env", None)

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            validate_env_backend(_MOCK_ENV, "not_a_backend")


class LoadEnvMergeTest:
    def test_deep_merge_replaces_lists_and_recurses_through_mappings(self):
        merged = _deep_merge(
            {"nested": {"left": 1, "items": [1, 2]}, "keep": True},
            {"nested": {"right": 2, "items": [3]}},
        )
        assert merged == {
            "nested": {"left": 1, "right": 2, "items": [3]},
            "keep": True,
        }

    def test_env_layers_apply_tokamak_then_base_then_leaf_then_wrappers(self):
        merged = load_env_layers("iter/hybrid/flattop")
        torax = merged["torax"]
        assert torax["plasma_composition"]["main_ion"] == {"D": 0.5, "T": 0.5}
        assert torax["sources"]["generic_particle"]["S_total"] == 2.05e20
        assert torax["numerics"]["t_final"] == 440.0
        assert merged["observations"]["realistic"]["resolution"] == {
            "T_e": 5,
            "T_i": 5,
            "n_e": 5,
        }
        assert [item["name"] for item in merged["actuators"]] == [
            "P_nbi",
            "P_eccd",
            "rho_eccd",
            "gas_puff_rate",
        ]
        assert merged["actuators"][0]["low"] == 0.0

    def test_env_backend_merge_is_a_disjoint_union(self):
        merged = _merge_env_and_backend("iter/hybrid/flattop", "cgm")
        assert merged["torax"]["numerics"]["t_final"] == 440.0
        assert merged["torax"]["transport"]["model_name"] == "CGM"
        assert merged["torax"]["solver"]["solver_type"] == "linear"
        assert merged["stepping"] == {"max_event_substeps": 1}

    def test_disjoint_union_rejects_duplicate_nested_leaf(self):
        with pytest.raises(ValueError, match=r"both define leaf torax\.solver\.kind"):
            _union_no_duplicate(
                {"torax": {"solver": {"kind": "env"}}},
                {"torax": {"solver": {"kind": "backend"}}},
            )

    def test_step_transport_is_owned_only_by_step_backends(self):
        env = load_env_layers(_STEP_ENV)
        assert "transport" not in env["torax"]
        assert "solver" not in env["torax"]

        bg_b = load_backend("bohm_gyrobohm_step")
        assert bg_b["torax"]["transport"]["model_name"] == "bohm-gyrobohm"
        assert bg_b["torax"]["transport"]["chi_e_bohm_multiplier"] == 0.15

        spherical = _merge_env_and_backend(_STEP_ENV, "tglfnn_spherical")
        assert spherical["torax"]["transport"]["machine"] == "step"
        assert "chi_e_bohm_multiplier" not in spherical["torax"]["transport"]

    def test_all_registered_backend_pairs_merge_without_duplicate_leaves(self):
        for env, backends in valid_env_backend_combos().items():
            for backend in backends:
                merged = _merge_env_and_backend(env, backend)
                assert "transport" in merged["torax"], (env, backend)
                assert "solver" in merged["torax"], (env, backend)

    def test_extended_yaml_inherits_and_leaf_overrides(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(
            "plasmax.environment.registry.CONFIGS_DIR", tmp_path.resolve()
        )
        base = tmp_path / "base.yaml"
        child = tmp_path / "child.yaml"
        base.write_text("nested: {left: 1, value: base}\nitems: [1, 2]\n")
        child.write_text(
            "extends: base.yaml\nnested: {right: 2, value: child}\nitems: [3]\n"
        )
        assert _load_extended_yaml(child) == {
            "nested": {"left": 1, "right": 2, "value": "child"},
            "items": [3],
        }

    def test_extended_yaml_rejects_cycles(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ):
        root = tmp_path / "configs"
        root.mkdir()
        monkeypatch.setattr("plasmax.environment.registry.CONFIGS_DIR", root)
        first = root / "first.yaml"
        second = root / "second.yaml"
        first.write_text("extends: second.yaml\n")
        second.write_text("extends: first.yaml\n")
        with pytest.raises(ValueError, match="Cyclic configuration extends"):
            _load_extended_yaml(first)

    def test_extended_yaml_rejects_path_escape(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ):
        root = tmp_path / "configs"
        root.mkdir()
        monkeypatch.setattr("plasmax.environment.registry.CONFIGS_DIR", root)
        outside = tmp_path / "outside.yaml"
        outside.write_text("value: outside\n")
        child = root / "child.yaml"
        child.write_text("extends: ../outside.yaml\n")
        with pytest.raises(ValueError, match="escapes"):
            _load_extended_yaml(child)


class FinalSchemaTest:
    def test_parse_env_and_backend_returns_frozen_final_torax_schema(self):
        config = parse_env_and_backend(_MOCK_ENV, _MOCK_BACKEND)
        assert isinstance(config, PlasmaxConfig)
        assert isinstance(config.torax, torax_model_config.ToraxConfig)
        assert config.environment_key == _MOCK_ENV
        with pytest.raises(pydantic.ValidationError, match="frozen"):
            config.environment_key = "other"

    def test_loader_strips_composition_directives_from_final_schema(self):
        config = parse_env_and_backend("iter/hybrid/flattop", "cgm")
        dumped = config.model_dump(mode="json")
        assert "tokamak" not in dumped
        assert "scenario" not in dumped
        assert "phase" not in dumped
        assert "reset_reference" not in dumped
        assert dumped["environment_key"] == "iter/hybrid/flattop"

    def test_initialization_is_asset_resolved_and_checksum_normalized(self):
        config = parse_env_and_backend("iter/hybrid/flattop", "cgm")
        assert config.initialization is not None
        assert config.initialization.path.is_absolute()
        assert config.initialization.path.is_file()
        assert config.initialization.path.is_relative_to(CONFIGS_DIR / "data")
        assert config.initialization.sha256 == config.initialization.sha256.lower()

    def test_kstar_is_one_complete_config_with_explicit_weights(self):
        config = parse_env_and_backend("kstar_worldmodel")
        assert isinstance(config, WorldModelConfig)
        assert config.world_model.name == "kstar_lstm"
        assert config.world_model.weights_path.is_absolute()
        assert config.world_model.weights_path.is_file()
        assert config.world_model.max_steps_in_episode == 100


class PhaseInitializationConfigTest:
    def test_validates_and_freezes_snapshot_reference(self):
        config = PhaseInitializationConfig(path="state.npz", sha256="A" * 64)
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
        assert SteppingConfig() == SteppingConfig(
            max_solver_substeps=1,
            max_event_substeps=0,
        )

    @pytest.mark.parametrize("value", [0, -1, True])
    def test_solver_limit_must_be_positive_integer(self, value):
        with pytest.raises(pydantic.ValidationError, match="max_solver_substeps"):
            SteppingConfig(max_solver_substeps=value)

    @pytest.mark.parametrize("value", [-1, True])
    def test_event_limit_must_be_nonnegative_integer(self, value):
        with pytest.raises(pydantic.ValidationError, match="max_event_substeps"):
            SteppingConfig(max_event_substeps=value)


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


class LoadScenarioOracleTest:
    def test_loader_returns_scalar_non_autoresetting_envelope_env(self):
        env = make(_MOCK_ENV, backend=_MOCK_BACKEND, variant="oracle")
        state, info = env.init(jax.random.key(0))
        assert isinstance(env, Environment)
        assert isinstance(env, PlasmaxTruncationWrapper)
        assert not any(
            isinstance(layer, (AutoResetWrapper, VmapWrapper))
            for layer in iter_wrappers(env)
        )
        assert isinstance(env.action_space, Continuous)
        assert env.action_space.shape == (2,)
        assert info.obs.shape == env.observation_space.shape == (24,)
        assert state.steps == 0

    def test_envelope_step_returns_state_and_structured_info(self):
        env = make(_MOCK_ENV, backend=_MOCK_BACKEND, variant="oracle")
        state, _ = env.init(jax.random.key(0))
        next_state, info = jax.jit(env.step)(state, jnp.zeros(env.action_space.shape))
        jax.block_until_ready((next_state, info))
        assert next_state.steps == 1
        assert jnp.all(jnp.isfinite(info.obs))
        assert jnp.isfinite(info.reward)


class LoaderMaxStepsContractTest:
    def test_default_is_derived_from_torax_safe_horizon(self):
        assert make(_MOCK_ENV, backend=_MOCK_BACKEND).max_steps == 5

    def test_shorter_caller_horizon_is_allowed(self):
        assert make(_MOCK_ENV, backend=_MOCK_BACKEND, max_steps=1).max_steps == 1

    def test_max_steps_cannot_exceed_backend_safe_horizon(self):
        with pytest.raises(ValueError, match="safe horizon|at most"):
            make(_MOCK_ENV, backend=_MOCK_BACKEND, max_steps=6)

    @pytest.mark.parametrize("value", [0, -1])
    def test_max_steps_must_be_positive(self, value):
        with pytest.raises(ValueError, match="max_steps"):
            make(_MOCK_ENV, backend=_MOCK_BACKEND, max_steps=value)

    @pytest.mark.parametrize("value", [True, 1.5])
    def test_max_steps_must_be_integral(self, value):
        with pytest.raises(ValueError, match="max_steps"):
            make(_MOCK_ENV, backend=_MOCK_BACKEND, max_steps=value)


class VariantWrapperTest:
    def test_full_wrapper_stack_has_contract_order(self):
        env = make(
            "iter/hybrid/rampup",
            "cgm",
            variant="realistic",
            max_steps=1,
            time_aware=True,
            quantize_bins=3,
        )
        layers = list(iter_wrappers(env))
        assert [type(layer) for layer in layers[:-1]] == [
            PlasmaxTruncationWrapper,
            QuantizeActionWrapper,
            TimeAwareWrapper,
            ActionRescaleWrapper,
            ObsDelayWrapper,
            ObsFilterWrapper,
            ObsFilterWrapper,
            NoiseWrapper,
        ]
        assert layers[-1] is env.unwrapped
        assert isinstance(env.action_space, Discrete)
        assert env.obs_layout().scalar_names[-1] == "elapsed_time"

    def test_delay_must_survive_filter(self):
        with pytest.raises(pydantic.ValidationError, match="unknown delay sensors"):
            ObservationsConfig(
                profiles=(ObsProfileConfig(name="T_e", scale=1.0),),
                scalars=(ObsScalarConfig(name="q95", scale=1.0),),
                realistic=RealisticObsConfig(
                    filter=ObsFilterSpec(scalars=("q95",)),
                    delay={"T_e": 0.5},
                ),
            )

    def test_history_length_must_be_positive(self):
        with pytest.raises(pydantic.ValidationError):
            HistoryConfig(length=0)

    def test_quantize_bins_validation(self):
        with pytest.raises(ValueError, match=">= 2|at least 2"):
            make(
                _MOCK_ENV,
                backend=_MOCK_BACKEND,
                variant="realistic",
                quantize_bins=1,
            )
        with pytest.raises(ValueError, match="realistic action degradation"):
            make(
                _MOCK_ENV,
                backend=_MOCK_BACKEND,
                variant="oracle",
                quantize_bins=3,
            )

    def test_unknown_variant_raises(self):
        with pytest.raises(ValueError, match="variant"):
            make(_MOCK_ENV, backend=_MOCK_BACKEND, variant="invalid")


class ShippedConfigSmokeTest:
    def test_load_env_quantize_bins(self):
        env = make(
            _MOCK_ENV,
            backend=_MOCK_BACKEND,
            variant="realistic",
            quantize_bins=5,
        )
        assert isinstance(env.action_space, Discrete)
        low = jnp.array([spec.low for spec in env.unwrapped.actuator_specs])
        high = jnp.array([spec.high for spec in env.unwrapped.actuator_specs])
        np.testing.assert_allclose(
            env.to_physical(jnp.zeros(env.action_space.shape)),
            low,
            rtol=1e-7,
            atol=0.0,
        )
        np.testing.assert_allclose(
            env.to_physical(jnp.array(env.action_space.n) - 1),
            high,
            rtol=1e-7,
            atol=0.0,
        )
