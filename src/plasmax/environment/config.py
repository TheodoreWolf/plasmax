"""Parse validated plasmax environment configuration."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from torax._src.torax_pydantic import model_config as torax_model_config

from plasmax.environment import registry
from plasmax.environment.initialization_data import (
    KstarInitialization,
    ToraxInitialization,
    load_initialization,
)
from plasmax.environment.merge import (
    _merge_env_and_backend,
    load_env_layers,
    validate_env_backend,
)
from plasmax.environment.schema import PlasmaxConfig, WorldModelConfig

# Metadata keys carried in env/scenario YAMLs that name a merge layer rather
# than contribute config; stripped from the final validated output.
_METADATA_KEYS = frozenset({"tokamak", "scenario", "phase", "reset_reference"})


def _resolve_assets(value: Any) -> Any:
    """Expand packaged path variables recursively without changing YAML shape."""
    if isinstance(value, Mapping):
        return {key: _resolve_assets(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_assets(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_resolve_assets(item) for item in value)
    if isinstance(value, str):
        return value.replace("${DATA_DIR}", str(registry.CONFIGS_DIR / "data"))
    return value


def _load_initialization(
    raw: Mapping[str, Any], env: str
) -> tuple[Path, ToraxInitialization | KstarInitialization]:
    path = raw.get("initialization")
    if not isinstance(path, str) or not path:
        raise ValueError(f"Environment {env!r} requires an initialization YAML path")
    source = Path(path)
    if not source.is_absolute():
        source = registry.CONFIGS_DIR / source
    source = source.resolve()
    return source, load_initialization(
        source, kind="kstar" if env == "kstar_worldmodel" else "torax"
    )


def apply_initialization(
    torax: Mapping[str, Any],
    document: ToraxInitialization,
) -> dict[str, Any]:
    """Apply atomic resolved profiles after all task/backend layers are composed."""
    result = copy.deepcopy(dict(torax))
    conditions = dict(result.get("profile_conditions") or {})
    owned = {
        "T_i",
        "T_e",
        "n_e",
        "psi",
        "nbar",
        "normalize_n_e_to_nbar",
        "n_e_nbar_is_fGW",
        "initial_psi_mode",
        "initial_psi_from_j",
    }
    duplicate = owned.intersection(conditions)
    if duplicate:
        raise ValueError(
            "initialization profiles must live in the initialization YAML: "
            f"{sorted(duplicate)}"
        )
    conditions.update(document.profile_conditions())
    result["profile_conditions"] = conditions
    if document.composition is not None:
        composition = dict(result.get("plasma_composition") or {})
        composition.update(document.composition.torax_parameters())
        result["plasma_composition"] = composition
    return result


def _without_metadata(raw: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in raw.items() if key not in _METADATA_KEYS}


def parse_env_and_backend(
    env: str,
    backend: str | None = None,
) -> PlasmaxConfig | WorldModelConfig:
    """Load and fully validate one registered environment configuration."""
    validate_env_backend(env, backend)

    if env == "kstar_worldmodel":
        raw = _resolve_assets(load_env_layers(env))
        raw = {"environment_key": env, **_without_metadata(raw)}
        raw["initialization"], initial = _load_initialization(raw, env)
        assert isinstance(initial, KstarInitialization)
        config = WorldModelConfig.model_validate(raw)
        config._initial_state = initial
        return config

    assert backend is not None
    raw = _merge_env_and_backend(env, backend)
    raw = _resolve_assets(raw)
    raw = _without_metadata(raw)
    torax = raw.get("torax")
    if not isinstance(torax, Mapping):
        raise ValueError(f"Environment {env!r} must define a TORAX mapping")
    raw["initialization"], initial = _load_initialization(raw, env)
    assert isinstance(initial, ToraxInitialization)
    raw["torax"] = torax_model_config.ToraxConfig.from_dict(
        apply_initialization(torax, initial)
    )
    raw["environment_key"] = env
    config = PlasmaxConfig.model_validate(raw)
    config._initial_state = initial
    return config


__all__ = ["parse_env_and_backend", "apply_initialization"]
