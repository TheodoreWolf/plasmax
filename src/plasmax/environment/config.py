"""Parse validated plasmax environment configuration."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from torax._src.torax_pydantic import model_config as torax_model_config

from plasmax.environment import registry
from plasmax.environment.merge import (
    _deep_merge,
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


def _apply_imas_init(torax: Mapping[str, Any]) -> dict[str, Any]:
    """Resolves a top-level ``_imas_init`` directive in a torax block."""
    result = copy.deepcopy(dict(torax))
    directive = result.pop("_imas_init", None)
    if directive is None:
        return result
    if not isinstance(directive, Mapping):
        raise ValueError("torax._imas_init must be a mapping")
    unknown = sorted(
        set(directive)
        - {
            "path",
            "profile_conditions",
            "plasma_composition",
            "explicit_convert",
        }
    )
    if unknown:
        raise ValueError(f"Unknown torax._imas_init fields: {unknown}")
    path_value = directive.get("path")
    if not isinstance(path_value, str) or not path_value:
        raise ValueError("torax._imas_init.path must be a non-empty path")
    path = Path(path_value)
    if not path.is_file():
        raise FileNotFoundError(f"IMAS initialization file does not exist: {path}")
    for name in ("profile_conditions", "plasma_composition", "explicit_convert"):
        if not isinstance(directive.get(name, False), bool):
            raise ValueError(f"torax._imas_init.{name} must be a boolean")
    if not any(
        directive.get(name, False)
        for name in ("profile_conditions", "plasma_composition")
    ):
        return result

    # Imports are intentionally lazy: registry, pair, and public option errors
    # are reported before opening a comparatively expensive IMAS artifact.
    from torax._src.imas_tools.input import core_profiles, loader

    explicit_convert = directive.get("explicit_convert", False)
    ids = loader.load_imas_data(
        uri=path.name,
        ids_name="core_profiles",
        directory=path.parent,
        explicit_convert=explicit_convert,
    )
    t_initial = (result.get("numerics") or {}).get("t_initial")
    if directive.get("profile_conditions", False):
        imported = core_profiles.profile_conditions_from_IMAS(ids, t_initial)
        # Some public IMAS releases omit core_profiles/global_quantities/ip
        # while carrying the authoritative current on the equilibrium slice.
        imported_ip = imported.get("Ip")
        ip_values = imported_ip[1] if isinstance(imported_ip, tuple) else imported_ip
        if ip_values is None or np.asarray(ip_values).size == 0:
            equilibrium = loader.load_imas_data(
                uri=path.name,
                ids_name="equilibrium",
                directory=path.parent,
                explicit_convert=explicit_convert,
            )
            imported = dict(imported)
            imported["Ip"] = float(
                np.asarray(equilibrium.time_slice[0].global_quantities.ip)
            )
        explicit = result.get("profile_conditions") or {}
        if not isinstance(explicit, Mapping):
            raise ValueError("torax.profile_conditions must be a mapping")
        result["profile_conditions"] = _deep_merge(imported, explicit)
    if directive.get("plasma_composition", False):
        imported = core_profiles.plasma_composition_from_IMAS(ids, t_initial)
        explicit = result.get("plasma_composition") or {}
        if not isinstance(explicit, Mapping):
            raise ValueError("torax.plasma_composition must be a mapping")
        result["plasma_composition"] = _deep_merge(imported, explicit)
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
        return WorldModelConfig.model_validate(raw)

    assert backend is not None
    raw = _merge_env_and_backend(env, backend)
    raw = _resolve_assets(raw)
    raw = _without_metadata(raw)
    torax = raw.get("torax")
    if not isinstance(torax, Mapping):
        raise ValueError(f"Environment {env!r} must define a TORAX mapping")
    raw["torax"] = torax_model_config.ToraxConfig.from_dict(_apply_imas_init(torax))
    raw["environment_key"] = env
    return PlasmaxConfig.model_validate(raw)


__all__ = ["parse_env_and_backend"]
