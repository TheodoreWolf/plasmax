"""Env/backend YAML merge helpers and compatibility registry."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from plasmax.environment import registry

RawConfig = dict[str, Any]


def _read_mapping(path: Path) -> RawConfig:
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file does not exist: {path}")
    with path.open() as stream:
        value = yaml.safe_load(stream) or {}
    if not isinstance(value, dict):
        raise ValueError(f"Configuration {path} must contain a YAML mapping")
    return value


def _deep_merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> RawConfig:
    """Recursively merge ``overlay`` onto ``base`` (overlay wins on leaves)."""
    out = copy.deepcopy(dict(base))
    for key, value in overlay.items():
        existing = out.get(key)
        if isinstance(existing, Mapping) and isinstance(value, Mapping):
            out[key] = _deep_merge(existing, value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _safe_extended_path(source: Path, value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"extends in {source} must be a non-empty relative path")
    extended = Path(value)
    if extended.is_absolute():
        raise ValueError(f"extends in {source} must be relative")
    target = (source.parent / extended).resolve()
    config_root = registry.CONFIGS_DIR.resolve()
    if not target.is_relative_to(config_root):
        raise ValueError(f"extends in {source} escapes the packaged config tree")
    return target


def _load_extended_yaml(
    path: str | Path,
    *,
    stack: tuple[Path, ...] = (),
) -> RawConfig:
    """Load one packaged fragment and its safe relative ``extends`` chain."""
    source = Path(path).resolve()
    config_root = registry.CONFIGS_DIR.resolve()
    if not source.is_relative_to(config_root):
        raise ValueError(f"Configuration path escapes the packaged tree: {source}")
    if source in stack:
        chain = " -> ".join(str(item) for item in (*stack, source))
        raise ValueError(f"Cyclic configuration extends: {chain}")

    raw = _read_mapping(source)
    extends = raw.pop("extends", None)
    if extends is None:
        return raw
    values = [extends] if isinstance(extends, str) else extends
    if not isinstance(values, list) or not values:
        raise ValueError(f"extends in {source} must be a string or non-empty list")

    merged: RawConfig = {}
    for value in values:
        parent = _safe_extended_path(source, value)
        merged = _deep_merge(
            merged,
            _load_extended_yaml(parent, stack=(*stack, source)),
        )
    return _deep_merge(merged, raw)


def _mapping(value: object, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be a mapping")
    return value


def _validate_env_ownership(raw: Mapping[str, Any], env: str) -> None:
    torax = _mapping(raw.get("torax", {}), f"environment {env!r} torax")
    owned_by_backend = sorted(set(torax) & {"transport", "solver"})
    if owned_by_backend:
        raise ValueError(
            f"Environment {env!r} cannot define backend-owned "
            "TORAX sections: "
            f"{owned_by_backend}"
        )
    stepping = _mapping(raw.get("stepping", {}), f"environment {env!r} stepping")
    if "max_solver_substeps" in stepping:
        raise ValueError(
            f"Environment {env!r} cannot define backend-owned "
            "stepping.max_solver_substeps"
        )
    randomization = _mapping(
        raw.get("physics_randomization", {}),
        f"environment {env!r} physics_randomization",
    )
    model_paths = sorted(
        path for path in randomization if str(path).startswith("transport_model.")
    )
    if model_paths:
        raise ValueError(
            f"Environment {env!r} cannot define model-specific "
            f"physics randomization: {model_paths}"
        )


def _validate_backend_ownership(raw: Mapping[str, Any], backend: str) -> None:
    allowed_top_level = {"torax", "physics_randomization", "stepping"}
    unknown = sorted(set(raw) - allowed_top_level)
    if unknown:
        raise ValueError(
            f"Backend {backend!r} contains environment-owned fields: {unknown}"
        )

    torax = _mapping(raw.get("torax", {}), f"backend {backend!r} torax")
    unknown_torax = sorted(set(torax) - {"transport", "solver"})
    if unknown_torax:
        raise ValueError(
            f"Backend {backend!r} contains environment-owned TORAX sections: "
            f"{unknown_torax}"
        )
    for section in torax:
        _mapping(torax[section], f"backend {backend!r} torax.{section}")

    stepping = _mapping(raw.get("stepping", {}), f"backend {backend!r} stepping")
    unknown_stepping = sorted(set(stepping) - {"max_solver_substeps"})
    if unknown_stepping:
        raise ValueError(
            f"Backend {backend!r} contains environment-owned stepping fields: "
            f"{unknown_stepping}"
        )

    randomization = _mapping(
        raw.get("physics_randomization", {}),
        f"backend {backend!r} physics_randomization",
    )
    non_model_paths = sorted(
        path for path in randomization if not str(path).startswith("transport_model.")
    )
    if non_model_paths:
        raise ValueError(
            f"Backend {backend!r} contains non-model physics randomization: "
            f"{non_model_paths}"
        )


def _union_no_duplicate(
    env: Mapping[str, Any],
    backend: Mapping[str, Any],
    *,
    prefix: tuple[str, ...] = (),
) -> RawConfig:
    """Union orthogonal nested mappings and reject every duplicate leaf."""
    result = copy.deepcopy(dict(env))
    for key, value in backend.items():
        if key not in result:
            result[key] = copy.deepcopy(value)
            continue
        existing = result[key]
        path = (*prefix, str(key))
        if isinstance(existing, Mapping) and isinstance(value, Mapping):
            result[key] = _union_no_duplicate(existing, value, prefix=path)
            continue
        raise ValueError("Environment and backend both define leaf " + ".".join(path))
    return result


def load_env_layers(env: str) -> RawConfig:
    """Compose ``tokamak < sibling base < phase leaf < wrappers``."""
    leaf_path = registry.resolve_env(env)
    leaf = _load_extended_yaml(leaf_path)
    if env == "kstar_worldmodel":
        return leaf

    base_path = leaf_path.parent / "base.yaml"
    base = _load_extended_yaml(base_path) if base_path.is_file() else {}
    base_tokamak = base.get("tokamak")
    leaf_tokamak = leaf.get("tokamak")
    if (
        base_tokamak is not None
        and leaf_tokamak is not None
        and base_tokamak != leaf_tokamak
    ):
        raise ValueError(
            f"Environment {env!r} conflicts with its sibling base on tokamak"
        )
    tokamak = leaf_tokamak if leaf_tokamak is not None else base_tokamak
    if tokamak is not None and not isinstance(tokamak, str):
        raise ValueError(f"Environment {env!r} tokamak must be a string")

    tokamak_layer: RawConfig = {}
    if tokamak is not None:
        tokamak_path = registry.CONFIGS_DIR / "tokamaks" / f"{tokamak}.yaml"
        tokamak_layer = _load_extended_yaml(tokamak_path)

    raw = _deep_merge(tokamak_layer, base)
    raw = _deep_merge(raw, leaf)
    if tokamak is not None:
        raw = _deep_merge(
            raw,
            _load_extended_yaml(registry.CONFIGS_DIR / "wrappers.yaml"),
        )
    _validate_env_ownership(raw, env)
    return raw


def load_backend(backend: str) -> RawConfig:
    """Load and ownership-check one registered backend fragment."""
    raw = _load_extended_yaml(registry.resolve_backend(backend))
    _validate_backend_ownership(raw, backend)
    return raw


def valid_env_backend_combos() -> Mapping[str, frozenset[str]]:
    """Return the immutable environment/backend compatibility registry."""
    return registry._VALID_ENV_BACKEND_COMBOS  # noqa: SLF001


def validate_env_backend(env: str, backend: str | None) -> None:
    """Raises ValueError if (env, backend) is not a registered combination."""
    registry.resolve_env(env)
    allowed = valid_env_backend_combos()[env]
    if not allowed:
        if backend is not None:
            raise ValueError(f"Environment {env!r} is standalone and takes no backend")
        return
    if backend is None:
        raise ValueError(
            f"Environment {env!r} requires a backend. "
            f"Compatible backends: {sorted(allowed)}"
        )
    registry.resolve_backend(backend)
    if backend not in allowed:
        raise ValueError(
            f"Backend {backend!r} is not compatible with environment {env!r}. "
            f"Compatible backends: {sorted(allowed)}"
        )


def _merge_env_and_backend(env: str, backend: str) -> RawConfig:
    """Compose one registered TORAX environment/backend pair without precedence."""
    validate_env_backend(env, backend)
    return _union_no_duplicate(load_env_layers(env), load_backend(backend))


__all__ = [
    "load_backend",
    "load_env_layers",
    "_merge_env_and_backend",
    "valid_env_backend_combos",
    "validate_env_backend",
]
