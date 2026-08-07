"""Env/backend YAML merge helpers and compatibility registry."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml

_SCENARIO_DIR_TOKEN = "${SCENARIO_DIR}"
_DATA_DIR_TOKEN = "${DATA_DIR}"
_WRAPPERS_CONFIG_NAME = "wrappers.yaml"
_TOKAMAKS_DIR_NAME = "tokamaks"
_ENVS_DIR_NAME = "envs"
_SCENARIO_BASE_NAME = "base.yaml"
# Metadata keys carried in env/scenario YAMLs that name a merge layer rather
# than contribute config; stripped from the merged output.
_META_KEYS = (
    "tokamak",
    "scenario",
    "phase",
    "reset_reference",
)


def _configs_root(path: str) -> Path:
    """Returns the packaged config root by walking up to the ``envs`` ancestor.

    Envs may nest arbitrarily under ``<config-root>/envs`` (e.g.
    ``envs/iter/hybrid/flattop.yaml``), so ``tokamaks/`` and ``data/``
    are located relative to the ``envs`` parent rather than a fixed depth.
    """
    p = Path(path).resolve()
    for anc in p.parents:
        if anc.name == _ENVS_DIR_NAME:
            return anc.parent
    # Fallback for non-envs paths (e.g. the single-file test.yaml scenario).
    return p.parent


def _env_key(env_path: str) -> str:
    """Returns the registry key for an env: its path relative to ``envs/``.

    e.g. ``envs/iter/hybrid/flattop.yaml`` -> ``iter/hybrid/flattop``.
    This encodes the (device, scenario, phase) address for nested envs and a
    bare device name for single-file ones (e.g. ``envs/step.yaml`` -> ``step``).
    """
    p = Path(env_path).resolve()
    for anc in p.parents:
        if anc.name == _ENVS_DIR_NAME:
            return p.relative_to(anc).with_suffix("").as_posix()
    return p.stem


def _load_tokamak_defaults(env_path: str, env_raw: dict[str, Any]) -> dict[str, Any]:
    """Loads device-level defaults named by the env's ``tokamak:`` key.

    A scenario env sets ``tokamak: iter`` to inherit the shared device config
    from ``tokamaks/iter.yaml`` (geometry plumbing, plasma_composition,
    observations, actuators, disruption, …); the env overlays only its
    scenario-specific deltas. Returns ``{}`` when no ``tokamak:`` key is present.
    """
    name = env_raw.get("tokamak")
    if name is None:
        return {}
    tokamak_path = _configs_root(env_path) / _TOKAMAKS_DIR_NAME / f"{name}.yaml"
    if not tokamak_path.exists():
        raise FileNotFoundError(
            f"env {_env_key(env_path)!r} sets tokamak: {name!r} but "
            f"{str(tokamak_path)!r} does not exist"
        )
    with open(tokamak_path) as f:
        return yaml.safe_load(f) or {}


def _load_scenario_base(env_path: str) -> dict[str, Any]:
    """Loads a sibling ``base.yaml`` holding the scenario's phase-shared config.

    Multi-phase scenarios (e.g. ``envs/iter/hybrid/{rampup,flattop,
    rampdown}.yaml``) factor the physics common to all phases
    (composition, sources, pedestal, …) into ``base.yaml`` in the same
    directory; each phase file carries only its deltas (Ip schedule, t_final,
    geometry, current-drive fraction). Returns ``{}`` when the env is itself
    ``base.yaml`` or has no sibling base (single-file devices like ``step``).
    """
    p = Path(env_path).resolve()
    if p.name == _SCENARIO_BASE_NAME:
        return {}
    base_path = p.parent / _SCENARIO_BASE_NAME
    if not base_path.exists():
        return {}
    with open(base_path) as f:
        return yaml.safe_load(f) or {}


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``overlay`` onto ``base`` (overlay wins on leaves)."""
    out = dict(base)
    for k, v in overlay.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _load_extended_yaml(
    path: str | Path,
    *,
    root: Path | None = None,
    stack: tuple[Path, ...] = (),
) -> dict[str, Any]:
    """Load a YAML mapping with a repository-internal ``extends`` chain.

    Extension paths are relative to the file declaring them and cannot escape
    the initial file's directory. Leaf mappings override their shared base.
    """
    resolved = Path(path).resolve()
    allowed_root = resolved.parent if root is None else root
    if not resolved.is_relative_to(allowed_root):
        raise ValueError(
            f"extends path {str(resolved)!r} escapes {str(allowed_root)!r}"
        )
    if resolved in stack:
        chain = " -> ".join(str(item) for item in (*stack, resolved))
        raise ValueError(f"cyclic YAML extends chain: {chain}")
    with resolved.open() as stream:
        raw = yaml.safe_load(stream) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"YAML config {str(resolved)!r} must contain a mapping")
    extends = raw.pop("extends", None)
    if extends is None:
        return raw
    if not isinstance(extends, str):
        raise ValueError(f"extends in {str(resolved)!r} must be a relative path")
    parent = (resolved.parent / extends).resolve()
    base = _load_extended_yaml(
        parent,
        root=allowed_root,
        stack=(*stack, resolved),
    )
    return _deep_merge(base, raw)


def _merge_backend_transport(
    backend_torax: dict[str, Any], *overlay_torax: dict[str, Any]
) -> dict[str, Any] | None:
    """Merges transport tuning without changing the selected backend model.

    A scenario may carry calibration for one transport model while supporting
    another backend. Model-specific calibration applies only when its
    ``model_name`` matches the backend; model-agnostic transport settings still
    overlay normally.
    """
    backend_transport = backend_torax.get("transport")
    if not isinstance(backend_transport, dict):
        return None

    selected_model = backend_transport.get("model_name")
    merged = dict(backend_transport)
    for torax_overlay in overlay_torax:
        transport_overlay = torax_overlay.get("transport")
        if not isinstance(transport_overlay, dict):
            continue
        overlay_model = transport_overlay.get("model_name")
        if (
            selected_model is not None
            and overlay_model is not None
            and overlay_model != selected_model
        ):
            continue
        merged = _deep_merge(merged, transport_overlay)
    return merged


def _resolve_geometry_dir(torax_dict: dict[str, Any], yaml_path: str) -> dict[str, Any]:
    """Expands geometry_directory tokens against ``yaml_path``.

    ``${DATA_DIR}`` -> ``<configs_root>/data`` (depth-independent, used by the
    tokamak files since envs nest arbitrarily under the packaged ``envs/``);
    ``${SCENARIO_DIR}`` -> ``dirname(yaml_path)`` (legacy relative token).
    """
    geo = torax_dict.get("geometry", {})
    gdir = geo.get("geometry_directory")
    if not isinstance(gdir, str):
        return torax_dict
    if _DATA_DIR_TOKEN in gdir:
        data_dir = str(_configs_root(yaml_path) / "data")
        gdir = gdir.replace(_DATA_DIR_TOKEN, data_dir)
    if _SCENARIO_DIR_TOKEN in gdir:
        gdir = gdir.replace(
            _SCENARIO_DIR_TOKEN, os.path.dirname(os.path.abspath(yaml_path))
        )
    if gdir != geo.get("geometry_directory"):
        return {**torax_dict, "geometry": {**geo, "geometry_directory": gdir}}
    return torax_dict


def _has_nonempty_ip(ip_field: Any) -> bool:
    """True iff ``ip_field`` carries a usable (non-empty) Ip value."""
    if ip_field is None:
        return False
    # core_profiles_from_IMAS returns (time_array, ip_array). If the IDS has
    # no Ip filled, ip_array is an empty numpy array.
    if isinstance(ip_field, tuple) and len(ip_field) == 2:
        try:
            return len(ip_field[1]) > 0
        except TypeError:
            return True
    return True


def _apply_imas_init(torax_dict: dict[str, Any], yaml_path: str) -> dict[str, Any]:
    """Resolves a top-level ``_imas_init`` directive in a torax block."""
    init = torax_dict.get("_imas_init")
    if init is None:
        return torax_dict

    if "path" not in init:
        raise ValueError("_imas_init requires a 'path' key")

    # Lazy import -- only require ``imas`` when actually used.
    import imas
    from torax._src.imas_tools.input import (
        core_profiles as imas_core_profiles,
    )
    from torax._src.imas_tools.input import (
        loader as imas_loader,
    )

    scenario_dir = os.path.dirname(os.path.abspath(yaml_path))
    nc_path = init["path"].replace(_SCENARIO_DIR_TOKEN, scenario_dir)
    nc_path = os.path.abspath(nc_path)
    if not os.path.exists(nc_path):
        raise FileNotFoundError(
            f"_imas_init.path resolves to {nc_path!r} which does not exist"
        )

    want_pc = bool(init.get("profile_conditions", False))
    want_comp = bool(init.get("plasma_composition", False))
    if not (want_pc or want_comp):
        return {k: v for k, v in torax_dict.items() if k != "_imas_init"}

    explicit_convert = bool(init.get("explicit_convert", False))
    cp_ids = imas_loader.load_imas_data(
        nc_path,
        "core_profiles",
        explicit_convert=explicit_convert,
    )
    out = {k: v for k, v in torax_dict.items() if k != "_imas_init"}

    if want_pc:
        imas_pc = dict(imas_core_profiles.profile_conditions_from_IMAS(cp_ids))
        # core_profiles IDS Ip is sometimes empty; pull from equilibrium IDS.
        if not _has_nonempty_ip(imas_pc.get("Ip")):
            eq_ids = imas_loader.load_imas_data(
                nc_path,
                "equilibrium",
                explicit_convert=explicit_convert,
            )
            eq_xr = imas.util.to_xarray(eq_ids)
            imas_pc["Ip"] = float(eq_xr["time_slice.global_quantities.ip"][0].item())
        yaml_pc = dict(out.get("profile_conditions") or {})
        out["profile_conditions"] = {**imas_pc, **yaml_pc}

    if want_comp:
        imas_comp = dict(imas_core_profiles.plasma_composition_from_IMAS(cp_ids))
        yaml_comp = dict(out.get("plasma_composition") or {})
        out["plasma_composition"] = _deep_merge(imas_comp, yaml_comp)

    return out


def _load_wrapper_defaults(env_path: str) -> dict[str, Any]:
    """Loads shared wrapper defaults from packaged ``wrappers.yaml`` when present."""
    p = Path(env_path).resolve()
    if not any(anc.name == _ENVS_DIR_NAME for anc in p.parents):
        return {}
    wrapper_path = _configs_root(env_path) / _WRAPPERS_CONFIG_NAME
    if not wrapper_path.exists():
        return {}

    with open(wrapper_path) as f:
        raw = yaml.safe_load(f) or {}

    # Wrapper YAMLs may only carry the wrapper-stack blocks; anything else
    # belongs in a tokamak/scenario file.
    for block, keys, allowed in (
        ("top-level", raw, {"observations", "actions"}),
        ("observations", raw.get("observations") or {}, {"realistic", "history"}),
        ("actions", raw.get("actions") or {}, {"realistic"}),
    ):
        extra = set(keys) - allowed
        if extra:
            raise ValueError(
                f"Wrapper YAML {str(wrapper_path)!r}: unexpected {block} keys "
                f"{sorted(extra)}; allowed: {sorted(allowed)}"
            )
    return raw


def _merge_env_and_backend(env_path: str, backend_path: str) -> dict[str, Any]:
    """Returns the merged raw dict for an env+backend YAML pair."""
    with open(env_path) as f:
        phase_raw = yaml.safe_load(f) or {}
    backend_raw = _load_extended_yaml(backend_path)

    allowed_backend_blocks = {
        "torax",
        "physics_randomization",
        "stepping",
    }
    extra = set(backend_raw) - allowed_backend_blocks
    if extra:
        raise ValueError(
            f"Backend YAML {backend_path!r} may only contain "
            "`torax:`, `physics_randomization:`, and `stepping:` blocks; "
            f"unexpected top-level keys: {sorted(extra)}"
        )

    # Precedence, low -> high:
    #   backend < wrappers < tokamak < scenario base < phase (env).
    # The scenario base (sibling base.yaml) holds the physics shared across a
    # scenario's phases; the addressed env file carries the phase deltas. For
    # single-file devices (sparc/step/kstar) base is empty and env == phase.
    # The backend remains authoritative for the transport model discriminator;
    # model-specific env calibration applies only to a matching model.
    base_raw = _load_scenario_base(env_path)
    env_raw = _deep_merge(base_raw, phase_raw)
    tokamak_raw = _load_tokamak_defaults(env_path, env_raw)
    wrapper_raw = _load_wrapper_defaults(env_path)

    def _non_torax(d: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in d.items() if k != "torax" and k not in _META_KEYS}

    merged = _deep_merge(_non_torax(backend_raw), wrapper_raw)
    merged = _deep_merge(merged, _non_torax(tokamak_raw))
    merged = _deep_merge(merged, _non_torax(env_raw))

    backend_torax = dict(backend_raw.get("torax") or {})
    tokamak_torax = dict(tokamak_raw.get("torax") or {})
    env_torax = dict(env_raw.get("torax") or {})

    merged_torax = _deep_merge(_deep_merge(backend_torax, tokamak_torax), env_torax)
    merged_transport = _merge_backend_transport(backend_torax, tokamak_torax, env_torax)
    if merged_transport is not None:
        merged_torax["transport"] = merged_transport
    merged["torax"] = merged_torax
    return merged


# Full-transport backends valid for every conventional-aspect-ratio ITER/SPARC
# scenario (any device, scenario, phase combination).
_STANDARD_BACKENDS = frozenset({"cgm", "qlknn", "bohm_gyrobohm", "tglfnn", "tglfnn_nr"})

# Env-backend compatibility, keyed by the env address (`_env_key`). ITER and
# SPARC span the (device, scenario, phase) grid: scenarios x 3 phases. A missing
# key means that (device, scenario, phase) combination does not exist.
_valid_env_backend_combos: dict[str, frozenset[str]] = {
    f"iter/{scenario}/{phase}": _STANDARD_BACKENDS
    for scenario in ("baseline", "hybrid", "advanced")
    for phase in ("rampup", "flattop", "rampdown")
}
_valid_env_backend_combos.update(
    {
        f"sparc/{scenario}/{phase}": _STANDARD_BACKENDS
        for scenario in ("prd", "reduced_field")
        for phase in ("rampup", "flattop", "rampdown")
    }
)
_valid_env_backend_combos.update(
    {
        "step": frozenset({"bohm_gyrobohm", "tglfnn_spherical"}),
        "kstar": frozenset({"fusion_lstm"}),
    }
)
_VALID_ENV_BACKEND_COMBOS: Mapping[str, frozenset[str]] = MappingProxyType(
    _valid_env_backend_combos
)
del _valid_env_backend_combos


def valid_env_backend_combos() -> Mapping[str, frozenset[str]]:
    """Return the immutable environment/backend compatibility registry."""
    return _VALID_ENV_BACKEND_COMBOS


def validate_env_backend(env_path: str, backend_path: str) -> None:
    """Raises ValueError if (env, backend) is not a registered combination.

    The env is addressed by its path relative to the packaged ``envs/`` (e.g.
    ``iter/hybrid/flattop``), which is the (device, scenario, phase) triple for
    nested scenarios; unknown addresses (bad device/scenario/phase) are
    rejected here before any JAX compilation.
    """
    env_name = _env_key(env_path)
    backend_name = Path(backend_path).stem
    if env_name not in _VALID_ENV_BACKEND_COMBOS:
        raise ValueError(
            f"env {env_name!r} not in valid_env_backend_combos(); register it "
            "in merge._VALID_ENV_BACKEND_COMBOS."
        )
    allowed = _VALID_ENV_BACKEND_COMBOS[env_name]
    if backend_name not in allowed:
        raise ValueError(
            f"env {env_name!r} is not compatible with backend {backend_name!r}. "
            f"Allowed: {sorted(allowed)}"
        )


def backend_kind(backend_path: str) -> str:
    """Returns a backend's kind: ``'torax'`` (default) or ``'world_model'``."""
    raw = _load_extended_yaml(backend_path)
    return raw.get("type", "torax")


def env_key(env_path: str) -> str:
    """Public alias for the env's registry address (path relative to ``envs/``).

    e.g. ``envs/iter/hybrid/flattop.yaml`` -> ``iter/hybrid/flattop``.
    """
    return _env_key(env_path)


__all__ = [
    "backend_kind",
    "env_key",
    "valid_env_backend_combos",
    "validate_env_backend",
]
