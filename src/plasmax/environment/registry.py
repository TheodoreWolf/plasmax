"""Alias registry for environment/backend YAML configs.

Maps short names to the config files shipped with the package
(``plasmax/configs/``), so callers can write
``make("iter/hybrid/flattop", "qlknn")`` or
``plasmax.make("iter/hybrid/flattop", backend="qlknn")`` from any working
directory. This is the single place to register a new environment or backend
alias.
"""

from __future__ import annotations

from pathlib import Path
from types import MappingProxyType
from typing import Final

# Configs ship inside the package so aliases resolve for installed wheels and
# repository checkouts from any working directory.
CONFIGS_DIR: Final = Path(__file__).resolve().parents[1] / "configs"

# Full-transport backends valid for every conventional-aspect-ratio ITER/SPARC
# scenario (any device, scenario, phase combination).
_STANDARD_BACKENDS = frozenset({"cgm", "qlknn", "bohm_gyrobohm", "tglfnn", "tglfnn_nr"})

# Env + backend scenarios are loaded in pairs via `make`. Nested envs are
# addressed as ``<device>/<scenario>/<phase>`` with a sibling ``base.yaml``
# holding the phase-shared physics. The phase is always explicit because each
# phase is a separate control task and policy. Keys match the compatibility
# registry below.
_ENV_ALIASES = {
    f"{device}/{scenario}/{phase}": (
        CONFIGS_DIR / "envs" / device / scenario / f"{phase}.yaml"
    )
    for device, scenarios in (
        ("iter", ("baseline", "hybrid", "advanced")),
        ("sparc", ("prd", "reduced_field")),
    )
    for scenario in scenarios
    for phase in ("rampup", "flattop", "rampdown")
}
_ENV_ALIASES.update(
    {
        "step/spp_001_ec_hd/flattop": (
            CONFIGS_DIR / "envs" / "step" / "spp_001_ec_hd" / "flattop.yaml"
        ),
        "mock/circular/smoke": (
            CONFIGS_DIR / "envs" / "mock" / "circular" / "smoke.yaml"
        ),
        "kstar_worldmodel": CONFIGS_DIR / "envs" / "kstar_worldmodel.yaml",
    }
)

_BACKEND_ALIASES = {
    name: CONFIGS_DIR / "backends" / f"{name}.yaml"
    for name in (
        "cgm",
        "qlknn",
        "bohm_gyrobohm",
        "bohm_gyrobohm_step",
        "tglfnn",
        "tglfnn_nr",
        "tglfnn_spherical",
        "mock",
    )
}

# Env-backend compatibility, keyed by the env address. ITER and SPARC span the
# (device, scenario, phase) grid: scenarios x 3 phases. A missing key means that
# (device, scenario, phase) combination does not exist.
_VALID_ENV_BACKEND_COMBOS = MappingProxyType(
    {
        env: (
            frozenset()
            if env == "kstar_worldmodel"
            else frozenset({"bohm_gyrobohm_step", "tglfnn_spherical"})
            if env == "step/spp_001_ec_hd/flattop"
            else frozenset({"mock"})
            if env == "mock/circular/smoke"
            else _STANDARD_BACKENDS
        )
        for env in _ENV_ALIASES
    }
)

ENV_ALIASES: Final = MappingProxyType(_ENV_ALIASES)
BACKEND_ALIASES: Final = MappingProxyType(_BACKEND_ALIASES)


def resolve_env(name: str) -> Path:
    """Resolves an env alias (e.g. ``"iter/hybrid/flattop"``) to its YAML path."""
    try:
        return ENV_ALIASES[name]
    except (KeyError, TypeError) as error:
        raise ValueError(
            f"Unknown environment {name!r}. "
            f"Registered environments: {sorted(ENV_ALIASES)}"
        ) from error


def resolve_backend(name: str) -> Path:
    """Resolves a backend alias (e.g. ``"qlknn"``) to its YAML path."""
    try:
        return BACKEND_ALIASES[name]
    except (KeyError, TypeError) as error:
        raise ValueError(
            f"Unknown backend {name!r}. Registered backends: {sorted(BACKEND_ALIASES)}"
        ) from error


__all__ = [
    "BACKEND_ALIASES",
    "CONFIGS_DIR",
    "ENV_ALIASES",
    "resolve_backend",
    "resolve_env",
]
