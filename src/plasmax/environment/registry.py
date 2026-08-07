"""Alias registry for environment/backend/scenario YAML configs.

Maps short names to the config files shipped with the package
(``plasmax/configs/``), so callers can write
``load_env("iter/hybrid/flattop", "qlknn")`` or
``plasmax.make("iter/hybrid/flattop", backend="qlknn")`` from any working
directory. This is the single place to register a new scenario or backend
alias. Unregistered values (e.g. an explicit YAML path) are returned
unchanged by the ``resolve_*`` helpers.
"""

from __future__ import annotations

from pathlib import Path

# Configs ship inside the package so aliases resolve for installed wheels and
# repository checkouts from any working directory.
CONFIGS_DIR = Path(__file__).resolve().parents[1] / "configs"

# Env + backend scenarios, loaded in pairs via `load_env`. Nested envs are
# addressed as ``<device>/<scenario>/<phase>`` with a sibling ``base.yaml``
# holding the phase-shared physics. The phase is always explicit because each
# phase is a separate control task and policy. Keys match the ``_env_key`` used
# in merge._VALID_ENV_BACKEND_COMBOS.
ENV_ALIASES: dict[str, str] = {
    f"{device}/{scenario}/{phase}": str(
        CONFIGS_DIR / "envs" / device / scenario / f"{phase}.yaml"
    )
    for device, scenarios in (
        ("iter", ("baseline", "hybrid", "advanced")),
        ("sparc", ("prd", "reduced_field")),
    )
    for scenario in scenarios
    for phase in ("rampup", "flattop", "rampdown")
}
ENV_ALIASES.update(
    {
        "step": str(CONFIGS_DIR / "envs" / "step.yaml"),
        "kstar": str(CONFIGS_DIR / "envs" / "kstar.yaml"),
    }
)

BACKEND_ALIASES: dict[str, str] = {
    name: str(CONFIGS_DIR / "backends" / f"{name}.yaml")
    for name in (
        "cgm",
        "qlknn",
        "bohm_gyrobohm",
        "tglfnn",
        "tglfnn_nr",
        "tglfnn_spherical",
        "fusion_lstm",
    )
}

# Single-file scenarios, loaded via `load_scenario` (no separate backend).
SCENARIO_ALIASES: dict[str, str] = {
    "test": str(CONFIGS_DIR / "test.yaml"),
}


def resolve_env(name: str) -> str:
    """Resolves an env alias (e.g. ``"iter/hybrid/flattop"``) to its YAML path."""
    return ENV_ALIASES.get(name, name)


def resolve_backend(name: str) -> str:
    """Resolves a backend alias (e.g. ``"qlknn"``) to its YAML path."""
    return BACKEND_ALIASES.get(name, name)


def resolve_scenario(name: str) -> str:
    """Resolves a single-file scenario alias (e.g. ``"test"``) to its YAML path."""
    return SCENARIO_ALIASES.get(name, name)
