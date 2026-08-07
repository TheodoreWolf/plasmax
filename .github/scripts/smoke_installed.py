"""Smoke-test an installed plasmax release artifact outside the checkout."""

from __future__ import annotations

import importlib.util
import inspect
from importlib import resources

import plasmax

EXPECTED_PUBLIC_API = {
    "EnvState",
    "PlasmaxEnv",
    "ScenarioConfig",
    "TrajectoryStep",
    "collect_episode",
    "collect_episodes",
    "load_env",
    "load_scenario",
    "make",
    "registry",
}


def main() -> None:
    if set(plasmax.__all__) != EXPECTED_PUBLIC_API:
        raise AssertionError(f"unexpected public API: {sorted(plasmax.__all__)}")

    retired_namespace = "".join(("torax", "_rl"))
    if importlib.util.find_spec(retired_namespace) is not None:
        raise AssertionError("retired import namespace is still installed")
    try:
        __import__(retired_namespace)
    except ModuleNotFoundError:
        pass
    else:
        raise AssertionError("retired import namespace remains importable")

    for repository_module in ("agents", "training", "scripts", "experiments"):
        if importlib.util.find_spec(repository_module) is not None:
            raise AssertionError(f"repository-only module leaked: {repository_module}")

    package = resources.files("plasmax")
    required_assets = (
        package / "configs" / "test.yaml",
        package / "configs" / "data" / "STEP_SPP_001_ECHD_ftop.nc",
        package / "configs" / "data" / "STEP_SPP_001_ECHD_ftop.NOTICE",
        package / "configs" / "data" / "kstar_lstm" / "weights.npz",
    )
    missing_assets = [str(path) for path in required_assets if not path.is_file()]
    if missing_assets:
        raise AssertionError(f"installed assets are missing: {missing_assets}")

    for loader in (plasmax.make, plasmax.load_env, plasmax.load_scenario):
        default_variant = inspect.signature(loader).parameters["variant"].default
        if default_variant != "realistic":
            raise AssertionError(f"{loader.__name__} defaults to {default_variant!r}")

    env = plasmax.make("test")
    if env.action_space.shape != (2,):
        raise AssertionError(f"unexpected action space: {env.action_space.shape}")


if __name__ == "__main__":
    main()
