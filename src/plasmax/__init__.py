"""Stable, JAX-native fusion-control environments built on TORAX."""

# Apply the TORAX Grid1D tracer fix before constructing any geometry or
# environment. This is intentionally an import side effect at the package
# boundary; see ``_torax_patches.py`` for the upstream compatibility context.
from plasmax import _torax_patches as _torax_patches  # noqa: F401  # isort: skip
from plasmax.environment import registry
from plasmax.environment.core import EnvState, PlasmaxEnv
from plasmax.environment.factory import make
from plasmax.environment.schema import PlasmaxConfig
from plasmax.rollout import TrajectoryStep, collect_episode, collect_episodes

__all__ = [
    "make",
    "PlasmaxConfig",
    "PlasmaxEnv",
    "EnvState",
    "TrajectoryStep",
    "collect_episode",
    "collect_episodes",
    "registry",
]
