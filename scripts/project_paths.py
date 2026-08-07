"""Repository paths shared by development-only scripts."""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = REPO_ROOT / "src" / "plasmax" / "configs"
OUTPUTS_DIR = REPO_ROOT / "outputs"
PLOTS_DIR = REPO_ROOT / "plots"


def wandb_dir() -> Path:
    """Return the W&B run directory, preserving an explicit cluster override."""

    override = os.environ.get("WANDB_DIR")
    return Path(override).expanduser() if override else OUTPUTS_DIR / "wandb"


__all__ = [
    "CONFIGS_DIR",
    "OUTPUTS_DIR",
    "PLOTS_DIR",
    "REPO_ROOT",
    "wandb_dir",
]
