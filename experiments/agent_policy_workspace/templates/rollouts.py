"""Editable exploratory analysis of saved numeric rollout artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tyro


def main(path: Path) -> None:
    """Summarize a saved NPZ; this does not create an official evaluation score."""
    with np.load(path, allow_pickle=False) as saved:
        valid = np.asarray(saved["valid"], dtype=bool)
        rewards = np.asarray(saved["reward"])
        if valid.ndim != 2 or rewards.shape != valid.shape:
            raise ValueError(
                "expected reward and valid arrays shaped (episodes, steps)"
            )
        returns = np.where(valid, rewards, 0).sum(axis=-1)
        lengths = valid.sum(axis=-1)
    episodes = [
        {
            "episode": index,
            "return": float(value) if np.isfinite(value) else str(value),
            "length": int(length),
        }
        for index, (value, length) in enumerate(zip(returns, lengths, strict=True))
    ]
    print(
        json.dumps(
            {
                "kind": "exploratory",
                "source": str(path.resolve()),
                "episodes": episodes,
            },
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    tyro.cli(main)
