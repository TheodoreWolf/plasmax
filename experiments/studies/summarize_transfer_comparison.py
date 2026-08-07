"""Summarize the six-method oracle/realistic backend-transfer comparison."""

from __future__ import annotations

import csv
import dataclasses
import json
from pathlib import Path
from typing import Any

import tyro

ALGORITHMS = (
    "direct_policy",
    "direct_knots_1",
    "direct_knots_10",
    "direct_knots_100",
    "sac",
    "ppo",
)
LABELS = {
    "direct_policy": "Policy learning",
    "direct_knots_1": "1 knot",
    "direct_knots_10": "10 knots",
    "direct_knots_100": "100 knots",
    "sac": "SAC",
    "ppo": "PPO",
}
BACKEND_LABELS = {
    "bohm_gyrobohm": "BGB",
    "cgm": "CGM",
    "qlknn": "QLKNN",
    "tglfnn": "TLGFNN",
}


@dataclasses.dataclass
class Config:
    root: str = "outputs/bgb2tglfnn_10m"
    csv_out: str = "outputs/bgb2tglfnn_10m/transfer_comparison.csv"
    markdown_out: str = "outputs/bgb2tglfnn_10m/transfer_comparison.md"
    source_backend: str = "bohm_gyrobohm"
    target_backend: str = "tglfnn"
    allow_incomplete: bool = False


def _algorithm_and_variant(path: Path) -> tuple[str, str]:
    text = "/".join(path.parts).lower()
    algorithm = next(
        (name for name in sorted(ALGORITHMS, key=len, reverse=True) if name in text),
        None,
    )
    variant = next(
        (name for name in ("oracle", "realistic") if name in text),
        None,
    )
    if algorithm is None or variant is None:
        raise ValueError(f"cannot infer algorithm/variant from {path}")
    return algorithm, variant


def _metrics(
    payload: dict[str, Any],
    source_backend: str,
    target_backend: str,
) -> dict[str, Any]:
    values = payload.get("metrics", payload)
    if values.get("transfer/source_backend") != source_backend:
        raise ValueError(f"comparison requires {source_backend} source metrics")
    if values.get("transfer/target_backend") != target_backend:
        raise ValueError(f"comparison requires {target_backend} target metrics")
    return values


def load_rows(
    root: Path,
    target_backend: str = "tglfnn",
    source_backend: str = "bohm_gyrobohm",
) -> list[dict[str, Any]]:
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    candidates = sorted(root.rglob("*_transfer.json"))
    candidates.extend(sorted((root / "direct").glob("*.json")))
    for path in candidates:
        algorithm, variant = _algorithm_and_variant(path)
        metrics = _metrics(
            json.loads(path.read_text()),
            source_backend,
            target_backend,
        )
        key = (algorithm, variant)
        if key in rows:
            raise ValueError(f"duplicate transfer summary for {key}: {path}")
        target_length = float(metrics["transfer/target_episode_length_mean"])
        rows[key] = {
            "algorithm": algorithm,
            "method": LABELS[algorithm],
            "variant": variant,
            "training_seeds": int(
                metrics.get(
                    "transfer/num_training_seeds",
                    10 if algorithm.startswith("direct_") else 2,
                )
            ),
            "source_return": float(metrics["transfer/source_return_mean"]),
            "source_seed_std": float(
                metrics.get("transfer/source_return_seed_std", 0.0)
            ),
            "target_return": float(metrics["transfer/target_return_mean"]),
            "target_seed_std": float(
                metrics.get("transfer/target_return_seed_std", 0.0)
            ),
            "return_gap": float(metrics["transfer/return_gap"]),
            "return_ratio": float(metrics["transfer/return_ratio"]),
            "target_episode_length": target_length,
            "target_horizon_fraction": target_length / 4400.0,
            "path": str(path),
        }
    return [
        rows[(algorithm, variant)]
        for variant in ("oracle", "realistic")
        for algorithm in ALGORITHMS
        if (algorithm, variant) in rows
    ]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(
    path: Path,
    rows: list[dict[str, Any]],
    source_backend: str,
    target_backend: str,
) -> None:
    source_label = BACKEND_LABELS.get(source_backend, source_backend.upper())
    target_label = BACKEND_LABELS.get(target_backend, target_backend.upper())
    lines = [
        f"| Variant | Method | Seeds | {source_label} return | "
        f"{target_label} return | "
        "Retained | Target horizon |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['variant']} | {row['method']} | {row['training_seeds']} | "
            f"{row['source_return']:.2f} | {row['target_return']:.2f} | "
            f"{100.0 * row['return_ratio']:.3f}% | "
            f"{100.0 * row['target_horizon_fraction']:.2f}% |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main(cfg: Config) -> None:
    rows = load_rows(
        Path(cfg.root),
        cfg.target_backend,
        cfg.source_backend,
    )
    if not cfg.allow_incomplete and len(rows) != 12:
        raise ValueError(f"expected 12 comparison cells, found {len(rows)}")
    if not rows:
        raise ValueError(f"no transfer summaries found below {cfg.root}")
    _write_csv(Path(cfg.csv_out), rows)
    _write_markdown(
        Path(cfg.markdown_out),
        rows,
        cfg.source_backend,
        cfg.target_backend,
    )
    print(Path(cfg.markdown_out).read_text())


if __name__ == "__main__":
    main(tyro.cli(Config))
