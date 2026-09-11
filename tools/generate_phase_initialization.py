"""Capture a nominal task state as one readable YAML initialization.

Use --source-steps 0 to export the selected nominal state, or explicitly request
held-action evolution with --source-steps N. Initializations use YAML exclusively.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import tyro

from plasmax.environment import factory as factory_lib
from plasmax.environment.config import parse_env_and_backend
from plasmax.environment.initialization import (
    initialization_from_snapshot,
    snapshot_from_state,
)
from plasmax.environment.initialization_data import (
    KstarInitialization,
    ToraxInitialization,
    write_initialization,
)
from plasmax.environment.references import ReferenceExtraction
from plasmax.environment.registry import CONFIGS_DIR
from plasmax.environment.schema import PlasmaxConfig, WorldModelConfig


@dataclasses.dataclass(frozen=True)
class Config:
    environment: str = "iter/hybrid/flattop"
    source_backend: str | None = None
    source_steps: int = 0
    seed: int = 0
    output: Path | None = None


def _backend(config: Config) -> str | None:
    if config.source_backend is not None:
        return config.source_backend
    if config.environment == "kstar_worldmodel":
        return None
    if config.environment.startswith("step/"):
        return "bohm_gyrobohm_step"
    if config.environment.startswith("mock/"):
        return "mock"
    return "bohm_gyrobohm"


def _source_path(path: Path) -> str:
    path = path.resolve()
    return str(
        path.relative_to(CONFIGS_DIR) if path.is_relative_to(CONFIGS_DIR) else path
    )


def capture(config: Config) -> ToraxInitialization | KstarInitialization:
    """Return a resolved state without changing runtime precision or reset RNGs."""
    if config.source_steps < 0:
        raise ValueError("source_steps must be non-negative")
    backend = _backend(config)
    parsed = parse_env_and_backend(config.environment, backend)
    document = parsed._initial_state
    digest = hashlib.sha256(
        json.dumps(parsed.model_dump(mode="json"), sort_keys=True).encode()
    ).hexdigest()

    if isinstance(parsed, WorldModelConfig):
        if config.source_steps:
            raise ValueError("KSTAR export captures its nominal learned initialization")
        from plasmax.models.world_model import load_bundle, predict_nn
        from plasmax.models.world_model_env import _steady_features

        assert isinstance(document, KstarInitialization)
        inputs = jnp.asarray(
            [document.inputs[name] for name in document.input_order], jnp.float32
        )
        features = _steady_features(inputs)
        outputs = predict_nn(load_bundle(parsed.world_model.weights_path), features)
        row = np.concatenate([np.asarray(outputs), np.asarray(features)])
        weights = parsed.world_model.weights_path
        provenance = document.provenance.model_copy(
            update={
                "source_config_sha256": digest,
                "sources": (
                    {
                        "title": "NeoRL2 KSTAR weights",
                        "url": "https://github.com/polixir/NeoRL2",
                        "local_path": _source_path(weights),
                        "sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
                    },
                    *(
                        source
                        for source in document.provenance.sources
                        if source.get("title") != "NeoRL2 KSTAR weights"
                    ),
                ),
            }
        )
        return document.model_copy(
            update={"history_row": tuple(row.tolist()), "provenance": provenance}
        )

    assert isinstance(parsed, PlasmaxConfig)
    assert isinstance(document, ToraxInitialization)
    if config.source_steps == 0:
        return document
    parsed = parsed.model_copy(update={"state_noise": {}, "physics_randomization": {}})
    env = factory_lib._build_env(parsed, reward=None, disruption_penalty=None)
    state, _ = env.init(jax.random.key(config.seed))
    step = jax.jit(env.step)
    for index in range(config.source_steps):
        state, info = step(state, state.prev_action)
        jax.block_until_ready((state, info))
        if not bool(info.control_step_complete) or bool(
            info.terminated | info.truncated
        ):
            raise RuntimeError(f"source trajectory ended at step {index + 1}")
    snapshot = snapshot_from_state(
        state.plasma.sim,
        environment=config.environment,
        source_backend=backend,
        source_step=config.source_steps,
        seed=config.seed,
        source_config_sha256=digest,
    )
    provenance = document.provenance.model_copy(update=snapshot.metadata.model_dump())
    source = parsed.initialization
    provenance = provenance.model_copy(
        update={
            "sources": (
                *provenance.sources,
                {
                    "title": "Capture input state",
                    "local_path": _source_path(source),
                    "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                },
            ),
            "extraction": ReferenceExtraction(
                method="simulation_capture",
                grid=f"{len(snapshot.rho_norm)} TORAX cells and "
                f"{len(snapshot.rho_face_norm)} faces",
                notes=(
                    f"Captured after {config.source_steps} held-action steps "
                    "with reset noise and physics randomization disabled.",
                ),
            ),
        }
    )
    exported = initialization_from_snapshot(
        snapshot, description=document.description, provenance=provenance
    )
    return exported.model_copy(update={"composition": document.composition})


def main(config: Config) -> None:
    document = capture(config)
    output = config.output
    if output is None:
        output = parse_env_and_backend(
            config.environment, _backend(config)
        ).initialization
    checksum = write_initialization(document, output)
    print(f"Saved {output}: sha256={checksum}", flush=True)


if __name__ == "__main__":
    main(tyro.cli(Config))
