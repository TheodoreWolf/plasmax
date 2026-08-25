"""Golden checks for lossless YAML deduplication."""

# ruff: noqa: E501 -- immutable SHA-256 values are clearer on one line.

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from plasmax.environment.merge import _deep_merge, _load_extended_yaml

_CONFIGS = Path(__file__).parents[1] / "src" / "plasmax" / "configs"

# SHA-256 of the canonical merged base + phase mappings after the physical
# reference migration and restoration of the hybrid control-task horizons.
# Task metadata is intentionally omitted because this test protects lossless
# YAML factoring rather than task-level reward settings.
_PHASE_GOLDENS = {
    "iter/advanced/rampup": "acf79107e863c87967b21bb310d73c959c4654d81cbc2cc54e148e203d9fbddb",
    "iter/advanced/flattop": "a5671c179312267c10de12c58cf6ce3d606f2351c606325bb904ab65d91a37a9",
    "iter/advanced/rampdown": "82d2fb33f2e616586f22a6b851dbc56e9eee0011905926e13d02d515e744f098",
    "iter/baseline/rampup": "412685ca41bb8889ef7c01178279b44cdace92ce2ef19ce102b2e75e854b003e",
    "iter/baseline/flattop": "779eaa7aa2c952c1c3a924db8de9f7142faf9458b4c3e777d515a6b56f9de450",
    "iter/baseline/rampdown": "61b0d8eb6afef9e9a57390d647ff2e7661ac26656908b512ef3724d9eed9e969",
    "iter/hybrid/rampup": "f1129c9a4598718d666b97cd9ac2d96fd03422e7b4cd8d7a71c0788f715d651d",
    "iter/hybrid/flattop": "123af9a22bd600a53bf399cc6c826fdd2eb4f82744e06106144587c85f36692e",
    "iter/hybrid/rampdown": "f091272fc308eedd311f34273c70c532af979a0c270498fa43c071786f39fdd6",
    "sparc/prd/rampup": "3c4bd1f53b208af21709303a42ce73d6e956609ef8815679d384628e45f5cd53",
    "sparc/prd/flattop": "2abcec99b9c4a56aed84e2473defadccc36b24b3187dc371638a941366e1c11c",
    "sparc/prd/rampdown": "876f98836a0c3c4cb1890df4ee81c678474b78820910b92b079930d3b76af83c",
    "sparc/reduced_field/rampup": "aff12ffcf12c66150e40e0a6cd816a0cf3cfb80cf997e2e14600f96d3e5713cd",
    "sparc/reduced_field/flattop": "10f8c743040ae0458eb6ed3fb5954db25f3be0f77ed95832cb020735526ca86e",
    "sparc/reduced_field/rampdown": "83f3eb4416c954f2ba051beaf39deb2588c20fb391885545ded23d8edd1e8f7f",
}

# SHA-256 of each complete backend mapping after the calibration search
# spaces moved to tools/calibration/search_spaces.yaml.
_TGLF_GOLDENS = {
    "tglfnn": "07bc2f4f6934824175ca2709437f5d97a31fae011e6f9d1719f16a519ba20a45",
    "tglfnn_nr": "8ad1a1594cef0b8919737307c8513f9724bfb1f9813745429e80f58ba59c8e55",
    "tglfnn_spherical": "a91087f1e546b88f4b18e69e95c654691bcbc43f20aee34b2940e2e5e768de19",
}


def _digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def test_phase_profile_deduplication_preserves_merged_configs() -> None:
    envs = _CONFIGS / "envs"
    for key, expected in _PHASE_GOLDENS.items():
        phase_path = envs / f"{key}.yaml"
        with (phase_path.parent / "base.yaml").open() as stream:
            base = yaml.safe_load(stream)
        with phase_path.open() as stream:
            phase = yaml.safe_load(stream)
        phase.pop("task")
        # Binary reset snapshots are phase-owned runtime assets, not part of
        # the factored TORAX/profile mapping protected by these goldens.
        phase.pop("initialization", None)
        assert _digest(_deep_merge(base, phase)) == expected, key


def test_tglfnn_extends_preserves_backend_configs() -> None:
    backends = _CONFIGS / "backends"
    for name, expected in _TGLF_GOLDENS.items():
        assert _digest(_load_extended_yaml(backends / f"{name}.yaml")) == expected
