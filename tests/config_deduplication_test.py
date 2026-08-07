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
# reference migration. Task metadata is intentionally omitted because this test
# protects lossless YAML factoring rather than task-level reward settings.
_PHASE_GOLDENS = {
    "iter/advanced/rampup": "acf79107e863c87967b21bb310d73c959c4654d81cbc2cc54e148e203d9fbddb",
    "iter/advanced/flattop": "a5671c179312267c10de12c58cf6ce3d606f2351c606325bb904ab65d91a37a9",
    "iter/advanced/rampdown": "82d2fb33f2e616586f22a6b851dbc56e9eee0011905926e13d02d515e744f098",
    "iter/baseline/rampup": "412685ca41bb8889ef7c01178279b44cdace92ce2ef19ce102b2e75e854b003e",
    "iter/baseline/flattop": "779eaa7aa2c952c1c3a924db8de9f7142faf9458b4c3e777d515a6b56f9de450",
    "iter/baseline/rampdown": "61b0d8eb6afef9e9a57390d647ff2e7661ac26656908b512ef3724d9eed9e969",
    "iter/hybrid/rampup": "e8d1ce24d7139e18f0259b6d8be8b5d3b18bbec45a25f9f56211229aaf7a5d49",
    "iter/hybrid/flattop": "281193bdd71c13a650d202bd0fefabd1676dfad6fd8f99e03ceace53cf94e493",
    "iter/hybrid/rampdown": "f091272fc308eedd311f34273c70c532af979a0c270498fa43c071786f39fdd6",
    "sparc/prd/rampup": "cc754b900698910bdb13cbb36a1dbf4c966de21717c87d405d348b08524a4df6",
    "sparc/prd/flattop": "79c26e9b9d056a0ff4ee0128230df27d5414dd386362af3b7209268e1da5f86a",
    "sparc/prd/rampdown": "44214a8a87e0f0ce73c0491815ee5227cdf656b23f6bce754c41d527386885d1",
    "sparc/reduced_field/rampup": "aa19bc8c79ddd499bfd319d4fb22f02f87900cd492b130bfb558777e2f62befc",
    "sparc/reduced_field/flattop": "0da83c6ec1baf0b4a92179e8c38298871f50d711fd8e36edb940a932791555c9",
    "sparc/reduced_field/rampdown": "fb8d13bfd0dab6b3ee18542d79ee8ce086eef9a861251d32319f589a57c6658a",
}

# SHA-256 of each complete backend mapping after the calibration search
# spaces moved to tools/calibration/search_spaces.yaml.
_TGLF_GOLDENS = {
    "tglfnn": "1431bdfa64689ebc1004a1e9f71abeb9c059059d5a3c130a85d2f0c9c206f9f2",
    "tglfnn_nr": "c5b2eaf6be89ff901b4d6082c4930b2c509050fe1b7663ed421b67405be303c2",
    "tglfnn_spherical": "4bb6499b2abca46ecce4a786cfc0f664780be372fb6efc164c959afdfa55a9af",
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
        assert _digest(_deep_merge(base, phase)) == expected, key


def test_tglfnn_extends_preserves_backend_configs() -> None:
    backends = _CONFIGS / "backends"
    for name, expected in _TGLF_GOLDENS.items():
        assert _digest(_load_extended_yaml(backends / f"{name}.yaml")) == expected
