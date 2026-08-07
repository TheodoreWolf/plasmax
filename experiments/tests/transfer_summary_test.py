"""Contracts for the six-method transfer comparison summarizer."""

import json

from experiments.studies.summarize_transfer_comparison import (
    _algorithm_and_variant,
    load_rows,
)


def _payload(target_backend="tglfnn", source_backend="bohm_gyrobohm"):
    return {
        "metrics": {
            "transfer/source_backend": source_backend,
            "transfer/target_backend": target_backend,
            "transfer/source_return_mean": 100.0,
            "transfer/source_return_seed_std": 5.0,
            "transfer/target_return_mean": 10.0,
            "transfer/target_return_seed_std": 1.0,
            "transfer/return_gap": -90.0,
            "transfer/return_ratio": 0.1,
            "transfer/target_episode_length_mean": 440.0,
        }
    }


def test_algorithm_inference_does_not_confuse_knot_substrings(tmp_path):
    path = tmp_path / "direct" / "direct_knots_100_realistic.json"

    assert _algorithm_and_variant(path) == ("direct_knots_100", "realistic")


def test_load_rows_reads_nested_direct_summary(tmp_path):
    path = tmp_path / "direct" / "direct_knots_10_oracle.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(_payload()))

    rows = load_rows(tmp_path)

    assert len(rows) == 1
    assert rows[0]["algorithm"] == "direct_knots_10"
    assert rows[0]["training_seeds"] == 10
    assert rows[0]["target_horizon_fraction"] == 0.1


def test_load_rows_validates_requested_target_backend(tmp_path):
    path = tmp_path / "direct" / "direct_policy_realistic.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(_payload("qlknn")))

    rows = load_rows(tmp_path, "qlknn")

    assert len(rows) == 1
    assert rows[0]["target_return"] == 10.0


def test_load_rows_validates_requested_source_backend(tmp_path):
    path = tmp_path / "direct" / "direct_policy_oracle.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(_payload("bohm_gyrobohm", "cgm")))

    rows = load_rows(tmp_path, "bohm_gyrobohm", "cgm")

    assert len(rows) == 1
    assert rows[0]["source_return"] == 100.0
