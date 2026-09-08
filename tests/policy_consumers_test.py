"""Saved-policy replay and plots share collection and preserve study diagnostics."""

import csv
import dataclasses
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
import pytest
from envelope import TruncationWrapper
from flax import serialization
from helpers import CheapBoundaryEnv

matplotlib.use("Agg")

from agents.policy_io import LoadedPolicy, environment_interface
from experiments.plotting import plot_open_loop as plot
from experiments.studies import replay_direct_actuators as replay
from plasmax.spaces import ObsLayout


class NamedEnv(CheapBoundaryEnv):
    def obs_layout(self) -> ObsLayout:
        return ObsLayout({}, {"elapsed_time": slice(1, 2)}, (), ("elapsed_time",))


def _env() -> Any:
    return TruncationWrapper(
        NamedEnv(obs_dim=2, action_low=(-1.0,), action_high=(1.0,), terminate_after=2),
        max_steps=3,
    )


def _saved_policy(seed: int = 0, coefficient: float = 0.0) -> LoadedPolicy:
    return LoadedPolicy(
        algorithm="backprop_open_loop",
        inference={
            "params": np.full((10, 1), coefficient, np.float32),
            "source_times": np.array([0.0, 1.0, 2.0], np.float32),
            "time_index": 1,
        },
        interface=environment_interface(_env()),
        metadata={
            "seed": seed,
            "actual_timesteps": replay._FINAL_STEP,
            "config": {
                "algorithm": "backprop_open_loop",
                "backprop": {"num_knots": 10},
                "env": {
                    "env_setup": "iter/baseline/rampup",
                    "backend": "bohm_gyrobohm",
                    "eval_seed": 12,
                },
            },
        },
        deterministic=True,
    )


def _write_policy(path: Path, policy: LoadedPolicy) -> None:
    path.write_bytes(
        serialization.msgpack_serialize(
            {
                "format_version": 1,
                **dataclasses.asdict(policy),
            }
        )
    )


def test_replay_discovers_new_artifacts_and_rejects_duplicate_seed(
    tmp_path: Path,
) -> None:
    policy = _saved_policy(seed=2, coefficient=0.5)
    _write_policy(tmp_path / "run.msgpack", policy)
    (tmp_path / "ignored.npz").write_text("not a policy format")
    runs = replay._load_saved_runs(tmp_path)
    key = ("iter/baseline/rampup", "direct_knots_10", 2)
    assert set(runs) == {key}
    np.testing.assert_allclose(runs[key].flat_parameters, 0.5, rtol=0, atol=0)
    _write_policy(tmp_path / "duplicate.msgpack", policy)
    with pytest.raises(ValueError, match="Multiple saved policies"):
        replay._load_saved_runs(tmp_path)


def test_replay_collects_each_loaded_policy_with_the_same_key(monkeypatch) -> None:
    env = _env()
    monkeypatch.setattr(replay, "load_policy_env", lambda policy: env)
    calls = []
    evaluate = replay.evaluate_policy

    def record(policy, environment, key, **kwargs):
        calls.append(jax.random.key_data(key))
        return evaluate(policy, environment, key, **kwargs)

    monkeypatch.setattr(replay, "evaluate_policy", record)
    runs = [
        replay.SavedRun(
            seed,
            Path(f"seed{seed}.msgpack"),
            _saved_policy(seed, seed / 2),
            np.array([seed]),
        )
        for seed in range(2)
    ]
    _, trajectory = replay._run_cell_replay(runs, eval_index=3)
    np.testing.assert_array_equal(calls[0], calls[1])
    np.testing.assert_array_equal(trajectory.valid, [[True, True, False]] * 2)
    np.testing.assert_array_equal(trajectory.obs[0], trajectory.obs[1])
    np.testing.assert_allclose(trajectory.action[0, :2], 0.0, rtol=0, atol=0)
    np.testing.assert_allclose(
        trajectory.action[1, :2], np.tanh(0.5), rtol=1e-6, atol=1e-6
    )
    runs[1].policy.metadata["config"]["env"]["eval_seed"] = 13
    with pytest.raises(ValueError, match="Environment mismatch"):
        replay._run_cell_replay(runs, eval_index=3)


def test_replay_preserves_requested_applied_variation_and_valid_csv(
    tmp_path: Path,
) -> None:
    command = np.array([[[0.0], [0.0], [99.0]], [[1.0], [1.0], [99.0]]])
    valid = np.array([[True, True, False], [True, True, False]])
    trajectory = replay.ReplayStep(
        time_s=np.array([[1.0, 2.0, 2.0]] * 2),
        command_normalized=command,
        command_physical=command * 10,
        applied_normalized=np.zeros_like(command),
        applied_physical=np.full_like(command, 5),
        reward=np.array([[1.0, 2.0, 999.0], [2.0, 3.0, 999.0]]),
        terminated=np.array([[False, True, False]] * 2),
        truncated=np.zeros_like(valid),
        valid=valid,
    )
    runs = [
        replay.SavedRun(
            seed, Path(f"seed{seed}.msgpack"), _saved_policy(seed), np.array([1 + seed])
        )
        for seed in range(2)
    ]
    task = replay._TASKS[0]
    cell, actuators = replay._summarize_cell(
        task,
        "direct_knots_10",
        runs,
        trajectory,
        ("P_nbi",),
        np.array([0.0]),
        np.array([10.0]),
        0,
    )
    assert cell.parameter_set == "saved"
    assert cell.return_mean == 4.0
    assert cell.common_valid_steps == 2
    assert cell.command_normalized_seed_std_max > 0
    assert cell.applied_normalized_seed_std_max == 0
    assert actuators[0].physical_unit_span == 10
    path = tmp_path / "replay.csv"
    replay._write_trajectory_csv(
        path, task, "direct_knots_10", runs, trajectory, ("P_nbi",), 0
    )
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4
    assert {row["step"] for row in rows} == {"0", "1"}
    assert float(rows[-1]["P_nbi/command_physical"]) == 10
    assert float(rows[-1]["P_nbi/applied_physical"]) == 5


def _plot_arrays() -> dict[str, np.ndarray]:
    arrays = {
        name: np.array([[1.0, 2.0, 999.0]])
        for name in (
            "time_s",
            "Ip",
            "P_fusion",
            "P_aux_total",
            "q_min",
            "fgw_n_e_line_avg",
            "beta_N",
        )
    }
    arrays["valid"] = np.array([[True, True, False]])
    arrays["command_physical"] = np.ones((1, 3, 1))
    return arrays


def test_plot_pairs_hold_reset_and_shared_evaluations(monkeypatch) -> None:
    keys = []
    initialized = []
    state = SimpleNamespace(prev_action=jnp.array([5.0]))
    env = SimpleNamespace(
        init=lambda key: (initialized.append(jax.random.key_data(key)) or state, None),
        from_physical=lambda action: action / 10,
    )
    policy = _saved_policy()

    def evaluate(controller, environment, key, **kwargs):
        keys.append(jax.random.key_data(key))
        if callable(controller):
            np.testing.assert_allclose(controller(None, None), [0.5], rtol=0, atol=0)
        return {"returns": np.array([0.0])}, _plot_arrays()

    monkeypatch.setattr(plot, "evaluate_policy", evaluate)
    monkeypatch.setattr(plot, "trajectory_arrays", lambda arrays, env: arrays)
    key = jax.random.key(3)
    plot._compare(policy, env, key)
    np.testing.assert_array_equal(keys[0], keys[1])
    reset_key = jax.random.split(jax.random.split(key, 1)[0])[0]
    np.testing.assert_array_equal(initialized[0], jax.random.key_data(reset_key))


def test_plot_uses_new_results_and_masks_padding(tmp_path: Path, monkeypatch) -> None:
    results = {
        "global_step": np.array([0, 10]),
        "evaluation": {"evaluation/return_mean": np.array([0.0, 4.0])},
    }
    policy = dataclasses.replace(_saved_policy(), results=results)
    env = SimpleNamespace(
        unwrapped=SimpleNamespace(actuator_specs=[SimpleNamespace(name="P_nbi")])
    )
    monkeypatch.setattr(plot, "load_policy", lambda path: policy)
    monkeypatch.setattr(plot, "load_policy_env", lambda policy, **kwargs: env)
    monkeypatch.setattr(
        plot,
        "_compare",
        lambda *args: (
            {"returns": np.array([4.0])},
            _plot_arrays(),
            {"returns": np.array([0.0])},
            _plot_arrays(),
        ),
    )
    arrays = _plot_arrays()
    assert plot._scalars(arrays)["end"] == 2
    np.testing.assert_array_equal(plot._scalars(arrays)["q_min"], [1.0, 2.0])
    np.testing.assert_array_equal(plot._learning_curve(results)[1], [0.0, 4.0])
    np.testing.assert_array_equal(
        plot._learning_curve(
            {
                "global_step": [0, 10],
                "evaluation": [np.ones((2, 2)), np.array([[1.0, 3.0], [4.0, 6.0]])],
            }
        )[1],
        [2.0, 5.0],
    )
    path = tmp_path / "figure.png"
    plot.main(plot.Config(policy=Path("unused.msgpack"), out=path))
    assert path.stat().st_size > 1000
