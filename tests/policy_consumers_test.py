"""Saved-policy plots share collection and mask invalid trajectory steps."""

import dataclasses
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
from envelope import TruncationWrapper
from helpers import CheapBoundaryEnv

matplotlib.use("Agg")

from agents.policy_io import LoadedPolicy, environment_interface
from experiments.plotting import plot_open_loop as plot
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
