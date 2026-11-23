"""Tests for backend-agreement aggregation and failure reporting."""

import json

import numpy as np
import pytest

from benchmarks import backend_agreement
from benchmarks.backend_agreement import (
    Config,
    Rollout,
    _elementwise_error,
    _selected_sensor_mean,
    _sensor_balanced_mean,
    _throughput_sps,
    _validate_solver_error_states,
)
from plasmax.spaces import ObsLayout


def test_sensor_balanced_mean_averages_profile_before_sensors():
    layout = ObsLayout(
        profile_slices={"profile": slice(0, 2)},
        scalar_slices={"scalar": slice(2, 3)},
        profile_names=("profile",),
        scalar_names=("scalar",),
    )
    relative_error = np.array(
        [
            [0.0, 2.0, 6.0],
            [2.0, 4.0, 0.0],
        ]
    )

    np.testing.assert_allclose(
        _sensor_balanced_mean(relative_error, layout),
        2.5,
        atol=0.0,
        rtol=0.0,
    )


def test_sensor_balanced_mean_can_select_nonredundant_sensors():
    layout = ObsLayout(
        profile_slices={"profile": slice(0, 2)},
        scalar_slices={"scalar": slice(2, 3)},
        profile_names=("profile",),
        scalar_names=("scalar",),
    )
    elementwise_error = np.array([[0.0, 2.0, 6.0], [2.0, 4.0, 0.0]])

    np.testing.assert_allclose(
        _sensor_balanced_mean(elementwise_error, layout, ("profile",)),
        2.0,
        atol=0.0,
        rtol=0.0,
    )


def test_selected_sensor_mean_rejects_unknown_sensor():
    with pytest.raises(ValueError, match="unknown metric sensors"):
        _selected_sensor_mean({"T_e": 1.0}, ("T_i",))


def test_mse_has_no_reference_denominator():
    observations = np.array([[1.0, 3.0]])
    reference = np.array([[0.0, 1.0]])

    np.testing.assert_array_equal(
        _elementwise_error(
            observations,
            reference,
            metric="mse",
            floor_fraction=1.0e-3,
        ),
        np.array([[1.0, 4.0]]),
    )


def test_explicit_throughput_matches_backend_order():
    cfg = Config(
        backends=("bohm_gyrobohm", "tglfnn"),
        cpu_scalar_sps=(650.0, 9.0),
    )

    assert _throughput_sps(cfg) == {
        "bohm_gyrobohm": 650.0,
        "tglfnn": 9.0,
    }


def test_measured_throughput_uses_median_warm_run():
    cfg = Config(
        backends=("tglfnn_nr",),
        n_steps=100,
        timing_repeats=3,
        require_cpu=False,
    )
    rollouts = {
        "tglfnn_nr": Rollout(
            observations=np.empty((0, 0)),
            boundaries=np.empty(0),
            control_step_complete=np.ones(100, dtype=bool),
            solver_iterations=np.empty(0),
            solver_error_states=np.empty(0),
            elapsed_s=1.0,
            timing_durations_s=(5.0, 4.0, 6.0),
        )
    }

    assert _throughput_sps(cfg, rollouts) == {"tglfnn_nr": 20.0}


def test_measured_throughput_counts_only_completed_control_intervals():
    cfg = Config(
        backends=("tglfnn_nr",),
        n_steps=100,
        timing_repeats=1,
        require_cpu=False,
    )
    rollouts = {
        "tglfnn_nr": Rollout(
            observations=np.empty((0, 0)),
            boundaries=np.empty(0),
            control_step_complete=np.array([True, False, True]),
            solver_iterations=np.empty(0),
            solver_error_states=np.empty(0),
            elapsed_s=1.0,
            timing_durations_s=(0.5,),
        )
    }

    assert _throughput_sps(cfg, rollouts) == {"tglfnn_nr": 4.0}


def test_solver_coarse_convergence_is_accepted():
    _validate_solver_error_states(np.array([0, 2, 0, 2]))


def test_solver_failure_is_rejected_with_state_and_indices():
    with pytest.raises(RuntimeError, match=r"\{1: \[1, 3\]\}"):
        _validate_solver_error_states(np.array([0, 1, 2, 1]))


@pytest.mark.parametrize(
    ("output_index", "invalid_value", "error_message"),
    [
        (0, np.nan, "non-finite observations"),
        (1, True, "rollout terminated"),
        (2, False, "did not complete the physical control interval"),
        (4, 1, "solver reported failed or unknown states"),
    ],
)
def test_invalid_initial_rollout_is_rejected_before_warm_repeats(
    output_index, invalid_value, error_message, capsys
):
    outputs = (
        np.zeros((2, 4)),
        np.zeros(2, dtype=bool),
        np.ones(2, dtype=bool),
        np.ones(2, dtype=np.int32),
        np.zeros(2, dtype=np.int32),
    )
    outputs[output_index][0] = invalid_value
    calls = 0

    def run(_key):
        nonlocal calls
        calls += 1
        return outputs

    with pytest.raises(RuntimeError, match=error_message):
        backend_agreement._collect_rollout(
            object(),
            seed=8,
            n_steps=2,
            fail_on_boundary=True,
            timing_repeats=3,
            runner=run,
        )

    assert calls == 1
    progress = capsys.readouterr().out
    assert "seed=8 initial rollout:" in progress
    assert "warm repeat" not in progress


@pytest.mark.parametrize("cpu_scalar_sps", [(1.0, 2.0, 3.0), ()])
def test_multi_seed_failures_are_reported_without_pruning_other_pairs(
    monkeypatch, tmp_path, cpu_scalar_sps
):
    layout = ObsLayout(
        profile_slices={
            name: slice(index, index + 1)
            for index, name in enumerate(("T_e", "T_i", "n_e", "q"))
        },
        scalar_slices={},
        profile_names=("T_e", "T_i", "n_e", "q"),
        scalar_names=(),
    )

    class FakeEnv:
        def __init__(self, backend):
            self.backend = backend
            self.scalar_obs_specs = ()

        def obs_layout(self):
            return layout

    monkeypatch.setattr(
        backend_agreement,
        "make",
        lambda _env, backend: FakeEnv(backend),
    )
    monkeypatch.setattr(
        backend_agreement,
        "_make_rollout_runner",
        lambda env, _n_steps: env,
    )

    monkeypatch.setattr(
        backend_agreement, "PhysicsRandomizationWrapper", lambda env: env
    )

    values = {"reference": 0.0, "healthy": 1.0, "partial": 2.0}

    def collect(env, seed, *_args, **_kwargs):
        if env.backend == "partial" and seed == 1:
            raise RuntimeError("simulated backend failure")
        observations = np.full((2, 4), values[env.backend], dtype=np.float32)
        return Rollout(
            observations=observations,
            boundaries=np.zeros(2, dtype=bool),
            control_step_complete=np.ones(2, dtype=bool),
            solver_iterations=np.ones(2, dtype=np.int32),
            solver_error_states=np.zeros(2, dtype=np.int32),
            elapsed_s=0.1,
            timing_durations_s=(0.2, 0.1, 0.3) if _args[2] else (),
        )

    monkeypatch.setattr(backend_agreement, "_collect_rollout", collect)
    output = tmp_path / "agreement.json"
    backend_agreement.main(
        Config(
            backends=("reference", "healthy", "partial"),
            reference_backend="reference",
            seeds=(0, 1),
            n_steps=2,
            cpu_scalar_sps=cpu_scalar_sps,
            timing_repeats=3,
            output=output,
            require_cpu=False,
        )
    )

    result = json.loads(output.read_text())
    assert result["successful_seeds"] == {
        "reference": [0, 1],
        "healthy": [0, 1],
        "partial": [0],
    }
    assert result["failed_runs"] == {
        "reference": {},
        "healthy": {},
        "partial": {"1": "simulated backend failure"},
    }
    assert result["metrics"]["healthy"]["paired_seeds"] == [0, 1]
    assert result["metrics"]["partial"]["paired_seeds"] == [0]
    assert result["comparison_mode"] == "registered"
    assert result["shared_stack"] is None
    assert result["physics_randomization_enabled"] is False
    assert result["timing"]["partial"]["completed_control_intervals_by_seed"] == {
        "0": 2
    }
    if cpu_scalar_sps:
        assert result["timing"]["healthy"]["source"] == "provided"
        assert result["timing"]["healthy"]["warm_durations_s"] == []
    else:
        assert result["timing"]["healthy"]["source"] == "measured"
        assert result["timing"]["healthy"]["warm_durations_s"] == [0.2, 0.1, 0.3]
        assert result["metrics"]["healthy"]["sps"] == 10.0


def test_registered_mode_rejects_ignored_controlled_options():
    with pytest.raises(ValueError, match="require comparison_mode='controlled'"):
        backend_agreement._validate(Config(fixed_dt_s=0.5, require_cpu=False))


@pytest.mark.parametrize("physics_randomization", [False, True])
def test_registered_mode_applies_only_requested_physics_wrapper(
    monkeypatch, physics_randomization
):
    bare_env = object()
    wrapped_env = object()
    monkeypatch.setattr(backend_agreement, "make", lambda _env, _backend: bare_env)

    def wrap(env):
        assert env is bare_env
        return wrapped_env

    monkeypatch.setattr(backend_agreement, "PhysicsRandomizationWrapper", wrap)
    build, metadata = backend_agreement._comparison_setup(
        Config(physics_randomization=physics_randomization)
    )
    assert build("cgm") is (wrapped_env if physics_randomization else bare_env)
    assert metadata["physics_randomization_enabled"] is physics_randomization


def test_controlled_mode_reports_required_migration():
    with pytest.raises(NotImplementedError, match="require migration"):
        backend_agreement._comparison_setup(Config(comparison_mode="controlled"))
