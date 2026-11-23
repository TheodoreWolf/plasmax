"""Synthetic checks for cold and warm benchmark timing boundaries."""

from __future__ import annotations

from functools import partial
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from benchmarks import env_throughput, gymtorax_throughput, world_model_throughput


def test_env_timing_separates_compilation_and_uses_first_call_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    outputs = iter(
        (None, np.array([0, 1]), np.array(counts))
        for counts in ([2, 3], [3, 4], [4, 5])
    )

    def execute(key: str) -> Any:
        assert key == "key"
        events.append("execute")
        return next(outputs)

    def compile_run() -> Any:
        events.append("compile")
        return execute

    def lower(key: str) -> Any:
        assert key == "key"
        events.append("lower")
        return SimpleNamespace(compile=compile_run)

    def run(key: str) -> Any:
        raise AssertionError("Timing must execute the already-compiled callable")

    run.lower = lower
    monkeypatch.setattr(env_throughput, "_make_run", lambda *_: partial(run, "key"))
    clock = iter([0, 2, 2, 5, 5, 10, 10, 11, 11, 14])
    monkeypatch.setattr(env_throughput.time, "perf_counter", lambda: next(clock))
    monkeypatch.setattr(env_throughput.jax, "block_until_ready", lambda value: value)
    monkeypatch.setattr(env_throughput.jax, "device_get", lambda value: value)
    cfg = env_throughput.Config(n_steps=5, repeats=2)
    timing = env_throughput._time_run(None, cfg, 2)
    assert events == ["lower", "compile", "execute", "execute", "execute"]
    np.testing.assert_allclose(
        (timing.trace_lower_s, timing.compile_s, timing.first_execution_s),
        (2, 3, 5),
        rtol=0,
        atol=0,
    )
    assert timing.first_completed_steps == 5
    assert timing.completed_steps == 9
    payload = env_throughput._timing_payload(cfg, timing)
    assert payload["compile_and_first_run_s"] == 10
    assert payload["cold_steps_per_second"] == 0.5
    assert payload["steps_per_second"] == 4.5


def test_world_model_validation_stays_outside_first_and_warm_timers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    def execute(keys: str, actions: str) -> str:
        assert (keys, actions) == ("keys", "actions")
        events.append("execute")
        return "output"

    def compile_run() -> Any:
        events.append("compile")
        return execute

    def lower(keys: str, actions: str) -> Any:
        assert (keys, actions) == ("keys", "actions")
        events.append("lower")
        return SimpleNamespace(compile=compile_run)

    prepared = SimpleNamespace(
        run=SimpleNamespace(lower=lower), keys="keys", actions="actions", metadata=None
    )
    monkeypatch.setattr(world_model_throughput, "_make_run", lambda *_: prepared)
    clock = iter([0, 2, 2, 5, 5, 10, 100, 101])
    monkeypatch.setattr(
        world_model_throughput.time, "perf_counter", lambda: next(clock)
    )
    monkeypatch.setattr(
        world_model_throughput.jax, "block_until_ready", lambda value: value
    )

    def validate(output: str, **_: Any) -> Any:
        assert output == "output"
        events.append("validate")
        return SimpleNamespace(completed_steps=20)

    monkeypatch.setattr(world_model_throughput, "_validate_output", validate)
    env = SimpleNamespace(observation_space=SimpleNamespace(shape=(15,)))
    cfg = world_model_throughput.Config(n_steps=10, repeats=1)
    case = world_model_throughput._time_run(env, cfg, 2)
    assert events == ["lower", "compile", "execute", "validate", "execute", "validate"]
    assert case.timing.compile_and_first_run_s == 10
    assert case.timing.first_completed_steps == 20
    assert case.timing.durations_s == (1,)
    assert len(case.repeat_validations) == 1


def test_gym_compilation_listener_filters_events_and_unregisters_on_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def external_listener(*_: Any, **__: Any) -> None:
        pass

    listeners = [external_listener]
    monkeypatch.setattr(
        gymtorax_throughput.monitoring,
        "register_event_duration_secs_listener",
        listeners.append,
    )
    monkeypatch.setattr(
        gymtorax_throughput.monitoring,
        "unregister_event_duration_listener",
        listeners.remove,
    )
    with (
        pytest.raises(RuntimeError, match="rollout failed"),
        gymtorax_throughput._compilation_timer() as timing,
    ):
        listener = listeners[-1]
        listener(gymtorax_throughput._COMPILE_EVENT, 0.25, fun_name="first")
        listener("/jax/core/compile/jaxpr_trace_duration", 10.0)
        listener(gymtorax_throughput._COMPILE_EVENT, 0.75, fun_name="second")
        assert timing.duration_s == 1.0
        assert timing.events == 2
        raise RuntimeError("rollout failed")
    assert listeners == [external_listener]


def test_gym_rollout_times_reset_separately_and_counts_actual_intervals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeEnv:
        delta_t_a = 0.1

        def __init__(self) -> None:
            self.torax_app = SimpleNamespace(
                current_sim_state=SimpleNamespace(t=0.0), current_sim_output=None
            )
            self.calls = 0

        def reset(self, seed: int) -> None:
            assert seed == 0

        def step(self, action: dict[str, Any]) -> tuple:
            action.pop("power")
            self.calls += 1
            self.torax_app.current_sim_state.t += self.delta_t_a
            return None, 1.0, self.calls == 2, False, {}

    clock = iter([0, 2, 2, 5, 5, 9])
    monkeypatch.setattr(gymtorax_throughput.time, "perf_counter", lambda: next(clock))
    monkeypatch.setattr(
        gymtorax_throughput.jax, "block_until_ready", lambda value: value
    )
    actions = [{"power": 1.0} for _ in range(3)]
    timing = gymtorax_throughput._rollout(FakeEnv(), actions, 0)
    assert timing.reset_duration_s == 2
    assert timing.duration_s == 7
    assert timing.completed_steps == timing.attempted_steps == 2
    assert timing.terminated
    assert actions == [{"power": 1.0}] * 3
