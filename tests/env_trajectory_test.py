"""Fast tests for the environment trajectory benchmark and report tooling.

These tests use synthetic reports and mocked environment runs.  They must not
execute the full environment/backend matrix.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks import env_trajectory


def _runner(*groups: str) -> env_trajectory.RunnerMetadata:
    return env_trajectory.RunnerMetadata(
        groups=groups or ("mock",),
        platform="test-platform",
        python_version="3.12.0",
        plasmax_version="0.1.0",
        torax_version="1.4.3",
        jax_version="0.7.0",
        devices=("TFRT_CPU_0",),
    )


def _ok_result(
    key: env_trajectory.CaseKey,
    *,
    creation_seconds: float = 1.0,
    first_trajectory_seconds: float = 10.0,
    steps: int = 5,
    boundary: str = "terminated",
    termination_code: int = 0,
) -> env_trajectory.CaseResult:
    environment, backend, variant = key
    return env_trajectory.CaseResult(
        environment=environment,
        backend=backend,
        variant=variant,
        status="ok",
        creation_seconds=creation_seconds,
        first_trajectory_seconds=first_trajectory_seconds,
        steps=steps,
        boundary=boundary,
        termination_code=termination_code,
        error_type=None,
        error_message=None,
    )


def _error_result(
    key: env_trajectory.CaseKey,
    *,
    error_type: str = "RuntimeError",
    error_message: str = "synthetic failure",
) -> env_trajectory.CaseResult:
    environment, backend, variant = key
    return env_trajectory.CaseResult(
        environment=environment,
        backend=backend,
        variant=variant,
        status="error",
        creation_seconds=0.5,
        first_trajectory_seconds=1.0,
        steps=None,
        boundary=None,
        termination_code=None,
        error_type=error_type,
        error_message=error_message,
    )


def _report(
    results: tuple[env_trajectory.CaseResult, ...],
    *,
    group: str = "mock",
    revision: str = "abc123",
    complete: bool = True,
    expected_cases: int | None = None,
    generated_at_utc: str = "2026-09-01T00:00:00+00:00",
) -> env_trajectory.TrajectoryReport:
    keys = tuple(result.key for result in results)
    return env_trajectory.TrajectoryReport(
        schema_version=env_trajectory.SCHEMA_VERSION,
        complete=complete,
        revision=revision,
        generated_at_utc=generated_at_utc,
        runners=(_runner(group),),
        selection=env_trajectory.Selection(
            groups=(group,),
            environments=tuple(sorted({key[0] for key in keys})),
            backends=tuple(sorted({key[1] for key in keys})),
            variants=tuple(sorted({key[2] for key in keys})),
            excluded_backends=env_trajectory.EXCLUDED_BACKENDS,
            expected_cases=(len(results) if expected_cases is None else expected_cases),
            full_matrix=False,
        ),
        results=results,
    )


def _write_json(report: env_trajectory.TrajectoryReport, path: Path) -> None:
    env_trajectory.write_report(report, path)


def test_full_case_selection_and_seven_groups() -> None:
    assert len(env_trajectory.FULL_CASES) == 126
    assert all(
        backend not in env_trajectory.EXCLUDED_BACKENDS
        for _, backend, _ in env_trajectory.FULL_CASES
    )

    group_counts = {
        group: len(env_trajectory.select_cases(env_trajectory.RunConfig(group=group)))
        for group in env_trajectory.GROUPS
    }
    assert group_counts == {
        "iter-baseline": 24,
        "iter-hybrid": 24,
        "iter-advanced": 24,
        "sparc-prd": 24,
        "sparc-reduced-field": 24,
        "step": 4,
        "mock": 2,
    }
    selected = {
        case
        for group in env_trajectory.GROUPS
        for case in env_trajectory.select_cases(env_trajectory.RunConfig(group=group))
    }
    assert selected == set(env_trajectory.FULL_CASES)


def test_case_selection_filters_and_rejects_invalid_filters() -> None:
    selected = env_trajectory.select_cases(
        env_trajectory.RunConfig(
            environments=("iter/hybrid/flattop",),
            backends=("cgm",),
            variants=("realistic",),
        )
    )
    assert selected == (("iter/hybrid/flattop", "cgm", "realistic"),)

    with pytest.raises(ValueError, match="Unknown or unsupported environments"):
        env_trajectory.select_cases(
            env_trajectory.RunConfig(environments=("not/an/environment",))
        )
    with pytest.raises(ValueError, match="Unknown or excluded backends"):
        env_trajectory.select_cases(env_trajectory.RunConfig(backends=("tglfnn_nr",)))
    with pytest.raises(ValueError, match="variants must not be empty"):
        env_trajectory.select_cases(env_trajectory.RunConfig(variants=()))


def test_run_writes_incrementally_continues_and_finishes_nonzero(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cases = (
        ("mock/default", "fixed", "oracle"),
        ("mock/default", "fixed", "realistic"),
    )
    outcomes = (_error_result(cases[0]), _ok_result(cases[1]))
    snapshots: list[env_trajectory.TrajectoryReport] = []
    original_write = env_trajectory.write_report

    def record_write(
        report: env_trajectory.TrajectoryReport,
        path: Path,
    ) -> None:
        snapshots.append(report)
        original_write(report, path)

    calls: list[env_trajectory.CaseKey] = []

    def fake_run_case(
        environment: str,
        backend: str,
        variant: str,
    ) -> env_trajectory.CaseResult:
        calls.append((environment, backend, variant))
        return outcomes[len(calls) - 1]

    monkeypatch.setattr(env_trajectory, "select_cases", lambda _: cases)
    monkeypatch.setattr(env_trajectory, "_runner_metadata", lambda _: _runner())
    monkeypatch.setattr(env_trajectory, "run_case", fake_run_case)
    monkeypatch.setattr(env_trajectory, "write_report", record_write)
    monkeypatch.setattr(env_trajectory.jax, "clear_caches", lambda: None)
    monkeypatch.setattr(env_trajectory.gc, "collect", lambda: 0)

    output = tmp_path / "nested" / "results.json"
    status = env_trajectory.run_trajectories(
        env_trajectory.RunConfig(output=output, revision="pr456")
    )

    assert status == 1
    assert calls == list(cases)
    assert [(item.complete, len(item.results)) for item in snapshots] == [
        (False, 0),
        (False, 1),
        (False, 2),
        (True, 2),
    ]
    final = env_trajectory.load_report(output)
    assert final.complete is True
    assert final.revision == "pr456"
    assert tuple(result.status for result in final.results) == ("error", "ok")
    assert not tuple(output.parent.glob(f".{output.name}.*.tmp"))


def test_run_case_records_construction_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_make(*_args: object, **_kwargs: object) -> None:
        raise ValueError("cannot construct")

    monkeypatch.setattr(env_trajectory.plasmax, "make", fail_make)

    result = env_trajectory.run_case("mock/default", "fixed", "oracle")

    assert result.status == "error"
    assert result.creation_seconds is not None
    assert result.first_trajectory_seconds is None
    assert result.error_type == "ValueError"
    assert result.error_message == "cannot construct"


def test_run_case_records_trajectory_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(env_trajectory.plasmax, "make", lambda *_a, **_k: object())
    monkeypatch.setattr(env_trajectory, "OracleWrappers", lambda env: env)

    def fail_trajectory(_environment: object) -> None:
        raise ArithmeticError("cannot step")

    monkeypatch.setattr(env_trajectory, "_run_to_boundary", fail_trajectory)

    result = env_trajectory.run_case("mock/default", "fixed", "oracle")

    assert result.status == "error"
    assert result.creation_seconds is not None
    assert result.first_trajectory_seconds is not None
    assert result.error_type == "ArithmeticError"
    assert result.error_message == "cannot step"


def test_run_case_records_no_boundary_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(env_trajectory.plasmax, "make", lambda *_a, **_k: object())
    monkeypatch.setattr(env_trajectory, "OracleWrappers", lambda env: env)
    info = SimpleNamespace(terminated=False, truncated=False, termination_code=0)
    monkeypatch.setattr(
        env_trajectory,
        "_run_to_boundary",
        lambda _environment: (info, 5),
    )

    result = env_trajectory.run_case("mock/default", "fixed", "oracle")

    assert result.status == "error"
    assert result.error_type == "RuntimeError"
    assert result.error_message == "trajectory did not reach a boundary"


def test_json_round_trip_and_duplicate_rejection(tmp_path: Path) -> None:
    case = _ok_result(("mock/default", "fixed", "oracle"))
    path = tmp_path / "report.json"
    _write_json(_report((case,)), path)
    assert env_trajectory.load_report(path).results == (case,)

    payload = json.loads(path.read_text())
    payload["selection"]["expected_cases"] = 2
    payload["results"].append(payload["results"][0])
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="duplicate cases"):
        env_trajectory.load_report(path)


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), -0.1])
def test_write_or_load_rejects_invalid_timings(
    bad_value: float,
    tmp_path: Path,
) -> None:
    result = _ok_result(
        ("mock/default", "fixed", "oracle"),
        creation_seconds=bad_value,
    )
    path = tmp_path / "report.json"
    if bad_value < 0:
        _write_json(_report((result,)), path)
        with pytest.raises(ValueError, match="finite and non-negative"):
            env_trajectory.load_report(path)
    else:
        with pytest.raises(ValueError, match="Out of range float values"):
            _write_json(_report((result,)), path)
        assert not path.exists()


def test_load_rejects_malformed_unsupported_and_wrong_length_reports(
    tmp_path: Path,
) -> None:
    malformed = tmp_path / "malformed.json"
    malformed.write_text("not json")
    with pytest.raises(ValueError, match="Could not read trajectory report"):
        env_trajectory.load_report(malformed)

    valid = tmp_path / "valid.json"
    _write_json(
        _report(
            (_ok_result(("mock/default", "fixed", "oracle")),),
            expected_cases=2,
            complete=False,
        ),
        valid,
    )
    payload = json.loads(valid.read_text())
    payload["schema_version"] = 99
    unsupported = tmp_path / "unsupported.json"
    unsupported.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="Unsupported trajectory schema"):
        env_trajectory.load_report(unsupported)

    payload["schema_version"] = env_trajectory.SCHEMA_VERSION
    payload["complete"] = True
    wrong_length = tmp_path / "wrong-length.json"
    wrong_length.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="expected 2 cases"):
        env_trajectory.load_report(wrong_length)

    payload["selection"]["expected_cases"] = 1
    payload["selection"]["environments"] = ["different/environment"]
    wrong_selection = tmp_path / "wrong-selection.json"
    wrong_selection.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="selection.environments"):
        env_trajectory.load_report(wrong_selection)


def test_comparison_is_order_independent_and_finds_added_removed_cases() -> None:
    key_a = ("env/a", "backend", "oracle")
    key_b = ("env/b", "backend", "oracle")
    key_c = ("env/c", "backend", "oracle")
    result_a = _ok_result(key_a)
    result_b = _ok_result(key_b)
    result_c = _ok_result(key_c)

    reordered = env_trajectory.compare_reports(
        _report((result_a, result_b)),
        _report((result_b, result_a), revision="def456"),
    )
    assert reordered.unchanged == 2
    assert reordered.behavior_changes == ()

    changed = env_trajectory.compare_reports(
        _report((result_a, result_b)),
        _report((result_c, result_a), revision="def456"),
    )
    changes = {(change.key, change.description) for change in changed.behavior_changes}
    assert changes == {
        (key_b, "case removed"),
        (key_c, "case added"),
    }


def test_comparison_finds_exact_behavior_changes_and_current_errors() -> None:
    behavior_key = ("env/behavior", "backend", "oracle")
    error_key = ("env/error", "backend", "oracle")
    baseline = _report(
        (
            _ok_result(behavior_key),
            _ok_result(error_key),
        )
    )
    current = _report(
        (
            _error_result(error_key),
            _ok_result(
                behavior_key,
                steps=6,
                boundary="truncated",
                termination_code=7,
            ),
        ),
        revision="def456",
    )

    comparison = env_trajectory.compare_reports(baseline, current)

    assert comparison.unchanged == 0
    assert comparison.errors == (_error_result(error_key),)
    assert comparison.behavior_changes == (
        env_trajectory.BehaviorChange(
            behavior_key,
            "steps: 5 → 6; boundary: terminated → truncated; termination code: 0 → 7",
        ),
        env_trajectory.BehaviorChange(
            error_key,
            "status: ok → error; steps: 5 → None; "
            "boundary: terminated → None; termination code: 0 → None",
        ),
    )


def test_timing_warnings_require_both_thresholds() -> None:
    exact_key = ("env/exact", "backend", "oracle")
    ratio_short_key = ("env/ratio-short", "backend", "oracle")
    seconds_small_key = ("env/seconds-small", "backend", "oracle")
    baseline = _report(
        (
            _ok_result(exact_key, first_trajectory_seconds=8.0),
            _ok_result(ratio_short_key, first_trajectory_seconds=8.0),
            _ok_result(seconds_small_key, first_trajectory_seconds=1.0),
        )
    )
    current = _report(
        (
            _ok_result(exact_key, first_trajectory_seconds=10.0),
            _ok_result(ratio_short_key, first_trajectory_seconds=9.99),
            _ok_result(seconds_small_key, first_trajectory_seconds=2.25),
        ),
        revision="def456",
    )

    comparison = env_trajectory.compare_reports(baseline, current)

    assert comparison.timing_warnings == (
        env_trajectory.TimingWarning(
            key=exact_key,
            metric="first_trajectory_seconds",
            baseline_seconds=8.0,
            current_seconds=10.0,
        ),
    )


def test_zero_second_baseline_can_warn_without_dividing_by_zero() -> None:
    key = ("env/zero", "backend", "oracle")
    baseline = _report((_ok_result(key, creation_seconds=0.0),))
    current = _report(
        (_ok_result(key, creation_seconds=2.0),),
        revision="def456",
    )

    comparison = env_trajectory.compare_reports(baseline, current)
    markdown = env_trajectory.render_markdown(current, baseline=baseline)

    assert comparison.timing_warnings == (
        env_trajectory.TimingWarning(
            key=key,
            metric="creation_seconds",
            baseline_seconds=0.0,
            current_seconds=2.0,
        ),
    )
    assert "+2.000s (baseline was 0s)" in markdown


def test_markdown_contains_commits_stale_warning_counts_and_escaped_cells() -> None:
    key = ("mock/a|b\nnext<script>", "fixed", "oracle")
    baseline = _report(
        (_ok_result(key, first_trajectory_seconds=8.0),),
        revision="base-old",
    )
    current = _report(
        (
            _error_result(
                key,
                error_message="bad | row\nsecond line `code` <tag>",
            ),
        ),
        revision="pr-new",
    )

    markdown = env_trajectory.render_markdown(
        current,
        baseline=baseline,
        expected_baseline_revision="base-wanted",
    )

    assert "Current revision: `pr-new`" in markdown
    assert "Baseline revision: `base-old`" in markdown
    assert "exact PR base baseline was unavailable" in markdown
    assert "Expected `base-wanted` but used `base-old`" in markdown
    assert "| 0 | 1 | 0 | 1 |" in markdown
    assert "mock/a\\|b next&lt;script&gt;" in markdown
    escaped_error = "RuntimeError: bad \\| row second line &#96;code&#96; &lt;tag&gt;"
    assert escaped_error in markdown
    assert "<script>" not in markdown


def test_merge_all_seven_shards_and_detect_missing_shard() -> None:
    reports = []
    for group in env_trajectory.GROUPS:
        group_results = tuple(
            _ok_result(case)
            for case in env_trajectory.FULL_CASES
            if env_trajectory._group_for_environment(case[0]) == group
        )
        reports.append(_report(group_results, group=group))

    merged = env_trajectory.merge_reports(reports, env_trajectory.GROUPS)

    assert merged.complete is True
    assert merged.selection.full_matrix is True
    assert merged.selection.expected_cases == 126
    assert set(merged.selection.groups) == set(env_trajectory.GROUPS)
    assert {result.key for result in merged.results} == set(env_trajectory.FULL_CASES)

    with pytest.raises(ValueError, match=r"missing=\['mock'\]"):
        env_trajectory.merge_reports(reports[:-1], env_trajectory.GROUPS)


def test_merge_rejects_incomplete_and_duplicate_shards() -> None:
    key = ("mock/default", "fixed", "oracle")
    incomplete = _report((_ok_result(key),), complete=False, expected_cases=2)
    with pytest.raises(ValueError, match="Incomplete trajectory shards"):
        env_trajectory.merge_reports((incomplete,))

    first = _report((_ok_result(key),), group="mock")
    duplicate_group = _report(
        (_ok_result(("mock/other", "fixed", "oracle")),),
        group="mock",
    )
    with pytest.raises(ValueError, match="groups are duplicated"):
        env_trajectory.merge_reports((first, duplicate_group))

    duplicate_case = dataclasses.replace(
        first,
        runners=(_runner("step"),),
        selection=dataclasses.replace(first.selection, groups=("step",)),
    )
    with pytest.raises(ValueError, match="duplicate cases"):
        env_trajectory.merge_reports((first, duplicate_case))


def test_comparison_rejects_incomplete_or_erroneous_baseline() -> None:
    key = ("mock/default", "fixed", "oracle")
    current = _report((_ok_result(key),), revision="def456")
    incomplete = _report(
        (_ok_result(key),),
        complete=False,
        expected_cases=2,
    )
    with pytest.raises(ValueError, match="Baseline trajectory report is incomplete"):
        env_trajectory.compare_reports(incomplete, current)

    erroneous = _report((_error_result(key),))
    with pytest.raises(ValueError, match="contains failed cases"):
        env_trajectory.compare_reports(erroneous, current)


def test_build_report_has_nonblocking_timing_warnings_and_stale_baseline(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    key = ("mock/default", "fixed", "oracle")
    current_path = tmp_path / "current.json"
    baseline_path = tmp_path / "baseline.json"
    output = tmp_path / "out" / "current.json"
    markdown_output = tmp_path / "out" / "summary.md"
    github_output = tmp_path / "github-output"
    github_output.write_text("existing=value\n")
    monkeypatch.setenv("GITHUB_OUTPUT", str(github_output))
    _write_json(
        _report(
            (
                _ok_result(
                    key, creation_seconds=3.0, first_trajectory_seconds=10.0, steps=6
                ),
            ),
            revision="pr-commit",
        ),
        current_path,
    )
    _write_json(
        _report(
            (_ok_result(key, first_trajectory_seconds=8.0),),
            revision="old-main",
        ),
        baseline_path,
    )

    status = env_trajectory.build_report(
        env_trajectory.ReportConfig(
            inputs=current_path,
            output=output,
            markdown_output=markdown_output,
            baseline=baseline_path,
            expected_baseline_revision="exact-base",
            require_baseline=True,
        )
    )

    assert status == 0
    assert env_trajectory.load_report(output).revision == "pr-commit"
    markdown = markdown_output.read_text()
    assert "> **Warning:** 1 behavior change(s) and 1 possible slowdown(s)." in markdown
    assert "Behavior changes" in markdown
    assert "Possible slowdowns" in markdown
    assert "exact PR base baseline was unavailable" in markdown
    assert github_output.read_text() == (
        "existing=value\nbehavior_changes=1\npossible_slowdowns=1\n"
    )


@pytest.mark.parametrize(
    ("scenario", "expected_status", "expected_output"),
    [
        ("unchanged", 0, "behavior_changes=0\npossible_slowdowns=0\n"),
        ("current-error", 1, ""),
        ("no-baseline", 0, ""),
        ("local", 0, ""),
    ],
)
def test_build_report_emits_outputs_only_for_successful_comparisons(
    scenario: str,
    expected_status: int,
    expected_output: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    key = ("mock/default", "fixed", "oracle")
    current_path = tmp_path / "current.json"
    baseline_path = tmp_path / "baseline.json"
    github_output = tmp_path / "github-output"
    github_output.write_text("")
    if scenario == "local":
        monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
    else:
        monkeypatch.setenv("GITHUB_OUTPUT", str(github_output))
    result = _error_result(key) if scenario == "current-error" else _ok_result(key)
    _write_json(_report((result,), revision="pr-commit"), current_path)
    _write_json(_report((_ok_result(key),), revision="main-commit"), baseline_path)

    status = env_trajectory.build_report(
        env_trajectory.ReportConfig(
            inputs=current_path,
            output=tmp_path / "out" / "current.json",
            markdown_output=tmp_path / "out" / "summary.md",
            baseline=None if scenario == "no-baseline" else baseline_path,
        )
    )

    assert status == expected_status
    assert github_output.read_text() == expected_output


def test_build_report_without_required_baseline_writes_current_and_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    key = ("mock/default", "fixed", "oracle")
    current_path = tmp_path / "current.json"
    output = tmp_path / "out" / "current.json"
    markdown_output = tmp_path / "out" / "summary.md"
    github_output = tmp_path / "github-output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(github_output))
    _write_json(_report((_ok_result(key),), revision="pr-commit"), current_path)

    status = env_trajectory.build_report(
        env_trajectory.ReportConfig(
            inputs=current_path,
            output=output,
            markdown_output=markdown_output,
            require_baseline=True,
        )
    )

    assert status == 1
    assert env_trajectory.load_report(output).revision == "pr-commit"
    assert "no valid main baseline was found" in markdown_output.read_text()
    assert not github_output.exists()


@pytest.mark.parametrize("baseline_kind", ["incomplete", "error", "incompatible"])
def test_build_report_rejects_unusable_baselines_but_keeps_current_result(
    baseline_kind: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    key = ("mock/default", "fixed", "oracle")
    current_path = tmp_path / "current.json"
    baseline_path = tmp_path / "baseline.json"
    output = tmp_path / "out" / "current.json"
    markdown_output = tmp_path / "out" / "summary.md"
    github_output = tmp_path / "github-output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(github_output))
    _write_json(_report((_ok_result(key),), revision="pr-commit"), current_path)

    if baseline_kind == "incomplete":
        baseline = _report(
            (_ok_result(key),),
            complete=False,
            expected_cases=2,
        )
    elif baseline_kind == "error":
        baseline = _report((_error_result(key),), revision="main-commit")
    else:
        baseline = _report((_ok_result(key),), revision="main-commit")
    _write_json(baseline, baseline_path)
    if baseline_kind == "incompatible":
        payload = json.loads(baseline_path.read_text())
        payload["schema_version"] = 99
        baseline_path.write_text(json.dumps(payload))

    status = env_trajectory.build_report(
        env_trajectory.ReportConfig(
            inputs=current_path,
            output=output,
            markdown_output=markdown_output,
            baseline=baseline_path,
            require_baseline=True,
        )
    )

    assert status == 1
    assert env_trajectory.load_report(output).revision == "pr-commit"
    assert "Comparison unavailable" in markdown_output.read_text()
    assert not github_output.exists()


def test_build_report_detects_missing_shard(tmp_path: Path) -> None:
    inputs = tmp_path / "shards"
    inputs.mkdir()
    for group in env_trajectory.GROUPS[:-1]:
        group_results = tuple(
            _ok_result(case)
            for case in env_trajectory.FULL_CASES
            if env_trajectory._group_for_environment(case[0]) == group
        )
        _write_json(_report(group_results, group=group), inputs / f"{group}.json")

    markdown_output = tmp_path / "summary.md"
    status = env_trajectory.build_report(
        env_trajectory.ReportConfig(
            inputs=inputs,
            output=tmp_path / "merged.json",
            markdown_output=markdown_output,
            expected_groups=env_trajectory.GROUPS,
        )
    )

    assert status == 1
    assert not (tmp_path / "merged.json").exists()
    assert "missing=['mock']" in markdown_output.read_text()
