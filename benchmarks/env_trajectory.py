"""Run and compare full environment trajectories.

The default command exercises the complete TORAX environment/backend/variant
matrix with a fixed seed and zero normalized actions. It is intentionally a
repository benchmark rather than an installed-package API.

Run the complete local diagnostic with::

    uv run python benchmarks/env_trajectory.py run

Run one small subset with::

    uv run python benchmarks/env_trajectory.py run \
        --environments iter/hybrid/flattop \
        --backends cgm \
        --variants realistic
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import gc
import html
import importlib.metadata
import json
import math
import os
import platform
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Annotated, Any, Literal

import jax
import jax.numpy as jnp
import tyro

import plasmax
from plasmax.environment.merge import valid_env_backend_combos
from plasmax.wrappers import OracleWrappers, RealisticWrappers

SCHEMA_VERSION = 1
SEED = 0
EXCLUDED_BACKENDS = ("tglfnn_nr",)
VARIANTS = ("oracle", "realistic")
GROUPS = (
    "iter-baseline",
    "iter-hybrid",
    "iter-advanced",
    "sparc-prd",
    "sparc-reduced-field",
    "step",
    "mock",
)
SLOWDOWN_RATIO = 1.25
SLOWDOWN_SECONDS = 2.0

type CaseKey = tuple[str, str, str]


def _full_cases() -> tuple[CaseKey, ...]:
    return tuple(
        (environment, backend, variant)
        for environment, backends in sorted(valid_env_backend_combos().items())
        for backend in sorted(backends)
        if backend not in EXCLUDED_BACKENDS
        for variant in VARIANTS
    )


FULL_CASES = _full_cases()


@dataclasses.dataclass(frozen=True)
class RunConfig:
    """Execute selected environment trajectories."""

    group: Literal[
        "all",
        "iter-baseline",
        "iter-hybrid",
        "iter-advanced",
        "sparc-prd",
        "sparc-reduced-field",
        "step",
        "mock",
    ] = "all"
    environments: tuple[str, ...] = ()
    backends: tuple[str, ...] = ()
    variants: tuple[Literal["oracle", "realistic"], ...] = VARIANTS
    output: Path = Path("outputs/env_trajectory_results.json")
    revision: str | None = None


@dataclasses.dataclass(frozen=True)
class ReportConfig:
    """Merge trajectory shards and optionally compare them with a baseline."""

    inputs: Path
    output: Path = Path("outputs/env_trajectory_results.json")
    markdown_output: Path = Path("outputs/env_trajectory_report.md")
    baseline: Path | None = None
    expected_baseline_revision: str | None = None
    expected_groups: tuple[str, ...] = ()
    require_baseline: bool = False


Command = (
    Annotated[RunConfig, tyro.conf.subcommand(name="run")]
    | Annotated[ReportConfig, tyro.conf.subcommand(name="report")]
)


@dataclasses.dataclass(frozen=True)
class RunnerMetadata:
    groups: tuple[str, ...]
    platform: str
    python_version: str
    plasmax_version: str
    torax_version: str
    jax_version: str
    devices: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class Selection:
    groups: tuple[str, ...]
    environments: tuple[str, ...]
    backends: tuple[str, ...]
    variants: tuple[str, ...]
    excluded_backends: tuple[str, ...]
    expected_cases: int
    full_matrix: bool


@dataclasses.dataclass(frozen=True)
class CaseResult:
    environment: str
    backend: str
    variant: str
    status: Literal["ok", "error"]
    creation_seconds: float | None
    first_trajectory_seconds: float | None
    steps: int | None
    boundary: Literal["terminated", "truncated"] | None
    termination_code: int | None
    error_type: str | None
    error_message: str | None

    @property
    def key(self) -> CaseKey:
        return (self.environment, self.backend, self.variant)


@dataclasses.dataclass(frozen=True)
class TrajectoryReport:
    schema_version: int
    complete: bool
    revision: str | None
    generated_at_utc: str
    runners: tuple[RunnerMetadata, ...]
    selection: Selection
    results: tuple[CaseResult, ...]


@dataclasses.dataclass(frozen=True)
class BehaviorChange:
    key: CaseKey
    description: str


@dataclasses.dataclass(frozen=True)
class TimingWarning:
    key: CaseKey
    metric: str
    baseline_seconds: float
    current_seconds: float

    @property
    def delta_seconds(self) -> float:
        return self.current_seconds - self.baseline_seconds

    @property
    def delta_percent(self) -> float:
        if self.baseline_seconds == 0.0:
            return math.inf
        return 100.0 * self.delta_seconds / self.baseline_seconds


@dataclasses.dataclass(frozen=True)
class Comparison:
    unchanged: int
    behavior_changes: tuple[BehaviorChange, ...]
    timing_warnings: tuple[TimingWarning, ...]
    errors: tuple[CaseResult, ...]


def _utc_now() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _runner_metadata(group: str) -> RunnerMetadata:
    return RunnerMetadata(
        groups=(group,),
        platform=platform.platform(),
        python_version=platform.python_version(),
        plasmax_version=_package_version("plasmax"),
        torax_version=_package_version("torax"),
        jax_version=jax.__version__,
        devices=tuple(str(device) for device in jax.devices()),
    )


def _group_for_environment(environment: str) -> str:
    if environment.startswith("iter/baseline/"):
        return "iter-baseline"
    if environment.startswith("iter/hybrid/"):
        return "iter-hybrid"
    if environment.startswith("iter/advanced/"):
        return "iter-advanced"
    if environment.startswith("sparc/prd/"):
        return "sparc-prd"
    if environment.startswith("sparc/reduced_field/"):
        return "sparc-reduced-field"
    if environment.startswith("step/"):
        return "step"
    if environment.startswith("mock/"):
        return "mock"
    raise ValueError(f"No trajectory group is defined for {environment!r}")


def select_cases(cfg: RunConfig) -> tuple[CaseKey, ...]:
    """Return the deterministic case selection for a run configuration."""
    available_environments = {case[0] for case in FULL_CASES}
    available_backends = {case[1] for case in FULL_CASES}
    unknown_environments = set(cfg.environments) - available_environments
    unknown_backends = set(cfg.backends) - available_backends
    if unknown_environments:
        raise ValueError(
            f"Unknown or unsupported environments: {sorted(unknown_environments)}"
        )
    if unknown_backends:
        raise ValueError(f"Unknown or excluded backends: {sorted(unknown_backends)}")
    if not cfg.variants:
        raise ValueError("variants must not be empty")

    selected = tuple(
        case
        for case in FULL_CASES
        if (cfg.group == "all" or _group_for_environment(case[0]) == cfg.group)
        and (not cfg.environments or case[0] in cfg.environments)
        and (not cfg.backends or case[1] in cfg.backends)
        and case[2] in cfg.variants
    )
    if not selected:
        raise ValueError("The requested filters select no valid trajectory cases")
    return selected


def _selection(cfg: RunConfig, cases: Sequence[CaseKey]) -> Selection:
    is_full = tuple(cases) == FULL_CASES
    return Selection(
        groups=(cfg.group,),
        environments=tuple(sorted({case[0] for case in cases})),
        backends=tuple(sorted({case[1] for case in cases})),
        variants=tuple(sorted({case[2] for case in cases})),
        excluded_backends=EXCLUDED_BACKENDS,
        expected_cases=len(cases),
        full_matrix=is_full,
    )


def _run_to_boundary(environment: Any) -> tuple[Any, jax.Array]:
    state, info = environment.init(jax.random.key(SEED))
    action = jnp.zeros(
        environment.action_space.shape,
        dtype=environment.action_space.dtype,
    )
    active = ~(info.terminated | info.truncated)
    steps = jnp.zeros((), dtype=jnp.int32)

    def scan_step(carry: tuple[Any, Any, jax.Array, jax.Array], _):
        def step(active_carry):
            state, _, _, steps = active_carry
            state, info = environment.step(state, action)
            active = ~(info.terminated | info.truncated)
            return state, info, active, steps + 1

        carry = jax.lax.cond(carry[2], step, lambda frozen: frozen, carry)
        return carry, None

    final, _ = jax.lax.scan(
        scan_step,
        (state, info, active, steps),
        None,
        length=environment.max_steps,
    )
    jax.block_until_ready(final)
    _, info, _, steps = final
    return info, steps


def run_case(environment: str, backend: str, variant: str) -> CaseResult:
    """Run one case, returning an error result instead of raising."""
    creation_started = time.perf_counter()
    creation_seconds: float | None = None
    trajectory_started: float | None = None
    first_trajectory_seconds: float | None = None
    try:
        env = (RealisticWrappers if variant == "realistic" else OracleWrappers)(
            plasmax.make(environment, backend)
        )
        creation_seconds = time.perf_counter() - creation_started
        trajectory_started = time.perf_counter()
        info, steps = _run_to_boundary(env)
        first_trajectory_seconds = time.perf_counter() - trajectory_started
        terminated = bool(info.terminated)
        truncated = bool(info.truncated)
        if not terminated and not truncated:
            raise RuntimeError("trajectory did not reach a boundary")
        boundary = "terminated" if terminated else "truncated"
        return CaseResult(
            environment=environment,
            backend=backend,
            variant=variant,
            status="ok",
            creation_seconds=creation_seconds,
            first_trajectory_seconds=first_trajectory_seconds,
            steps=int(steps),
            boundary=boundary,
            termination_code=int(info.termination_code),
            error_type=None,
            error_message=None,
        )
    except Exception as error:  # noqa: BLE001 - every case must be reported.
        now = time.perf_counter()
        if creation_seconds is None:
            creation_seconds = now - creation_started
        elif trajectory_started is not None and first_trajectory_seconds is None:
            first_trajectory_seconds = now - trajectory_started
        return CaseResult(
            environment=environment,
            backend=backend,
            variant=variant,
            status="error",
            creation_seconds=creation_seconds,
            first_trajectory_seconds=first_trajectory_seconds,
            steps=None,
            boundary=None,
            termination_code=None,
            error_type=type(error).__name__,
            error_message=str(error),
        )


def _report_dict(report: TrajectoryReport) -> dict[str, Any]:
    return dataclasses.asdict(report)


def write_report(report: TrajectoryReport, path: Path) -> None:
    """Atomically write one report, rejecting non-standard JSON numbers."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        _report_dict(report),
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(payload + "\n")
            temporary = Path(handle.name)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _refresh_report(
    report: TrajectoryReport,
    *,
    complete: bool | None = None,
    results: Sequence[CaseResult] | None = None,
) -> TrajectoryReport:
    return dataclasses.replace(
        report,
        complete=report.complete if complete is None else complete,
        generated_at_utc=_utc_now(),
        results=report.results if results is None else tuple(results),
    )


def _format_seconds(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def print_results(results: Sequence[CaseResult]) -> None:
    print("\n" + "=" * 28 + " environment trajectory results " + "=" * 28)
    print(
        "| environment | backend | variant | creation (s) | "
        "first trajectory (s) | steps | boundary | termination code | status |"
    )
    print("|---|---|---|---:|---:|---:|---|---:|---|")
    for result in results:
        error = "ok" if result.status == "ok" else f"error ({result.error_type})"
        print(
            f"| {result.environment} | {result.backend} | {result.variant} | "
            f"{_format_seconds(result.creation_seconds)} | "
            f"{_format_seconds(result.first_trajectory_seconds)} | "
            f"{'-' if result.steps is None else result.steps} | "
            f"{'-' if result.boundary is None else result.boundary} | "
            f"{'-' if result.termination_code is None else result.termination_code} | "
            f"{error} |"
        )


def run_trajectories(cfg: RunConfig) -> int:
    cases = select_cases(cfg)
    revision = cfg.revision or os.environ.get("GITHUB_SHA")
    report = TrajectoryReport(
        schema_version=SCHEMA_VERSION,
        complete=False,
        revision=revision,
        generated_at_utc=_utc_now(),
        runners=(_runner_metadata(cfg.group),),
        selection=_selection(cfg, cases),
        results=(),
    )
    write_report(report, cfg.output)

    print(
        f"Running {len(cases)} trajectory case(s); results: {cfg.output}",
        flush=True,
    )
    results: list[CaseResult] = []
    for index, (environment, backend, variant) in enumerate(cases, start=1):
        label = f"{environment} / {backend} / {variant}"
        print(f"[{index}/{len(cases)}] {label}", flush=True)
        result = run_case(environment, backend, variant)
        results.append(result)
        report = _refresh_report(report, results=results)
        write_report(report, cfg.output)
        if result.status == "ok":
            print(
                f"  ok: creation={result.creation_seconds:.3f}s, "
                f"first trajectory={result.first_trajectory_seconds:.3f}s, "
                f"steps={result.steps}, boundary={result.boundary}",
                flush=True,
            )
        else:
            print(
                f"  error: {result.error_type}: {result.error_message}",
                flush=True,
            )
        jax.clear_caches()
        gc.collect()

    report = _refresh_report(report, complete=True, results=results)
    write_report(report, cfg.output)
    print_results(results)
    errors = sum(result.status == "error" for result in results)
    print(
        f"\nCompleted {len(results)} case(s) with {errors} error(s). "
        f"JSON: {cfg.output}",
        flush=True,
    )
    return 1 if errors else 0


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _require_sequence(value: Any, name: str) -> Sequence[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a JSON array")
    return value


def _require_string(value: Any, name: str, *, optional: bool = False) -> str | None:
    if optional and value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    return value


def _require_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _require_int(value: Any, name: str, *, optional: bool = False) -> int | None:
    if optional and value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _require_seconds(value: Any, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number or null")
    seconds = float(value)
    if not math.isfinite(seconds) or seconds < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return seconds


def _strings(value: Any, name: str) -> tuple[str, ...]:
    values = _require_sequence(value, name)
    if not all(isinstance(item, str) for item in values):
        raise ValueError(f"{name} must contain only strings")
    return tuple(values)


def _parse_runner(raw: Any, index: int) -> RunnerMetadata:
    data = _require_mapping(raw, f"runners[{index}]")
    return RunnerMetadata(
        groups=_strings(data.get("groups"), f"runners[{index}].groups"),
        platform=str(_require_string(data.get("platform"), "runner.platform")),
        python_version=str(
            _require_string(data.get("python_version"), "runner.python_version")
        ),
        plasmax_version=str(
            _require_string(data.get("plasmax_version"), "runner.plasmax_version")
        ),
        torax_version=str(
            _require_string(data.get("torax_version"), "runner.torax_version")
        ),
        jax_version=str(_require_string(data.get("jax_version"), "runner.jax_version")),
        devices=_strings(data.get("devices"), f"runners[{index}].devices"),
    )


def _parse_selection(raw: Any) -> Selection:
    data = _require_mapping(raw, "selection")
    expected_cases = _require_int(data.get("expected_cases"), "expected_cases")
    assert expected_cases is not None
    if expected_cases < 0:
        raise ValueError("expected_cases must be non-negative")
    return Selection(
        groups=_strings(data.get("groups"), "selection.groups"),
        environments=_strings(data.get("environments"), "selection.environments"),
        backends=_strings(data.get("backends"), "selection.backends"),
        variants=_strings(data.get("variants"), "selection.variants"),
        excluded_backends=_strings(
            data.get("excluded_backends"), "selection.excluded_backends"
        ),
        expected_cases=expected_cases,
        full_matrix=_require_bool(data.get("full_matrix"), "selection.full_matrix"),
    )


def _parse_case(raw: Any, index: int) -> CaseResult:
    data = _require_mapping(raw, f"results[{index}]")
    status = _require_string(data.get("status"), f"results[{index}].status")
    if status not in {"ok", "error"}:
        raise ValueError(f"results[{index}].status must be 'ok' or 'error'")
    boundary = _require_string(
        data.get("boundary"), f"results[{index}].boundary", optional=True
    )
    if boundary not in {None, "terminated", "truncated"}:
        raise ValueError(f"results[{index}].boundary is invalid")

    result = CaseResult(
        environment=str(
            _require_string(data.get("environment"), f"results[{index}].environment")
        ),
        backend=str(_require_string(data.get("backend"), f"results[{index}].backend")),
        variant=str(_require_string(data.get("variant"), f"results[{index}].variant")),
        status=status,
        creation_seconds=_require_seconds(
            data.get("creation_seconds"), f"results[{index}].creation_seconds"
        ),
        first_trajectory_seconds=_require_seconds(
            data.get("first_trajectory_seconds"),
            f"results[{index}].first_trajectory_seconds",
        ),
        steps=_require_int(data.get("steps"), f"results[{index}].steps", optional=True),
        boundary=boundary,
        termination_code=_require_int(
            data.get("termination_code"),
            f"results[{index}].termination_code",
            optional=True,
        ),
        error_type=_require_string(
            data.get("error_type"), f"results[{index}].error_type", optional=True
        ),
        error_message=_require_string(
            data.get("error_message"),
            f"results[{index}].error_message",
            optional=True,
        ),
    )
    if result.status == "ok":
        if result.creation_seconds is None or result.first_trajectory_seconds is None:
            raise ValueError(f"successful results[{index}] must include both timings")
        if result.steps is None or result.steps < 0:
            raise ValueError(f"successful results[{index}] must include valid steps")
        if result.boundary is None or result.termination_code is None:
            raise ValueError(
                f"successful results[{index}] must include boundary information"
            )
        if result.error_type is not None or result.error_message is not None:
            raise ValueError(f"successful results[{index}] cannot contain an error")
    else:
        if not result.error_type or result.error_message is None:
            raise ValueError(f"failed results[{index}] must describe the error")
        if any(
            value is not None
            for value in (result.steps, result.boundary, result.termination_code)
        ):
            raise ValueError(
                f"failed results[{index}] cannot contain boundary information"
            )
    return result


def load_report(path: Path) -> TrajectoryReport:
    """Load and validate one trajectory report."""
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read trajectory report {path}: {error}") from error
    data = _require_mapping(raw, "report")
    schema_version = _require_int(data.get("schema_version"), "schema_version")
    if schema_version != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported trajectory schema {schema_version}; expected {SCHEMA_VERSION}"
        )
    generated_at = _require_string(data.get("generated_at_utc"), "generated_at_utc")
    assert generated_at is not None
    try:
        generated = dt.datetime.fromisoformat(generated_at)
    except ValueError as error:
        raise ValueError("generated_at_utc must be an ISO-8601 timestamp") from error
    if generated.tzinfo is None:
        raise ValueError("generated_at_utc must include a timezone")

    runners_raw = _require_sequence(data.get("runners"), "runners")
    if not runners_raw:
        raise ValueError("runners must not be empty")
    results_raw = _require_sequence(data.get("results"), "results")
    results = tuple(_parse_case(item, index) for index, item in enumerate(results_raw))
    keys = [result.key for result in results]
    if len(keys) != len(set(keys)):
        raise ValueError(f"Trajectory report {path} contains duplicate cases")

    selection = _parse_selection(data.get("selection"))
    complete = _require_bool(data.get("complete"), "complete")
    if complete and len(results) != selection.expected_cases:
        raise ValueError(
            f"Complete report expected {selection.expected_cases} cases but contains "
            f"{len(results)}"
        )
    if not complete and len(results) > selection.expected_cases:
        raise ValueError("Incomplete report contains more cases than expected")
    if complete:
        actual_dimensions = {
            "environments": {result.environment for result in results},
            "backends": {result.backend for result in results},
            "variants": {result.variant for result in results},
        }
        selected_dimensions = {
            "environments": set(selection.environments),
            "backends": set(selection.backends),
            "variants": set(selection.variants),
        }
        for name, actual in actual_dimensions.items():
            if actual != selected_dimensions[name]:
                raise ValueError(
                    f"Complete report selection.{name} does not match its cases"
                )
    return TrajectoryReport(
        schema_version=schema_version,
        complete=complete,
        revision=_require_string(data.get("revision"), "revision", optional=True),
        generated_at_utc=generated_at,
        runners=tuple(
            _parse_runner(item, index) for index, item in enumerate(runners_raw)
        ),
        selection=selection,
        results=results,
    )


def _input_paths(inputs: Path) -> tuple[Path, ...]:
    if inputs.is_file():
        return (inputs,)
    if not inputs.is_dir():
        raise ValueError(f"Report input does not exist: {inputs}")
    paths = tuple(sorted(inputs.rglob("*.json")))
    if not paths:
        raise ValueError(f"No JSON trajectory reports found below {inputs}")
    return paths


def merge_reports(
    reports: Sequence[TrajectoryReport],
    expected_groups: Sequence[str] = (),
) -> TrajectoryReport:
    """Merge complete, non-overlapping shard reports."""
    if not reports:
        raise ValueError("At least one trajectory report is required")
    incomplete = [report.selection.groups for report in reports if not report.complete]
    if incomplete:
        raise ValueError(f"Incomplete trajectory shards: {incomplete}")

    revisions = {report.revision for report in reports}
    if len(revisions) != 1:
        raise ValueError(f"Trajectory shards use different revisions: {revisions}")
    groups = tuple(group for report in reports for group in report.selection.groups)
    if len(groups) != len(set(groups)):
        raise ValueError(f"Trajectory shard groups are duplicated: {groups}")
    if expected_groups and set(groups) != set(expected_groups):
        missing = sorted(set(expected_groups) - set(groups))
        unexpected = sorted(set(groups) - set(expected_groups))
        raise ValueError(
            f"Trajectory shard groups do not match: missing={missing}, "
            f"unexpected={unexpected}"
        )

    results = tuple(result for report in reports for result in report.results)
    keys = [result.key for result in results]
    if len(keys) != len(set(keys)):
        raise ValueError("Trajectory shards contain duplicate cases")
    full_matrix = set(keys) == set(FULL_CASES)
    if expected_groups and not full_matrix:
        missing = sorted(set(FULL_CASES) - set(keys))
        unexpected = sorted(set(keys) - set(FULL_CASES))
        raise ValueError(
            f"Merged trajectory cases do not match the full matrix: "
            f"missing={missing}, unexpected={unexpected}"
        )

    return TrajectoryReport(
        schema_version=SCHEMA_VERSION,
        complete=True,
        revision=reports[0].revision,
        generated_at_utc=_utc_now(),
        runners=tuple(runner for report in reports for runner in report.runners),
        selection=Selection(
            groups=tuple(sorted(groups)),
            environments=tuple(sorted({key[0] for key in keys})),
            backends=tuple(sorted({key[1] for key in keys})),
            variants=tuple(sorted({key[2] for key in keys})),
            excluded_backends=EXCLUDED_BACKENDS,
            expected_cases=len(results),
            full_matrix=full_matrix,
        ),
        results=tuple(sorted(results, key=lambda result: result.key)),
    )


def _behavior_description(baseline: CaseResult, current: CaseResult) -> str | None:
    fields = (
        ("status", baseline.status, current.status),
        ("steps", baseline.steps, current.steps),
        ("boundary", baseline.boundary, current.boundary),
        ("termination code", baseline.termination_code, current.termination_code),
    )
    changes = [f"{name}: {old} → {new}" for name, old, new in fields if old != new]
    return "; ".join(changes) if changes else None


def compare_reports(
    baseline: TrajectoryReport,
    current: TrajectoryReport,
) -> Comparison:
    """Compare deterministic behavior exactly and timings tolerantly."""
    if not baseline.complete:
        raise ValueError("Baseline trajectory report is incomplete")
    baseline_errors = [
        result for result in baseline.results if result.status == "error"
    ]
    if baseline_errors:
        raise ValueError("Baseline trajectory report contains failed cases")

    baseline_by_key = {result.key: result for result in baseline.results}
    current_by_key = {result.key: result for result in current.results}
    behavior_changes: list[BehaviorChange] = []
    timing_warnings: list[TimingWarning] = []
    errors = tuple(
        sorted(
            (result for result in current.results if result.status == "error"),
            key=lambda result: result.key,
        )
    )
    unchanged = 0

    for key in sorted(current_by_key.keys() - baseline_by_key.keys()):
        behavior_changes.append(BehaviorChange(key, "case added"))
    for key in sorted(baseline_by_key.keys() - current_by_key.keys()):
        behavior_changes.append(BehaviorChange(key, "case removed"))

    for key in sorted(current_by_key.keys() & baseline_by_key.keys()):
        current_result = current_by_key[key]
        baseline_result = baseline_by_key[key]
        description = _behavior_description(baseline_result, current_result)
        if description is None:
            unchanged += 1
        else:
            behavior_changes.append(BehaviorChange(key, description))

        if current_result.status == "error" or baseline_result.status == "error":
            continue

        for metric in ("creation_seconds", "first_trajectory_seconds"):
            baseline_seconds = getattr(baseline_result, metric)
            current_seconds = getattr(current_result, metric)
            if baseline_seconds is None or current_seconds is None:
                continue
            delta = current_seconds - baseline_seconds
            ratio_is_slow = (
                current_seconds > 0.0
                if baseline_seconds == 0.0
                else current_seconds / baseline_seconds >= SLOWDOWN_RATIO
            )
            if ratio_is_slow and delta >= SLOWDOWN_SECONDS:
                timing_warnings.append(
                    TimingWarning(
                        key=key,
                        metric=metric,
                        baseline_seconds=baseline_seconds,
                        current_seconds=current_seconds,
                    )
                )

    return Comparison(
        unchanged=unchanged,
        behavior_changes=tuple(behavior_changes),
        timing_warnings=tuple(timing_warnings),
        errors=errors,
    )


def _escape_markdown(value: Any) -> str:
    text = str(value).replace("\n", " ").replace("\r", " ")
    text = html.escape(text, quote=False).replace("`", "&#96;")
    return text.replace("\\", "\\\\").replace("|", "\\|")


def _case_cells(key: CaseKey) -> str:
    return " | ".join(_escape_markdown(value) for value in key)


def _baseline_age(generated_at: str) -> str:
    generated = dt.datetime.fromisoformat(generated_at)
    age = dt.datetime.now(dt.UTC) - generated.astimezone(dt.UTC)
    if age.total_seconds() < 0:
        return "from the future"
    hours = int(age.total_seconds() // 3600)
    if hours < 48:
        return f"{hours} hour(s) old"
    return f"{hours // 24} day(s) old"


def _runner_differences(
    baseline: TrajectoryReport,
    current: TrajectoryReport,
) -> list[tuple[str, str, str]]:
    attributes = (
        "platform",
        "python_version",
        "plasmax_version",
        "torax_version",
        "jax_version",
        "devices",
    )
    differences = []
    for attribute in attributes:
        baseline_values = sorted(
            {str(getattr(runner, attribute)) for runner in baseline.runners}
        )
        current_values = sorted(
            {str(getattr(runner, attribute)) for runner in current.runners}
        )
        if baseline_values != current_values:
            differences.append(
                (attribute, ", ".join(baseline_values), ", ".join(current_values))
            )
    return differences


def render_markdown(
    current: TrajectoryReport,
    *,
    baseline: TrajectoryReport | None = None,
    expected_baseline_revision: str | None = None,
    comparison_error: str | None = None,
) -> str:
    """Render the compact CI-facing report."""
    lines = ["# Environment trajectory report", ""]
    current_revision = _escape_markdown(current.revision or "unknown")
    lines.append(f"Current revision: `{current_revision}`")
    if baseline is not None:
        lines.append(
            f"Baseline revision: `{_escape_markdown(baseline.revision or 'unknown')}` "
            f"({_baseline_age(baseline.generated_at_utc)})"
        )
        if (
            expected_baseline_revision is not None
            and baseline.revision != expected_baseline_revision
        ):
            lines.extend(
                [
                    "",
                    "> **Warning:** the exact PR base baseline was unavailable. "
                    f"Expected `{_escape_markdown(expected_baseline_revision)}` but "
                    f"used `{_escape_markdown(baseline.revision or 'unknown')}`.",
                ]
            )
    if comparison_error is not None:
        lines.extend(
            [
                "",
                f"> **Comparison unavailable:** {_escape_markdown(comparison_error)}",
            ]
        )

    current_errors = tuple(
        result for result in current.results if result.status == "error"
    )
    comparison = None if baseline is None else compare_reports(baseline, current)
    if comparison is None:
        lines.extend(
            [
                "",
                "| Completed cases | Errors |",
                "|---:|---:|",
                f"| {len(current.results)} | {len(current_errors)} |",
            ]
        )
    else:
        slowdown_cases = len({warning.key for warning in comparison.timing_warnings})
        lines.extend(
            [
                "",
                "| Unchanged cases | Behavior changes | Possible slowdowns | Errors |",
                "|---:|---:|---:|---:|",
                f"| {comparison.unchanged} | {len(comparison.behavior_changes)} | "
                f"{slowdown_cases} | {len(comparison.errors)} |",
            ]
        )
        warning_parts = []
        if comparison.behavior_changes:
            warning_parts.append(
                f"{len(comparison.behavior_changes)} behavior change(s)"
            )
        if slowdown_cases:
            warning_parts.append(f"{slowdown_cases} possible slowdown(s)")
        if warning_parts:
            lines.extend(
                [
                    "",
                    "> **Warning:** " + " and ".join(warning_parts) + ". Review below.",
                ]
            )

        if comparison.behavior_changes:
            lines.extend(
                [
                    "",
                    "## Behavior changes",
                    "",
                    "| Environment | Backend | Variant | Change |",
                    "|---|---|---|---|",
                ]
            )
            lines.extend(
                f"| {_case_cells(change.key)} | "
                f"{_escape_markdown(change.description)} |"
                for change in comparison.behavior_changes
            )

        if comparison.timing_warnings:
            lines.extend(
                [
                    "",
                    "## Possible slowdowns",
                    "",
                    "Timings are advisory because GitHub-hosted runners vary.",
                    "",
                    "| Environment | Backend | Variant | Timing | Main (s) | "
                    "PR (s) | Change |",
                    "|---|---|---|---|---:|---:|---:|",
                ]
            )
            for warning in comparison.timing_warnings:
                label = warning.metric.replace("_", " ")
                if warning.baseline_seconds == 0.0:
                    change = f"+{warning.delta_seconds:.3f}s (baseline was 0s)"
                else:
                    change = (
                        f"+{warning.delta_percent:.1f}% (+{warning.delta_seconds:.3f}s)"
                    )
                lines.append(
                    f"| {_case_cells(warning.key)} | {label} | "
                    f"{warning.baseline_seconds:.3f} | "
                    f"{warning.current_seconds:.3f} | "
                    f"{change} |"
                )

        runner_differences = _runner_differences(baseline, current)
        if runner_differences:
            lines.extend(
                [
                    "",
                    "<details>",
                    "<summary>Runner and version differences</summary>",
                    "",
                    "| Field | Main | Current |",
                    "|---|---|---|",
                ]
            )
            lines.extend(
                f"| {_escape_markdown(name)} | {_escape_markdown(old)} | "
                f"{_escape_markdown(new)} |"
                for name, old, new in runner_differences
            )
            lines.extend(["", "</details>"])

    if current_errors:
        lines.extend(
            [
                "",
                "## Errors",
                "",
                "| Environment | Backend | Variant | Error |",
                "|---|---|---|---|",
            ]
        )
        for result in current_errors:
            message = f"{result.error_type}: {result.error_message}"[:500]
            lines.append(f"| {_case_cells(result.key)} | {_escape_markdown(message)} |")

    lines.extend(
        [
            "",
            f"Full result: {len(current.results)} case(s); "
            f"schema {current.schema_version}.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_markdown(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def build_report(cfg: ReportConfig) -> int:
    """Merge shard files, write outputs, and return the aggregate status."""
    try:
        paths = _input_paths(cfg.inputs)
        reports = tuple(load_report(path) for path in paths)
        current = merge_reports(reports, cfg.expected_groups)
    except ValueError as error:
        cfg.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        _write_markdown(
            "# Environment trajectory report\n\n"
            f"> **Report failed:** {_escape_markdown(error)}\n",
            cfg.markdown_output,
        )
        print(f"Environment trajectory report failed: {error}", file=sys.stderr)
        return 1

    write_report(current, cfg.output)
    baseline: TrajectoryReport | None = None
    comparison_error: str | None = None
    if cfg.baseline is not None:
        try:
            baseline = load_report(cfg.baseline)
            if not baseline.complete:
                raise ValueError("baseline is incomplete")
            if any(result.status == "error" for result in baseline.results):
                raise ValueError("baseline contains failed cases")
            if cfg.expected_groups and not baseline.selection.full_matrix:
                raise ValueError("baseline does not contain a full trajectory matrix")
            if cfg.expected_groups and set(baseline.selection.groups) != set(
                cfg.expected_groups
            ):
                raise ValueError(
                    "baseline groups do not match the full trajectory matrix"
                )
            if cfg.require_baseline and not baseline.revision:
                raise ValueError("baseline does not record a commit revision")
        except ValueError as error:
            comparison_error = (
                f"{error}; manually run the Environment trajectories workflow on main"
            )
            baseline = None
    elif cfg.require_baseline:
        comparison_error = (
            "no valid main baseline was found; manually run the Environment "
            "trajectories workflow on main"
        )

    try:
        markdown = render_markdown(
            current,
            baseline=baseline,
            expected_baseline_revision=cfg.expected_baseline_revision,
            comparison_error=comparison_error,
        )
    except ValueError as error:
        comparison_error = str(error)
        markdown = render_markdown(current, comparison_error=comparison_error)
    _write_markdown(markdown, cfg.markdown_output)
    print(markdown)

    current_errors = any(result.status == "error" for result in current.results)
    return 1 if current_errors or comparison_error is not None else 0


def main(command: Command) -> int:
    if isinstance(command, RunConfig):
        return run_trajectories(command)
    return build_report(command)


if __name__ == "__main__":
    raise SystemExit(main(tyro.cli(Command)))
