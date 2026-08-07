"""Minimal backend calibration against immutable physical reset references.

The search never modifies a reset state.  It first evaluates each backend's
nominal YAML, then increases numerical budgets, and only then performs a
deterministic forward selection of at most two declared physical groups.  Hot
hold acceptance is written to JSON/CSV rather than enforced in pytest.

The default is a cheap plan/metadata validation.  Pass ``--execute`` to run the
TORAX holds (the first compilation of each backend can take several minutes).
"""

from __future__ import annotations

import copy
import csv
import dataclasses
import functools
import hashlib
import itertools
import json
import math
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np
import tyro
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator
from torax._src.torax_pydantic import model_config

from plasmax.environment.config import parse_env_and_backend
from plasmax.environment.factory import _build_env
from plasmax.environment.merge import (
    _apply_imas_init,
    _load_extended_yaml,
    _resolve_geometry_dir,
)
from plasmax.environment.references import load_reference_manifest, reset_reference_id
from plasmax.environment.registry import resolve_backend, resolve_env
from plasmax.environment.schema import ScenarioConfig, SteppingConfig
from plasmax.wrappers import unwrap_to_env_state

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "outputs" / "calibration"
SEARCH_SPACES_PATH = Path(__file__).resolve().parent / "search_spaces.yaml"
CONVENTIONAL_HOT_ENVS = (
    "iter/baseline/flattop",
    "iter/hybrid/flattop",
    "iter/advanced/flattop",
    "sparc/prd/flattop",
    "sparc/reduced_field/flattop",
)
CONVENTIONAL_COLD_ENVS = tuple(
    env.removesuffix("flattop") + "rampup" for env in CONVENTIONAL_HOT_ENVS
)
CONVENTIONAL_BACKENDS = (
    "bohm_gyrobohm",
    "cgm",
    "qlknn",
    "tglfnn",
    "tglfnn_nr",
)
STEP_BACKENDS = ("bohm_gyrobohm", "tglfnn_spherical")


class CalibrationParameter(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    runtime_path: str
    randomization_path: str


class PhysicalGroup(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    relative_grid: tuple[float, float, float, float, float]
    parameters: tuple[CalibrationParameter, ...]

    @model_validator(mode="after")
    def _validate_grid(self) -> PhysicalGroup:
        if tuple(sorted(self.relative_grid)) != self.relative_grid:
            raise ValueError("relative_grid must be increasing")
        if 1.0 not in self.relative_grid:
            raise ValueError("relative_grid must contain the nominal factor 1")
        if not self.parameters:
            raise ValueError("a physical group must contain at least one parameter")
        return self


class CalibrationMetadata(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    scope: Literal["conventional_global", "step_spherical"]
    max_physical_groups: int = Field(ge=0, le=2)
    numerical: dict[str, tuple[float | int | bool, ...]] = Field(default_factory=dict)
    physical_groups: dict[str, PhysicalGroup] = Field(default_factory=dict)
    locked_nominal: tuple[str, ...] = ()
    physical_from: str | None = None
    solver_only: bool = False
    preserved_envs: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _validate_solver_only(self) -> CalibrationMetadata:
        if self.solver_only and (self.max_physical_groups or self.physical_groups):
            raise ValueError("solver-only calibration cannot declare physical groups")
        return self


@dataclasses.dataclass(frozen=True)
class Candidate:
    numerical: tuple[tuple[str, float | int | bool], ...] = ()
    physical: tuple[tuple[str, float], ...] = ()

    @property
    def label(self) -> str:
        if not self.numerical and not self.physical:
            return "nominal"
        parts = [f"{path}={value}" for path, value in self.numerical]
        parts.extend(f"{name}={factor:g}x" for name, factor in self.physical)
        return ";".join(parts)


@dataclasses.dataclass(frozen=True)
class Snapshot:
    profiles: dict[str, np.ndarray]
    scalars: dict[str, float]


@dataclasses.dataclass(frozen=True)
class CheckpointResult:
    seconds: float
    profile_drift: dict[str, float]
    scalar_drift: dict[str, float]
    max_profile_drift: float
    max_scalar_drift: float
    passed: bool


@dataclasses.dataclass(frozen=True)
class HoldResult:
    env: str
    backend: str
    candidate: str
    valid_rollout: bool
    termination_code: int
    elapsed_seconds: float
    checkpoints: tuple[CheckpointResult, ...]
    error: str | None = None

    @property
    def passed(self) -> bool:
        return self.valid_rollout and all(point.passed for point in self.checkpoints)

    @property
    def final_drift(self) -> float:
        if not self.checkpoints:
            return math.inf
        point = self.checkpoints[-1]
        return max(point.max_profile_drift, point.max_scalar_drift)


_PROFILE_NAMES = ("T_i", "T_e", "n_e", "psi", "q")
_SCALAR_NAMES = (
    "Ip",
    "q_min",
    "q95",
    "beta_N",
    "W_thermal_total",
    "P_fusion",
    "f_non_inductive",
)


@functools.cache
def _load_search_spaces() -> dict[str, Any]:
    with SEARCH_SPACES_PATH.open() as stream:
        return yaml.safe_load(stream) or {}


def _load_metadata(backend: str) -> CalibrationMetadata:
    metadata = _load_search_spaces().get(backend)
    if not isinstance(metadata, dict):
        raise ValueError(f"backend {backend!r} has no calibration metadata")
    parsed = CalibrationMetadata.model_validate(metadata)
    raw = _load_extended_yaml(resolve_backend(backend))
    _validate_physical_bounds(backend, parsed, raw.get("physics_randomization") or {})
    return parsed


def _validate_physical_bounds(
    backend: str,
    metadata: CalibrationMetadata,
    randomization: Mapping[str, Any],
) -> None:
    """Ensure every candidate grid stays inside an existing uncertainty range."""

    for group_name, group in metadata.physical_groups.items():
        for parameter in group.parameters:
            spec = randomization.get(parameter.randomization_path)
            if not isinstance(spec, dict) or "relative" not in spec:
                raise ValueError(
                    f"{backend}.{group_name}: {parameter.randomization_path!r} "
                    "has no relative physics_randomization bound"
                )
            low, high = map(float, spec["relative"])
            if group.relative_grid[0] < low or group.relative_grid[-1] > high:
                raise ValueError(
                    f"{backend}.{group_name} grid {group.relative_grid} exceeds "
                    f"the declared [{low}, {high}] bound"
                )


def _get_nested(mapping: Mapping[str, Any], path: str, default: Any = 1.0) -> Any:
    current: Any = mapping
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return default
        current = current[part]
    return current


def _set_nested(mapping: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    current = mapping
    for part in parts[:-1]:
        child = current.get(part)
        if not isinstance(child, dict):
            child = {}
            current[part] = child
        current = child
    current[parts[-1]] = value


def _candidate_cost(candidate: Candidate, metadata: CalibrationMetadata) -> float:
    cost = 0.0
    for path, value in candidate.numerical:
        choices = metadata.numerical[path]
        cost += choices.index(value) / max(len(choices) - 1, 1)
    return cost


def _candidate_distance(candidate: Candidate) -> float:
    return float(sum(abs(math.log(factor)) for _, factor in candidate.physical))


def _numerical_candidates(metadata: CalibrationMetadata) -> list[Candidate]:
    if not metadata.numerical:
        return [Candidate()]
    paths = tuple(metadata.numerical)
    candidates = [
        Candidate(numerical=tuple(zip(paths, values, strict=True)))
        for values in itertools.product(*(metadata.numerical[path] for path in paths))
    ]
    candidates.sort(
        key=lambda candidate: (
            _candidate_cost(candidate, metadata),
            candidate.label,
        )
    )
    return [
        Candidate(),
        *[candidate for candidate in candidates if candidate != Candidate()],
    ]


def _changes_numerical_config(candidate: Candidate, cfg: ScenarioConfig) -> bool:
    """Whether a labelled numerical candidate changes this nominal config."""

    stepping = cfg.stepping.model_dump()
    for path, value in candidate.numerical:
        nominal = (
            stepping.get(path.removeprefix("stepping."))
            if path.startswith("stepping.")
            else _get_nested(cfg.torax, path, default=None)
        )
        if nominal != value:
            return True
    return False


def _apply_candidate(
    cfg: ScenarioConfig,
    metadata: CalibrationMetadata,
    candidate: Candidate,
    physical_metadata: CalibrationMetadata | None = None,
) -> tuple[ScenarioConfig, dict[str, Any]]:
    torax = copy.deepcopy(dict(cfg.torax))
    stepping = cfg.stepping.model_dump()
    for path, value in candidate.numerical:
        if path.startswith("stepping."):
            stepping[path.removeprefix("stepping.")] = value
        else:
            _set_nested(torax, path, value)
    for group_name, factor in candidate.physical:
        group_source = metadata if physical_metadata is None else physical_metadata
        group = group_source.physical_groups[group_name]
        for parameter in group.parameters:
            nominal = float(_get_nested(torax, parameter.runtime_path))
            _set_nested(torax, parameter.runtime_path, nominal * factor)
    updated = cfg.model_copy(
        update={
            "state_noise": {},
            "physics_randomization": {},
            "stepping": SteppingConfig.model_validate(stepping),
        }
    )
    return updated, torax


def _hold_definition(env: str) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return checkpoint seconds and fractional limits for one reference."""

    if env == "step":
        return (10.0, 100.0), (0.10, 0.20)
    reference = load_reference_manifest().references[reset_reference_id(env)]
    if reference.kind == "digitized":
        return (1.0, 10.0), (0.15, 0.30)
    return (1.0, 10.0), (0.10, 0.20)


def _snapshot(state: Any) -> Snapshot:
    plasma = unwrap_to_env_state(state).plasma
    profiles = {
        name: np.asarray(getattr(plasma, name), dtype=np.float64)
        for name in _PROFILE_NAMES
    }
    scalars = {name: float(getattr(plasma, name)) for name in _SCALAR_NAMES}
    return Snapshot(profiles=profiles, scalars=scalars)


def _snapshot_arrays(
    state: Any,
) -> tuple[tuple[jax.Array, ...], tuple[jax.Array, ...]]:
    """Return a JAX-pytree snapshot suitable for a compiled scan."""

    plasma = unwrap_to_env_state(state).plasma
    return (
        tuple(getattr(plasma, name) for name in _PROFILE_NAMES),
        tuple(getattr(plasma, name) for name in _SCALAR_NAMES),
    )


def _snapshot_from_arrays(
    values: tuple[tuple[jax.Array, ...], tuple[jax.Array, ...]],
    index: int | None = None,
) -> Snapshot:
    """Convert one compiled snapshot (or one index of a sequence) to NumPy."""

    profiles, scalars = values

    def select(value: jax.Array) -> jax.Array:
        return value if index is None else value[index]

    return Snapshot(
        profiles={
            name: np.asarray(select(value), dtype=np.float64)
            for name, value in zip(_PROFILE_NAMES, profiles, strict=True)
        },
        scalars={
            name: float(select(value))
            for name, value in zip(_SCALAR_NAMES, scalars, strict=True)
        },
    )


def _make_compiled_rollout(base_env: Any, num_steps: int):
    """Compile one constant-action rollout and freeze after invalid states."""

    from_physical = getattr(base_env, "from_physical", None)

    @jax.jit
    def rollout(key: jax.Array):
        state, info = base_env.init(key)
        physical_action = unwrap_to_env_state(state).prev_action
        action = (
            physical_action if from_physical is None else from_physical(physical_action)
        )
        initial = _snapshot_arrays(state)
        initial_finite = jnp.all(jnp.isfinite(info.obs)) & jnp.isfinite(info.reward)
        active = (~info.terminated) & initial_finite
        termination_code = jnp.where(
            initial_finite, info.termination_code, jnp.int32(3)
        )

        def advance(carry: tuple[Any, Any, jax.Array, jax.Array], _: None):
            state, info, active, termination_code = carry

            def real_step(_: None):
                next_state, next_info = base_env.step(state, action)
                finite = jnp.all(jnp.isfinite(next_info.obs)) & jnp.isfinite(
                    next_info.reward
                )
                next_active = (~next_info.terminated) & finite
                next_code = jnp.where(
                    finite,
                    next_info.termination_code,
                    jnp.int32(3),
                )
                return next_state, next_info, next_active, next_code

            next_carry = jax.lax.cond(active, real_step, lambda _: carry, operand=None)
            next_state, _, next_active, _ = next_carry
            return next_carry, (_snapshot_arrays(next_state), next_active)

        final, (snapshots, active_steps) = jax.lax.scan(
            advance,
            (state, info, active, termination_code),
            xs=None,
            length=num_steps,
        )
        return initial, snapshots, active_steps, final[2], final[3]

    return rollout


def _normalized_profile_drift(reference: np.ndarray, current: np.ndarray) -> float:
    rho = (np.arange(reference.size, dtype=np.float64) + 0.5) / reference.size
    weights = 2.0 * rho
    numerator = np.sum(weights * np.square(current - reference))
    denominator = np.sum(weights * np.square(reference))
    return float(np.sqrt(numerator / max(denominator, np.finfo(float).tiny)))


_SCALAR_FLOORS = {
    "Ip": 1e5,
    "q_min": 0.1,
    "q95": 0.1,
    "beta_N": 0.1,
    "W_thermal_total": 1e6,
    "P_fusion": 1e6,
    "f_non_inductive": 0.01,
}


def _checkpoint_result(
    seconds: float,
    limit: float,
    reference: Snapshot,
    current: Snapshot,
) -> CheckpointResult:
    profiles = {
        name: _normalized_profile_drift(reference.profiles[name], values)
        for name, values in current.profiles.items()
    }
    scalars = {
        name: abs(value - reference.scalars[name])
        / max(abs(reference.scalars[name]), _SCALAR_FLOORS[name])
        for name, value in current.scalars.items()
    }
    max_profile = max(profiles.values())
    max_scalar = max(scalars.values())
    return CheckpointResult(
        seconds=seconds,
        profile_drift=profiles,
        scalar_drift=scalars,
        max_profile_drift=max_profile,
        max_scalar_drift=max_scalar,
        passed=max(max_profile, max_scalar) <= limit,
    )


def _build_hold_env(
    env: str,
    backend: str,
    metadata: CalibrationMetadata,
    candidate: Candidate,
    *,
    noisy: bool = False,
    realistic: bool = False,
    include_wrappers: bool = False,
    physical_metadata: CalibrationMetadata | None = None,
):
    env_path = resolve_env(env)
    nominal_cfg = parse_env_and_backend(env, backend)
    cfg, torax = _apply_candidate(
        nominal_cfg, metadata, candidate, physical_metadata=physical_metadata
    )
    if noisy:
        cfg = cfg.model_copy(
            update={
                "state_noise": nominal_cfg.state_noise,
                "physics_randomization": nominal_cfg.physics_randomization,
            }
        )
    seconds, _ = _hold_definition(env)
    numerics = torax.setdefault("numerics", {})
    fixed_dt = float(numerics.get("fixed_dt", 0.1))
    numerics["t_final"] = max(
        float(numerics.get("t_final", 0.0)), seconds[-1] + fixed_dt
    )
    torax = _apply_imas_init(torax, env_path)
    torax = _resolve_geometry_dir(torax, env_path)
    torax_config = model_config.ToraxConfig.from_dict(torax)
    num_steps = int(math.ceil(seconds[-1] / fixed_dt))
    built = _build_env(
        cfg,
        torax_config,
        reward="P_diff",
        variant="realistic" if realistic else "oracle",
        max_steps=num_steps,
        disruption_penalty=0.0,
    )
    return (built if include_wrappers else built.unwrapped), fixed_dt


def run_hold(
    env: str,
    backend: str,
    metadata: CalibrationMetadata,
    candidate: Candidate,
    *,
    seed: int = 0,
    physical_metadata: CalibrationMetadata | None = None,
) -> HoldResult:
    """Run one noise-free constant-action hot hold."""

    started = time.monotonic()
    try:
        base_env, fixed_dt = _build_hold_env(
            env,
            backend,
            metadata,
            candidate,
            physical_metadata=physical_metadata,
        )
        seconds, limits = _hold_definition(env)
        step_targets = tuple(int(math.ceil(value / fixed_dt)) for value in seconds)
        rollout = _make_compiled_rollout(base_env, step_targets[-1])
        initial, snapshots, active_steps, valid, termination_code = rollout(
            jax.random.key(seed)
        )
        jax.block_until_ready(
            (initial, snapshots, active_steps, valid, termination_code)
        )
        reference = _snapshot_from_arrays(initial)
        checkpoints: list[CheckpointResult] = []
        active_array = np.asarray(active_steps, dtype=np.bool_)
        for seconds_value, limit, step in zip(
            seconds, limits, step_targets, strict=True
        ):
            if active_array[step - 1]:
                checkpoints.append(
                    _checkpoint_result(
                        seconds_value,
                        limit,
                        reference,
                        _snapshot_from_arrays(snapshots, step - 1),
                    )
                )
        valid_rollout = bool(valid) and len(checkpoints) == len(step_targets)
        return HoldResult(
            env=env,
            backend=backend,
            candidate=candidate.label,
            valid_rollout=valid_rollout,
            termination_code=int(termination_code),
            elapsed_seconds=time.monotonic() - started,
            checkpoints=tuple(checkpoints),
        )
    except Exception as error:  # Record unsupported pairs instead of losing a sweep.
        return HoldResult(
            env=env,
            backend=backend,
            candidate=candidate.label,
            valid_rollout=False,
            termination_code=3,
            elapsed_seconds=time.monotonic() - started,
            checkpoints=(),
            error=f"{type(error).__name__}: {error}",
        )


def run_cold_anchor(
    env: str,
    backend: str,
    metadata: CalibrationMetadata,
    candidate: Candidate,
    *,
    physical_metadata: CalibrationMetadata | None = None,
) -> dict[str, Any]:
    """Validate a constrained cold anchor and its first physical second."""

    base_env, fixed_dt = _build_hold_env(
        env,
        backend,
        metadata,
        candidate,
        physical_metadata=physical_metadata,
    )
    state, info = base_env.init(jax.random.key(0))
    initial = unwrap_to_env_state(state).plasma
    payload = parse_env_and_backend(env, backend).torax["profile_conditions"]
    reference = load_reference_manifest().references[reset_reference_id(env)]

    def first_value(value: Any) -> Any:
        if (
            isinstance(value, Mapping)
            and value
            and all(
                isinstance(key, int | float) and not isinstance(key, bool)
                for key in value
            )
        ):
            return value[min(value, key=float)]
        return value

    def profile_endpoints(name: str) -> tuple[float, float]:
        radial = first_value(payload[name])
        if not isinstance(radial, Mapping):
            raise ValueError(f"cold-anchor {name} must expose a radial profile")
        return float(radial[min(radial, key=float)]), float(
            radial[max(radial, key=float)]
        )

    ip_value = payload["Ip"]
    target_ip = float(first_value(ip_value))
    target_fgw = float(reference.targets["f_GW"])
    target_axis_temperature = float(reference.targets["axis_temperature_keV"])
    target_edge_temperature = float(reference.targets["edge_temperature_keV"])
    ti_axis, ti_edge = profile_endpoints("T_i")
    te_axis, te_edge = profile_endpoints("T_e")
    steps = max(1, int(math.ceil(1.0 / fixed_dt)))
    rollout = _make_compiled_rollout(base_env, steps)
    _, _, _, valid, termination_code = rollout(jax.random.key(0))
    jax.block_until_ready((valid, termination_code))
    initial_valid = not bool(info.terminated) and bool(
        np.all(np.isfinite(np.asarray(info.obs)))
    )
    ip_actual = float(initial.Ip)
    fgw_actual = float(initial.fgw_n_e_line_avg)
    return {
        "env": env,
        "backend": backend,
        "candidate": candidate.label,
        "target_Ip_A": target_ip,
        "actual_Ip_A": ip_actual,
        "target_f_GW": target_fgw,
        "actual_f_GW_line": fgw_actual,
        "target_axis_Te_Ti_keV": target_axis_temperature,
        "target_edge_Te_Ti_keV": target_edge_temperature,
        "configured_axis_T_i_keV": ti_axis,
        "configured_axis_T_e_keV": te_axis,
        "configured_edge_T_i_keV": ti_edge,
        "configured_edge_T_e_keV": te_edge,
        "initial_first_cell_T_i_keV": float(initial.T_i[0]),
        "initial_first_cell_T_e_keV": float(initial.T_e[0]),
        "initial_last_cell_T_i_keV": float(initial.T_i[-1]),
        "initial_last_cell_T_e_keV": float(initial.T_e[-1]),
        "Ip_within_2pct": abs(ip_actual / target_ip - 1.0) <= 0.02,
        "f_GW_within_2pct": abs(fgw_actual / target_fgw - 1.0) <= 0.02,
        "temperature_anchors_exact": all(
            math.isclose(actual, target, rel_tol=0.0, abs_tol=1.0e-12)
            for actual, target in (
                (ti_axis, target_axis_temperature),
                (te_axis, target_axis_temperature),
                (ti_edge, target_edge_temperature),
                (te_edge, target_edge_temperature),
            )
        ),
        "survived_first_second": initial_valid and bool(valid),
        "termination_code": int(termination_code),
    }


def _evaluate_candidate(
    backend: str,
    envs: Sequence[str],
    metadata: CalibrationMetadata,
    candidate: Candidate,
    physical_metadata: CalibrationMetadata | None = None,
) -> tuple[HoldResult, ...]:
    return tuple(
        run_hold(
            env,
            backend,
            metadata,
            candidate,
            physical_metadata=physical_metadata,
        )
        for env in envs
    )


def _candidate_score(
    candidate: Candidate,
    results: Sequence[HoldResult],
    metadata: CalibrationMetadata,
) -> tuple[bool, bool, int, float, float, float]:
    valid = all(result.valid_rollout for result in results)
    passed = valid and all(result.passed for result in results)
    drift = max((result.final_drift for result in results), default=math.inf)
    return (
        not valid,
        not passed,
        len(candidate.physical),
        _candidate_distance(candidate),
        _candidate_cost(candidate, metadata),
        drift,
    )


def calibrate_backend(
    backend: str,
    envs: Sequence[str],
    *,
    candidate_limit: int = 0,
    inherited_physical: tuple[tuple[str, float], ...] = (),
    physical_metadata: CalibrationMetadata | None = None,
) -> tuple[
    Candidate | None,
    list[tuple[Candidate, tuple[HoldResult, ...]]],
    bool,
]:
    """Run numerical escalation and two-group deterministic forward selection."""

    metadata = _load_metadata(backend)
    evaluated: list[tuple[Candidate, tuple[HoldResult, ...]]] = []

    def evaluate(candidate: Candidate) -> tuple[HoldResult, ...]:
        if candidate_limit and len(evaluated) >= candidate_limit:
            raise RuntimeError(f"candidate limit {candidate_limit} reached")
        print(f"[{backend}] {candidate.label}")
        results = _evaluate_candidate(
            backend,
            envs,
            metadata,
            candidate,
            physical_metadata=physical_metadata,
        )
        evaluated.append((candidate, results))
        return results

    nominal_configs = tuple(parse_env_and_backend(env, backend) for env in envs)
    numerical_candidates = [
        dataclasses.replace(candidate, physical=inherited_physical)
        for candidate in _numerical_candidates(metadata)
        if candidate == Candidate()
        or any(_changes_numerical_config(candidate, cfg) for cfg in nominal_configs)
    ]
    best_numerical: Candidate | None = None
    best_numerical_results: tuple[HoldResult, ...] | None = None
    try:
        for candidate in numerical_candidates:
            results = evaluate(candidate)
            if all(result.passed for result in results):
                return candidate, evaluated, True
            if all(result.valid_rollout for result in results) and (
                best_numerical is None
                or _candidate_score(candidate, results, metadata)
                < _candidate_score(
                    best_numerical,
                    best_numerical_results or (),
                    metadata,
                )
            ):
                best_numerical = candidate
                best_numerical_results = results
    except RuntimeError:
        return None, evaluated, False

    if metadata.solver_only or not metadata.physical_groups or best_numerical is None:
        return None, evaluated, True

    # Forward step one: evaluate every non-nominal value of every group using
    # the least-cost valid numerical budget selected above.
    single_candidates: list[tuple[Candidate, tuple[HoldResult, ...]]] = []
    try:
        for group_name, group in metadata.physical_groups.items():
            for factor in group.relative_grid:
                if factor == 1.0:
                    continue
                candidate = dataclasses.replace(
                    best_numerical, physical=((group_name, factor),)
                )
                results = evaluate(candidate)
                single_candidates.append((candidate, results))
    except RuntimeError:
        return None, evaluated, False
    passing_singles = [
        item for item in single_candidates if all(result.passed for result in item[1])
    ]
    if passing_singles:
        selected, _ = min(
            passing_singles,
            key=lambda item: _candidate_score(item[0], item[1], metadata),
        )
        return selected, evaluated, True
    if metadata.max_physical_groups < 2 or not single_candidates:
        return None, evaluated, True

    valid_singles = [
        item
        for item in single_candidates
        if all(result.valid_rollout for result in item[1])
    ]
    if not valid_singles:
        return None, evaluated, True
    first, _ = min(
        valid_singles,
        key=lambda item: _candidate_score(item[0], item[1], metadata),
    )
    selected_group = first.physical[0][0]
    double_candidates: list[tuple[Candidate, tuple[HoldResult, ...]]] = []
    try:
        for group_name, group in metadata.physical_groups.items():
            if group_name == selected_group:
                continue
            for factor in group.relative_grid:
                if factor == 1.0:
                    continue
                candidate = dataclasses.replace(
                    first, physical=(*first.physical, (group_name, factor))
                )
                results = evaluate(candidate)
                double_candidates.append((candidate, results))
    except RuntimeError:
        return None, evaluated, False
    passing_doubles = [
        item for item in double_candidates if all(result.passed for result in item[1])
    ]
    if passing_doubles:
        selected, _ = min(
            passing_doubles,
            key=lambda item: _candidate_score(item[0], item[1], metadata),
        )
        return selected, evaluated, True
    return None, evaluated, True


def _distribution(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "p10": float(np.quantile(array, 0.10)),
        "median": float(np.quantile(array, 0.50)),
        "p90": float(np.quantile(array, 0.90)),
    }


def _profile_sha256(snapshot: Snapshot) -> str:
    """Hash physical reset profiles in a stable name/shape/value order."""

    digest = hashlib.sha256()
    for name in _PROFILE_NAMES:
        values = np.ascontiguousarray(snapshot.profiles[name], dtype="<f8")
        digest.update(name.encode())
        digest.update(np.asarray(values.shape, dtype="<i8").tobytes())
        digest.update(values.tobytes())
    return digest.hexdigest()


def paired_realistic_oracle(
    env: str,
    backend: str,
    candidate: Candidate,
    seeds: int = 32,
    physical_metadata: CalibrationMetadata | None = None,
) -> dict[str, Any]:
    """Compare paired seeded holds without tuning against noisy outcomes."""

    metadata = _load_metadata(backend)
    nominal_env, fixed_dt = _build_hold_env(
        env,
        backend,
        metadata,
        candidate,
        noisy=False,
        realistic=False,
        physical_metadata=physical_metadata,
    )
    nominal_state, _ = nominal_env.init(jax.random.key(0))
    nominal_reference = _snapshot(nominal_state)
    oracle_env, oracle_dt = _build_hold_env(
        env,
        backend,
        metadata,
        candidate,
        noisy=True,
        realistic=False,
        include_wrappers=True,
        physical_metadata=physical_metadata,
    )
    realistic_env, realistic_dt = _build_hold_env(
        env,
        backend,
        metadata,
        candidate,
        noisy=True,
        realistic=True,
        include_wrappers=True,
        physical_metadata=physical_metadata,
    )
    if not math.isclose(fixed_dt, oracle_dt) or not math.isclose(
        fixed_dt, realistic_dt
    ):
        raise ValueError("paired variants must use the same physical control interval")
    seconds, limits = _hold_definition(env)
    steps = int(math.ceil(seconds[-1] / fixed_dt))
    oracle_rollout = _make_compiled_rollout(oracle_env, steps)
    realistic_rollout = _make_compiled_rollout(realistic_env, steps)

    def rollout(compiled: Any, seed: int) -> tuple[bool, Snapshot, Snapshot]:
        initial, snapshots, _, valid, termination_code = compiled(jax.random.key(seed))
        jax.block_until_ready((initial, snapshots, valid, termination_code))
        return (
            bool(valid),
            _snapshot_from_arrays(initial),
            _snapshot_from_arrays(snapshots, steps - 1),
        )

    oracle_drift: list[float] = []
    realistic_drift: list[float] = []
    paired_delta: list[float] = []
    reset_profile_hashes: list[str] = []
    terminations = {"oracle": 0, "realistic": 0}
    for seed in range(seeds):
        # Calibration candidates are fixed before this function.  We retain the
        # same reset key for each pair; realistic additionally samples the
        # configured transition-wise physics randomization.
        oracle_valid, oracle_initial, oracle_final = rollout(oracle_rollout, seed)
        realistic_valid, realistic_initial, realistic_final = rollout(
            realistic_rollout, seed
        )
        for name in oracle_initial.profiles:
            np.testing.assert_array_equal(
                oracle_initial.profiles[name], realistic_initial.profiles[name]
            )
        for name in oracle_initial.scalars:
            np.testing.assert_equal(
                oracle_initial.scalars[name], realistic_initial.scalars[name]
            )
        reset_profile_hashes.append(_profile_sha256(oracle_initial))
        if oracle_valid:
            oracle_result = _checkpoint_result(
                seconds[-1], limits[-1], nominal_reference, oracle_final
            )
            oracle_value = max(
                oracle_result.max_profile_drift, oracle_result.max_scalar_drift
            )
            oracle_drift.append(oracle_value)
        else:
            terminations["oracle"] += 1
        if realistic_valid:
            result = _checkpoint_result(
                seconds[-1], limits[-1], nominal_reference, realistic_final
            )
            drift = max(result.max_profile_drift, result.max_scalar_drift)
            realistic_drift.append(drift)
            if oracle_valid:
                paired_delta.append(drift - oracle_value)
        else:
            terminations["realistic"] += 1
    return {
        "seeds": seeds,
        "reset_profile_sha256": reset_profile_hashes,
        "oracle": _distribution(oracle_drift) if oracle_drift else None,
        "realistic": _distribution(realistic_drift) if realistic_drift else None,
        "paired_realistic_minus_oracle": (
            _distribution(paired_delta) if paired_delta else None
        ),
        "terminations": terminations,
    }


def _result_dict(result: HoldResult) -> dict[str, Any]:
    final_drift = result.final_drift
    return {
        **dataclasses.asdict(result),
        "passed": result.passed,
        "final_drift": final_drift if math.isfinite(final_drift) else None,
    }


def _write_csv(
    path: Path,
    records: Iterable[tuple[str, Candidate, HoldResult]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "backend",
        "candidate",
        "env",
        "valid_rollout",
        "passed",
        "termination_code",
        "checkpoint_s",
        "max_profile_drift",
        "max_scalar_drift",
        "elapsed_seconds",
        "error",
    )
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for backend, candidate, result in records:
            checkpoints = result.checkpoints or (None,)
            for checkpoint in checkpoints:
                writer.writerow(
                    {
                        "backend": backend,
                        "candidate": candidate.label,
                        "env": result.env,
                        "valid_rollout": result.valid_rollout,
                        "passed": result.passed,
                        "termination_code": result.termination_code,
                        "checkpoint_s": (
                            "" if checkpoint is None else checkpoint.seconds
                        ),
                        "max_profile_drift": (
                            "" if checkpoint is None else checkpoint.max_profile_drift
                        ),
                        "max_scalar_drift": (
                            "" if checkpoint is None else checkpoint.max_scalar_drift
                        ),
                        "elapsed_seconds": result.elapsed_seconds,
                        "error": result.error or "",
                    }
                )


def _plan(backends: Sequence[str]) -> dict[str, Any]:
    plan: dict[str, Any] = {}
    for backend in backends:
        metadata = _load_metadata(backend)
        envs = (
            ("step",) if metadata.scope == "step_spherical" else CONVENTIONAL_HOT_ENVS
        )
        nominal_configs = tuple(parse_env_and_backend(env, backend) for env in envs)
        numerical_candidate_count = sum(
            candidate == Candidate()
            or any(_changes_numerical_config(candidate, cfg) for cfg in nominal_configs)
            for candidate in _numerical_candidates(metadata)
        )
        plan[backend] = {
            "metadata": metadata.model_dump(mode="json"),
            "environments": list(envs),
            "cold_anchor_validation": (
                list(CONVENTIONAL_COLD_ENVS)
                if metadata.scope == "conventional_global"
                else []
            ),
            "numerical_candidate_count": numerical_candidate_count,
            "maximum_physical_groups": metadata.max_physical_groups,
        }
    if "bohm_gyrobohm" in backends:
        plan["bohm_gyrobohm"]["validation_only_environments"] = ["step"]
    return plan


def main(
    backends: tuple[str, ...] = (*CONVENTIONAL_BACKENDS, "tglfnn_spherical"),
    execute: bool = False,
    candidate_limit: int = 0,
    paired_seeds: int = 32,
    output_json: Path = DEFAULT_OUTPUT_DIR / "backend_calibration.json",
    output_csv: Path = DEFAULT_OUTPUT_DIR / "backend_calibration.csv",
) -> None:
    """Validate the search plan or execute calibration and paired evaluation."""

    unknown = set(backends) - {*CONVENTIONAL_BACKENDS, *STEP_BACKENDS}
    if unknown:
        raise ValueError(f"unknown calibration backends: {sorted(unknown)}")
    if paired_seeds < 0:
        raise ValueError("paired_seeds must be non-negative")
    plan = _plan(backends)
    report: dict[str, Any] = {"executed": execute, "plan": plan, "backends": {}}
    csv_records: list[tuple[str, Candidate, HoldResult]] = []
    paired_reset_hashes: dict[str, tuple[str, ...]] = {}
    if execute:
        selected: dict[str, Candidate] = {}
        for backend in backends:
            metadata = _load_metadata(backend)
            envs = (
                ("step",)
                if metadata.scope == "step_spherical"
                else CONVENTIONAL_HOT_ENVS
            )
            inherited_physical: tuple[tuple[str, float], ...] = ()
            physical_metadata: CalibrationMetadata | None = None
            if metadata.physical_from is not None:
                source_candidate = selected.get(metadata.physical_from)
                if source_candidate is None:
                    raise ValueError(
                        f"{backend} requires an accepted {metadata.physical_from} "
                        "physical calibration earlier in --backends"
                    )
                inherited_physical = source_candidate.physical
                physical_metadata = _load_metadata(metadata.physical_from)
            candidate, evaluations, search_complete = calibrate_backend(
                backend,
                envs,
                candidate_limit=candidate_limit,
                inherited_physical=inherited_physical,
                physical_metadata=physical_metadata,
            )
            if candidate is not None:
                selected[backend] = candidate
            unresolved = []
            unsupported = []
            if candidate is None and search_complete:
                for env in envs:
                    if not any(
                        result.env == env and result.passed
                        for _, results in evaluations
                        for result in results
                    ):
                        unsupported.append(env)
            elif candidate is None:
                unresolved = list(envs)
            backend_report: dict[str, Any] = {
                "search_complete": search_complete,
                "selected": (
                    None if candidate is None else dataclasses.asdict(candidate)
                ),
                "unsupported": unsupported,
                "unresolved": unresolved,
                "evaluations": [
                    {
                        "candidate": dataclasses.asdict(current),
                        "results": [_result_dict(result) for result in results],
                    }
                    for current, results in evaluations
                ],
            }
            for current, results in evaluations:
                csv_records.extend((backend, current, result) for result in results)
            if candidate is not None and paired_seeds:
                paired = {
                    env: paired_realistic_oracle(
                        env,
                        backend,
                        candidate,
                        seeds=paired_seeds,
                        physical_metadata=physical_metadata,
                    )
                    for env in envs
                }
                for env, comparison in paired.items():
                    hashes = tuple(comparison["reset_profile_sha256"])
                    expected = paired_reset_hashes.setdefault(env, hashes)
                    if hashes != expected:
                        raise ValueError(
                            f"backend {backend} received different paired reset "
                            f"perturbations for {env}"
                        )
                backend_report["paired_realistic_oracle"] = paired
            if candidate is not None and metadata.scope == "conventional_global":
                backend_report["cold_anchor_validation"] = [
                    run_cold_anchor(
                        env,
                        backend,
                        metadata,
                        candidate,
                        physical_metadata=physical_metadata,
                    )
                    for env in CONVENTIONAL_COLD_ENVS
                ]
            report["backends"][backend] = backend_report

        report["paired_reset_contract"] = {
            "seeds": paired_seeds,
            "profile_sha256_by_environment": {
                env: list(hashes) for env, hashes in paired_reset_hashes.items()
            },
        }

        # Preserve the upstream STEP BgB exception: validate it unchanged after
        # conventional BgB selection, but never include it in the global fit.
        if "bohm_gyrobohm" in backends:
            metadata = _load_metadata("bohm_gyrobohm")
            step_result = run_hold("step", "bohm_gyrobohm", metadata, Candidate())
            report["backends"]["bohm_gyrobohm"]["step_validation"] = _result_dict(
                step_result
            )
            csv_records.append(("bohm_gyrobohm", Candidate(), step_result))

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    _write_csv(output_csv, csv_records)
    print(f"Calibration {'report' if execute else 'plan'}: {output_json}")


if __name__ == "__main__":
    tyro.cli(main)
