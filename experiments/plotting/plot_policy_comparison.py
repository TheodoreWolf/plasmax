"""Generate matched PPO and direct-backprop frozen-policy learning curves.

The x-axis is the number of training environment transitions. Evaluation
transitions are excluded. Every plotted y-value is the mean undiscounted return
of a frozen policy over a complete episode, read from
``evaluation/return_mean`` in a single W&B comparison group.
"""

from __future__ import annotations

import csv
import dataclasses
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import tyro
import wandb

plt.switch_backend("Agg")

Method = Literal["ppo", "direct_backprop"]


@dataclasses.dataclass(frozen=True)
class Args:
    project: str = "flair/plasmax"
    group: str = "policy-compare-final-iter-hybrid-flattop-bgb"
    expected_steps: int = 7_040_000
    env: str = "iter/hybrid/flattop"
    backend: str = "bohm_gyrobohm"
    reward: str = "P_diff"
    variant: str = "oracle"
    disruption_penalty: float = -5.0
    eval_seed: int = 20_000
    eval_rollouts: int = 64
    eval_freq: int = 704_000
    seeds: tuple[int, ...] = (0, 1, 2)
    winners_json: str = "outputs/policy_comparison_extension_winners.json"
    include_running: bool = False
    out_csv: str = "outputs/policy_comparison.csv"
    out_png: str = "plots/policy_comparison.png"


@dataclasses.dataclass(frozen=True)
class EvaluationPoint:
    method: Method
    run_id: str
    run_name: str
    seed: int
    state: str
    step: int
    return_mean: float
    return_std: float
    episode_length_mean: float


def _nested(config: dict[str, Any], *keys: str) -> Any:
    value: Any = config
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def _method(config: dict[str, Any]) -> Method | None:
    if config.get("method") == "direct_backprop_policy":
        return "direct_backprop"
    if isinstance(config.get("ppo"), dict):
        return "ppo"
    return None


def _config_value(config: dict[str, Any], method: Method, name: str) -> Any:
    if method == "direct_backprop":
        return config.get(name)
    if name in {"env", "backend", "reward", "variant"}:
        ppo_name = "env_setup" if name == "env" else name
        return _nested(config, "env", ppo_name)
    raise KeyError(name)


def _training_budget(config: dict[str, Any], method: Method) -> int | None:
    if method == "ppo":
        value = _nested(config, "ppo", "total_timesteps")
        return None if value is None else int(value)
    required = ("rollout_steps", "num_rollouts", "iters")
    if any(config.get(name) is None for name in required):
        return None
    return int(np.prod([int(config[name]) for name in required]))


def _matches(args: Args, config: dict[str, Any], method: Method) -> bool:
    expected = {
        "env": args.env,
        "backend": args.backend,
        "reward": args.reward,
        "variant": args.variant,
    }
    return (
        all(
            _config_value(config, method, name) == value
            for name, value in expected.items()
        )
        and _training_budget(config, method) == args.expected_steps
    )


def _contract_errors(
    args: Args,
    config: dict[str, Any],
    method: Method,
) -> list[str]:
    if method == "direct_backprop":
        expected = {
            "disruption_penalty": args.disruption_penalty,
            "rollout_steps": 4_400,
            "truncation_steps": 100,
            "num_rollouts": 16,
            "eval_rollouts": args.eval_rollouts,
            "eval_seed": args.eval_seed,
            "eval_every_passes": args.eval_freq // (4_400 * 16),
            "iters": args.expected_steps // (4_400 * 16),
            "hidden_sizes": [64, 64],
            "grad_clip": 1.0,
            "remat": True,
        }
        return [
            f"{name}={config.get(name)!r}, expected {value!r}"
            for name, value in expected.items()
            if config.get(name) != value
        ]

    env_config = config.get("env", {})
    ppo_config = config.get("ppo", {})
    expected_env = {
        "transfer_backend": None,
        "disruption_penalty": args.disruption_penalty,
        "eval_n_envs": args.eval_rollouts,
        "eval_seed": args.eval_seed,
        "deterministic_eval": True,
        "force_minimal_callback": True,
        "ablate": "none",
        "time_aware": False,
        "quantize_bins": None,
    }
    expected_ppo = {
        "num_envs": 64,
        "num_steps": 100,
        "num_epochs": 1,
        "num_minibatches": 4,
        "eval_freq": args.eval_freq,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_eps": 0.2,
        "vf_coef": 0.5,
        "ent_coef": 0.01,
        "max_grad_norm": 0.5,
        "normalize_rewards": True,
        "hidden_sizes": [64, 64],
        "activation": "swish",
        "residual_policy": True,
        "normalize_observations": False,
    }
    errors = [
        f"env.{name}={env_config.get(name)!r}, expected {value!r}"
        for name, value in expected_env.items()
        if env_config.get(name) != value
    ]
    errors.extend(
        f"ppo.{name}={ppo_config.get(name)!r}, expected {value!r}"
        for name, value in expected_ppo.items()
        if ppo_config.get(name) != value
    )
    return errors


def _optimization_signature(config: dict[str, Any], method: Method) -> tuple[Any, ...]:
    if method == "direct_backprop":
        return (config.get("learning_rate"), config.get("grad_clip"))
    ppo_config = config["ppo"]
    return (
        ppo_config.get("learning_rate"),
        ppo_config.get("initial_log_std"),
        ppo_config.get("gamma"),
        ppo_config.get("gae_lambda"),
        ppo_config.get("ent_coef"),
        ppo_config.get("normalize_rewards"),
    )


def _same(actual: Any, expected: Any) -> bool:
    if isinstance(expected, float):
        return isinstance(actual, int | float) and math.isclose(
            float(actual), expected, rel_tol=1e-10, abs_tol=1e-12
        )
    return actual == expected


def _winner_errors(
    config: dict[str, Any],
    method: Method,
    winner_manifest: dict[str, Any],
) -> list[str]:
    winner = winner_manifest["winners"][method]
    overrides = winner["launch_overrides"]
    if method == "direct_backprop":
        expected = {"learning_rate": overrides["DIRECT_LR"]}
        return [
            f"{name}={config.get(name)!r}, selected winner requires {value!r}"
            for name, value in expected.items()
            if not _same(config.get(name), value)
        ]
    ppo_config = config.get("ppo", {})
    expected = {
        "learning_rate": overrides["PPO_LR"],
        "initial_log_std": overrides["PPO_INITIAL_LOG_STD"],
        "normalize_observations": False,
        "normalize_rewards": True,
        "max_grad_norm": 0.5,
    }
    errors = [
        f"ppo.{name}={ppo_config.get(name)!r}, selected winner requires {value!r}"
        for name, value in expected.items()
        if not _same(ppo_config.get(name), value)
    ]
    if config.get("num_seeds") != 1:
        errors.append(f"num_seeds={config.get('num_seeds')!r}, expected 1")
    return errors


def fetch_points(args: Args) -> list[EvaluationPoint]:
    winner_path = Path(args.winners_json)
    if not winner_path.exists():
        raise FileNotFoundError(
            f"selected-winner manifest does not exist: {winner_path}"
        )
    winner_manifest = json.loads(winner_path.read_text())
    api = wandb.Api()
    states = {"finished"}
    if args.include_running:
        states.add("running")

    points: list[EvaluationPoint] = []
    run_keys: set[tuple[Method, int]] = set()
    signatures: dict[Method, set[tuple[Any, ...]]] = defaultdict(set)
    for run in api.runs(args.project, filters={"group": args.group}):
        if run.state not in states:
            continue
        config = dict(run.config)
        method = _method(config)
        if method is None:
            continue
        if not _matches(args, config, method):
            raise ValueError(f"run {run.id} does not match the comparison task")
        errors = _contract_errors(args, config, method)
        errors.extend(_winner_errors(config, method, winner_manifest))
        if errors:
            raise ValueError(
                f"run {run.id} violates the final comparison contract: "
                + "; ".join(errors)
            )
        seed = int(config["seed"])
        run_key = (method, seed)
        if run_key in run_keys:
            raise ValueError(
                f"comparison group contains multiple {method} runs for seed {seed}"
            )
        run_keys.add(run_key)
        signatures[method].add(_optimization_signature(config, method))

        step_key = "simulator_steps" if method == "direct_backprop" else "_step"
        by_step: dict[int, tuple[float, float, float]] = {}
        for row in run.scan_history(
            keys=[
                step_key,
                "evaluation/return_mean",
                "evaluation/return_std",
                "evaluation/episode_length_mean",
            ],
            page_size=1_000,
        ):
            step = row.get(step_key)
            return_mean = row.get("evaluation/return_mean")
            return_std = row.get("evaluation/return_std")
            episode_length_mean = row.get("evaluation/episode_length_mean")
            if (
                step is None
                or return_mean is None
                or return_std is None
                or episode_length_mean is None
            ):
                continue
            step = int(step)
            if step <= args.expected_steps:
                metrics = (
                    float(return_mean),
                    float(return_std),
                    float(episode_length_mean),
                )
                if not all(math.isfinite(value) for value in metrics):
                    raise ValueError(
                        f"run {run.id} has non-finite evaluation metrics at "
                        f"step {step}: {metrics}"
                    )
                if not 1.0 <= metrics[2] <= 4_400.0:
                    raise ValueError(
                        f"run {run.id} has episode_length_mean={metrics[2]} at "
                        f"step {step}, expected a value in [1, 4400]"
                    )
                by_step[step] = metrics

        if args.expected_steps not in by_step and not args.include_running:
            raise ValueError(
                f"run {run.id} has no evaluation at the matched endpoint "
                f"{args.expected_steps:,}"
            )
        points.extend(
            EvaluationPoint(
                method=method,
                run_id=run.id,
                run_name=run.name,
                seed=seed,
                state=run.state,
                step=step,
                return_mean=metrics[0],
                return_std=metrics[1],
                episode_length_mean=metrics[2],
            )
            for step, metrics in sorted(by_step.items())
        )

    methods = {point.method for point in points}
    if methods != {"ppo", "direct_backprop"}:
        raise ValueError(
            "expected matched PPO and direct-backprop runs; "
            f"found {sorted(methods)} in group {args.group!r}"
        )
    expected_seeds = set(args.seeds)
    for method in ("ppo", "direct_backprop"):
        actual_seeds = {point.seed for point in points if point.method == method}
        if actual_seeds != expected_seeds:
            raise ValueError(
                f"{method} seeds are {sorted(actual_seeds)}, "
                f"expected {sorted(expected_seeds)}"
            )
        if len(signatures[method]) != 1:
            raise ValueError(f"{method} final runs do not share one tuned config")

    expected_grid = set(range(0, args.expected_steps + 1, args.eval_freq))
    by_run: dict[str, set[int]] = defaultdict(set)
    for point in points:
        by_run[point.run_id].add(point.step)
    for run_id, actual_grid in by_run.items():
        if actual_grid != expected_grid:
            raise ValueError(
                f"run {run_id} evaluation grid is {sorted(actual_grid)}, "
                f"expected {sorted(expected_grid)}"
            )
    return points


def _shared_steps(points: list[EvaluationPoint]) -> list[int]:
    by_run: dict[str, set[int]] = defaultdict(set)
    for point in points:
        by_run[point.run_id].add(point.step)
    shared = set.intersection(*by_run.values())
    if not shared:
        raise ValueError("comparison runs have no shared evaluation steps")
    return sorted(shared)


def write_csv(path: str | Path, points: list[EvaluationPoint]) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[field.name for field in dataclasses.fields(EvaluationPoint)],
        )
        writer.writeheader()
        writer.writerows(dataclasses.asdict(point) for point in points)
    return out


def make_plot(
    path: str | Path,
    points: list[EvaluationPoint],
    expected_steps: int,
) -> Path:
    shared_steps = _shared_steps(points)
    point_lookup = {
        (point.method, point.run_id, point.step): point.return_mean for point in points
    }
    runs: dict[Method, dict[str, list[EvaluationPoint]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for point in points:
        runs[point.method][point.run_id].append(point)

    colors: dict[Method, str] = {
        "ppo": "#3569b7",
        "direct_backprop": "#d06b28",
    }
    labels: dict[Method, str] = {
        "ppo": "PPO",
        "direct_backprop": "Direct backprop",
    }

    fig, axis = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
    x = np.asarray(shared_steps, dtype=np.float64)
    for method in ("ppo", "direct_backprop"):
        method_runs = runs[method]
        values = np.asarray(
            [
                [point_lookup[(method, run_id, step)] for step in shared_steps]
                for run_id in sorted(method_runs)
            ]
        )
        for run_values in values:
            axis.plot(x, run_values, color=colors[method], alpha=0.22, linewidth=1.0)
        mean = values.mean(axis=0)
        std = values.std(axis=0)
        axis.fill_between(
            x,
            mean - std,
            mean + std,
            color=colors[method],
            alpha=0.14,
            linewidth=0,
        )
        axis.plot(
            x,
            mean,
            color=colors[method],
            linewidth=2.4,
            marker="o",
            markersize=3.5,
            label=f"{labels[method]} (mean +/- std, n={values.shape[0]})",
        )

    axis.set_xlabel("Training environment transitions")
    axis.set_ylabel("Frozen-policy full-episode return")
    axis.set_title(
        "ITER hybrid flat-top, Bohm-GyroBohm"
        f" ({expected_steps / 1e6:.2f}M-step matched budget)"
    )
    axis.grid(alpha=0.22, linewidth=0.7)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False)
    axis.set_xlim(0, expected_steps)
    axis.xaxis.set_major_formatter(lambda value, _: f"{value / 1e6:g}M")

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def main(args: Args) -> None:
    points = fetch_points(args)
    csv_path = write_csv(args.out_csv, points)
    png_path = make_plot(args.out_png, points, args.expected_steps)
    methods: dict[Method, set[int]] = defaultdict(set)
    for point in points:
        methods[point.method].add(point.seed)
    print(
        f"wrote {csv_path} and {png_path}; "
        f"PPO seeds={len(methods['ppo'])}, "
        f"direct-backprop seeds={len(methods['direct_backprop'])}"
    )


if __name__ == "__main__":
    main(tyro.cli(Args))
