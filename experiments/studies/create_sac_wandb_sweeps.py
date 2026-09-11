"""Generate and optionally create per-environment W&B SAC random sweeps."""

from __future__ import annotations

import csv
import dataclasses
from pathlib import Path
from typing import Any

import tyro
import wandb
import yaml

from plasmax.environment.merge import valid_env_backend_combos


@dataclasses.dataclass(frozen=True)
class Args:
    out_dir: Path = Path("sweeps/sac_civo")
    manifest: Path = Path("sweeps/sac_civo/sweep_ids.tsv")
    entity: str = "flair"
    project: str = "plasmax"
    run_cap: int = 12
    total_timesteps: int = 2_000_000
    eval_freq: int = 250_000
    num_seeds: int = 3
    seed: int = 0
    create: bool = False
    resume: bool = False


def _realistic_envs() -> tuple[str, ...]:
    """Return every registered ITER/SPARC environment."""
    combos = valid_env_backend_combos()
    envs = tuple(
        env
        for env in sorted(combos)
        if env.startswith("iter/") or env.startswith("sparc/")
    )
    for env in envs:
        if "bohm_gyrobohm" not in combos[env]:
            raise ValueError(
                f"{env!r} cannot use the sweep backend: {sorted(combos[env])}"
            )
    return envs


def _sweep_config(env: str, args: Args) -> dict[str, Any]:
    slug = env.replace("/", "_")
    command = [
        "python",
        "${program}",
        f"--env.env-setup={env}",
        "--env.backend=bohm_gyrobohm",
        "--env.variant=realistic",
        "--env.eval-n-envs=8",
        f"--num-seeds={args.num_seeds}",
        f"--seed={args.seed}",
        f"--sac.total-timesteps={args.total_timesteps}",
        f"--sac.eval-freq={min(args.eval_freq, args.total_timesteps)}",
        "--sac.num-envs=64",
        "--study=civo-sac-hparam-all-envs",
        f"--wandb.group=sac_hparam_{slug}_realistic",
        "--wandb.mode=online",
    ]
    command.append("${args}")
    return {
        "program": "training/train_sac.py",
        "project": args.project,
        "entity": args.entity,
        "name": f"sac-hparam-{slug}-realistic",
        "description": (
            f"Random SAC hyperparameter search on {env}, realistic observations, "
            f"{args.num_seeds} vmapped seeds per setting."
        ),
        "method": "random",
        "run_cap": args.run_cap,
        "metric": {"name": "evaluation/return_mean", "goal": "maximize"},
        "command": command,
        "parameters": {
            "sac.learning_rate": {
                "distribution": "log_uniform_values",
                "min": 5.0e-5,
                "max": 1.0e-3,
            },
            "sac.gamma": {"values": [0.95, 0.97, 0.99, 0.995]},
            "sac.num_epochs": {"values": [16, 32, 64, 128]},
            "sac.batch_size": {"values": [128, 256, 512]},
            "sac.polyak": {"values": [0.99, 0.995, 0.999]},
            "sac.fill_buffer": {"values": [5_000, 10_000, 50_000]},
            "sac.max_grad_norm": {"values": [1.0, 5.0, 10.0]},
        },
    }


def main(args: Args) -> None:
    if args.run_cap <= 0:
        raise ValueError("run_cap must be positive")
    if args.total_timesteps <= 0 or args.eval_freq <= 0:
        raise ValueError("total_timesteps and eval_freq must be positive")
    if args.num_seeds != 3:
        raise ValueError("this sweep contract requires exactly three vmapped seeds")
    if args.create and args.manifest.exists() and not args.resume:
        raise FileExistsError(
            f"{args.manifest} already exists; pass --resume to create only missing "
            "environment sweeps"
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    sweep_configs = []
    for task_id, env in enumerate(_realistic_envs()):
        config = _sweep_config(env, args)
        config_path = args.out_dir / f"{env.replace('/', '_')}.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False))
        print(f"Wrote {config_path}")
        sweep_configs.append((task_id, env, config, config_path))

    if not args.create:
        print(
            f"Generated {len(sweep_configs)} random sweep definitions; "
            "pass --create after review to register them with W&B."
        )
        return

    existing_envs = set()
    if args.manifest.exists():
        with args.manifest.open(newline="") as manifest:
            existing_envs = {
                row["env"] for row in csv.DictReader(manifest, delimiter="\t")
            }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    append = bool(existing_envs)
    with args.manifest.open("a" if append else "w", newline="") as manifest:
        writer = csv.DictWriter(
            manifest,
            fieldnames=("task_id", "env", "sweep_id", "config"),
            delimiter="\t",
            lineterminator="\n",
        )
        if not append:
            writer.writeheader()
            manifest.flush()
        for task_id, env, config, config_path in sweep_configs:
            if env in existing_envs:
                print(f"Keeping existing W&B sweep for {env}")
                continue
            sweep_id = wandb.sweep(
                sweep=config,
                entity=args.entity,
                project=args.project,
            )
            writer.writerow(
                {
                    "task_id": task_id,
                    "env": env,
                    "sweep_id": f"{args.entity}/{args.project}/{sweep_id}",
                    "config": str(config_path),
                }
            )
            manifest.flush()
            print(f"Created W&B sweep for {env}: {sweep_id}")
    print(f"W&B sweep manifest is complete at {args.manifest}")


if __name__ == "__main__":
    main(tyro.cli(Args))
