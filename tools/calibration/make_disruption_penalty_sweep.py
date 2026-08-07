"""Generate the Isambard PPO disruption-penalty sweep manifest."""

from __future__ import annotations

import csv
import dataclasses
from pathlib import Path

import tyro

from experiments.studies.disruption_sweep import (
    DEFAULT_KAPPAS,
    make_disruption_sweep_jobs,
)


@dataclasses.dataclass
class Args:
    out: Path = Path("sweeps/ppo_disruption_kappa_isambard.tsv")
    backend: str = "bohm_gyrobohm"
    kappas: tuple[float, ...] = DEFAULT_KAPPAS
    total_steps: int = 2_000_000
    eval_freq: int = 250_000
    num_seeds: int = 3
    seed: int = 0
    study: str = "ppo-disruption-kappa-pilot"


def main(args: Args) -> None:
    jobs = make_disruption_sweep_jobs(
        backend=args.backend,
        kappas=args.kappas,
        total_steps=args.total_steps,
        eval_freq=args.eval_freq,
        num_seeds=args.num_seeds,
        seed=args.seed,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        "task_id",
        "env",
        "backend",
        "variant",
        "reward",
        "task_terminal_penalty",
        "kappa",
        "disruption_penalty",
        "total_steps",
        "eval_freq",
        "num_seeds",
        "seed",
        "slug",
    )
    with args.out.open("w", newline="") as manifest:
        writer = csv.DictWriter(
            manifest,
            fieldnames=fieldnames,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        for task_id, job in enumerate(jobs):
            row = dataclasses.asdict(job)
            row["task_id"] = task_id
            row["disruption_penalty"] = row.pop("penalty")
            row["slug"] = job.slug
            writer.writerow(row)

    print(f"Wrote {len(jobs)} jobs to {args.out}")
    print(
        f"Submit with: MANIFEST={args.out} STUDY={args.study} "
        f"sbatch --array=0-{len(jobs) - 1}%4 "
        "experiments/cluster/isambard/isambard_disruption_sweep.slurm"
    )


if __name__ == "__main__":
    main(tyro.cli(Args))
