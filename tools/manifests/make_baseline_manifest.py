"""Generate the Isambard job-array manifest for the baseline study."""

from __future__ import annotations

import csv
import dataclasses
from pathlib import Path

import tyro

from experiments.studies.baseline_study import (
    BASELINE_ALGORITHMS,
    BASELINE_VARIANTS,
    baseline_envs,
    make_baseline_jobs,
)


@dataclasses.dataclass
class Args:
    out: str = "sweeps/baseline_v1.tsv"
    study: str = "baseline-v1"
    backend: str = "bohm_gyrobohm"
    envs: tuple[str, ...] = baseline_envs()
    algorithms: tuple[str, ...] = BASELINE_ALGORITHMS
    variants: tuple[str, ...] = BASELINE_VARIANTS
    num_seeds: int = 10
    seed: int = 0
    policy_steps: int = 10_000_000
    knot_steps: int = 1_000_000
    policy_eval_freq: int = 1_000_000
    knot_eval_freq: int = 50_000


def main(args: Args) -> None:
    jobs = make_baseline_jobs(
        backend=args.backend,
        envs=args.envs,
        algorithms=args.algorithms,
        variants=args.variants,
        num_seeds=args.num_seeds,
        seed=args.seed,
        policy_steps=args.policy_steps,
        knot_steps=args.knot_steps,
        policy_eval_freq=args.policy_eval_freq,
        knot_eval_freq=args.knot_eval_freq,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        "task_id",
        "algorithm",
        "env",
        "backend",
        "variant",
        "reward",
        "total_steps",
        "eval_freq",
        "num_seeds",
        "seed",
        "slug",
    )
    with out.open("w", newline="") as manifest:
        writer = csv.DictWriter(
            manifest,
            fieldnames=fieldnames,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        for task_id, job in enumerate(jobs):
            writer.writerow(
                {
                    "task_id": task_id,
                    **dataclasses.asdict(job),
                    "slug": job.slug,
                }
            )
    print(f"Wrote {len(jobs)} jobs to {out}")
    print(
        "Submit with: "
        f"MANIFEST={out} STUDY={args.study} "
        f"sbatch --array=0-{len(jobs) - 1} "
        "experiments/cluster/isambard/isambard_baselines.slurm"
    )


if __name__ == "__main__":
    main(tyro.cli(Args))
