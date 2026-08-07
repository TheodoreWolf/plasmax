"""Task-metadata-driven terminal-penalty sensitivity sweep contracts."""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Iterable

from experiments.studies.baseline_study import baseline_envs, reward_for_env
from plasmax.environment.config import parse_env_and_backend

DEFAULT_KAPPA = 1.0
DEFAULT_KAPPAS = (0.5, DEFAULT_KAPPA, 2.0)


def phase_envs() -> tuple[str, ...]:
    """Return registered ramp-up/down tasks with calibrated metadata."""
    return tuple(
        env
        for env in baseline_envs(include_step=False)
        if env.endswith(("/rampup", "/rampdown"))
    )


def calibrated_disruption_penalty(
    env: str,
    backend: str,
    *,
    kappa: float = DEFAULT_KAPPA,
) -> float:
    """Scale the terminal penalty stored in the task YAML by ``kappa``."""
    if kappa <= 0.0 or not math.isfinite(kappa):
        raise ValueError("kappa must be positive and finite")
    terminal_penalty = parse_env_and_backend(env, backend).task.terminal_penalty
    if terminal_penalty is None:
        raise ValueError(f"task {env!r} has no terminal penalty")
    return kappa * terminal_penalty


def _kappa_label(kappa: float) -> str:
    return f"{kappa:g}".replace(".", "p")


@dataclasses.dataclass(frozen=True)
class DisruptionSweepJob:
    """One cluster array task containing a vmapped PPO seed set."""

    env: str
    backend: str
    variant: str
    reward: str
    task_terminal_penalty: float
    kappa: float
    penalty: float
    total_steps: int
    eval_freq: int
    num_seeds: int
    seed: int

    def __post_init__(self) -> None:
        if self.variant != "realistic":
            raise ValueError("new disruption sweeps must use variant='realistic'")
        if self.env not in phase_envs():
            raise ValueError(f"unknown disruption sweep environment {self.env!r}")
        if self.reward != reward_for_env(self.env, self.backend):
            raise ValueError(f"reward does not match task metadata for {self.env!r}")
        if self.total_steps <= 0 or self.eval_freq <= 0 or self.num_seeds <= 0:
            raise ValueError("steps, eval_freq, and num_seeds must be positive")
        expected = self.kappa * self.task_terminal_penalty
        if not math.isclose(self.penalty, expected, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(
                f"penalty {self.penalty} does not match task-scaled value {expected}"
            )

    @property
    def slug(self) -> str:
        return "-".join(
            (
                "ppo",
                self.env.replace("/", "_"),
                self.backend,
                self.variant,
                self.reward,
                f"kappa{_kappa_label(self.kappa)}",
                f"{self.num_seeds}seeds",
            )
        )


def make_disruption_sweep_jobs(
    *,
    backend: str = "bohm_gyrobohm",
    envs: Iterable[str] | None = None,
    kappas: Iterable[float] = DEFAULT_KAPPAS,
    variant: str = "realistic",
    total_steps: int = 2_000_000,
    eval_freq: int = 250_000,
    num_seeds: int = 3,
    seed: int = 0,
) -> tuple[DisruptionSweepJob, ...]:
    """Build an environment × task-penalty multiplier matrix."""
    selected_envs = phase_envs() if envs is None else tuple(envs)
    selected_kappas = tuple(kappas)
    if not selected_kappas:
        raise ValueError("kappas cannot be empty")

    jobs: list[DisruptionSweepJob] = []
    for env in selected_envs:
        task = parse_env_and_backend(env, backend).task
        if task.terminal_penalty is None:
            raise ValueError(f"task {env!r} has no terminal penalty")
        for kappa in selected_kappas:
            jobs.append(
                DisruptionSweepJob(
                    env=env,
                    backend=backend,
                    variant=variant,
                    reward=task.reward,
                    task_terminal_penalty=task.terminal_penalty,
                    kappa=kappa,
                    penalty=kappa * task.terminal_penalty,
                    total_steps=total_steps,
                    eval_freq=eval_freq,
                    num_seeds=num_seeds,
                    seed=seed,
                )
            )
    return tuple(jobs)


__all__ = [
    "DEFAULT_KAPPA",
    "DEFAULT_KAPPAS",
    "DisruptionSweepJob",
    "calibrated_disruption_penalty",
    "make_disruption_sweep_jobs",
    "phase_envs",
]
