"""Shared contracts for the phase-wise algorithm baseline study.

This module deliberately contains no training code. It reads each task's reward
contract from packaged metadata and defines deterministic seed keys plus stable
run labels used by the individual trainers.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable

import jax
import jax.numpy as jnp

from plasmax.environment.config import parse_env_and_backend
from plasmax.environment.merge import valid_env_backend_combos

BASELINE_ALGORITHMS = (
    "ppo",
    "sac",
    "direct_policy",
    "direct_knots_1",
    "direct_knots_10",
    "direct_knots_100",
)
BASELINE_VARIANTS = ("oracle", "realistic")


def baseline_envs(*, include_step: bool = True) -> tuple[str, ...]:
    """Return every registered TORAX phase task, excluding KSTAR."""
    envs = [env for env in sorted(valid_env_backend_combos()) if env != "kstar"]
    if not include_step:
        envs.remove("step")
    return tuple(envs)


def env_phase(env: str) -> str:
    """Return the reward phase represented by an environment registry key."""
    if env == "step":
        # STEP is a stationary non-inductive flat-top, despite being a
        # single-file device rather than a nested ``.../flattop`` alias.
        return "flattop"
    phase = env.rsplit("/", 1)[-1]
    if phase not in {"rampup", "flattop", "rampdown"}:
        raise ValueError(
            f"baseline env {env!r} must be rampup, flattop, or rampdown; "
            "KSTAR is excluded"
        )
    return phase


def reward_for_env(env: str, backend: str = "bohm_gyrobohm") -> str:
    """Return the reward stored with an environment's task metadata."""
    return parse_env_and_backend(env, backend).task.reward


def validate_reward(
    env: str,
    reward: str | None,
    backend: str = "bohm_gyrobohm",
) -> None:
    """Validate an explicit override; omitted rewards inherit task metadata."""
    if reward is None:
        return
    expected = reward_for_env(env, backend)
    if reward != expected:
        raise ValueError(
            f"baseline reward mismatch for {env!r}: got {reward!r}, "
            f"expected {expected!r}"
        )


def seed_keys(base_seed: int, num_seeds: int) -> jax.Array:
    """Return keys invariant to how a seed set is batched across jobs.

    ``jax.random.split(key, n)`` changes the key assigned to seed zero when
    ``n`` changes. Folding an explicit seed index into one base key lets a
    pilot, a five-seed shard, and the final ten-seed job use the same seed IDs.
    """
    if num_seeds <= 0:
        raise ValueError("num_seeds must be positive")
    indices = jnp.arange(num_seeds, dtype=jnp.uint32)
    key = jax.random.PRNGKey(base_seed)
    return jax.vmap(lambda index: jax.random.fold_in(key, index))(indices)


def indexed_keys(key: jax.Array, count: int) -> jax.Array:
    """Fold stable indices into one key without count-dependent splitting."""
    if count <= 0:
        raise ValueError("count must be positive")
    indices = jnp.arange(count, dtype=jnp.uint32)
    return jax.vmap(lambda index: jax.random.fold_in(key, index))(indices)


def run_slug(
    algorithm: str,
    env: str,
    backend: str,
    variant: str,
    reward: str | None,
    num_seeds: int,
) -> str:
    """Return a filesystem- and W&B-friendly stable baseline run label."""
    return "-".join(
        (
            algorithm,
            env.replace("/", "_"),
            backend.replace("/", "_"),
            variant,
            reward or "task",
            f"{num_seeds}seeds",
        )
    )


@dataclasses.dataclass(frozen=True)
class BaselineJob:
    """One accelerator job containing a vmapped set of training seeds."""

    algorithm: str
    env: str
    backend: str
    variant: str
    reward: str
    total_steps: int
    eval_freq: int
    num_seeds: int
    seed: int = 0

    def __post_init__(self) -> None:
        if self.algorithm not in BASELINE_ALGORITHMS:
            raise ValueError(f"unknown baseline algorithm {self.algorithm!r}")
        if self.env not in baseline_envs():
            raise ValueError(f"unknown or excluded baseline env {self.env!r}")
        if self.variant not in BASELINE_VARIANTS:
            raise ValueError(f"unknown baseline variant {self.variant!r}")
        allowed = valid_env_backend_combos()[self.env]
        if self.backend not in allowed:
            raise ValueError(
                f"backend {self.backend!r} is invalid for {self.env!r}; "
                f"allowed: {sorted(allowed)}"
            )
        validate_reward(self.env, self.reward, self.backend)
        if self.total_steps <= 0 or self.eval_freq <= 0 or self.num_seeds <= 0:
            raise ValueError("total_steps, eval_freq, and num_seeds must be positive")

    @property
    def slug(self) -> str:
        return run_slug(
            self.algorithm,
            self.env,
            self.backend,
            self.variant,
            self.reward,
            self.num_seeds,
        )


def make_baseline_jobs(
    *,
    backend: str = "bohm_gyrobohm",
    envs: Iterable[str] | None = None,
    algorithms: Iterable[str] = BASELINE_ALGORITHMS,
    variants: Iterable[str] = BASELINE_VARIANTS,
    num_seeds: int = 10,
    seed: int = 0,
    policy_steps: int = 10_000_000,
    knot_steps: int = 1_000_000,
    policy_eval_freq: int = 1_000_000,
    knot_eval_freq: int = 50_000,
) -> tuple[BaselineJob, ...]:
    """Build the complete non-KSTAR phase-task experiment matrix."""
    jobs: list[BaselineJob] = []
    selected_envs = baseline_envs() if envs is None else tuple(envs)
    for env in selected_envs:
        reward = reward_for_env(env, backend)
        for variant in variants:
            for algorithm in algorithms:
                is_knot = algorithm.startswith("direct_knots_")
                total_steps = knot_steps if is_knot else policy_steps
                eval_freq = knot_eval_freq if is_knot else policy_eval_freq
                jobs.append(
                    BaselineJob(
                        algorithm=algorithm,
                        env=env,
                        backend=backend,
                        variant=variant,
                        reward=reward,
                        total_steps=total_steps,
                        eval_freq=min(eval_freq, total_steps),
                        num_seeds=num_seeds,
                        seed=seed,
                    )
                )
    return tuple(jobs)


__all__ = [
    "BASELINE_ALGORITHMS",
    "BASELINE_VARIANTS",
    "BaselineJob",
    "baseline_envs",
    "env_phase",
    "indexed_keys",
    "make_baseline_jobs",
    "reward_for_env",
    "run_slug",
    "seed_keys",
    "validate_reward",
]
