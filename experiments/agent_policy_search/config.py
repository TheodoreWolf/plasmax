"""Serializable configuration shared by the master and evaluation workers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class TaskConfig:
    env_setup: str
    backend: str | None = None
    variant: Literal["realistic", "oracle"] = "realistic"
    max_steps: int | None = None
    reward: str | None = None
    time_aware: bool = False


@dataclass(frozen=True)
class ModelConfig:
    model: str
    api_base: str | None = None
    max_tokens: int = 8192
    temperature: float | None = None
    timeout_seconds: float = 180.0


@dataclass(frozen=True)
class EvaluationConfig:
    task: TaskConfig
    seeds: tuple[int, ...]
    timeout_seconds: float = 1800.0


@dataclass(frozen=True)
class TrackingConfig:
    project: str = "plasmax"
    entity: str | None = None
    group: str = "agent-policy-search"
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class SearchConfig:
    task: TaskConfig
    model: ModelConfig
    seed_policy: Path
    proposals: int = 10
    development_seed: int = 0
    development_episodes: int = 64
    heldout_seed: int = 10000
    heldout_episodes: int = 64
    evaluation_timeout_seconds: float = 1800.0
    output_root: Path = Path("outputs/agent_policy_search")
    tracking: TrackingConfig = field(default_factory=TrackingConfig)

    def seed_banks(self) -> dict[str, tuple[int, ...]]:
        """Make episode seeds explicit; never expose heldout outcomes to search."""
        if self.proposals < 0:
            raise ValueError("proposals must be nonnegative")
        banks = {}
        for name, start, count in (
            ("development", self.development_seed, self.development_episodes),
            ("heldout", self.heldout_seed, self.heldout_episodes),
        ):
            if count <= 0 or start < 0 or start + count > 2**32:
                raise ValueError(f"{name} requires a positive count and uint32 seeds")
            banks[name] = tuple(range(start, start + count))
        if set(banks["development"]) & set(banks["heldout"]):
            raise ValueError("development and heldout seeds must be disjoint")
        if self.evaluation_timeout_seconds <= 0 or self.model.timeout_seconds <= 0:
            raise ValueError("timeouts must be positive")
        if self.model.max_tokens <= 0 or not self.model.model.strip():
            raise ValueError("model name and a positive max_tokens are required")
        return banks
