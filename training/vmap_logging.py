"""Cross-seed logging for vmapped PPO training.

When ``algo.train`` is vmapped over several training seeds, the eval callback
fires once per seed via :func:`jax.debug.callback` (which, unlike
:func:`jax.experimental.io_callback`, batches a vmap by *looping* the callback
over the mapped axis — one concrete call per seed). Those calls arrive in
arbitrary order, so :class:`SeedBufferLogger` buffers metrics keyed by
``(global_step, run_idx)`` and, once all ``num_seeds`` entries for a step have
landed, averages across seeds and emits a single wandb log at that step —
attaching the cross-seed std of every metric so runs carry statistical error
bars.

Adapted from FLAIROx/envelope-bench ``ppo_vmap/logger.py`` (buffer-by-step
idea), trimmed to plasmax's needs: no HDF5/orbax/nnx dependency, wandb + stdout
plus an optional final ``.npz`` dump of the full per-seed history.
"""

import csv
import json
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import wandb

from scripts.project_paths import wandb_dir

__all__ = ["SeedBufferLogger"]


def _finite_mean_and_std(values: list[float], ddof: int) -> tuple[float, float]:
    """Return finite-only statistics without warning on an all-NaN metric."""

    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan"), float("nan")
    mean = float(finite.mean())
    std = float(finite.std(ddof=ddof)) if finite.size > ddof else float("nan")
    return mean, std


class SeedBufferLogger:
    """Buffers per-seed metrics from a vmapped run and logs their mean ± std.

    Args:
        num_seeds: Number of vmapped training seeds. A step is flushed once
            this many per-seed entries have been collected for it.
        run_name: wandb run name.
        project/entity/group/mode: wandb.init passthrough.
        config: Config dict recorded to wandb.
        out_dir: If set, ``finish()`` dumps the full per-seed metric history to
            ``{out_dir}/{run_name}_history.npz`` (no extra deps; numpy only).
        print_fn: Sink for the per-step status line (defaults to ``print``).
    """

    def __init__(
        self,
        *,
        num_seeds: int,
        run_name: str,
        project: str = "plasmax",
        entity: str = "flair",
        group: str = "debug",
        mode: str = "online",
        config: dict | None = None,
        out_dir: str | None = None,
        seed_ids: list[int] | tuple[int, ...] | None = None,
        job_type: str | None = None,
        tags: list[str] | tuple[str, ...] | None = None,
        print_fn: Callable[[str], None] = print,
    ) -> None:
        if num_seeds <= 0:
            raise ValueError("num_seeds must be positive")
        self.num_seeds = num_seeds
        self.seed_ids = tuple(range(num_seeds)) if seed_ids is None else tuple(seed_ids)
        if len(self.seed_ids) != num_seeds:
            raise ValueError("seed_ids length must equal num_seeds")
        self.print_fn = print_fn
        self.start_time: float | None = None
        self._prev_flush_time: float | None = None
        self._prev_flush_step: int = 0
        self._sps: float = 0.0

        # step -> {run_idx: {metric: float}}
        self._buffers: dict[int, dict[int, dict[str, float]]] = {}
        # Retained per-seed history for the optional npz dump: step -> per_run dict.
        self._history: dict[int, dict[str, list[float]]] = {}

        self.out_dir = Path(out_dir) if out_dir is not None else None
        self._config = config or {}
        self._run_name = run_name
        self._csv_path: Path | None = None
        self._csv_keys: tuple[str, ...] | None = None
        if self.out_dir is not None:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            self._csv_path = self.out_dir / f"{run_name}_metrics.csv"
            config_path = self.out_dir / f"{run_name}_config.json"
            config_path.write_text(
                json.dumps(self._config, indent=2, sort_keys=True, default=str) + "\n"
            )

        self._run = wandb.init(
            project=project,
            entity=entity,
            dir=wandb_dir(),
            name=run_name,
            group=group,
            mode=mode,
            config=config,
            job_type=job_type,
            tags=list(tags) if tags is not None else None,
        )

    # ------------------------------------------------------------------
    # Called from inside vmap via jax.debug.callback (once per seed).
    # ------------------------------------------------------------------
    def log(self, global_step, run_idx, metrics: dict) -> None:
        """Buffer one seed's metrics; flush the step once all seeds report."""
        step = int(global_step)
        run_idx = int(run_idx)
        metrics = {k: float(v) for k, v in metrics.items()}

        self._buffers.setdefault(step, {})[run_idx] = metrics
        if len(self._buffers[step]) == self.num_seeds:
            self._flush_step(step)

    def _flush_step(self, step: int) -> None:
        per_seed = self._buffers.pop(step)
        keys = list(next(iter(per_seed.values())).keys())

        per_run = {k: [per_seed[r][k] for r in sorted(per_seed)] for k in keys}
        ddof = 1 if self.num_seeds > 1 else 0
        stats = {k: _finite_mean_and_std(per_run[k], ddof) for k in keys}
        mean = {k: stats[k][0] for k in keys}
        std = {k: stats[k][1] for k in keys}
        self._history[step] = per_run
        self._write_local_rows(step, per_run)

        # Steps-per-second from wall time between flushes.
        now = time.time()
        if self._prev_flush_time is not None:
            dt = now - self._prev_flush_time
            if dt > 0:
                self._sps = (step - self._prev_flush_step) / dt
        self._prev_flush_time = now
        self._prev_flush_step = step
        elapsed = now - self.start_time if self.start_time is not None else 0.0

        log_data = {"time/sps": self._sps, "time/total_time": elapsed}
        for k in keys:
            log_data[k] = mean[k]
            if self.num_seeds > 1:
                log_data[f"{k}_seed_std"] = std[k]
        self._run.log(log_data, step=step)

        ret = mean.get("evaluation/return_mean", float("nan"))
        ret_std = std.get("evaluation/return_mean", 0.0)
        self.print_fn(
            f"step={step:>10d}  return={ret:.3f} ±{ret_std:.3f} (across seeds)"
            f"  sps={self._sps:.0f}"
        )

    def _write_local_rows(self, step: int, per_run: dict[str, list[float]]) -> None:
        """Append one long-form row per training seed at a flushed checkpoint."""
        if self._csv_path is None:
            return
        keys = tuple(per_run)
        if self._csv_keys is None:
            self._csv_keys = keys
        elif keys != self._csv_keys:
            raise ValueError(
                "metric keys changed during a run: "
                f"expected {self._csv_keys}, got {keys}"
            )
        write_header = not self._csv_path.exists()
        with self._csv_path.open("a", newline="") as csv_file:
            writer = csv.DictWriter(
                csv_file,
                fieldnames=("train_steps", "seed", "seed_index", *keys),
            )
            if write_header:
                writer.writeheader()
            for run_idx, seed_id in enumerate(self.seed_ids):
                writer.writerow(
                    {
                        "train_steps": step,
                        "seed": seed_id,
                        "seed_index": run_idx,
                        **{key: per_run[key][run_idx] for key in keys},
                    }
                )
            csv_file.flush()

    def log_batch(self, global_step: int, metrics: dict[str, np.ndarray]) -> None:
        """Log a host-side ``(num_seeds,)`` metric batch in one call."""
        arrays = {name: np.asarray(value) for name, value in metrics.items()}
        for name, value in arrays.items():
            if value.shape != (self.num_seeds,):
                raise ValueError(
                    f"metric {name!r} must have shape ({self.num_seeds},), "
                    f"got {value.shape}"
                )
        for run_idx in range(self.num_seeds):
            self.log(
                global_step,
                run_idx,
                {name: value[run_idx] for name, value in arrays.items()},
            )

    # ------------------------------------------------------------------
    def log_once(self, data: dict) -> None:
        """Record one-off scalars (compile/lower/train times) to wandb summary."""
        for k, v in data.items():
            self.print_fn(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
        self._run.summary.update(data)

    def finish(self) -> None:
        try:
            import jax

            memory_stats = jax.local_devices()[0].memory_stats()
        except (IndexError, RuntimeError):
            memory_stats = None
        if memory_stats:
            bytes_per_gib = 1024**3
            memory_summary = {
                f"memory/{name}_gib": value / bytes_per_gib
                for name, value in memory_stats.items()
                if name in {"bytes_in_use", "bytes_limit", "peak_bytes_in_use"}
            }
            if memory_summary:
                self.log_once(memory_summary)
        if self.out_dir is not None and self._history:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            steps = sorted(self._history)
            keys = list(self._history[steps[0]])
            # arrays: (num_logged_steps, num_seeds) per metric, plus a steps vector.
            dump = {
                "steps": np.asarray(steps, dtype=np.int64),
                "seed_ids": np.asarray(self.seed_ids, dtype=np.int64),
            }
            for k in keys:
                dump[k] = np.asarray([self._history[s][k] for s in steps])
            path = self.out_dir / f"{self._run_name}_history.npz"
            np.savez(path, **dump)
            self.print_fn(f"Saved per-seed history: {path}")
        self._run.finish()
