"""Re-aggregate saved per-sensor backend errors without rerunning rollouts."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import tyro

from benchmarks.backend_agreement import (
    METRIC_SENSORS,
    _pareto_backends,
    _selected_sensor_mean,
)


@dataclasses.dataclass(frozen=True)
class Config:
    input: Path = Path("outputs/backend_pareto_flattop_a100.json")
    output: Path = Path("outputs/backend_pareto_flattop_a100_nonredundant.json")
    metric_sensors: tuple[str, ...] = METRIC_SENSORS


def main(cfg: Config) -> None:
    data = json.loads(cfg.input.read_text())
    for point in data["metrics"].values():
        point["all_sensor_balanced_mse"] = point["sensor_balanced_mse"]
        point["all_sensor_balanced_mre"] = point["sensor_balanced_mre"]
        point["sensor_balanced_mse"] = _selected_sensor_mean(
            point["sensor_mse"], cfg.metric_sensors
        )
        point["sensor_balanced_mre"] = _selected_sensor_mean(
            point["sensor_mre"], cfg.metric_sensors
        )
        point["error"] = point[f"sensor_balanced_{data['error_metric']}"]

    data["metric_sensors"] = list(cfg.metric_sensors)
    data["metric_sensor_count"] = len(cfg.metric_sensors)
    data["aggregation"] = (
        "profile errors are averaged over radius first, then the selected "
        "metric sensors are averaged equally over time"
    )
    data["reaggregated_from"] = str(cfg.input)
    data["pareto_frontier"] = _pareto_backends(data["metrics"])

    cfg.output.parent.mkdir(parents=True, exist_ok=True)
    cfg.output.write_text(json.dumps(data, indent=2) + "\n")
    print(f"Saved {cfg.output}")


if __name__ == "__main__":
    main(tyro.cli(Config))
