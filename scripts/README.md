# Evaluation and rollout launchers

These entrypoints are available to repository clones and are not included in
the installed `plasmax` distribution. Run them from the repository root with
`uv run python scripts/<name>.py --help`.

| Entrypoint | Purpose |
|---|---|
| `evaluate_policy.py` | Compare saved policies with paired episode keys, including backend transfer. |
| `rollout_discharge.py` | Roll out a constant-action discharge and render its traces. |

`_runtime.py` and `project_paths.py` are shared private helpers rather than
entrypoints. Training launchers and the agent API are documented in
[training/README.md](../training/README.md).

Research studies and publication plotting live under `experiments/`. Artifact,
calibration, and equilibrium builders live under `tools/`.

```bash
uv run python scripts/evaluate_policy.py --policies outputs/policies/policy.msgpack \
    --target-backend qlknn --trajectories
```

The evaluator uses the saved task configuration, permits explicit task/backend
overrides, and checks observation/action compatibility. JSON reports contain
returns and comparison metrics; optional numeric `.npz` trajectory files contain
validity/boundary masks, observations, rewards, and requested/applied controls.
Use `--env-setup` and `--backend` when a programmatic export has no named run
configuration; `--target-backend` adds a paired transfer comparison.
Evaluation defaults to online W&B logging.
