# Clone-only launchers

These entrypoints are available to repository clones and are not included in
the installed `plasmax` distribution. Run them from the repository root with
`uv run python scripts/<name>.py --help`.

| Entrypoint | Purpose |
|---|---|
| `train_ppo.py` | Train an upstream Rejax PPO policy through the thin Envelope/Gymnax adapter. |
| `train_sac.py` | Train an upstream Rejax SAC policy through the same adapter. |
| `evaluate_baselines.py` | Evaluate the repository-only baseline controllers. |
| `rollout_discharge.py` | Roll out a constant-action discharge and render its traces. |

`_runtime.py` and `project_paths.py` are shared private helpers rather than
entrypoints. New runs default to the realistic environment variant and inherit
their reward and terminal penalty from task YAML metadata unless explicitly
overridden.

Research studies and publication plotting live under `experiments/`; cluster
launchers are under `experiments/cluster/`. Artifact, calibration, manifest,
and equilibrium builders live under `tools/`.
