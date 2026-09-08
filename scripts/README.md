# Clone-only launchers

These entrypoints are available to repository clones and are not included in
the installed `plasmax` distribution. Run them from the repository root with
`uv run python scripts/<name>.py --help`.

| Entrypoint | Purpose |
|---|---|
| `train_ppo.py` | Train an upstream Rejax PPO policy through the thin Envelope/Gymnax adapter. |
| `train_sac.py` | Train an upstream Rejax SAC policy through the same adapter. |
| `train_backprop.py` | Train a feedback policy or open-loop knot schedule through native Envelope gradients. |
| `train_mpc.py` | Train an online dynamics model and save its frozen MPC planner. |
| `evaluate_policy.py` | Compare saved policies with paired episode keys, including backend transfer. |
| `rollout_discharge.py` | Roll out a constant-action discharge and render its traces. |

`_runtime.py` and `project_paths.py` are shared private helpers rather than
entrypoints. New runs default to the `realistic` research label, which selects
`RealisticWrappers(make(...))`; `oracle` selects `OracleWrappers`. The launchers
inherit reward and terminal penalty from task YAML metadata unless explicitly
overridden. Wrapper options are passed to the composition helper.

Research studies and publication plotting live under `experiments/`. Artifact,
calibration, and equilibrium builders live under `tools/`.

All agents expose `train(rng) -> (state, results)` and `make_act(state)`. Backprop
and MPC use native Envelope training; PPO/SAC use upstream Rejax. Whole training
can be jitted and vmapped across seeds with the same static configuration.
TGLFNN training seeds must run as separate single-seed processes.

Every training launcher saves one inference-only MessagePack policy per seed.
Files default to unique paths under `outputs/policies`; `--checkpoint-dir`
selects a directory with run/seed filenames. The files include the inference
architecture, parameters, normalizers, interface, and run metadata, rather than
optimizer or replay state. Loading does not need a training-state template or an
environment. There is no legacy checkpoint reader or training-resumption API.

```python
import jax

from agents.policy_io import load_policy, save_policy

state, results = jax.jit(agent.train)(rng)
path = save_policy(agent, state, results=results, metadata=run_metadata)
policy = load_policy(path)
print(policy.summary())
act = policy.make_act()  # Uses the recorded deterministic/stochastic mode.
```

```bash
uv run python scripts/train_backprop.py --mode open_loop --env.variant realistic
uv run python scripts/train_mpc.py --env.variant realistic
uv run python scripts/evaluate_policy.py --policies outputs/policies/policy.msgpack \
    --target-backend qlknn --trajectories
```

The evaluator uses the saved task configuration, permits explicit task/backend
overrides, and checks observation/action compatibility. JSON reports contain
returns and comparison metrics; optional numeric `.npz` trajectory files contain
validity/boundary masks, observations, rewards, and requested/applied controls.
Use `--env-setup` and `--backend` when a programmatic export has no named run
configuration; `--target-backend` adds a paired transfer comparison.
Training and evaluation launchers default to online W&B logging.
