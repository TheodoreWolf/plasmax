# Training launchers

These entrypoints are available to repository clones and are not included in
the installed `plasmax` distribution. Run them from the repository root with
`uv run python training/<name>.py --help`.

| Entrypoint | Purpose |
|---|---|
| `train_ppo.py` | Train an upstream Rejax PPO policy through the thin Envelope/Gymnax adapter. |
| `train_sac.py` | Train an upstream Rejax SAC policy through the same adapter. |
| `train_backprop.py` | Train a feedback policy or open-loop knot schedule through native Envelope gradients. |
| `train_es.py` | Train either parameterization with full-episode OpenES or CMA-ES. |
| `train_mpc.py` | Train an online dynamics model and save its frozen MPC planner. |

This directory also contains shared training and evaluation utilities. The
algorithms live in `agents/`; research studies and publication plotting live
under `experiments/`. Generic evaluation and rollout commands remain in
[scripts/](../scripts/README.md).

New runs default to the `realistic` research label, which selects
`RealisticWrappers(make(...))`; `oracle` selects `OracleWrappers`. The launchers
inherit the reward from task YAML metadata unless explicitly overridden.
Termination shaping belongs to the reward function. Wrapper options are passed
to the composition helper. Training
defaults to online W&B logging.

All agents expose `train(rng) -> (state, results)` and `make_act(state)`. Backprop,
ES and MPC use native Envelope training; PPO/SAC use upstream Rejax. Whole training
can be jitted and vmapped across seeds with the same static configuration.
TGLFNN training seeds must run as separate single-seed processes.

Agents predict absolute actions using the environment's existing action space
and scaling. Backprop and ES feedback policies use a zero-initialized output
head followed by `tanh`; APG applies the same transform across its planned
sequence and scales to the native action bounds. These deterministic policies
and the zero-initialized Backprop/ES open-loop knots start at the action-range
midpoint. PPO/SAC retain upstream actor initialization; SAPO/MPC retain their
existing action parameterizations. No agent reads reset actuator defaults to
initialize its policy. Simulator reset values, action history, and rate limits
still use those defaults.

The `residual_policy` and `action_setpoint` options, PPO's residual-only
`initial_log_std`, and reset-only Backprop/APG `init_seed` have been removed.
ES retains `init_seed` for its optimizer's prototype parameter tree.

Evaluation callbacks log `evaluation/nonfinite_reward_rate` over valid rollout
transitions. This diagnostic does not replace nonfinite rewards, stop execution,
or inspect training rewards.

TORAX evaluations for PPO, SAC, Backprop, and ES share the physics scalar
definitions in `training/wandb_logging.py`. In particular,
`obs/beta_N` averages the final valid normalized beta across evaluation episodes.
Training logs also retain `seeds/{seed}/obs/beta_N` and, for multiple training
seeds, `obs/beta_N_seed_std`. Saved-policy evaluation uses the same definitions;
environments without TORAX plasma state do not emit these physics quantities.
Live chart helpers are maintained in `training/wandb_figures.py`; training does
not depend on the ignored local plotting workspace.

By default, every training launcher saves one inference-only MessagePack policy
per seed. Files default to unique paths under `outputs/policies`; `--checkpoint-dir`
selects a directory with run/seed filenames. The files include the inference
architecture, parameters, normalizers, interface, and run metadata, rather than
optimizer or replay state. Loading does not need a training-state template or an
environment. New artifacts use format version 2 and contain no policy setpoint.
Version-1 artifacts are rejected; evaluate them with the code revision that
created them. There is no legacy checkpoint reader or training-resumption API.
PPO accepts `--no-save-policy` to skip policy export and W&B model artifacts.

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
uv run python training/train_backprop.py --mode open_loop --env.variant realistic
uv run python training/train_mpc.py --env.variant realistic
```

## Evolution strategies

ES holds each candidate's parameters fixed for a full episode and optimizes
total return. The shared agent supports feedback policies and open-loop knot
schedules independently of the search strategy:

```bash
uv run python training/train_es.py \
    --parameterization policy --es.strategy open_es --env.variant realistic
uv run python training/train_es.py \
    --parameterization open_loop --es.strategy cma_es --env.variant realistic
```

OpenES uses antithetic sampling, centred fitness ranks and clipped Adam; CMA-ES
uses its native covariance adaptation. `--es.population-size` counts candidates
(64 by default, including both signs of the 32 OpenES perturbation pairs).
OpenES requires an even population. `--es.num-rollouts` counts full episodes per
candidate (one by default), with shared reset seeds across candidates and fresh
seeds each generation. Knot runs enable time-aware observations automatically.

A generation costs `population_size × num_rollouts × episode_steps` scheduled
simulator steps. Training runs only complete generations within
`--es.total-timesteps` and rejects budgets smaller than one generation. Episode
horizons are never shortened to fit the budget. Evaluation runs initially, at
generation boundaries crossing `--es.eval-freq`, and finally; it and inference
export use the current distribution mean. W&B records the strategy and
parameterization alongside the existing evaluation metrics, generation and
search scale. Logging defaults to online and training defaults to one seed.

The defaults are starting settings, not tuned results. Full CMA-ES stores a
dense covariance, so the small open-loop parameterization is the practical
starting point for that strategy. Artifacts store inference parameters and
metadata only; ES training resumption is not supported.

## PPO wrapper ablations

PPO's `--env.variant` accepts the `oracle` and `realistic` presets, or a
slash-separated selection of wrapper aliases. Selected wrappers always apply
in this order, regardless of their position in the string:

| Alias | Effect |
|---|---|
| `physics` | Per-transition physics randomization |
| `noise` | Observation noise |
| `resolution` | Profile downsampling |
| `filter` | Observation selection |
| `delay` | Observation delay |

Each wrapper uses the task/backend configuration for its settings. Repeated
aliases apply only once; unknown aliases are errors. Use `oracle` for no optional
effects. Action scaling, configured history, optional time awareness, and
truncation remain part of every composition. `--env.quantize-bins` is supported
only with the `realistic` preset.

For a cumulative ablation in realistic order, use these five conditions:

| Condition | `--env.variant` |
|---|---|
| Oracle | `oracle` |
| Add physics randomization | `physics` |
| Add observation noise | `physics/noise` |
| Add observation downsampling/filtering | `physics/noise/resolution/filter` |
| Add observation delay | `realistic` |

For example, run the full-realism condition with three training seeds:

```bash
uv run python training/train_ppo.py \
    --env.env-setup iter/hybrid/flattop \
    --env.backend bohm_gyrobohm \
    --env.variant realistic \
    --num-seeds 3 \
    --no-save-policy
```

Change `--env.variant` to select another condition. W&B records the variant,
configuration, and evaluation metrics, including per-seed curves for multi-seed
runs. Leave `--history-dir` unset to skip local history and transfer-summary
exports. Use `--no-save-policy` for these ablations: saved-policy evaluation does
not yet reconstruct custom wrapper selections.
