# Phase-wise control baselines

This study compares learned policies and direct differentiable control on every
registered TORAX phase task, excluding the KSTAR world-model environment.
It holds the Bohm–GyroBohm backend fixed so the environment axis is the device,
scenario, and phase rather than a mixed environment/backend sweep.

## Matrix

- Environments: ITER baseline/hybrid/advanced and SPARC PRD/reduced-field,
  each at ramp-up, flat-top, and ramp-down, plus the stationary STEP flat-top
  (16 tasks).
- Variants: `oracle`, `realistic`.
- Algorithms: PPO, SAC, direct truncated-BPTT policy, and direct open-loop
  schedules with 1, 10, or 100 requested knots.
- Training seeds: 10 independent optimizer seeds, vmapped inside each job.
- Evaluation: deterministic policy actions on the same fixed bank of four episode
  keys for every training seed and algorithm.
- Policy budget: 10 million training simulator transitions per seed.
- Knot-schedule budget: 10 million training simulator transitions per seed.

The canonical publication manifest is `sweeps/baseline_v1_10m.tsv` and contains
192 jobs. The original 1M-knot submission manifest is retained separately as
provenance. A 100-knot schedule has at most one effective
control per simulator transition: on 50-step SPARC ramp-down and STEP episodes,
the artifact records `requested_knots=100` and `effective_knots=50` rather than
pretending that unobservable extra controls were learned.

## Reward contract

| Phase | Reward |
|---|---|
| Ramp-up | `lh_transition` |
| Flat-top (including STEP) | `P_diff` |
| Ramp-down | `rampdown` |

This table summarizes the `task.reward` values stored in the leaf YAMLs; the
study code reads those values instead of maintaining a second mapping. The
`lh_transition` time weighting is bound to the loaded ramp-up's actual
`numerics.t_final`; it is not fixed at 100 seconds.

## Algorithm settings

PPO uses the current project defaults: 10M transitions, 1024 environments,
100-step rollouts, four epochs/minibatches, learning rate 3e-4, gamma 0.99,
GAE lambda 0.95, clipping 0.2, and 64x64 Swish actor/critic networks.

SAC is imported from Rejax. Its defaults follow the original continuous-control
configuration: two 256-unit ReLU layers, learning rate 3e-4, gamma 0.99, replay
buffer 1e6, minibatch 256, and target smoothing tau 0.005 (`polyak=0.995` in
Rejax). It uses 64 parallel environments and 64 gradient epochs per collection
step, giving update-to-data ratio 1. Observation and reward normalization are
off by default: Rejax 0.1.2's RMS count widens to float64 after TORAX enables
x64, which violates the JAX scan-carry contract. The repository does not patch
or copy Rejax internals.

The direct policy uses a deterministic 64x64 residual MLP and 64 parallel
trajectories. It differentiates windows near 32 steps (the largest divisor of
the episode length no greater than 32), carries simulator state between windows,
and cuts gradients at window boundaries. Adam uses learning rate 1e-6 and
betas `(0.7, 0.95)`; ten-seed SPARC pilots found non-finite follow-up gradients
at both SHAC's 0.002 setting and 5e-4, while 1e-6 remained finite at every
checkpoint. Non-finite updates are skipped rather than silently zeroing
individual gradient entries. This is a direct truncated-BPTT baseline, not SHAC: it does
not add SHAC's learned terminal critic.

Direct knot schedules use the same 64 parallel trajectories and truncated-BPTT
windows as the direct policy, with physical simulator state carried between
windows and gradients cut at each boundary. Adam uses learning rate 0.05. All
direct methods retain the best fixed-evaluation checkpoint because
differentiable control can cross a hard disruption or actuator-saturation cliff
after finding a good iterate.

The direct transform order is deliberately
`vmap(seed -> vmap(value_and_grad(single_rollout)))`. Moving reverse-mode AD
outside the rollout vmap triggers TORAX's adaptive-loop custom-JVP transpose
failure.

These settings are motivated by the original
[SAC paper and supplement](https://proceedings.mlr.press/v80/haarnoja18b.html),
[SHAC](https://openreview.net/forum?id=ZSKRQMvttc),
[PODS](https://proceedings.mlr.press/v139/mora21a.html), and the analysis of
[first-order gradients in stiff differentiable simulators](https://proceedings.mlr.press/v162/suh22b.html).

## Logging and confidence intervals

W&B uses project `flair/plasmax`, group `baseline-v1`, and `job_type` equal to
the algorithm. Each vmapped job logs cross-seed means and sample standard
deviations; downstream study summaries compute any confidence intervals.

Local Isambard artifacts are written below
`$SCRATCHDIR/plasmax-baselines/<study>/<array-job>_<task>-<slug>/`:

- `*_metrics.csv`: one row per training seed and evaluation checkpoint;
- `*_history.npz`: the same raw per-seed arrays;
- `*_config.json`: complete run configuration;
- `*_checkpoint.npz`: retained parameters for direct methods.

`experiments/plotting/plot_baseline_learning_curves.py` first combines every recorded
per-seed evaluation statistic into `baseline_returns.csv`, then writes
`baseline_learning_curves.csv` with a two-sided 95% Student-t interval across
training-seed means (for ten seeds, 9 degrees of freedom). Tables go under
`outputs/` and one PNG per environment goes under `plots/`, with oracle and
realistic variants as side-by-side panels when both are present.
Evaluation-episode variation within a seed is not incorrectly treated as an
independent training seed.

## Generate and submit

```bash
uv run python tools/manifests/make_baseline_manifest.py \
  --out sweeps/baseline_v1.tsv

rsync -az --exclude='.venv/' --exclude='.git/' \
  --exclude='outputs/' --exclude='plots/' ./ \
  u6oz.aip2.isambard:/home/u6oz/theow.u6oz/plasmax/

ssh u6oz.aip2.isambard \
  'cd ~/plasmax && MANIFEST=sweeps/baseline_v1.tsv STUDY=baseline-v1 \
   sbatch --array=0-191 experiments/cluster/isambard/isambard_baselines.slurm'
```

Submitting the full array queues every environment immediately. Slurm
fair-share and QOS control actual concurrency; add a `%N` running-task limit
only when it follows a measured project or workload constraint.

The production array is submitted only after shorter ten-seed pilots establish
memory use, wall time, finite gradients, and convergence behavior on the GH200
compute nodes.
