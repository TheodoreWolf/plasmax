TORAX 1.4.3 rollout evaluation — 15 October 2026

**No upgrade-induced behavior change was detected on the tested stack.**
Branch `codex/torax-1.4.3-rollouts` starts from `8e48dc8`; only the TORAX
lock changes from 1.4.2 to 1.4.3. All other dependency versions are unchanged.
No environment code or configuration adaptation was needed.

The [release notes](https://github.com/google-deepmind/torax/releases/tag/v1.4.3)
add direct Fortran TGLF coupling and experimental gas-puff feedback, neither
enabled by our configurations. The
[Newton differentiation fix](https://github.com/google-deepmind/torax/commit/15d21f39091330c7be983c02f483f717b20ddd12)
removes a scheduling wrapper already inactive with our locked JAX 0.10.1.
The source review therefore predicted unchanged existing rollouts.

Both versions ran the unchanged [CI benchmark](env_trajectory.py) on f11 CPUs,
Python 3.12.12 and JAX 0.10.1: all seven groups, seed 0, zero normalized actions,
oracle and realistic wrappers, through the first termination or configured horizon.

- **126/126 cases match** in status, steps, boundary and termination cause:
  93 truncations, 32 Greenwald terminations and one q-min termination; no
  simulator errors.
- **Eight full traces match exactly** across 232 saved arrays, including plasma
  profiles, rewards, observations, actuator values and solver diagnostics.
  All saved trace values were finite. Newton values and forward/reverse gradients
  also match; finite differences agree within `1.7e-12`. All 36 targeted tests,
  including the `tglfnn_nr` canary, passed.
- **No repeatable slowdown:** median first-rollout ratio is **1.001×**, including
  compilation. Three initial timing warnings did not repeat in two
  additional paired runs each. These measurements do not establish steady-state
  throughput.

Three ITER hybrid ramp-up outcomes differ from the
[archived CI baseline](https://github.com/TheodoreWolf/plasmax/actions/runs/37907855134):

| Backend / variant | Archived CI 1.4.2 | f11, both versions |
|---|---|---|
| QLKNN / realistic | 85 steps, Greenwald | 1000 steps, truncation |
| TGLFNN / oracle | 469 steps, Greenwald | 52 steps, Greenwald |
| TGLFNN / realistic | 224 steps, Greenwald | 131 steps, Greenwald |

Fresh 1.4.2 repeats reproduce the f11 outcomes. This is a separate reproducibility
issue: Python, kernel and libc differ between runners; the cause is unproven.

Six cases were recovered with matched reruns after a temporary cache filesystem
filled. Original reports, recovery provenance, commands and detailed comparisons
are retained in [the local evidence bundle](../outputs/torax-143-audit/README.md).
The matrix excludes KSTAR and `tglfnn_nr`; other policies, seeds and JAX versions
were not evaluated.
