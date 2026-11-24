<p align="center">
  <img src="https://raw.githubusercontent.com/TheodoreWolf/plasmax/main/assets/plasmax-logo.png" alt="plasmax logo" width="520">
</p>

# Plasmax: differentiable & parallelizable environments for kinetic control in Tokamaks

`plasmax` provides JAX-native fusion-control environments built on
[TORAX](https://github.com/google-deepmind/torax).
The environments are inspired by real tokamak devices; ITER, SPARC, STEP, and KSTAR.
Each tokamak contains different tasks for different scenarios and phases.

Environment design is inspired by recommendations from [Challenges of Real World Reinforcement Learning](https://arxiv.org/abs/1904.12901), where missing, noisy observations are explicitly implemented.

## Install

```bash
pip install plasmax
```
We did not package the training and agent stack to make the dependencies lighter.
If you want to try training agents using our stack:

```bash
git clone https://github.com/TheodoreWolf/plasmax
cd plasmax
pip install -e . --group research
```

## Quick start

```python
import jax
import jax.numpy as jnp

import plasmax
from plasmax.wrappers import OracleWrappers, RealisticWrappers

env = plasmax.make(
    "iter/hybrid/flattop",
    backend="bohm_gyrobohm",
)
env = RealisticWrappers(env, max_steps=100)

state, info = env.init(jax.random.key(0))
action = jnp.zeros(env.action_space.shape, dtype=env.action_space.dtype)
state, info = env.step(state, action)

print(info.obs, info.reward)
print(info.terminated, info.truncated)
```

Environments are loaded through the `make` function. TORAX environment names
are structured as `{tokamak}/{scenario}/{phase}` and require an explicit
compatible backend. KSTAR is a standalone environment and takes no backend.
Convenience wrappers are available for use, to make the environments more close to reality, these are all configurable through the API.

## State and termination semantics

The environments use the [Envelope](https://github.com/keraJLi/envelope) API and contracts, this includes e.g. explicit truncation versus termination.
`init`, `reset`, and `step` return `(state, info)`.

On TORAX environments, `state` is an `EnvState` JAX pytree dataclass (`plasmax.EnvState`) with fields:

- `plasma`: a `PlasmaState` view over TORAX's `sim` (`SimState`) and `post` (`PostProcessedOutputs`). It exposes shortcut properties such as `t` or `Ip`. In particular, it contains`phys_params` physics runtime parameters which can be changed at each step.

`info` is an Envelope `InfoContainer` exposing:
- `obs`: flat observation array.
- `reward`: scalar
- `terminated`: `bool` (`termination_code != -1`).
- `truncated`: `bool` which has subtleties compared to termination, see e.g. this [paper](https://arxiv.org/abs/2506.15050).
- `termination_code`: termination reason (`-1` when not terminated).
- We also include TORAX stepping diagnostics, which help with debugging.

Termination codes (`info.termination_code`):

| Code | Meaning |
|---|---|
| `-1` | No termination (normal transition, `init`, or `reset`) |
| `1` | Safety-factor disruption (`min(q_face) < q_min_threshold`, default `0.8`) |
| `2` | Greenwald density limit exceeded (`fgw_n_e_{line_avg,volume_avg} > greenwald_threshold`, default `1.1`) |
| `3` | Solver failure (nonlinear solve failed to converge, invalid `dt`, or internal substep budget exhausted) |
| `4` | Invalid / non-finite state or observation (candidate transition is rejected and the previous valid `EnvState` and observation are retained) |

If multiple termination conditions trigger on the same step, precedence is `4 > 3 > 1 > 2`.

## Tasks and backends

| Task aliases | Compatible backend aliases |
|---|---|
| `iter/{baseline,hybrid,advanced}/{rampup,flattop,rampdown}` | `cgm`, `qlknn`, `bohm_gyrobohm`, `tglfnn`, `tglfnn_nr` |
| `sparc/{prd,reduced_field}/{rampup,flattop,rampdown}` | `cgm`, `qlknn`, `bohm_gyrobohm`, `tglfnn`, `tglfnn_nr` |
| `step/spp_001_ec_hd/flattop` | `bohm_gyrobohm_step`, `tglfnn_spherical` |
| `mock/circular/smoke` | `mock` |
| `kstar_worldmodel` | None; standalone environment |

Environments and backends are defined through YAML files, where common settings are shared across tokamaks/scenarios.
Rewards are defined in the leaf YAML. Backends are similarly defined, both can be extended by using the `extends` keyword.
```yaml
extends: flattop.yaml

task:
    reward: beta_N
```
Registering a new environment or backend can then be done through the `registry.py`, following the pattern shown there.

## Reward functions

All reward functions are defined and registered in `reward.py` and have the signature `f(state, action, next_state, termination_code) -> reward`. The default reward functions apply penalties for different terminations, where e.g. solver failures are treated differently to constraint violations (disruptions).
Some default reward functions in the API are the following.

```python
_REWARD_ALIASES: dict[str, RewardFn] = {
    "Q_fusion": Q_fusion, 
    "beta_N": beta_N,
    "P_diff_MW": P_diff_MW,
    "W_thermal": W_thermal,
    "rampdown": rampdown,
    "lh_transition": lh_transition, # for ramp-up
}
```

## Wrappers

We include wrappers that help with training and realism. 
```python
from plasmax.wrappers import NoiseWrapper, PhysicsRandomizationWrapper

env = plasmax.make("iter/baseline/flattop", "bohm_gyrobohm")
env = NoiseWrapper(PhysicsRandomizationWrapper(env))
```

The full list of wrappers is the following is available in `wrappers.py`.
Due to the number of wrappers, convenience helpers compose the standard training stacks:
```python
from plasmax.wrappers import OracleWrappers, RealisticWrappers

# OracleWrappers(env, *, max_steps=None, time_aware=False)
# ActionRescaleWrapper -> [TimeAwareWrapper] -> [ObsHistoryWrapper] -> TruncationWrapper
oracle_env = OracleWrappers(env, max_steps=100, time_aware=False)

# RealisticWrappers composes 10 wrappers. 
realistic_env = RealisticWrappers(env, max_steps=100, time_aware=False, quantize_bins=None)

```

## Contribution and Development

We welcome contributions!
To contribute, first fork the repository, then:

```bash
git clone {your_gh_username}/plasmax
cd plasmax
# We highly encourage uv for developement
uv sync --group dev
git checkout {name}/{what_you_are_changing}
```

Then you can open a PR in this repository. Make sure to run tests, CI will do this for you as well.
Please have respect for the developer's time and do not submit PRs that can not be reasonably reviewed (even with the help of agents).

See [Testing plasmax](TESTING.md) for the test commands and the local
environment trajectory benchmark used by the pull-request report.

## Repository layout

```text
src/plasmax/        installed environments, tooling, configs, and data
agents/             clone-only baseline agents
training/           clone-only training launchers, adapters, and shared utilities
scripts/            generic evaluation and rollout launchers
tools/              artifact and equilibrium generation
benchmarks/         backend agreement and throughput benchmarks
experiments/        local-only research studies (gitignored)
tests/              library and release tests
```

See [training commands](training/README.md) and
[evaluation and rollout commands](scripts/README.md) for the clone-only entrypoints.

## License and attribution

The library is licensed under the
[Apache License 2.0](https://github.com/TheodoreWolf/plasmax/blob/main/LICENSE).
TORAX and
packaged third-party data/model assets retain their own attribution and license
terms; the relevant notices are shipped adjacent to those assets.

## Agents

An AGENTS.md file is included, which has my own personal code preferences. We recommend users who want to use agents to obtain an explicit JAX skill (I've written my own, that I will open-source, when I'm happy with it), as current agents are still not great at this.
Agents were utilized throughout this work, while I did my best to check the code, mistakes remain.
In my experience, the most dangerous are comments that state mistakes or bad assumption as facts, this then further reinforce the agents in their bad ideas.

## Citation
```bibtex
@inproceedings{wolf2026plasmax,
    title={Plasmax: Differentiable \& Parallelizable Environments for Tokamak Kinetic Control},
    author={Theo Wolf and Jarek Luca Liesen and Ingmar Posner and Jakob Nicolaus Foerster},
    booktitle={Sim2Science: ML with Imperfect Scientific Models},
    year={2026},
    url={https://openreview.net/forum?id=hX9aaUMB2a}
}
```
