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

env = plasmax.make(
    "iter/hybrid/flattop",
    backend="bohm_gyrobohm",
)

state, info = env.init(jax.random.key(0))
action = jnp.zeros(env.action_space.shape, dtype=env.action_space.dtype)
state, info = env.step(state, action)

print(info.obs, info.reward)
print(info.terminated, info.truncated)
```

Environments are loaded through the `make` function. TORAX environment names
are structured as `{tokamak}/{scenario}/{phase}` and require an explicit
compatible backend. KSTAR is a standalone environment and takes no backend.

By default, it applies the modifications recommended by [Challenges of Real World Reinforcement Learning](https://arxiv.org/abs/1904.12901), use `variant="oracle"`, to remove these.

The environments use the [Envelope](https://github.com/keraJLi/envelope) API and contracts, this includes e.g. explicit truncation versus termination.

## Tasks and backends

| Task aliases | Compatible backend aliases |
|---|---|
| `iter/{baseline,hybrid,advanced}/{rampup,flattop,rampdown}` | `cgm`, `qlknn`, `bohm_gyrobohm`, `tglfnn`, `tglfnn_nr` |
| `sparc/{prd,reduced_field}/{rampup,flattop,rampdown}` | `cgm`, `qlknn`, `bohm_gyrobohm`, `tglfnn`, `tglfnn_nr` |
| `step/spp_001_ec_hd/flattop` | `bohm_gyrobohm_step`, `tglfnn_spherical` |
| `mock/circular/smoke` | `mock` |
| `kstar_worldmodel` | None; standalone environment |

Unsupported environment/backend pairs are rejected before construction.

```python
smoke = plasmax.make("mock/circular/smoke", backend="mock")
kstar = plasmax.make("kstar_worldmodel")
```

NB: the `tglfnn_spherical` backend requires a repository clone for now.
[TGLFNN-UKAEA](https://github.com/ukaea/tglfnn-ukaea) is still an eager
transitive TORAX dependency. TORAX 1.4.3 pins `fusion-surrogates` 0.4.6, whose
TGLFNN extra pins the older 0.1.0 weights. Repository clones use a uv-only
override to the final PyPI 0.2.0 weights until TORAX adopts `fusion-surrogates`
0.4.7. Published installs still follow TORAX's dependency metadata.

Equilibria generated with
[FreeGSNKE](https://github.com/FusionComputingLab/freegsnke) are committed
artifacts, so FreeGSNKE is not a runtime dependency.

Every leaf task YAML owns its reward and terminal-penalty defaults:

```yaml
task:
  reward: lh_transition
  terminal_penalty: -100
```

By default, `reward` and `disruption_penalty`; uses the task metadata.
Explicit overrides are supported, including
`disruption_penalty=0.0`. Ramp-up tasks use `lh_transition`, flat-top and STEP
tasks use `P_diff`, and ramp-down tasks use `rampdown`. KSTAR uses its native
learned-model reward and has no terminal penalty.

```python
env = plasmax.make("iter/advanced/rampup", backend="qlknn")

oracle_ablation = plasmax.make(
    "iter/advanced/rampup",
    backend="qlknn",
    variant="oracle",
    reward="Q_fusion",
    disruption_penalty=0.0,
)
```

## Environment boundary

`init`, `step`, and `reset` return `(state, info)`. A transition exposes:

- `info.obs`: the post-transition flat observation;
- `info.reward`: a scalar float32 RL-boundary reward;
- `info.terminated`: a physical or solver termination;
- `info.truncated`: the configured time-limit cutoff;
- `info.termination_code`: the environment's termination reason.

If termination and the time limit coincide, termination wins.

Fixed-shape rollout collection is part of the installed library:

```python
from plasmax import collect_episode


def act(obs, key):
    del obs, key
    return jnp.zeros(env.action_space.shape, dtype=env.action_space.dtype)


trajectory = collect_episode(
    act,
    env,
    jax.random.key(1),
    num_steps=env.max_steps,
)
```

The collector retains the first terminal transition, stops stepping the
environment, and pads the remaining fixed-size output with `valid=False`.

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
training/           clone-only training adapters
scripts/            generic baseline and rollout launchers
tools/              artifact and equilibrium generation
benchmarks/         backend agreement and throughput benchmarks
experiments/        research studies and plotting
tests/              library and release tests
```

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
Coming soon, once I get the paper out...
