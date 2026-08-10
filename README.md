# plasmax

`plasmax` provides JAX-native fusion-control environments built on
[TORAX](https://github.com/google-deepmind/torax). It packages machine-inspired
ITER, SPARC, STEP, and KSTAR tasks behind one explicit-state Envelope API while
keeping agents, training systems, and publication code outside the installed
library.

The environments are intended for control research. Their configurations retain
the physical values, equilibria, transport choices, and learned-model assets
tracked by this repository, but they are reduced control benchmarks rather than
complete device digital twins. Physical and solver terminations are part of the
control problem and remain observable through the environment boundary.

## Install

`plasmax` requires Python 3.12 or newer.

```bash
pip install plasmax
```

The published distribution contains only `plasmax` and its runtime task data.
It does not install agents, Gymnax adapters, Rejax, W&B, training launchers, or
study code.

## Quick start

```python
import jax
import jax.numpy as jnp

import plasmax

env = plasmax.make(
    "iter/hybrid/flattop",
    backend="cgm",
)

state, info = env.init(jax.random.key(0))
action = jnp.zeros(env.action_space.shape, dtype=env.action_space.dtype)
state, info = env.step(state, action)

print(info.obs, info.reward)
print(info.terminated, info.truncated)
```

`realistic` is the default variant for `make`, `load_env`, and
`load_scenario`. It applies the task's configured sensor, resolution, filter,
delay, action, and transition-wise physics-randomization behavior. Use
`variant="oracle"` only for an explicit oracle ablation.

Spaces are properties, so use `env.action_space.shape` and
`env.observation_space.shape`. The scalar environment returned by a loader does
not autoreset or vectorize itself. Its state owns environment PRNG streams;
policies continue to own their own keys.

## Tasks and backends

Named tasks resolve packaged YAML and data from any working directory:

| Task aliases | Compatible backend aliases |
|---|---|
| `iter/{baseline,hybrid,advanced}/{rampup,flattop,rampdown}` | `cgm`, `qlknn`, `bohm_gyrobohm`, `tglfnn`, `tglfnn_nr` |
| `sparc/{prd,reduced_field}/{rampup,flattop,rampdown}` | `cgm`, `qlknn`, `bohm_gyrobohm`, `tglfnn`, `tglfnn_nr` |
| `step` | `bohm_gyrobohm`, `tglfnn_spherical` |
| `kstar` | `fusion_lstm` |

The fast packaged fixture is available through `plasmax.make("test")`.
Explicit YAML paths are also accepted by `load_env` and `load_scenario`.
Unsupported environment/backend pairs are rejected before construction.

TGLFNN-UKAEA is still an eager transitive TORAX dependency. Repository clones
retain the uv-only `v0.2.0-draft` override until the compatible release is on
PyPI; true backend-level optionality is therefore deferred rather than hidden
behind an installation claim.

Every leaf task YAML owns its reward and terminal-penalty defaults:

```yaml
task:
  reward: lh_transition
  terminal_penalty: -99.99568287525884
```

Callers normally omit `reward` and `disruption_penalty`; the loader then uses
the task metadata. Explicit overrides are still supported, including
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

If termination and the time limit coincide, termination wins. Internal TORAX
physics precision is unchanged by the float32 reward boundary.

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

## Public API

The supported top-level API is deliberately small:

```python
from plasmax import (
    EnvState,
    PlasmaxEnv,
    ScenarioConfig,
    TrajectoryStep,
    collect_episode,
    collect_episodes,
    load_env,
    load_scenario,
    make,
    registry,
)
```

Control types, rewards, spaces, and specialized wrappers are available from
their explicit modules. Training and agent APIs are intentionally not exported
by the installed package.

## Clone-only baselines and development

Clone the repository when running baselines or studies:

```bash
git clone https://github.com/TheodoreWolf/plasmax.git
cd plasmax

uv sync --no-dev --group research

uv run python scripts/train_ppo.py \
  --env.env-setup iter/hybrid/flattop \
  --env.backend cgm \
  --env.variant realistic
```

PPO, SAC, baseline evaluation, and discharge rollout are generic clone-only
launchers under `scripts/`. Their reward and terminal penalty are inherited
from task metadata unless explicitly overridden. New W&B runs use the
`flair/plasmax` project.

For development and artifact-generation dependencies:

```bash
uv sync --group dev
uv sync --group dev --group artifacts

uv run ruff check .
uv run pytest
uv run pytest experiments/tests/
uv run pytest -o addopts="" -m integration tests/
```

Default pytest runs exclude integration tests. CI runs the explicit integration
matrix separately across seven scenario shards.

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

The library is licensed under the [Apache License 2.0](LICENSE). TORAX and
packaged third-party data/model assets retain their own attribution and license
terms; the relevant notices are shipped adjacent to those assets.
