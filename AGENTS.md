# AGENTS.md

This file describes repository-specific conventions for working on `plasmax`.
User-facing installation and examples are in [README.md](README.md); testing
contracts are in [TESTING.md](TESTING.md).

## Environment and commands

Use uv for every Python environment operation.

```bash
# Environment-only installed dependencies.
uv sync --no-dev

# Clone-only agents, training, plotting, and experiment dependencies.
uv sync --no-dev --group research

# Research plus tests, lint, and pre-commit.
uv sync --group dev

# ONNX conversion and artifact-generation dependencies.
uv sync --no-dev --group artifacts

uv run pytest
uv run pytest experiments/tests/
uv run pytest -o addopts="" -m integration tests/
uv run ruff check .
uv run ruff format .
uv build
```

Use `uv run python ...` for scripts and one-liners. Never use `pip install` or a
bare `python` command. New command-line entry points use Tyro rather than
argparse.

First TORAX/JAX compilation can take minutes. Repeated calls should reuse the
JAX compilation cache.

## Package boundary

The distribution is an environment-only library. Hatch builds only
`src/plasmax/`; no repository-only module may become an installed dependency.

The supported top-level API is:

```text
make, load_env, load_scenario, ScenarioConfig, PlasmaxEnv, EnvState,
TrajectoryStep, collect_episode, collect_episodes, registry
```

Keep agents, Gymnax/Rejax adaptation, programmatic training, W&B, plotting,
studies, and cluster launchers outside `src/`. Concrete clone-only modules may
be imported from a checkout, but `agents/` and `training/` do not provide
package façades.

Do not recreate the retired import namespace or a compatibility shim. Use the
lowercase brand `plasmax` in prose, paths, distribution metadata, infrastructure,
and new external identifiers. Python classes use conventional capitalization,
for example `PlasmaxEnv` and `PlasmaxTruncationWrapper`.

TORAX remains the upstream simulator name. Keep legitimate upstream identifiers
such as the `torax` dependency/imports, `ToraxConfig`, `_ToraxDynamics`,
`_torax_patches.py`, `torax:` YAML sections, and TORAX-owned environment
variables.

## Repository layout

```text
src/plasmax/        published environment library and packaged task data
agents/             clone-only baseline agents
training/           clone-only adapters and training wrappers
scripts/            generic PPO, SAC, evaluation, and rollout launchers
tools/              artifact and equilibrium generation
benchmarks/         backend agreement and throughput measurements
docs/               environment documentation
experiments/        studies, plotting, cluster infrastructure, and legacy code
tests/              installed-library and integration tests
```

Study-specific behavior belongs under `experiments/`, not in a generic launcher
or the installed package. Generated run outputs and plots stay in ignored local
directories. Sweeps and manifests are intentionally local and ignored.

## Core architecture

`plasmax` wraps TORAX and the packaged KSTAR learned model behind the Envelope
explicit-state lifecycle:

1. `ControlInputs` names actuator values.
2. Internal provider appliers map them to TORAX runtime-parameter overrides.
3. `PlasmaxEnv` owns reset/transition behavior and returns `EnvState`.
4. Small Envelope wrappers add the configured realistic sensor/action behavior.
5. `collect_episode` and `collect_episodes` provide generic fixed-shape rollout
   collection.

Environment state and wrapper state are JAX pytrees. Preserve JIT, `lax.scan`,
`lax.while_loop`, and `vmap` compatibility. Keep the differentiable scan stepper
and event-driven while-loop stepper behaviorally aligned, including their
custom-JVP contract.

Internal TORAX physics uses its existing precision. Rewards and clone-only RL
adapters cross a float32 boundary. Do not restore global Rejax monkey patches or
copied PPO/SAC dtype workarounds; use upstream algorithms through thin adapters.

Importing `plasmax` applies `plasmax._torax_patches`. This workaround prevents a
TORAX `Grid1D.cell_widths` cached tracer from leaking between compiled traces on
EQDSK/CHEASE geometry. Keep it until the upstream issue is resolved.

## Configuration

Packaged aliases resolve paths relative to `src/plasmax/configs/` so installed
artifacts work from any current directory.

- Single-file scenarios use `load_scenario`, including the fast `test` fixture.
- Machine tasks use `load_env` or `make` with an environment and compatible
  backend alias.
- Environment values override backend defaults through the one canonical merge
  path.
- Keep the full registered ITER, SPARC, STEP, and KSTAR matrix and every packaged
  data asset. A trajectory terminating is a control outcome, not a reason to
  remove a task.

Every leaf YAML stores task defaults:

```yaml
task:
  reward: lh_transition
  terminal_penalty: -99.99568287525884
```

Loaders default to `variant="realistic"`. Omitted reward and disruption-penalty
arguments inherit task metadata; explicit overrides, including zero, must be
preserved. KSTAR uses its native reward and null penalty.

Physics randomization remains transition-wise: each configured scalar is sampled
and applied on each transition. Do not change it to episode-only sampling. Keep
existing sensor-noise behavior. Do not add new noise validation, clipping,
reset-state positivity/quasineutrality gates, disruption gates, or stricter core
schema constraints.

The uv-only TGLFNN override remains in `pyproject.toml` until a compatible
release is available from the package index. Published metadata itself must not
contain a direct Git dependency. Keep the current TORAX and JAX pins.

## Tests

Tests use pytest and NumPy testing helpers. Prefer:

- `np.testing.assert_allclose` with explicit tolerances for floating results;
- `np.testing.assert_array_equal` for exact arrays and masks;
- `chex.assert_trees_all_close` for pytrees;
- `pytest.raises(..., match=...)` for error behavior.

The default command excludes tests marked `integration`. Do not change that
policy: CI runs the integration suite explicitly in seven shards. Keep geometry,
STEP, KSTAR, fixed-duration stepping, and TORAX-reference parity in the
appropriate integration tier.

Release checks build and install both wheel and source distribution outside the
checkout. They must prove that only `plasmax` is packaged, required task/model
assets are present, the public API matches, the retired namespace is absent, and
metadata contains no direct Git requirements.

Do not add physical-trajectory acceptance tests or noise-validation tests.
Clone-only PPO, SAC, and MPC smoke tests should exercise upstream algorithms and
thin adapters rather than optimizer internals.

## Experiments and tracking

New tracked runs use W&B entity `flair` and project `plasmax`. If online logging
is requested, never silently downgrade to offline. Stop if authorization or
initialization fails.

Generic launchers inherit reward and terminal penalty from task metadata. Pass
`--env.variant realistic` explicitly in recorded experiment commands even
though it is already the default.

Cluster-specific Docker, Slurm, and synchronization guidance lives with the
corresponding launchers under `experiments/cluster/`. Follow shared-cluster
safety rules before running or stopping remote jobs.

## Release hygiene

- Track `uv.lock` and use `uv sync --locked` in CI.
- The package version starts at `0.1.0` under the clean-break name.
- Build metadata and repository links use
  `https://github.com/TheodoreWolf/plasmax`.
- `jax-envelope==0.4.2` is the index-hosted Envelope dependency.
- Run `.github/scripts/check_dist.py` after `uv build`.
- Active tracked text must not contain retired branding. Immutable historical
  run identifiers belong only in ignored local manifests.
