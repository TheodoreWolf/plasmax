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
uv run pytest -m integration tests/
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
make, PlasmaxConfig, PlasmaxEnv, EnvState,
TrajectoryStep, collect_episode, collect_episodes, registry
```

Keep agents, Gymnax/Rejax adaptation, programmatic training, W&B, plotting,
and studies outside `src/`. Concrete clone-only modules may be imported from a
checkout, but `agents/` and `training/` do not provide package façades.

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
experiments/        research studies and plotting
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
4. Small Envelope wrappers add physics randomization and configured sensor/action behavior.
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

- `make` is the only public constructor. TORAX environments use
  `{tokamak}/{scenario}/{phase}` and require an explicit compatible backend;
  `kstar_worldmodel` is standalone and takes no backend. Raw YAML paths are not
  constructor inputs.
- TORAX environment fragments compose as
  `tokamak < scenario base < phase < wrappers`. Backend fragments are orthogonal:
  they own transport, solver, solver-substep budget, and transport-model
  randomization leaves, and duplicate environment/backend leaves are errors.
- YAML fragments are not independently schema-validated. The loader resolves
  assets, creates the upstream `ToraxConfig`, and validates the one complete
  `PlasmaxConfig`. KSTAR instead validates one complete `WorldModelConfig`.
- `make` validates the registered pair and options, loads the complete config,
  and returns the matching bare core environment. `RealisticWrappers` and
  `OracleWrappers` compose training behavior explicitly, check the requested
  horizon, and add final truncation.
- Keep the full registered ITER, SPARC, STEP, KSTAR, and mock matrix and every
  packaged data asset. A trajectory terminating is a control outcome, not a
  reason to remove a task.

Every leaf YAML stores task defaults:

```yaml
task:
  reward: lh_transition
  terminal_penalty: -100
```

`make` has no variant or wrapper options. Wrappers resolve defaults from the
already parsed `env.plasmax_config`; explicit wrapper arguments remain supported.
Omitted reward and disruption-penalty arguments inherit task metadata; explicit
overrides, including zero, must be preserved. KSTAR uses its native reward and
null penalty.

`PhysicsRandomizationWrapper` owns its RNG and samples each configured scalar
before every transition. The core consumes persistent `EnvState.phys_params`;
`with_physics` immutably updates selected entries through nested wrapper state,
and reset restores nominal values. Relative sampling uses configured nominals.
Do not change it to episode-only sampling. Keep existing sensor-noise behavior.
Do not add new noise validation, clipping,
reset-state positivity/quasineutrality gates, disruption gates, or stricter core
schema constraints.

The uv-only TGLFNN override remains in `pyproject.toml` until TORAX adopts
`fusion-surrogates` 0.4.7 and its compatible TGLFNN 0.2.0 dependency. Keep the
override index-hosted; published metadata cannot express uv overrides.

Runtime metadata uses tested lower bounds while `uv.lock` keeps clone and CI
installs exact. Do not add a TORAX upper bound without a demonstrated
incompatibility. Envelope is pre-1.0, so keep it within a tested compatible
minor series. Depend on JAX rather than duplicating its matching `jaxlib` pin.

## Tests

Tests use pytest and NumPy testing helpers. Prefer:

- `np.testing.assert_allclose` with explicit tolerances for floating results;
- `np.testing.assert_array_equal` for exact arrays and masks;
- `chex.assert_trees_all_close` for pytrees;
- `pytest.raises(..., match=...)` for error behavior.

The default `uv run pytest` command includes tests marked `integration`. CI
partitions execution explicitly: the fast job selects `not integration`, and
one integration job runs the slow specialised checks, including
external-reference parity and one representative `tglfnn_nr` canary. Keep
focused geometry, STEP, KSTAR, and fixed-duration tests in the fast suite.

The dedicated environment trajectory workflow owns the seven-way
environment/backend/variant matrix. Do not duplicate that full matrix in the
pytest integration suite.

`tests/env_trajectory_test.py` is the fast test suite for
`benchmarks/env_trajectory.py`; it must never execute the real 126-case
trajectory matrix. Use synthetic reports and mocked environments there to test
case selection, incremental output, failures, JSON validation, comparisons,
Markdown summaries, baselines, and seven-shard merging. Run real trajectories
only through the benchmark command or its dedicated GitHub Actions workflow.

Release checks build a wheel and a source distribution and run `twine check` on
them. Nothing installs or smoke-tests the built artifacts.

Do not add physical-trajectory acceptance tests or noise-validation tests.
Clone-only PPO, SAC, and MPC smoke tests should exercise upstream algorithms and
thin adapters rather than optimizer internals.

## Experiments and tracking

Run control algorithms through their dedicated repository scripts, such as
`scripts/train_ppo.py` and `scripts/train_sac.py`; do not substitute ad hoc
training runners. All transport backends except TGLFNN may be vmapped across
training seeds. Run TGLFNN seeds as independent single-seed processes or jobs.

Control-algorithm runs use online W&B logging unless the user explicitly requests otherwise. Never
silently downgrade to offline or disable logging; stop if authorization or
initialization fails.

Generic launchers inherit reward and terminal penalty from task metadata. Pass
`--env.variant realistic` explicitly in recorded experiment commands even
though it is already the CLI default. These research labels select explicit
composition helpers in the launchers; they are not arguments to `make`.

## Release hygiene

- Track `uv.lock` and use `uv sync --locked` in CI.
- The package version starts at `0.1.0` under the clean-break name.
- Build metadata and repository links use
  `https://github.com/TheodoreWolf/plasmax`.
- `jax-envelope~=0.6.1` is the index-hosted Envelope dependency.
- Active tracked text must not contain retired branding. Immutable historical
  run identifiers belong only in ignored local manifests.
