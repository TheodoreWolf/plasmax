# Testing plasmax

The test suite protects the installed environment contract, configuration
metadata, JAX behavior, and the repository-only baseline adapters. It does not
require a nominal controller to keep every physical environment alive: a
disruption or solver termination is a valid control outcome.

## Commands

```bash
# Full library suite, including integration tests.
uv run pytest

# Unit and fast regression tests only.
uv run pytest -m "not integration" tests/

# Clone-only publication, plotting, transfer, and study tests.
uv run pytest experiments/tests/

# Slow specialised integration checks, including the TGLFNN canary.
uv run pytest -m integration tests/

# Full environment trajectory benchmark (126 cases).
uv run python benchmarks/env_trajectory.py run

# One of the seven CI trajectory groups.
uv run python benchmarks/env_trajectory.py run --group iter-hybrid

# Lint all installed and clone-only code.
uv run ruff check .
```

Tests use pytest. Test classes use `Test*` or `*Test` names, such as
`PlasmaxEnvTest`, with pytest fixtures or `setup_method`/`setup_class`.

## Test tiers

The fast suite contains behavior, schema, reward, wrapper, control, collection,
and packaging regressions. Prefer the packaged `mock/circular/smoke` environment
with the `mock` backend for these tests.

The default pytest configuration includes tests marked `integration`. Focused
geometry, STEP, KSTAR, and fixed-duration contracts remain in the fast suite.
Slow specialised checks, including external-reference parity and one
representative `tglfnn_nr` canary, use `@pytest.mark.integration`. CI runs them
in one integration job, separate from the fast suite.

The dedicated environment trajectory workflow owns the complete
environment/backend/variant matrix. It divides that benchmark into seven
parallel groups; pytest does not repeat this matrix.

Publication, plotting, transfer, and study-matrix tests live beside their code
under `experiments/tests/`; they are repository tests, not package contents.

## Environment trajectory benchmark

The trajectory benchmark is a user-facing diagnostic rather than part of the
pytest suite. Run the full 126-case matrix with:

```bash
uv run python benchmarks/env_trajectory.py run
```

It prints progress while each case runs, followed by a complete timing table,
and writes the machine-readable result to
`outputs/env_trajectory_results.json`. The result is updated after every case,
so partial information remains available if a later case fails.

For a quicker local check, select one or more environments, backends, or
variants. For example:

```bash
uv run python benchmarks/env_trajectory.py run \
  --environments iter/hybrid/flattop \
  --backends cgm \
  --variants realistic
```

You can also run one of the seven CI-sized groups, for example:

```bash
uv run python benchmarks/env_trajectory.py run --group iter-hybrid
```

The two timings have deliberately narrow meanings:

- `creation_seconds` measures `plasmax.make` plus the selected wrapper composition;
- `first_trajectory_seconds` measures initialization, first JAX compilation,
  and the zero-action trajectory to its first boundary.

These are useful for inspecting changes on the same machine, but timings from
different machines are only advisory. The benchmark excludes the expensive
`tglfnn_nr` backend and the standalone KSTAR world model.

GitHub Actions runs the same matrix in seven parallel groups. Successful runs
on `main` store a JSON baseline as a workflow artifact. Pull requests run the
benchmark once, compare it with that stored baseline, and show a compact report
in the workflow summary; the complete JSON remains downloadable. Behavior and
large timing differences are warnings for review. Broken, missing, incomplete,
or invalid results make the report check fail.

## Behavioral assertions

Test caller-visible behavior rather than private implementation details. A test
name should state the contract it proves, and a passing test must exercise that
contract without a conditional assertion path.

Use NumPy's testing helpers for arrays:

- `np.testing.assert_allclose` for floating-point behavior, always with explicit
  `atol` and `rtol`;
- `np.testing.assert_array_equal` for exact arrays, masks, integer values, and
  reproducibility checks;
- `chex.assert_trees_all_close` for complete JAX pytrees;
- ordinary `assert` for Python scalars and structural relationships;
- `pytest.raises(..., match=...)` for errors, including a stable message fragment.

Shape, dtype, finiteness, or “did not raise” is sufficient only when that is the
named contract. Otherwise assert an expected value, algebraic relationship, or
directional response.

## JAX contracts

Use typed `jax.random.key(...)` values at the public Envelope boundary. Seed
tests explicitly and compare same-key executions for deterministic behavior.
Important reusable properties include:

- pytree flatten/unflatten round trips for `EnvState`, `ControlInputs`, and
  `TrajectoryStep`;
- eager/JIT equality for environment and reward calls;
- vmapped results matching stacked scalar calls;
- `collect_episode` retaining the first terminal transition and masking padding;
- the public reward and adapter boundary remaining float32 while internal TORAX
  precision remains unchanged.

Avoid compiling the same expensive call twice unless a test specifically proves
JIT/eager parity.

## Configuration and task metadata

Every leaf task YAML must specify `task.reward` and `task.terminal_penalty`.
Tests cover:

- the phase-appropriate reward for every leaf task;
- all ten calibrated ITER/SPARC ramp penalties exactly;
- explicit `0.0` penalties for flat-top, STEP, and the mock smoke fixture;
- KSTAR's native reward and null terminal penalty;
- loader inheritance when reward and penalty are omitted;
- string and callable reward overrides;
- explicit zero overriding nonzero task metadata;
- bare construction plus explicit realistic/oracle composition;
- wrapper default resolution and persistent physics updates/reset behavior.

Golden merged-config comparisons must show that profile and TGLF YAML
deduplication changes no physics values. The allowed differences are task
metadata and renamed package paths.

## Environment and wrapper regressions

Keep direct tests for controls, rewards, loaders, wrappers, collection, STEP,
KSTAR, EQDSK geometry, fixed-duration stepping, and TORAX reference parity.
Wrapper regression tests specifically protect:

- action history initialized from configured actuator setpoints;
- elapsed time measured from episode start;
- action rescaling at `-1`, `0`, and `1` and its inverse round trip;
- termination taking precedence over truncation at a coincident boundary.

Do not add new sensor-noise validation tests, statistical noise acceptance
tests, reset positivity/quasineutrality gates, or physical-trajectory acceptance
tests. Per-transition physics randomization remains part of the existing
contract and should retain its current sampling tests.

## Distribution tests

There are none. CI builds a wheel and a source distribution and runs `twine
check` on them; nothing installs or exercises the built artifacts.

```bash
uv build
uvx --from "twine==6.2.0" twine check dist/*
```

## Clone-only algorithms

PPO, SAC, and MPC smoke tests exercise upstream algorithms through thin
repository adapters. They should use tiny horizons and fixed seeds, and assert
finite outputs and adapter behavior—not copied optimizer internals or long
training quality. Keep slow research studies and W&B integration out of the
installed-package tests.

## Avoid testing

- Private helpers solely to mirror implementation structure.
- TORAX, JAX, Envelope, or Rejax behavior that plasmax does not wrap or depend on.
- Plot pixels; test the data passed to a renderer.
- A controller's ability to prevent disruption as a prerequisite for testing an
  environment transition.
