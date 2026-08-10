# Testing plasmax

The test suite protects the installed environment contract, configuration
metadata, JAX behavior, and the repository-only baseline adapters. It does not
require a nominal controller to keep every physical environment alive: a
disruption or solver termination is a valid control outcome.

## Commands

```bash
# Unit and fast regression tests; integration tests are excluded by pyproject.
uv run pytest

# Clone-only publication, plotting, transfer, and study tests.
uv run pytest experiments/tests/

# The full integration suite.
uv run pytest -o addopts="" -m integration tests/

# One of the seven CI shards.
uv run pytest -o addopts="" -m integration -k iter_hybrid tests/

# Lint all installed and clone-only code.
uv run ruff check .
```

Tests use pytest. Test classes use `Test*` or `*Test` names, such as
`PlasmaxEnvTest`, with pytest fixtures or `setup_method`/`setup_class`.

## Test tiers

The default suite contains fast behavior, schema, reward, wrapper, control,
collection, and packaging regressions. Prefer the packaged `test` scenario for
these tests.

Integration tests use `@pytest.mark.integration` for the full environment,
backend, geometry, STEP, KSTAR, fixed-duration, and TORAX-reference checks. The
default pytest configuration intentionally excludes them. CI opts in explicitly
and preserves seven shards:

1. ITER baseline
2. ITER hybrid
3. ITER advanced
4. SPARC PRD
5. SPARC reduced field
6. STEP
7. remaining integration tests

Publication, plotting, transfer, and study-matrix tests live beside their code
under `experiments/tests/`; they are repository tests, not package contents.

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
- explicit `0.0` penalties for flat-top, STEP, and the test fixture;
- KSTAR's native reward and null terminal penalty;
- loader inheritance when reward and penalty are omitted;
- string and callable reward overrides;
- explicit zero overriding nonzero task metadata;
- the realistic default on `make`.

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

Release checks build both a wheel and source distribution, install each outside
the checkout, and prove that:

- `import plasmax` works and the retired namespace is absent;
- both artifacts expose the same small `plasmax.__all__`;
- only `plasmax` is installed—agents, training, scripts, experiments, tests,
  tools, and benchmarks do not leak into an archive;
- all required YAML, EQDSK, STEP, and KSTAR assets are present;
- `plasmax.make("test")` uses the realistic default;
- spaces use property access such as `env.action_space.shape`;
- published metadata contains no direct Git requirements.

Run the local archive checks with:

```bash
uv build
uv run --no-project python .github/scripts/check_dist.py dist
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
