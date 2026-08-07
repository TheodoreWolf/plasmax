"""Smoke tests for the env x backend x variant matrix.

Driven by the env-backend compatibility registry in
:mod:`plasmax.environment.config` so only valid pairs are exercised — adding
an env / backend YAML plus an entry in ``_VALID_ENV_BACKEND_COMBOS``
automatically expands both tiers below.

Two tiers:

- :class:`EnvCanaryTest` — one representative phase per ITER/SPARC scenario,
  plus STEP, using a cheap valid backend in oracle mode. Runs on every commit.
- :class:`EnvMatrixSmokeTest` — every registered ``(env, backend, variant)``
  combination. Marked ``integration``; excluded from the default pytest
  run. Trigger explicitly with ``pytest -m integration``.

Both tiers share :func:`_assert_step_contract`, the single source of truth
for "what a healthy env must do".
"""

import jax
import jax.numpy as jnp
import pytest
from envelope import AutoResetWrapper, Environment, Info, VmapWrapper

from plasmax.environment.config import (
    backend_kind,
    valid_env_backend_combos,
)
from plasmax.environment.factory import load_env
from plasmax.wrappers import (
    PlasmaxTruncationWrapper,
    iter_wrappers,
    unwrap_to_env_state,
)

_VARIANTS = ("oracle", "realistic")

# TORAX env x backend matrix only: world-model backends (e.g. fusion_lstm) use a
# different state contract (no plasma) and have their own smoke tests in
# tests/world_model_env_test.py.
_VALID_PAIRS: list[tuple[str, str]] = [
    (env, backend)
    for env, backends in sorted(valid_env_backend_combos().items())
    for backend in sorted(backends)
    if backend_kind(backend) == "torax"
]


def _pair_id(pair: tuple[str, str]) -> str:
    env, backend = pair
    return f"{env.replace('/', '_')}-{backend}"


# Cheapest valid backend per env, used as the per-commit canary.
_CANARY_BACKENDS_BY_ENV: dict[str, str] = {
    # One representative phase per ITER/SPARC scenario. The integration matrix
    # below covers every registered phase/backend/variant combination.
    "iter/baseline/flattop": "cgm",
    "iter/hybrid/flattop": "cgm",
    "iter/advanced/flattop": "cgm",
    "sparc/prd/flattop": "cgm",
    "sparc/reduced_field/flattop": "cgm",
    "step": "bohm_gyrobohm",
}

_CANARY_PAIRS: list[tuple[str, str]] = [
    pair for pair in sorted(_CANARY_BACKENDS_BY_ENV.items())
]


def _assert_step_contract(env, key=None, *, single_solver_call: bool = False):
    """Initializes and takes one bounded step, asserting contract invariants.

    No physics ranges or scenario-specific outcomes are asserted. The action
    is the environment's own declared initial setpoint, mapped into the
    wrapper's normalized ``[-1, 1]`` cube.

    The nonlinear TGLFNN reference backend can adaptively split one public
    control transition into as many as 129 expensive TORAX solves. Matrix
    coverage is a smoke test, not that GPU-oriented acceptance study, so
    ``single_solver_call`` applies a test-only one-call budget while retaining
    the real reset and transition paths. Packaged task configuration is not
    modified.
    """
    if key is None:
        key = jax.random.key(0)

    state, init_info = env.init(key)
    assert isinstance(env, Environment)
    assert isinstance(env, PlasmaxTruncationWrapper)
    assert not any(
        isinstance(layer, (AutoResetWrapper, VmapWrapper))
        for layer in iter_wrappers(env)
    )
    assert isinstance(init_info, Info)
    assert init_info.obs.shape == env.observation_space.shape
    assert jnp.all(jnp.isfinite(init_info.obs))
    assert not bool(init_info.terminated)
    assert not bool(init_info.truncated)
    assert int(init_info.termination_code) == -1

    action = env.from_physical(state.unwrapped.prev_action)

    if single_solver_call:
        dynamics = env.unwrapped._dynamics
        dynamics._stepping = dynamics._stepping.model_copy(
            update={"max_solver_substeps": 1, "max_event_substeps": 0}
        )

    state2, info = env.step(state, action)
    assert isinstance(info, Info)
    assert info.obs.shape == env.observation_space.shape
    assert info.reward.shape == ()
    assert info.terminated.shape == ()
    assert info.truncated.shape == ()
    assert int(info.internal_steps) >= 1
    if single_solver_call:
        assert int(info.internal_steps) == 1

    termination_code = int(info.termination_code)
    assert termination_code in {-1, 0, 1, 2, 3}
    assert bool(info.terminated) == (termination_code != -1)
    assert float(unwrap_to_env_state(state2).plasma.t) >= float(
        unwrap_to_env_state(state).plasma.t
    )


class EnvCanaryTest:
    """Per-scenario canary against a cheap valid backend. Runs every commit."""

    @pytest.mark.parametrize(
        "env_yaml,backend", _CANARY_PAIRS, ids=[_pair_id(p) for p in _CANARY_PAIRS]
    )
    def test_load_env_canary(self, env_yaml, backend):
        env = load_env(env_yaml, backend, variant="oracle")
        _assert_step_contract(env)


@pytest.mark.integration
class EnvMatrixSmokeTest:
    """Every registered (env, backend, variant). Opt-in (~10-15 min cold)."""

    @pytest.mark.parametrize("variant", _VARIANTS)
    @pytest.mark.parametrize(
        "env_yaml,backend", _VALID_PAIRS, ids=[_pair_id(p) for p in _VALID_PAIRS]
    )
    def test_smoke(self, env_yaml, backend, variant):
        env = load_env(env_yaml, backend, variant=variant)
        _assert_step_contract(env, single_solver_call=backend == "tglfnn_nr")
