"""Focused smoke tests for the environment step contract.

The fast canaries cover one representative phase per ITER/SPARC scenario,
plus STEP and the mock environment. A separate integration canary exercises
the expensive TGLFNN backend. The complete environment matrix is covered by
``benchmarks/env_trajectory.py`` and its dedicated CI workflow.
"""

import jax
import jax.numpy as jnp
import pytest
from envelope import AutoResetWrapper, Environment, Info, VmapWrapper

from plasmax.environment.factory import make
from plasmax.wrappers import (
    OracleWrappers,
    PlasmaxTruncationWrapper,
    iter_wrappers,
    unwrap_to_env_state,
)


def _pair_id(pair: tuple[str, str]) -> str:
    env, backend = pair
    return f"{env.replace('/', '_')}-{backend}"


# Cheapest valid backend per env, used as the per-commit canary.
_CANARY_BACKENDS_BY_ENV: dict[str, str] = {
    # One representative phase per ITER/SPARC scenario. The trajectory
    # benchmark covers every supported phase/backend/variant combination.
    "iter/baseline/flattop": "cgm",
    "iter/hybrid/flattop": "cgm",
    "iter/advanced/flattop": "cgm",
    "sparc/prd/flattop": "cgm",
    "sparc/reduced_field/flattop": "cgm",
    "step/spp_001_ec_hd/flattop": "bohm_gyrobohm_step",
    "mock/circular/smoke": "mock",
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
    control transition into as many as 129 expensive TORAX solves. This canary
    only needs to check the real reset and transition paths, so
    ``single_solver_call`` applies a test-only one-call budget. Packaged task
    configuration is not modified.
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
        env = OracleWrappers(make(env_yaml, backend))
        _assert_step_contract(env)


@pytest.mark.integration
def test_tglfnn_nr_canary():
    """Exercise one real TGLFNN transition with a bounded solver budget."""
    env = OracleWrappers(make("iter/hybrid/flattop", "tglfnn_nr"))
    _assert_step_contract(env, single_solver_call=True)
