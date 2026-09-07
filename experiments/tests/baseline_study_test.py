"""Behavioral contracts for the phase-wise baseline experiment matrix."""

import functools

import numpy as np
import pytest

from experiments.studies.baseline_study import (
    BASELINE_ALGORITHMS,
    baseline_envs,
    make_baseline_jobs,
    reward_for_env,
    seed_keys,
    validate_reward,
)
from plasmax import rewards as rewards_lib
from plasmax.environment.factory import make
from plasmax.wrappers import RealisticWrappers


def test_matrix_excludes_kstar_and_contains_all_phase_envs():
    envs = baseline_envs()

    assert "kstar_worldmodel" not in envs
    assert "mock/circular/smoke" not in envs
    assert "step/spp_001_ec_hd/flattop" not in envs
    assert len(envs) == 15


@pytest.mark.parametrize(
    ("env", "backend", "reward"),
    [
        ("iter/hybrid/rampup", "bohm_gyrobohm", "lh_transition"),
        ("sparc/prd/flattop", "bohm_gyrobohm", "P_diff"),
        ("iter/advanced/rampdown", "bohm_gyrobohm", "rampdown"),
        (
            "step/spp_001_ec_hd/flattop",
            "bohm_gyrobohm_step",
            "P_diff",
        ),
    ],
)
def test_phase_reward_contract(env, backend, reward):
    assert reward_for_env(env, backend) == reward
    validate_reward(env, reward, backend)


def test_wrong_phase_reward_fails_before_training():
    with pytest.raises(ValueError, match="baseline reward mismatch"):
        validate_reward("iter/hybrid/rampup", "P_diff")


def test_complete_matrix_has_every_algorithm_variant_and_env():
    jobs = make_baseline_jobs()

    assert len(jobs) == 15 * 2 * len(BASELINE_ALGORITHMS)
    assert len({job.slug for job in jobs}) == len(jobs)


def test_knot_budget_and_evaluation_frequency_are_configurable():
    jobs = make_baseline_jobs(
        backend="bohm_gyrobohm_step",
        envs=("step/spp_001_ec_hd/flattop",),
        algorithms=("direct_knots_10",),
        variants=("oracle",),
        knot_steps=10_000_000,
        knot_eval_freq=1_000_000,
    )

    assert len(jobs) == 1
    assert jobs[0].total_steps == 10_000_000
    assert jobs[0].eval_freq == 1_000_000


def test_seed_keys_do_not_depend_on_batch_size():
    first_five = seed_keys(7, 5)
    first_ten = seed_keys(7, 10)[:5]

    np.testing.assert_array_equal(first_five, first_ten)


def test_lh_transition_binds_the_environment_rampup_duration():
    env = RealisticWrappers(
        make("sparc/prd/rampup", "bohm_gyrobohm", reward="lh_transition")
    )
    reward_fn = env.unwrapped._dynamics._reward_fn

    assert isinstance(reward_fn, functools.partial)
    assert reward_fn.func is rewards_lib.lh_transition
    assert reward_fn.keywords == {"t_final": 10.0}
