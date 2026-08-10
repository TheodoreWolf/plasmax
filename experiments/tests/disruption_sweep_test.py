import math

import numpy as np

from experiments.studies.disruption_sweep import (
    DEFAULT_KAPPA,
    DEFAULT_KAPPAS,
    calibrated_disruption_penalty,
    make_disruption_sweep_jobs,
    phase_envs,
)


class DisruptionPenaltyTest:
    def test_default_kappa_returns_task_terminal_penalty(self):
        np.testing.assert_allclose(DEFAULT_KAPPA, 1.0, atol=0.0, rtol=0.0)
        np.testing.assert_allclose(
            calibrated_disruption_penalty(
                "sparc/prd/rampup",
                "bohm_gyrobohm",
            ),
            -10,
            atol=0.0,
            rtol=0.0,
        )

    def test_penalty_scales_linearly_with_kappa(self):
        penalties = [
            calibrated_disruption_penalty(
                "iter/hybrid/rampdown",
                "bohm_gyrobohm",
                kappa=kappa,
            )
            for kappa in DEFAULT_KAPPAS
        ]
        np.testing.assert_allclose(
            penalties,
            np.asarray(penalties[0]) * np.asarray([1.0, 2.0, 4.0]),
            atol=1e-12,
            rtol=1e-12,
        )


class DisruptionSweepJobsTest:
    def test_complete_realistic_phase_by_kappa_matrix(self):
        jobs = make_disruption_sweep_jobs()

        assert len(jobs) == len(phase_envs()) * len(DEFAULT_KAPPAS) == 30
        assert {job.env for job in jobs} == set(phase_envs())
        assert {job.kappa for job in jobs} == set(DEFAULT_KAPPAS)
        assert all(job.variant == "realistic" for job in jobs)
        assert all(job.penalty < 0.0 and math.isfinite(job.penalty) for job in jobs)

    def test_known_task_penalties_and_phase_rewards(self):
        jobs = make_disruption_sweep_jobs(kappas=(1.0,))
        by_env = {job.env: job for job in jobs}

        assert by_env["iter/baseline/rampup"].reward == "lh_transition"
        np.testing.assert_allclose(
            by_env["iter/baseline/rampup"].task_terminal_penalty,
            -100,
            atol=0.0,
            rtol=0.0,
        )
        assert by_env["sparc/prd/rampdown"].reward == "rampdown"
        np.testing.assert_allclose(
            by_env["sparc/prd/rampdown"].penalty,
            -380,
            atol=0.0,
            rtol=0.0,
        )
