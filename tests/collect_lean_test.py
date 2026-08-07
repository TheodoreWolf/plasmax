"""Regression check for the memory-lean eval trajectory (collect ``lean=True``).

The lean path nulls heavy per-step SimState/CoreProfiles fields eval never reads
so ``n_seeds`` can be large without OOM. This asserts it (a) drops the heavy
fields, (b) keeps the ones the eval figures plot, and (c) is actually smaller.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from envelope import WrappedState

from plasmax.environment.factory import load_scenario
from plasmax.rollout import (
    collect_episodes,
    trajectories_to_state_history,
)
from plasmax.wrappers import unwrap_to_env_state


def _nbytes(traj) -> int:
    return sum(leaf.nbytes for leaf in jax.tree_util.tree_leaves(traj))


class CollectLeanTest:
    @classmethod
    def setup_class(cls):
        cls.env = load_scenario("test")

    def _collect(self, lean, num_steps=4):
        return collect_episodes(
            lambda obs, rng: jnp.zeros((self.env.action_space.shape[0],)),
            self.env,
            jax.random.key(0),
            num_steps=num_steps,
            n_seeds=2,
            lean=lean,
        )

    def test_lean_is_smaller(self):
        assert _nbytes(self._collect(lean=True)) < _nbytes(self._collect(lean=False))

    def test_lean_preserves_wrapper_state_nesting(self):
        assert isinstance(self._collect(lean=True).env_state, WrappedState)

    def test_lean_keeps_plotted_fields_drops_heavy(self):
        es = unwrap_to_env_state(self._collect(lean=True).env_state)
        cp = es.plasma.core
        # Kept: what the eval figures/scalars read. (n_seeds, num_steps, ...)
        assert cp.T_e.value.shape[:2] == (2, 4)
        assert es.plasma.Q_fusion.shape == (2, 4)
        assert cp.q_face is not None
        # Dropped: heavy, eval-unused.
        assert es.plasma.geo is None
        assert cp.psi is None

    def test_full_keeps_everything(self):
        es = unwrap_to_env_state(self._collect(lean=False).env_state)
        assert es.plasma.geo is not None
        assert es.plasma.core.psi is not None

    def test_state_history_rejects_lean_trajectory(self):
        traj = self._collect(lean=True)
        with pytest.raises(ValueError, match="full trajectory|lean=False"):
            trajectories_to_state_history(
                traj,
                self.env.unwrapped.config,
                seed_idx=0,
            )

    def test_state_history_filters_padding_but_keeps_terminal_plasma_state(self):
        traj = self._collect(lean=False, num_steps=7)
        valid = np.asarray(traj.valid[0], dtype=bool)
        assert valid.any()
        assert not valid.all()

        history = trajectories_to_state_history(
            traj,
            self.env.unwrapped.config,
            seed_idx=0,
        )
        env_state = unwrap_to_env_state(traj.env_state)
        last_valid = np.flatnonzero(valid)[-1]

        assert history.times.shape == (int(valid.sum()),)
        np.testing.assert_allclose(
            history.times[-1],
            np.asarray(env_state.plasma.t[0, last_valid]),
            rtol=1e-7,
            atol=0.0,
        )
        assert bool(np.asarray(traj.done[0, last_valid]))
