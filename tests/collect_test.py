"""Envelope-native trajectory collection contract tests."""

import inspect
from functools import cached_property

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from envelope import (
    AutoResetWrapper,
    Continuous,
    Environment,
    FrozenPyTreeNode,
    InfoContainer,
    static_field,
)

from plasmax.rollout import (
    TrajectoryStep,
    collect_episode,
    collect_episodes,
)
from plasmax.wrappers import (
    ActionRescaleWrapper,
    NoiseWrapper,
    ObsDelayWrapper,
    PlasmaxTruncationWrapper,
    QuantizeActionWrapper,
)


class _CounterState(FrozenPyTreeNode):
    value: jax.Array
    steps: jax.Array


class _BoundaryEnv(Environment):
    """Small deterministic-step env with seed-dependent initial observations."""

    terminate_after: int | None = static_field(default=None)
    truncate_after: int | None = static_field(default=None)

    @cached_property
    def observation_space(self):
        return Continuous(
            low=jnp.asarray([-jnp.inf], jnp.float32),
            high=jnp.asarray([jnp.inf], jnp.float32),
        )

    @cached_property
    def action_space(self):
        return Continuous(
            low=jnp.asarray([-1.0], jnp.float32),
            high=jnp.asarray([1.0], jnp.float32),
        )

    @staticmethod
    def _info(state, *, reward, terminated, truncated, termination_code):
        return InfoContainer(
            obs=state.value[None],
            reward=jnp.asarray(reward, jnp.float32),
            terminated=jnp.asarray(terminated),
            truncated=jnp.asarray(truncated),
        ).update(termination_code=jnp.asarray(termination_code, jnp.int32))

    def init(self, key):
        state = _CounterState(
            value=jax.random.uniform(key, (), dtype=jnp.float32),
            steps=jnp.asarray(0, jnp.int32),
        )
        return state, self._info(
            state,
            reward=0.0,
            terminated=False,
            truncated=False,
            termination_code=-1,
        )

    def reset(self, state, key):
        del state
        return self.init(key)

    def step(self, state, action):
        next_state = state.replace(
            value=state.value + action[0] + 1.0,
            steps=state.steps + 1,
        )
        terminated = (
            jnp.asarray(False)
            if self.terminate_after is None
            else next_state.steps >= self.terminate_after
        )
        truncated = (
            jnp.asarray(False)
            if self.truncate_after is None
            else next_state.steps >= self.truncate_after
        )
        code = jnp.where(terminated, 1, -1)
        return next_state, self._info(
            next_state,
            reward=next_state.value,
            terminated=terminated,
            truncated=truncated,
            termination_code=code,
        )


class _PhysicalBoundaryEnv(_BoundaryEnv):
    """Boundary env whose inner action coordinates are physical units."""

    @cached_property
    def action_space(self):
        return Continuous(
            low=jnp.asarray([10.0], jnp.float32),
            high=jnp.asarray([20.0], jnp.float32),
        )


class _SeedBoundaryState(FrozenPyTreeNode):
    steps: jax.Array
    limit: jax.Array


class _SeedBoundaryEnv(Environment):
    """Each init key deterministically selects a one-to-three-step episode."""

    @cached_property
    def observation_space(self):
        return Continuous(
            low=jnp.asarray([0.0], jnp.float32),
            high=jnp.asarray([jnp.inf], jnp.float32),
        )

    @cached_property
    def action_space(self):
        return Continuous(
            low=jnp.asarray([-1.0], jnp.float32),
            high=jnp.asarray([1.0], jnp.float32),
        )

    @staticmethod
    def _info(state, terminated=False):
        return InfoContainer(
            obs=state.steps[None].astype(jnp.float32),
            reward=state.steps.astype(jnp.float32),
            terminated=jnp.asarray(terminated),
            truncated=jnp.asarray(False),
        ).update(termination_code=jnp.where(terminated, jnp.int32(1), jnp.int32(-1)))

    def init(self, key):
        limit = 1 + jax.random.key_data(key)[0] % jnp.uint32(3)
        state = _SeedBoundaryState(
            steps=jnp.asarray(0, jnp.int32),
            limit=limit.astype(jnp.int32),
        )
        return state, self._info(state)

    def reset(self, state, key):
        del state
        return self.init(key)

    def step(self, state, action):
        del action
        next_state = state.replace(steps=state.steps + 1)
        return next_state, self._info(
            next_state, terminated=next_state.steps >= next_state.limit
        )


def _zero_policy(obs, key):
    del obs, key
    return jnp.zeros((1,), jnp.float32)


def test_collect_episode_drops_env_params_from_signature():
    assert "env_params" not in inspect.signature(collect_episode).parameters
    assert "env_params" not in inspect.signature(collect_episodes).parameters


@pytest.mark.parametrize(
    ("env", "expected_terminated", "expected_truncated"),
    [
        (
            _BoundaryEnv(terminate_after=2),
            [False, True, False, False, False],
            [False] * 5,
        ),
        (
            _BoundaryEnv(truncate_after=2),
            [False] * 5,
            [False, True, False, False, False],
        ),
    ],
)
def test_collect_episode_retains_boundary_then_freezes_and_masks_padding(
    env, expected_terminated, expected_truncated
):
    traj = collect_episode(
        _zero_policy,
        env,
        jax.random.key(0),
        num_steps=5,
    )

    np.testing.assert_array_equal(traj.valid, [True, True, False, False, False])
    np.testing.assert_array_equal(traj.terminated, expected_terminated)
    np.testing.assert_array_equal(traj.truncated, expected_truncated)
    np.testing.assert_array_equal(traj.done, [False, True, False, False, False])

    # The boundary transition is emitted. Later scan slots neither advance nor
    # replace the post-boundary state.
    np.testing.assert_array_equal(traj.env_state.steps, [1, 2, 2, 2, 2])
    np.testing.assert_allclose(
        traj.next_obs[1], traj.env_state.value[1, None], rtol=1e-7, atol=0.0
    )
    np.testing.assert_array_equal(traj.info.termination_code[2:], [-1, -1, -1])


def test_done_is_computed_from_flags_and_validity_not_stored():
    traj = collect_episode(
        _zero_policy,
        _BoundaryEnv(terminate_after=1),
        jax.random.key(0),
        num_steps=3,
    )
    assert "done" not in TrajectoryStep.__annotations__
    np.testing.assert_array_equal(traj.done, [True, False, False])
    np.testing.assert_array_equal(traj.valid, [True, False, False])


def test_collect_episodes_vmaps_independent_typed_keys():
    traj = collect_episodes(
        _zero_policy,
        _BoundaryEnv(),
        jax.random.key(7),
        num_steps=1,
        n_seeds=3,
    )
    assert traj.obs.shape == (3, 1, 1)
    np.testing.assert_array_equal(traj.valid, np.ones((3, 1), dtype=bool))
    assert np.unique(np.asarray(traj.obs[:, 0, 0])).size == 3


def test_collect_episodes_accepts_array_valued_wrapper_configuration():
    env = PlasmaxTruncationWrapper(
        env=ObsDelayWrapper(
            env=NoiseWrapper(
                env=_BoundaryEnv(),
                noise_scale=jnp.asarray([0.1], dtype=jnp.float32),
            ),
            hold_prob=jnp.asarray([0.2], dtype=jnp.float32),
        ),
        max_steps=2,
    )

    traj = collect_episodes(
        _zero_policy,
        env,
        jax.random.key(13),
        num_steps=2,
        n_seeds=3,
    )

    assert traj.obs.shape == (3, 2, 1)
    np.testing.assert_array_equal(traj.valid, np.ones((3, 2), dtype=bool))
    np.testing.assert_array_equal(traj.truncated[:, -1], np.ones(3, dtype=bool))


@pytest.mark.parametrize(
    ("env", "policy", "expected"),
    [
        (
            ActionRescaleWrapper(env=_PhysicalBoundaryEnv()),
            lambda obs, key: jnp.asarray([0.5], jnp.float32),
            17.5,
        ),
        (
            QuantizeActionWrapper(
                env=ActionRescaleWrapper(env=_PhysicalBoundaryEnv()),
                bin_counts=(3,),
            ),
            lambda obs, key: jnp.asarray([2], jnp.int32),
            20.0,
        ),
    ],
    ids=("continuous", "quantized"),
)
def test_collect_episode_records_physical_actions(env, policy, expected):
    traj = collect_episode(policy, env, jax.random.key(0), num_steps=1)
    np.testing.assert_allclose(traj.action, [[expected]], rtol=0.0, atol=1e-6)


def test_collect_episodes_selective_endings_freeze_each_lane_without_leakage():
    traj = collect_episodes(
        _zero_policy,
        _SeedBoundaryEnv(),
        jax.random.key(11),
        num_steps=5,
        n_seeds=6,
    )
    limits = np.asarray(traj.env_state.limit[:, 0])
    assert np.unique(limits).size > 1

    for lane, limit in enumerate(limits):
        valid = np.asarray(traj.valid[lane])
        steps = np.asarray(traj.env_state.steps[lane])
        terminated = np.asarray(traj.terminated[lane])
        np.testing.assert_array_equal(valid, np.arange(5) < limit)
        np.testing.assert_array_equal(steps, np.minimum(np.arange(1, 6), limit))
        np.testing.assert_array_equal(terminated, np.arange(5) == int(limit) - 1)


def test_collect_rejects_autoresetting_environment():
    env = AutoResetWrapper(env=_BoundaryEnv(terminate_after=1))
    with pytest.raises(ValueError, match="(?i)autoreset"):
        collect_episode(
            _zero_policy,
            env,
            jax.random.key(0),
            num_steps=2,
        )
