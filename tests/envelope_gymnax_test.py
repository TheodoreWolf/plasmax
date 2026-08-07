"""Contracts for the local Envelope-to-Gymnax compatibility adapter."""

from functools import cached_property

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from envelope import (
    BatchedSpace,
    Continuous,
    Discrete,
    Environment,
    FrozenPyTreeNode,
    InfoContainer,
    PyTreeSpace,
    TruncationWrapper,
    static_field,
)
from gymnax.environments import spaces as gymnax_spaces

from training.envelope_gymnax import (
    EnvelopeGymnax,
    GymnaxMultiDiscrete,
    _convert_space,
    to_typed_key,
)


class _State(FrozenPyTreeNode):
    steps: jax.Array
    episodes: jax.Array


class _AdapterEnv(Environment):
    terminate_after: int | None = static_field(default=None)
    truncate_after: int | None = static_field(default=None)
    action_n: int | tuple[int, ...] | None = static_field(default=None)

    @cached_property
    def observation_space(self):
        return Continuous(
            low=jnp.asarray([-100.0], jnp.float32),
            high=jnp.asarray([100.0], jnp.float32),
        )

    @cached_property
    def action_space(self):
        if self.action_n is None:
            return Continuous(
                low=jnp.asarray([-1.0], jnp.float32),
                high=jnp.asarray([1.0], jnp.float32),
            )
        return Discrete(n=jnp.asarray(self.action_n, jnp.int32))

    @staticmethod
    def _info(state, *, terminated=False, truncated=False):
        obs = (state.episodes * 10 + state.steps).astype(jnp.float32)[None]
        return InfoContainer(
            obs=obs,
            reward=state.steps.astype(jnp.float32),
            terminated=jnp.asarray(terminated),
            truncated=jnp.asarray(truncated),
        ).update(episode=state.episodes)

    def init(self, key):
        del key
        state = _State(
            steps=jnp.asarray(0, jnp.int32),
            episodes=jnp.asarray(0, jnp.int32),
        )
        return state, self._info(state)

    def reset(self, state, key):
        del key
        state = state.replace(
            steps=jnp.zeros_like(state.steps), episodes=state.episodes + 1
        )
        return state, self._info(state)

    def step(self, state, action):
        del action
        state = state.replace(steps=state.steps + 1)
        terminated = (
            jnp.asarray(False)
            if self.terminate_after is None
            else state.steps >= self.terminate_after
        )
        truncated = (
            jnp.asarray(False)
            if self.truncate_after is None
            else state.steps >= self.truncate_after
        )
        return state, self._info(state, terminated=terminated, truncated=truncated)


def test_space_conversion_continuous_scalar_and_vector_discrete():
    continuous = EnvelopeGymnax(_AdapterEnv(), max_steps_in_episode=3)
    assert isinstance(continuous.action_space(), gymnax_spaces.Box)
    np.testing.assert_array_equal(continuous.action_space().low, [-1.0])
    np.testing.assert_array_equal(continuous.action_space().high, [1.0])

    scalar = EnvelopeGymnax(
        _AdapterEnv(action_n=3), max_steps_in_episode=3
    ).action_space()
    assert isinstance(scalar, gymnax_spaces.Discrete)
    assert scalar.n == 3

    vector = EnvelopeGymnax(
        _AdapterEnv(action_n=(3, 5)), max_steps_in_episode=3
    ).action_space()
    assert isinstance(vector, GymnaxMultiDiscrete)
    assert vector.nvec == (3, 5)
    sample = vector.sample(jax.random.key(0))
    assert sample.shape == (2,)
    assert bool(vector.contains(sample))
    assert not bool(vector.contains(jnp.asarray([3, 0])))


@pytest.mark.parametrize(
    "space",
    [
        PyTreeSpace({"x": Continuous(low=0.0, high=1.0)}),
        BatchedSpace(Continuous(low=0.0, high=1.0), batch_size=2),
    ],
)
def test_unsupported_spaces_fail_clearly(space):
    with pytest.raises(TypeError, match="scalar Continuous and Discrete"):
        _convert_space(space)


def test_horizon_is_inferred_or_must_be_explicit_and_positive():
    wrapped = TruncationWrapper(env=_AdapterEnv(), max_steps=4)
    assert EnvelopeGymnax(wrapped).default_params.max_steps_in_episode == 4

    with pytest.raises(ValueError, match="required.*no TruncationWrapper"):
        EnvelopeGymnax(_AdapterEnv())
    for value in (True, 0, 1.5):
        with pytest.raises(ValueError, match="max_steps_in_episode"):
            EnvelopeGymnax(_AdapterEnv(), max_steps_in_episode=value)


def test_typed_and_legacy_keys_are_accepted_and_other_shapes_rejected():
    typed = jax.random.key(0)
    legacy = jax.random.PRNGKey(0)
    assert jnp.issubdtype(to_typed_key(typed).dtype, jax.dtypes.prng_key)
    assert jnp.issubdtype(to_typed_key(legacy).dtype, jax.dtypes.prng_key)
    with pytest.raises(ValueError, match="legacy uint32"):
        to_typed_key(jnp.zeros((3,), jnp.uint32))

    adapter = EnvelopeGymnax(_AdapterEnv(), max_steps_in_episode=3)
    typed_obs, _ = adapter.reset(typed)
    legacy_obs, _ = adapter.reset(legacy)
    np.testing.assert_array_equal(typed_obs, legacy_obs)


@pytest.mark.parametrize(
    ("env", "expected_terminated", "expected_truncated"),
    [
        (_AdapterEnv(terminate_after=1), True, False),
        (_AdapterEnv(truncate_after=1), False, True),
    ],
)
def test_boundary_preserves_terminal_info_and_state_aware_autoresets(
    env, expected_terminated, expected_truncated
):
    adapter = EnvelopeGymnax(env, max_steps_in_episode=3)
    obs, state = adapter.reset(jax.random.PRNGKey(0))
    np.testing.assert_array_equal(obs, [0.0])

    obs, state, reward, done, info = jax.jit(adapter.step)(
        jax.random.PRNGKey(1), state, jnp.asarray([0.0]), adapter.default_params
    )

    assert bool(done)
    assert bool(info.terminated) is expected_terminated
    assert bool(info.truncated) is expected_truncated
    np.testing.assert_array_equal(info.obs, [1.0])
    np.testing.assert_array_equal(reward, 1.0)
    np.testing.assert_array_equal(obs, [10.0])
    np.testing.assert_array_equal(state.steps, 0)
    np.testing.assert_array_equal(state.episodes, 1)


def test_reset_and_step_are_jittable_and_vmappable():
    adapter = EnvelopeGymnax(_AdapterEnv(terminate_after=2), max_steps_in_episode=3)
    keys = jax.random.split(jax.random.PRNGKey(2), 4)
    obs, states = jax.jit(jax.vmap(adapter.reset, in_axes=(0, None)))(
        keys, adapter.default_params
    )
    assert obs.shape == (4, 1)

    step_keys = jax.random.split(jax.random.PRNGKey(3), 4)
    actions = jnp.zeros((4, 1), jnp.float32)
    outputs = jax.jit(jax.vmap(adapter.step, in_axes=(0, 0, 0, None)))(
        step_keys, states, actions, adapter.default_params
    )
    assert outputs[0].shape == (4, 1)
    np.testing.assert_array_equal(outputs[3], np.zeros(4, dtype=bool))
