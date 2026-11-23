"""Train CleanRL's unmodified PPO on a plasmax environment via Gymnasium.

plasmax environments are explicit-state JAX functions, so a Gymnasium view is a
few lines: hold the state, call the jitted ``init``/``step``, convert to NumPy.

    uv run python examples/cleanrl_ppo.py --total-timesteps 100000
"""

import runpy
import sys

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np

import plasmax
from plasmax.wrappers import RealisticWrappers

ENV_ID = "plasmax/IterHybridFlattop-v0"

ENV = RealisticWrappers(plasmax.make("iter/hybrid/flattop", backend="bohm_gyrobohm"))
INIT = jax.jit(ENV.init)
STEP = jax.jit(ENV.step)


def _box(space) -> gym.spaces.Box:
    low = np.broadcast_to(np.asarray(space.low, np.float32), space.shape)
    high = np.broadcast_to(np.asarray(space.high, np.float32), space.shape)
    return gym.spaces.Box(low=low, high=high, dtype=np.float32)


class PlasmaxGymEnv(gym.Env):
    observation_space = _box(ENV.observation_space)
    action_space = _box(ENV.action_space)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        key = jax.random.key(int(self.np_random.integers(2**31)))
        self._state, info = INIT(key)
        return np.asarray(info.obs, np.float32), {}

    def step(self, action):
        action = jnp.asarray(action, ENV.action_space.dtype)
        self._state, info = STEP(self._state, action)
        return (
            np.asarray(info.obs, np.float32),
            float(info.reward),
            bool(info.terminated),
            bool(info.truncated),
            {},
        )


gym.register(id=ENV_ID, entry_point=PlasmaxGymEnv)

if __name__ == "__main__":
    sys.argv = ["ppo_continuous_action", "--env-id", ENV_ID, *sys.argv[1:]]
    runpy.run_module("cleanrl.ppo_continuous_action", run_name="__main__")
