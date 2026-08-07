"""Programmatic PPO training for Envelope environments through Rejax."""

import jax
from envelope import Environment

from agents.ppo import PPOAdapter
from training.envelope_gymnax import EnvelopeGymnax


def train_ppo(
    env: Environment | EnvelopeGymnax,
    *,
    total_timesteps: int = 10_000,
    num_envs: int = 1,
    learning_rate: float = 3e-4,
    num_steps: int = 100,
    seed: int = 0,
    **ppo_kwargs,
) -> tuple:
    """Train upstream Rejax PPO and return its state and evaluations."""

    gymnax_env = env if isinstance(env, EnvelopeGymnax) else EnvelopeGymnax(env)

    algo = PPOAdapter.create(
        env=gymnax_env,
        env_params=gymnax_env.default_params,
        learning_rate=learning_rate,
        total_timesteps=total_timesteps,
        num_envs=num_envs,
        num_steps=num_steps,
        **ppo_kwargs,
    )
    rng = jax.random.PRNGKey(seed)
    return jax.jit(algo.train)(rng)
