"""Rejax PPO on a plasmax environment.

uv sync --group research
uv run python examples/rejax_ppo.py
"""

import jax
from rejax import PPO

import plasmax
from plasmax.wrappers import RealisticWrappers
from training.envelope_gymnax import EnvelopeGymnax

# Envelope env -> Gymnax API, which is what Rejax consumes.
env = EnvelopeGymnax(
    RealisticWrappers(plasmax.make("iter/hybrid/flattop", backend="bohm_gyrobohm"))
)

algo = PPO.create(
    env=env,
    env_params=env.default_params,
    total_timesteps=100_000,
    num_envs=16,
    num_steps=64,
    num_epochs=4,
    num_minibatches=4,
    eval_freq=10_240,
    learning_rate=3e-4,
    gamma=0.99,
    gae_lambda=0.95,
    clip_eps=0.2,
    max_grad_norm=0.5,
)

train = jax.jit(algo.train)
train_state, (episode_lengths, episode_returns) = train(jax.random.PRNGKey(0))

# One row per evaluation, one column per evaluation episode.
for i, returns in enumerate(episode_returns):
    print(f"eval {i}: mean return {returns.mean():.3f}")
