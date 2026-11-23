"""Generated interface-only starter; replace the control logic for this task."""

# ruff: noqa: F821
# Template markers are replaced from the instantiated environment interface.

import jax
import jax.numpy as jnp

POLICY_API_VERSION = 1
POLICY_METADATA = {"name": "initial_command", "action_size": __ACTION_SIZE__}


def init_memory() -> jax.Array:
    return jnp.asarray(0, dtype=jnp.int32)


def act(
    obs: jax.Array,
    reset: jax.Array,
    memory: jax.Array,
    rng: jax.Array,
) -> tuple[jax.Array, jax.Array, dict[str, jax.Array]]:
    del obs, rng
    action = jnp.asarray(__INITIAL_ACTION__, dtype=jnp.__ACTION_DTYPE__)
    next_memory = jnp.where(reset, jnp.asarray(0, dtype=jnp.int32), memory)
    return action, next_memory, {}
