"""Collect representative full profiles for the backend comparison."""

from __future__ import annotations

import json

import jax
import numpy as np

from plasmax.environment import make
from plasmax.wrappers import PhysicsRandomizationWrapper, unwrap_to_env_state

BACKENDS = ("cgm", "bohm_gyrobohm", "qlknn", "tglfnn")
N_STEPS = 100


def _snapshot(state):
    plasma = unwrap_to_env_state(state).plasma
    return (
        plasma.t,
        plasma.geo.rho_norm,
        plasma.T_e,
        plasma.T_i,
        plasma.n_e,
        plasma.q,
    )


def _collect_backend(backend: str) -> dict[str, object]:
    env = PhysicsRandomizationWrapper(make("iter/hybrid/flattop", backend))

    @jax.jit
    def rollout(key: jax.Array):
        state, _ = env.init(key)
        action = unwrap_to_env_state(state).prev_action
        initial = _snapshot(state)

        def advance(carry, _):
            next_state, info = env.step(carry, action)
            return next_state, (
                _snapshot(next_state),
                info.control_step_complete,
                info.terminated,
            )

        _, (snapshots, complete, terminated) = jax.lax.scan(
            advance,
            state,
            xs=None,
            length=N_STEPS,
        )
        return initial, snapshots, complete, terminated

    initial, snapshots, complete, terminated = rollout(jax.random.key(0))
    jax.block_until_ready((initial, snapshots, complete, terminated))
    complete_np = np.asarray(complete)
    terminated_np = np.asarray(terminated)
    if not np.all(complete_np):
        raise RuntimeError(
            f"{backend}: incomplete control steps "
            f"{np.flatnonzero(~complete_np).tolist()}"
        )
    if np.any(terminated_np):
        raise RuntimeError(
            f"{backend}: terminated at steps "
            f"{(np.flatnonzero(terminated_np) + 1).tolist()}"
        )

    snapshots_np = tuple(np.asarray(value) for value in snapshots)
    selected = {
        "start": tuple(np.asarray(value) for value in initial),
        "middle": tuple(value[49] for value in snapshots_np),
        "end": tuple(value[99] for value in snapshots_np),
    }
    result: dict[str, object] = {}
    for phase, values in selected.items():
        t, rho, t_e, t_i, n_e, q = values
        arrays = (rho, t_e, t_i, n_e, q)
        if not all(np.all(np.isfinite(value)) for value in arrays):
            raise RuntimeError(f"{backend}: non-finite {phase} profile")
        result[phase] = {
            "t": float(t),
            "rho": np.asarray(rho).tolist(),
            "T_e": np.asarray(t_e).tolist(),
            "T_i": np.asarray(t_i).tolist(),
            "n_e": np.asarray(n_e).tolist(),
            "q": np.asarray(q).tolist(),
        }
    return result


def main() -> None:
    data = {backend: _collect_backend(backend) for backend in BACKENDS}
    print("PROFILE_DATA=" + json.dumps(data, separators=(",", ":")))


if __name__ == "__main__":
    main()
