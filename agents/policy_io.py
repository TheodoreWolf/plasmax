"""Small, inference-only policy artifacts for the repository's agents."""

from __future__ import annotations

import dataclasses
import functools
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax import serialization
from rejax.networks import DiscretePolicy, GaussianPolicy, SquashedGaussianPolicy

from agents.ppo import MultiDiscretePolicy, PPOAdapter, ResidualGaussianPolicy
from agents.sac import SACAdapter

__all__ = ["LoadedPolicy", "environment_interface", "load_policy", "save_policy"]


def _plain(value: Any) -> Any:
    """Convert metadata and inference pytrees to MessagePack's plain values."""
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if hasattr(value, "model_dump"):
        return _plain(value.model_dump(mode="json"))
    if dataclasses.is_dataclass(value):
        return {
            field.name: _plain(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, jax.Array):
        if jnp.issubdtype(value.dtype, jax.dtypes.prng_key):
            value = jax.random.key_data(value)
        return np.asarray(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _native_env(env: Any) -> Any:
    # Rejax may put FloatObsWrapper around the Gymnax adapter.
    while callable(getattr(env, "action_space", None)) and hasattr(env, "env"):
        env = env.env
    return getattr(env, "envelope_env", env)


def environment_interface(env: Any) -> dict[str, Any]:
    """Describe policy inputs/outputs, without restricting simulator transfer."""
    with jax.ensure_compile_time_eval():
        env = _native_env(env)
        action = env.action_space
        observation = env.observation_space
        if callable(action):
            action, observation = action(), observation()
    interface: dict[str, Any] = {
        "observation_shape": list(observation.shape),
        "action_shape": list(action.shape),
        "action_dtype": str(np.dtype(action.dtype)),
    }
    if hasattr(action, "nvec"):
        interface["action_bins"] = list(action.nvec)
    elif hasattr(action, "n"):
        interface["action_bins"] = np.asarray(action.n).tolist()
    else:
        interface["action_low"] = np.asarray(action.low).tolist()
        interface["action_high"] = np.asarray(action.high).tolist()
    if hasattr(env, "obs_layout"):
        layout = env.obs_layout()
        interface["observations"] = {
            name: [section.start, section.stop]
            for name, section in {
                **layout.profile_slices,
                **layout.scalar_slices,
            }.items()
        }
    for name in ("profile_obs_specs", "scalar_obs_specs", "actuator_specs"):
        if hasattr(env, name):
            specs = getattr(env, name)
            fields = (
                ("name", "scale")
                if name != "actuator_specs"
                else ("name", "low", "high")
            )
            interface[name] = [
                {key: _plain(getattr(spec, key)) for key in fields} for spec in specs
            ]
    interface["history"] = []
    layer = env
    while hasattr(layer, "env"):
        if type(layer).__name__ == "ObsHistoryWrapper":
            interface["history"].append(layer.k)
        layer = layer.env
    return interface


def _actor_spec(actor: nn.Module) -> dict[str, Any]:
    if type(actor) not in (
        DiscretePolicy,
        GaussianPolicy,
        SquashedGaussianPolicy,
        MultiDiscretePolicy,
        ResidualGaussianPolicy,
    ):
        raise ValueError(f"unsupported policy model {type(actor).__name__}")
    activation = next(
        (
            name
            for name in (
                "relu",
                "tanh",
                "swish",
                "silu",
                "gelu",
                "elu",
                "sigmoid",
                "leaky_relu",
                "softplus",
            )
            if getattr(nn, name) is actor.activation
        ),
        None,
    )
    if activation is None:
        raise ValueError("policy export requires a supported named Flax activation")
    result = {
        "type": type(actor).__name__,
        "hidden_layer_sizes": actor.hidden_layer_sizes,
        "activation": activation,
    }
    for name in (
        "action_dim",
        "action_range",
        "nvec",
        "action_setpoint",
        "initial_log_std",
        "log_std_range",
    ):
        if hasattr(actor, name):
            result[name] = getattr(actor, name)
    return result


def _restore_actor(spec: dict[str, Any]) -> nn.Module:
    fields = dict(spec)
    kind = fields.pop("type")
    activation = fields.pop("activation")
    if activation not in (
        "relu",
        "tanh",
        "swish",
        "silu",
        "gelu",
        "elu",
        "sigmoid",
        "leaky_relu",
        "softplus",
    ):
        raise ValueError(f"unsupported policy activation {activation!r}")
    fields["activation"] = getattr(nn, activation)
    fields["hidden_layer_sizes"] = tuple(fields["hidden_layer_sizes"])
    for name in ("action_range", "log_std_range", "nvec", "action_setpoint"):
        if name in fields:
            fields[name] = tuple(fields[name])
    if kind == "DiscretePolicy":
        return DiscretePolicy(**fields)
    if kind == "GaussianPolicy":
        return GaussianPolicy(**fields)
    if kind == "SquashedGaussianPolicy":
        return SquashedGaussianPolicy(**fields)
    if kind == "MultiDiscretePolicy":
        return MultiDiscretePolicy(**fields)
    if kind == "ResidualGaussianPolicy":
        return ResidualGaussianPolicy(**fields)
    raise ValueError(f"unsupported policy model {kind!r}")


@dataclasses.dataclass(frozen=True)
class LoadedPolicy:
    """An evaluation snapshot; no environment or training state is restored."""

    algorithm: str
    inference: dict[str, Any]
    interface: dict[str, Any]
    metadata: dict[str, Any]
    deterministic: bool
    results: Any = None

    def summary(self) -> str:
        mode = "deterministic" if self.deterministic else "stochastic"
        return (
            f"{self.algorithm}: {mode}; "
            f"observations {tuple(self.interface['observation_shape'])}, "
            f"actions {tuple(self.interface['action_shape'])}"
        )

    def make_act(self, deterministic: bool | None = None) -> Callable:
        """Bind frozen parameters to the common ``act(obs, rng)`` interface."""
        deterministic = self.deterministic if deterministic is None else deterministic
        payload = self.inference
        if self.algorithm in ("ppo", "sac"):
            actor = _restore_actor(payload["model"])
            params = payload["params"]
            rms = payload.get("observation_rms")

            def act(obs, rng):
                if rms is not None:
                    obs = (obs - jnp.asarray(rms["mean"])) / jnp.sqrt(
                        jnp.asarray(rms["var"]) + 1e-8
                    )
                obs = jnp.expand_dims(obs, 0)
                if not deterministic:
                    action = actor.apply(params, obs, rng, method="act")
                elif isinstance(actor, MultiDiscretePolicy):
                    action = jnp.argmax(
                        actor.apply(params, obs, method="_padded_logits"), axis=-1
                    )
                else:
                    action = actor.apply(params, obs, method="_action_dist").mode()
                    if isinstance(actor, SquashedGaussianPolicy):
                        action = (
                            actor.action_loc + jnp.tanh(action) * actor.action_scale
                        )
                    elif isinstance(actor, (GaussianPolicy, ResidualGaussianPolicy)):
                        action = jnp.clip(action, *actor.action_range)
                return jnp.reshape(action, tuple(self.interface["action_shape"]))

            return act
        if self.algorithm == "backprop_policy":
            from agents.backprop import ResidualPolicy, policy_action

            policy = ResidualPolicy(
                action_dim=payload["action_dim"],
                hidden_sizes=tuple(payload["hidden_sizes"]),
            )
            return lambda obs, rng: policy_action(
                policy, payload["params"], jnp.asarray(payload["action_setpoint"]), obs
            )
        if self.algorithm == "backprop_open_loop":
            from agents.backprop import open_loop_action

            return lambda obs, rng: open_loop_action(
                jnp.asarray(payload["params"]),
                obs,
                jnp.asarray(payload["source_times"]),
                payload["time_index"],
            )
        if self.algorithm == "mpc":
            from agents.mpc import WorldModel, plan_action

            model = WorldModel(obs_dim=payload["obs_dim"], hidden=payload["hidden"])
            section = slice(*payload["reward_slice"])

            def reward(obs, action, next_obs):
                del obs, action
                return jnp.sum(next_obs[section])

            def act(obs, rng):
                if deterministic:
                    rng = jax.random.key(payload["planning_seed"])
                return plan_action(
                    model,
                    payload["params"],
                    jnp.asarray(payload["action_low"]),
                    jnp.asarray(payload["action_high"]),
                    reward,
                    obs,
                    rng,
                    horizon=payload["horizon"],
                    num_samples=payload["num_samples"],
                )

            return act
        raise ValueError(f"unsupported policy algorithm {self.algorithm!r}")


def save_policy(
    agent: Any,
    state: Any,
    path: str | Path | None = None,
    *,
    results: Any = None,
    metadata: dict[str, Any] | None = None,
    deterministic: bool | None = None,
) -> Path:
    """Atomically save one successful seed's inference state as MessagePack."""
    from agents.backprop import BackpropOpenLoopAgent, BackpropPolicyAgent
    from agents.mpc import MPCAgent

    if np.any(np.asarray(getattr(state, "failed", False))):
        raise ValueError("cannot export a policy from failed training")
    if np.ndim(getattr(state, "global_step", 0)) != 0:
        raise ValueError("save_policy requires one training seed's state")
    if isinstance(agent, (PPOAdapter, SACAdapter)):
        algorithm = "ppo" if isinstance(agent, PPOAdapter) else "sac"
        inference = {
            "model": _actor_spec(agent.actor),
            "params": serialization.to_state_dict(state.actor_ts.params),
        }
        if agent.normalize_observations:
            inference["observation_rms"] = {
                "mean": state.obs_rms_state.mean,
                "var": state.obs_rms_state.var,
            }
    elif isinstance(agent, BackpropPolicyAgent):
        algorithm = "backprop_policy"
        inference = {
            "params": serialization.to_state_dict(state.params),
            "action_dim": agent.policy.action_dim,
            "hidden_sizes": agent.policy.hidden_sizes,
            "action_setpoint": agent.action_setpoint,
        }
    elif isinstance(agent, BackpropOpenLoopAgent):
        algorithm = "backprop_open_loop"
        inference = {
            "params": state.params,
            "source_times": agent.source_times,
            "time_index": agent.time_index,
        }
    elif isinstance(agent, MPCAgent):
        if agent.reward_scalar is None or agent.reward_slice is None:
            raise ValueError(
                "MPC export requires a named observation-scalar objective; "
                "arbitrary reward callables are unsupported"
            )
        algorithm = "mpc"
        inference = {
            "params": serialization.to_state_dict(state.model_ts.params),
            "obs_dim": agent.model.obs_dim,
            "hidden": agent.model.hidden,
            "action_low": agent.action_low,
            "action_high": agent.action_high,
            "horizon": agent.horizon,
            "num_samples": agent.num_samples,
            "reward_scalar": agent.reward_scalar,
            "reward_slice": agent.reward_slice,
            "planning_seed": agent.planning_seed,
        }
    else:
        raise ValueError(f"unsupported agent {type(agent).__name__}")
    for leaf in jax.tree.leaves(inference["params"]):
        if not np.all(np.isfinite(np.asarray(leaf))):
            raise ValueError("cannot export non-finite policy parameters")
    provenance = dict(metadata or {})
    provenance.setdefault("created_at", datetime.now(UTC).isoformat())
    if hasattr(state, "global_step"):
        provenance["actual_timesteps"] = state.global_step
    env = _native_env(agent.env)
    if getattr(env, "plasmax_config", None) is not None:
        provenance.setdefault("source_config", env.plasmax_config)
        dynamics = getattr(env.unwrapped, "_dynamics", None)
        if hasattr(dynamics, "_disruption_penalty"):
            reward = dynamics._reward_fn
            while isinstance(reward, functools.partial):
                reward = reward.func
            provenance["effective_task"] = {
                "reward": getattr(reward, "__name__", None),
                "terminal_penalty": float(dynamics._disruption_penalty),
            }
    from plasmax.wrappers import find_max_steps

    provenance["source_max_steps"] = find_max_steps(env)
    snapshot = LoadedPolicy(
        algorithm,
        inference,
        environment_interface(env),
        provenance,
        getattr(agent, "deterministic", algorithm.startswith("backprop_"))
        if deterministic is None
        else deterministic,
        results,
    )
    payload = _plain(
        {
            "format_version": 1,
            **{
                field.name: getattr(snapshot, field.name)
                for field in dataclasses.fields(snapshot)
            },
        }
    )
    if path is None:
        seed = f"_seed{provenance['seed']}" if "seed" in provenance else ""
        path = (
            Path("outputs/policies")
            / f"{datetime.now(UTC):%Y%m%dT%H%M%S.%fZ}_{uuid4().hex[:8]}{seed}.msgpack"
        )
    output = Path(path)
    if output.suffix != ".msgpack":
        raise ValueError("policy artifacts must use the .msgpack extension")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_bytes(serialization.msgpack_serialize(payload))
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def load_policy(path: str | Path) -> LoadedPolicy:
    """Read the new policy format without initializing a model or environment."""
    payload = serialization.msgpack_restore(Path(path).read_bytes())
    if not isinstance(payload, dict) or payload.get("format_version") != 1:
        raise ValueError("unsupported policy artifact format")
    if payload.get("algorithm") not in (
        "ppo",
        "sac",
        "backprop_policy",
        "backprop_open_loop",
        "mpc",
    ):
        raise ValueError(f"unsupported policy algorithm {payload.get('algorithm')!r}")
    payload.pop("format_version")
    return LoadedPolicy(**payload)
