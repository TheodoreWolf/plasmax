"""KSTAR world-model environment (JAX sibling of PlasmaxEnv).

A learned-dynamics RL env built on the NeoRL2 ``fusion_lstm`` KSTAR model
(Seo et al., Nucl. Fusion 2021). Unlike :class:`~plasmax.environment.PlasmaxEnv`
(a TORAX physics solver), the dynamics here are the JAX LSTM ensemble in
:mod:`plasmax.models.world_model`; both expose Envelope's state-and-info
lifecycle and property-based spaces, so the same wrapper stack can drive them.
Episode horizons are deliberately supplied by an outer truncation wrapper.

The task: hold three plasma scalars (βp, q95, li) at sampled targets by steering
6 engineering actuators (Ip + plasma shape). A faithful port of NeoRL2's
``FusionEnv`` reset/step/_predict0d, minus the h89/h98/wmhd diagnostics (which
feed neither the observation nor the reward).

Observation (15): ``[Ip, Elon, Up.Tri, Lo.Tri, In.Mid, Out.Mid, βp, q95, li,
βp*, q95*, li*, Pnb1a, Pnb1b, Pnb1c]`` (``*`` = target). Action (6) in
``[-1, 1]`` -> ``[Ip, Elon, Up.Tri, Lo.Tri, In.Mid, Out.Mid]``. Reward:
``-log(RMS((pred - target)/[0.2, 0.5, 0.05]))``.
"""

from __future__ import annotations

import dataclasses
from functools import cached_property
from typing import Self

import jax
import jax.numpy as jnp
import numpy as np
from envelope import Continuous, Environment, Info, InfoContainer, static_field

from plasmax.models.world_model import (
    load_bundle,
    predict_bpw,
    predict_lstm,
    predict_nn,
)
from plasmax.spaces import ObsLayout

# --- constants (verbatim from NeoRL2 neorl2/envs/fusion.py) -----------------
# float64 on purpose: these feed the slider quantization (_quantize), which must
# see the true decimal (e.g. 1.8, not float32's 1.79999995) to match NeoRL2's
# Python-float ``f2i``/``i2f`` round-trip exactly.
INPUT_INIT = np.array(
    [0.5, 1.8, 0.275, 1.5, 1.5, 0.5, 0.0, 0.0, 0.0, 0.0, 1.32, 2.22, 1.7, 0.3, 0.75]
)
INPUT_MINS = np.array(
    [0.3, 1.5, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, -10, -10, 1.265, 2.18, 1.6, 0.1, 0.5]
)
INPUT_MAXS = np.array(
    [0.8, 2.7, 0.6, 1.75, 1.75, 1.5, 0.8, 0.8, 10, 10, 1.36, 2.29, 2.0, 0.5, 0.9]
)
TARGET_INIT = np.array([1.6, 5.0, 0.95])
TARGET_MINS = np.array([1.1, 3.8, 0.84])
TARGET_MAXS = np.array([2.1, 6.2, 1.06])

CTRL_IDX = np.array([0, 12, 13, 14, 10, 11])  # action -> input_params indices
NBI_IDX = np.array([3, 4, 5])  # Pnb1a/b/c
LSTM_IN_IDX = np.array([0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 10, 2])
BPW_IN_IDX = np.array([0, 1, 10, 11, 12, 13, 14])  # after the βn slot

INTERVAL = 20
YEAR_IN = 2021.0
_SCALE = 10.0 ** np.log10(1000.0)  # slider int<->float quantization
REWARD_SCALE = np.array([0.2, 0.5, 0.05], np.float32)

OBS_NAMES = (
    "Ip",
    "Elon",
    "Up.Tri",
    "Lo.Tri",
    "In.Mid",
    "Out.Mid",
    "betap",
    "q95",
    "li",
    "betap_target",
    "q95_target",
    "li_target",
    "Pnb1a",
    "Pnb1b",
    "Pnb1c",
)
OBS_LOW = np.array(
    [
        0.3,
        1.6,
        0.1,
        0.5,
        1.265,
        2.18,
        -np.inf,
        -np.inf,
        -np.inf,
        1.1,
        3.8,
        0.84,
        0.0,
        0.0,
        0.0,
    ],
    np.float32,
)
OBS_HIGH = np.array(
    [
        0.8,
        2.0,
        0.5,
        0.9,
        1.36,
        2.29,
        np.inf,
        np.inf,
        np.inf,
        2.1,
        6.2,
        1.06,
        1.75,
        1.75,
        1.5,
    ],
    np.float32,
)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class WorldModelEnvState:
    """Episode state for :class:`WorldModelEnv` (a JAX pytree).

    Attributes:
      x: ``(10, 21)`` LSTM history buffer (cols 0-3 = denormalised [βn,q95,q0,li]
        outputs, cols 4-20 = transformed input features incl. a fixed year).
      inputs: ``(15,)`` current actuator/engineering values (physical, quantized).
      targets: ``(3,)`` per-episode [βp, q95, li] setpoints.
      t: scalar step counter.
      prev_action: last action (``(6,)``), for rate-limit-style wrappers.
    """

    x: jax.Array
    inputs: jax.Array
    targets: jax.Array
    t: jax.Array
    prev_action: jax.Array


def _quantize(v: jax.Array) -> jax.Array:
    """Replicate NeoRL2's slider round-trip ``i2f(f2i(v))`` (truncate to ~3 dp).

    Done in float64 because NeoRL2's scale is ``10**log10(1000) = 999.999…``
    (not exactly 1000); float32 would collapse it to 1000 and shift values.
    """
    return (jnp.trunc(v.astype(jnp.float64) * _SCALE) / _SCALE).astype(jnp.float32)


def _lstm_features(inputs: jax.Array) -> jax.Array:
    """The 16 transformed input columns (buffer cols 4-19) for the LSTM."""
    f = inputs[LSTM_IN_IDX]
    rin, rout = f[9], f[10]
    f = f.at[9].set(0.5 * (rin + rout)).at[10].set(0.5 * (rout - rin))  # geo -> (R, a)
    f = f.at[14].set(jnp.where(f[14] > 1.265 + 1e-4, 1.0, 0.0))  # In.Mid -> bit
    return f


def _steady_features(inputs: jax.Array) -> jax.Array:
    """17-vector for the steady-state ``nn`` init: LSTM features + fixed year."""
    return jnp.concatenate([_lstm_features(inputs), jnp.array([YEAR_IN], jnp.float32)])


def _bpw_features(beta_n: jax.Array, inputs: jax.Array) -> jax.Array:
    """8-vector for the β-power head: ``[βn, Ip, Bt, R, a, Elon, Up.Tri, Lo.Tri]``."""
    arr = jnp.concatenate([beta_n[None], inputs[BPW_IN_IDX]])
    rin, rout = arr[3], arr[4]
    return arr.at[3].set(0.5 * (rin + rout)).at[4].set(0.5 * (rout - rin))


def target_tracking_reward(pred: jax.Array, targets: jax.Array) -> jax.Array:
    """``-log(RMS((pred - target)/scale))`` over [βp, q95, li] (NeoRL2 reward)."""
    err = (pred - targets) / REWARD_SCALE
    rms = jnp.sqrt(jnp.mean(err**2))
    return -jnp.log(jnp.maximum(rms, 1e-6))


def _validate_typed_key(key: jax.Array) -> None:
    dtype = getattr(key, "dtype", None)
    shape = getattr(key, "shape", None)
    if dtype is None or shape != () or not jnp.issubdtype(dtype, jax.dtypes.prng_key):
        raise ValueError("key must be a scalar typed (new-style) jax.random.key")


def _make_info(obs: jax.Array, reward: jax.Array) -> InfoContainer:
    return InfoContainer(
        obs=obs,
        reward=jnp.asarray(reward, dtype=jnp.float32),
        terminated=jnp.asarray(False, dtype=jnp.bool_),
        truncated=jnp.asarray(False, dtype=jnp.bool_),
    ).update(termination_code=jnp.asarray(-1, dtype=jnp.int32))


class _WorldModelDynamics:
    """Identity-hashable holder for the learned dynamics and weight bundle.

    Args:
      bundle: weights from :func:`plasmax.models.world_model.load_bundle` (defaults to
        the vendored npz).
    """

    def __init__(self, bundle=None):
        self._bundle = load_bundle() if bundle is None else bundle
        # Construct spaces eagerly, outside any JAX transformation.  In
        # particular, this prevents a first property access during tracing
        # from leaving cached tracer-valued bounds on the dynamics object.
        self._action_space = Continuous(
            low=-jnp.ones(6, jnp.float32), high=jnp.ones(6, jnp.float32)
        )
        self._observation_space = Continuous(
            low=jnp.asarray(OBS_LOW), high=jnp.asarray(OBS_HIGH)
        )

    # --- spaces ---
    @property
    def action_space(self) -> Continuous:
        return self._action_space

    @property
    def observation_space(self) -> Continuous:
        return self._observation_space

    def obs_layout(self) -> ObsLayout:
        slices = {n: slice(i, i + 1) for i, n in enumerate(OBS_NAMES)}
        return ObsLayout(
            profile_slices={},
            scalar_slices=slices,
            profile_names=(),
            scalar_names=OBS_NAMES,
        )

    # --- dynamics ---
    def _assemble_obs(self, inputs, x, targets, beta_p) -> jax.Array:
        obs = jnp.zeros(15, jnp.float32)
        obs = obs.at[:6].set(inputs[CTRL_IDX])
        obs = obs.at[6].set(beta_p)
        obs = obs.at[7].set(x[-1, 1])  # q95
        obs = obs.at[8].set(x[-1, 3])  # li
        obs = obs.at[9:12].set(targets)
        obs = obs.at[12:15].set(inputs[NBI_IDX])
        return obs

    def init(
        self, key: jax.Array, *, random_target: bool
    ) -> tuple[WorldModelEnvState, InfoContainer]:
        _validate_typed_key(key)
        inputs = _quantize(jnp.asarray(INPUT_INIT))
        if random_target:
            targets = jax.random.uniform(
                key,
                (3,),
                minval=jnp.asarray(TARGET_MINS),
                maxval=jnp.asarray(TARGET_MAXS),
            )
        else:
            targets = jnp.asarray(TARGET_INIT)
        targets = targets.astype(jnp.float32)

        # Steady-state init: fill every buffer row with [nn output | features].
        feats = _steady_features(inputs)  # (17,)
        y0 = predict_nn(self._bundle, feats)  # (4,) [βn,q95,q0,li]
        x = jnp.zeros((10, 21), jnp.float32)
        x = x.at[:, :4].set(y0)
        x = x.at[:, 4:].set(feats)

        beta_p = predict_bpw(self._bundle, _bpw_features(x[-1, 0], inputs))[0]
        obs = self._assemble_obs(inputs, x, targets, beta_p)
        state = WorldModelEnvState(
            x=x,
            inputs=inputs,
            targets=targets,
            t=jnp.array(0, jnp.int32),
            prev_action=jnp.zeros(6, jnp.float32),
        )
        return state, _make_info(obs, jnp.zeros((), dtype=jnp.float32))

    def step(
        self, env_state: WorldModelEnvState, action: jax.Array
    ) -> tuple[WorldModelEnvState, InfoContainer]:
        action = jnp.clip(action, -1.0, 1.0).astype(jnp.float32)
        a01 = (action + 1.0) / 2.0
        lo = jnp.asarray(INPUT_MINS)[CTRL_IDX]
        hi = jnp.asarray(INPUT_MAXS)[CTRL_IDX]
        inputs = env_state.inputs.at[CTRL_IDX].set(_quantize((hi - lo) * a01 + lo))

        feat = _lstm_features(inputs)  # constant over the relaxation phase

        def relax(x, _):
            x = x.at[:-1, 4:].set(x[1:, 4:])  # shift input cols (year preserved)
            x = x.at[-1, 4:20].set(feat)
            y = predict_lstm(self._bundle, x)  # (4,)
            x = x.at[:-1, :4].set(x[1:, :4])  # shift output cols
            x = x.at[-1, :4].set(y)
            return x, None

        x, _ = jax.lax.scan(relax, env_state.x, None, length=INTERVAL - 1)

        beta_p = predict_bpw(self._bundle, _bpw_features(x[-1, 0], inputs))[0]
        obs = self._assemble_obs(inputs, x, env_state.targets, beta_p)
        reward = target_tracking_reward(obs[6:9], env_state.targets)

        t = env_state.t + 1
        new_state = WorldModelEnvState(
            x=x,
            inputs=inputs,
            targets=env_state.targets,
            t=t,
            prev_action=action,
        )
        return new_state, _make_info(obs, reward)


def _default_world_model_dynamics() -> _WorldModelDynamics:
    return _WorldModelDynamics()


class WorldModelEnv(Environment):
    """Envelope-native KSTAR fusion_lstm environment.

    The learned transition is deterministic. ``random_target`` affects only
    typed-key initialization/reset, while an outer truncation wrapper owns the
    YAML-configured KSTAR episode horizon.
    """

    random_target: bool = static_field(default=True)
    _dynamics: _WorldModelDynamics = static_field(
        default_factory=_default_world_model_dynamics, repr=False
    )

    @classmethod
    def from_bundle(cls, bundle, *, random_target: bool = True) -> Self:
        """Constructs the backend from a caller-supplied JAX weight bundle."""
        return cls(
            random_target=random_target,
            _dynamics=_WorldModelDynamics(bundle=bundle),
        )

    def init(self, key: jax.Array) -> tuple[WorldModelEnvState, Info]:
        return self._dynamics.init(key, random_target=self.random_target)

    def reset(
        self, state: WorldModelEnvState, key: jax.Array
    ) -> tuple[WorldModelEnvState, Info]:
        del state
        return self._dynamics.init(key, random_target=self.random_target)

    def step(
        self, state: WorldModelEnvState, action: jax.Array
    ) -> tuple[WorldModelEnvState, Info]:
        return self._dynamics.step(state, action)

    @cached_property
    def action_space(self) -> Continuous:
        return self._dynamics.action_space

    @cached_property
    def observation_space(self) -> Continuous:
        return self._dynamics.observation_space

    def obs_layout(self) -> ObsLayout:
        return self._dynamics.obs_layout()
