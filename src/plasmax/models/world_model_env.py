"""KSTAR world-model environment (JAX sibling of PlasmaxEnv).

A learned-dynamics RL env built on the NeoRL2 ``fusion_lstm`` KSTAR model
(Seo et al., Nucl. Fusion 2021). Unlike :class:`~plasmax.environment.PlasmaxEnv`
(a TORAX physics solver), the dynamics here are the JAX LSTM ensemble in
:mod:`plasmax.models.world_model`; both expose Envelope's state-and-info
lifecycle, physical actuator units, persistent ``phys_params``, and
property-based spaces, so the same wrapper stack can drive them. Episode
horizons and normalized ``[-1, 1]`` policy actions are supplied by outer
wrappers (such as :class:`~plasmax.wrappers.ActionRescaleWrapper` and
:class:`~plasmax.wrappers.TruncationWrapper`).

The task: hold three plasma scalars (βp, q95, li) at sampled targets by steering
6 engineering actuators (Ip + plasma shape). Adapted from NeoRL2's
``FusionEnv`` reset/step/_predict0d with continuous physical actuator inputs for
backpropagation, minus the h89/h98/wmhd diagnostics (which feed neither the
observation nor the reward).

Observation (15): ``[Ip, Elon, Up.Tri, Lo.Tri, In.Mid, Out.Mid, βp, q95, li,
βp*, q95*, li*, Pnb1a, Pnb1b, Pnb1c]`` (``*`` = target). Action (6) in physical
engineering units -> ``[Ip, Elon, Up.Tri, Lo.Tri, In.Mid, Out.Mid]``. Reward:
``-log(RMS((pred - target)/[0.2, 0.5, 0.05]))``.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from functools import cached_property
from pathlib import Path
from typing import Self

import jax
import jax.numpy as jnp
import numpy as np
from envelope import (
    Continuous,
    Environment,
    Info,
    InfoContainer,
    WrappedState,
    static_field,
)

from plasmax.environment.initialization_data import (
    KstarInitialization,
    load_initialization,
)
from plasmax.environment.schema import (
    KSTAR_ACTUATOR_NAMES,
    KSTAR_OBS_NAMES,
    PhysicsRandomizationSpec,
    WorldModelConfig,
)
from plasmax.models.world_model import (
    load_bundle,
    predict_bpw,
    predict_lstm,
)
from plasmax.spaces import ActuatorSpec, ObsLayout, ObsSpec

# --- constants (verbatim from NeoRL2 neorl2/envs/fusion.py) -----------------
INPUT_MINS = np.array(
    [0.3, 1.5, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, -10, -10, 1.265, 2.18, 1.6, 0.1, 0.5]
)
INPUT_MAXS = np.array(
    [0.8, 2.7, 0.6, 1.75, 1.75, 1.5, 0.8, 0.8, 10, 10, 1.36, 2.29, 2.0, 0.5, 0.9]
)

CTRL_IDX = np.array([0, 12, 13, 14, 10, 11])  # action -> input_params indices
NBI_IDX = np.array([3, 4, 5])  # Pnb1a/b/c
LSTM_IN_IDX = np.array([0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 10, 2])
BPW_IN_IDX = np.array([0, 1, 10, 11, 12, 13, 14])  # after the βn slot

INTERVAL = 20
YEAR_IN = 2021.0
REWARD_SCALE = np.array([0.2, 0.5, 0.05], np.float32)

OBS_NAMES = KSTAR_OBS_NAMES
_ACTUATOR_TO_INPUT_IDX: dict[str, int] = dict(
    zip(KSTAR_ACTUATOR_NAMES, CTRL_IDX.tolist(), strict=True)
)
_DEFAULT_ACTUATOR_SPECS: tuple[ActuatorSpec, ...] = tuple(
    ActuatorSpec(
        name=name,
        low=float(INPUT_MINS[idx]),
        high=float(INPUT_MAXS[idx]),
    )
    for name, idx in _ACTUATOR_TO_INPUT_IDX.items()
)
_ENSEMBLE_MEMBER_PATHS = frozenset(
    {
        "world_model.ensemble_member",
        "world_model.lstm_member",
        "world_model.bpw_member",
    }
)
_NO_PHYSICS_RANDOMIZATION: dict[str, PhysicsRandomizationSpec] = {}
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
        -np.inf,
        -np.inf,
        -np.inf,
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
        np.inf,
        np.inf,
        np.inf,
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
      inputs: ``(15,)`` current actuator/engineering values (physical, continuous).
      targets: ``(3,)`` per-episode [βp, q95, li] setpoints.
      t: scalar step counter.
      prev_action: last applied physical action vector (``(6,)``), used for
        rate limiting and observation-history initialization.
      phys_params: persistent world-model randomization parameters keyed by
        dot-path, initialized to configured nominal values at reset.
    """

    x: jax.Array
    inputs: jax.Array
    targets: jax.Array
    t: jax.Array
    prev_action: jax.Array
    phys_params: dict[str, jax.Array]


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
    # Floor before sqrt so exact target matches also have finite derivatives.
    rms = jnp.sqrt(jnp.maximum(jnp.mean(err**2), 1e-12))
    return -jnp.log(rms)


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


def _validate_actuator_specs(actuator_specs: Sequence[ActuatorSpec]) -> None:
    names = [spec.name for spec in actuator_specs]
    unknown = [name for name in names if name not in _ACTUATOR_TO_INPUT_IDX]
    if unknown:
        raise ValueError(
            f"Unknown KSTAR actuator names: {unknown}. "
            f"Valid names are: {list(_ACTUATOR_TO_INPUT_IDX)}"
        )
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate KSTAR actuator names are not allowed: {names}")
    if set(names) != set(_ACTUATOR_TO_INPUT_IDX):
        missing = sorted(set(_ACTUATOR_TO_INPUT_IDX) - set(names))
        raise ValueError(f"Missing KSTAR actuators: {missing}")
    for spec in actuator_specs:
        if (
            not np.isfinite(spec.low)
            or not np.isfinite(spec.high)
            or not (spec.low < spec.high)
        ):
            raise ValueError(
                f"Actuator {spec.name!r} must have finite low < high, got "
                f"low={spec.low}, high={spec.high}."
            )
        if spec.max_delta < 0:
            raise ValueError(
                f"Actuator {spec.name!r} must have non-negative max_delta, got "
                f"{spec.max_delta}."
            )


class _WorldModelDynamics:
    """Identity-hashable holder for the learned dynamics and weight bundle.

    Args:
      bundle: weights from :func:`plasmax.models.world_model.load_bundle` (defaults to
        the vendored npz).
    """

    def __init__(
        self,
        bundle=None,
        plasmax_config: WorldModelConfig | None = None,
        initialization: KstarInitialization | None = None,
        actuator_specs: Sequence[ActuatorSpec] | None = None,
        physics_randomization: Mapping[str, PhysicsRandomizationSpec] | None = None,
        clip_by_max_action_delta: bool = True,
    ):
        self._bundle = load_bundle() if bundle is None else bundle
        self.plasmax_config = plasmax_config
        if initialization is None:
            initialization = load_initialization(
                Path(__file__).parents[1]
                / "configs/data/initializations/kstar/nominal.yaml",
                kind="kstar",
            )
        assert isinstance(initialization, KstarInitialization)
        self._initialization = initialization
        self._initial_inputs = jnp.asarray(
            [initialization.inputs[name] for name in initialization.input_order],
            dtype=jnp.float32,
        )
        self._initial_history = jnp.broadcast_to(
            jnp.asarray(initialization.history_row, dtype=jnp.float32),
            (initialization.history_length, 21),
        )
        targets = [initialization.targets[name] for name in ("betap", "q95", "li")]
        # Preserve the original float64 uniform sampling, followed by float32.
        self._target_default = jnp.asarray(
            [t.default for t in targets], dtype=jnp.float64
        )
        self._target_min = jnp.asarray([t.minimum for t in targets], dtype=jnp.float64)
        self._target_max = jnp.asarray([t.maximum for t in targets], dtype=jnp.float64)

        if actuator_specs is None:
            if plasmax_config is not None:
                actuator_specs = [item.to_spec() for item in plasmax_config.actuators]
            else:
                actuator_specs = list(_DEFAULT_ACTUATOR_SPECS)
        _validate_actuator_specs(actuator_specs)
        self._actuator_specs = list(actuator_specs)
        self._ctrl_idx = jnp.asarray(
            [_ACTUATOR_TO_INPUT_IDX[spec.name] for spec in self._actuator_specs],
            dtype=jnp.int32,
        )
        self._action_low = jnp.asarray(
            [spec.low for spec in self._actuator_specs], dtype=jnp.float32
        )
        self._action_high = jnp.asarray(
            [spec.high for spec in self._actuator_specs], dtype=jnp.float32
        )
        self._max_action_delta = jnp.asarray(
            [spec.max_delta for spec in self._actuator_specs], dtype=jnp.float32
        )
        self._has_finite_max_delta = bool(
            np.any(np.isfinite([spec.max_delta for spec in self._actuator_specs]))
        )
        self._clip_by_max_action_delta = clip_by_max_action_delta
        self._initial_prev_action = self._build_initial_prev_action()

        if physics_randomization is None:
            physics_randomization = (
                plasmax_config.physics_randomization
                if plasmax_config is not None
                else _NO_PHYSICS_RANDOMIZATION
            )
        self._physics_specs, self._physics_nominals, self._input_param_indices = (
            self._build_physics_randomization(physics_randomization, initialization)
        )

        # Construct spaces eagerly, outside any JAX transformation.  In
        # particular, this prevents a first property access during tracing
        # from leaving cached tracer-valued bounds on the dynamics object.
        self._action_space = Continuous(low=self._action_low, high=self._action_high)
        obs_low = np.array(OBS_LOW, dtype=np.float32, copy=True)
        obs_high = np.array(OBS_HIGH, dtype=np.float32, copy=True)
        obs_index = {name: i for i, name in enumerate(OBS_NAMES)}
        for spec in self._actuator_specs:
            slot = obs_index[spec.name]
            obs_low[slot] = spec.low
            obs_high[slot] = spec.high
        obs_low[9:12] = np.asarray(self._target_min, dtype=np.float32)
        obs_high[9:12] = np.asarray(self._target_max, dtype=np.float32)
        self._observation_space = Continuous(
            low=jnp.asarray(obs_low, dtype=jnp.float32),
            high=jnp.asarray(obs_high, dtype=jnp.float32),
        )
        self._profile_obs_specs: list[ObsSpec] = []
        self._scalar_obs_specs: list[ObsSpec] = [
            ObsSpec(
                name=name,
                scale=1.0,
                bounds=(
                    (float(obs_low[i]), float(obs_high[i]))
                    if np.isfinite(obs_low[i]) and np.isfinite(obs_high[i])
                    else None
                ),
            )
            for i, name in enumerate(OBS_NAMES)
        ]

    def _build_initial_prev_action(self) -> jax.Array:
        values: list[float] = []
        for spec in self._actuator_specs:
            if spec.init is not None:
                value = float(spec.init)
            else:
                idx = _ACTUATOR_TO_INPUT_IDX[spec.name]
                value = float(self._initial_inputs[idx])
            value = max(spec.low, min(spec.high, value))
            values.append(value)
        return jnp.asarray(values, dtype=jnp.float32)

    def _build_physics_randomization(
        self,
        physics_randomization: Mapping[str, PhysicsRandomizationSpec],
        initialization: KstarInitialization,
    ) -> tuple[
        dict[str, PhysicsRandomizationSpec],
        dict[str, jax.Array],
        tuple[tuple[str, int], ...],
    ]:
        specs = dict(physics_randomization)
        if "world_model.ensemble_member" in specs and (
            "world_model.lstm_member" in specs or "world_model.bpw_member" in specs
        ):
            raise ValueError(
                "world_model.ensemble_member cannot be combined with "
                "world_model.lstm_member or world_model.bpw_member"
            )
        n_lstm = int(self._bundle["lstm"]["W0"].shape[0])
        n_bpw = int(self._bundle["bpw"]["W1"].shape[0])
        input_indices = {
            name: idx for idx, name in enumerate(initialization.input_order)
        }
        actuated_indices = set(CTRL_IDX.tolist())

        nominals: dict[str, jax.Array] = {}
        input_param_indices: list[tuple[str, int]] = []
        for path, spec in specs.items():
            if path in _ENSEMBLE_MEMBER_PATHS:
                if not spec.is_discrete:
                    raise ValueError(
                        f"physics_randomization path {path!r} requires discrete choices"
                    )
                assert spec.choices is not None
                limit = (
                    min(n_lstm, n_bpw)
                    if path == "world_model.ensemble_member"
                    else n_lstm
                    if path == "world_model.lstm_member"
                    else n_bpw
                )
                invalid = [c for c in spec.choices if c < 0 or c >= limit]
                if invalid:
                    raise ValueError(
                        f"physics_randomization path {path!r} choices {invalid} "
                        f"out of range [0, {limit})"
                    )
                nominals[path] = jnp.asarray(-1, dtype=jnp.int32)
            elif path.startswith("inputs."):
                name = path.removeprefix("inputs.")
                if name not in input_indices:
                    valid_inputs = [
                        n for n, i in input_indices.items() if i not in actuated_indices
                    ]
                    raise ValueError(
                        f"Unknown world-model input randomization path {path!r}. "
                        f"Valid unactuated inputs: {valid_inputs}"
                    )
                idx = input_indices[name]
                if idx in actuated_indices:
                    raise ValueError(
                        f"Actuated KSTAR input {path!r} cannot be physics-randomized"
                    )
                if spec.is_discrete:
                    raise ValueError(
                        f"physics_randomization path {path!r} requires a continuous "
                        "absolute or relative range"
                    )
                nominals[path] = jnp.asarray(
                    initialization.inputs[name], dtype=jnp.float32
                )
                input_param_indices.append((path, idx))
            else:
                expected = sorted(_ENSEMBLE_MEMBER_PATHS)
                raise ValueError(
                    f"Unknown world-model physics_randomization path {path!r}. "
                    f"Expected one of {expected} or 'inputs.<name>'."
                )
        return specs, nominals, tuple(input_param_indices)

    # --- spaces & metadata ---
    @property
    def action_space(self) -> Continuous:
        return self._action_space

    @property
    def observation_space(self) -> Continuous:
        return self._observation_space

    @property
    def actuator_specs(self) -> list[ActuatorSpec]:
        return self._actuator_specs

    @property
    def profile_obs_specs(self) -> list[ObsSpec]:
        return list(self._profile_obs_specs)

    @property
    def scalar_obs_specs(self) -> list[ObsSpec]:
        return list(self._scalar_obs_specs)

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
        inputs = self._initial_inputs
        if random_target:
            targets = jax.random.uniform(
                key,
                (3,),
                minval=self._target_min,
                maxval=self._target_max,
            )
        else:
            targets = self._target_default
        targets = targets.astype(jnp.float32)

        x = self._initial_history

        beta_p = predict_bpw(self._bundle, _bpw_features(x[-1, 0], inputs))[0]
        obs = self._assemble_obs(inputs, x, targets, beta_p)
        state = WorldModelEnvState(
            x=x,
            inputs=inputs,
            targets=targets,
            t=jnp.array(0, jnp.int32),
            prev_action=self._initial_prev_action,
            phys_params=dict(self._physics_nominals),
        )
        return state, _make_info(obs, jnp.zeros((), dtype=jnp.float32))

    def step(
        self, env_state: WorldModelEnvState, action: jax.Array
    ) -> tuple[WorldModelEnvState, InfoContainer]:
        action = jnp.asarray(action, dtype=jnp.float32)
        if self._clip_by_max_action_delta and self._has_finite_max_delta:
            action = jnp.clip(
                action,
                env_state.prev_action - self._max_action_delta,
                env_state.prev_action + self._max_action_delta,
            )
        action = jnp.clip(action, self._action_low, self._action_high)
        inputs = env_state.inputs.at[self._ctrl_idx].set(action)

        phys_params = env_state.phys_params
        for path, idx in self._input_param_indices:
            inputs = inputs.at[idx].set(
                jnp.asarray(phys_params[path], dtype=jnp.float32)
            )

        lstm_member = phys_params.get(
            "world_model.lstm_member",
            phys_params.get("world_model.ensemble_member"),
        )
        bpw_member = phys_params.get(
            "world_model.bpw_member",
            phys_params.get("world_model.ensemble_member"),
        )

        feat = _lstm_features(inputs)  # constant over the relaxation phase

        def relax(x, _):
            x = x.at[:-1, 4:].set(x[1:, 4:])  # shift input cols (year preserved)
            x = x.at[-1, 4:20].set(feat)
            y = predict_lstm(self._bundle, x, member=lstm_member)  # (4,)
            x = x.at[:-1, :4].set(x[1:, :4])  # shift output cols
            x = x.at[-1, :4].set(y)
            return x, None

        x, _ = jax.lax.scan(relax, env_state.x, None, length=INTERVAL - 1)

        beta_p = predict_bpw(
            self._bundle, _bpw_features(x[-1, 0], inputs), member=bpw_member
        )[0]
        obs = self._assemble_obs(inputs, x, env_state.targets, beta_p)
        reward = target_tracking_reward(obs[6:9], env_state.targets)

        t = env_state.t + 1
        new_state = WorldModelEnvState(
            x=x,
            inputs=inputs,
            targets=env_state.targets,
            t=t,
            prev_action=action,
            phys_params=phys_params,
        )
        return new_state, _make_info(obs, reward)


def _default_world_model_dynamics() -> _WorldModelDynamics:
    return _WorldModelDynamics()


class WorldModelEnv(Environment):
    """Envelope-native KSTAR fusion_lstm environment.

    The learned transition is deterministic given ``state.phys_params``.
    ``random_target`` affects typed-key initialization/reset, while outer
    wrappers own action scaling, dynamics/sensor randomization, and the
    YAML-configured KSTAR episode horizon.
    """

    random_target: bool = static_field(default=True)
    _dynamics: _WorldModelDynamics = static_field(
        default_factory=_default_world_model_dynamics, repr=False
    )

    @classmethod
    def from_bundle(
        cls,
        bundle,
        *,
        random_target: bool = True,
        actuator_specs: Sequence[ActuatorSpec] | None = None,
        physics_randomization: Mapping[str, PhysicsRandomizationSpec] | None = None,
        clip_by_max_action_delta: bool = True,
        _plasmax_config: WorldModelConfig | None = None,
        _initialization: KstarInitialization | None = None,
    ) -> Self:
        """Constructs the backend from a caller-supplied JAX weight bundle."""
        return cls(
            random_target=random_target,
            _dynamics=_WorldModelDynamics(
                bundle=bundle,
                plasmax_config=_plasmax_config,
                initialization=_initialization,
                actuator_specs=actuator_specs,
                physics_randomization=physics_randomization,
                clip_by_max_action_delta=clip_by_max_action_delta,
            ),
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

    def with_physics(
        self,
        state: WorldModelEnvState | WrappedState,
        parameters: dict[str, jax.Array],
    ) -> WorldModelEnvState | WrappedState:
        """Immutably update persistent parameters through any wrapper state."""
        if isinstance(state, WrappedState):
            return dataclasses.replace(
                state, inner_state=self.with_physics(state.inner_state, parameters)
            )
        return dataclasses.replace(
            state, phys_params={**state.phys_params, **parameters}
        )

    @property
    def physics_randomization(self) -> dict[str, PhysicsRandomizationSpec]:
        return self._dynamics._physics_specs

    @property
    def physics_nominals(self) -> dict[str, jax.Array]:
        return self._dynamics._physics_nominals

    @cached_property
    def action_space(self) -> Continuous:
        return self._dynamics.action_space

    @cached_property
    def observation_space(self) -> Continuous:
        return self._dynamics.observation_space

    @property
    def actuator_specs(self) -> list[ActuatorSpec]:
        return self._dynamics.actuator_specs

    @property
    def profile_obs_specs(self) -> list[ObsSpec]:
        return self._dynamics.profile_obs_specs

    @property
    def scalar_obs_specs(self) -> list[ObsSpec]:
        return self._dynamics.scalar_obs_specs

    def obs_layout(self) -> ObsLayout:
        return self._dynamics.obs_layout()

    @property
    def plasmax_config(self) -> WorldModelConfig | None:
        return self._dynamics.plasmax_config

    @property
    def safe_max_steps(self) -> int | None:
        if self.plasmax_config is None:
            return None
        return self.plasmax_config.world_model.max_steps_in_episode
