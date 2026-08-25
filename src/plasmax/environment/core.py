"""Barebones TORAX RL environment."""

import dataclasses
from collections.abc import Callable, Sequence
from functools import cached_property
from typing import Any, Self

import jax
import numpy as np
from envelope import Continuous, Environment, Info, InfoContainer, static_field
from jax import numpy as jnp
from torax._src import jax_utils
from torax._src import state as torax_state
from torax._src.config import build_runtime_params
from torax._src.orchestration import initial_state as initial_state_lib
from torax._src.orchestration import run_simulation
from torax._src.orchestration import sim_state as sim_state_lib
from torax._src.orchestration.step_function import SimulationStepFn
from torax._src.output_tools import post_processing
from torax._src.physics import formulas
from torax._src.torax_pydantic import interpolated_param_1d, model_config

from plasmax.control import ControlInputs, _ControlInputsApplier, _PhysicsParamsApplier
from plasmax.environment import initialization as initialization_lib
from plasmax.environment import schema as scenario_models
from plasmax.environment import validation as environment_validation
from plasmax.environment.stepping import fixed_duration_step
from plasmax.spaces import (
    PROFILE_REGISTRY,
    SCALAR_REGISTRY,
    ActuatorSpec,
    ObsLayout,
    ObsSpec,
    build_obs_bounds,
    extract_obs,
)

_NO_STATE_NOISE: dict[str, float] = {}
_NO_PHYSICS_RANDOMIZATION: dict[str, scenario_models.PhysicsRandomizationSpec] = {}


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class PlasmaState:
    """Stable plasmax view over TORAX's simulation state."""

    sim: sim_state_lib.SimState
    post: post_processing.PostProcessedOutputs

    @property
    def t(self) -> jax.Array:
        return self.sim.t

    @property
    def Ip(self) -> jax.Array:
        """Total plasma current at the LCFS [A]."""
        return self.core.Ip_profile_face[-1]

    @property
    def core(self):
        return self.sim.core_profiles

    @property
    def geo(self):
        return self.sim.geometry

    @property
    def mode(self) -> jax.Array:
        return self.sim.pedestal_transition_state.confinement_mode

    @property
    def T_e(self) -> jax.Array:
        return self.core.T_e.value

    @property
    def T_i(self) -> jax.Array:
        return self.core.T_i.value

    @property
    def n_e(self) -> jax.Array:
        return self.core.n_e.value

    @property
    def psi(self) -> jax.Array:
        return self.core.psi.value

    @property
    def q(self) -> jax.Array:
        q_face = self.core.q_face
        return 0.5 * (q_face[:-1] + q_face[1:])

    def __getattr__(self, name: str):
        """Expose post-processed scalars without leaking ``post`` to consumers."""
        try:
            post = object.__getattribute__(self, "post")
            return getattr(post, name)
        except AttributeError as exc:
            raise AttributeError(name) from exc


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class EnvState:
    """Carries all mutable state for a PlasmaxEnv episode.

    Attributes:
      plasma: Stable view over the current TORAX state and derived outputs.
      prev_action: Last applied action vector, used for rate limiting.
      phys_params: Physics runtime-param overrides used for the most recent
        transition, keyed by provider dot-path (empty when disabled; nominal
        values immediately after reset).
      key: Typed PRNG key used for the next stochastic transition.
    """

    plasma: PlasmaState
    prev_action: jax.Array
    phys_params: dict[str, jax.Array]
    key: jax.Array


def _scale_cell_variable(cell_var, multiplier: jax.Array):
    """Scales a TORAX CellVariable value and its right-face value consistently."""
    right_face_constraint = cell_var.right_face_constraint
    if right_face_constraint is not None:
        right_face_constraint = right_face_constraint * multiplier[-1]
    return dataclasses.replace(
        cell_var,
        value=cell_var.value * multiplier,
        right_face_constraint=right_face_constraint,
    )


def _with_initial_internal_energy(core_profiles, geo):
    """Refreshes reset-time thermal energy after direct profile perturbations."""
    W_thermal_e, W_thermal_i, W_thermal_total = (
        formulas.calculate_stored_thermal_energy(
            core_profiles.pressure_thermal_e,
            core_profiles.pressure_thermal_i,
            core_profiles.pressure_thermal_total,
            geo,
        )
    )
    zero = jnp.array(0.0, dtype=jax_utils.get_dtype())
    return dataclasses.replace(
        core_profiles,
        internal_plasma_energy=torax_state.PlasmaInternalEnergy(
            W_thermal_i=W_thermal_i,
            W_thermal_e=W_thermal_e,
            W_thermal_total=W_thermal_total,
            dW_thermal_i_dt=zero,
            dW_thermal_e_dt=zero,
            dW_thermal_i_dt_smoothed=zero,
            dW_thermal_e_dt_smoothed=zero,
        ),
    )


def _validate_typed_key(key: jax.Array) -> None:
    """Rejects legacy uint32 PRNG keys at the public lifecycle boundary."""
    dtype = getattr(key, "dtype", None)
    shape = getattr(key, "shape", None)
    if dtype is None or shape != () or not jnp.issubdtype(dtype, jax.dtypes.prng_key):
        raise ValueError("key must be a scalar typed (new-style) jax.random.key")


def _make_info(
    obs: jax.Array,
    reward: jax.Array,
    terminated: jax.Array | bool,
    termination_code: jax.Array | int,
    *,
    internal_steps: jax.Array | int = 0,
    sawtooth_crashes: jax.Array | int = 0,
    control_step_complete: jax.Array | bool = False,
    step_limit_reached: jax.Array | bool = False,
) -> InfoContainer:
    """Builds the structurally stable emission shared by every lifecycle path."""
    return InfoContainer(
        obs=obs,
        reward=reward,
        terminated=jnp.asarray(terminated, dtype=jnp.bool_),
        truncated=jnp.asarray(False, dtype=jnp.bool_),
    ).update(
        termination_code=jnp.asarray(termination_code, dtype=jnp.int32),
        internal_steps=jnp.asarray(internal_steps, dtype=jnp.int32),
        sawtooth_crashes=jnp.asarray(sawtooth_crashes, dtype=jnp.int32),
        control_step_complete=jnp.asarray(control_step_complete, dtype=jnp.bool_),
        step_limit_reached=jnp.asarray(step_limit_reached, dtype=jnp.bool_),
    )


def _derive_safe_max_steps(config: model_config.ToraxConfig) -> int:
    """Counts physical control intervals from ``t_initial`` through ``t_final``.

    The unrandomized ``fixed_dt`` schedule defines those intervals. This
    mirrors the fixed calculator's exact-final clamp and completion tolerance
    and is metadata for an outer truncation wrapper.
    """
    calculator_type = config.time_step_calculator.calculator_type
    calculator_name = getattr(calculator_type, "value", calculator_type)
    if calculator_name != "fixed":
        raise ValueError(
            "PlasmaxEnv requires a fixed time-step calculator to derive its safe "
            f"horizon, got {calculator_name!r}."
        )

    t = float(config.numerics.t_initial)
    t_final = float(config.numerics.t_final)
    tolerance = float(config.time_step_calculator.tolerance)
    exact_t_final = bool(config.numerics.exact_t_final)
    steps = 0

    while t < t_final - tolerance:
        dt = float(config.numerics.fixed_dt.get_value(t))
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError(
                "Cannot derive a safe TORAX horizon: fixed_dt must remain "
                f"positive and finite, got {dt} at t={t}."
            )
        crosses_t_final = t < t_final and t + dt > t_final
        next_t = t_final if exact_t_final and crosses_t_final else t + dt
        if next_t <= t:
            raise ValueError(
                "Cannot derive a safe TORAX horizon: fixed_dt does not advance "
                f"time at t={t} (dt={dt})."
            )
        t = next_t
        steps += 1

    return steps


def _torax_state_is_finite(
    sim_state: sim_state_lib.SimState,
    postout: post_processing.PostProcessedOutputs,
) -> jax.Array:
    """JAX-compatible finite-state gate for each accepted internal step."""
    core = sim_state.core_profiles
    return (
        jnp.isfinite(sim_state.t)
        & jnp.isfinite(sim_state.dt)
        & jnp.all(jnp.isfinite(core.T_e.value))
        & jnp.all(jnp.isfinite(core.T_i.value))
        & jnp.all(jnp.isfinite(core.n_e.value))
        & jnp.all(jnp.isfinite(core.psi.value))
        & jnp.isfinite(postout.P_fusion)
        & jnp.isfinite(postout.q_min)
    )


class _ToraxDynamics:
    """Identity-hashable private holder for configured TORAX dynamics.

    Keeping TORAX/Pydantic objects and callables behind one ordinary Python
    object prevents array-valued equality or unhashable mappings from becoming
    Envelope pytree metadata. The public :class:`PlasmaxEnv` is a frozen facade.
    """

    def __init__(
        self,
        config: model_config.ToraxConfig,
        actuator_specs: Sequence[ActuatorSpec],
        reward_fn: Callable[[jax.Array, EnvState, jax.Array, EnvState], jax.Array],
        disruption_penalty: float = 0.0,
        clip_by_max_action_delta: bool = True,
        disruption: scenario_models.DisruptionConfig | None = None,
        obs_fn: (
            Callable[
                [PlasmaState],
                jax.Array,
            ]
            | None
        ) = None,
        state_noise_config: dict[str, float] = _NO_STATE_NOISE,
        physics_randomization: dict[
            str, scenario_models.PhysicsRandomizationSpec
        ] = _NO_PHYSICS_RANDOMIZATION,
        stepping: scenario_models.SteppingConfig | None = None,
        initialization: initialization_lib.PhaseSnapshot | None = None,
        *,
        profile_obs_specs: Sequence[ObsSpec],
        scalar_obs_specs: Sequence[ObsSpec],
    ):
        """Builds an environment from validated TORAX and RL configuration.

        Actuator specs define action order, bounds, and rate limits. Physics
        randomization supports scalar runtime-parameter paths only.
        """
        self._config = config
        environment_validation.validate_observation_specs(
            profile_obs_specs,
            scalar_obs_specs,
            PROFILE_REGISTRY,
            SCALAR_REGISTRY,
        )
        environment_validation.validate_state_noise_config(state_noise_config)

        actuators = [s.name for s in actuator_specs]
        self._actuator_specs = list(actuator_specs)
        self._actuators = actuators
        self._action_low = jnp.array([s.low for s in actuator_specs])
        self._action_high = jnp.array([s.high for s in actuator_specs])
        self._reward_fn = reward_fn
        self._max_action_delta = jnp.array([s.max_delta for s in actuator_specs])
        self._clip_by_max_action_delta = clip_by_max_action_delta
        self._disruption_penalty = jnp.asarray(disruption_penalty)
        self._disruption_cfg = disruption or scenario_models.DisruptionConfig()
        self._stepping = stepping or scenario_models.SteppingConfig()

        self._profile_obs_specs = tuple(profile_obs_specs)
        self._scalar_obs_specs = tuple(scalar_obs_specs)

        self._uses_custom_obs = obs_fn is not None
        if obs_fn is None:
            self._raw_obs_fn = lambda plasma: extract_obs(
                plasma, self._profile_obs_specs, self._scalar_obs_specs
            )
        else:
            self._raw_obs_fn = obs_fn

        self.safe_max_steps = _derive_safe_max_steps(config)

        self._step_fn = run_simulation.make_step_fn(config)
        provider = self._step_fn.runtime_params_provider
        environment_validation.validate_actuator_specs(actuator_specs, provider)
        environment_validation.validate_physics_randomization(
            physics_randomization, provider
        )
        if "numerics.fixed_dt" in physics_randomization:
            raise ValueError(
                "numerics.fixed_dt defines the physical control interval and "
                "cannot be physics-randomized"
            )
        self._applier = _ControlInputsApplier(self._actuators)
        # Physics-randomization paths. Set before _compute_reset, which seeds
        # EnvState.phys_params with nominal values for a stable pytree shape.
        self._physics_specs = dict(physics_randomization)
        self._phys_paths = list(self._physics_specs)

        def _nominal_value(path: str) -> jax.Array:
            node = provider.get_node_from_path(path)
            if isinstance(node, interpolated_param_1d.TimeVaryingScalar):
                node = node.get_value(0.0)
            return jnp.asarray(node)

        self._physics_nominals = {
            path: _nominal_value(path) for path in self._phys_paths
        }
        if self._phys_paths:
            self._physics_nominal_vector = jnp.stack(
                tuple(self._physics_nominals[path] for path in self._phys_paths)
            )
            physics_dtype = self._physics_nominal_vector.dtype
            self._physics_lows = jnp.asarray(
                tuple(self._physics_specs[path].bounds[0] for path in self._phys_paths),
                dtype=physics_dtype,
            )
            self._physics_highs = jnp.asarray(
                tuple(self._physics_specs[path].bounds[1] for path in self._phys_paths),
                dtype=physics_dtype,
            )
            self._physics_is_relative = jnp.asarray(
                tuple(
                    self._physics_specs[path].is_relative for path in self._phys_paths
                )
            )
        else:
            self._physics_nominal_vector = jnp.empty((0,))
            self._physics_lows = jnp.empty((0,))
            self._physics_highs = jnp.empty((0,))
            self._physics_is_relative = jnp.empty((0,), dtype=jnp.bool_)
        self._phys_applier = _PhysicsParamsApplier(self._phys_paths)

        self._state_noise_config: dict[str, float] = state_noise_config
        self._state_noise_names = tuple(self._state_noise_config)
        self._state_noise_scales = jnp.asarray(tuple(self._state_noise_config.values()))

        self._action_space = Continuous(
            low=jnp.asarray(self._action_low, dtype=jnp.float32),
            high=jnp.asarray(self._action_high, dtype=jnp.float32),
        )

        # prev_action used as the rate-limit anchor at reset. We resolve in
        # priority order: explicit ActuatorSpec.init → scenario source default
        # at t=0 via the same dot-path the action override uses → midpoint of
        # [low, high] as a last-resort fallback. Midpoint is unsafe when the
        # actuator range spans many orders of magnitude (e.g. pellet_rate
        # 1e20–5e22), where any reasonable max_delta is dwarfed by the gap
        # between midpoint and the design operating point.
        initial_prev_action = self._build_initial_prev_action(self._step_fn)
        self._initial_obs, self._initial_env_state = self._compute_reset(
            self._step_fn,
            initial_prev_action,
            initialization,
        )
        self._obs_shape = self._initial_obs.shape
        # Keep TORAX physics in its configured precision while exposing a
        # stable float32 reward boundary to RL algorithms and scan carries.
        self._reward_dtype = jnp.float32
        self._disruption_penalty = jnp.asarray(
            self._disruption_penalty, dtype=self._reward_dtype
        )

        obs_size = self._initial_obs.shape[0]

        # Reset-time state noise recomputes post-processing so scalar obs and
        # disruption checks stay consistent with the perturbed profiles. That
        # needs the t_initial runtime params; resolve them once here
        # (only when noise is enabled — the default {} path stays untouched).
        self._reset_runtime_params = None
        if self._state_noise_config:
            self._reset_runtime_params = (
                build_runtime_params.get_consistent_runtime_params_and_geometry(
                    t=provider.numerics.t_initial,
                    runtime_params_provider=provider,
                    geometry_provider=self._step_fn.geometry_provider,
                    is_initialization=True,
                )[0]
            )

        if self._uses_custom_obs:
            self._obs_layout = ObsLayout(
                profile_slices={},
                scalar_slices={},
                profile_names=(),
                scalar_names=(),
                vector_size=obs_size,
            )
            obs_low = np.full(obs_size, -np.inf, dtype=np.float32)
            obs_high = np.full(obs_size, np.inf, dtype=np.float32)
        else:
            n_rho = environment_validation.config_n_rho(config)
            self._obs_layout = ObsLayout.from_specs(
                self._profile_obs_specs, self._scalar_obs_specs, n_rho
            )
            obs_low, obs_high = build_obs_bounds(
                obs_size,
                self._obs_layout,
                self._profile_obs_specs,
                self._scalar_obs_specs,
            )
        self._observation_space = Continuous(
            low=jnp.asarray(obs_low, dtype=self._initial_obs.dtype),
            high=jnp.asarray(obs_high, dtype=self._initial_obs.dtype),
        )

    def _observe(self, plasma: PlasmaState) -> jax.Array:
        """Evaluates and validates the public flat floating observation."""
        obs = jnp.asarray(self._raw_obs_fn(plasma))
        if obs.ndim != 1:
            raise ValueError(
                "obs_fn must return a flat, one-dimensional array; "
                f"got shape {obs.shape}."
            )
        if not jnp.issubdtype(obs.dtype, jnp.floating):
            raise ValueError(
                f"obs_fn must return a floating array; got dtype {obs.dtype}."
            )
        expected_shape = getattr(self, "_obs_shape", None)
        if expected_shape is not None and obs.shape != expected_shape:
            raise ValueError(
                "obs_fn output shape must remain fixed across the lifecycle: "
                f"expected {expected_shape}, got {obs.shape}."
            )
        return obs

    def _build_initial_prev_action(self, step_fn: SimulationStepFn) -> jax.Array:
        """Returns the prev_action vector to use at reset for this config."""
        from plasmax.control import _FIELD_TO_PATH  # local import to avoid cycle

        provider = step_fn.runtime_params_provider
        values: list[float] = []
        for spec in self._actuator_specs:
            if spec.init is not None:
                v = float(spec.init)
            else:
                path = _FIELD_TO_PATH.get(spec.name)
                v = None
                if path is not None:
                    try:
                        node = provider.get_node_from_path(path)
                        v = float(node.get_value(0.0))
                    except (ValueError, AttributeError):
                        v = None
                if v is None:
                    v = 0.5 * (spec.low + spec.high)
            # Clamp into the actuator's declared physical range.
            v = max(spec.low, min(spec.high, v))
            values.append(v)
        return jnp.array(values)

    def _compute_reset(
        self,
        step_fn: SimulationStepFn,
        prev_action: jax.Array,
        initialization: initialization_lib.PhaseSnapshot | None,
    ) -> tuple[jax.Array, EnvState]:
        if initialization is None:
            sim_state, postout = (
                initial_state_lib.get_initial_state_and_post_processed_outputs(step_fn)
            )
        else:
            sim_state, postout = initialization_lib.materialize_snapshot(
                initialization,
                step_fn=step_fn,
            )
        # Seed phys_params with each path's nominal value at t=0, so reset and
        # transition states have the same pytree structure.
        phys_params = dict(self._physics_nominals)
        env_state = EnvState(
            plasma=PlasmaState(sim=sim_state, post=postout),
            prev_action=prev_action,
            phys_params=phys_params,
            key=jax.random.key(0),
        )
        obs = self._observe(env_state.plasma)
        return obs, env_state

    def init(self, key: jax.Array) -> tuple[EnvState, InfoContainer]:
        """Initializes physical state and independent reset/step RNG streams."""
        _validate_typed_key(key)
        step_key, noise_key = jax.random.split(key)
        env_state = dataclasses.replace(self._initial_env_state, key=step_key)

        # Reset-time state noise. Recomputes core_profiles internal energy and
        # post-processing from the perturbed profiles so profile obs, scalar obs,
        # and disruption checks all read the same (noisy) state. Gated on a
        # non-empty config so the default physical state stays byte-identical.
        if self._state_noise_config:
            env_state = self._apply_state_noise(
                noise_key, env_state, self._reset_runtime_params
            )

        obs = self._observe(env_state.plasma)
        reward = jnp.zeros((), dtype=self._reward_dtype)
        return env_state, _make_info(obs, reward, False, -1)

    def _sample_physics_params(self, key: jax.Array) -> dict[str, jax.Array]:
        """Samples the runtime overrides for one transition."""
        phys_keys = jax.random.split(key, len(self._physics_specs))

        def sample_one(sample_key, low, high):
            return jax.random.uniform(
                sample_key,
                (),
                minval=low,
                maxval=high,
                dtype=self._physics_nominal_vector.dtype,
            )

        samples = jax.vmap(sample_one)(
            phys_keys, self._physics_lows, self._physics_highs
        )
        samples = jnp.where(
            self._physics_is_relative,
            self._physics_nominal_vector * samples,
            samples,
        )
        return dict(zip(self._phys_paths, samples, strict=True))

    def _apply_state_noise(self, key, env_state: EnvState, runtime_params) -> EnvState:
        """Perturbs reset profiles and rebuilds the dependent (sim_state, postout).

        Each configured field's CellVariable value (and its boundary
        ``right_face_constraint``) is scaled by ``1 + scale * N(0, 1)``. Thermal
        pressures are ``cached_property``s and refresh automatically; the stored
        ``internal_plasma_energy`` and ``postout`` are recomputed explicitly so
        every consumer reads the perturbed state. Quasineutral partner densities
        (``n_i``/``n_impurity``) are *not* re-derived, so large ``n_e`` noise
        leaves a small quasineutrality residual — fine for the few-percent
        domain randomisation these configs use.
        """
        sim_state = env_state.plasma.sim
        cp = sim_state.core_profiles
        geo = sim_state.geometry
        cell_vars = tuple(getattr(cp, name) for name in self._state_noise_names)
        batched_cell_vars = jax.tree.map(lambda *leaves: jnp.stack(leaves), *cell_vars)
        noise_keys = jax.random.split(key, len(self._state_noise_names))

        def sample_multiplier(noise_key, scale):
            noise = jax.random.normal(noise_key, batched_cell_vars.value.shape[1:])
            return 1.0 + scale * noise

        multipliers = jax.vmap(sample_multiplier)(noise_keys, self._state_noise_scales)
        scaled_cell_vars = jax.vmap(_scale_cell_variable)(
            batched_cell_vars, multipliers
        )
        updates = {
            name: jax.tree.map(lambda leaf, i=i: leaf[i], scaled_cell_vars)
            for i, name in enumerate(self._state_noise_names)
        }
        new_cp = _with_initial_internal_energy(dataclasses.replace(cp, **updates), geo)
        new_sim = dataclasses.replace(sim_state, core_profiles=new_cp)
        new_postout = post_processing.make_post_processed_outputs(
            sim_state=new_sim,
            runtime_params=runtime_params,
            previous_post_processed_outputs=(
                post_processing.PostProcessedOutputs.zeros(geo)
            ),
        )
        return dataclasses.replace(
            env_state, plasma=PlasmaState(sim=new_sim, post=new_postout)
        )

    def _control_dt(self, t: jax.Array) -> jax.Array:
        """Resolves the unrandomized physical control interval at ``t``."""
        numerics = self._step_fn.runtime_params_provider.numerics
        dt = jnp.asarray(numerics.fixed_dt.get_value(t))
        t_final = jnp.asarray(numerics.t_final, dtype=dt.dtype)
        crosses_t_final = (t < t_final) & (t + dt > t_final)
        if numerics.exact_t_final:
            dt = jnp.where(crosses_t_final, t_final - t, dt)
        return dt

    def step(
        self,
        env_state: EnvState,
        action: jax.Array,
    ) -> tuple[EnvState, InfoContainer]:
        """Advances one simulation step without owning a time horizon."""
        next_key, physics_key = jax.random.split(env_state.key)
        if self._clip_by_max_action_delta:
            action = jnp.clip(
                action,
                env_state.prev_action - self._max_action_delta,
                env_state.prev_action + self._max_action_delta,
            )
        action = jnp.clip(action, self._action_low, self._action_high)

        kwargs = {name: action[i] for i, name in enumerate(self._actuators)}
        control_inputs = ControlInputs(**kwargs)

        phys_params = (
            self._sample_physics_params(physics_key)
            if self._physics_specs
            else env_state.phys_params
        )

        provider = self._phys_applier(
            phys_params,
            self._applier(control_inputs, self._step_fn.runtime_params_provider),
        )
        control_dt = self._control_dt(env_state.plasma.t)
        step_result = fixed_duration_step(
            self._step_fn,
            _torax_state_is_finite,
            self._stepping.max_solver_substeps,
            self._stepping.max_event_substeps,
            control_dt,
            env_state.plasma.sim,
            env_state.plasma.post,
            provider,
        )

        new_env_state = EnvState(
            plasma=PlasmaState(
                sim=step_result.sim_state,
                post=step_result.post_processed_outputs,
            ),
            prev_action=action,
            phys_params=phys_params,
            key=next_key,
        )
        obs = self._observe(new_env_state.plasma)

        q_min_disruption, greenwald_exceeded, solver_failure = self._disruption_terms(
            new_env_state.plasma, obs
        )
        disruption = q_min_disruption | greenwald_exceeded | solver_failure
        reward = jnp.asarray(
            self._reward_fn(env_state.prev_action, env_state, action, new_env_state),
            dtype=self._reward_dtype,
        )
        # On a disrupting/unphysical step the reward is the terminal penalty: the
        # episode ends here, and reward_fn may itself be NaN if the state
        # diverged, so select the penalty rather than propagate it.
        reward = jnp.where(disruption, self._disruption_penalty, reward)

        # Termination code priority is solver > q_min > greenwald. Time limits
        # are exclusively the responsibility of PlasmaxTruncationWrapper.
        termination_code = jnp.where(
            solver_failure,
            jnp.int32(3),
            jnp.where(
                q_min_disruption,
                jnp.int32(1),
                jnp.where(greenwald_exceeded, jnp.int32(2), jnp.int32(-1)),
            ),
        )

        return new_env_state, _make_info(
            obs,
            reward,
            disruption,
            termination_code,
            internal_steps=step_result.internal_steps,
            sawtooth_crashes=step_result.sawtooth_crashes,
            control_step_complete=step_result.control_step_complete,
            step_limit_reached=step_result.step_limit_reached,
        )

    def _disruption_terms(
        self, plasma: PlasmaState, obs: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Returns the ``(q_min, greenwald, solver_failure)`` termination booleans.

        The internal macro-step already checks every evolved core profile plus
        its critical post-processed outputs after each accepted solve. This
        final gate adds the public observation and selected Greenwald metric,
        and recognizes the macro-step's aggregated failure status. The
        implicit solve can diverge when actuators drive the plasma into a state
        the model cannot resolve; that is terminal and unrecoverable.
        """
        cfg = self._disruption_cfg
        fgw = getattr(plasma, cfg.greenwald_field)
        q_min_disruption = plasma.q_min < cfg.q_min_threshold
        greenwald_exceeded = fgw > cfg.greenwald_threshold
        solver_failure = (
            ~jnp.all(jnp.isfinite(obs))
            | ~jnp.isfinite(fgw)
            | (plasma.sim.solver_numeric_outputs.solver_error_state == 1)
        )
        return q_min_disruption, greenwald_exceeded, solver_failure

    @property
    def actuator_specs(self) -> list[ActuatorSpec]:
        return self._actuator_specs

    @property
    def profile_obs_specs(self) -> list[ObsSpec]:
        """The profile observation specs (name + scale + bounds) this env uses."""
        return list(self._profile_obs_specs)

    @property
    def scalar_obs_specs(self) -> list[ObsSpec]:
        """The scalar observation specs (name + scale + bounds) this env uses."""
        return list(self._scalar_obs_specs)

    @property
    def config(self) -> model_config.ToraxConfig:
        """The ToraxConfig backing this env."""
        return self._config

    @property
    def action_space(self) -> Continuous:
        return self._action_space

    @property
    def observation_space(self) -> Continuous:
        return self._observation_space

    def obs_layout(self) -> ObsLayout:
        """Returns the slice map describing where each named obs lives."""
        return self._obs_layout


class PlasmaxEnv(Environment):
    """Envelope-native frozen facade over configured TORAX dynamics.

    The base environment owns physical disruptions only. Episode horizons,
    autoreset, normalization, and vectorization are supplied by Envelope
    wrappers. Stochastic transition state is carried in :class:`EnvState`.
    """

    _dynamics: _ToraxDynamics = static_field(repr=False)

    @classmethod
    def _from_config(
        cls,
        *,
        initialization: initialization_lib.PhaseSnapshot | None,
        **dynamics_kwargs: Any,
    ) -> Self:
        """Constructs the environment with an optional private phase reset."""
        return cls(
            _dynamics=_ToraxDynamics(
                initialization=initialization,
                **dynamics_kwargs,
            )
        )

    @classmethod
    def from_config(
        cls,
        config: model_config.ToraxConfig,
        actuator_specs: Sequence[ActuatorSpec],
        reward_fn: Callable[[jax.Array, EnvState, jax.Array, EnvState], jax.Array],
        disruption_penalty: float = 0.0,
        clip_by_max_action_delta: bool = True,
        disruption: scenario_models.DisruptionConfig | None = None,
        obs_fn: Callable[[PlasmaState], jax.Array] | None = None,
        state_noise_config: dict[str, float] = _NO_STATE_NOISE,
        physics_randomization: dict[
            str, scenario_models.PhysicsRandomizationSpec
        ] = _NO_PHYSICS_RANDOMIZATION,
        stepping: scenario_models.SteppingConfig | None = None,
        *,
        profile_obs_specs: Sequence[ObsSpec],
        scalar_obs_specs: Sequence[ObsSpec],
    ) -> Self:
        """Constructs and validates the private TORAX transition component."""
        return cls._from_config(
            config=config,
            actuator_specs=actuator_specs,
            reward_fn=reward_fn,
            disruption_penalty=disruption_penalty,
            clip_by_max_action_delta=clip_by_max_action_delta,
            disruption=disruption,
            obs_fn=obs_fn,
            state_noise_config=state_noise_config,
            physics_randomization=physics_randomization,
            stepping=stepping,
            initialization=None,
            profile_obs_specs=profile_obs_specs,
            scalar_obs_specs=scalar_obs_specs,
        )

    def init(self, key: jax.Array) -> tuple[EnvState, Info]:
        return self._dynamics.init(key)

    def reset(self, state: EnvState, key: jax.Array) -> tuple[EnvState, Info]:
        del state
        return self._dynamics.init(key)

    def step(self, state: EnvState, action: jax.Array) -> tuple[EnvState, Info]:
        return self._dynamics.step(state, action)

    @cached_property
    def action_space(self) -> Continuous:
        return self._dynamics.action_space

    @cached_property
    def observation_space(self) -> Continuous:
        return self._dynamics.observation_space

    @property
    def safe_max_steps(self) -> int:
        """Largest configured number of valid TORAX transitions."""
        return self._dynamics.safe_max_steps

    @property
    def actuator_specs(self) -> list[ActuatorSpec]:
        return self._dynamics.actuator_specs

    @property
    def profile_obs_specs(self) -> list[ObsSpec]:
        return self._dynamics.profile_obs_specs

    @property
    def scalar_obs_specs(self) -> list[ObsSpec]:
        return self._dynamics.scalar_obs_specs

    @property
    def config(self) -> model_config.ToraxConfig:
        return self._dynamics.config

    def obs_layout(self) -> ObsLayout:
        return self._dynamics.obs_layout()
