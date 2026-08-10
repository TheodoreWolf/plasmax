# Backends

A **backend** is the simulator engine: it defines the transport model, the
solver, the neoclassical model, and the radiation sources. It is deep-merged
with an **env** (the RL task — actuators, observations, geometry, scenario) at
load time via `make(env_path, backend_path)`; the backend supplies
defaults and the env wins on any leaf it sets explicitly. Not every env pairs
with every backend — the allowed pairs are enforced by
`plasmax.environment.merge._VALID_ENV_BACKEND_COMBOS`
(`valid_env_backend_combos()`).

Every backend but `fusion_lstm` is a real TORAX config. Each file's header
comment documents its full parameter set and rationale; this README covers only
the **differences** between them.

## Current default

**Bohm–GyroBohm (`bohm_gyrobohm.yaml`) is the default backend for training and
rollout for now.** It is fast, differentiable, robust under exploratory policies,
and available for ITER, SPARC, and STEP. It is a controlled baseline rather than
a claim that Bohm–GyroBohm is the highest-fidelity transport model. Reported
results should be cross-checked on QLKNN and TGLFNN, and selected reference cases
should be checked with the nonlinear solver or an offline higher-fidelity model.

ITER and SPARC environment aliases always include an explicit `rampup`,
`flattop`, or `rampdown` phase. A different transport model is always an
explicit backend choice; models in the same category are alternatives, not
effects that are added together.

Upstream TORAX reference examples live at
https://github.com/google-deepmind/torax/tree/main/torax/examples. The most
relevant files for these backends are `iterhybrid_predictor_corrector.py`,
`iterhybrid_rampup.py`, and `step_flattop_bgb.py`.

## Two axes: transport model × solver

A TORAX backend is essentially a choice on two independent axes.

### Transport model — how turbulent heat/particle fluxes are computed

| Model | Type | Physics | Notes |
|-------|------|---------|-------|
| **CGM** | analytic, theory-based | ITG critical-gradient (Guo–Romanelli); gyro-Bohm scaling with a critical-threshold nonlinearity | Fast, differentiable, no out-of-distribution (OOD) risk — a useful analytic cross-check on the default. |
| **QLKNN** | ML surrogate | NN (`qlknn_7_11_v1`) trained on QuaLiKiz; ITG + TEM + ETG modes | High fidelity but valid only inside its training range; an exploring policy can drive it OOD, so inner/outer patches + chi/D/V clipping guard against runaway fluxes. |
| **TGLFNN-UKAEA** | ML surrogate | NN surrogate of TGLF (Trapped-Gyro-Landau-Fluid) | Independent high-fidelity check on the QuaLiKiz-lineage QLKNN. Conventional and STEP-trained weights are selected by explicit backend files. |
| **Bohm–GyroBohm (BgB)** | analytic, semi-empirical | Bohm + gyro-Bohm turbulent transport | **Current default.** Geometry-agnostic TORAX model; STEP's OpenSTEP-calibrated coefficients + pedestal live env-side in `envs/step.yaml`. |

All ITER/SPARC backends use **Redl bootstrap + Angioni–Sauter** neoclassical
transport (Sauter-family analytic, differentiable), except `cgm` which uses a
simple `bootstrap_multiplier`.

### Solver — how the coupled transport PDEs are advanced each step

| Solver | How | Cost | When |
|--------|-----|------|------|
| **linear** (theta-method + predictor–corrector) | Fixed-point: freeze coefficients at the previous iterate, solve the linear system, repeat `n_corrector_steps` times | Cheap per step; vectorises cleanly under `vmap` | Default for RL training/rollout. Lower accuracy over stiff transients. |
| **newton_raphson** | Gradient-based; JAX auto-diffs the Jacobian, iterates residual → 0, with line search + adaptive `dt` fallback | Model-dependent; can exceed 100× linear cost when differentiating an NN transport model on CPU. **Serialises poorly under `vmap`** (every vmapped env blocks on the slowest one's Newton iterations). | Highly nonlinear/stiff cases where linear convergence is inadequate. |
| **optimizer** | Recasts the PDE residual as a loss, minimised via jaxopt | Similar to NR | TORAX flags it "relatively untested" — not used by any backend here. |

`solver_type` accepts `linear`, `newton_raphson`, or `optimizer`. Training
backends use the linear solver with predictor-corrector iterations;
`tglfnn_nr.yaml` is the packaged nonlinear reference for fidelity checks.

### Fixed-duration control steps

`torax.numerics.fixed_dt` is the external RL control interval, not a promise
that TORAX will accept one PDE solve of that size. A transition holds its
action and randomized runtime provider fixed while completing the interval
with bounded internal solves. `stepping.max_solver_substeps` is normally
backend-owned (`1` by default and `32` for `tglfnn_nr`), while a scenario with
sawtooth MHD sets `stepping.max_event_substeps: 1`; the two blocks deep-merge.
Both limits are static and changing either recompiles the JAX transition.

Transition info exposes `internal_steps`, `sawtooth_crashes`,
`control_step_complete`, and `step_limit_reached`. Exhaustion or a failed solve
terminates with code `3` at its actual partial time; a successful transition
lands exactly on the configured control grid.

All solvers use **Pereverzev–Corrigan artificial diffusion**
(`use_pereverzev`, `chi_pereverzev`, `D_pereverzev`): a large artificial
diffusion term balanced by an inward convection term so that *zero* net
transport is added at time *t*. It stabilises stiff turbulent transport (QLKNN,
TGLFNN, CGM) at the cost of accuracy over short transients.

## The backends

| File | Transport | Solver | Env family | One-liner |
|------|-----------|--------|-----------|-----------|
| `cgm.yaml` | CGM | linear | ITER/SPARC | Fast analytic alternative and robustness cross-check (~5 s compile, ~15 ms/step). |
| `qlknn.yaml` | QLKNN | linear | ITER/SPARC | ML-fidelity transport, cheap solver — preferred for vectorised QLKNN rollouts/training. |
| `bohm_gyrobohm.yaml` | Bohm–GyroBohm | linear | ITER/SPARC/STEP | **Default training/rollout backend.** Generic BgB model + guards; machine-specific calibration lives env-side. |
| `tglfnn.yaml` | TGLFNN-UKAEA | linear | ITER/SPARC | Conventional TGLFNN on the cheap solver; independent check on QLKNN. |
| `tglfnn_nr.yaml` | TGLFNN-UKAEA | Newton-Raphson | ITER/SPARC | Nonlinear reference backend for validation rollouts; too costly for vectorized training. |
| `tglfnn_spherical.yaml` | TGLFNN-UKAEA | linear | STEP | STEP-trained TGLFNN weights plus STEP pedestal values. |
| `fusion_lstm.yaml` | — (learned dynamics) | — | KSTAR only | **Not a TORAX backend** — see below. |

The three TGLFNN files extend the private `_common/tglfnn.yaml` fragment.
That fragment is packaging-only configuration reuse, not a registered backend;
each public backend still resolves to the same complete TORAX mapping.

## Physics coverage

TORAX exposes a menu of modular physics. One geometry and one turbulent
transport model are selected per run, while compatible source terms are summed.
The table describes the resolved main-environment stacks; `test.yaml` instead
uses circular geometry and constant transport, and
`experiments/studies/reproduce_torax_paper_case.py` is a separate CHEASE
reference case.

| Physics area | Current use | Important limitation |
|--------------|-------------|----------------------|
| Magnetic geometry | Time-keyed EQDSK equilibria for ITER/SPARC; IMAS equilibrium for STEP | No self-consistent equilibrium solve; FBT is unused and CHEASE is reference-only. |
| Composition and evolved profiles | D–T main-ion mix plus one impurity mixture; evolving Ti, Te, ne, and psi | ITER/SPARC currently use Ne as the single impurity and prescribe Zeff. Heavy-impurity transport is not modelled. |
| Turbulent transport | Default BgB; optional CGM, QLKNN, and TGLFNN-UKAEA | Surrogates need OOD guards; BgB/CGM require calibration and do not reproduce all gyrokinetic effects. |
| Neoclassical physics | Redl bootstrap plus Angioni–Sauter transport on BgB/QLKNN/TGLFNN; CGM uses a bootstrap multiplier | Model choice is currently backend-owned even though machine/scenario calibration may differ. |
| Pedestal and L–H transition | Prescribed `set_T_ped_n_ped`; Martin formation model with adaptive source or transport | This is not a predictive ELMy-H pedestal model. |
| Auxiliary heating/current | Gaussian generic heat/current plus Gaussian Lin–Liu ECCD | ITER NBI and SPARC ICRF are deposition approximations, not dedicated source solvers. |
| Particle sources | Gas puff plus generic Gaussian source for ITER/SPARC; pellet model for STEP | Deposition is prescribed rather than coupled to neutral/pellet ablation physics. |
| Core heat sources | Bosch–Hale D–T fusion, ion–electron heat exchange, and ohmic heating for ITER/SPARC/STEP | Ohmic uses the standard resistive model wherever current is evolved. |
| Radiation | Relativistic bremsstrahlung, Mavrin impurity radiation, and Albajar cyclotron radiation, all tokamak-owned for ITER/SPARC; STEP keeps its OpenSTEP lumped-radiation sink | Cyclotron wall-reflection coefficient is the TORAX default (0.9), not machine-calibrated. |
| Rotation | Disabled on TGLFNN and not configured for QLKNN | No calibrated toroidal-rotation input or momentum evolution in the scenarios. |
| Edge/divertor | No coupled edge model | TORAX's Extended Lengyel model is not enabled. |
| Fast ions and ICRH | Fusion-power partitioning only | ToricNN/scaled-profile ICRH, fast-ion pressure/dilution, and ITG stabilization are not enabled. |
| MHD | Disruption termination plus TORAX's simple sawtooth trigger/redistribution in ITER baseline/hybrid and both SPARC scenarios. Inner-core transport patches remain surrogate OOD guards, not crash models. | Sawtooth crashes are internal event substeps inside one fixed-duration control transition. ITER advanced and STEP intentionally have no sawtooth model; QLKNN's proxy stays off because it is not a physical substitute. NTMs are not modelled. |

## Where physics configuration belongs

The merge precedence is, from lowest to highest:

`backend < wrappers < tokamak < scenario base < phase`

Use the narrowest layer that owns the physics:

- **Backend:** transport implementation, transport-specific corrections and
  bounds, transport-dependent neoclassical choices, solver, and numerical
  stabilization. A backend must remain usable by every registered compatible
  machine.
- **Tokamak (`tokamaks/*.yaml`):** machine-wide geometry plumbing, composition,
  wall/edge constants, and source-model choices that are valid for every
  scenario on that machine.
- **Scenario base (`envs/<tokamak>/<scenario>/base.yaml`):** discharge-specific
  Zeff, source powers and deposition, minority fractions, pedestal targets, and
  other physics shared by ramp-up, flat-top, and ramp-down. Initial profiles
  shared by flat-top and ramp-down also live here.
- **Phase YAML:** phase schedules, boundary conditions, geometry sequence,
  horizon, ramp-up profile overrides, and genuine phase-specific overrides.

For SPARC specifically, models common to both PRD and reduced-field operation
are selected once in `tokamaks/sparc.yaml`. Their operating points belong
in `envs/sparc/prd/base.yaml` and `envs/sparc/reduced_field/base.yaml`. The
bremsstrahlung/Mavrin/cyclotron model selection now lives at the tokamak layer
(shared with ITER), while Zeff, radiation multipliers, ICRH power, minority
concentration, and pedestal values remain scenario-level. Deep merge then keeps
the backend focused on transport and avoids duplicating the same SPARC physics
in every backend.

## Fidelity roadmap

These are proposed additions, not validated defaults. Each should land with a
resolved-config test, a short forward-run regression, power/particle accounting,
and comparison against a published or upstream TORAX reference case.

| Priority | Addition | Recommended owner | Rationale and validation gate |
|----------|----------|-------------------|-------------------------------|
| Done | `sources.ohmic` for ITER and SPARC | `tokamaks/iter.yaml`, `tokamaks/sparc.yaml` | Landed at the tokamak layer (standard resistive model). Resistive heating now enters the power balance wherever current is evolved. |
| Done | Bremsstrahlung and Mavrin impurity radiation backend-independent | ITER/SPARC tokamak files; scenario bases retain Zeff/multipliers | Moved to the tokamak layer, closing the ITER+BgB omission. STEP keeps its env-side lumped-radiation sink (no backend brems), removing a double-count against its OpenSTEP reference. |
| Done | Cyclotron radiation (Albajar) | `tokamaks/iter.yaml`, `tokamaks/sparc.yaml` | Enabled for ITER and SPARC at the tokamak layer. **Not** on STEP: its lumped `P_in_scaled_flat_profile` sink already includes synchrotron. Wall reflection is the TORAX default (0.9) pending machine calibration. |
| P1 | Replace SPARC's generic ICRF stand-in with ToricNN ICRH | Model/machine constants in `tokamaks/sparc.yaml`; power and minority mix in each SPARC scenario base | TORAX's ToricNN model is SPARC-specific and supplies species-resolved deposition. Validate supported field range, He3 composition, absorbed power, and deposition profiles before removing `generic_heat`. |
| P1 | Enable fast-ion pressure, dilution, and ITG stabilization after ICRH | Fast-ion source/composition in SPARC; stabilization switches in `qlknn.yaml` and `tglfnn*.yaml` | Restores important ICRH confinement effects. Verify that zero-fast-ion cases are unchanged and avoid double-counting fusion-alpha heating. |
| P1 | Couple the Extended Lengyel edge model | Machine constants in tokamak files; target temperature/seeding policy in scenario bases | Gives core boundary conditions and impurity seeding a physical divertor response. Start with SPARC, where compact high-power exhaust is central; validate explicit-coupling stability. |
| P2 | Enable calibrated rotation corrections | Rotation profiles/parameters in scenario bases; `use_rotation`/`rotation_mode` in TGLFNN/QLKNN backends | Adds ExB shear suppression only once toroidal rotation inputs are defensible. Do not enable a default correction with an implicit zero or guessed rotation profile. |
| Done | Fixed-duration differentiable sawtooth stepping | ITER baseline/hybrid and both SPARC scenario bases | Each RL transition now completes the configured physical interval through bounded internal PDE/event substeps, retaining crash/iteration diagnostics and reverse-mode differentiation. QLKNN's proxy remains off and inner transport patches remain surrogate OOD guards. |
| P2 | Add a direct QuaLiKiz validation backend | New validation-only backend, not a training default | Provides an offline ground-truth check for selected QLKNN states. It requires disabled JIT/file I/O and the linear solver, so it is unsuitable for vectorised RL. |
| Upstream | Predictive pedestal, heavy-impurity transport, NTM dynamics, self-consistent equilibrium | Not configurable until TORAX supplies validated models | Track as fidelity gaps rather than approximating them silently in unrelated backend knobs. |

## Reference Checks

- `envs/step.yaml` (on `bohm_gyrobohm.yaml`) matches TORAX
  `step_flattop_bgb.py` for the OpenSTEP BgB multiplier (`0.15`), base BgB
  coefficients, clipping bounds, and pedestal values; the backend supplies the
  Redl + Angioni-Sauter neoclassical models and the linear solver.
- `qlknn.yaml` uses the ITER hybrid QLKNN patch and clipping values from
  the TORAX ITER examples with a fixed-cost linear solver for rollouts/training.
- `tglfnn.yaml`, `tglfnn_nr.yaml`, and `tglfnn_spherical.yaml` share the same
  transport guards. The NR variant changes only the nonlinear solver controls;
  the spherical variant selects the STEP-trained TGLFNN machine weights.

### `fusion_lstm` is not a TORAX backend

`fusion_lstm.yaml` (`type: world_model`) is a learned-dynamics world model — a
NN ensemble trained on KSTAR discharges that emulates the 0D plasma response,
not a TORAX transport model. `make` dispatches it to a separate
learned-dynamics env. Unlike TORAX backends it is 1:1 with its scenario (pairs
only with the `kstar` env).
