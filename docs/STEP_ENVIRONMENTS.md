# STEP SPP-001 EC-HD flat-top

The public environment name is `step`. It refers permanently to the
**STEP SPP-001 EC-HD flat-top** and not to the latest STEP design.

This is a design-stage core-plasma environment, similar in scope to the ITER
and SPARC environments in this repository. It is not a whole-plant digital
twin. OpenSTEP supplies one stationary state at 300 s; the environment uses
that state as its initial condition and then predicts 100 s of evolution with
TORAX. That evolution is not a released STEP time trajectory.

## Scope and public interface

| Item | Definition |
|---|---|
| Environment | `step` |
| Official target | SPP-001 EC-HD flat-top |
| Initial state | OpenSTEP `ec-hd/ftop` at 300 s |
| Primary backend | `bohm_gyrobohm` |
| Optional backend | `tglfnn_spherical` |
| Actions | `P_eccd`, `rho_eccd`, `pellet_rate` |
| Episode | 100 s, 2 s control interval, 50 steps |
| Added aliases or APIs | None |

Only this flat-top is in scope. There is no ramp-up, ramp-down, full-discharge,
EB-CC, other SPP-001 operating point, or SPP-002 environment here.

## Provenance

| Field | Value |
|---|---|
| Verification date | 30 July 2026 |
| Repository base used for the audit | `fca1df55a667536f762cc05b3d3a40ea8f763233` |
| TORAX | `1.4.2` |
| Official release | [OpenSTEP `v1.0`](https://github.com/ukaea/OpenSTEP/tree/v1.0) |
| OpenSTEP tag commit | `60b0ec2f6a036b9dc6a79dd52e9a20dec24c9930` |
| Dataset DOI | [`10.14468/07jt-s540`](https://doi.org/10.14468/07jt-s540) |
| Licence | [CC-BY-4.0](https://github.com/ukaea/OpenSTEP/blob/v1.0/LICENSE.txt) |
| Upstream asset | [`ec-hd/ftop/STEP_SPP_001_ECHD_ftop.nc`](https://github.com/ukaea/OpenSTEP/blob/v1.0/ec-hd/ftop/STEP_SPP_001_ECHD_ftop.nc) |
| Vendored asset | [`src/plasmax/configs/data/STEP_SPP_001_ECHD_ftop.nc`](../src/plasmax/configs/data/STEP_SPP_001_ECHD_ftop.nc) |
| Packaged attribution | [`STEP_SPP_001_ECHD_ftop.NOTICE`](../src/plasmax/configs/data/STEP_SPP_001_ECHD_ftop.NOTICE) |
| Size | 1,260,824 bytes |
| SHA-256 | `64fe9e0d7f634be8ce4b748c08090ecdca2a4a5212473fb5a2c33cb2b6fd4bad` |
| Producer recorded in the asset | JETTO `36.0.4`, commit `6d0278ba50b33ffbcfdb1828e2d956026fbd1aa3` |

The vendored asset is byte-for-byte identical to the file at the OpenSTEP
`v1.0` tag.

The file records `Conventions=IMAS` and
`data_dictionary_version=4.0.0`. IMAS Data Dictionary major version 4 is
[COCOS 17](https://imas-data-dictionary.readthedocs.io/en/latest/cocos.html).
The file has no separate COCOS scalar. Asset-specific metadata therefore
governs this environment.

## What OpenSTEP provides

This is one exported flat-top slice, not a time series:

| IDS group | Released content |
|---|---|
| `core_profiles` | One 300 s slice; 150-point temperatures, density, flux, q, composition, and related profiles |
| `equilibrium` | One 300 s fixed-boundary equilibrium; 150-point 1-D data, a 151 × 151 2-D grid, and a 72-point boundary |
| `core_sources` | One 300 s slice; 15 named source records and their radial profiles |
| `core_transport` | One 300 s slice; `combined`, `transport_solver`, `background`, `neoclassical`, and `anomalous` coefficient sets |
| `ntms` | Empty metadata/time placeholder at 0 s; no usable NTM physics state |

Missing or fill-valued fields are absent, not zero.

Selected direct reference values are:

| Quantity | Released value | Exact locator |
|---|---:|---|
| Plasma current | 21.228640 MA | `equilibrium/0/time_slice.global_quantities.ip` |
| Major-radius field reference | 3.6 m, +3.2 T | `equilibrium/0/vacuum_toroidal_field.{r0,b0}` |
| Minor radius | 2.000984 m | `equilibrium/0/time_slice.boundary.minor_radius` |
| Elongation | 2.989611 | `equilibrium/0/time_slice.boundary.elongation` |
| `beta_N` | 4.489618 | `equilibrium/0/time_slice.global_quantities.beta_tor_norm` |
| `li(3)` | 0.276933 | `equilibrium/0/time_slice.global_quantities.li_3` |
| EC electron heating | 150.000000 MW | `core_sources/0/source[ec].global_quantities.electrons.power` |
| EC parallel current | 1.723623 MA | `core_sources/0/source[ec].global_quantities.current_parallel` |
| Bootstrap parallel current | 21.022642 MA | `core_sources/0/source[bootstrap_current].global_quantities.current_parallel` |
| Pellet ion source | `1.175488242e22 s^-1` | `core_sources/0/source[pellet].global_quantities.total_ion_particles` |

The peer-reviewed SPP-001 paper reports 1.87 MA EC current and
`9.35e21 s^-1` pellet fuelling for EC-HD
([Tholerus et al. 2024, table 5](https://doi.org/10.1088/1741-4326/ad6ea2)).
Those differ from the exported IDS values above. The environment uses the
machine-readable release and keeps the paper values visible as a conflict; it
does not average or merge them.

## What the environment implements

The configuration is
[`src/plasmax/configs/envs/step.yaml`](../src/plasmax/configs/envs/step.yaml).

### 1. Official reset state

- `T_e`, `T_i`, `n_e`, and `psi` are interpolated directly from
  `core_profiles` onto the 25-cell TORAX grid.
- The fixed-boundary geometry is loaded directly from `equilibrium`.
- D/T, Xe, and He ratios are retained from the release. TORAX solves the Ar
  fraction with `n_e_ratios_Z_eff`, reproducing the released `Z_eff` profile
  while keeping all impurity fractions non-negative.
- Physical reset noise is disabled so an oracle reset is reproducible.
  Measurement degradation remains available through the `realistic` wrapper.

The reset reproduces the imported profiles to floating-point interpolation
precision. Plasma current is exact; volume and area agree within `5e-5`
relative; q-min and `li(3)` agree within 0.5%.

There is a known representation gap in pressure-derived quantities:
TORAX reconstructs `beta_N≈4.046` and thermal energy `≈583.9 MJ`, versus
the released equilibrium `beta_N=4.4896` and the `core_profiles` thermal
pressure integral `≈607.8 MJ`. The imported temperature and density profiles
are not rescaled to hide this difference.

### 2. Calibrated Bohm–gyroBohm transport

No new transport model is introduced. The existing TORAX Bohm–gyroBohm model
is fitted to the OpenSTEP `anomalous` transport coefficients on
`0.05 <= rho_tor_norm <= 0.90`. The axis convention and pedestal region are
excluded from the fit.

The fitted settings are:

```yaml
chi_e_bohm_multiplier: 0.12
chi_i_bohm_multiplier: 0.12
chi_e_gyrobohm_multiplier: 0.12
chi_i_gyrobohm_multiplier: 0.12
chi_min: 0.01
D_face_c1: 3.3627577722
D_face_c2: 1.8200709945
V_face_coeff: -0.2205599932
```

On the production grid, normalized profile error changes as follows:

| Coefficient | Before | Calibrated |
|---|---:|---:|
| Electron anomalous heat diffusivity | 24.9% | 2.0% |
| Ion anomalous heat diffusivity | 24.6% | 4.6% |
| Particle diffusivity | 69.4% | 9.6% |
| Particle convection | 95.6% | 41.2% |

The remaining convection error is structural: TORAX exposes one scalar `V/D`
coefficient while the released convection varies radially. The released
`core_transport` profiles are calibration targets; they are not imported as a
prescribed transport trajectory.

The pedestal is fixed at the released profiles interpolated to
`rho_tor_norm=0.95`:

- `T_e = 4.423820 keV`
- `T_i = 3.245460 keV`
- `n_e = 5.887068e19 m^-3`

`tglfnn_spherical` remains an optional comparison backend. Registration and a
one-step integration smoke test do not make it validated against OpenSTEP.

### 3. Simplified controllable sources

The action interface is unchanged:

| Action | Bounds | Nominal value |
|---|---:|---:|
| `P_eccd` | 50–300 MW | 150 MW |
| `rho_eccd` | 0.3–0.7 | 0.3384 |
| `pellet_rate` | `1e20–5e22 s^-1` | `1.175488242e22 s^-1` |

OpenSTEP's EC deposition is multi-peaked. TORAX uses one controllable Gaussian,
with location `0.3384` and width `0.1704`, chosen to reproduce the released
volume-weighted heat-deposition moments. The pellet Gaussian uses location
`0.7993` and width `0.1697`. These are reduced control models, not launcher- or
pellet-resolved replays.

The IDS current scalars are parallel-current integrals; TORAX reports toroidal
current after a geometry conversion. On the 25-cell production geometry, the
released radial profiles convert to approximately:

- EC toroidal current: 1.926020 MA
- bootstrap toroidal current: 19.342936 MA
- total non-inductive toroidal current: 21.268956 MA

The EC current-drive efficiency (`0.1362405203`) and bootstrap multiplier
(`0.9115699759`) are calibrated against those like-for-like values. Comparing
the raw IDS parallel-current scalars directly with TORAX's toroidal totals is
incorrect.

The flat radiation sink remains a reduced core power-balance model. It is not
divertor, scrape-off-layer, or exhaust physics. OpenSTEP's detailed source
profiles are used for calibration and regression, but are not replayed at
runtime.

## Validation status

The STEP-specific tests are in
[`tests/step_env_test.py`](../tests/step_env_test.py).

| Gate | Status |
|---|---|
| OpenSTEP tag, byte identity, checksum, IDS layout | Passed |
| IMAS DD4 / COCOS-17 convention and equilibrium signs | Passed |
| Reset `T_e`, `T_i`, `n_e`, `psi`, `Z_eff`, non-negative composition | Passed |
| Plasma current, geometry, q-min, `li(3)`, and declared pressure gap | Passed |
| EC heat, pellet rate, fusion deposition, radiation, and net source power | Passed |
| EC, bootstrap, non-inductive, and total-current balance in one current basis | Passed |
| Bohm–gyroBohm anomalous transport-profile fit | Passed |
| 100 s nominal Bohm–gyroBohm rollout | Passed: finite, no disruption, 50/50 steps |
| Timestep check | Passed for principal globals |
| Dynamic radial-grid convergence | Not passed |
| `tglfnn_spherical` physics validation | Not passed; optional smoke comparison only |

For the selected calibration, the nominal 100 s rollout keeps q-min above about
1.9 and the line-averaged Greenwald fraction below about 0.99. The main drift is
roughly -9% in plasma current, -6% in thermal energy, +3% in `beta_N`, and -14%
in fusion power. These are simulated drifts from the released snapshot, not
claims about the real STEP flat-top.

A 1 s timestep changes the 100 s endpoint by less than 0.6% for plasma current,
thermal energy, `beta_N`, fusion power, and Greenwald fraction relative to the
2 s production step. Its q-min differs by about 0.16 but remains safely above
the 0.8 limit.

Reset sweeps at 25, 50, and 100 radial cells change the main thermodynamic and
equilibrium globals by less than 0.5% after interpolation. Dynamic radial
convergence is not established: the same fitted model at 50 cells develops an
unphysical q-min failure even with a 0.5 s timestep. The environment therefore
retains the tested 25-cell configuration. It must be described as the validated
production discretization, not as a grid-converged evolution.

## Fidelity summary

| Dimension | Fidelity |
|---|---|
| Asset identity and equilibrium | Replayed |
| Reset temperature, density, flux, and `Z_eff` | Replayed after interpolation |
| Detailed impurity composition | Calibrated reduced representation |
| Bohm–gyroBohm core transport | Calibrated on `0.05 <= rho <= 0.90` |
| EC and pellet integrated sources | Calibrated |
| EC and pellet deposition shapes | Simplified Gaussians |
| Radiation | Simplified core sink |
| 100 s evolution | Synthetic TORAX prediction from one official slice |
| Divertor/SOL, plant systems, diagnostics, PF circuits | Absent |

## Remaining work

1. Resolve the 50-cell current/q evolution before claiming dynamic radial-grid
   convergence.
2. Validate `tglfnn_spherical` independently against the same reset, source,
   transport, and rollout checks; until then it remains optional.
3. Improve the reduced EC and particle-convection shapes only if this can be
   done without changing the public action interface.
4. Add uncertainty ranges only where OpenSTEP data or a validated model
   comparison supports them.

Everything else—other SPP-001 points, EB-CC, SPP-002, ramp phases,
full-discharge modelling, free-boundary PF evolution, walls, diagnostics,
divertor/SOL physics, and plant control—requires a separate proposal and
appropriate official data. None should be inferred from this flat-top or added
as an alias of `step`.
