# Physics randomization ranges

`variant="realistic"` samples every configured parameter independently and
uniformly for every transition. The sample used to produce `s[t+1]` is stored
in `s[t+1].phys_params`; immediately after reset that mapping contains nominal
values. `oracle` always uses the nominal configuration. Declare either an
absolute range or a range relative to the nominal runtime value at `t=0` (all
shipped targets are time-constant):

```yaml
physics_randomization:
  pedestal.T_e_ped: {relative: [0.8, 1.2]}
  numerics.resistivity_multiplier: {absolute: [0.95, 1.05]}
```

Device/scenario uncertainties live in `tokamaks/`; transport-model
uncertainties live in `backends/`, and the loader deep-merges them.

## Evidence and interpretation

The TORAX paper identifies turbulent transport, pedestal formation, radiation,
and analytic neoclassical closures as important modelling approximations. The
ranges below are not all statistical confidence intervals. They fall into
three categories, kept distinct here so the configuration is not more certain
than the evidence.

This transition-wise sampling follows the robust/non-stationary MDP framing in
Dulac-Arnold et al. (2019): the transition kernel can change with time within a
trajectory, rather than selecting one stationary perturbed MDP per episode.

| Parameters | Range | Kind | Basis |
|---|---:|---|---|
| `numerics.resistivity_multiplier`, `neoclassical.bootstrap_current.bootstrap_multiplier` | 0.95-1.05 x nominal | fit envelope | Sauter et al. report their conductivity/bootstrap formulae reproduce the numerical Fokker-Planck results to within about 5%. |
| `pedestal.T_i_ped`, `pedestal.T_e_ped`, `pedestal.n_e_ped` | 0.8-1.2 x nominal | validation envelope mapped to inputs | Cross-machine EPED validation reports roughly 20-25% scatter in pedestal predictions. TORAX exposes prescribed pedestal temperatures and density rather than an EPED pressure output, so the envelope is applied independently to those primitives. This is deliberately conservative for pedestal pressure. |
| `pedestal.formation_model.P_LH_prefactor` | 0.8-1.25 x nominal | propagated fit uncertainty | Martin scaling is `0.0488 exp(+-0.057) n^0.717+-0.035 B^0.803+-0.032 S^0.941+-0.019`. Combining coefficient/exponent extremes at ITER-like values gives approximately 0.81-1.23 x nominal, rounded outward. |
| QLKNN `collisionality_multiplier` | 0.85-1.15 x nominal | sensitivity proxy | QLKNN reports profile discrepancies of about 1-15% and dynamic errors of about 4-10% against QuaLiKiz/JETTO. Source-defined ITG/ETG nominal corrections stay locked; collisionality is the declared local calibration/randomization proxy. |
| TGLFNN `collisionality_multiplier` | 0.8-1.2 x nominal | sensitivity proxy | Published TGLF validation shows profile errors around 15-26%. TORAX has no global TGLFNN flux multiplier, so collisionality is randomized as a proxy, not claimed as a measured uncertainty on collisionality itself. |
| CGM stiffness/partition coefficients | 0.9-1.1 x nominal | interim engineering model-form prior | TORAX describes CGM's stiffness and exponent as free coefficients in a deliberately simplified critical-gradient model. No calibrated uncertainty is supplied, so the training prior is capped at +/-10% pending calibration. |
| BgB transport multipliers | 0.9-1.1 x nominal | interim engineering model-form prior | Bohm-GyroBohm is semi-empirical and its coefficients are machine/calibration dependent. No transferable statistical uncertainty is available, so the training prior is capped at +/-10% pending calibration. |
| Fixed inner/outer transport patches | 0.9-1.1 x nominal | interim engineering model-form prior | TORAX adds these patches specifically where the surrogate transport model is not validated (magnetic axis/sawtooth core and outer edge). The training prior is capped at +/-10% pending calibration. |
| Mavrin `radiation_multiplier` | 0.9-1.1 x nominal | interim engineering model-form prior | Radiation is highly sensitive to impurity state and atomic modelling; older cooling-rate datasets differ by as much as a factor of two for some low-Z impurities. The training prior is conservatively capped at +/-10% pending calibration rather than treating that spread as a Mavrin-fit uncertainty. |
| STEP `fraction_P_heating` | 0.9-1.1 x nominal | interim engineering scenario prior | The flat radiation fraction is a scenario calibration rather than a first-principles prediction, so the training prior is capped at +/-10% pending calibration. |

Sources:

- Dulac-Arnold et al., [Challenges of Real-World Reinforcement Learning](https://arxiv.org/abs/1904.12901).
- Citrin et al., [TORAX: A Fast and Differentiable Tokamak Transport Simulator in JAX](https://arxiv.org/abs/2406.06718).
- Sauter et al., [Neoclassical conductivity and bootstrap current formulas for general axisymmetric equilibria and arbitrary collisionality regime](https://infoscience.epfl.ch/entities/publication/f8bcbda0-d231-42b5-b556-2d3f0ca0b59d).
- van de Plassche et al., [Fast modeling of turbulent transport in fusion plasmas using neural networks](https://arxiv.org/abs/1911.05617).
- Martin et al. scaling as documented by [PROCESS: Plasma H-mode](https://ukaea.github.io/PROCESS/physics-models/plasma_h_mode/).
- Snyder et al., [Cross-machine validation of the EPED model](https://meetings-archive.aps.org/dpp/2015/tp12/90/).
- Rodriguez-Fernandez et al., [Validation of TGLF across a database of MAST, MAST-U and NSTX discharges](https://archive.aps.org/dpp/2024/pp12/97/).
- Post et al., [Steady-state radiative cooling rates for low-density, high-temperature plasmas](https://arxiv.org/abs/plasm-ph/9506001).
