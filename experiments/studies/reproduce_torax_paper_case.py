"""Reproduce the TORAX paper's ITER-like stationary validation case.

Paper target (Citrin et al. 2024, section IV.A):
- Ip = 11.5 MA
- 50 MW external heating, equally split ions/electrons
- constant transport: chi_i=2, chi_e=1, D_e=1, V_e=-0.15
- D-T fusion, bootstrap, ohmic, and ion-electron exchange enabled
- 10 s run with dt=0.05 s, then inspect the stationary state

The paper used TORAX's CHEASE ITER hybrid equilibrium. This script resolves the
installed TORAX package's bundled `iterhybrid.mat2cols` geometry at runtime.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import torax
from torax._src.torax_pydantic import model_config

from plasmax.environment import factory as env_config
from plasmax.environment.schema import (
    ActuatorConfig,
    DisruptionConfig,
    ObservationsConfig,
    ObsProfileConfig,
    ObsScalarConfig,
    PlasmaxConfig,
    TaskConfig,
)
from plasmax.wrappers import unwrap_to_env_state


def _paper_scenario(geometry_directory: Path, t_final: float, dt: float):
    torax_cfg = {
        "plasma_composition": {
            "main_ion": {"D": 0.5, "T": 0.5},
            "impurity": "Ne",
            "Z_eff": 1.6,
        },
        "profile_conditions": {
            "Ip": 11.5e6,
            "T_i": {0.0: {0.0: 8.0, 1.0: 0.2}},
            "T_i_right_bc": 0.2,
            "T_e": {0.0: {0.0: 8.0, 1.0: 0.2}},
            "T_e_right_bc": 0.2,
            # Paper table uses nbar=0.8 with nbar_is_fGW=False. Current TORAX
            # uses SI units on that path, so spell the value as 0.8e20 m^-3.
            "n_e_right_bc": 0.5e20,
            "n_e_nbar_is_fGW": False,
            "normalize_n_e_to_nbar": True,
            "nbar": 0.8e20,
            "n_e": {0: {0.0: 1.0, 1.0: 1.0}},
        },
        "numerics": {
            "t_final": t_final,
            "exact_t_final": True,
            "fixed_dt": dt,
            "resistivity_multiplier": 1.0,
            "evolve_ion_heat": True,
            "evolve_electron_heat": True,
            "evolve_current": True,
            "evolve_density": True,
        },
        "geometry": {
            "geometry_type": "chease",
            "geometry_directory": str(geometry_directory),
            "geometry_file": "iterhybrid.mat2cols",
            "Ip_from_parameters": True,
            "R_major": 6.2,
            "a_minor": 2.0,
            "B_0": 5.3,
            "n_rho": 50,
        },
        "neoclassical": {
            "bootstrap_current": {
                "bootstrap_multiplier": 1.0,
            },
        },
        "sources": {
            "generic_particle": {
                "S_total": 3.0e21,
                "deposition_location": 0.2,
                "particle_width": 0.25,
            },
            "generic_heat": {
                "gaussian_location": 0.11,
                "gaussian_width": 0.2,
                "P_total": 50.0e6,
                "electron_heat_fraction": 0.5,
            },
            "ohmic": {},
            "fusion": {},
            "ei_exchange": {
                "Qei_multiplier": 1.0,
            },
        },
        "pedestal": {
            "model_name": "set_T_ped_n_ped",
            "set_pedestal": False,
        },
        "transport": {
            "model_name": "constant",
            "chi_i": 2.0,
            "chi_e": 1.0,
            "D_e": 1.0,
            "V_e": -0.15,
        },
        "solver": {
            "solver_type": "newton_raphson",
            "use_predictor_corrector": True,
            "n_corrector_steps": 5,
            "use_pereverzev": True,
            "chi_pereverzev": 30,
            "D_pereverzev": 15,
        },
        "time_step_calculator": {
            "calculator_type": "fixed",
        },
    }
    torax_config = model_config.ToraxConfig.from_dict(torax_cfg)
    return PlasmaxConfig(
        environment_key="torax_paper_iter_stationary",
        torax=torax_config,
        task=TaskConfig(reward="P_diff", terminal_penalty=0.0),
        actuators=[
            ActuatorConfig(
                name="P_nbi",
                low=0.0,
                high=100.0e6,
                max_delta=float("inf"),
                init=50.0e6,
            )
        ],
        observations=ObservationsConfig(
            profiles=[
                ObsProfileConfig(name="T_e", scale=10.0, bounds=(0.0, 50.0)),
                ObsProfileConfig(name="T_i", scale=10.0, bounds=(0.0, 50.0)),
                ObsProfileConfig(name="n_e", scale=1.0e20, bounds=(0.0, 2.0e20)),
                ObsProfileConfig(name="psi", scale=10.0, bounds=(-200.0, 200.0)),
                ObsProfileConfig(name="q", scale=5.0, bounds=(0.0, 20.0)),
            ],
            scalars=[
                ObsScalarConfig(name="W_thermal", scale=1.0e8, bounds=(0.0, 1.0e9)),
                ObsScalarConfig(name="tau_E", scale=1.0, bounds=(0.0, 20.0)),
                ObsScalarConfig(name="P_fusion", scale=1.0e8, bounds=(0.0, 1.0e9)),
                ObsScalarConfig(name="t", scale=t_final, bounds=(0.0, t_final)),
                ObsScalarConfig(name="q_min", scale=3.0, bounds=(0.0, 10.0)),
                ObsScalarConfig(name="q95", scale=5.0, bounds=(0.0, 20.0)),
                ObsScalarConfig(name="beta_N", scale=3.0, bounds=(0.0, 5.0)),
                ObsScalarConfig(name="f_non_inductive", scale=1.0, bounds=(0.0, 1.5)),
            ],
        ),
        disruption=DisruptionConfig(q_min_threshold=0.01, greenwald_threshold=10.0),
    )


def _build_env(t_final: float, dt: float):
    geometry_directory = Path(torax.__file__).parent / "data" / "third_party" / "geo"
    scenario = _paper_scenario(geometry_directory, t_final, dt)
    num_steps = int(round(t_final / dt))
    return env_config._build_env(  # noqa: SLF001 - script-local reproduction helper.
        scenario,
        reward="P_diff",
        variant="oracle",
        max_steps=num_steps,
        disruption_penalty=0.0,
        time_aware=False,
        quantize_bins=None,
    )


def _rollout(env, key, num_steps: int):
    state, _ = env.init(key)
    action = jnp.asarray([0.0])  # normalized midpoint = 50 MW by construction.
    rows = []
    for _ in range(num_steps):
        state, info = env.step(state, action)
        base_state = unwrap_to_env_state(state)
        po = base_state.plasma
        cp = po.core
        rows.append(
            {
                "reward": info.reward,
                "terminated": info.terminated,
                "truncated": info.truncated,
                "t": po.t,
                "P_fusion": po.P_fusion,
                "P_aux": po.P_aux_total,
                "Q_fusion": po.Q_fusion,
                "W_thermal": po.W_thermal_total,
                "tau_E": po.tau_E,
                "q_min": po.q_min,
                "q95": po.q95,
                "fgw": po.fgw_n_e_line_avg,
                "beta_N": po.beta_N,
                "T_i_core": cp.T_i.value[0],
                "T_e_core": cp.T_e.value[0],
                "n_e_line_avg": po.n_e_line_avg,
            }
        )
        if bool(np.asarray(info.terminated | info.truncated)):
            break
    return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *rows)


def _mean_last(x: np.ndarray, n: int) -> float:
    return float(np.mean(x[-min(n, x.size) :]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--t-final", type=float, default=10.0)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--tail-steps", type=int, default=20)
    args = parser.parse_args()

    env = _build_env(args.t_final, args.dt)
    num_steps = int(round(args.t_final / args.dt))
    print(f"Running TORAX paper stationary case: {num_steps} steps...")
    traj = _rollout(env, jax.random.key(0), num_steps)
    jax.block_until_ready(traj["Q_fusion"])
    data = {k: np.asarray(v) for k, v in traj.items()}
    alive_steps = data["reward"].size

    print(f"alive_steps: {alive_steps}/{num_steps}")
    print("final:")
    for key, scale, unit in [
        ("Q_fusion", 1.0, ""),
        ("P_fusion", 1e-6, "MW"),
        ("P_aux", 1e-6, "MW"),
        ("W_thermal", 1e-6, "MJ"),
        ("tau_E", 1.0, "s"),
        ("q_min", 1.0, ""),
        ("q95", 1.0, ""),
        ("fgw", 1.0, ""),
        ("beta_N", 1.0, ""),
        ("T_i_core", 1.0, "keV"),
        ("T_e_core", 1.0, "keV"),
        ("n_e_line_avg", 1e-20, "1e20 m^-3"),
    ]:
        print(f"  {key}: {float(data[key][-1]) * scale:.6g} {unit}".rstrip())

    print(f"tail mean over last {min(args.tail_steps, num_steps)} steps:")
    for key, scale, unit in [
        ("Q_fusion", 1.0, ""),
        ("P_fusion", 1e-6, "MW"),
        ("P_aux", 1e-6, "MW"),
        ("T_i_core", 1.0, "keV"),
        ("T_e_core", 1.0, "keV"),
        ("n_e_line_avg", 1e-20, "1e20 m^-3"),
    ]:
        value = _mean_last(data[key], args.tail_steps) * scale
        print(f"  {key}: {value:.6g} {unit}".rstrip())


if __name__ == "__main__":
    main()
