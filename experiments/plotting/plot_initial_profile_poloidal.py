"""Plot the distinct phase initialization profiles in poloidal geometry.

The phased ITER/SPARC environments have two distinct state initializations:
``rampup`` starts cold, while ``flattop`` and ``rampdown`` start hot. This
script maps their configured one-dimensional T_e and n_e profiles onto the
corresponding EQDSK poloidal-flux grid, reusing the LCFS and first-wall
treatment from :mod:`plot_equilibrium_shapes`.

Usage::

    uv run python experiments/plotting/plot_initial_profile_poloidal.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import tyro
from eqdsk import EQDSKInterface
from matplotlib.path import Path as MplPath
from ruamel.yaml import YAML

from experiments.plotting.plot_equilibrium_shapes import _lcfs, _panel_title
from scripts.project_paths import CONFIGS_DIR, PLOTS_DIR
from tools.equilibria.equilibrium_specs import RAMP_IP_LEVELS_MA, ip_tag

_INK = "#1a1a17"
_MUTED = "#6b6a66"
_GRID = "#d4d3cf"
_PHASES = (
    ("rampup", "Ramp-up", 0),
    ("flattop", "Flat-top / ramp-down", -1),
)
_SCENARIO_DIRS = {
    "iter_baseline": "iter/baseline",
    "iter_hybrid": "iter/hybrid",
    "iter_advanced": "iter/advanced",
    "sparc_prd": "sparc/prd",
    "sparc_reduced_field": "sparc/reduced_field",
}


@dataclass(frozen=True)
class RadialProfile:
    """A configured cell-centred profile on normalized radius."""

    rho: np.ndarray
    values: np.ndarray


@dataclass(frozen=True)
class InitialState:
    """Profiles and geometry needed for one phase initialization panel."""

    T_e: RadialProfile
    T_i: RadialProfile
    n_e: RadialProfile
    equilibrium: EQDSKInterface
    ip_ma: float


@dataclass(frozen=True)
class Args:
    """CLI for the poloidal initialization atlas."""

    env_dir: Path = CONFIGS_DIR / "envs"
    """Directory containing the environment YAMLs."""
    data_dir: Path = CONFIGS_DIR / "data"
    """Directory containing the generated EQDSK files."""
    out_path: Path = PLOTS_DIR / "initial_profile_poloidal.png"
    """Output PNG or PDF path."""
    scenarios: tuple[str, ...] = tuple(_SCENARIO_DIRS)
    """Scenario generator keys to include."""


def _radial_profile(profile_conditions: dict[str, Any], name: str) -> RadialProfile:
    time_values = profile_conditions[name]
    radial_values = time_values[min(time_values, key=float)]
    ordered = sorted(radial_values.items(), key=lambda item: float(item[0]))
    return RadialProfile(
        rho=np.asarray([float(rho) for rho, _ in ordered]),
        values=np.asarray([float(value) for _, value in ordered]),
    )


def _load_state(
    scenario: str,
    phase: str,
    ip_index: int,
    env_dir: Path,
    data_dir: Path,
) -> InitialState:
    yaml = YAML(typ="safe")
    env_path = env_dir / _SCENARIO_DIRS[scenario] / f"{phase}.yaml"
    data = yaml.load(env_path.read_text())
    profile_conditions = data["torax"]["profile_conditions"]
    ip_ma = RAMP_IP_LEVELS_MA[scenario][ip_index]
    equilibrium = EQDSKInterface.from_file(
        str(data_dir / f"{scenario}_{ip_tag(ip_ma)}.eqdsk"),
        no_cocos=True,
    )
    return InitialState(
        T_e=_radial_profile(profile_conditions, "T_e"),
        T_i=_radial_profile(profile_conditions, "T_i"),
        n_e=_radial_profile(profile_conditions, "n_e"),
        equilibrium=equilibrium,
        ip_ma=ip_ma,
    )


def _poloidal_profile(
    equilibrium: EQDSKInterface,
    profile: RadialProfile,
) -> tuple[np.ndarray, np.ndarray, np.ma.MaskedArray, np.ma.MaskedArray, np.ndarray]:
    """Map a radial profile onto the EQDSK poloidal-flux grid."""
    r = np.linspace(
        equilibrium.xgrid1,
        equilibrium.xgrid1 + equilibrium.xdim,
        equilibrium.nx,
    )
    z = np.linspace(
        equilibrium.zmid - equilibrium.zdim / 2,
        equilibrium.zmid + equilibrium.zdim / 2,
        equilibrium.nz,
    )
    r_grid, z_grid = np.meshgrid(r, z)
    psi = np.asarray(equilibrium.psi).T
    normalized_flux = (psi - equilibrium.psimag) / (
        equilibrium.psibdry - equilibrium.psimag
    )
    # TORAX profile rho is normalized toroidal-flux radius, whereas the 2D
    # EQDSK grid carries poloidal flux. Since d(Phi_tor)/d(psi_pol) = q, the
    # normalized cumulative integral of q(psi) supplies the required mapping.
    q_psi = np.abs(np.asarray(equilibrium.qpsi, dtype=float))
    dpsi = 1.0 / (q_psi.size - 1)
    toroidal_flux = np.concatenate(
        [[0.0], np.cumsum(0.5 * (q_psi[:-1] + q_psi[1:]) * dpsi)]
    )
    rho_tor = np.sqrt(toroidal_flux / toroidal_flux[-1])
    rho = np.interp(
        np.clip(normalized_flux, 0.0, 1.0),
        np.linspace(0.0, 1.0, q_psi.size),
        rho_tor,
    )

    lcfs = _lcfs(equilibrium)
    points = np.column_stack([r_grid.ravel(), z_grid.ravel()])
    inside = MplPath(lcfs, closed=True).contains_points(points).reshape(rho.shape)
    valid = (
        inside & np.isfinite(rho) & (normalized_flux >= 0.0) & (normalized_flux <= 1.0)
    )
    values = np.interp(
        rho,
        profile.rho,
        profile.values,
        left=profile.values[0],
        right=profile.values[-1],
    )
    return (
        r_grid,
        z_grid,
        np.ma.array(values, mask=~valid),
        np.ma.array(rho, mask=~valid),
        lcfs,
    )


def _limiter(equilibrium: EQDSKInterface) -> np.ndarray:
    limiter = np.column_stack([equilibrium.xlim, equilibrium.zlim])
    if not np.allclose(limiter[0], limiter[-1]):
        limiter = np.vstack([limiter, limiter[:1]])
    return limiter


def _draw_panel(
    ax: plt.Axes,
    state: InitialState,
    profile: RadialProfile,
    *,
    cmap: str,
    norm: mpl.colors.Normalize,
    annotation: str,
) -> None:
    r, z, values, rho, lcfs = _poloidal_profile(state.equilibrium, profile)
    ax.pcolormesh(r, z, values, cmap=cmap, norm=norm, shading="auto", rasterized=True)
    ax.contour(
        r,
        z,
        rho,
        levels=(0.2, 0.4, 0.6, 0.8),
        colors="white",
        linewidths=0.45,
        alpha=0.65,
    )
    limiter = _limiter(state.equilibrium)
    ax.plot(limiter[:, 0], limiter[:, 1], color=_MUTED, lw=0.9)
    ax.plot(lcfs[:, 0], lcfs[:, 1], color=_INK, lw=1.0)
    ax.text(
        0.03,
        0.03,
        annotation,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7,
        color=_INK,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 2},
    )
    ax.set_aspect("equal")
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_color(_GRID)
    ax.tick_params(colors=_MUTED, labelsize=7, length=2)


def main(args: Args) -> None:
    unknown = set(args.scenarios) - set(_SCENARIO_DIRS)
    if unknown:
        raise ValueError(f"Unknown scenarios: {sorted(unknown)}")

    states = {
        (scenario, phase): _load_state(
            scenario,
            phase,
            ip_index,
            args.env_dir,
            args.data_dir,
        )
        for scenario in args.scenarios
        for phase, _, ip_index in _PHASES
    }
    max_T_e = max(float(state.T_e.values.max()) for state in states.values())
    max_n_e_1e20 = max(
        float(state.n_e.values.max() * 1e-20) for state in states.values()
    )
    T_e_norm = mpl.colors.Normalize(vmin=0.0, vmax=max_T_e)
    n_e_norm = mpl.colors.Normalize(vmin=0.0, vmax=max_n_e_1e20)

    fig = plt.figure(
        figsize=(12.0, 3.15 * len(args.scenarios) + 0.65),
        constrained_layout=True,
    )
    grid = fig.add_gridspec(
        len(args.scenarios) + 1,
        4,
        height_ratios=(*([1.0] * len(args.scenarios)), 0.045),
    )
    axes = np.asarray(
        [
            [fig.add_subplot(grid[row, col]) for col in range(4)]
            for row in range(len(args.scenarios))
        ]
    )
    T_e_colorbar_ax = fig.add_subplot(grid[-1, :2])
    n_e_colorbar_ax = fig.add_subplot(grid[-1, 2:])
    column_titles = (
        "Ramp-up · Tₑ",
        "Flat-top / ramp-down · Tₑ",
        "Ramp-up · nₑ",
        "Flat-top / ramp-down · nₑ",
    )
    for col, title in enumerate(column_titles):
        axes[0, col].set_title(title, color=_INK, fontsize=10)

    for row, scenario in enumerate(args.scenarios):
        for phase_index, (phase, _, _) in enumerate(_PHASES):
            state = states[(scenario, phase)]
            T_e_annotation = (
                f"Iₚ={state.ip_ma:g} MA\n"
                f"Tₑ₀={state.T_e.values.max():.1f} keV\n"
                f"Tᵢ₀={state.T_i.values.max():.1f} keV"
            )
            _draw_panel(
                axes[row, phase_index],
                state,
                state.T_e,
                cmap="magma",
                norm=T_e_norm,
                annotation=T_e_annotation,
            )
            n_e_profile = RadialProfile(
                rho=state.n_e.rho,
                values=state.n_e.values * 1e-20,
            )
            _draw_panel(
                axes[row, phase_index + 2],
                state,
                n_e_profile,
                cmap="viridis",
                norm=n_e_norm,
                annotation=(
                    f"Iₚ={state.ip_ma:g} MA\n"
                    f"nₑ₀={state.n_e.values.max() * 1e-20:.2f}×10²⁰ m⁻³"
                ),
            )

        axes[row, 0].set_ylabel(
            f"{_panel_title(scenario)}\nZ [m]",
            color=_INK,
            fontsize=9,
        )
        for col in range(4):
            if row == len(args.scenarios) - 1:
                axes[row, col].set_xlabel("R [m]", color=_INK, fontsize=8)
            else:
                axes[row, col].set_xticklabels([])
            if col != 0:
                axes[row, col].set_yticklabels([])

    T_e_mappable = mpl.cm.ScalarMappable(norm=T_e_norm, cmap="magma")
    n_e_mappable = mpl.cm.ScalarMappable(norm=n_e_norm, cmap="viridis")
    T_e_colorbar = fig.colorbar(
        T_e_mappable,
        cax=T_e_colorbar_ax,
        orientation="horizontal",
    )
    T_e_colorbar.set_label("Electron temperature Tₑ [keV]", fontsize=9)
    n_e_colorbar = fig.colorbar(
        n_e_mappable,
        cax=n_e_colorbar_ax,
        orientation="horizontal",
    )
    n_e_colorbar.set_label("Electron density nₑ [10²⁰ m⁻³]", fontsize=9)
    fig.suptitle(
        "Validated initial states on poloidal flux surfaces",
        color=_INK,
        fontsize=13,
    )
    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_path, dpi=220, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out_path}")


if __name__ == "__main__":
    main(tyro.cli(Args))
