"""Plot the per-scenario LCFS shape evolution against the machine limiter.

One panel per scenario: the limiter contour plus the plasma boundary from each
generated ``{scenario}_ip{NNN}.eqdsk`` waypoint, shaded light -> dark with
increasing Ip. The lowest-Ip boundary is the inboard-limited startup shape, so
the panel shows the geometry actually interpolated through a ramp: small
limited startup -> blended mid -> full diverted flat-top.

The boundary is contoured from the psi map at the boundary flux rather than
read from the file's ``rbdry``/``zbdry`` arrays: for the pinned (startup/mid)
solves those arrays are deliberately inflated 2% so TORAX's LCFS-bbox psi mask
clears the plasma by a grid spacing, and would draw as poking through the
wall (see ``generate_equilibrium._CroppedEq``).

Usage::

    uv run python experiments/plotting/plot_equilibrium_shapes.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import contourpy
import matplotlib.pyplot as plt
import numpy as np
import tyro
from eqdsk import EQDSKInterface

from scripts.project_paths import CONFIGS_DIR, PLOTS_DIR
from tools.equilibria.equilibrium_specs import RAMP_IP_LEVELS_MA, ip_tag

_INK = "#1a1a17"
_MUTED = "#6b6a66"


def _ip_colors(n: int) -> np.ndarray:
    """Blue (low Ip) -> purple (high Ip) from matplotlib's plasma colormap."""
    return plt.cm.plasma(np.linspace(0.0, 0.42, n))


def _panel_title(name: str) -> str:
    words = []
    for w in name.split("_"):
        words.append(w.upper() if w in ("iter", "sparc", "prd") else w.capitalize())
    return " ".join(words)


def _lcfs(d: EQDSKInterface) -> np.ndarray:
    """True plasma boundary: the psi contour at (almost) the boundary flux."""
    r_grid = np.linspace(d.xgrid1, d.xgrid1 + d.xdim, d.nx)
    z_grid = np.linspace(d.zmid - d.zdim / 2, d.zmid + d.zdim / 2, d.nz)
    gen = contourpy.contour_generator(x=r_grid, y=z_grid, z=np.asarray(d.psi).T)
    # 0.995: at exactly psibdry the contour degenerates at the X-point, and
    # even at 0.999 the compact SPARC solves trace a spur down the divertor
    # leg; 0.5% inside is visually indistinguishable from the LCFS.
    level = d.psimag + 0.995 * (d.psibdry - d.psimag)
    loops = gen.lines(level)

    def area(loop: np.ndarray) -> float:
        x, z = loop[:, 0], loop[:, 1]
        return 0.5 * abs(np.dot(x, np.roll(z, 1)) - np.dot(z, np.roll(x, 1)))

    return max(loops, key=area)


@dataclass(frozen=True)
class Args:
    """CLI for the shape-evolution plot."""

    data_dir: Path = CONFIGS_DIR / "data"
    """Directory holding the generated .eqdsk files."""
    out_path: Path = PLOTS_DIR / "equilibrium_shapes.pdf"
    """Output figure path."""
    scenarios: tuple[str, ...] = tuple(RAMP_IP_LEVELS_MA)
    """Scenarios to plot (default: all in the manifest)."""


def _plot_scenario(ax: plt.Axes, name: str, data_dir: Path) -> None:
    levels = RAMP_IP_LEVELS_MA[name]
    first = True
    for ip_ma, color in zip(levels, _ip_colors(len(levels)), strict=True):
        d = EQDSKInterface.from_file(
            str(data_dir / f"{name}_{ip_tag(ip_ma)}.eqdsk"), no_cocos=True
        )
        if first:
            lim = np.column_stack([d.xlim, d.zlim])
            if not np.allclose(lim[0], lim[-1]):
                lim = np.vstack([lim, lim[:1]])
            ax.plot(lim[:, 0], lim[:, 1], color=_MUTED, lw=1.2, label="first wall")
            first = False
        lcfs = _lcfs(d)
        ax.plot(
            lcfs[:, 0],
            lcfs[:, 1],
            color=color,
            lw=2.0,
            label=f"{ip_ma:g} MA",
        )
    ax.set_title(_panel_title(name), color=_INK, fontsize=11)
    ax.set_xlabel("R [m]", fontsize=9)
    ax.set_ylabel("Z [m]", fontsize=9)
    ax.set_aspect("equal")
    ax.grid(color="#e8e7e3", lw=0.6)
    for spine in ax.spines.values():
        spine.set_color("#d4d3cf")
    ax.tick_params(colors=_MUTED, labelsize=8)
    ax.legend(fontsize=8, loc="upper right", framealpha=0.9, edgecolor="#d4d3cf")


def main(args: Args) -> None:
    fig, axes = plt.subplots(
        1, len(args.scenarios), figsize=(3.2 * len(args.scenarios), 6.0)
    )
    axes = np.atleast_1d(axes)
    for ax, name in zip(axes, args.scenarios, strict=True):
        _plot_scenario(ax, name, args.data_dir)
    fig.tight_layout()
    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_path, dpi=150, facecolor="white")
    print(f"wrote {args.out_path}")


if __name__ == "__main__":
    main(tyro.cli(Args))
