"""Generate per-scenario tokamak equilibria (EQDSK g-files) with FreeGSnke.

Every ITER scenario in the packaged config tree historically shared TORAX's single
bundled CHEASE equilibrium (``iterhybrid.mat2cols``), which is only physically
right for the hybrid case. This script solves free-boundary Grad-Shafranov
equilibria with FreeGSnke and writes EQDSK files that TORAX ingests via
``geometry_type: eqdsk``. The ITER inverse-solve recipe follows FreeGSnke's
``example08`` (and SPARC follows ``example07``).

For each scenario it writes a *sequence* of equilibria at the Ip waypoints in
``RAMP_IP_LEVELS_MA`` (``{scenario}_ip{NNN}.eqdsk``, e.g.
``iter_hybrid_ip105.eqdsk`` for 10.5 MA). The env YAMLs stitch these into a
time-keyed ``geometry_configs`` block so the flux-surface geometry interpolates
through the current ramp — there is no on-the-fly Grad-Shafranov solve at
training time.

The *lowest* Ip waypoint is solved as an inboard-limited startup plasma
(smaller minor radius, reduced elongation/triangularity, no X-point
constraint) rather than the flat-top LCFS, mimicking a real ramp-up/-down: the
discharge starts and ends limited on the inner wall and diverts between the
first two waypoints. Mid/top waypoints keep the flat-top diverted shape.

IMPORTANT: FreeGSnke pins ``numpy<2`` while TORAX requires ``numpy>2`` — they
cannot share an environment. Run this script from the dedicated venv, NOT the
main project venv:

    tools/equilibria/.venv/bin/python \
        tools/equilibria/generate_equilibrium.py --scenario all

(See tools/equilibria/README.md for how that venv is created.) The generated
``.eqdsk`` files are committed artifacts, so training never needs FreeGSnke.

The machine descriptions (PF coils, passive structures, and limiter/wall) are
the vendored FreeGSnke pickles under
``tools/equilibria/machines/{ITER,SPARC}``. FreeGSnke receives the identical
limiter contour for both its limiter and wall inputs.

Usage::

    # everything
    tools/equilibria/.venv/bin/python tools/equilibria/generate_equilibrium.py
    # a single scenario
    tools/equilibria/.venv/bin/python tools/equilibria/generate_equilibrium.py \
        --scenario iter_hybrid
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

import numpy as np
import tyro

try:
    from .equilibrium_specs import (
        MID_BLEND,
        RAMP_IP_LEVELS_MA,
        STARTUP_SHAPES,
        ip_tag,
    )
except ImportError:  # Direct execution from tools/equilibria/.
    from equilibrium_specs import (  # type: ignore[no-redef]
        MID_BLEND,
        RAMP_IP_LEVELS_MA,
        STARTUP_SHAPES,
        ip_tag,
    )

_REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = _REPO_ROOT / "src" / "plasmax" / "configs"

# Scenarios this script knows how to build. "all" is a convenience alias.
ScenarioName = Literal[
    "iter_baseline",
    "iter_hybrid",
    "iter_advanced",
    "sparc_prd",
    "sparc_reduced_field",
]


@dataclass(frozen=True)
class StartupShape:
    """Miller-parameterized limited boundary for the lowest-Ip solve.

    ``R0 - a`` is chosen to slightly overlap the inboard limiter so the
    free-boundary solve clips on it -> a genuinely limited (not diverted)
    plasma, as in the early ramp-up / late ramp-down of a real discharge.

    The startup solve carries no X-point constraint; the boundary flux is
    pinned to the wall touch point via ``_pin_limited_boundary`` instead
    (FreeGSnke has no native no-X-point limited path — without the pin the
    core mask degenerates and current fills the vessel).
    """

    R0: float  # geometric major radius [m]
    a: float  # minor radius [m]
    kappa: float  # elongation
    delta: float  # triangularity
    Z0: float = 0.0  # vertical centroid [m]

    @property
    def touch_point(self) -> tuple[float, float]:
        """Inboard wall touch point: 1 cm inside the target surface."""
        return (self.R0 - self.a + 0.01, self.Z0)


def _miller_isoflux(shape: StartupShape, n: int = 32) -> list:
    """Isoflux target set on a Miller boundary: R0 + a*cos(t + delta*sin t)."""
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    r = shape.R0 + shape.a * np.cos(theta + shape.delta * np.sin(theta))
    z = shape.Z0 + shape.kappa * shape.a * np.sin(theta)
    return [[r.tolist(), z.tolist()]]


def _resample_polar(
    r: np.ndarray, z: np.ndarray, center: np.ndarray, thetas: np.ndarray
) -> np.ndarray:
    """Radius of a closed boundary at the given poloidal angles about center."""
    theta_pts = np.arctan2(z - center[1], r - center[0])
    rad_pts = np.hypot(r - center[0], z - center[1])
    order = np.argsort(theta_pts)
    theta_s, rad_s = theta_pts[order], rad_pts[order]
    theta_ext = np.concatenate([theta_s - 2 * np.pi, theta_s, theta_s + 2 * np.pi])
    return np.interp(thetas, theta_ext, np.tile(rad_s, 3))


def _blend_mid_targets(
    eq_top: Any, startup: StartupShape, frac: float, n: int = 32
) -> tuple[list, list[list[float]]]:
    """Mid-waypoint constraints: LCFS ``frac`` of the way from the limited
    startup boundary to the solved flat-top LCFS, with the X-point target
    blended onto the new boundary (an off-boundary null would leave the LCFS
    at the flat-top X-point surface again).

    Returns (isoflux_set, null_points).
    """
    mill = np.asarray(_miller_isoflux(startup, n=256)[0])
    flat = np.asarray(eq_top.separatrix(ntheta=256))
    center = 0.5 * (np.array([mill[0].mean(), mill[1].mean()]) + flat.mean(axis=0))
    # Theta grid anchored on the flat-top X-point so the blended corner (the
    # mid X-point target) is itself an isoflux point, as in the flat-top set.
    rx, zx = float(eq_top.xpt[0][0]), float(eq_top.xpt[0][1])
    theta_x = np.arctan2(zx - center[1], rx - center[0])
    thetas = theta_x + np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    thetas = np.mod(thetas + np.pi, 2.0 * np.pi) - np.pi
    rad = (1.0 - frac) * _resample_polar(
        mill[0], mill[1], center, thetas
    ) + frac * _resample_polar(flat[:, 0], flat[:, 1], center, thetas)
    r = center[0] + rad * np.cos(thetas)
    z = center[1] + rad * np.sin(thetas)
    ro, zo = float(eq_top.opt[0][0]), float(eq_top.opt[0][1])
    o_point = (1.0 - frac) * np.array([startup.R0, startup.Z0]) + frac * np.array(
        [ro, zo]
    )
    null_points = [[float(r[0]), float(o_point[0])], [float(z[0]), float(o_point[1])]]
    return [[r.tolist(), z.tolist()]], null_points


@dataclass(frozen=True)
class ScenarioSpec:
    """Everything needed to solve one equilibrium and write its EQDSK.

    The profile/constraint/solver values are seeded from FreeGSnke's shipped
    ITER (``example08``) and SPARC (``example07``) notebooks. The hybrid and
    advanced ITER variants reuse the baseline ITER machine + LCFS shape and
    differ only in plasma current and current-profile peaking — a first-pass
    approximation (see module docstring; rigorous q-profile/beta_N matching is
    a follow-up).
    """

    machine: str  # subdirectory under the machine-config root
    Rmin: float
    Rmax: float
    Zmin: float
    Zmax: float
    profile_kind: Literal["Fiesta_Topeol", "ConstrainPaxisIp"]
    profile_kwargs: dict[str, float]
    null_points: list[list[float]] | None
    isoflux_set: list  # nested list -> np.ndarray at solve time
    inverse_kwargs: dict[str, Any]
    forward_refine: bool
    label: str
    startup_shape: StartupShape | None = None  # limited shape for lowest Ip
    nx: int = 129
    ny: int = 129


# --------------------------------------------------------------------------- #
# ITER — shared machine + LCFS shape (example08). Baseline is example08 verbatim
# (Ip = 15 MA). Hybrid/advanced retune Ip + current-profile peaking only.
# --------------------------------------------------------------------------- #
_ITER_NULL_POINTS = [[5.02, 6.34], [-3.23, 0.66]]  # [[Rx, Ro], [Zx, Zo]]
_ITER_ISOFLUX = [
    [
        [
            4.25455147,
            4.1881875,
            4.2625,
            4.45683769,
            4.78746942,
            5.31835938,
            5.91875,
            6.44275166,
            6.92285156,
            7.35441013,
            7.73027344,
            8.03046875,
            8.19517497,
            8.05673414,
            7.75308013,
            7.37358451,
            6.94355469,
            6.47773438,
            5.99121094,
            5.49433594,
            5.02,
            4.79042969,
            4.56269531,
            4.36038615,
        ],
        [
            0.0,
            1.13554688,
            2.2565134,
            3.16757813,
            3.80507812,
            4.0499021,
            3.93089003,
            3.665625,
            3.31076604,
            2.86875,
            2.3199601,
            1.62832118,
            0.657421875,
            -0.35859375,
            -1.05585937,
            -1.59375,
            -2.03492727,
            -2.41356403,
            -2.75622016,
            -3.08699278,
            -3.23,
            -2.40548044,
            -1.55982142,
            -0.67734375,
        ],
    ]
]
_ITER_GRID = dict(Rmin=3.2, Rmax=8.8, Zmin=-5.0, Zmax=5.0)
# Limited startup boundary (lowest Ip waypoint), from the shared manifest.
# kappa/delta relaxed vs flat-top (1.75/0.4); near-full aperture, as ITER's
# limited phase ends (X-point formation ~3.5-4 MA) close to full size.
_ITER_STARTUP = StartupShape(**STARTUP_SHAPES["iter_baseline"])
# Inverse-solve knobs from example08 (robust for the ITER machine).
_ITER_INVERSE = dict(
    target_relative_tolerance=1e-4,
    target_relative_psit_update=1e-3,
    l2_reg=1e-14,
    Picard_handover=1e-4,
    full_jacobian_handover=[1e-3, 1e-2],
    verbose=True,
)

# SPARC limited startup, from the shared manifest. The compact machine ramps
# fast, so the 1 MA startup plasma is genuinely smaller (a: 0.55 -> 0.40) as
# well as less shaped than flat-top (kappa 2.07 -> 1.6, delta 0.5 -> 0.15).
_SPARC_STARTUP = StartupShape(**STARTUP_SHAPES["sparc_prd"])

SCENARIOS: dict[str, ScenarioSpec] = {
    # ITER baseline: 15 MA Q=10 inductive — example08 shape, physical field.
    # fvac = R0*B0 = 6.2*5.3 = 32.86 (example08's toy fvac=0.5 gives unphysical
    # B0~0.08 T and q<<1; we need the real ITER toroidal field).
    "iter_baseline": ScenarioSpec(
        machine="ITER",
        **_ITER_GRID,
        profile_kind="Fiesta_Topeol",
        profile_kwargs=dict(
            Beta0=0.5978,
            Ip=15e6,
            fvac=32.86,
            Raxis=6.2,
            alpha_m=2.0,
            alpha_n=1.395,
        ),
        null_points=_ITER_NULL_POINTS,
        isoflux_set=_ITER_ISOFLUX,
        inverse_kwargs=_ITER_INVERSE,
        forward_refine=False,
        startup_shape=_ITER_STARTUP,
        label="ITER baseline (15 MA Q=10 inductive)",
    ),
    # ITER hybrid: ~10.5 MA, broader/flatter core current (lower alpha_m) for a
    # flat low-shear core q>~1. Higher Beta0 (hybrid runs at higher beta_N).
    "iter_hybrid": ScenarioSpec(
        machine="ITER",
        **_ITER_GRID,
        profile_kind="Fiesta_Topeol",
        profile_kwargs=dict(
            Beta0=0.7,
            Ip=10.5e6,
            fvac=32.86,
            Raxis=6.2,
            alpha_m=1.5,
            alpha_n=1.4,
        ),
        null_points=_ITER_NULL_POINTS,
        isoflux_set=_ITER_ISOFLUX,
        inverse_kwargs=_ITER_INVERSE,
        forward_refine=False,
        startup_shape=_ITER_STARTUP,
        label="ITER hybrid (~10.5 MA, flat low-shear core)",
    ),
    # ITER advanced/steady-state: ~8 MA, lower current for a weak-shear,
    # q_min>1 (no-sawtooth) regime. NOTE: Fiesta_Topeol is monotonic, so a true
    # reversed-shear ITB profile (off-axis q_min) is not captured here — this is
    # the most approximate of the three (see module docstring).
    "iter_advanced": ScenarioSpec(
        machine="ITER",
        **_ITER_GRID,
        profile_kind="Fiesta_Topeol",
        profile_kwargs=dict(
            Beta0=0.85,
            Ip=8.0e6,
            fvac=32.86,
            Raxis=6.2,
            alpha_m=1.8,
            alpha_n=1.1,
        ),
        null_points=_ITER_NULL_POINTS,
        isoflux_set=_ITER_ISOFLUX,
        inverse_kwargs=_ITER_INVERSE,
        forward_refine=False,
        startup_shape=_ITER_STARTUP,
        label="ITER advanced (~8 MA, weak-shear, q_min>1)",
    ),
    # SPARC PRD (Primary Reference Discharge) — example07 verbatim (compact
    # high-field H-mode, 12.2 T / 8.7 MA, Q~11).
    "sparc_prd": ScenarioSpec(
        machine="SPARC",
        Rmin=1.1,
        Rmax=2.7,
        Zmin=-1.8,
        Zmax=1.8,
        # fvac = R0*B0 = 1.85*12.2 = 22.57 (SPARC compact high-field).
        profile_kind="ConstrainPaxisIp",
        profile_kwargs=dict(
            paxis=5e4,
            Ip=8.7e6,
            fvac=22.57,
            alpha_m=1.8,
            alpha_n=1.2,
        ),
        null_points=[[1.55, 1.55], [1.15, -1.15]],
        isoflux_set=[
            [[1.55, 1.55, 2.4, 1.3, 1.7, 1.7], [1.15, -1.15, 0.0, 0.0, 1.5, -1.5]]
        ],
        # SPARC has 11 control coils; regularise the last one (vertical-stability)
        # more strongly, per example07.
        inverse_kwargs=dict(
            target_relative_tolerance=1e-6,
            target_relative_psit_update=1e-3,
            l2_reg=np.array([1e-16] * 10 + [1e-5]),
            verbose=False,
        ),
        forward_refine=True,
        startup_shape=_SPARC_STARTUP,
        label="SPARC PRD (8.7 MA, 12.2 T)",
    ),
    # SPARC reduced-field H-mode ("H8") — early-operation scenario at 8 T; Ip
    # lowered to 5.7 MA to hold q*. Same SPARC machine + LCFS shape as the PRD;
    # only the toroidal field (fvac = 1.85*8.0 = 14.8) and Ip change.
    "sparc_reduced_field": ScenarioSpec(
        machine="SPARC",
        Rmin=1.1,
        Rmax=2.7,
        Zmin=-1.8,
        Zmax=1.8,
        profile_kind="ConstrainPaxisIp",
        profile_kwargs=dict(
            paxis=5e4,
            Ip=5.7e6,
            fvac=14.8,
            alpha_m=1.8,
            alpha_n=1.2,
        ),
        null_points=[[1.55, 1.55], [1.15, -1.15]],
        isoflux_set=[
            [[1.55, 1.55, 2.4, 1.3, 1.7, 1.7], [1.15, -1.15, 0.0, 0.0, 1.5, -1.5]]
        ],
        inverse_kwargs=dict(
            target_relative_tolerance=1e-6,
            target_relative_psit_update=1e-3,
            l2_reg=np.array([1e-16] * 10 + [1e-5]),
            verbose=False,
        ),
        forward_refine=True,
        startup_shape=_SPARC_STARTUP,
        label="SPARC reduced-field (5.7 MA, 8 T)",
    ),
}


def _build_profiles(spec: ScenarioSpec, eq: Any) -> Any:
    """Construct the FreeGSnke profile object for ``spec`` bound to ``eq``."""
    from freegsnke import jtor_update

    cls = getattr(jtor_update, spec.profile_kind)
    return cls(eq=eq, **spec.profile_kwargs)


def _pin_limited_boundary(profiles: Any, touch_r: float, touch_z: float) -> None:
    """Make every Jtor evaluation use the flux at the wall touch point as the
    plasma boundary flux (limited plasma, no X-point)."""
    from scipy.interpolate import RectBivariateSpline

    orig_jtor = profiles.Jtor

    def jtor_limited(
        R: Any, Z: Any, psi: Any, psi_bndry: Any = None, **kwargs: Any
    ) -> Any:
        psi_touch = float(RectBivariateSpline(R[:, 0], Z[0, :], psi)(touch_r, touch_z))
        jtor = orig_jtor(R, Z, psi, psi_bndry=psi_touch, **kwargs)
        if not np.all(np.isfinite(jtor)) or not np.any(jtor):
            # Early Picard iterations can put the pin flux outside the
            # axis-boundary range (empty core mask, I_R = 0, NaN spiral —
            # this is fatal for the reduced-field SPARC mid solve). Fall
            # back to the unpinned Jtor for this iteration; the pin
            # re-engages once the flux map is sane.
            jtor = orig_jtor(R, Z, psi, psi_bndry=psi_bndry, **kwargs)
        return jtor

    profiles.Jtor = jtor_limited


def _solve_one(
    spec: ScenarioSpec,
    machine_root: Path,
    boundary_pin: tuple[float, float] | None = None,
) -> Any:
    """Build the machine, solve the (inverse) equilibrium, return the eq."""
    from freegsnke import GSstaticsolver, build_machine, equilibrium_update
    from freegsnke.inverse import Inverse_optimizer

    mdir = machine_root / spec.machine
    tokamak = build_machine.tokamak(
        active_coils_path=str(mdir / f"{spec.machine}_active_coils.pickle"),
        passive_coils_path=str(mdir / f"{spec.machine}_passive_coils.pickle"),
        limiter_path=str(mdir / f"{spec.machine}_limiter.pickle"),
        # The vendored upstream wall and limiter pickles were byte-identical.
        # Reuse the one retained artifact instead of packaging duplicate data.
        wall_path=str(mdir / f"{spec.machine}_limiter.pickle"),
    )

    eq = equilibrium_update.Equilibrium(
        tokamak=tokamak,
        Rmin=spec.Rmin,
        Rmax=spec.Rmax,
        Zmin=spec.Zmin,
        Zmax=spec.Zmax,
        nx=spec.nx,
        ny=spec.ny,
    )

    profiles = _build_profiles(spec, eq)
    if boundary_pin is not None:
        _pin_limited_boundary(profiles, *boundary_pin)
    solver = GSstaticsolver.NKGSsolver(eq)
    constrain = Inverse_optimizer(
        null_points=spec.null_points,
        isoflux_set=np.array(spec.isoflux_set),
    )

    solver.inverse_solve(
        eq=eq, profiles=profiles, constrain=constrain, **spec.inverse_kwargs
    )
    if spec.forward_refine:
        # Currents are now known; tighten with a forward solve (example07).
        solver.solve(
            eq=eq,
            profiles=profiles,
            constrain=None,
            target_relative_tolerance=1e-9,
            verbose=False,
        )
    return eq


def _summarize(eq: Any) -> str:
    """Best-effort one-line physics summary for a sanity check."""
    bits = [f"Ip={eq.plasmaCurrent() / 1e6:.2f} MA"]
    for name, fn in (("betaP", "poloidalBeta"), ("q95", None)):
        try:
            if name == "q95":
                bits.append(f"q95={float(eq.q(0.95)):.2f}")
            else:
                bits.append(f"{name}={float(getattr(eq, fn)()):.3f}")
        except Exception:  # noqa: BLE001 - summary only, never fatal
            pass
    return ", ".join(bits)


class _CroppedEq:
    """Present ``eq`` to the EQDSK writer on a smaller psi grid.

    Shadows the grid attributes and ``psi()``; everything else (profiles,
    separatrix, q, tokamak) delegates to the real equilibrium.
    """

    def __init__(
        self, eq: Any, rmin: float, rmax: float, zmin: float, zmax: float
    ) -> None:
        from scipy.interpolate import RectBivariateSpline

        self._eq = eq
        self.Rmin, self.Rmax, self.Zmin, self.Zmax = rmin, rmax, zmin, zmax
        r = np.linspace(rmin, rmax, eq.R.shape[0])
        z = np.linspace(zmin, zmax, eq.Z.shape[1])
        self.R, self.Z = np.meshgrid(r, z, indexing="ij")
        spline = RectBivariateSpline(eq.R[:, 0], eq.Z[0, :], eq.psi())
        self._psi = spline(r, z)

    def psi(self) -> Any:
        return self._psi

    def separatrix(self, ntheta: int = 101, **kwargs: Any) -> Any:
        # TORAX masks psi to the written boundary's bounding box + 1 cm and
        # clips contours at masked grid NODES, so the box must clear the
        # plasma by at least one grid spacing (~4 cm here, > the 1 cm
        # offset). Sample densely (101 points undersample the flat limited
        # sections by >1 cm) and inflate 2% about the centroid — the written
        # boundary's only structural consumer is that mask bbox; TORAX
        # derives the actual geometry from the psi map.
        pts = self._eq.separatrix(ntheta=max(ntheta, 720), **kwargs)
        return pts.mean(axis=0) + 1.02 * (pts - pts.mean(axis=0))

    def __getattr__(self, name: str) -> Any:
        return getattr(self._eq, name)


def _write_eqdsk(eq: Any, out_path: Path, label: str, pinned: bool = False) -> None:
    """Write ``eq`` to an EQDSK g-file via FreeGS4E's geqdsk writer."""
    from freegs4e import critical, geqdsk

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not pinned:
        with out_path.open("w") as fh:
            geqdsk.write(eq, fh, label=label[:48])
        return

    # Limited plasma (startup waypoint): the LCFS is the pinned touch-point
    # surface, but freegs4e's writer, separatrix() and q() all take the
    # boundary flux from xpoint[0][2] — and any X-points in the field lie
    # OUTSIDE the LCFS. Stand in a pseudo X-point carrying the limiter
    # boundary flux for the duration of the write; only its flux value (and,
    # harmlessly, its poloidal angle from the O-point) are ever read.
    psi_b = float(eq._profiles.psi_bndry)
    r0, z0 = eq.opt[0][:2]
    pseudo = np.array([[r0, z0 - 1.0, psi_b]])
    eq.xpt = pseudo
    eq.psi_bndry = psi_b
    # The user-psi_bndry Jtor path never stores psi_axis on the profiles
    # object, but profiles.fpol()/pressure() (used by the writer) need it.
    if not hasattr(eq._profiles, "psi_axis"):
        eq._profiles.psi_axis = eq.psi_axis
    orig_find_critical = critical.find_critical

    def _with_pseudo_xpoint(*args: Any, **kwargs: Any) -> tuple[Any, Any]:
        opoint, _ = orig_find_critical(*args, **kwargs)
        return opoint, pseudo

    # Crop the written psi grid to a box around the LCFS: the full solve box
    # contains vacuum flux structure near the PF coils (open arcs at the grid
    # edge) in the same psi range as the core surfaces of this small plasma,
    # and TORAX's contour selection takes the first loop it finds.
    bdry = eq.separatrix(ntheta=101)
    pad_r = 0.4 * (bdry[:, 0].max() - bdry[:, 0].min()) / 2.0
    pad = max(pad_r, 0.3)
    cropped = _CroppedEq(
        eq,
        rmin=max(eq.Rmin, bdry[:, 0].min() - pad),
        rmax=min(eq.Rmax, bdry[:, 0].max() + pad),
        zmin=max(eq.Zmin, bdry[:, 1].min() - pad),
        zmax=min(eq.Zmax, bdry[:, 1].max() + pad),
    )

    critical.find_critical = _with_pseudo_xpoint
    try:
        with out_path.open("w") as fh:
            geqdsk.write(cropped, fh, label=label[:48])
    finally:
        critical.find_critical = orig_find_critical


@dataclass(frozen=True)
class Args:
    """CLI for the equilibrium generator."""

    scenario: ScenarioName | Literal["all"] = "all"
    """Which scenario to generate (default: all)."""
    out_dir: Path = CONFIGS_DIR / "data"
    """Directory the .eqdsk files are written to."""
    machine_dir: Path = Path(__file__).resolve().parent / "machines"
    """Root holding the vendored {ITER,SPARC} machine-config pickles."""


def main(args: Args) -> None:
    names = list(SCENARIOS) if args.scenario == "all" else [args.scenario]
    for name in names:
        spec = SCENARIOS[name]
        levels = RAMP_IP_LEVELS_MA[name]
        print(f"\n=== {name}: {spec.label} — {len(levels)} Ip levels {levels} MA ===")

        def _run(
            ip_ma: float,
            level_spec: ScenarioSpec,
            boundary_pin: tuple[float, float] | None = None,
            name: str = name,
        ) -> Any:
            eq = _solve_one(level_spec, args.machine_dir, boundary_pin)
            out_path = args.out_dir / f"{name}_{ip_tag(ip_ma)}.eqdsk"
            _write_eqdsk(
                eq, out_path, level_spec.label, pinned=boundary_pin is not None
            )
            print(f"  {ip_ma:.1f} MA: {_summarize(eq)}  -> {out_path.name}")
            return eq

        def _at(
            ip_ma: float, spec: ScenarioSpec = spec, **overrides: Any
        ) -> ScenarioSpec:
            return replace(
                spec,
                profile_kwargs={**spec.profile_kwargs, "Ip": ip_ma * 1e6},
                **overrides,
            )

        # Flat-top first: its solved LCFS seeds the mid-level blend. The
        # lowest waypoint is the limited startup shape; mid waypoints target
        # a boundary MID_BLEND of the way from startup to flat-top.
        top = levels[-1]
        eq_top = _run(top, _at(top, label=f"{spec.label} @ {top:.1f} MA"))
        startup = spec.startup_shape
        low = levels[0]
        _run(
            low,
            _at(
                low,
                isoflux_set=_miller_isoflux(startup),
                null_points=None,
                label=f"{spec.label} @ {low:.1f} MA (limited startup)",
            ),
            boundary_pin=startup.touch_point,
        )
        for mid in levels[1:-1]:
            isoflux, nulls = _blend_mid_targets(eq_top, startup, MID_BLEND)
            # Pin the boundary flux to the blended X-point corner: isoflux +
            # null constraints alone don't tie the LCFS to the target ring,
            # and the free boundary drifts outward until it limits on the
            # inboard wall (~4 cm past the target).
            # Blended targets don't merit SPARC's 1e-6 inverse tolerance: the
            # pinned reduced-field mid oscillates just above it for the full
            # 100-iteration budget (~30 min). 1e-4 (the ITER setting) is
            # plenty for an interpolation waypoint.
            mid_inverse = {
                **spec.inverse_kwargs,
                "target_relative_tolerance": max(
                    spec.inverse_kwargs.get("target_relative_tolerance", 1e-4),
                    1e-4,
                ),
            }
            _run(
                mid,
                _at(
                    mid,
                    isoflux_set=isoflux,
                    null_points=nulls,
                    inverse_kwargs=mid_inverse,
                    label=f"{spec.label} @ {mid:.1f} MA (blended mid)",
                ),
                boundary_pin=(nulls[0][0], nulls[1][0]),
            )


if __name__ == "__main__":
    main(tyro.cli(Args))
