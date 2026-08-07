"""Validates the FreeGSnke-generated EQDSK geometries through TORAX.

These EQDSK files are committed build artifacts produced by
``tools/equilibria/generate_equilibrium.py`` (which needs FreeGSnke, a
separate venv).
This test needs only TORAX and the packaged EQDSK files, so it runs in the
normal project venv and acts as the regression gate on the geometry pipeline —
in particular it pins the COCOS convention (FreeGS4E writes COCOS 7) and
confirms each equilibrium reads back with physical R0/a/B0/Ip/q.

Each scenario ships a *sequence* of per-Ip equilibria (``{scenario}_ip{NNN}``,
NNN = round(Ip_MA*10)) interpolated through the current ramp by the env's
time-keyed ``geometry_configs`` block. The highest level is the flat-top
current. The lowest level is an inboard-limited startup plasma (smaller
R0/a, from ``STARTUP_SHAPES``); the others share the flat-top LCFS. B0 is
Ip-independent and checked on every slice;
the read-back Ip must match the slice's level; q is finite and positive
(the sharpest COCOS sign check) on every slice, and q95 sits in a sane range on
the flat-top slice (low-Ip slices have high q95 ~ 1/Ip, which is physical).
"""

import eqdsk
import numpy as np
import pytest
from torax._src.geometry import pydantic_model as geo_pm

from plasmax.environment.registry import CONFIGS_DIR
from tools.equilibria.equilibrium_specs import (
    MID_BLEND,
    RAMP_IP_LEVELS_MA,
    STARTUP_SHAPES,
    ip_tag,
)

_DATA_DIR = CONFIGS_DIR / "data"

# FreeGS4E's geqdsk writer emits full poloidal flux (psi not divided by 2pi)
# with sigma_RphiZ = +1 -> COCOS 7. TORAX auto-detects within {7,8,17,18} and
# converts internally to COCOS 11; 7 is the odd (no-warning), correct member.
_COCOS = 7

# Ip-independent machine parameters per scenario: (R0 [m], a [m], B0 [T]).
_MACHINE = {
    "iter_baseline": dict(R0=6.2, a=2.0, B0=5.3),
    "iter_hybrid": dict(R0=6.2, a=2.0, B0=5.3),
    "iter_advanced": dict(R0=6.2, a=2.0, B0=5.3),
    "sparc_prd": dict(R0=1.85, a=0.55, B0=12.2),
    "sparc_reduced_field": dict(R0=1.85, a=0.55, B0=8.0),
}

# Per-scenario Ip waypoints [MA]; the last (highest) level is the flat-top.
# Shared with tools/equilibria/generate_equilibrium.py through the dependency-free
# equilibrium manifest so generated filenames and regression coverage agree.
_LEVELS_MA = RAMP_IP_LEVELS_MA


def _slices():
    """(scenario, ip_ma, filename_stem) for every committed per-Ip equilibrium."""
    for scenario, levels in _LEVELS_MA.items():
        for ip_ma in levels:
            yield scenario, ip_ma, f"{scenario}_{ip_tag(ip_ma)}"


_SLICES = sorted(_slices())
_FLAT_TOP = {s: f"{s}_{ip_tag(max(levels))}" for s, levels in _LEVELS_MA.items()}


class GeometryEqdskTest:
    """Each committed per-Ip EQDSK builds a physical TORAX geometry at COCOS 7."""

    @pytest.mark.parametrize("scenario, ip_ma, stem", _SLICES)
    def test_eqdsk_file_exists(self, scenario, ip_ma, stem):
        assert (_DATA_DIR / f"{stem}.eqdsk").is_file()

    @pytest.mark.parametrize("scenario, ip_ma, stem", _SLICES)
    def test_geometry_is_physical(self, scenario, ip_ma, stem):
        machine = _MACHINE[scenario]
        cfg = geo_pm.Geometry.from_dict(
            {
                "geometry_type": "eqdsk",
                "geometry_file": f"{stem}.eqdsk",
                "geometry_directory": str(_DATA_DIR),
                "cocos": _COCOS,
                "Ip_from_parameters": False,  # read Ip straight from the file
            }
        )
        geo = cfg.build_provider(0.0)

        # The lowest waypoint is the inboard-limited startup shape, mid
        # waypoints are MID_BLEND of the way from startup to flat-top, and the
        # highest is the flat-top LCFS. The vacuum field R*B is machine-fixed,
        # so B_0 (reported at the plasma's R_major) scales as R0_machine/R_major.
        startup = STARTUP_SHAPES[scenario]
        levels = _LEVELS_MA[scenario]
        if ip_ma == min(levels):
            expected_r0, expected_a = startup["R0"], startup["a"]
        elif ip_ma == max(levels):
            expected_r0, expected_a = machine["R0"], machine["a"]
        else:
            expected_r0 = (1 - MID_BLEND) * startup["R0"] + MID_BLEND * machine["R0"]
            expected_a = (1 - MID_BLEND) * startup["a"] + MID_BLEND * machine["a"]
        np.testing.assert_allclose(geo.R_major, expected_r0, rtol=0.03, atol=0.0)
        np.testing.assert_allclose(geo.a_minor, expected_a, rtol=0.05, atol=0.0)
        expected_b0 = machine["B0"] * machine["R0"] / expected_r0
        np.testing.assert_allclose(geo.B_0, expected_b0, rtol=0.05, atol=0.0)

        # Read-back Ip must match this slice's level.
        ip_read = np.asarray(geo.Ip_profile_face)[-1] / 1e6
        np.testing.assert_allclose(ip_read, ip_ma, rtol=0.05, atol=0.0)

    @pytest.mark.parametrize("scenario, ip_ma, stem", _SLICES)
    def test_q_profile_is_finite_and_positive(self, scenario, ip_ma, stem):
        # Sign/finiteness of q is the sharpest COCOS check; applies to every
        # slice (magnitude varies ~1/Ip, so the range is checked separately).
        d = eqdsk.file.EQDSKInterface.from_file(
            str(_DATA_DIR / f"{stem}.eqdsk"), no_cocos=True
        )
        q = np.asarray(d.qpsi)
        assert np.all(np.isfinite(q))
        assert np.all(q > 0.0), f"{stem}: non-positive q (COCOS sign error?)"

    @pytest.mark.parametrize("scenario", sorted(_FLAT_TOP))
    def test_flat_top_q95_is_sane(self, scenario):
        # q95 in a plausible flat-top range (low-Ip ramp slices have high q95).
        d = eqdsk.file.EQDSKInterface.from_file(
            str(_DATA_DIR / f"{_FLAT_TOP[scenario]}.eqdsk"), no_cocos=True
        )
        q = np.asarray(d.qpsi)
        q95 = float(np.interp(0.95, np.linspace(0, 1, len(q)), q))
        assert 1.5 < q95 < 8.0, f"{scenario}: implausible q95={q95:.2f}"

    def test_scenarios_are_distinct(self):
        # Guard against every env silently sharing one equilibrium again:
        # the ITER trio flat-top equilibria must have distinct plasma currents.
        ips = {}
        for scenario in ("iter_baseline", "iter_hybrid", "iter_advanced"):
            d = eqdsk.file.EQDSKInterface.from_file(
                str(_DATA_DIR / f"{_FLAT_TOP[scenario]}.eqdsk"), no_cocos=True
            )
            ips[scenario] = abs(float(d.cplasma))
        assert ips["iter_baseline"] > ips["iter_hybrid"] > ips["iter_advanced"]
