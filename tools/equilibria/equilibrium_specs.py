"""Dependency-free manifest for generated equilibrium files."""

__all__ = ["MID_BLEND", "RAMP_IP_LEVELS_MA", "STARTUP_SHAPES", "ip_tag"]

# Per-scenario plasma-current waypoints [MA] for the time-keyed ramp geometry.
# The mid waypoint sits just after each machine's limiter->divertor transition
# (ITER ~3.5-4 MA, SPARC ~1.5-2 MA), where the shape is genuinely intermediate;
# above it the LCFS holds the flat-top shape while only Ip keeps ramping.
RAMP_IP_LEVELS_MA: dict[str, tuple[float, ...]] = {
    "iter_baseline": (3.0, 4.5, 15.0),
    "iter_hybrid": (3.0, 4.5, 10.5),
    "iter_advanced": (3.0, 4.5, 9.0),
    "sparc_prd": (1.0, 2.0, 8.7),
    "sparc_reduced_field": (1.0, 2.0, 5.7),
}

# The mid waypoint's LCFS target is this fraction of the way from the limited
# startup boundary to the solved flat-top LCFS (0 = startup, 1 = flat-top).
MID_BLEND = 0.5

# Miller parameters (R0, a, kappa, delta, Z0) of the inboard-limited startup
# boundary solved at each scenario's LOWEST Ip waypoint; the higher waypoints
# keep the flat-top diverted LCFS. Shared between the FreeGSnke generator
# (which targets these) and the geometry regression tests (which read them
# back). R0 - a slightly overlaps the machine's inboard wall (ITER 4.105 m,
# SPARC 1.264 m) so the solve limits there.
_ITER_STARTUP = dict(R0=5.90, a=1.80, kappa=1.5, delta=0.15, Z0=0.3)
_SPARC_STARTUP = dict(R0=1.66, a=0.40, kappa=1.6, delta=0.15, Z0=0.0)
STARTUP_SHAPES: dict[str, dict[str, float]] = {
    "iter_baseline": _ITER_STARTUP,
    "iter_hybrid": _ITER_STARTUP,
    "iter_advanced": _ITER_STARTUP,
    "sparc_prd": _SPARC_STARTUP,
    "sparc_reduced_field": _SPARC_STARTUP,
}


def ip_tag(ip_ma: float) -> str:
    """Return the filename tag for a plasma-current level in MA."""
    return f"ip{round(ip_ma * 10):03d}"
