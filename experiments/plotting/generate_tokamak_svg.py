"""Generate a mathematical tokamak cutaway as a standalone SVG or PDF.

The output format follows the ``--output`` suffix; ``.pdf`` gives a vector
figure for ``\\includegraphics``.

Usage:
    uv run python experiments/plotting/generate_tokamak_svg.py
    uv run python experiments/plotting/generate_tokamak_svg.py \
        --output plots/my_tokamak.pdf
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree

import tyro

from scripts.project_paths import PLOTS_DIR


@dataclass(frozen=True)
class Args:
    output: Path = PLOTS_DIR / "tokamak_cutaway.svg"
    width: int = 1200
    height: int = 800


Vec3 = tuple[float, float, float]
# depth (larger is nearer the camera), projected screen points, fill colour
Patch = tuple[float, tuple[tuple[float, float], ...], str]
Group = tuple[str, float, list[Patch]]

_VIEW_WIDTH = 1200
_VIEW_HEIGHT = 800
_CENTER_X = 600.0
_CENTER_Y = 439.0
_SCALE = 106.0
_ELEVATION = math.radians(24.0)
_CAMERA = (0.0, -math.cos(_ELEVATION), math.sin(_ELEVATION))
_SCREEN_UP = (0.0, math.sin(_ELEVATION), math.cos(_ELEVATION))
_LIGHT = (-0.35, -0.48, 0.80)
_MAJOR_RADIUS = 2.75
_VESSEL_RADIUS = 1.10
_VESSEL_INNER_RADIUS = 0.88
_PLASMA_RADIUS = 0.62
_ELONGATION = 1.75
_PLASMA_TRIANGULARITY = 0.42
_CUTAWAY_HALF_ANGLE = math.radians(72.0)
_COIL_COLOR = (52, 96, 196)
_COIL_COUNT = 5
_COIL_INNER_RADIUS = 1.25
_COIL_OUTER_RADIUS = 4.15
_COIL_HALF_HEIGHT = 2.32
_COIL_HALF_WIDTH = 0.14
_COIL_HALF_DEPTH = 0.19
_COIL_SEGMENTS = 72


def _dot(a: Vec3, b: Vec3) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def _project(point: Vec3) -> tuple[float, float, float]:
    x, _, _ = point
    screen_x = _CENTER_X + _SCALE * x
    screen_y = _CENTER_Y - _SCALE * _dot(point, _SCREEN_UP)
    return screen_x, screen_y, _dot(point, _CAMERA)


def _patch(points_3d: tuple[Vec3, ...], color: str) -> Patch:
    projected = [_project(point) for point in points_3d]
    depth = sum(point[2] for point in projected) / len(projected)
    return depth, tuple((x, y) for x, y, _ in projected), color


def _torus_point(
    major_radius: float,
    minor_radius: float,
    u: float,
    v: float,
    triangularity: float = 0.0,
) -> Vec3:
    radial = major_radius + minor_radius * math.cos(v + triangularity * math.sin(v))
    vertical = _ELONGATION * minor_radius * math.sin(v)
    return radial * math.cos(u), radial * math.sin(u), vertical


def _torus_normal(u: float, v: float, triangularity: float = 0.0) -> Vec3:
    # outward normal of the elongated, D-shaped poloidal cross-section
    radial = _ELONGATION * math.cos(v)
    vertical = math.sin(v + triangularity * math.sin(v)) * (
        1.0 + triangularity * math.cos(v)
    )
    length = math.hypot(radial, vertical)
    return (
        radial * math.cos(u) / length,
        radial * math.sin(u) / length,
        vertical / length,
    )


def _shade(base: tuple[int, int, int], normal: Vec3) -> str:
    diffuse = max(0.0, _dot(normal, _LIGHT))
    brightness = 0.68 + 0.43 * diffuse
    specular = 26.0 * diffuse**8
    channels = [min(255, round(channel * brightness + specular)) for channel in base]
    return "#" + "".join(f"{channel:02x}" for channel in channels)


def _angle_distance(a: float, b: float) -> float:
    return abs((a - b + math.pi) % (2.0 * math.pi) - math.pi)


def _torus_patches(
    major_radius: float,
    minor_radius: float,
    base_color: tuple[int, int, int],
    *,
    cutaway: bool,
    back_only: bool = False,
    reverse_faces: bool = False,
    omit_u_ranges: tuple[tuple[float, float], ...] = (),
    triangularity: float = 0.0,
    u_steps: int,
    v_steps: int,
) -> tuple[list[Patch], list[Patch]]:
    opaque: list[Patch] = []
    transparent: list[Patch] = []
    for u_index in range(u_steps):
        u0 = 2.0 * math.pi * u_index / u_steps
        u1 = 2.0 * math.pi * (u_index + 1) / u_steps
        u_mid = (u0 + u1) / 2.0
        if any(start <= u_mid < end for start, end in omit_u_ranges):
            continue
        is_cutaway = cutaway and (
            _angle_distance(u_mid, 3.0 * math.pi / 2.0) < _CUTAWAY_HALF_ANGLE
        )
        if back_only and is_cutaway:
            continue
        for v_index in range(v_steps):
            v0 = 2.0 * math.pi * v_index / v_steps
            v1 = 2.0 * math.pi * (v_index + 1) / v_steps
            v_mid = (v0 + v1) / 2.0
            normal = _torus_normal(u_mid, v_mid, triangularity)
            visible_normal = (
                tuple(-value for value in normal) if reverse_faces else normal
            )
            if _dot(visible_normal, _CAMERA) < -0.04:
                continue

            points_3d = (
                _torus_point(major_radius, minor_radius, u0, v0, triangularity),
                _torus_point(major_radius, minor_radius, u1, v0, triangularity),
                _torus_point(major_radius, minor_radius, u1, v1, triangularity),
                _torus_point(major_radius, minor_radius, u0, v1, triangularity),
            )
            patch = _patch(points_3d, _shade(base_color, visible_normal))
            (transparent if is_cutaway and not back_only else opaque).append(patch)
    return opaque, transparent


def _cut_face(u: float, normal_sign: float) -> list[Patch]:
    tangent = (-math.sin(u) * normal_sign, math.cos(u) * normal_sign, 0.0)
    if _dot(tangent, _CAMERA) < 0.0:
        tangent = tuple(-value for value in tangent)
    outer_radius = _VESSEL_RADIUS * 1.09 * 0.95
    rim_width = (_VESSEL_RADIUS - _VESSEL_INNER_RADIUS) * 1.09 * 1.10 * 1.03
    inner_radius = outer_radius - rim_width

    patches = []
    for index in range(40):
        v0 = 2.0 * math.pi * index / 40
        v1 = 2.0 * math.pi * (index + 1) / 40
        points_3d = (
            _torus_point(_MAJOR_RADIUS, outer_radius, u, v0),
            _torus_point(_MAJOR_RADIUS, outer_radius, u, v1),
            _torus_point(_MAJOR_RADIUS, inner_radius, u, v1),
            _torus_point(_MAJOR_RADIUS, inner_radius, u, v0),
        )
        patches.append(_patch(points_3d, _shade((205, 211, 216), tangent)))
    return patches


def _d_shape(t: float) -> tuple[float, float]:
    """Closed D-shaped poloidal contour: flat inner leg, rounded outer arc."""
    center = (_COIL_INNER_RADIUS + _COIL_OUTER_RADIUS) / 2.0
    amplitude = (_COIL_OUTER_RADIUS - _COIL_INNER_RADIUS) / 2.0
    cos_t, sin_t = math.cos(t), math.sin(t)
    # ponytail: superellipse with a flatter exponent on the inboard half; a true
    # Princeton-D constant-tension curve if the shape ever needs to be physical.
    exponent = 0.95 if cos_t >= 0.0 else 0.30
    radial = center + amplitude * math.copysign(abs(cos_t) ** exponent, cos_t)
    vertical = _COIL_HALF_HEIGHT * math.copysign(abs(sin_t) ** 0.82, sin_t)
    return radial, vertical


def _coil_patches(phi: float) -> list[Patch]:
    """One D-shaped toroidal field coil, extruded as a rectangular conductor."""
    cos_phi, sin_phi = math.cos(phi), math.sin(phi)
    toroidal = (-sin_phi, cos_phi, 0.0)
    curve = [
        _d_shape(2.0 * math.pi * index / _COIL_SEGMENTS)
        for index in range(_COIL_SEGMENTS)
    ]
    normals = []
    for index in range(_COIL_SEGMENTS):
        radial_prev, vertical_prev = curve[index - 1]
        radial_next, vertical_next = curve[(index + 1) % _COIL_SEGMENTS]
        tangent_r, tangent_z = radial_next - radial_prev, vertical_next - vertical_prev
        length = math.hypot(tangent_r, tangent_z) or 1.0
        normals.append((tangent_z / length, -tangent_r / length))

    def corner(index: int, width_sign: float, depth_sign: float) -> Vec3:
        radial, vertical = curve[index]
        normal_r, normal_z = normals[index]
        radial += depth_sign * _COIL_HALF_DEPTH * normal_r
        vertical += depth_sign * _COIL_HALF_DEPTH * normal_z
        offset = width_sign * _COIL_HALF_WIDTH
        return (
            radial * cos_phi + offset * toroidal[0],
            radial * sin_phi + offset * toroidal[1],
            vertical,
        )

    patches: list[Patch] = []
    for index in range(_COIL_SEGMENTS):
        next_index = (index + 1) % _COIL_SEGMENTS
        normal_r = (normals[index][0] + normals[next_index][0]) / 2.0
        normal_z = (normals[index][1] + normals[next_index][1]) / 2.0
        poloidal_normal = (normal_r * cos_phi, normal_r * sin_phi, normal_z)
        faces = (
            (poloidal_normal, ((-1.0, 1.0), (1.0, 1.0))),
            (tuple(-value for value in poloidal_normal), ((1.0, -1.0), (-1.0, -1.0))),
            (toroidal, ((1.0, -1.0), (1.0, 1.0))),
            (tuple(-value for value in toroidal), ((-1.0, 1.0), (-1.0, -1.0))),
        )
        for normal, (first, second) in faces:
            if _dot(normal, _CAMERA) < 0.0:
                continue
            points_3d = (
                corner(index, *first),
                corner(index, *second),
                corner(next_index, *second),
                corner(next_index, *first),
            )
            patches.append(_patch(points_3d, _shade(_COIL_COLOR, normal)))
    return patches


def _plasma_profile(u: float) -> list[Patch]:
    core_color = (255, 186, 233)
    edge_color = (232, 15, 145)
    radial_steps = 7
    angular_steps = 48
    profile_radius = _PLASMA_RADIUS * 1.05
    patches = []

    for radial_index in range(radial_steps):
        radius0 = profile_radius * radial_index / radial_steps
        radius1 = profile_radius * (radial_index + 1) / radial_steps
        profile_position = (radial_index + 0.5) / radial_steps
        color = "#" + "".join(
            f"{round(core + profile_position * (edge - core)):02x}"
            for core, edge in zip(core_color, edge_color, strict=True)
        )
        for angular_index in range(angular_steps):
            v0 = 2.0 * math.pi * angular_index / angular_steps
            v1 = 2.0 * math.pi * (angular_index + 1) / angular_steps
            points_3d = (
                _torus_point(_MAJOR_RADIUS, radius0, u, v0, _PLASMA_TRIANGULARITY),
                _torus_point(_MAJOR_RADIUS, radius1, u, v0, _PLASMA_TRIANGULARITY),
                _torus_point(_MAJOR_RADIUS, radius1, u, v1, _PLASMA_TRIANGULARITY),
                _torus_point(_MAJOR_RADIUS, radius0, u, v1, _PLASMA_TRIANGULARITY),
            )
            patches.append(_patch(points_3d, color))
    return patches


def _build_scene() -> list[Group]:
    """Depth-sorted draw groups, painted back to front."""
    left_cut_u = 3.0 * math.pi / 2.0 - _CUTAWAY_HALF_ANGLE
    right_cut_u = 3.0 * math.pi / 2.0 + _CUTAWAY_HALF_ANGLE
    vessel, transparent_vessel = _torus_patches(
        _MAJOR_RADIUS,
        _VESSEL_RADIUS,
        (158, 164, 169),
        cutaway=True,
        u_steps=84,
        v_steps=28,
    )
    interior, _ = _torus_patches(
        _MAJOR_RADIUS,
        _VESSEL_INNER_RADIUS,
        (88, 103, 114),
        cutaway=True,
        back_only=True,
        reverse_faces=True,
        u_steps=84,
        v_steps=28,
    )
    plasma, _ = _torus_patches(
        _MAJOR_RADIUS,
        _PLASMA_RADIUS,
        (255, 45, 176),
        cutaway=True,
        # the whole wedge of plasma is removed; both cut planes show its profile
        omit_u_ranges=((left_cut_u, right_cut_u),),
        triangularity=_PLASMA_TRIANGULARITY,
        u_steps=84,
        v_steps=22,
    )
    cut_faces = _cut_face(
        left_cut_u,
        normal_sign=1.0,
    ) + _cut_face(
        right_cut_u,
        normal_sign=-1.0,
    )
    poloidal_profile = _plasma_profile(left_cut_u) + _plasma_profile(right_cut_u)
    coils: list[Patch] = []
    for index in range(_COIL_COUNT):
        # anchored on the right cut plane so a coil frames each edge of the cut
        phi = (right_cut_u + 2.0 * math.pi * index / _COIL_COUNT) % (2.0 * math.pi)
        if left_cut_u + 1e-6 < phi < right_cut_u - 1e-6:
            continue  # coils inside the cutaway wedge would hide the plasma
        coils += _coil_patches(phi)
    return [
        ("solid-tori", 1.0, sorted(vessel + interior + plasma + coils)),
        ("cut-faces", 1.0, sorted(cut_faces)),
        ("poloidal-profile", 1.0, sorted(poloidal_profile)),
        ("transparent-front", 0.07, sorted(transparent_vessel)),
    ]


def build_svg(width: int, height: int) -> str:
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")

    lines = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {_VIEW_WIDTH} {_VIEW_HEIGHT}" '
            'preserveAspectRatio="xMidYMid meet" role="img" '
            'aria-label="Tokamak vessel with D-shaped field coils and '
            'exposed plasma">'
        ),
        f'<rect width="{_VIEW_WIDTH}" height="{_VIEW_HEIGHT}" fill="#ffffff"/>',
    ]
    for name, opacity, patches in _build_scene():
        lines.append(f'<g id="{name}" opacity="{opacity:g}">')
        lines += [
            '<polygon points="{}" fill="{}" stroke="{}" stroke-width="0.8"/>'.format(
                " ".join(f"{x:.1f},{y:.1f}" for x, y in points), color, color
            )
            for _, points, color in patches
        ]
        lines.append("</g>")
    lines.append("</svg>")
    return "\n".join(lines)


def write_pdf(path: Path, width: int, height: int) -> None:
    """Vector PDF for LaTeX \\includegraphics, same geometry as the SVG."""
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")

    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.collections import PolyCollection
    from matplotlib.figure import Figure

    figure = Figure(figsize=(width / 100.0, height / 100.0), facecolor="white")
    axes = figure.add_axes((0.0, 0.0, 1.0, 1.0), facecolor="white")
    axes.set_xlim(0.0, _VIEW_WIDTH)
    axes.set_ylim(_VIEW_HEIGHT, 0.0)
    axes.set_aspect("equal")
    axes.set_axis_off()
    # SVG stroke-width 0.8 in view units, expressed in points
    line_width = 0.8 * 72.0 * width / 100.0 / _VIEW_WIDTH
    for _, opacity, patches in _build_scene():
        colors = [color for _, _, color in patches]
        # SVG applies group opacity once to the flattened group; matplotlib applies
        # it per polygon, so drop the seam strokes there to avoid a visible mesh.
        solid = opacity >= 1.0
        axes.add_collection(
            PolyCollection(
                [points for _, points, _ in patches],
                facecolors=colors,
                edgecolors=colors if solid else "none",
                linewidths=line_width if solid else 0.0,
                alpha=None if solid else opacity,
            )
        )
    figure.savefig(path, format="pdf", facecolor="white")


def main(args: Args) -> None:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.suffix.lower() == ".pdf":
        write_pdf(args.output, args.width, args.height)
    else:
        svg = build_svg(args.width, args.height)
        ElementTree.fromstring(svg)
        args.output.write_text(svg, encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main(tyro.cli(Args))
