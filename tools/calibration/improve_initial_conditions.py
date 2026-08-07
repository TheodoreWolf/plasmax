"""Build and verify backend-independent physical reset references.

This replaces the retired backend-relaxation search.  A reference is imported
or digitized once, then supplied unchanged to every compatible simulator
backend.  This module deliberately contains no environment construction or
transition calls: reference construction can never relax a profile through a
backend or write a simulated ``psi`` back into YAML.

Examples::

    # Verify packaged provenance, reset hashes, and derived artifacts.
    uv run python tools/calibration/improve_initial_conditions.py

    # Checksum-download non-redistributable inputs into the ignored cache,
    # regenerate the ITPA-derived CSV/EQDSK, and render digitization overlays.
    uv run python tools/calibration/improve_initial_conditions.py \
        --download-missing --write-derived --render-overlays
"""

from __future__ import annotations

import csv
import dataclasses
import hashlib
import json
import shutil
import urllib.request
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import tyro

from plasmax.environment.config import parse_env_and_backend, valid_env_backend_combos
from plasmax.environment.references import (
    load_reference_manifest,
    reset_reference_id,
    reset_reference_payload,
    reset_reference_sha256,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = REPOSITORY_ROOT / "src" / "plasmax" / "configs"
REFERENCE_DATA_DIR = CONFIGS_DIR / "data" / "references"
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "outputs" / "calibration"
CELL_CENTRES = np.arange(0.02, 1.0, 0.04, dtype=np.float64)


@dataclasses.dataclass(frozen=True)
class Download:
    """One immutable checksum-download input."""

    filename: str
    url: str
    sha256: str


DOWNLOADS: Mapping[str, Download] = {
    "itpa_450s": Download(
        filename="ITPA-TC33-450s.nc",
        url=("https://zenodo.org/records/21391776/files/ITPA-TC33-450s.nc?download=1"),
        sha256="ec73a0c64afa0352a5dc8173187a03682c27b8bf8f7b5cc3f6bfba533b1fbf71",
    ),
    "iter_advanced_pdf": Download(
        filename="ITER_Physics_Campbell.pdf",
        url=(
            "https://www.iter.org/sites/default/files/education/"
            "ITER_Physics_Campbell.pdf"
        ),
        sha256="29b25019c8fb1b47ed354b5ecb218c7fb55e77a567f6ea1ec0c7beebe2b7a3c3",
    ),
    "sparc_h8_pdf": Download(
        filename="SPARC_H8_Muraca_2025.pdf",
        url="https://arxiv.org/pdf/2502.00187",
        sha256="1175d9d66894fda9024dce51368be1e435a2e6f8fdca9d6e4b7584798058f76d",
    ),
    "iter_hybrid_rampup": Download(
        filename="torax-iterhybrid-rampup-v1.4.2.py",
        url=(
            "https://raw.githubusercontent.com/google-deepmind/torax/"
            "v1.4.2/torax/examples/iterhybrid_rampup.py"
        ),
        sha256="f3a66f5767a9cb3a2a6c6a7d6596a7433704d0439a32d89912ccf45f1fd00c50",
    ),
    "iter_hybrid_flattop": Download(
        filename="torax-iterhybrid-flat-v1.4.2.py",
        url=(
            "https://raw.githubusercontent.com/google-deepmind/torax/"
            "v1.4.2/torax/examples/iterhybrid_predictor_corrector.py"
        ),
        sha256="0b2ee110defc5979b78be8ecf5807a04d329fc71ad310873e176fecfd9712f79",
    ),
    "torax_step": Download(
        filename="torax-step-v1.4.2.py",
        url=(
            "https://raw.githubusercontent.com/google-deepmind/torax/"
            "v1.4.2/torax/examples/step_flattop_bgb.py"
        ),
        sha256="94234eaa673530c07041ea633a2b6657fd0ab7fd367a4ddd5d4730030980ff08",
    ),
}


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 of a file without loading it all into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_sha256(path: str | Path, expected: str) -> None:
    """Raise when ``path`` does not match its immutable source digest."""

    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(
            f"SHA-256 mismatch for {str(path)!r}: expected {expected}, got {actual}"
        )


def download_source(spec: Download, cache_dir: Path) -> Path:
    """Checksum-download one source atomically into an ignored local cache."""

    cache_dir.mkdir(parents=True, exist_ok=True)
    destination = cache_dir / spec.filename
    if destination.exists():
        verify_sha256(destination, spec.sha256)
        return destination
    temporary = destination.with_suffix(destination.suffix + ".part")
    try:
        with (
            urllib.request.urlopen(spec.url) as response,  # noqa: S310
            temporary.open("wb") as output,
        ):
            shutil.copyfileobj(response, output)
        verify_sha256(temporary, spec.sha256)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def parse_sectioned_profile(path: str | Path) -> dict[str, np.ndarray]:
    """Parse SPARCPublic's ``# name | unit`` two-column profile format."""

    sections: dict[str, list[float]] = {}
    current: str | None = None
    with Path(path).open() as stream:
        for line_number, line in enumerate(stream, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                current = stripped[1:].split("|", maxsplit=1)[0].strip()
                if not current:
                    raise ValueError(f"empty section name on line {line_number}")
                sections[current] = []
                continue
            if current is None:
                raise ValueError(f"data before first section on line {line_number}")
            fields = stripped.split()
            if len(fields) != 2:
                raise ValueError(f"expected index/value pair on line {line_number}")
            expected_index = len(sections[current]) + 1
            if int(fields[0]) != expected_index:
                raise ValueError(
                    f"non-consecutive index on line {line_number}: "
                    f"expected {expected_index}, got {fields[0]}"
                )
            sections[current].append(float(fields[1]))
    lengths = {len(values) for values in sections.values()}
    if lengths != {101}:
        raise ValueError(f"SPARCPublic sections must each have 101 rows, got {lengths}")
    return {name: np.asarray(values) for name, values in sections.items()}


def interpolate_profile(
    rho: Sequence[float],
    values: Sequence[float],
    target_rho: Sequence[float] = CELL_CENTRES,
) -> np.ndarray:
    """Deterministically interpolate a physical profile onto cell centres."""

    source_rho = np.asarray(rho, dtype=np.float64)
    source_values = np.asarray(values, dtype=np.float64)
    target = np.asarray(target_rho, dtype=np.float64)
    if source_rho.ndim != 1 or source_values.shape != source_rho.shape:
        raise ValueError("profile rho and values must be equal-length 1D arrays")
    if not np.all(np.diff(source_rho) > 0):
        raise ValueError("profile rho must be strictly increasing")
    if not np.all(np.isfinite(source_values)):
        raise ValueError("profile values must be finite")
    return np.interp(target, source_rho, source_values)


def reconstruct_psi_from_q(
    rho: Sequence[float],
    q: Sequence[float],
    edge_psi: float,
) -> np.ndarray:
    """Reconstruct a monotone psi shape by integrating ``rho / q``.

    The proportionality constant is fixed by the chosen physical equilibrium's
    edge poloidal flux. This is the documented fallback for a figure that
    publishes q but no equilibrium artifact.
    """

    radial_grid = np.asarray(rho, dtype=np.float64)
    safety_factor = np.asarray(q, dtype=np.float64)
    if radial_grid.ndim != 1 or safety_factor.shape != radial_grid.shape:
        raise ValueError("rho and q must be equal-length 1D arrays")
    if not np.all(np.diff(radial_grid) > 0.0):
        raise ValueError("rho must be strictly increasing")
    if not np.all(np.isfinite(safety_factor)) or np.any(safety_factor <= 0.0):
        raise ValueError("q must be positive and finite")
    if not np.isfinite(edge_psi) or edge_psi <= 0.0:
        raise ValueError("edge_psi must be positive and finite")

    extended_rho = np.concatenate(([0.0], radial_grid))
    integrand = np.concatenate(([0.0], radial_grid / safety_factor))
    cumulative = np.asarray(
        [
            np.trapezoid(integrand[: index + 2], extended_rho[: index + 2])
            for index in range(radial_grid.size)
        ]
    )
    return cumulative * (edge_psi / cumulative[-1])


def cylindrical_volume_error(
    source_rho: Sequence[float],
    source_values: Sequence[float],
    cell_values: Sequence[float],
) -> float:
    """Relative error in a cylindrical volume-weighted profile integral."""

    rho = np.asarray(source_rho, dtype=np.float64)
    values = np.asarray(source_values, dtype=np.float64)
    cells = np.asarray(cell_values, dtype=np.float64)
    source_integral = np.trapezoid(values * 2.0 * rho, rho)
    faces = np.linspace(0.0, 1.0, cells.size + 1)
    cell_integral = np.sum(cells * np.diff(faces**2))
    return float(abs(cell_integral - source_integral) / abs(source_integral))


@dataclasses.dataclass(frozen=True)
class GaussianProjection:
    """Released source moments and their TORAX Gaussian representation."""

    total: float
    rho_centroid: float
    rho_rms_width: float
    gaussian_location: float
    gaussian_width: float
    electron_fraction: float | None = None


def _profile_moments(
    rho: np.ndarray,
    profile: np.ndarray,
    volume: np.ndarray,
) -> tuple[float, float, float]:
    """Integrate a released profile against its physical volume coordinate."""

    volume_derivative = np.gradient(volume, rho)
    weighted = profile * volume_derivative
    total = float(np.trapezoid(weighted, rho))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("source profile must have a positive finite integral")
    centroid = float(np.trapezoid(rho * weighted, rho) / total)
    variance = float(np.trapezoid(np.square(rho - centroid) * weighted, rho) / total)
    return total, centroid, float(np.sqrt(variance))


def _gaussian_moments(
    rho: np.ndarray,
    volume_derivative: np.ndarray,
    location: float,
    width: float,
) -> np.ndarray:
    profile = np.exp(-0.5 * np.square((rho - location) / width))
    weights = profile * volume_derivative
    normalization = np.sum(weights)
    centroid = np.sum(rho * weights) / normalization
    rms_width = np.sqrt(np.sum(np.square(rho - centroid) * weights) / normalization)
    return np.asarray((centroid, rms_width), dtype=np.float64)


def fit_gaussian_moments(
    rho: Sequence[float],
    volume_derivative: Sequence[float],
    target_centroid: float,
    target_rms_width: float,
) -> tuple[float, float]:
    """Fit TORAX Gaussian inputs to physical dV-weighted source moments.

    A deterministic two-variable Newton solve is used instead of treating the
    Gaussian input location/width as the resulting physical moments. The two
    differ on a shaped torus because TORAX normalizes the source against
    ``dV/drho``.
    """

    radial_grid = np.asarray(rho, dtype=np.float64)
    measure = np.asarray(volume_derivative, dtype=np.float64)
    target = np.asarray((target_centroid, target_rms_width), dtype=np.float64)
    if radial_grid.ndim != 1 or measure.shape != radial_grid.shape:
        raise ValueError("rho and volume_derivative must be equal-length 1D arrays")
    if target_rms_width <= 0.0:
        raise ValueError("target_rms_width must be positive")

    parameters = target.copy()
    for _ in range(64):
        residual = (
            _gaussian_moments(radial_grid, measure, parameters[0], parameters[1])
            - target
        )
        if np.max(np.abs(residual)) <= 1.0e-13:
            return float(parameters[0]), float(parameters[1])

        jacobian = np.empty((2, 2), dtype=np.float64)
        for column in range(2):
            delta = 1.0e-6 * max(1.0, abs(parameters[column]))
            upper = parameters.copy()
            lower = parameters.copy()
            upper[column] += delta
            lower[column] -= delta
            jacobian[:, column] = (
                _gaussian_moments(radial_grid, measure, upper[0], upper[1])
                - _gaussian_moments(radial_grid, measure, lower[0], lower[1])
            ) / (2.0 * delta)
        step = np.linalg.solve(jacobian, -residual)
        current_norm = np.linalg.norm(residual)
        for scale in (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625):
            proposed = parameters + scale * step
            if proposed[1] <= 1.0e-5:
                continue
            proposed_residual = (
                _gaussian_moments(radial_grid, measure, proposed[0], proposed[1])
                - target
            )
            if np.linalg.norm(proposed_residual) < current_norm:
                parameters = proposed
                break
        else:
            raise RuntimeError("Gaussian moment solve could not reduce its residual")
    raise RuntimeError("Gaussian moment solve did not converge in 64 iterations")


def _iter_baseline_geometry_measure() -> tuple[np.ndarray, np.ndarray]:
    """Build only the converted ITPA geometry, without an environment/backend."""

    from torax._src.geometry import pydantic_model as geometry_pydantic_model

    geometry_config = geometry_pydantic_model.Geometry.model_validate(
        {
            "geometry_type": "eqdsk",
            "geometry_directory": str(CONFIGS_DIR / "data"),
            "geometry_file": "references/iter_baseline_450s.eqdsk",
            "cocos": 7,
            "n_rho": 25,
            "Ip_from_parameters": True,
            "last_surface_factor": 0.989,
        }
    )
    geometry = geometry_config.build_provider(0.0)
    return (
        np.asarray(geometry.rho_norm, dtype=np.float64),
        np.asarray(geometry.vpr, dtype=np.float64),
    )


def itpa_source_projections(path: str | Path) -> dict[str, GaussianProjection]:
    """Derive all declared ITPA source projections without stepping a backend."""

    source = Path(path)
    verify_sha256(source, DOWNLOADS["itpa_450s"].sha256)
    import h5py

    with h5py.File(source) as data:
        group = data["core_sources/0"]
        names = [
            value.decode() if isinstance(value, bytes) else str(value)
            for value in group["source.identifier.name"][:]
        ]

        def source_arrays(name: str) -> tuple[int, np.ndarray, np.ndarray]:
            index = names.index(name)
            source_rho = np.asarray(
                group["source.profiles_1d.grid.rho_tor_norm"][index, 0],
                dtype=np.float64,
            )
            volume = np.asarray(
                group["source.profiles_1d.grid.volume"][index, 0],
                dtype=np.float64,
            )
            return index, source_rho, volume

        nbi_index, nbi_rho, nbi_volume = source_arrays("nbi")
        nbi_electron_heat = np.asarray(
            group["source.profiles_1d.electrons.energy"][nbi_index, 0]
        )
        nbi_ion_heat = np.asarray(
            group["source.profiles_1d.total_ion_energy"][nbi_index, 0]
        )
        nbi_total_heat = nbi_electron_heat + nbi_ion_heat
        integrated_nbi_heat, nbi_centroid, nbi_width = _profile_moments(
            nbi_rho, nbi_total_heat, nbi_volume
        )
        integrated_nbi_electron_heat, _, _ = _profile_moments(
            nbi_rho, nbi_electron_heat, nbi_volume
        )
        nbi_particles = np.asarray(
            group["source.profiles_1d.ion.particles"][nbi_index, 0]
        ).sum(axis=0)
        _, nbi_particle_centroid, nbi_particle_width = _profile_moments(
            nbi_rho, nbi_particles, nbi_volume
        )

        ec_index, ec_rho, ec_volume = source_arrays("ec")
        ec_heat = np.asarray(group["source.profiles_1d.electrons.energy"][ec_index, 0])
        _, ec_centroid, ec_width = _profile_moments(ec_rho, ec_heat, ec_volume)

        pellet_index, pellet_rho, pellet_volume = source_arrays("pellet")
        pellet_particles = np.asarray(
            group["source.profiles_1d.ion.particles"][pellet_index, 0]
        ).sum(axis=0)
        _, pellet_centroid, pellet_width = _profile_moments(
            pellet_rho, pellet_particles, pellet_volume
        )

        nbi_power = float(group["source.global_quantities.power"][nbi_index, 0])
        ec_power = float(group["source.global_quantities.electrons.power"][ec_index, 0])
        nbi_particle_rate = float(
            group["source.global_quantities.total_ion_particles"][nbi_index, 0]
        )
        pellet_rate = float(
            group["source.global_quantities.total_ion_particles"][pellet_index, 0]
        )

    torax_rho, torax_dv_drho = _iter_baseline_geometry_measure()

    def projected(
        total: float,
        centroid: float,
        width: float,
        electron_fraction: float | None = None,
    ) -> GaussianProjection:
        location, gaussian_width = fit_gaussian_moments(
            torax_rho, torax_dv_drho, centroid, width
        )
        return GaussianProjection(
            total=total,
            rho_centroid=centroid,
            rho_rms_width=width,
            gaussian_location=location,
            gaussian_width=gaussian_width,
            electron_fraction=electron_fraction,
        )

    return {
        "nbi_heat": projected(
            nbi_power,
            nbi_centroid,
            nbi_width,
            integrated_nbi_electron_heat / integrated_nbi_heat,
        ),
        "ec_heat": projected(ec_power, ec_centroid, ec_width, 1.0),
        "nbi_particles": projected(
            nbi_particle_rate, nbi_particle_centroid, nbi_particle_width
        ),
        "pellet_particles": projected(pellet_rate, pellet_centroid, pellet_width),
    }


def _itpa_profiles(path: Path) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Import the exact ITPA profiles and interpolate them to 25 cells."""

    verify_sha256(path, DOWNLOADS["itpa_450s"].sha256)
    import imas
    from torax._src.imas_tools.input import core_profiles, loader

    core_ids = loader.load_imas_data(str(path), "core_profiles", explicit_convert=True)
    conditions = core_profiles.profile_conditions_from_IMAS(core_ids)
    equilibrium_ids = loader.load_imas_data(
        str(path), "equilibrium", explicit_convert=True
    )
    equilibrium = imas.util.to_xarray(equilibrium_ids)
    psi_axis = float(equilibrium["time_slice.global_quantities.psi_axis"][0])
    eq_rho = np.asarray(
        equilibrium["time_slice.profiles_1d.rho_tor_norm"][0], dtype=np.float64
    )
    q_source = np.asarray(equilibrium["time_slice.profiles_1d.q"][0], dtype=np.float64)

    imported: dict[str, np.ndarray] = {"rho": CELL_CENTRES.copy()}
    errors: dict[str, float] = {}
    # TORAX's IMAS adapter already returns temperatures in keV and density in
    # m^-3, matching ``profile_conditions`` units.
    for name, unit_scale in (("T_i", 1.0), ("T_e", 1.0), ("n_e", 1.0)):
        rho = np.asarray(conditions[name][1][0], dtype=np.float64)
        values = np.asarray(conditions[name][2][0], dtype=np.float64) * unit_scale
        imported[name] = interpolate_profile(rho, values)
        errors[name] = cylindrical_volume_error(rho, values, imported[name])
    psi_rho = np.asarray(conditions["psi"][1][0], dtype=np.float64)
    psi = np.asarray(conditions["psi"][2][0], dtype=np.float64) - psi_axis
    imported["psi"] = interpolate_profile(psi_rho, psi)
    imported["q"] = interpolate_profile(eq_rho, q_source)
    return imported, errors


def write_itpa_csv(profiles: Mapping[str, np.ndarray], destination: Path) -> None:
    """Write the compact 25-cell ITPA profile artifact deterministically."""

    columns = ("rho", "T_i", "T_e", "n_e", "psi", "q")
    labels = ("rho", "T_i_keV", "T_e_keV", "n_e_m-3", "psi_Wb", "q")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(labels)
        for row in zip(*(profiles[name] for name in columns), strict=True):
            writer.writerow(
                (
                    f"{row[0]:.2f}",
                    f"{row[1]:.8g}",
                    f"{row[2]:.8g}",
                    f"{row[3]:.8g}",
                    f"{row[4]:.8g}",
                    f"{row[5]:.8g}",
                )
            )


def write_itpa_eqdsk(source: Path, destination: Path) -> None:
    """Convert the ITPA IMAS equilibrium to a deterministic COCOS-7 GEQDSK."""

    verify_sha256(source, DOWNLOADS["itpa_450s"].sha256)
    import eqdsk
    import imas
    from eqdsk.cocos import COCOS

    with imas.DBEntry(uri=str(source), mode="r", dd_version="3.42.0") as database:
        data = eqdsk.imas.from_imas(database)
    # GEQDSK requires the 1D flux functions to have the rectangular grid's
    # radial length.  The IDS provides 70 cell-centred values on a 71x71 grid.
    flux_grid = np.linspace(0.0, 1.0, int(data["nx"]))
    source_grid = np.asarray(data["psinorm"], dtype=np.float64)
    for name in ("ffprime", "fpol", "pprime", "pressure", "qpsi"):
        data[name] = np.interp(flux_grid, source_grid, np.asarray(data[name]))
    data["psinorm"] = flux_grid
    interface = eqdsk.EQDSKInterface(**data)
    # IMAS DD4 uses COCOS 17.  Convert the 2pi-normalized flux to COCOS 7 so it
    # can coexist with the existing ramp EQDSKs under one TORAX geometry block.
    interface._cocos = COCOS.C17  # noqa: SLF001 - library has no public setter
    converted = interface.to_cocos(COCOS.C7)
    destination.parent.mkdir(parents=True, exist_ok=True)
    converted.write(destination, file_format="eqdsk", strict_spec=True)
    # eqdsk-python inserts today's date in the identifier.  Replace only that
    # non-physical header with a stable, source-labelled GEQDSK header.
    lines = destination.read_text().splitlines(keepends=True)
    lines[0] = f"{'ITPA TC33 450S':<48}{0:4d}{converted.nx:4d}{converted.nz:4d}\n"
    destination.write_text("".join(lines))


def _read_csv(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 25:
        raise ValueError(f"expected 25 profile rows in {path}, got {len(rows)}")
    return {
        name: np.asarray([float(row[name]) for row in rows], dtype=np.float64)
        for name in rows[0]
    }


def _numeric_mapping_values(mapping: Mapping[str, Any]) -> np.ndarray:
    return np.asarray(
        [mapping[key] for key in sorted(mapping, key=float)], dtype=np.float64
    )


def _assert_close_columns(
    actual: Mapping[str, np.ndarray],
    expected: Mapping[str, np.ndarray],
    names: Iterable[str],
    *,
    rtol: float = 1e-6,
) -> None:
    for name in names:
        np.testing.assert_allclose(actual[name], expected[name], rtol=rtol, atol=0.0)


def _pixel_points(
    rho: np.ndarray,
    values: np.ndarray,
    *,
    x_pixels: tuple[float, float],
    y_pixels: tuple[float, float],
    y_values: tuple[float, float],
) -> list[tuple[float, float]]:
    x = x_pixels[0] + rho * (x_pixels[1] - x_pixels[0])
    y = y_pixels[0] + (values - y_values[0]) * (
        (y_pixels[1] - y_pixels[0]) / (y_values[1] - y_values[0])
    )
    return list(zip(x, y, strict=True))


def _render_pdf_page(path: Path, page: int):
    try:
        import pypdfium2
    except ImportError as error:  # pragma: no cover - optional research group
        raise RuntimeError(
            "rendering digitization overlays requires the research dependency pypdfium2"
        ) from error
    document = pypdfium2.PdfDocument(path)
    try:
        return document[page].render(scale=3).to_pil().convert("RGB")
    finally:
        document.close()


def render_iter_advanced_overlay(pdf: Path, destination: Path) -> None:
    """Overlay the stored page-25 digitization on the source slide."""

    verify_sha256(pdf, DOWNLOADS["iter_advanced_pdf"].sha256)
    from PIL import ImageDraw

    image = _render_pdf_page(pdf, 24)
    draw = ImageDraw.Draw(image)
    data = _read_csv(REFERENCE_DATA_DIR / "iter_advanced_slide25_profiles_25.csv")
    rho = data["rho"]
    for name, colour in (("T_i_keV", "#ff00ff"), ("T_e_keV", "#00ffff")):
        draw.line(
            _pixel_points(
                rho,
                data[name],
                x_pixels=(1334, 1948),
                y_pixels=(881, 444),
                y_values=(0, 30),
            ),
            fill=colour,
            width=4,
        )
    draw.line(
        _pixel_points(
            rho,
            data["n_e_m-3"] / 1e20,
            x_pixels=(1334, 1948),
            y_pixels=(881, 444),
            y_values=(0, 1),
        ),
        fill="#ffff00",
        width=4,
    )
    draw.line(
        _pixel_points(
            rho,
            data["q"],
            x_pixels=(1354, 1951),
            y_pixels=(1448, 1040),
            y_values=(0, 10),
        ),
        fill="#ff00ff",
        width=4,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination)


def render_sparc_h8_overlay(pdf: Path, destination: Path) -> None:
    """Overlay the stored Figure-16 10/50/90% curves on the source paper."""

    verify_sha256(pdf, DOWNLOADS["sparc_h8_pdf"].sha256)
    from PIL import ImageDraw

    image = _render_pdf_page(pdf, 24)
    draw = ImageDraw.Draw(image)
    data = _read_csv(REFERENCE_DATA_DIR / "sparc_h8_figure16_profiles_25.csv")
    rho = data["rho"]
    specifications = (
        ("T_i", (310, 636), (449, 218), (0, 12), 1.0),
        ("T_e", (720, 1045), (449, 222), (0, 14), 1.0),
        ("n_e", (519, 840), (750, 548), (0, 40), 1e19),
    )
    for prefix, x_pixels, y_pixels, y_values, scale in specifications:
        for suffix, colour, width in (
            ("p10", "#00ffff", 3),
            ("median", "#ff00ff", 4),
            ("p90", "#ffff00", 3),
        ):
            unit = "keV" if prefix.startswith("T_") else "m-3"
            values = data[f"{prefix}_{suffix}_{unit}"] / scale
            draw.line(
                _pixel_points(
                    rho,
                    values,
                    x_pixels=x_pixels,
                    y_pixels=y_pixels,
                    y_values=y_values,
                ),
                fill=colour,
                width=width,
            )
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination)


def _source_path(explicit: Path | None, cache_dir: Path, key: str) -> Path | None:
    path = explicit if explicit is not None else cache_dir / DOWNLOADS[key].filename
    return path if path.exists() else None


def _verify_itpa_source_projections(
    projections: Mapping[str, GaussianProjection],
) -> None:
    """Cross-check source-derived moments, manifest records, and live YAML."""

    reference = load_reference_manifest().references["iter_baseline_hot_450s"]
    by_source = {item.source_component: item for item in reference.projections}
    manifest_names = {
        "nbi_heat": "ITPA core_sources/nbi total heat",
        "ec_heat": "ITPA core_sources/ec electron heat",
        "nbi_particles": "ITPA core_sources/nbi ion particles",
        "pellet_particles": "ITPA core_sources/pellet ion particles",
    }
    for name, source_component in manifest_names.items():
        projection = projections[name]
        declared = by_source[source_component]
        targets = declared.source_targets
        total_key = "power_W" if name.endswith("heat") else "particles_per_s"
        np.testing.assert_allclose(targets[total_key], projection.total, rtol=1e-12)
        np.testing.assert_allclose(
            targets["rho_centroid"], projection.rho_centroid, rtol=1e-12
        )
        np.testing.assert_allclose(
            targets["rho_rms_width"], projection.rho_rms_width, rtol=1e-12
        )
        parameters = declared.torax_parameters
        location_key = {
            "nbi_heat": "gaussian_location",
            "ec_heat": "gaussian_location",
            "nbi_particles": "deposition_location",
            "pellet_particles": "pellet_deposition_location",
        }[name]
        width_key = {
            "nbi_heat": "gaussian_width",
            "ec_heat": "gaussian_width",
            "nbi_particles": "particle_width",
            "pellet_particles": "pellet_width",
        }[name]
        np.testing.assert_allclose(
            parameters[location_key], projection.gaussian_location, rtol=1e-12
        )
        np.testing.assert_allclose(
            parameters[width_key], projection.gaussian_width, rtol=1e-12
        )

    config = parse_env_and_backend("iter/baseline/flattop", "bohm_gyrobohm")
    sources = config.torax["sources"]
    yaml_specs = {
        "nbi_heat": ("generic_heat", "P_total", "gaussian_location", "gaussian_width"),
        "ec_heat": ("ecrh", "P_total", "gaussian_location", "gaussian_width"),
        "nbi_particles": (
            "generic_particle",
            "S_total",
            "deposition_location",
            "particle_width",
        ),
        "pellet_particles": (
            "pellet",
            "S_total",
            "pellet_deposition_location",
            "pellet_width",
        ),
    }
    for name, (source_name, total_key, location_key, width_key) in yaml_specs.items():
        projection = projections[name]
        source_config = sources[source_name]
        np.testing.assert_allclose(source_config[total_key], projection.total)
        np.testing.assert_allclose(
            source_config[location_key], projection.gaussian_location, rtol=1e-12
        )
        np.testing.assert_allclose(
            source_config[width_key], projection.gaussian_width, rtol=1e-12
        )

    actuator_inits = {item.name: item.init for item in config.actuators}
    np.testing.assert_allclose(
        actuator_inits["rho_eccd"], projections["ec_heat"].gaussian_location
    )


def verify_packaged_references() -> dict[str, Any]:
    """Verify manifest coverage, provenance hashes, and cMDP reset equality."""

    manifest = load_reference_manifest()
    environments = [name for name in valid_env_backend_combos() if name != "kstar"]
    references_by_env: dict[str, str] = {}
    reset_hashes: dict[str, str] = {}
    for env in environments:
        reference_id = reset_reference_id(env)
        if reference_id not in manifest.references:
            raise ValueError(f"{env} names unknown reset reference {reference_id!r}")
        references_by_env[env] = reference_id
        digest = reset_reference_sha256(env)
        expected = manifest.references[reference_id].profile_sha256
        if digest != expected:
            raise ValueError(
                f"reset payload hash mismatch for {env}: expected {expected}, "
                f"got {digest}"
            )
        reset_hashes[env] = digest
        reference_payload = reset_reference_payload(env)
        for backend in sorted(valid_env_backend_combos()[env]):
            if reset_reference_payload(env, backend) != reference_payload:
                raise ValueError(
                    f"backend {backend} changes the reset payload for {env}"
                )

    for scenario in (
        "iter/baseline",
        "iter/hybrid",
        "iter/advanced",
        "sparc/prd",
        "sparc/reduced_field",
    ):
        flat = f"{scenario}/flattop"
        down = f"{scenario}/rampdown"
        if references_by_env[flat] != references_by_env[down]:
            raise ValueError(f"{scenario} flat-top and ramp-down use different IDs")
        if reset_hashes[flat] != reset_hashes[down]:
            raise ValueError(f"{scenario} flat-top and ramp-down payloads differ")

    for reference in manifest.references.values():
        for relative_path, expected in reference.artifacts.items():
            artifact = CONFIGS_DIR / relative_path
            verify_sha256(artifact, expected)
        for source in reference.sources:
            if source.local_path is not None:
                verify_sha256(CONFIGS_DIR / source.local_path, source.sha256)

    # Independent SPARCPublic parsing pins the exact released profile values.
    sparc_raw = parse_sectioned_profile(
        REFERENCE_DATA_DIR / "sparc_prd_transp_20221013.txt"
    )
    sparc_cells = {
        "rho": CELL_CENTRES,
        "T_i": interpolate_profile(sparc_raw["rho"], sparc_raw["ti"]),
        "T_e": interpolate_profile(sparc_raw["rho"], sparc_raw["te"]),
        "n_e": interpolate_profile(sparc_raw["rho"], sparc_raw["ne"] * 1e19),
        "psi": interpolate_profile(
            sparc_raw["rho"], sparc_raw["polflux"] * 2.0 * np.pi
        ),
    }
    prd_payload = reset_reference_payload("sparc/prd/flattop")
    conditions = prd_payload["profile_conditions"]
    for name in ("T_i", "T_e", "n_e", "psi"):
        actual = _numeric_mapping_values(conditions[name])
        np.testing.assert_allclose(actual, sparc_cells[name], rtol=2e-6, atol=0.0)

    advanced = _read_csv(REFERENCE_DATA_DIR / "iter_advanced_slide25_profiles_25.csv")
    advanced_payload = reset_reference_payload("iter/advanced/flattop")
    advanced_conditions = advanced_payload["profile_conditions"]
    for condition_name, column_name in (
        ("T_i", "T_i_keV"),
        ("T_e", "T_e_keV"),
        ("n_e", "n_e_m-3"),
    ):
        np.testing.assert_allclose(
            _numeric_mapping_values(advanced_conditions[condition_name]),
            advanced[column_name],
            rtol=1e-7,
            atol=0.0,
        )
    advanced_reference = manifest.references["iter_advanced_hot_slide25"]
    reconstructed_psi = reconstruct_psi_from_q(
        advanced["rho"],
        advanced["q"],
        float(advanced_reference.targets["edge_poloidal_flux_Wb"]),
    )
    np.testing.assert_allclose(
        _numeric_mapping_values(advanced_conditions["psi"]),
        reconstructed_psi,
        rtol=2e-7,
        atol=1e-8,
    )

    h8 = _read_csv(REFERENCE_DATA_DIR / "sparc_h8_figure16_profiles_25.csv")
    h8_payload = reset_reference_payload("sparc/reduced_field/flattop")
    h8_conditions = h8_payload["profile_conditions"]
    for prefix in ("T_i", "T_e", "n_e"):
        unit = "keV" if prefix.startswith("T_") else "m-3"
        if not np.all(h8[f"{prefix}_p10_{unit}"] <= h8[f"{prefix}_median_{unit}"]):
            raise ValueError(f"{prefix} median leaves its digitization envelope")
        if not np.all(h8[f"{prefix}_median_{unit}"] <= h8[f"{prefix}_p90_{unit}"]):
            raise ValueError(f"{prefix} median leaves its digitization envelope")
        np.testing.assert_allclose(
            _numeric_mapping_values(h8_conditions[prefix]),
            h8[f"{prefix}_median_{unit}"],
            rtol=1e-7,
            atol=0.0,
        )

    return {
        "schema_version": manifest.schema_version,
        "reference_count": len(manifest.references),
        "environment_count": len(environments),
        "references_by_environment": references_by_env,
        "reset_hashes": reset_hashes,
    }


def main(
    cache_dir: Path = DEFAULT_OUTPUT_DIR / "reference_cache",
    output: Path = DEFAULT_OUTPUT_DIR / "physical_reference_report.json",
    itpa_450s: Path | None = None,
    iter_advanced_pdf: Path | None = None,
    sparc_h8_pdf: Path | None = None,
    download_missing: bool = False,
    write_derived: bool = False,
    render_overlays: bool = False,
) -> None:
    """Verify references and optionally regenerate source-derived artifacts."""

    downloaded: dict[str, str] = {}
    if download_missing:
        for key, spec in DOWNLOADS.items():
            downloaded[key] = str(download_source(spec, cache_dir))

    itpa_path = _source_path(itpa_450s, cache_dir, "itpa_450s")
    advanced_path = _source_path(iter_advanced_pdf, cache_dir, "iter_advanced_pdf")
    h8_path = _source_path(sparc_h8_pdf, cache_dir, "sparc_h8_pdf")
    derivation: dict[str, Any] = {}
    if itpa_path is not None:
        profiles, errors = _itpa_profiles(itpa_path)
        source_projections = itpa_source_projections(itpa_path)
        _verify_itpa_source_projections(source_projections)
        packaged = _read_csv(REFERENCE_DATA_DIR / "iter_baseline_450s_profiles_25.csv")
        _assert_close_columns(
            profiles,
            {
                "rho": packaged["rho"],
                "T_i": packaged["T_i_keV"],
                "T_e": packaged["T_e_keV"],
                "n_e": packaged["n_e_m-3"],
                "psi": packaged["psi_Wb"],
                "q": packaged["q"],
            },
            ("rho", "T_i", "T_e", "n_e", "psi", "q"),
            rtol=2e-6,
        )
        if any(error > 0.01 for error in errors.values()):
            raise ValueError(f"ITPA volume-weighted interpolation error >1%: {errors}")
        derivation["itpa_450s"] = {
            "source": str(itpa_path),
            "source_projections": {
                name: dataclasses.asdict(projection)
                for name, projection in source_projections.items()
            },
            "volume_weighted_interpolation_error": errors,
        }
        if write_derived:
            write_itpa_csv(
                profiles,
                REFERENCE_DATA_DIR / "iter_baseline_450s_profiles_25.csv",
            )
            write_itpa_eqdsk(
                itpa_path,
                REFERENCE_DATA_DIR / "iter_baseline_450s.eqdsk",
            )
    elif write_derived:
        raise ValueError("--write-derived requires --itpa-450s or --download-missing")

    overlays: dict[str, str] = {}
    if render_overlays:
        if advanced_path is None or h8_path is None:
            raise ValueError(
                "--render-overlays requires both PDFs or --download-missing"
            )
        overlay_dir = output.parent / "digitization_overlays"
        advanced_overlay = overlay_dir / "iter_advanced_slide25_overlay.png"
        h8_overlay = overlay_dir / "sparc_h8_figure16_overlay.png"
        render_iter_advanced_overlay(advanced_path, advanced_overlay)
        render_sparc_h8_overlay(h8_path, h8_overlay)
        overlays = {
            "iter_advanced": str(advanced_overlay),
            "sparc_h8": str(h8_overlay),
        }

    verification = verify_packaged_references()
    report = {
        "construction_steps_backends": 0,
        "downloaded": downloaded,
        "derivation": derivation,
        "overlays": overlays,
        "verification": verification,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        f"Verified {verification['reference_count']} physical references across "
        f"{verification['environment_count']} environments; report: {output}"
    )


if __name__ == "__main__":
    tyro.cli(main)
