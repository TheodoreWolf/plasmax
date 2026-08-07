"""Validate that release archives contain only the installed environment API."""

from __future__ import annotations

import re
import sys
import tarfile
import zipfile
from pathlib import Path

CRITICAL_PACKAGE_FILES = {
    "plasmax/configs/test.yaml",
    "plasmax/configs/data/STEP_SPP_001_ECHD_ftop.nc",
    "plasmax/configs/data/STEP_SPP_001_ECHD_ftop.NOTICE",
    "plasmax/configs/data/kstar_lstm/NOTICE",
    "plasmax/configs/data/kstar_lstm/weights.npz",
}
REPOSITORY_ONLY_DIRS = {
    "agents",
    "benchmarks",
    "experiments",
    "scripts",
    "tests",
    "tools",
    "training",
}
DIRECT_REFERENCE = re.compile(r"^Requires-Dist:.*(?:git\+| @ https?://)", re.MULTILINE)


def _required_package_files() -> set[str]:
    package_root = Path("src/plasmax")
    source_files = {
        f"plasmax/{path.relative_to(package_root).as_posix()}"
        for path in package_root.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.name != ".DS_Store"
        and path.suffix != ".pyc"
    }
    return CRITICAL_PACKAGE_FILES | source_files


def _one_match(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if len(matches) != 1:
        raise AssertionError(f"expected one {pattern!r} in {directory}, got {matches}")
    return matches[0]


def _check_metadata(metadata: str, archive: Path) -> None:
    if "Name: plasmax" not in metadata or "Version: 0.1.0" not in metadata:
        raise AssertionError(f"wrong project identity in {archive}")
    if "Requires-Dist: jax-envelope==0.4.2" not in metadata:
        raise AssertionError(f"wrong Envelope requirement in {archive}")
    if "Provides-Extra:" in metadata:
        raise AssertionError(f"published extras found in {archive}")
    direct = DIRECT_REFERENCE.findall(metadata)
    if direct:
        raise AssertionError(f"published direct references in {archive}: {direct}")


def _check_wheel(wheel: Path) -> None:
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
        missing = _required_package_files() - names
        if missing:
            raise AssertionError(f"wheel is missing package data: {sorted(missing)}")

        dist_info_prefix = "plasmax-0.1.0.dist-info/"
        leaked = sorted(
            name
            for name in names
            if not name.startswith(("plasmax/", dist_info_prefix))
        )
        if leaked:
            raise AssertionError(f"non-plasmax files leaked into wheel: {leaked}")

        metadata = archive.read(f"{dist_info_prefix}METADATA").decode()
        _check_metadata(metadata, wheel)


def _check_sdist(sdist: Path) -> None:
    with tarfile.open(sdist, "r:gz") as archive:
        names = {name.rstrip("/") for name in archive.getnames()}
        roots = {name.split("/", maxsplit=1)[0] for name in names if name}
        if len(roots) != 1:
            raise AssertionError(
                f"source archive has unexpected roots: {sorted(roots)}"
            )
        root = roots.pop()

        required = {f"{root}/src/{name}" for name in _required_package_files()}
        missing = required - names
        if missing:
            raise AssertionError(f"sdist is missing package data: {sorted(missing)}")

        leaked = sorted(
            name
            for name in names
            if len(parts := name.split("/")) > 1 and parts[1] in REPOSITORY_ONLY_DIRS
        )
        if leaked:
            raise AssertionError(f"repository-only files leaked into sdist: {leaked}")
        if any(
            name.startswith(f"{root}/src/")
            and not name.startswith(f"{root}/src/plasmax/")
            for name in names
        ):
            raise AssertionError("sdist contains a second Python package")

        member = archive.extractfile(f"{root}/PKG-INFO")
        if member is None:
            raise AssertionError("sdist is missing PKG-INFO")
        _check_metadata(member.read().decode(), sdist)


def main() -> None:
    dist = Path(sys.argv[1] if len(sys.argv) > 1 else "dist")
    _check_wheel(_one_match(dist, "plasmax-*.whl"))
    _check_sdist(_one_match(dist, "plasmax-*.tar.gz"))


if __name__ == "__main__":
    main()
