"""Build and validate the installed wheel and source distribution APIs."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SMOKE = _ROOT / ".github" / "scripts" / "smoke_installed.py"
_CHECK_DIST = _ROOT / ".github" / "scripts" / "check_dist.py"
_API_PROBE = "import json, plasmax; print(json.dumps(sorted(plasmax.__all__)))"


@pytest.fixture(scope="session")
def built_distributions(tmp_path_factory):
    output_dir = tmp_path_factory.mktemp("plasmax-distributions")
    subprocess.run(
        ["uv", "build", "--out-dir", str(output_dir)],
        cwd=_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    wheel = next(output_dir.glob("plasmax-*.whl"))
    sdist = next(output_dir.glob("plasmax-*.tar.gz"))
    return wheel, sdist


def _run_isolated(artifact: Path, script: Path | None = None, code: str | None = None):
    command = [
        "uv",
        "run",
        "--isolated",
        "--no-project",
        "--with",
        str(artifact),
        "python",
    ]
    if script is not None:
        command.append(str(script))
    else:
        command.extend(("-c", code or ""))
    return subprocess.run(
        command,
        cwd=artifact.parent,
        check=True,
        capture_output=True,
        text=True,
    )


def test_distribution_archives_contain_only_the_environment_library(
    built_distributions,
):
    wheel, _ = built_distributions
    subprocess.run(
        ["uv", "run", "--no-project", "python", str(_CHECK_DIST), str(wheel.parent)],
        cwd=_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("artifact_index", [0, 1], ids=["wheel", "sdist"])
def test_each_distribution_installs_and_smokes_in_isolation(
    built_distributions, artifact_index
):
    artifact = built_distributions[artifact_index]
    _run_isolated(artifact, script=_SMOKE)


def test_wheel_and_sdist_expose_identical_public_apis(built_distributions):
    observed = []
    for artifact in built_distributions:
        result = _run_isolated(artifact, code=_API_PROBE)
        observed.append(json.loads(result.stdout.splitlines()[-1]))
    assert observed[0] == observed[1]
