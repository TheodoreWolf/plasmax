"""Dispatch one evaluation to a credential-free FLAIR GPU container."""

from __future__ import annotations

import json
import math
import re
import shlex
import shutil
import subprocess
import tempfile
import time
from pathlib import Path, PurePosixPath
from typing import Any
from uuid import uuid4

from experiments.agent_policy_search.artifacts import read_json, write_json
from experiments.agent_policy_workspace.ledger import _replace_bytes


def _remaining(deadline: float, reserve: float = 0.0) -> float:
    remaining = deadline - time.time() - reserve
    if remaining <= 0:
        raise TimeoutError("FLAIR worker deadline exhausted")
    return remaining


def _run(command: list[str], *, deadline: float) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=True,
        stdout=subprocess.PIPE,
        text=True,
        timeout=_remaining(deadline),
    )


def _ssh(
    host: str, command: list[str], *, deadline: float
) -> subprocess.CompletedProcess[str]:
    return _run(["ssh", host, shlex.join(command)], deadline=deadline)


def main(request: Path, host: str, remote_root: str, gpu: int, image: str) -> None:
    """Stage one request, run it without credentials/network, and retrieve results."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", host) or gpu < 0:
        raise ValueError("host must be a simple SSH alias and GPU nonnegative")
    request = request.resolve()
    root = PurePosixPath(remote_root)
    identifier = uuid4().hex
    job = root / "jobs" / identifier
    metadata: dict[str, Any] = {
        "host": host,
        "gpu": gpu,
        "image": image,
        "container": None,
        "remote_job": str(job),
        "remote_source": str(root / "source"),
        "exit_code": None,
    }
    metadata_path = request.parent / "remote_execution.json"
    _replace_bytes(metadata_path, json.dumps(metadata, indent=2).encode())
    try:
        payload = read_json(request)
        timeout = float(payload["config"]["timeout_seconds"])
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("worker timeout must be positive and finite")
        now = time.time()
        deadline = float(payload.get("deadline_unix_seconds", now + timeout))
        if not math.isfinite(deadline):
            raise ValueError("worker deadline must be finite")
        deadline = min(deadline, now + timeout)
        metadata["deadline_unix_seconds"] = deadline
        identity = _ssh(
            host,
            [
                "sh",
                "-c",
                'id -u; id -g; id -un; printf "%s\\n" "$HOME"; date +%s',
            ],
            deadline=deadline,
        ).stdout.splitlines()
        uid, gid, user, home, remote_time = identity
        if (
            int(uid) == 0
            or not root.is_relative_to(home)
            or root == PurePosixPath(home)
        ):
            raise ValueError("worker must run non-root below the SSH user's home")
        if ".." in root.parts or not re.fullmatch(r"[a-z_][a-z0-9_-]*", user):
            raise ValueError("invalid worker directory or remote username")
        # Receiving the sampled server clock makes this conservative by SSH latency.
        remote_deadline = deadline + int(remote_time) - time.time()
        container = f"{user}_policy_worker_{identifier}"
        metadata.update(
            container=container,
            remote_deadline_unix_seconds=remote_deadline,
            termination_grace_seconds=5,
            retrieval_reserve_seconds=30,
        )
        _replace_bytes(metadata_path, json.dumps(metadata, indent=2).encode())
        _ssh(
            host,
            ["mkdir", "-p", str(root / "jobs"), str(root / "jax-cache")],
            deadline=deadline,
        )
        with tempfile.TemporaryDirectory(prefix="plasmax-flair-worker-") as temp:
            staging = Path(temp)
            if payload["source"] is not None:
                source = Path(payload["source"])
                candidate = staging / "candidate"
                candidate.mkdir()
                for name in ("policy.py", "custom_primitives.py"):
                    original = source if name == "policy.py" else source.parent / name
                    if (
                        name != "policy.py"
                        and not original.exists()
                        and not original.is_symlink()
                    ):
                        continue
                    if original.is_symlink() or not original.is_file():
                        raise ValueError("policy bundle inputs must be regular files")
                    shutil.copyfile(original, candidate / name)
                payload["source"] = "/evaluation/candidate/policy.py"
            payload["output_dir"] = "/evaluation/output"
            write_json(staging / "request.json", payload)
            _run(
                [
                    "ftk",
                    "sync",
                    "--target",
                    host,
                    "--src",
                    str(staging),
                    "--dest",
                    str(job),
                ],
                deadline=deadline,
            )
        _ssh(host, ["mkdir", "-p", str(job / "output")], deadline=deadline)
        availability = json.loads(
            _run(
                [
                    "ftk",
                    "find-gpus",
                    host,
                    "--gpus",
                    str(gpu),
                    "--need",
                    "1",
                    "--json",
                ],
                deadline=deadline,
            ).stdout
        )
        if not availability.get("satisfied"):
            raise RuntimeError(
                f"GPU {gpu} on {host} is unavailable; no worker launched"
            )
        _remaining(deadline, reserve=35.0)
        worker_deadline = math.floor(remote_deadline - 35.0)
        metadata["worker_deadline_unix_seconds"] = worker_deadline
        _replace_bytes(metadata_path, json.dumps(metadata, indent=2).encode())
        # Check inside the container so staging and Docker startup spend the budget.
        entrypoint = (
            "remaining=$(( $1 - $(date +%s) - 1 )); shift; "
            'test "$remaining" -gt 0 || exit 124; '
            'exec /usr/bin/timeout --kill-after=5s "${remaining}s" "$@"'
        )
        command = [
            *shlex.split(
                "docker run --network none --cap-drop ALL "
                "--security-opt no-new-privileges --cpus 8 --memory 32g "
                "--pids-limit 1024 --read-only --tmpfs /tmp:rw,size=2g"
            ),
            "--name",
            container,
            "--user",
            f"{uid}:{gid}",
            "--gpus",
            f"device={gpu}",
            "--mount",
            f"type=bind,src={root / 'source'},dst=/workspace,readonly",
            "--mount",
            f"type=bind,src={job},dst=/evaluation",
            "--mount",
            f"type=bind,src={root / 'jax-cache'},dst=/jax-cache",
            *shlex.split(
                "--workdir /workspace --env PYTHONPATH=/workspace/src:/workspace "
                "--env PYTHONDONTWRITEBYTECODE=1 "
                "--env JAX_COMPILATION_CACHE_DIR=/jax-cache "
                "--env JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=1 "
                "--env XLA_FLAGS=--xla_gpu_enable_command_buffer= "
                "--env XLA_PYTHON_CLIENT_PREALLOCATE=false "
                "--entrypoint /bin/sh"
            ),
            image,
            "-c",
            entrypoint,
            "plasmax-policy-deadline",
            str(worker_deadline),
            *shlex.split(
                "/usr/bin/uv run --no-cache --no-project "
                "--python /opt/plasmax/.venv/bin/python python -m "
                "experiments.agent_policy_workspace.runtime"
            ),
            "--request",
            "/evaluation/request.json",
        ]
        metadata["exit_code"] = subprocess.run(
            ["ssh", host, shlex.join(command)],
            check=False,
            timeout=_remaining(deadline),
        ).returncode
        # A symlinked transfer root would bypass rsync's per-entry --no-links.
        output = shlex.quote(str(job / "output"))
        _ssh(
            host,
            ["sh", "-c", f"test -d {output} && test ! -L {output}"],
            deadline=deadline,
        )
        _run(
            [
                "rsync",
                "-r",
                "--no-links",
                *[
                    f"--exclude=/{name}"
                    for name in (
                        "request.json",
                        "stdout.txt",
                        "stderr.txt",
                        "remote_execution.json",
                    )
                ],
                f"{host}:{job}/output/",
                f"{request.parent}/",
            ],
            deadline=deadline,
        )
        if not (request.parent / "worker_result.json").is_file():
            raise RuntimeError(
                f"remote worker exited {metadata['exit_code']} without a result"
            )
    except Exception as error:
        metadata["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        _replace_bytes(metadata_path, json.dumps(metadata, indent=2).encode())


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
