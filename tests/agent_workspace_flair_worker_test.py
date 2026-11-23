"""FLAIR dispatch preserves host logs and isolates generated policy workers."""

from __future__ import annotations

import json
import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from experiments.agent_policy_search.artifacts import read_json, write_json
from experiments.agent_policy_workspace import flair_worker, runtime


@dataclass
class Transport:
    available: bool = True
    docker_exit: int = 0
    result: bool = True
    commands: list[list[str]] = field(default_factory=list)
    staged: dict[str, bytes] = field(default_factory=dict)
    now: float = 1000.0
    staging_seconds: float = 0.0
    clock_offset_seconds: int = 0
    timeouts: list[float] = field(default_factory=list)

    def run(self, command: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
        self.commands.append(command)
        self.timeouts.append(kwargs["timeout"])
        stdout = ""
        code = 0
        if command[:2] == ["ftk", "sync"]:
            root = Path(command[command.index("--src") + 1])
            self.staged = {
                str(path.relative_to(root)): path.read_bytes()
                for path in root.rglob("*")
                if path.is_file()
            }
            self.now += self.staging_seconds
        elif command[:2] == ["ftk", "find-gpus"]:
            stdout = json.dumps({"satisfied": self.available})
        elif command[0] == "ssh":
            remote = shlex.split(command[2])
            if remote[:2] == ["sh", "-c"] and remote[2].startswith("id -u;"):
                stdout = (
                    "3689\n100\ntheo\n/homes/cdt24/theo\n"
                    f"{int(self.now) + self.clock_offset_seconds}\n"
                )
            elif remote[0] == "docker":
                code = self.docker_exit
        elif command[0] == "rsync" and self.result:
            write_json(Path(command[-1]) / "worker_result.json", {"status": "ok"})
        return subprocess.CompletedProcess(command, code, stdout, "")


@pytest.fixture
def request_file(tmp_path: Path) -> Path:
    candidate = tmp_path / "source"
    candidate.mkdir()
    (candidate / "policy.py").write_text("POLICY_API_VERSION = 1\n")
    (candidate / "custom_primitives.py").write_text("GAIN = 1\n")
    (candidate / ".netrc").write_text("must not be staged\n")
    output = tmp_path / "evaluation"
    output.mkdir()
    write_json(
        output / "request.json",
        {
            "source": str(candidate / "policy.py"),
            "output_dir": str(output),
            "config": {"timeout_seconds": 120},
            "validate_only": False,
        },
    )
    (output / "stdout.txt").write_text("host stdout\n")
    (output / "stderr.txt").write_text("host stderr\n")
    return output / "request.json"


def dispatch(
    monkeypatch: pytest.MonkeyPatch, request: Path, transport: Transport
) -> None:
    monkeypatch.setattr(flair_worker.subprocess, "run", transport.run)
    monkeypatch.setattr(flair_worker.time, "time", lambda: transport.now)
    monkeypatch.setattr(flair_worker, "uuid4", lambda: SimpleNamespace(hex="a" * 32))
    flair_worker.main(
        request,
        "f2",
        "/homes/cdt24/theo/agent-smoke",
        1,
        "theo_plasmax:test",
    )


def test_stages_only_bundle_and_runs_credential_free_worker(
    request_file: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = Transport()
    original = request_file.read_bytes()
    dispatch(monkeypatch, request_file, transport)
    assert set(transport.staged) == {
        "request.json",
        "candidate/policy.py",
        "candidate/custom_primitives.py",
    }
    rewritten = json.loads(transport.staged["request.json"])
    assert rewritten["source"] == "/evaluation/candidate/policy.py"
    assert rewritten["output_dir"] == "/evaluation/output"
    docker = next(
        shlex.split(c[2])
        for c in transport.commands
        if c[0] == "ssh" and shlex.split(c[2])[0] == "docker"
    )
    assert docker[docker.index("--network") + 1] == "none"
    assert docker[docker.index("--user") + 1] == "3689:100"
    assert docker[docker.index("--gpus") + 1] == "device=1"
    assert docker[docker.index("--entrypoint") + 1] == "/bin/sh"
    entrypoint = docker[docker.index("-c") + 1]
    assert "--kill-after=5s" in entrypoint and "date +%s" in entrypoint
    assert docker[docker.index("plasmax-policy-deadline") + 1] == "1085"
    assert "--read-only" in docker and "--rm" not in docker
    mounts = [docker[i + 1] for i, value in enumerate(docker) if value == "--mount"]
    assert len(mounts) == 3 and mounts[0].endswith("dst=/workspace,readonly")
    assert all(".netrc" not in value and "docker.sock" not in value for value in mounts)
    environment = [docker[i + 1] for i, value in enumerate(docker) if value == "--env"]
    assert all("API_KEY" not in value and "TOKEN" not in value for value in environment)
    rsync = next(c for c in transport.commands if c[0] == "rsync")
    assert "--no-links" in rsync
    assert all(
        f"--exclude=/{name}" in rsync
        for name in (
            "request.json",
            "stdout.txt",
            "stderr.txt",
            "remote_execution.json",
        )
    )
    assert request_file.read_bytes() == original
    assert (request_file.parent / "stdout.txt").read_text() == "host stdout\n"
    assert read_json(request_file.parent / "remote_execution.json")["exit_code"] == 0


def test_staging_spends_absolute_deadline_and_all_transport_is_bounded(
    request_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = read_json(request_file)
    payload["deadline_unix_seconds"] = 1110.0
    request_file.write_text(json.dumps(payload))
    transport = Transport(staging_seconds=45.0, clock_offset_seconds=5000)
    dispatch(monkeypatch, request_file, transport)
    metadata = read_json(request_file.parent / "remote_execution.json")
    assert metadata["deadline_unix_seconds"] == 1110.0
    assert metadata["worker_deadline_unix_seconds"] == 6075
    assert transport.timeouts[:3] == [110.0] * 3
    assert transport.timeouts[3:] == [65.0] * (len(transport.timeouts) - 3)
    assert json.loads(transport.staged["request.json"])["deadline_unix_seconds"] == 1110


def test_expired_during_staging_never_launches_container(
    request_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    transport = Transport(staging_seconds=90.0)
    with pytest.raises(TimeoutError, match="deadline exhausted"):
        dispatch(monkeypatch, request_file, transport)
    assert not any(
        c[0] == "ssh" and shlex.split(c[2])[0] == "docker" for c in transport.commands
    )
    metadata = read_json(request_file.parent / "remote_execution.json")
    assert metadata["exit_code"] is None
    assert "deadline exhausted" in metadata["error"]


def test_container_starting_after_deadline_never_executes_worker(
    request_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = subprocess.run
    transport = Transport()
    dispatch(monkeypatch, request_file, transport)
    docker = next(
        shlex.split(c[2])
        for c in transport.commands
        if c[0] == "ssh" and shlex.split(c[2])[0] == "docker"
    )
    entrypoint = docker[docker.index("-c") + 1]
    result = run(
        ["/bin/sh", "-c", entrypoint, "deadline-test", "0", "/does-not-exist"],
        timeout=5,
        check=False,
    )
    assert result.returncode == 124


def test_transport_failure_still_records_remote_metadata(
    request_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(*_args: Any, **_kwargs: Any) -> subprocess.CompletedProcess:
        raise TimeoutError("identity timed out")

    monkeypatch.setattr(flair_worker.subprocess, "run", fail)
    with pytest.raises(TimeoutError, match="identity timed out"):
        flair_worker.main(
            request_file, "f2", "/homes/cdt24/theo/test", 1, "theo_plasmax:test"
        )
    metadata = read_json(request_file.parent / "remote_execution.json")
    assert metadata["container"] is None
    assert "identity timed out" in metadata["error"]


def test_runtime_sends_absolute_deadline_before_worker_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_json(tmp_path / "request.json", {"source": "policy.py"})
    monkeypatch.setattr(runtime.time, "time", lambda: 2000.0)

    def run(_arguments: list[str], **kwargs: Any) -> SimpleNamespace:
        request = read_json(tmp_path / "request.json")
        assert request == {"source": "policy.py", "deadline_unix_seconds": 2023.0}
        assert kwargs["timeout"] == 23.0
        write_json(tmp_path / "worker_result.json", {"status": "ok", "finite": True})
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(runtime.subprocess, "run", run)
    assert runtime._subprocess(tmp_path, 23.0)["status"] == "ok"


def test_unavailable_gpu_never_launches_container(
    request_file: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = Transport(available=False)
    with pytest.raises(RuntimeError, match="unavailable"):
        dispatch(monkeypatch, request_file, transport)
    assert not any(
        c[0] == "ssh" and shlex.split(c[2])[0] == "docker" for c in transport.commands
    )
    metadata = read_json(request_file.parent / "remote_execution.json")
    assert metadata["exit_code"] is None and "unavailable" in metadata["error"]


def test_nonzero_container_exit_still_retrieves_worker_report(
    request_file: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dispatch(monkeypatch, request_file, Transport(docker_exit=1))
    assert (request_file.parent / "worker_result.json").exists()
    assert read_json(request_file.parent / "remote_execution.json")["exit_code"] == 1


def test_missing_report_raises_after_retrieving_outputs(
    request_file: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = Transport(docker_exit=124, result=False)
    with pytest.raises(RuntimeError, match="124 without a result"):
        dispatch(monkeypatch, request_file, transport)
    assert any(c[0] == "rsync" for c in transport.commands)
    assert read_json(request_file.parent / "remote_execution.json")["exit_code"] == 124


def test_rejects_symlink_policy_input(
    request_file: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = Path(read_json(request_file)["source"])
    source.unlink()
    source.symlink_to(source.parent / ".netrc")
    transport = Transport()
    with pytest.raises(ValueError, match="regular files"):
        dispatch(monkeypatch, request_file, transport)
    assert not any(c[0] == "ftk" for c in transport.commands)
