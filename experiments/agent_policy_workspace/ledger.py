"""Version policy bundles and keep an append-only experiment notebook."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import stat
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Any
from uuid import uuid4

from experiments.agent_policy_search.artifacts import plain, timestamp

__all__ = ["Ledger", "bundle_sha256", "restore_bundle", "snapshot_bundle"]

_BUNDLE_FILES = ("policy.py", "custom_primitives.py")


def _read_bundle(directory: Path, *, required: bool = True) -> dict[str, bytes | None]:
    result = {}
    for name in _BUNDLE_FILES:
        path = directory / name
        try:
            descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        except FileNotFoundError:
            if name == "policy.py" and required:
                raise
            result[name] = None
            continue
        except OSError as error:
            raise ValueError(
                f"bundle file must be a regular non-symlink file: {path}"
            ) from error
        with os.fdopen(descriptor, "rb") as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise ValueError(f"bundle file must be regular: {path}")
            result[name] = stream.read()
    return result


def _hash_bundle(bundle: dict[str, bytes | None]) -> str:
    digest = hashlib.sha256()
    for name in _BUNDLE_FILES:
        data = bundle[name]
        digest.update(name.encode() + b"\0" + (b"0" if data is None else b"1"))
        if data is not None:
            digest.update(len(data).to_bytes(8, "big") + data)
    return digest.hexdigest()


def _checked_hash(value: str) -> str:
    if re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("candidate hash must be 64 lowercase hexadecimal characters")
    return value


def bundle_sha256(directory: Path) -> str:
    """Hash filenames, exact bytes, and the optional helper's presence."""
    return _hash_bundle(_read_bundle(directory))


def snapshot_bundle(workspace: Path, archive_root: Path) -> tuple[str, Path]:
    """Archive exact source once; reject a changed or incomplete existing archive."""
    bundle = _read_bundle(workspace)
    identity = _hash_bundle(bundle)
    archive_root.mkdir(parents=True, exist_ok=True)
    candidate = archive_root / identity
    if not candidate.exists() and not candidate.is_symlink():
        with tempfile.TemporaryDirectory(
            prefix=".candidate-", dir=archive_root
        ) as temp:
            staging = Path(temp)
            for name, data in bundle.items():
                if data is not None:
                    with (staging / name).open("xb") as stream:
                        stream.write(data)
            with suppress(FileExistsError):
                staging.rename(candidate)
    if candidate.is_symlink() or bundle_sha256(candidate) != identity:
        raise ValueError(f"candidate archive does not match its hash: {identity}")
    return identity, candidate


def _replace_bytes(path: Path, data: bytes) -> None:
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def restore_bundle(workspace: Path, candidate: Path) -> Path:
    """Restore both source files, preserving displaced contents before any change."""
    identity = _checked_hash(candidate.name)
    if candidate.is_symlink():
        raise ValueError("candidate archive must not be a symlink")
    incoming = _read_bundle(candidate)
    if _hash_bundle(incoming) != identity:
        raise ValueError("candidate archive does not match its hash")
    original = _read_bundle(workspace, required=False)
    backup = workspace / ".policy_backups" / uuid4().hex
    backup.mkdir(parents=True, exist_ok=False)
    for name, data in original.items():
        if data is not None:
            with (backup / name).open("xb") as stream:
                stream.write(data)
    for name, data in incoming.items():
        if data is None:
            (workspace / name).unlink(missing_ok=True)
        else:
            _replace_bytes(workspace / name, data)
    if bundle_sha256(workspace) != identity:
        raise RuntimeError(
            f"restored bundle hash mismatch; previous files are in {backup}"
        )
    return backup


class Ledger:
    """Structured history; official score eligibility belongs to the evaluator."""

    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.path = workspace / "policy_ledger.jsonl"
        self.markdown = workspace / "experiment_ledger.md"

    @contextmanager
    def _locked(self) -> Iterator[None]:
        self.workspace.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(
            self.workspace / "policy_ledger.jsonl.lock",
            os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW,
            0o600,
        )
        with os.fdopen(descriptor, "a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def events(self) -> list[dict[str, Any]]:
        if self.path.is_symlink():
            raise ValueError("ledger must not be a symlink")
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text().splitlines()]

    @staticmethod
    def _current(events: list[dict[str, Any]]) -> str | None:
        current = None
        for entry in events:
            if entry["event"] == "start":
                current = entry["attempt"]
            elif entry["event"] == "conclude":
                current = None
        return current

    def current_attempt(self) -> str | None:
        return self._current(self.events())

    def best_candidate(self) -> str | None:
        for entry in reversed(self.events()):
            if "best_candidate_sha256" in entry:
                return entry["best_candidate_sha256"]
        return None

    def _append(
        self, events: list[dict[str, Any]], event: str, **fields: Any
    ) -> dict[str, Any]:
        entry = plain({**fields, "event": event, "time": timestamp()})
        encoded = json.dumps(entry, allow_nan=False) + "\n"
        descriptor = os.open(
            self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW, 0o600
        )
        with os.fdopen(descriptor, "a") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        events.append(entry)
        lines = [
            "# Policy experiment ledger",
            "",
            "> Generated from `policy_ledger.jsonl`; edit through the ledger commands.",
        ]
        for item in events:
            lines.extend(["", f"## {item.get('attempt', 'run')} — {item['event']}", ""])
            lines.extend(
                f"- {key.replace('_', ' ')}: {value}"
                for key, value in item.items()
                if key not in {"event", "attempt"}
            )
        _replace_bytes(self.markdown, ("\n".join(lines) + "\n").encode())
        return entry

    def append(self, event: str, **fields: Any) -> dict[str, Any]:
        """Record evaluator-owned events without bypassing attempt transitions."""
        if event in {"start", "conclude", "select_best"}:
            raise ValueError("use the named ledger method for attempt transitions")
        with self._locked():
            return self._append(self.events(), event, **fields)

    def start(
        self,
        *,
        title: str,
        hypothesis: str,
        planned_change: str,
        success_criterion: str,
        parent: str | None = None,
    ) -> str:
        if not all(
            value.strip()
            for value in (title, hypothesis, planned_change, success_criterion)
        ):
            raise ValueError(
                "an attempt needs title, hypothesis, planned change "
                "and success criterion"
            )
        with self._locked():
            events = self.events()
            if self._current(events) is not None:
                raise ValueError("conclude the current attempt before starting another")
            previous = [
                event["attempt"] for event in events if event["event"] == "start"
            ]
            if parent is not None and parent not in previous:
                raise ValueError("parent must identify an existing attempt")
            attempt = f"A{len(previous):02d}"
            self._append(
                events,
                "start",
                attempt=attempt,
                title=title,
                hypothesis=hypothesis,
                planned_change=planned_change,
                success_criterion=success_criterion,
                parent=parent,
            )
            return attempt

    def observe(
        self,
        *,
        summary: str,
        kind: str = "exploratory",
        candidate_sha256: str | None = None,
    ) -> dict[str, Any]:
        if candidate_sha256 is not None:
            _checked_hash(candidate_sha256)
        with self._locked():
            events = self.events()
            attempt = self._current(events)
            if attempt is None:
                raise ValueError("start an attempt before recording observations")
            return self._append(
                events,
                "observe",
                attempt=attempt,
                summary=summary,
                kind=kind,
                candidate_sha256=candidate_sha256,
            )

    def select_best(self, candidate_sha256: str, reason: str) -> dict[str, Any]:
        _checked_hash(candidate_sha256)
        if not reason.strip():
            raise ValueError("selection requires a reason")
        with self._locked():
            events = self.events()
            return self._append(
                events,
                "select_best",
                attempt=self._current(events),
                best_candidate_sha256=candidate_sha256,
                best_reason=reason,
            )

    def conclude(
        self,
        *,
        decision: str,
        summary: str,
        best_candidate_sha256: str,
        best_reason: str,
        candidate_sha256: str | None = None,
        limitation: str = "",
    ) -> dict[str, Any]:
        _checked_hash(best_candidate_sha256)
        if candidate_sha256 is not None:
            _checked_hash(candidate_sha256)
        if not all(value.strip() for value in (decision, summary, best_reason)):
            raise ValueError("conclusion requires a decision, summary and best reason")
        with self._locked():
            events = self.events()
            attempt = self._current(events)
            if attempt is None:
                raise ValueError("there is no open attempt to conclude")
            return self._append(
                events,
                "conclude",
                attempt=attempt,
                decision=decision,
                summary=summary,
                best_candidate_sha256=best_candidate_sha256,
                best_reason=best_reason,
                candidate_sha256=candidate_sha256,
                limitation=limitation,
            )
