"""Reject retired project branding in active tracked text files."""

from __future__ import annotations

import subprocess
from pathlib import Path

STALE_IDENTIFIERS = (
    "".join(("torax", "_rl")),
    "".join(("torax", "-rl")),
    "".join(("TORAX", "-RL")),
    "".join(("Plasma", "X")),
    "".join(("Torax", "Env")),
    "".join(("Torax", "TruncationWrapper")),
)

# Immutable historical run identifiers belong in ignored manifests, so active
# tracked files currently need no exceptions.
ALLOWLIST: frozenset[Path] = frozenset()


def main() -> None:
    active_paths = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        check=True,
        capture_output=True,
    ).stdout.split(b"\0")
    failures: list[str] = []
    for raw_path in active_paths:
        if not raw_path:
            continue
        path = Path(raw_path.decode())
        if path in ALLOWLIST or not path.is_file():
            continue
        try:
            text = path.read_text()
        except UnicodeDecodeError:
            continue
        for line_number, line in enumerate(text.splitlines(), start=1):
            for identifier in STALE_IDENTIFIERS:
                if identifier in line:
                    failures.append(f"{path}:{line_number}: {identifier}")
    if failures:
        details = "\n".join(failures)
        raise SystemExit(f"retired project identifiers found:\n{details}")


if __name__ == "__main__":
    main()
