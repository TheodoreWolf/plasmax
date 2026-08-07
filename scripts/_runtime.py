"""Runtime setup shared by experiment entrypoints."""

import os


def set_default_xla_flags(*flags: str) -> None:
    """Add XLA flags that the caller has not already configured."""
    existing = os.environ.get("XLA_FLAGS", "")
    configured = {token.split("=", maxsplit=1)[0] for token in existing.split()}
    missing = [
        flag for flag in flags if flag.split("=", maxsplit=1)[0] not in configured
    ]
    if missing:
        os.environ["XLA_FLAGS"] = " ".join([existing, *missing]).strip()
