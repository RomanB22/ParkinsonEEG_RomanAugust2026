"""Set safe non-interactive runtime defaults before importing MNE/Matplotlib."""

from __future__ import annotations

import os
from pathlib import Path


def configure_runtime() -> None:
    """Keep library caches out of the user's home and disable GUI plotting."""
    cache_root = Path(os.environ.get("TMPDIR", "/tmp")) / "parkinson_eeg_runtime"
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MNE_DONTWRITE_HOME", "true")
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "cache"))


configure_runtime()
