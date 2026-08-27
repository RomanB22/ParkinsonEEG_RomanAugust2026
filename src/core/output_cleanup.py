"""Bounded cleanup helpers for retired generated analysis products."""

from __future__ import annotations

import shutil
from pathlib import Path


def remove_retired_band_outputs(
    output_dir: str | Path,
    retired_bands: tuple[str, ...] = (
        "broad_5_15",
        "low_gamma",
        "low_beta",
        "high_beta",
    ),
) -> list[Path]:
    """Remove generated paths whose names contain a retired band identifier."""
    root = Path(output_dir)
    if not root.exists():
        return []
    matches = sorted(
        (
            path
            for path in root.rglob("*")
            if any(band in path.name for band in retired_bands)
        ),
        key=lambda path: len(path.parts),
        reverse=True,
    )
    removed: list[Path] = []
    for path in matches:
        if not path.exists():
            continue
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
        removed.append(path)
    return removed
