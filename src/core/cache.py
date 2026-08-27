"""Small, shared helpers for validated cross-stage feature caches."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def same_json_settings(left: Any, right: Any) -> bool:
    """Compare JSON-compatible settings without integer/float spelling noise."""
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(
            same_json_settings(left[key], right[key]) for key in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            same_json_settings(a, b) for a, b in zip(left, right)
        )
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return float(left) == float(right)
    return left == right


def replace_with_relative_symlink(source: Path, destination: Path) -> None:
    """Expose one cache file at a stable legacy path without copying it."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink() or destination.exists():
        destination.unlink()
    relative_source = os.path.relpath(source.resolve(), destination.parent.resolve())
    destination.symlink_to(relative_source)
