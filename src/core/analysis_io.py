"""Shared, strict input helpers for every downstream analysis."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import pandas as pd


SUBJECT_PATTERN = re.compile(r"(sub-\d+)")


def load_participants(
    path: str | Path,
    *,
    required: Iterable[str] = ("participant_id", "GROUP"),
) -> pd.DataFrame:
    """Read CSV/TSV metadata and enforce unique participant identifiers."""
    source = Path(path)
    separator = "\t" if source.suffix.lower() == ".tsv" else ","
    table = pd.read_csv(source, sep=separator)
    missing = sorted(set(required) - set(table.columns))
    if missing:
        raise ValueError(f"Participant table is missing columns: {missing}")
    if table["participant_id"].duplicated().any():
        raise ValueError("Participant IDs must be unique")
    return table


def discover_epoch_files(directory: str | Path, pattern: str) -> dict[str, Path]:
    """Map each participant to exactly one cleaned epoch file."""
    files: dict[str, Path] = {}
    for path in sorted(Path(directory).glob(pattern)):
        match = SUBJECT_PATTERN.search(path.name)
        if match is None:
            continue
        subject_id = match.group(1)
        if subject_id in files:
            raise ValueError(f"Multiple epoch files found for {subject_id}")
        files[subject_id] = path
    return files

