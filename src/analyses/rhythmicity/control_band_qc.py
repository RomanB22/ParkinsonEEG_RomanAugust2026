"""Shared helpers for the Control-defined ABBA-band sensitivity analyses."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


QC_DIRECTORY = "control_band_qc"
QC_BAND_DEFINITION = "dataset_control_group_mean_abba"


def control_defined_segments(segments: pd.DataFrame) -> pd.DataFrame:
    """Return valid Control segments with direction-stable QC band names.

    The returned rows are dataset-level definitions: their frequency limits
    are applied unchanged to every diagnostic/medication group in that
    dataset. The original ABBA name is retained in ``source_band_name`` while
    ``band_name`` follows the requested ``{canonical_band}_{low|high}`` form.
    """
    required = {
        "dataset", "group", "band_name", "canonical_region", "direction",
        "start_hz", "end_hz",
    }
    missing = sorted(required.difference(segments.columns))
    if missing:
        raise ValueError(f"ABBA segment table is missing columns: {', '.join(missing)}")

    selected = segments.loc[
        segments["group"].astype(str).eq("Control")
        & segments["end_hz"].gt(segments["start_hz"])
    ].copy()
    if selected.empty:
        raise RuntimeError("No valid Control group-mean ABBA segments were found")
    selected["source_group"] = "Control"
    selected["source_band_name"] = selected["band_name"].astype(str)
    selected["band_name"] = (
        selected["canonical_region"].astype(str)
        + "_"
        + selected["direction"].astype(str)
    )
    selected["band_definition"] = QC_BAND_DEFINITION

    duplicates = selected.duplicated(["dataset", "band_name"], keep=False)
    if duplicates.any():
        labels = (
            selected.loc[duplicates, ["dataset", "band_name"]]
            .drop_duplicates()
            .astype(str)
            .agg(":".join, axis=1)
            .tolist()
        )
        raise ValueError(
            "Control ABBA segments do not map uniquely to {band}_{direction}: "
            + ", ".join(labels)
        )
    return selected


def qc_output_root(output_root: str | Path) -> Path:
    """Keep sensitivity artifacts isolated from the primary analysis."""
    return Path(output_root) / QC_DIRECTORY
