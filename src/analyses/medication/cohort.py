"""Build session-aware participant and recording metadata for ds002778."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from core.dataset import (
    discover_recordings,
    load_participants,
    recording_id_from_path,
    session_id_from_path,
    subject_id_from_path,
)


CONDITION_BY_SESSION = {
    "ses-hc": "HC",
    "ses-off": "PD_OFF",
    "ses-on": "PD_ON",
}


def _numeric(value: Any) -> float:
    parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(parsed) if pd.notna(parsed) else np.nan


def _behavior_metadata(recording_path: Path) -> dict[str, Any]:
    session_dir = recording_path.parents[1]
    matches = sorted((session_dir / "beh").glob("*_task-rest_beh.json"))
    if len(matches) != 1:
        return {
            "behavior_json": "",
            "total_updrs": np.nan,
            "updrs_18_26": np.nan,
            "hoehn_yahr": np.nan,
        }
    path = matches[0]
    payload = json.loads(path.read_text(encoding="utf-8"))
    questionnaires = payload.get("questionairres", payload.get("questionnaires", {}))
    return {
        "behavior_json": str(path.resolve()),
        "total_updrs": _numeric(questionnaires.get("Total UPDRS")),
        "updrs_18_26": _numeric(questionnaires.get("UPDRS 18-26")),
        "hoehn_yahr": _numeric(questionnaires.get("H&Y")),
    }


def build_cohort(
    dataset_dir: str | Path,
    *,
    task: str = "rest",
    expected_counts: dict[str, int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return canonical participant-level and recording-level cohort tables."""
    dataset_dir = Path(dataset_dir)
    source = load_participants(dataset_dir).copy()
    required = {"participant_id", "age", "gender", "MMSE"}
    missing = sorted(required - set(source))
    if missing:
        raise ValueError(f"ds002778 participants.tsv is missing columns: {missing}")
    if source["participant_id"].duplicated().any():
        raise ValueError("ds002778 participant IDs must be unique")

    participants = source.rename(
        columns={
            "age": "age_years",
            "gender": "sex",
            "MMSE": "mmse",
            "NAART": "naart",
        }
    ).copy()
    participants["participant_id"] = participants["participant_id"].astype(str)
    participants["diagnosis"] = np.where(
        participants["participant_id"].str.startswith("sub-pd"), "PD", "HC"
    )
    participants["age_years"] = pd.to_numeric(
        participants["age_years"], errors="raise"
    )
    participants["mmse"] = pd.to_numeric(participants["mmse"], errors="raise")
    participants["sex"] = participants["sex"].astype(str).str.upper()
    if not set(participants["sex"]).issubset({"F", "M"}):
        raise ValueError("ds002778 gender must contain only f/m")
    participants["sex_male"] = participants["sex"].eq("M").astype(int)
    participants["disease_duration_years"] = pd.to_numeric(
        participants.get("disease_duration"), errors="coerce"
    )
    participants["naart"] = pd.to_numeric(participants.get("naart"), errors="coerce")
    participants["curation_note"] = participants.get("notes", "").fillna("").astype(str)
    participants["provenance_sensitivity_exclusion"] = participants[
        "curation_note"
    ].str.contains("preprocessed data", case=False, regex=False)

    lookup = participants.set_index("participant_id")
    rows: list[dict[str, Any]] = []
    for path in discover_recordings(dataset_dir, task):
        participant_id = subject_id_from_path(path)
        if participant_id not in lookup.index:
            raise ValueError(f"Recording lacks participants.tsv metadata: {path}")
        session_id = session_id_from_path(path)
        if session_id not in CONDITION_BY_SESSION:
            raise ValueError(f"Unsupported ds002778 session in {path}: {session_id}")
        condition = CONDITION_BY_SESSION[session_id]
        diagnosis = str(lookup.loc[participant_id, "diagnosis"])
        if condition == "HC" and diagnosis != "HC":
            raise ValueError(f"{path}: HC session has non-HC participant metadata")
        if condition != "HC" and diagnosis != "PD":
            raise ValueError(f"{path}: medication session has non-PD participant metadata")
        rows.append(
            {
                "recording_id": recording_id_from_path(path),
                "participant_id": participant_id,
                "session_id": session_id,
                "condition": condition,
                "diagnosis": diagnosis,
                "age_years": float(lookup.loc[participant_id, "age_years"]),
                "sex": str(lookup.loc[participant_id, "sex"]),
                "sex_male": int(lookup.loc[participant_id, "sex_male"]),
                "mmse": float(lookup.loc[participant_id, "mmse"]),
                "disease_duration_years": float(
                    lookup.loc[participant_id, "disease_duration_years"]
                ),
                "naart": float(lookup.loc[participant_id, "naart"]),
                "provenance_sensitivity_exclusion": bool(
                    lookup.loc[participant_id, "provenance_sensitivity_exclusion"]
                ),
                "curation_note": str(lookup.loc[participant_id, "curation_note"]),
                "source_eeg_file": str(path.resolve()),
                **_behavior_metadata(path),
            }
        )
    recordings = pd.DataFrame.from_records(rows).sort_values(
        ["participant_id", "condition"]
    ).reset_index(drop=True)
    if recordings["recording_id"].duplicated().any():
        raise ValueError("Recording IDs must be unique")

    sessions = recordings.groupby("participant_id")["condition"].agg(set)
    for participant_id, conditions in sessions.items():
        expected = {"HC"} if participant_id.startswith("sub-hc") else {
            "PD_OFF",
            "PD_ON",
        }
        if conditions != expected:
            raise ValueError(
                f"{participant_id}: expected conditions {sorted(expected)}, "
                f"found {sorted(conditions)}"
            )
    if expected_counts:
        observed = recordings["condition"].value_counts().to_dict()
        if observed != expected_counts:
            raise ValueError(
                f"Unexpected ds002778 condition counts: {observed}; "
                f"expected {expected_counts}"
            )

    participant_columns = [
        "participant_id",
        "diagnosis",
        "age_years",
        "sex",
        "sex_male",
        "mmse",
        "naart",
        "disease_duration_years",
        "rl_deficits",
        "curation_note",
        "provenance_sensitivity_exclusion",
    ]
    return (
        participants.reindex(columns=participant_columns)
        .sort_values("participant_id")
        .reset_index(drop=True),
        recordings,
    )


def write_cohort_tables(
    dataset_dir: str | Path,
    output_dir: str | Path,
    *,
    task: str = "rest",
    expected_counts: dict[str, int] | None = None,
) -> tuple[Path, Path]:
    participants, recordings = build_cohort(
        dataset_dir, task=task, expected_counts=expected_counts
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    participant_path = output_dir / "participants.csv"
    recording_path = output_dir / "recordings.csv"
    participants.to_csv(participant_path, index=False)
    recordings.to_csv(recording_path, index=False)
    return participant_path, recording_path
