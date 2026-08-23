"""Dataset discovery, metadata access, and EEGLAB loading."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .runtime import configure_runtime

configure_runtime()

import mne
import numpy as np
import pandas as pd


SUBJECT_PATTERN = re.compile(r"(sub-\d+)")


def subject_id_from_path(path: str | Path) -> str:
    match = SUBJECT_PATTERN.search(str(path))
    if not match:
        raise ValueError(f"Could not determine participant ID from {path}")
    return match.group(1)


def discover_recordings(dataset_dir: str | Path, task: str = "Rest") -> list[Path]:
    paths = sorted(Path(dataset_dir).glob(f"sub-*/eeg/*_task-{task}_eeg.set"))
    if not paths:
        raise FileNotFoundError(f"No task-{task} EEGLAB .set files found under {dataset_dir}")
    return paths


def recording_for_subject(dataset_dir: str | Path, subject_id: str, task: str = "Rest") -> Path:
    matches = list(Path(dataset_dir).glob(f"{subject_id}/eeg/{subject_id}_task-{task}_eeg.set"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one {task} recording for {subject_id}; found {len(matches)}")
    return matches[0]


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def sidecar_paths(set_path: str | Path) -> dict[str, Path]:
    set_path = Path(set_path)
    stem = set_path.name.removesuffix("_eeg.set")
    return {
        "eeg_json": set_path.with_name(f"{stem}_eeg.json"),
        "channels_tsv": set_path.with_name(f"{stem}_channels.tsv"),
        "electrodes_tsv": set_path.with_name(f"{stem}_electrodes.tsv"),
        "coordsystem_json": set_path.with_name(f"{stem}_coordsystem.json"),
    }


def load_participants(dataset_dir: str | Path) -> pd.DataFrame:
    path = Path(dataset_dir) / "participants.tsv"
    frame = pd.read_csv(path, sep="\t", dtype={"participant_id": str})
    if frame["participant_id"].duplicated().any():
        raise ValueError("participants.tsv contains duplicate participant_id values")
    return frame


def participant_metadata(dataset_dir: str | Path, subject_id: str) -> dict[str, Any]:
    frame = load_participants(dataset_dir)
    row = frame.loc[frame["participant_id"] == subject_id]
    if len(row) != 1:
        raise ValueError(f"Expected one participants.tsv row for {subject_id}; found {len(row)}")
    return row.iloc[0].to_dict()


def load_subject(set_path: str | Path, auxiliary_names: list[str] | None = None) -> tuple[mne.io.BaseRaw, dict[str, Any]]:
    """Load one recording and return EEG plus provenance metadata.

    Auxiliary channels are explicitly classified and dropped from the EEG
    analysis copy. Their names remain in the returned provenance record.
    """
    set_path = Path(set_path)
    subject_id = subject_id_from_path(set_path)
    paths = sidecar_paths(set_path)
    sidecar = read_json(paths["eeg_json"])
    raw = mne.io.read_raw_eeglab(set_path, preload=True, verbose="ERROR")

    raw.info["line_freq"] = float(sidecar.get("PowerLineFrequency", 60.0))
    auxiliaries = [name for name in (auxiliary_names or []) if name in raw.ch_names]
    if auxiliaries:
        raw.set_channel_types({name: "misc" for name in auxiliaries}, verbose="ERROR")
        raw.drop_channels(auxiliaries)

    # Preserve the EEGLAB electrode positions whenever every recorded EEG
    # channel has a finite coordinate. Use an extended standard montage only as
    # a fallback; this still never creates an unrecorded channel.
    imported_montage = raw.get_montage()
    imported_positions = imported_montage.get_positions()["ch_pos"] if imported_montage else {}
    positions_complete = all(
        name in imported_positions
        and np.all(np.isfinite(imported_positions[name]))
        and np.linalg.norm(imported_positions[name]) > 0
        for name in raw.ch_names
    )
    if positions_complete:
        montage_source = "EEGLAB source positions"
    else:
        montage = mne.channels.make_standard_montage("standard_1005")
        raw.set_montage(montage, on_missing="ignore", match_case=False, verbose="ERROR")
        montage_source = "standard_1005 fallback for incomplete source positions"

    provenance = {
        "subject_id": subject_id,
        "original_file": str(set_path.resolve()),
        "sidecar_file": str(paths["eeg_json"].resolve()),
        "original_reference": str(sidecar.get("EEGReference", "unknown")),
        "power_line_frequency": sidecar.get("PowerLineFrequency"),
        "sidecar_sampling_frequency": sidecar.get("SamplingFrequency"),
        "sidecar_recording_duration": sidecar.get("RecordingDuration"),
        "sidecar_eeg_channel_count": sidecar.get("EEGChannelCount"),
        "stored_channels": list(raw.ch_names) + auxiliaries,
        "analysis_eeg_channels": list(raw.ch_names),
        "dropped_auxiliary_channels": auxiliaries,
        "montage_source": montage_source,
    }
    return raw, provenance


def channel_names_from_sidecar(set_path: str | Path, auxiliary_names: list[str] | None = None) -> list[str]:
    channels = pd.read_csv(sidecar_paths(set_path)["channels_tsv"], sep="\t")
    excluded = set(auxiliary_names or [])
    return [str(name) for name in channels["name"] if str(name) not in excluded]
