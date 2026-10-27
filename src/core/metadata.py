"""Dataset-wide inspection and tabular metadata outputs."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from .dataset import (
    channel_names_from_sidecar,
    discover_recordings,
    load_participants,
    load_subject,
    read_json,
    recording_id_from_path,
    session_id_from_path,
    sidecar_paths,
    subject_id_from_path,
)
from . import qc


def expected_channels_from_dataset(dataset_dir: str | Path, task: str, auxiliary_names: list[str]) -> list[str]:
    channel_sets = [
        set(channel_names_from_sidecar(path, auxiliary_names))
        for path in discover_recordings(dataset_dir, task)
    ]
    return sorted(set().union(*channel_sets))


def inspect_dataset(config: dict[str, Any]) -> dict[str, Any]:
    dataset_dir = Path(config["project"]["dataset_dir"])
    output_dir = Path(config["project"]["output_dir"])
    metadata_dir = output_dir / "metadata"
    inspection_dir = output_dir / "qc" / "dataset_inspection"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    inspection_dir.mkdir(parents=True, exist_ok=True)

    participants = load_participants(dataset_dir)
    recordings = discover_recordings(dataset_dir, config["project"]["task"])
    auxiliary = config["channels"]["auxiliary_names"]
    channel_sets: dict[str, set[str]] = {}
    rows = []
    channel_metadata_rows = []
    for set_path in recordings:
        participant_id = subject_id_from_path(set_path)
        subject_id = recording_id_from_path(set_path)
        session_id = session_id_from_path(set_path)
        paths = sidecar_paths(set_path)
        sidecar = read_json(paths["eeg_json"])
        channels = pd.read_csv(paths["channels_tsv"], sep="\t")
        stored = [str(name) for name in channels["name"]]
        eeg = [name for name in stored if name not in auxiliary]
        auxiliaries = [name for name in stored if name in auxiliary]
        channel_sets[subject_id] = set(eeg)
        participant_row = participants.loc[
            participants["participant_id"] == participant_id
        ]
        if len(participant_row) == 1 and "GROUP" in participants:
            group = str(participant_row.iloc[0]["GROUP"])
        elif participant_id.removeprefix("sub-").lower().startswith("pd"):
            group = "PD"
        elif participant_id.removeprefix("sub-").lower().startswith("hc"):
            group = "Control"
        else:
            group = "unknown"
        reference = str(sidecar.get("EEGReference", "unknown"))
        rows.append(
            {
                "subject_id": subject_id,
                "participant_id": participant_id,
                "session_id": session_id,
                "group": group,
                "source_file": str(set_path.resolve()),
                "source_format": set_path.suffix.lower().lstrip("."),
                # Backward-compatible aliases retained for the original
                # EEGLAB inspection outputs.
                "set_file": (
                    str(set_path.resolve())
                    if set_path.suffix.lower() == ".set"
                    else ""
                ),
                "fdt_file_exists": (
                    set_path.with_suffix(".fdt").exists()
                    if set_path.suffix.lower() == ".set"
                    else pd.NA
                ),
                "companion_fdt_exists": (
                    set_path.with_suffix(".fdt").exists()
                    if set_path.suffix.lower() == ".set"
                    else pd.NA
                ),
                "sampling_rate": sidecar.get("SamplingFrequency"),
                "recording_duration_sec": sidecar.get("RecordingDuration"),
                "stored_channel_count": len(stored),
                "sidecar_eeg_channel_count": sidecar.get("EEGChannelCount"),
                "analysis_eeg_channel_count": len(eeg),
                "auxiliary_channels": ";".join(auxiliaries),
                "original_reference": reference,
                "reference_present_as_recorded_channel": reference in eeg,
                "line_frequency": sidecar.get("PowerLineFrequency"),
                "institution": sidecar.get("InstitutionAddress", sidecar.get("InstitutionName", "unknown")),
            }
        )
        for _, channel in channels.iterrows():
            channel_metadata_rows.append(
                {
                    "subject_id": subject_id,
                    "channel": str(channel["name"]),
                    "sidecar_type": channel.get("type", "n/a"),
                    "sidecar_units": channel.get("units", "n/a"),
                    "classified_as": "auxiliary" if str(channel["name"]) in auxiliary else "eeg",
                }
            )

    recording_table = pd.DataFrame(rows).sort_values("subject_id")
    union = sorted(set().union(*channel_sets.values()))
    common = sorted(set.intersection(*channel_sets.values()))
    availability = pd.DataFrame(
        [
            {"subject_id": subject_id, **{channel: int(channel in channels) for channel in union}}
            for subject_id, channels in sorted(channel_sets.items())
        ]
    )
    signatures = Counter(tuple(sorted(channels)) for channels in channel_sets.values())
    signature_rows = [
        {"n_subjects": count, "n_channels": len(channels), "channels": ";".join(channels)}
        for channels, count in signatures.items()
    ]

    participants.to_csv(metadata_dir / "subjects.csv", index=False)
    recording_table.to_csv(metadata_dir / "recordings.csv", index=False)
    pd.DataFrame(channel_metadata_rows).to_csv(metadata_dir / "channel_metadata.csv", index=False)
    availability.to_csv(metadata_dir / "channel_availability.csv", index=False)
    pd.DataFrame(signature_rows).sort_values(["n_subjects", "n_channels"], ascending=False).to_csv(
        metadata_dir / "channel_signatures.csv", index=False
    )
    common_payload = {
        "common_channels_present_in_all_subjects": common,
        "n_common_channels": len(common),
        "union_of_recorded_eeg_channels": union,
        "n_union_channels": len(union),
        "note": "This list is for later group analysis. Individual cleaned files retain every usable recorded EEG channel.",
    }
    (metadata_dir / "common_channels.json").write_text(
        json.dumps(common_payload, indent=2), encoding="utf-8"
    )

    groups = recording_table[["participant_id", "group"]].drop_duplicates()[
        "group"
    ].value_counts(dropna=False).to_dict()
    rates = sorted(recording_table["sampling_rate"].dropna().unique().tolist())
    references = recording_table["original_reference"].value_counts().to_dict()
    source_formats = recording_table["source_format"].value_counts().to_dict()
    if set(source_formats) == {"set"}:
        decision_notes = """- EEGLAB `.set` files require their external `.fdt` companions.
- Sidecars label channel type and units as `n/a`; MNE converts EEG samples to volts.
- Pz is the reported online reference but is not stored as a data channel; CPz is recorded and Pz is not synthesized.
- Medication status is unavailable in `participants.tsv`."""
    else:
        decision_notes = """- BioSemi BDF recordings are loaded directly with MNE and converted to volts.
- Repeated sessions retain recording IDs such as `sub-pd3_ses-off`; biological participant IDs remain separate.
- Medication state is encoded by the BIDS session (`ses-off`/`ses-on`)."""
    report = f"""# Dataset inspection report

## Dataset structure

- Participants in `participants.tsv`: {len(participants)}
- Resting-state EEG recordings: {len(recordings)}
- Group counts: {groups}
- Sampling rates found: {rates} Hz
- Recording duration range: {recording_table['recording_duration_sec'].min():.2f}–{recording_table['recording_duration_sec'].max():.2f} s
- Original reference values: {references}
- Distinct recorded EEG layouts: {len(signatures)}
- EEG channels present in every recording: {len(common)}
- EEG-channel union: {len(union)}

## Decisions and cautions

- Source formats: {source_formats}.
{decision_notes}
- Configured auxiliary channels are excluded from scalp EEG but retained in provenance.
- Valid imported electrode positions are preserved; a standard montage is only a fallback for recorded channels with missing coordinates.
- Final data are filtered {config['filter']['l_freq']:g}–{config['filter']['h_freq']:g} Hz, notched at {config['filter']['notch_freq_hz']:g} Hz, common-average referenced, and resampled to {config['resampling']['target_sfreq']:g} Hz.

## Generated metadata

- `subjects.csv`: participant metadata preserved exactly as supplied.
- `recordings.csv`: one row per source recording.
- `channel_metadata.csv`: one row per participant/channel.
- `channel_availability.csv`: recorded EEG-channel presence matrix.
- `common_channels.json`: common and union channel lists for later analysis.
- `channel_signatures.csv`: counts of distinct layouts.
"""
    (metadata_dir / "dataset_inspection_report.md").write_text(report, encoding="utf-8")

    # Basic raw inspection for one participant from each group.
    representative_ids = []
    for group in ("PD", "Control"):
        if "GROUP" in participants:
            matches = participants.loc[
                participants["GROUP"] == group, "participant_id"
            ]
        else:
            prefix = "sub-pd" if group == "PD" else "sub-hc"
            matches = participants.loc[
                participants["participant_id"].astype(str).str.startswith(prefix),
                "participant_id",
            ]
        if len(matches):
            representative_ids.append(str(matches.iloc[0]))
    for subject_id in representative_ids:
        set_path = next(path for path in recordings if subject_id_from_path(path) == subject_id)
        raw, _ = load_subject(set_path, auxiliary)
        channels = qc.select_channels(raw, config["qc"]["preferred_channels"])
        qc.plot_signal(
            raw,
            channels,
            float(config["qc"]["trace_start_sec"]),
            float(config["qc"]["trace_duration_sec"]),
            f"{subject_id} raw EEG inspection",
            inspection_dir / f"{subject_id}_raw_signal.png",
            int(config["qc"]["dpi"]),
        )
        qc.plot_psd(
            raw,
            channels,
            float(config["qc"]["psd_fmin_hz"]),
            float(config["qc"]["psd_fmax_hz"]),
            f"{subject_id} raw PSD inspection",
            inspection_dir / f"{subject_id}_raw_psd.png",
            int(config["qc"]["dpi"]),
        )

    return {
        "n_subjects": len(participants),
        "n_recordings": len(recordings),
        "group_counts": groups,
        "common_channels": common,
        "expected_channels": union,
        "n_layouts": len(signatures),
    }


def update_preprocessing_qc(output_dir: str | Path, rows: list[dict[str, Any]]) -> Path:
    """Update subject rows without discarding QC from earlier completed runs."""
    path = Path(output_dir) / "metadata" / "preprocessing_qc.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame(rows)
    if path.exists():
        existing = pd.read_csv(path)
        existing = existing.loc[~existing["subject_id"].isin(new["subject_id"])]
        new = pd.concat([existing, new], ignore_index=True)
    new.sort_values("subject_id").to_csv(path, index=False)
    return path
