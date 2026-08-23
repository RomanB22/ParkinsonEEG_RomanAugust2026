"""Continuous artifact annotation and fixed-epoch rejection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mne
import numpy as np
import pandas as pd

from .channels import robust_z


def _merge_intervals(records: list[dict[str, Any]], gap_sec: float) -> list[dict[str, Any]]:
    if not records:
        return []
    ordered = sorted(records, key=lambda item: (item["onset_sec"], item["description"]))
    merged = [ordered[0].copy()]
    for record in ordered[1:]:
        current = merged[-1]
        current_end = current["onset_sec"] + current["duration_sec"]
        record_end = record["onset_sec"] + record["duration_sec"]
        if record["onset_sec"] <= current_end + gap_sec:
            current["duration_sec"] = max(current_end, record_end) - current["onset_sec"]
            descriptions = set(current["description"].split("+"))
            descriptions.update(record["description"].split("+"))
            current["description"] = "+".join(sorted(descriptions))
            current["max_peak_to_peak_uv"] = max(
                current["max_peak_to_peak_uv"], record["max_peak_to_peak_uv"]
            )
            current["global_peak_to_peak_robust_z"] = max(
                current["global_peak_to_peak_robust_z"],
                record["global_peak_to_peak_robust_z"],
            )
        else:
            merged.append(record.copy())
    return merged


def annotate_large_artifacts(raw, config: dict[str, Any]):
    """Add BAD annotations for only very large amplitude/global transients."""
    annotated = raw.copy()
    picks = mne.pick_types(annotated.info, eeg=True, exclude="bads")
    data_uv = annotated.get_data(picks=picks) * 1e6
    sfreq = float(annotated.info["sfreq"])
    window = max(1, int(round(float(config["window_sec"]) * sfreq)))
    step = max(1, int(round(float(config["step_sec"]) * sfreq)))

    starts = np.arange(0, max(1, data_uv.shape[1] - window + 1), step, dtype=int)
    channel_ptp = np.asarray(
        [np.ptp(data_uv[:, start : start + window], axis=1) for start in starts]
    )
    max_ptp = np.max(channel_ptp, axis=1)
    global_ptp = np.median(channel_ptp, axis=1)
    global_z = robust_z(global_ptp)

    absolute_threshold = float(config["absolute_peak_to_peak_uv"])
    global_threshold = float(config["global_peak_to_peak_robust_z"])
    global_minimum = float(config["minimum_global_peak_to_peak_uv"])
    padding = float(config["padding_sec"])
    duration = float(config["window_sec"]) + 2.0 * padding

    records: list[dict[str, Any]] = []
    for index, start in enumerate(starts):
        labels: list[str] = []
        if max_ptp[index] >= absolute_threshold:
            labels.append("BAD_amplitude")
        if global_z[index] >= global_threshold and global_ptp[index] >= global_minimum:
            labels.append("BAD_movement")
        if not labels:
            continue
        onset = max(0.0, start / sfreq - padding)
        end = min(annotated.times[-1], onset + duration)
        records.append(
            {
                "onset_sec": onset,
                "duration_sec": max(0.0, end - onset),
                "description": "+".join(labels),
                "max_peak_to_peak_uv": float(max_ptp[index]),
                "global_peak_to_peak_uv": float(global_ptp[index]),
                "global_peak_to_peak_robust_z": float(global_z[index]),
            }
        )

    records = _merge_intervals(records, float(config["merge_gap_sec"]))
    if records:
        new_annotations = mne.Annotations(
            onset=[record["onset_sec"] for record in records],
            duration=[record["duration_sec"] for record in records],
            description=[record["description"] for record in records],
            orig_time=annotated.annotations.orig_time,
        )
        annotated.set_annotations(annotated.annotations + new_annotations)
    return annotated, pd.DataFrame.from_records(records)


@dataclass
class EpochResult:
    epochs: mne.Epochs
    all_epochs: mne.Epochs
    rejection_table: pd.DataFrame
    n_initial: int
    n_rejected: int
    n_retained: int


def create_and_reject_epochs(raw, config: dict[str, Any]) -> EpochResult:
    duration = float(config["duration_sec"])
    overlap = float(config["overlap_sec"])
    all_epochs = mne.make_fixed_length_epochs(
        raw,
        duration=duration,
        overlap=overlap,
        preload=True,
        reject_by_annotation=False,
        verbose="ERROR",
    )
    epochs = mne.make_fixed_length_epochs(
        raw,
        duration=duration,
        overlap=overlap,
        preload=True,
        reject_by_annotation=True,
        verbose="ERROR",
    )

    if len(epochs):
        data_uv = epochs.get_data(picks="eeg") * 1e6
        channel_ptp = np.ptp(data_uv, axis=2)
        epoch_ptp = np.max(channel_ptp, axis=1)
        epoch_z = robust_z(epoch_ptp)
        channel_log_z = np.column_stack(
            [
                robust_z(np.log10(np.maximum(channel_ptp[:, index], np.finfo(float).tiny)))
                for index in range(channel_ptp.shape[1])
            ]
        )
        max_channel_z = np.max(channel_log_z, axis=1)
        trigger_channel_index = np.argmax(channel_log_z, axis=1)
        fixed_bad = epoch_ptp >= float(config["peak_to_peak_uv"])
        robust_bad = epoch_z >= float(config["robust_z"])
        channel_robust_bad = max_channel_z >= float(config["robust_z"])
        bad_local = np.where(fixed_bad | robust_bad | channel_robust_bad)[0]
        amplitude_records = {
            int(epochs.selection[index]): {
                "max_peak_to_peak_uv": float(epoch_ptp[index]),
                "peak_to_peak_robust_z": float(epoch_z[index]),
                "max_channel_peak_to_peak_robust_z": float(max_channel_z[index]),
                "trigger_channel": epochs.ch_names[int(trigger_channel_index[index])],
                "amplitude_reason": ";".join(
                    reason
                    for condition, reason in (
                        (fixed_bad[index], "absolute_peak_to_peak"),
                        (robust_bad[index], "robust_peak_to_peak_outlier"),
                        (channel_robust_bad[index], "channel_wise_robust_peak_to_peak_outlier"),
                    )
                    if condition
                ),
            }
            for index in range(len(epochs))
        }
        if len(bad_local):
            epochs.drop(bad_local, reason="BAD_peak_to_peak", verbose="ERROR")
    else:
        amplitude_records = {}

    records = []
    for index, reasons in enumerate(epochs.drop_log):
        detail = amplitude_records.get(index, {})
        records.append(
            {
                "epoch_index": index,
                "onset_sec": index * (duration - overlap),
                "accepted": len(reasons) == 0,
                "reasons": ";".join(str(reason) for reason in reasons),
                "max_peak_to_peak_uv": detail.get("max_peak_to_peak_uv", np.nan),
                "peak_to_peak_robust_z": detail.get("peak_to_peak_robust_z", np.nan),
                "max_channel_peak_to_peak_robust_z": detail.get(
                    "max_channel_peak_to_peak_robust_z", np.nan
                ),
                "trigger_channel": detail.get("trigger_channel", ""),
                "amplitude_reason": detail.get("amplitude_reason", ""),
            }
        )
    table = pd.DataFrame.from_records(records)
    n_initial = len(records)
    n_retained = len(epochs)
    return EpochResult(epochs, all_epochs, table, n_initial, n_initial - n_retained, n_retained)
