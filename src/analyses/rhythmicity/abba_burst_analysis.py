"""Temporal burst analysis using group-specific ABBA frequency bands.

This is an optional sensitivity pipeline. It takes the group-mean frequency
intervals obtained from all-electrode LAVI profiles, filters each recording in
those intervals, detects temporal amplitude bursts, and summarizes burst
quantities and phase-aligned shapes by the named ABBA interval (for example,
``delta_1`` and ``delta_2``). The high/low ABBA direction is retained as
metadata and is never pooled into another interval. The detector follows the
logic of the rhythmicity paper's burst analysis (90th-percentile detection and
75th-percentile boundaries) while keeping the implementation deliberately
lightweight and reproducible for this project. Detection remains based on a
robust standardized all-electrode signal, whereas reported amplitudes and
burst shapes are extracted from an unstandardized representative EEG
electrode in microvolts.
"""

from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import mne
import numpy as np
import pandas as pd
from scipy.signal import butter, hilbert, sosfiltfilt
from scipy.stats import ttest_ind, t as student_t
from tqdm.auto import tqdm

from core.runtime import configure_runtime

configure_runtime()


GROUP_COLORS = {"Control": "#0072B2", "PD": "#D55E00", "PD_OFF": "#7570B3", "PD_ON": "#009E73"}
ABBA_COLORS = {"high": "#238B45", "low": "#756BB1"}
QUANTITIES = [
    "duration_s",
    "cycles",
    "bursts_per_minute",
    "occupancy_percent",
    "peak_amplitude_uv",
]
REFERENCE_QUANTITIES = ["relative_amplitude_db"]
QUANTITY_LABELS = {
    "duration_s": "Burst duration (s)",
    "cycles": "Burst cycles",
    "bursts_per_minute": "Bursts per minute",
    "occupancy_percent": "Burst occupancy (%)",
    "peak_amplitude_uv": "Peak EEG amplitude (µV)",
}
SELECTED_ABBA_BANDS = ("theta_1", "alpha_1", "beta_1", "beta_2", "gamma_1")


def _fdr_bh(values: pd.Series | np.ndarray) -> np.ndarray:
    p = np.asarray(values, dtype=float)
    q = np.full(p.shape, np.nan)
    valid = np.isfinite(p)
    if not valid.any():
        return q
    pv = p[valid]
    order = np.argsort(pv)
    ranked = pv[order] * len(pv) / np.arange(1, len(pv) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted = np.empty_like(ranked)
    adjusted[order] = np.minimum(ranked, 1.0)
    q[valid] = adjusted
    return q


def _stars(q: float) -> str:
    if not np.isfinite(q) or q >= 0.05:
        return ""
    if q < 0.001:
        return "***"
    if q < 0.01:
        return "**"
    return "*"


def _contiguous_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.pad(np.asarray(mask, dtype=np.int8), (1, 1))
    edges = np.diff(padded)
    starts = np.flatnonzero(edges == 1)
    stops = np.flatnonzero(edges == -1)
    return [(int(start), int(stop)) for start, stop in zip(starts, stops)]


def _global_signal(data: np.ndarray) -> np.ndarray:
    """Robust all-electrode average, retaining one signal per epoch."""
    values = np.asarray(data, dtype=float)
    med = np.nanmedian(values, axis=-1, keepdims=True)
    scale = np.nanmedian(np.abs(values - med), axis=-1, keepdims=True)
    scale = np.where(np.isfinite(scale) & (scale > 1e-12), scale, 1.0)
    standardized = (values - med) / scale
    return np.nanmean(standardized, axis=1)


def _voltage_signal(data: np.ndarray) -> np.ndarray:
    """Return one representative EEG channel in microvolts.

    No per-channel z-scoring or amplitude normalization is applied here. The
    signal is used only for reporting the physical burst voltage and shape;
    burst detection continues to use :func:`_global_signal` above.
    """
    values = np.asarray(data, dtype=float)
    # A spatial mean/median can be almost zero after common-average referencing
    # due to cancellation between electrodes. Select the highest-RMS channel
    # after rejecting broadband-RMS outliers (often residual artifacts). Its
    # signed waveform remains in physical units and is suitable for
    # voltage-scale burst shapes.
    centered = values - np.nanmedian(values, axis=-1, keepdims=True)
    channel_rms = np.sqrt(np.nanmean(centered**2, axis=(0, 2)))
    if not np.isfinite(channel_rms).any():
        return np.full((values.shape[0], values.shape[-1]), np.nan)
    finite_rms = np.isfinite(channel_rms)
    median_rms = float(np.nanmedian(channel_rms[finite_rms]))
    mad_rms = float(np.nanmedian(np.abs(channel_rms[finite_rms] - median_rms)))
    if np.isfinite(mad_rms) and mad_rms > 1e-12:
        valid = finite_rms & (channel_rms <= median_rms + 6.0 * mad_rms)
    else:
        valid = finite_rms & (channel_rms <= np.nanpercentile(channel_rms[finite_rms], 95.0))
    candidate_indices = np.flatnonzero(valid)
    if len(candidate_indices) == 0:
        candidate_indices = np.flatnonzero(finite_rms)
    ordered = candidate_indices[np.argsort(channel_rms[candidate_indices])]
    channel_index = int(ordered[-1])
    return values[:, channel_index, :]


def _detect_bursts(
    signal: np.ndarray,
    sfreq: float,
    low_hz: float,
    high_hz: float,
    *,
    detection_percentile: float,
    boundary_percentile: float,
    minimum_cycles: float,
) -> tuple[list[dict[str, float]], np.ndarray, np.ndarray]:
    """Detect bursts in an epoch-by-time band-passed signal."""
    analytic = hilbert(signal, axis=-1)
    amplitude = np.abs(analytic)
    finite = np.isfinite(amplitude)
    threshold = float(np.nanpercentile(amplitude[finite], detection_percentile)) if finite.any() else np.nan
    boundary = float(np.nanpercentile(amplitude[finite], boundary_percentile)) if finite.any() else np.nan
    if not np.isfinite(threshold) or not np.isfinite(boundary):
        return [], amplitude, np.full(signal.shape[0], np.nan)
    minimum_samples = max(1, int(math.ceil(float(minimum_cycles) * float(sfreq) / math.sqrt(low_hz * high_hz))))
    rows: list[dict[str, float]] = []
    center_frequency = math.sqrt(low_hz * high_hz)
    for epoch_index, values in enumerate(amplitude):
        detected = np.isfinite(values) & (values > threshold)
        for start, stop in _contiguous_runs(detected):
            if stop - start < minimum_samples:
                continue
            peak = start + int(np.nanargmax(values[start:stop]))
            left = peak
            while left > 0 and np.isfinite(values[left - 1]) and values[left - 1] > boundary:
                left -= 1
            right = peak + 1
            while right < len(values) and np.isfinite(values[right]) and values[right] > boundary:
                right += 1
            duration = (right - left) / float(sfreq)
            peak_amplitude = float(values[peak])
            rows.append({
                "epoch_index": float(epoch_index),
                "start_sample": float(left),
                "stop_sample_exclusive": float(right),
                "peak_sample": float(peak),
                "duration_s": duration,
                "cycles": duration * center_frequency,
                "relative_amplitude_db": 10.0 * math.log10(max(peak_amplitude**2 / (threshold**2), np.finfo(float).tiny)),
            })
    return rows, amplitude, np.full(signal.shape[0], threshold)


def _shape_from_burst(
    filtered: np.ndarray,
    epoch_index: int,
    peak_sample: int,
    sfreq: float,
    center_frequency: float,
    *,
    n_points: int = 121,
) -> np.ndarray | None:
    """Extract a trough-aligned waveform on a -3..3-cycle axis in input units."""
    period_samples = max(2, int(round(float(sfreq) / center_frequency)))
    search_start = max(0, peak_sample - period_samples)
    search_stop = min(filtered.shape[-1], peak_sample + period_samples + 1)
    if search_stop <= search_start:
        return None
    trough = search_start + int(np.argmin(filtered[epoch_index, search_start:search_stop]))
    half_window = 3.0 * float(sfreq) / center_frequency
    sample_axis = np.arange(filtered.shape[-1], dtype=float)
    target = trough + np.linspace(-half_window, half_window, n_points)
    if target[0] < 0 or target[-1] >= filtered.shape[-1]:
        return None
    waveform = np.interp(target, sample_axis, filtered[epoch_index])
    if not np.isfinite(waveform).any():
        return None
    return waveform


def _recording_task(task: tuple[Any, ...]) -> dict[str, Any]:
    (
        dataset,
        recording_id,
        participant_id,
        group,
        epoch_path,
        segments,
        filter_order,
        detection_percentile,
        boundary_percentile,
        minimum_cycles,
        max_shape_bursts,
    ) = task
    epochs = mne.read_epochs(epoch_path, preload=True, verbose="ERROR")
    sfreq = float(epochs.info["sfreq"])
    data_uv = epochs.get_data(copy=True) * 1e6
    detection_signal = _global_signal(data_uv)
    voltage_signal = _voltage_signal(data_uv)
    total_samples = detection_signal.shape[0] * detection_signal.shape[1]
    segment_rows: list[dict[str, Any]] = []
    shape_rows: list[dict[str, Any]] = []
    for segment in segments:
        segment_index = int(segment["segment_index"])
        direction = str(segment["direction"])
        band_name = str(segment["band_name"])
        canonical_region = str(segment.get("canonical_region", band_name.rsplit("_", 1)[0]))
        low_hz, high_hz = float(segment["start_hz"]), float(segment["end_hz"])
        if high_hz <= low_hz or low_hz <= 0 or high_hz >= sfreq / 2:
            continue
        sos = butter(int(filter_order), [low_hz, high_hz], btype="bandpass", fs=sfreq, output="sos")
        filtered_detection = sosfiltfilt(sos, detection_signal, axis=-1)
        filtered_voltage = sosfiltfilt(sos, voltage_signal, axis=-1)
        bursts, _, thresholds = _detect_bursts(
            filtered_detection,
            sfreq,
            low_hz,
            high_hz,
            detection_percentile=detection_percentile,
            boundary_percentile=boundary_percentile,
            minimum_cycles=minimum_cycles,
        )
        if not bursts:
            segment_rows.append({"dataset": dataset, "recording_id": recording_id, "participant_id": participant_id, "group": group, "segment_index": segment_index, "band_name": band_name, "canonical_region": canonical_region, "direction": direction, "start_hz": low_hz, "end_hz": high_hz, "burst_count": 0, "duration_s": np.nan, "cycles": np.nan, "bursts_per_minute": 0.0, "occupancy_percent": 0.0, "peak_amplitude_uv": np.nan, "relative_amplitude_db": np.nan})
            continue
        widths_hz = max(high_hz - low_hz, np.finfo(float).eps)
        duration = np.asarray([row["duration_s"] for row in bursts], dtype=float)
        cycles = np.asarray([row["cycles"] for row in bursts], dtype=float)
        amplitudes = np.asarray([row["relative_amplitude_db"] for row in bursts], dtype=float)
        voltage_peak_samples: list[int] = []
        voltage_amplitudes_list: list[float] = []
        for row in bursts:
            epoch_index = int(row["epoch_index"])
            start_sample = int(row["start_sample"])
            stop_sample = min(int(row["stop_sample_exclusive"]), filtered_voltage.shape[-1])
            voltage_window = np.abs(filtered_voltage[epoch_index, start_sample:stop_sample])
            if len(voltage_window) and np.isfinite(voltage_window).any():
                local_peak = int(np.nanargmax(voltage_window))
                voltage_peak_samples.append(start_sample + local_peak)
                voltage_amplitudes_list.append(float(voltage_window[local_peak]))
            else:
                voltage_peak_samples.append(int(row["peak_sample"]))
                voltage_amplitudes_list.append(np.nan)
        voltage_amplitudes = np.asarray(voltage_amplitudes_list, dtype=float)
        occupancy = float(np.sum(duration) / (total_samples / sfreq) * 100.0)
        segment_rows.append({"dataset": dataset, "recording_id": recording_id, "participant_id": participant_id, "group": group, "segment_index": segment_index, "band_name": band_name, "canonical_region": canonical_region, "direction": direction, "start_hz": low_hz, "end_hz": high_hz, "burst_count": len(bursts), "duration_s": float(np.mean(duration)), "cycles": float(np.mean(cycles)), "bursts_per_minute": float(len(bursts) / (total_samples / sfreq) * 60.0 / widths_hz), "occupancy_percent": occupancy, "peak_amplitude_uv": float(np.mean(voltage_amplitudes)), "relative_amplitude_db": float(np.mean(amplitudes))})
        center_frequency = math.sqrt(low_hz * high_hz)
        for burst, voltage_peak_sample in zip(bursts[: int(max_shape_bursts)], voltage_peak_samples[: int(max_shape_bursts)]):
            waveform = _shape_from_burst(filtered_voltage, int(burst["epoch_index"]), int(voltage_peak_sample), sfreq, center_frequency)
            if waveform is None:
                continue
            shape_rows.append({"dataset": dataset, "recording_id": recording_id, "participant_id": participant_id, "group": group, "segment_index": segment_index, "band_name": band_name, "canonical_region": canonical_region, "direction": direction, "start_hz": low_hz, "end_hz": high_hz, "waveform": waveform})
    return {"segment_rows": segment_rows, "shape_rows": shape_rows}


def _participant_aggregate(segment_table: pd.DataFrame) -> pd.DataFrame:
    if segment_table.empty:
        return segment_table
    # Average recordings within each participant and named ABBA interval, so
    # participants—not recordings or segments—are the independent observations
    # in the violins and tests.
    return segment_table.groupby(["dataset", "participant_id", "group", "band_name", "canonical_region", "direction"], as_index=False)[QUANTITIES + REFERENCE_QUANTITIES].mean(numeric_only=True)


def _group_statistics(participant_table: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    melted = participant_table.melt(id_vars=["dataset", "participant_id", "group", "band_name", "canonical_region", "direction"], value_vars=QUANTITIES, var_name="quantity", value_name="value")
    for (dataset, band_name, quantity), frame in melted.groupby(["dataset", "band_name", "quantity"]):
        groups = [group for group in ("Control", "PD", "PD_OFF", "PD_ON") if group in set(frame["group"])]
        for left_index, group_a in enumerate(groups):
            for group_b in groups[left_index + 1 :]:
                left = frame.loc[frame["group"].eq(group_a), "value"].dropna().to_numpy(float)
                right = frame.loc[frame["group"].eq(group_b), "value"].dropna().to_numpy(float)
                if len(left) >= 2 and len(right) >= 2 and np.unique(np.r_[left, right]).size > 1:
                    test = ttest_ind(left, right, equal_var=False, nan_policy="omit")
                    statistic, p_value = float(test.statistic), float(test.pvalue)
                else:
                    statistic, p_value = np.nan, np.nan
                rows.append({"dataset": dataset, "band_name": band_name, "quantity": quantity, "group_a": group_a, "group_b": group_b, "n_a": len(left), "n_b": len(right), "mean_a": np.nanmean(left) if len(left) else np.nan, "mean_b": np.nanmean(right) if len(right) else np.nan, "t_value": statistic, "p_value": p_value})
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["q_fdr_bh"] = np.nan
    for key, indices in result.groupby(["dataset", "quantity"]).groups.items():
        result.loc[indices, "q_fdr_bh"] = _fdr_bh(result.loc[indices, "p_value"])
    result["significant_fdr"] = result["q_fdr_bh"] < 0.05
    return result


def _band_direction_label(table: pd.DataFrame, band_names: list[str]) -> str:
    directions = sorted(set(table.loc[table["band_name"].isin(band_names), "direction"].dropna().astype(str)))
    if not directions:
        return "direction unavailable"
    if len(directions) == 1:
        return f"{directions[0]} LAVI"
    return "high/low LAVI by population"


def _focused_comparison_view(table: pd.DataFrame, band_name: str) -> pd.DataFrame:
    """Return a plotting view with the requested cross-population alignment.

    ABBA's native labels are retained in the metric files. For the focused
    theta comparison only, medication Control's broad low-LAVI ``theta_2`` is
    displayed as ``theta_1`` so it is compared with the low-LAVI ``theta_1``
    interval in PD-OFF and PD-ON. The narrow medication-Control high-LAVI
    theta interval is excluded from this focused comparison rather than mixed
    into it.
    """
    result = table.copy()
    if band_name != "theta_1" or "dataset" not in result or "group" not in result:
        return result
    control = result["dataset"].eq("medication_state") & result["group"].eq("Control")
    result = result.loc[~(control & result["band_name"].eq("theta_1"))].copy()
    result.loc[control.loc[result.index] & result["band_name"].eq("theta_2"), "band_name"] = "theta_1"
    return result


def _save_violins(participant: pd.DataFrame, statistics: pd.DataFrame, output: Path, band_names: list[str] | None = None) -> None:
    datasets = list(dict.fromkeys(participant["dataset"]))
    def band_key(value: str) -> tuple[int, int, str]:
        region, _, number = str(value).rpartition("_")
        order = {"delta": 0, "theta": 1, "alpha": 2, "beta": 3, "gamma": 4}
        return order.get(region, 99), int(number) if number.isdigit() else 0, str(value)
    band_names = sorted((band_names or participant["band_name"].dropna().unique()), key=band_key)
    band_names = [band for band in band_names if band in set(participant["band_name"].dropna())]
    if not band_names:
        return
    fig, axes = plt.subplots(len(datasets), len(QUANTITIES), figsize=(3.2 * len(QUANTITIES), 3.0 * len(datasets)), squeeze=False, sharex=False)
    for row_index, dataset in enumerate(datasets):
        frame = participant.loc[participant["dataset"].eq(dataset)]
        groups = [group for group in ("Control", "PD", "PD_OFF", "PD_ON") if group in set(frame["group"])]
        positions = {(band_name, group): band_index * (len(groups) + 1) + group_index for band_index, band_name in enumerate(band_names) for group_index, group in enumerate(groups)}
        ticks = [np.mean([positions[(band_name, group)] for group in groups]) for band_name in band_names]
        for col_index, quantity in enumerate(QUANTITIES):
            axis = axes[row_index, col_index]
            for band_name in band_names:
                for group in groups:
                    values = frame.loc[frame["band_name"].eq(band_name) & frame["group"].eq(group), quantity].dropna().to_numpy(float)
                    position = positions[(band_name, group)]
                    if len(values) > 1 and np.ptp(values) > 0:
                        violin = axis.violinplot([values], positions=[position], widths=0.8, showmeans=False, showmedians=True, showextrema=False)
                        body = violin["bodies"][0]; body.set_facecolor(GROUP_COLORS[group]); body.set_edgecolor("white"); body.set_linewidth(0.5); body.set_alpha(0.70)
                    if len(values):
                        rng = np.random.default_rng(1000 + row_index * 100 + col_index)
                        axis.scatter(np.full(len(values), position) + rng.uniform(-0.18, 0.18, len(values)), values, s=12, color=GROUP_COLORS[group], edgecolor="white", linewidth=0.3, alpha=0.85, zorder=3)
            axis.set_xticks(ticks, band_names, rotation=35, ha="right", fontsize=7)
            axis.grid(axis="y", alpha=0.2)
            axis.set_title(QUANTITY_LABELS[quantity], fontsize=9, fontweight="bold") if row_index == 0 else None
            if col_index == 0:
                axis.set_ylabel(dataset, fontsize=9, fontweight="bold")
            for band_index, band_name in enumerate(band_names):
                subset = (
                    statistics.loc[(statistics["dataset"].eq(dataset)) & (statistics["band_name"].eq(band_name)) & (statistics["quantity"].eq(quantity))]
                    if not statistics.empty
                    else pd.DataFrame()
                )
                # A band can have multiple pairwise tests (especially in the
                # medication dataset). Show one marker based on the strongest
                # FDR-adjusted comparison; the full pairwise results remain in
                # abba_burst_group_comparisons.csv.
                q_values = subset["q_fdr_bh"].to_numpy(float) if "q_fdr_bh" in subset else np.array([])
                q_values = q_values[np.isfinite(q_values)]
                stars = _stars(float(np.nanmin(q_values))) if len(q_values) else ""
                if stars:
                    values = frame.loc[frame["band_name"].eq(band_name), quantity].to_numpy(float)
                    values = values[np.isfinite(values)]
                    if len(values):
                        ymin, ymax = axis.get_ylim()
                        offset = 0.04 * max(ymax - ymin, np.finfo(float).eps)
                        axis.text(ticks[band_index], float(np.nanmax(values)) + offset, stars, ha="center", va="bottom", fontsize=10, fontweight="bold")
        axis = axes[row_index, 0]
        handles = [Patch(facecolor=GROUP_COLORS[group], label=group.replace("PD_", "PD-") if group.startswith("PD_") else group) for group in groups]
        axis.legend(handles=handles, frameon=False, fontsize=7, loc="upper left")
    if len(band_names) == 1:
        title = f"Temporal burst quantities — {band_names[0]} ({_band_direction_label(participant, band_names)})"
    else:
        title = "Temporal burst quantities in group-specific ABBA bands"
    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.995)
    subtitle = "Points are participant-level means; ABBA intervals are not pooled"
    if len(band_names) > 1:
        subtitle = "Each ABBA interval is retained as band_name_1, band_name_2, …; points are participant-level means"
    fig.text(0.5, 0.955, subtitle, ha="center", fontsize=9, color="#444444")
    fig.tight_layout(rect=(0.02, 0.03, 1.0, 0.90))
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _save_shapes(shape_table: pd.DataFrame, output: Path, confidence_level: float = 0.95, band_names: list[str] | None = None) -> None:
    if shape_table.empty:
        return
    datasets = list(dict.fromkeys(shape_table["dataset"]))
    def band_key(value: str) -> tuple[int, int, str]:
        region, _, number = str(value).rpartition("_")
        order = {"delta": 0, "theta": 1, "alpha": 2, "beta": 3, "gamma": 4}
        return order.get(region, 99), int(number) if number.isdigit() else 0, str(value)
    band_names = sorted((band_names or shape_table["band_name"].dropna().unique()), key=band_key)
    band_names = [band for band in band_names if band in set(shape_table["band_name"].dropna())]
    if not band_names:
        return
    x = np.linspace(-3.0, 3.0, 121)
    fig, axes = plt.subplots(len(datasets), len(band_names), figsize=(3.3 * len(band_names), 3.1 * len(datasets)), squeeze=False, sharex=True, sharey=False)
    for row_index, dataset in enumerate(datasets):
        for col_index, band_name in enumerate(band_names):
            axis = axes[row_index, col_index]
            frame = shape_table.loc[shape_table["dataset"].eq(dataset) & shape_table["band_name"].eq(band_name)]
            if frame.empty:
                axis.axis("off")
                continue
            groups = [group for group in ("Control", "PD", "PD_OFF", "PD_ON") if group in set(frame["group"])]
            for group in groups:
                curves = np.stack(frame.loc[frame["group"].eq(group), "waveform"].to_list()) if len(frame.loc[frame["group"].eq(group)]) else np.empty((0, len(x)))
                if len(curves) == 0:
                    continue
                mean = np.nanmean(curves, axis=0)
                if len(curves) > 1:
                    sem = np.nanstd(curves, axis=0, ddof=1) / np.sqrt(len(curves))
                    ci = float(student_t.ppf(0.5 + confidence_level / 2.0, len(curves) - 1))
                else:
                    ci = 0.0; sem = np.zeros_like(mean)
                color = GROUP_COLORS[group]
                axis.plot(x, mean, color=color, linewidth=1.7, label=f"{group.replace('PD_', 'PD-') if group.startswith('PD_') else group} (n={len(curves)})")
                axis.fill_between(x, mean - ci * sem, mean + ci * sem, color=color, alpha=0.18)
            axis.axvline(0.0, color="0.35", linestyle="--", linewidth=0.8)
            axis.axhline(0.0, color="0.5", linestyle=":", linewidth=0.8)
            axis.set_title(f"{dataset} — {band_name}", fontsize=10, fontweight="bold")
            axis.set_xlabel("Cycles from nearest trough")
            if col_index == 0:
                axis.set_ylabel("Band-passed EEG voltage (µV)")
            axis.grid(alpha=0.18)
            if groups:
                axis.legend(frameon=False, fontsize=7)
    if len(band_names) == 1:
        title = f"Average burst shape — {band_names[0]} ({_band_direction_label(shape_table, band_names)})"
    else:
        title = "Average burst shape in group-specific ABBA bands"
    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.995)
    fig.text(0.5, 0.955, "Curves are participant means; ribbons are 95% Student-t confidence intervals", ha="center", fontsize=9, color="#444444")
    fig.tight_layout(rect=(0.02, 0.03, 1.0, 0.90))
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run(output_root: Path, config_path: Path | None = None, *, workers: int | None = None, max_recordings: int | None = None) -> dict[str, int]:
    config = json.loads(config_path.read_text()) if config_path and config_path.is_file() else {}
    settings = config.get("abba_bursts", {})
    # Use the opaque group-mean limits shown in abba_band_limits_by_group.png.
    # Subject-specific limits remain available for the separate limits figure,
    # but mixing them would make delta_1/delta_2 incomparable across subjects.
    segments_path = output_root / "statistics" / "abba_group_mean_segments.csv"
    if not segments_path.is_file():
        raise FileNotFoundError(f"Missing ABBA group-mean segments: {segments_path}")
    segments = pd.read_csv(segments_path)
    canonical_root = Path(config.get("global_output_root", "outputs/global")) / "canonical"
    tasks: list[tuple[Any, ...]] = []
    for manifest_path in sorted(canonical_root.glob("*/recordings.csv.gz")):
        dataset = manifest_path.parent.name
        recordings = pd.read_csv(manifest_path, low_memory=False)
        for row in recordings.to_dict("records"):
            epoch_path = str(row.get("epoch_path", ""))
            if not epoch_path or not Path(epoch_path).is_file():
                continue
            subset = segments.loc[(segments["dataset"].eq(dataset)) & (segments["group"].astype(str).eq(str(row["group"]))) & (segments["end_hz"] > segments["start_hz"])].to_dict("records")
            if not subset:
                continue
            tasks.append((dataset, str(row["recording_id"]), str(row["participant_id"]), str(row["group"]), epoch_path, subset, int(settings.get("filter_order", 4)), float(settings.get("detection_percentile", 90.0)), float(settings.get("boundary_percentile", 75.0)), float(settings.get("minimum_cycles", 2.0)), int(settings.get("max_shape_bursts_per_segment", 100))))
    if max_recordings is not None:
        tasks = tasks[: int(max_recordings)]
    if not tasks:
        raise RuntimeError("No recordings matched ABBA group-mean segments")
    worker_count = int(workers if workers is not None else settings.get("workers", 4))
    results: list[dict[str, Any]] = []
    if worker_count == 1:
        for task in tqdm(tasks, desc="ABBA temporal bursts"):
            results.append(_recording_task(task))
    else:
        try:
            with ProcessPoolExecutor(max_workers=worker_count) as executor:
                futures = [executor.submit(_recording_task, task) for task in tasks]
                for future in tqdm(as_completed(futures), total=len(futures), desc="ABBA temporal bursts"):
                    results.append(future.result())
        except PermissionError:
            # macOS can deny semaphore creation in managed environments. The
            # thread fallback keeps the same progress reporting and results.
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                futures = [executor.submit(_recording_task, task) for task in tasks]
                for future in tqdm(as_completed(futures), total=len(futures), desc="ABBA temporal bursts (threads)"):
                    results.append(future.result())
    segment_table = pd.DataFrame([row for result in results for row in result["segment_rows"]])
    participant_table = _participant_aggregate(segment_table)
    statistics = _group_statistics(participant_table)
    shape_rows = [row for result in results for row in result["shape_rows"]]
    shape_table = pd.DataFrame(shape_rows)
    if not shape_table.empty:
        shape_table = shape_table.groupby(["dataset", "participant_id", "group", "band_name", "canonical_region", "direction"], as_index=False)["waveform"].agg(lambda values: np.nanmean(np.stack(values), axis=0))
    metrics_root = output_root / "metrics"; statistics_root = output_root / "statistics"; figures_root = output_root / "figures"
    metrics_root.mkdir(parents=True, exist_ok=True); statistics_root.mkdir(parents=True, exist_ok=True); figures_root.mkdir(parents=True, exist_ok=True)
    segment_table.to_csv(metrics_root / "abba_burst_segment_metrics.csv.gz", index=False)
    participant_table.to_csv(metrics_root / "abba_burst_participant_metrics.csv.gz", index=False)
    statistics.to_csv(statistics_root / "abba_burst_group_comparisons.csv", index=False)
    shape_table.to_pickle(metrics_root / "abba_burst_shapes_participant.pkl")
    if not shape_table.empty:
        np.savez_compressed(
            metrics_root / "abba_burst_shapes_participant.npz",
            dataset=shape_table["dataset"].to_numpy(dtype=str),
            participant_id=shape_table["participant_id"].to_numpy(dtype=str),
            group=shape_table["group"].to_numpy(dtype=str),
            band_name=shape_table["band_name"].to_numpy(dtype=str),
            canonical_region=shape_table["canonical_region"].to_numpy(dtype=str),
            direction=shape_table["direction"].to_numpy(dtype=str),
            cycles=np.linspace(-3.0, 3.0, 121),
            waveforms=np.stack(shape_table["waveform"].to_numpy()),
        )
    _save_violins(participant_table, statistics, figures_root / "abba_burst_quantity_violins.png")
    _save_shapes(shape_table, figures_root / "abba_burst_shapes.png", float(settings.get("confidence_level", 0.95)))
    # Also provide focused, publication-friendly figures for the intervals
    # requested for direct comparison across datasets and populations.
    focused_statistics: list[pd.DataFrame] = []
    for band_name in SELECTED_ABBA_BANDS:
        focused_participant = _focused_comparison_view(participant_table, band_name)
        focused_shape = _focused_comparison_view(shape_table, band_name)
        focused_test = _group_statistics(focused_participant)
        if not focused_test.empty:
            focused_statistics.append(focused_test.loc[focused_test["band_name"].eq(band_name)].copy())
        _save_violins(focused_participant, focused_test, figures_root / f"abba_burst_quantity_violins_{band_name}.png", [band_name])
        _save_shapes(focused_shape, figures_root / f"abba_burst_shapes_{band_name}.png", float(settings.get("confidence_level", 0.95)), [band_name])
    if focused_statistics:
        pd.concat(focused_statistics, ignore_index=True).to_csv(statistics_root / "abba_burst_focused_group_comparisons.csv", index=False)
    return {"recordings": len(tasks), "segment_rows": len(segment_table), "participant_rows": len(participant_table), "shape_participants": len(shape_table)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/rhythmicity"))
    parser.add_argument("--config", type=Path, default=Path("config/analyses/rhythmicity.json"))
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--max-recordings", type=int, default=None, help="Process only the first recordings for a quick smoke test")
    args = parser.parse_args()
    print(json.dumps(run(args.output_root, args.config, workers=args.workers, max_recordings=args.max_recordings), indent=2))


if __name__ == "__main__":
    main()
