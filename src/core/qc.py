"""Non-interactive QC figures for every scientifically important stage."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable

from .runtime import configure_runtime

configure_runtime()

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from scipy.signal import welch
from .plotting import save_figure as _save


def save_status(qc_dir: Path, stem: str, message: str) -> Path:
    """Record a stage that did not occur without manufacturing a fake plot."""
    path = qc_dir / f"{stem}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(message.rstrip() + "\n", encoding="utf-8")
    return path


def select_channels(raw, preferred: Iterable[str], minimum: int = 5) -> list[str]:
    preferred = list(preferred)
    channels = [name for name in preferred if name in raw.ch_names]
    eeg = [raw.ch_names[index] for index in mne.pick_types(raw.info, eeg=True, exclude=[])]
    if len(channels) < minimum and eeg:
        positions = np.linspace(0, len(eeg) - 1, min(minimum, len(eeg)), dtype=int)
        for index in positions:
            if eeg[index] not in channels:
                channels.append(eeg[index])
    return channels[: max(minimum, len(preferred))]


def _time_slice(raw, start_sec: float, duration_sec: float) -> tuple[int, int, np.ndarray]:
    max_start = max(0.0, raw.times[-1] - duration_sec)
    start = min(max(0.0, start_sec), max_start)
    first = int(round(start * raw.info["sfreq"]))
    last = min(raw.n_times, first + int(round(duration_sec * raw.info["sfreq"])))
    return first, last, raw.times[first:last]


def _trace_data(raw, channels: list[str], start_sec: float, duration_sec: float):
    first, last, times = _time_slice(raw, start_sec, duration_sec)
    return times, raw.get_data(picks=channels, start=first, stop=last) * 1e6


def _plot_trace_rows(axes, times, data_uv, channels, color="black", label=None, ylim=None):
    axes = np.atleast_1d(axes)
    for index, (axis, channel) in enumerate(zip(axes, channels)):
        axis.plot(times, data_uv[index], color=color, linewidth=0.65, label=label)
        axis.set_ylabel(f"{channel}\nµV")
        if ylim is not None:
            axis.set_ylim(ylim)
        axis.grid(alpha=0.15)
    axes[-1].set_xlabel("Time (s)")


def plot_signal(raw, channels: list[str], start_sec: float, duration_sec: float, title: str, path: Path, dpi: int) -> None:
    times, data = _trace_data(raw, channels, start_sec, duration_sec)
    fig, axes = plt.subplots(len(channels), 1, figsize=(13, 1.7 * len(channels)), sharex=True)
    _plot_trace_rows(axes, times, data, channels)
    fig.suptitle(title)
    fig.tight_layout()
    _save(fig, path, dpi)


def _median_psd(raw, channels: list[str], fmin: float, fmax: float):
    data = raw.get_data(picks=channels)
    sfreq = float(raw.info["sfreq"])
    nperseg = min(data.shape[1], int(round(4.0 * sfreq)))
    frequencies, power = welch(data, fs=sfreq, nperseg=nperseg, noverlap=nperseg // 2, axis=1)
    mask = (frequencies >= fmin) & (frequencies <= min(fmax, sfreq / 2.0))
    # V²/Hz -> µV²/Hz before converting to dB.
    db = 10.0 * np.log10(np.maximum(power[:, mask] * 1e12, np.finfo(float).tiny))
    return frequencies[mask], db


def plot_psd(raw, channels: list[str], fmin: float, fmax: float, title: str, path: Path, dpi: int) -> None:
    frequencies, db = _median_psd(raw, channels, fmin, fmax)
    fig, axis = plt.subplots(figsize=(10, 5))
    for channel, values in zip(channels, db):
        axis.plot(frequencies, values, alpha=0.35, linewidth=0.8, label=channel)
    axis.plot(frequencies, np.median(db, axis=0), color="black", linewidth=2.0, label="channel median")
    axis.set(title=title, xlabel="Frequency (Hz)", ylabel="Power (dB µV²/Hz)", xlim=(fmin, fmax))
    axis.grid(alpha=0.2)
    axis.legend(ncol=2, fontsize=8)
    _save(fig, path, dpi)


def plot_signal_comparison(before, after, channels: list[str], start_sec: float, duration_sec: float, before_label: str, after_label: str, title: str, path: Path, dpi: int) -> None:
    first_times, first_data = _trace_data(before, channels, start_sec, duration_sec)
    second_times, second_data = _trace_data(after, channels, start_sec, duration_sec)
    joint = np.concatenate([first_data.ravel(), second_data.ravel()])
    limit = max(1.0, float(np.nanpercentile(np.abs(joint), 99.5)))
    fig, axes = plt.subplots(len(channels), 1, figsize=(13, 1.8 * len(channels)), sharex=True)
    axes = np.atleast_1d(axes)
    for index, (axis, channel) in enumerate(zip(axes, channels)):
        axis.plot(first_times, first_data[index], color="#777777", linewidth=0.65, label=before_label)
        axis.plot(second_times, second_data[index], color="#0072B2", linewidth=0.65, label=after_label)
        axis.set_ylabel(f"{channel}\nµV")
        axis.set_ylim(-limit, limit)
        axis.grid(alpha=0.15)
    axes[0].legend(loc="upper right", ncol=2)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(title + " — identical channels, interval, and scale")
    fig.tight_layout()
    _save(fig, path, dpi)


def plot_psd_comparison(before, after, channels: list[str], fmin: float, fmax: float, before_label: str, after_label: str, title: str, path: Path, dpi: int) -> None:
    frequencies, before_db = _median_psd(before, channels, fmin, fmax)
    after_frequencies, after_db = _median_psd(after, channels, fmin, fmax)
    if not np.allclose(frequencies, after_frequencies):
        raise ValueError("PSD comparison frequency grids do not match")
    fig, axis = plt.subplots(figsize=(10, 5))
    axis.plot(frequencies, np.median(before_db, axis=0), color="#777777", linewidth=2, label=before_label)
    axis.plot(frequencies, np.median(after_db, axis=0), color="#0072B2", linewidth=2, label=after_label)
    axis.set(title=title, xlabel="Frequency (Hz)", ylabel="Median power (dB µV²/Hz)", xlim=(fmin, fmax))
    axis.grid(alpha=0.2)
    axis.legend()
    _save(fig, path, dpi)


def plot_bad_channels(raw, result, start_sec: float, duration_sec: float, path: Path, dpi: int) -> bool:
    if not result.bad_channels:
        return False
    times, traces = _trace_data(raw, result.bad_channels, start_sec, duration_sec)
    metrics = result.metrics.set_index("channel").loc[result.bad_channels]
    fig, axes = plt.subplots(len(result.bad_channels), 3, figsize=(15, 3.0 * len(result.bad_channels)))
    axes = np.atleast_2d(axes)
    for row, channel in enumerate(result.bad_channels):
        axes[row, 0].plot(times, traces[row], linewidth=0.7, color="#D55E00")
        axes[row, 0].set(title=f"{channel}: BAD RECORDED CHANNEL", xlabel="Time (s)", ylabel="µV")
        axes[row, 1].bar(["SD", "peak-to-peak"], [metrics.loc[channel, "std_uv"], metrics.loc[channel, "peak_to_peak_uv"]], color="#D55E00")
        axes[row, 1].set_ylabel("µV")
        axes[row, 2].axis("off")
        axes[row, 2].text(0, 0.8, "Reasons:\n" + "\n".join(result.reasons[channel]), va="top")
    fig.suptitle("Confirmed bad recorded channels before interpolation")
    fig.tight_layout()
    _save(fig, path, dpi)
    return True


def plot_artifact_annotations(raw, records: pd.DataFrame, channels: list[str], path: Path, dpi: int) -> bool:
    if records.empty:
        return False
    first = records.iloc[0]
    start = max(0.0, float(first["onset_sec"]) - 2.0)
    duration = min(12.0, raw.times[-1] - start)
    times, data = _trace_data(raw, channels, start, duration)
    fig, axes = plt.subplots(len(channels), 1, figsize=(13, 1.8 * len(channels)), sharex=True)
    axes = np.atleast_1d(axes)
    _plot_trace_rows(axes, times, data, channels)
    window_end = start + duration
    for _, record in records.iterrows():
        onset = float(record["onset_sec"])
        end = onset + float(record["duration_sec"])
        if end < start or onset > window_end:
            continue
        for axis in axes:
            axis.axvspan(onset, end, color="#D55E00", alpha=0.25)
        axes[0].text(max(onset, start), axes[0].get_ylim()[1], str(record["description"]), va="top", fontsize=8)
    fig.suptitle("Large temporal artifacts preserved as BAD annotations")
    fig.tight_layout()
    _save(fig, path, dpi)
    return True


def _ica_info(raw, ica):
    picks = [raw.ch_names.index(name) for name in ica.ch_names]
    info = mne.pick_info(raw.info, picks, copy=True)
    info["bads"] = []
    return info


def _plot_topomap(axis, values, info, title: str):
    try:
        mne.viz.plot_topomap(
            values,
            info,
            axes=axis,
            show=False,
            contours=0,
            sensors=True,
            cmap="viridis",
        )
    except Exception as error:  # keep QC informative if a montage is incomplete
        axis.bar(np.arange(len(values)), values, width=1.0)
        axis.set_xticks([])
        axis.text(0.02, 0.02, f"Topomap unavailable: {error}", transform=axis.transAxes, fontsize=6)
    axis.set_title(title, fontsize=9)


def _ica_order(scores: pd.DataFrame | None, n_components: int) -> list[int]:
    if scores is None:
        return list(range(n_components))
    return scores["component"].astype(int).tolist()


def _ica_score_lookup(scores: pd.DataFrame | None) -> dict[int, dict[str, Any]]:
    if scores is None:
        return {}
    return {
        int(row["component"]): row
        for row in scores.to_dict(orient="records")
    }


def _ica_ranked_label(component: int, lookup: dict[int, dict[str, Any]]) -> str:
    row = lookup.get(component)
    if row is None or "iclabel_predicted_label" not in row:
        return f"IC{component:03d}"
    candidate = " | CANDIDATE" if bool(row["proposed_exclusion"]) else ""
    return (
        f"#{int(row['artifact_rank']):02d} IC{component:03d}{candidate}\n"
        f"{row['iclabel_predicted_label']} {float(row['iclabel_predicted_probability']):.2f} | "
        f"artifact {float(row['iclabel_artifact_probability']):.2f}"
    )


def plot_ica_probabilities(scores: pd.DataFrame, path: Path, dpi: int) -> None:
    """Plot ICLabel class probabilities in artifact-to-brain review order."""
    probability_columns = [
        ("iclabel_eye_blink_probability", "eye blink", "#D55E00"),
        ("iclabel_muscle_artifact_probability", "muscle", "#CC79A7"),
        ("iclabel_heart_beat_probability", "heart", "#E69F00"),
        ("iclabel_line_noise_probability", "line", "#F0E442"),
        ("iclabel_channel_noise_probability", "channel", "#999999"),
        ("iclabel_other_probability", "other", "#56B4E9"),
        ("iclabel_brain_probability", "brain", "#009E73"),
    ]
    positions = np.arange(len(scores))
    fig, axis = plt.subplots(figsize=(12, max(7, 0.38 * len(scores))))
    left = np.zeros(len(scores), dtype=float)
    for column, label, color in probability_columns:
        values = scores[column].to_numpy(dtype=float)
        axis.barh(positions, values, left=left, color=color, label=label, height=0.78)
        left += values
    candidate = scores["proposed_exclusion"].to_numpy(dtype=bool)
    labels = [
        f"#{int(rank):02d} IC{int(component):03d}{'  *' if selected else ''}"
        for rank, component, selected in zip(
            scores["artifact_rank"], scores["component"], candidate
        )
    ]
    axis.set_yticks(positions, labels)
    axis.invert_yaxis()
    axis.set(
        xlabel="ICLabel class probability",
        xlim=(0, 1),
        title="ICA components ranked from most likely known artifact to most brain-like",
    )
    axis.grid(axis="x", alpha=0.2)
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=7, fontsize=8)
    axis.text(
        0,
        -0.035,
        "* = machine-proposed exclusion; every proposal requires visual confirmation",
        transform=axis.transAxes,
        fontsize=8,
    )
    fig.tight_layout()
    _save(fig, path, dpi)


def plot_ica_components(
    ica,
    raw,
    qc_dir: Path,
    dpi: int,
    per_page: int = 12,
    scores: pd.DataFrame | None = None,
) -> list[Path]:
    components = ica.get_components()
    info = _ica_info(raw, ica)
    order = _ica_order(scores, ica.n_components_)
    lookup = _ica_score_lookup(scores)
    paths = []
    for page_start in range(0, len(order), per_page):
        indices = order[page_start : page_start + per_page]
        ncols = 4
        nrows = math.ceil(len(indices) / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3 * nrows))
        axes = np.atleast_1d(axes).ravel()
        for axis, component in zip(axes, indices):
            _plot_topomap(
                axis,
                components[:, component],
                info,
                _ica_ranked_label(component, lookup),
            )
        for axis in axes[len(indices) :]:
            axis.axis("off")
        fig.suptitle("ICA component topographies — artifact-ranked; inspect candidates visually")
        fig.tight_layout()
        path = qc_dir / f"08_ica_components_ranked_p{page_start // per_page + 1:02d}.png"
        _save(fig, path, dpi)
        paths.append(path)
    return paths


def plot_ica_sources(
    ica,
    raw,
    qc_dir: Path,
    start_sec: float,
    duration_sec: float,
    dpi: int,
    per_page: int = 12,
    scores: pd.DataFrame | None = None,
) -> list[Path]:
    first, last, times = _time_slice(raw, start_sec, duration_sec)
    sources = ica.get_sources(raw).get_data(start=first, stop=last)
    order = _ica_order(scores, ica.n_components_)
    lookup = _ica_score_lookup(scores)
    paths = []
    for page_start in range(0, len(order), per_page):
        indices = order[page_start : page_start + per_page]
        fig, axes = plt.subplots(len(indices), 1, figsize=(13, 1.2 * len(indices)), sharex=True)
        axes = np.atleast_1d(axes)
        for axis, component in zip(axes, indices):
            values = sources[component]
            scale = np.nanpercentile(np.abs(values), 99.5) or 1.0
            axis.plot(times, values / scale, color="black", linewidth=0.6)
            row = lookup.get(component)
            marker = " *" if row and bool(row["proposed_exclusion"]) else ""
            rank = f"#{int(row['artifact_rank']):02d} " if row else ""
            axis.set_ylabel(f"{rank}IC{component:03d}{marker}", fontsize=8)
            axis.set_yticks([])
        axes[-1].set_xlabel("Time (s); each source scaled only for display")
        fig.suptitle("ICA source time courses — artifact-ranked; * = proposed exclusion")
        fig.tight_layout()
        path = qc_dir / f"09_ica_sources_ranked_p{page_start // per_page + 1:02d}.png"
        _save(fig, path, dpi)
        paths.append(path)
    return paths


def _component_property_figure(
    ica,
    raw,
    components: list[int],
    reasons: dict[int, str] | None = None,
    scores: pd.DataFrame | None = None,
):
    info = _ica_info(raw, ica)
    lookup = _ica_score_lookup(scores)
    maps = ica.get_components()
    sources = ica.get_sources(raw).get_data()
    sfreq = float(raw.info["sfreq"])
    n_show = min(sources.shape[1], int(round(20.0 * sfreq)))
    times = np.arange(n_show) / sfreq
    fig, axes = plt.subplots(len(components), 3, figsize=(15, 3.0 * len(components)))
    axes = np.atleast_2d(axes)
    for row, component in enumerate(components):
        reason = f" — {reasons[component]}" if reasons and component in reasons else ""
        title = _ica_ranked_label(component, lookup) if lookup else f"IC{component:03d}"
        _plot_topomap(axes[row, 0], maps[:, component], info, f"{title}{reason}")
        axes[row, 1].plot(times, sources[component, :n_show], linewidth=0.6)
        axes[row, 1].set(xlabel="Time (s)", ylabel="ICA units", title="Time course")
        frequency, power = welch(sources[component], fs=sfreq, nperseg=min(sources.shape[1], int(4 * sfreq)))
        mask = (frequency >= 1) & (frequency <= 50)
        axes[row, 2].plot(frequency[mask], 10 * np.log10(np.maximum(power[mask], np.finfo(float).tiny)), linewidth=0.8)
        axes[row, 2].set(xlabel="Frequency (Hz)", ylabel="Power (dB)", title="PSD")
    return fig


def plot_ica_properties(
    ica,
    raw,
    qc_dir: Path,
    dpi: int,
    per_page: int = 6,
    scores: pd.DataFrame | None = None,
) -> list[Path]:
    order = _ica_order(scores, ica.n_components_)
    paths = []
    for start in range(0, len(order), per_page):
        components = order[start : start + per_page]
        fig = _component_property_figure(ica, raw, components, scores=scores)
        fig.suptitle("ICA component properties — artifact-ranked: topography, time course, and PSD")
        fig.tight_layout()
        path = qc_dir / f"10_ica_properties_ranked_p{start // per_page + 1:02d}.png"
        _save(fig, path, dpi)
        paths.append(path)
    return paths


def plot_removed_ica_components(ica, raw, components: list[int], reasons: dict[int, str], path: Path, dpi: int) -> bool:
    if not components:
        return False
    fig = _component_property_figure(ica, raw, components, reasons)
    fig.suptitle("ICA components selected for removal")
    fig.tight_layout()
    _save(fig, path, dpi)
    return True


def plot_ica_removed_signal(before, after, channels: list[str], start_sec: float, duration_sec: float, path: Path, dpi: int) -> None:
    times, before_data = _trace_data(before, channels, start_sec, duration_sec)
    _, after_data = _trace_data(after, channels, start_sec, duration_sec)
    difference = before_data - after_data
    limit = max(1.0, float(np.percentile(np.abs(np.concatenate([before_data.ravel(), after_data.ravel(), difference.ravel()])), 99.5)))
    fig, axes = plt.subplots(len(channels), 3, figsize=(16, 2.4 * len(channels)), sharex=True, sharey=True)
    axes = np.atleast_2d(axes)
    for row, channel in enumerate(channels):
        for column, (values, title, color) in enumerate(((before_data[row], "Before ICA", "#777777"), (after_data[row], "After ICA", "#0072B2"), (difference[row], "Difference removed", "#D55E00"))):
            axes[row, column].plot(times, values, color=color, linewidth=0.6)
            axes[row, column].set_ylim(-limit, limit)
            axes[row, column].set_title(title if row == 0 else "")
            axes[row, column].set_ylabel(f"{channel} (µV)" if column == 0 else "")
    for axis in axes[-1]:
        axis.set_xlabel("Time (s)")
    fig.suptitle("Signal contribution removed by reviewed ICA components")
    fig.tight_layout()
    _save(fig, path, dpi)


def _nearest_neighbors(raw, channel: str, count: int = 3) -> list[str]:
    montage = raw.get_montage()
    if montage is None:
        return []
    positions = montage.get_positions()["ch_pos"]
    if channel not in positions:
        return []
    candidates = []
    for name in raw.ch_names:
        if name == channel or name not in positions:
            continue
        candidates.append((float(np.linalg.norm(positions[name] - positions[channel])), name))
    return [name for _, name in sorted(candidates)[:count]]


def plot_interpolation(before, after, bad_channels: list[str], start_sec: float, duration_sec: float, qc_dir: Path, dpi: int) -> list[Path]:
    paths = []
    for channel in bad_channels:
        neighbors = _nearest_neighbors(after, channel)
        channels = [channel] + neighbors
        times, before_data = _trace_data(before, channels, start_sec, duration_sec)
        _, after_data = _trace_data(after, channels, start_sec, duration_sec)
        fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
        axes[0].plot(times, before_data[0], color="#D55E00", linewidth=0.8, label=f"{channel} before")
        axes[0].plot(times, after_data[0], color="#0072B2", linewidth=0.8, label=f"{channel} interpolated")
        axes[0].set_ylabel("µV")
        axes[0].legend()
        for index, neighbor in enumerate(neighbors, start=1):
            axes[1].plot(times, after_data[index], linewidth=0.6, label=neighbor)
        axes[1].set(xlabel="Time (s)", ylabel="µV", title="Nearest recorded neighbors")
        axes[1].legend(ncol=3)
        fig.suptitle(f"Interpolation QC — {channel} was recorded and confirmed bad")
        fig.tight_layout()
        safe_channel = channel.replace("/", "-")
        path = qc_dir / f"14_interpolation_{safe_channel}.png"
        _save(fig, path, dpi)
        paths.append(path)
    return paths


def plot_raw_vs_clean_intervals(raw, cleaned, channels: list[str], clean_start: float, artifact_records: pd.DataFrame, duration_sec: float, path: Path, dpi: int) -> None:
    starts = [("relatively clean interval", clean_start)]
    if not artifact_records.empty:
        artifact_start = max(0.0, float(artifact_records.iloc[0]["onset_sec"]) - 1.0)
        starts.append(("annotated artifact interval", artifact_start))
    fig, axes = plt.subplots(len(channels), len(starts), figsize=(7 * len(starts), 1.8 * len(channels)), squeeze=False)
    for column, (interval_label, start) in enumerate(starts):
        raw_times, raw_data = _trace_data(raw, channels, start, duration_sec)
        clean_times, clean_data = _trace_data(cleaned, channels, start, duration_sec)
        limit = max(1.0, float(np.percentile(np.abs(np.concatenate([raw_data.ravel(), clean_data.ravel()])), 99.5)))
        for row, channel in enumerate(channels):
            axes[row, column].plot(raw_times, raw_data[row], color="#777777", linewidth=0.55, label="raw")
            axes[row, column].plot(clean_times, clean_data[row], color="#0072B2", linewidth=0.55, label="final cleaned")
            axes[row, column].set_ylim(-limit, limit)
            axes[row, column].set_ylabel(f"{channel}\nµV")
            if row == 0:
                axes[row, column].set_title(interval_label)
            if row == len(channels) - 1:
                axes[row, column].set_xlabel("Time (s)")
    axes[0, 0].legend(ncol=2)
    fig.suptitle("RAW EEG vs FINAL CLEANED EEG — identical channels and within-panel scale")
    fig.tight_layout()
    _save(fig, path, dpi)


def plot_epoch_rejection(all_epochs, table: pd.DataFrame, channels: list[str], path: Path, dpi: int) -> None:
    accepted = table.index[table["accepted"]].tolist()
    rejected = table.index[~table["accepted"]].tolist()
    examples = []
    if accepted:
        examples.append(("Accepted epoch", accepted[0], "#0072B2"))
    seen_reasons = set()
    for epoch_index in rejected:
        reason = table.loc[epoch_index, "reasons"]
        if reason in seen_reasons:
            continue
        detail = table.loc[epoch_index, "amplitude_reason"]
        title = f"Rejected epoch — {reason}"
        if detail:
            title += f"\n{detail}"
        examples.append((title, epoch_index, "#D55E00"))
        seen_reasons.add(reason)
        if len(examples) >= 3:
            break
    if not examples:
        raise ValueError("No epochs available for epoch QC")
    fig, axes = plt.subplots(len(channels), len(examples), figsize=(7 * len(examples), 1.8 * len(channels)), squeeze=False)
    epoch_times = all_epochs.times
    for column, (label, epoch_index, color) in enumerate(examples):
        data = all_epochs[epoch_index].get_data(picks=channels)[0] * 1e6
        limit = max(1.0, float(np.percentile(np.abs(data), 99.5)))
        for row, channel in enumerate(channels):
            axes[row, column].plot(epoch_times, data[row], color=color, linewidth=0.65)
            axes[row, column].set_ylim(-limit, limit)
            axes[row, column].set_ylabel(f"{channel}\nµV")
            if row == 0:
                axes[row, column].set_title(label)
            if row == len(channels) - 1:
                axes[row, column].set_xlabel("Epoch time (s)")
    fig.suptitle("Epoch rejection examples and recorded reasons")
    fig.tight_layout()
    _save(fig, path, dpi)


def plot_epoch_psd(epochs, channels: list[str], fmin: float, fmax: float, path: Path, dpi: int) -> None:
    data = epochs.get_data(picks=channels)
    sfreq = float(epochs.info["sfreq"])
    nperseg = data.shape[-1]
    frequencies, power = welch(data, fs=sfreq, nperseg=nperseg, axis=-1)
    mask = (frequencies >= fmin) & (frequencies <= fmax)
    db = 10 * np.log10(np.maximum(power[:, :, mask] * 1e12, np.finfo(float).tiny))
    fig, axis = plt.subplots(figsize=(10, 5))
    axis.plot(frequencies[mask], np.median(db, axis=(0, 1)), color="#0072B2", linewidth=2)
    axis.set(title="Final accepted-epoch PSD (no baseline, no normalization)", xlabel="Frequency (Hz)", ylabel="Median power (dB µV²/Hz)", xlim=(fmin, fmax))
    axis.grid(alpha=0.2)
    _save(fig, path, dpi)


def plot_summary(summary: dict[str, Any], path: Path, dpi: int) -> None:
    lines = [f"{key}: {value}" for key, value in summary.items()]
    fig, axis = plt.subplots(figsize=(11, max(7, 0.32 * len(lines))))
    axis.axis("off")
    axis.text(0.01, 0.99, "PREPROCESSING SUMMARY\n\n" + "\n".join(lines), va="top", family="monospace", fontsize=9)
    _save(fig, path, dpi)
