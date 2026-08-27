"""Transparent diagnostic and summary plots for bout ordinal analysis."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from src.runtime import configure_runtime

configure_runtime()

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from src.plotting import save_figure as _save

from .metrics import METRICS, ordinal_patterns


METRIC_LABELS = {
    "entropy": "Permutation entropy (H)",
    "complexity": "Statistical complexity (C)",
    "fisher_information": "Fisher information (F)",
}
METRIC_CMAPS = {
    "entropy": "viridis",
    "complexity": "viridis",
    "fisher_information": "viridis",
}


def _finite_limits(values: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return 0.0, 1.0
    low, high = float(finite.min()), float(finite.max())
    padding = max((high - low) * 0.05, 1e-6)
    return max(0.0, low - padding), high + padding


def _draw_distribution(
    axis: Any,
    values: np.ndarray,
    position: float,
    color: str,
    *,
    width: float = 0.65,
) -> None:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if len(finite) >= 2 and not np.allclose(finite, finite[0]):
        parts = axis.violinplot(
            [finite],
            positions=[position],
            widths=width,
            showmedians=True,
            showextrema=False,
        )
        for body in parts["bodies"]:
            body.set_facecolor(color)
            body.set_edgecolor(color)
            body.set_alpha(0.45)
        parts["cmedians"].set_color("black")
    elif len(finite):
        axis.scatter(np.full(len(finite), position), finite, color=color, s=18)
    if len(finite):
        jitter = np.linspace(-0.09, 0.09, len(finite))
        axis.scatter(
            np.full(len(finite), position) + jitter,
            np.sort(finite),
            color=color,
            s=8,
            alpha=0.30,
        )


def plot_detection_example(example: dict[str, Any], path: Path, dpi: int) -> None:
    """Show spectral background, thresholds, time-frequency power, and the bout mask."""
    sfreq = float(example["sfreq"])
    signal = np.asarray(example["signal_uv"], dtype=float)
    times = np.arange(len(signal)) / sfreq
    frequencies = np.asarray(example["wavelet_frequencies_hz"], dtype=float)
    power = np.asarray(example["wavelet_power"], dtype=float)
    thresholds = np.asarray(example["thresholds"], dtype=float)
    background = np.asarray(example["background"], dtype=float)
    detected = np.asarray(example["detected"], dtype=bool)
    band_mask = np.asarray(example["band_mask"], dtype=bool)

    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    axes[0, 0].loglog(
        example["frequencies_hz"],
        example["observed_psd_uv2_hz"],
        color="black",
        linewidth=1,
        label="Observed PSD",
    )
    axes[0, 0].loglog(
        example["frequencies_hz"],
        example["modeled_psd_uv2_hz"],
        color="#0072B2",
        linewidth=2,
        label="Full specparam model",
    )
    axes[0, 0].loglog(
        example["frequencies_hz"],
        example["aperiodic_psd_uv2_hz"],
        color="#D55E00",
        linewidth=2,
        label="Aperiodic background",
    )
    if "fixed_aperiodic_psd_uv2_hz" in example:
        axes[0, 0].loglog(
            example["frequencies_hz"],
            example["fixed_aperiodic_psd_uv2_hz"],
            color="#666666",
            linestyle="--",
            linewidth=1.0,
            label="Fixed candidate",
        )
    if "knee_aperiodic_psd_uv2_hz" in example and np.isfinite(
        example["knee_aperiodic_psd_uv2_hz"]
    ).all():
        axes[0, 0].loglog(
            example["frequencies_hz"],
            example["knee_aperiodic_psd_uv2_hz"],
            color="#CC79A7",
            linestyle=":",
            linewidth=1.1,
            label="Knee candidate",
        )
    axes[0, 0].set(
        xlabel="Frequency (Hz)", ylabel="PSD (µV²/Hz)", title="1. Specparam decomposition"
    )
    axes[0, 0].legend(frameon=False)
    axes[0, 0].grid(alpha=0.2)

    axes[0, 1].plot(frequencies, example["mean_wavelet_power"], color="black", label="Mean wavelet power")
    axes[0, 1].plot(frequencies, background, color="#0072B2", label="Mapped aperiodic background")
    axes[0, 1].plot(frequencies, thresholds, color="#D55E00", label="95% power threshold")
    axes[0, 1].set(
        xlabel="Frequency (Hz)", ylabel="Wavelet power (a.u.)", title="2. Aperiodic-relative threshold"
    )
    axes[0, 1].legend(frameon=False)
    axes[0, 1].grid(alpha=0.2)

    axes[1, 0].plot(times, signal, color="0.35", linewidth=0.9, label="Cleaned EEG")
    axes[1, 0].plot(
        times,
        np.where(band_mask, signal, np.nan),
        color="#D55E00",
        linewidth=1.8,
        label=f"Detected {example['band']} bout samples",
    )
    axes[1, 0].set(
        xlabel="Time within accepted epoch (s)",
        ylabel="Amplitude (µV)",
        title="3. Band-bout time mask",
    )
    axes[1, 0].legend(frameon=False)

    ratio = power / thresholds[:, np.newaxis]
    image = axes[1, 1].pcolormesh(
        times,
        frequencies,
        10.0 * np.log10(np.maximum(ratio, np.finfo(float).tiny)),
        shading="auto",
        cmap="viridis",
    )
    axes[1, 1].contour(
        times,
        frequencies,
        detected.astype(float),
        levels=[0.5],
        colors="cyan",
        linewidths=0.8,
    )
    axes[1, 1].set(
        xlabel="Time within accepted epoch (s)",
        ylabel="Frequency (Hz)",
        title="4. Duration-qualified detections",
    )
    fig.colorbar(image, ax=axes[1, 1], label="Power / threshold (dB)")
    fig.suptitle(
        f"{example['subject_id']} — {example['electrode']} — {example['band']} — "
        f"threshold background: {example.get('specparam_aperiodic_mode', 'unknown')}"
    )
    fig.tight_layout()
    _save(fig, path, dpi)


def plot_ordinal_example(example: dict[str, Any], path: Path, dpi: int) -> None:
    """Show one band-limited bout, its ordinal symbols, and probability vector."""
    signal = np.asarray(example["signal"], dtype=float)
    counts = np.asarray(example["counts"], dtype=np.int64)
    dx = int(example["embedding_dimension"])
    tau = int(example["delay_samples"])
    sfreq = float(example["sfreq"])
    times = np.arange(len(signal)) / sfreq
    span = (dx - 1) * tau
    windows = np.lib.stride_tricks.sliding_window_view(signal, span + 1)[..., ::tau]
    symbols = np.argsort(windows, axis=-1)
    centers = (np.arange(len(symbols)) + span / 2.0) / sfreq
    lookup = {pattern: index for index, pattern in enumerate(ordinal_patterns(dx))}
    symbol_indices = np.asarray(
        [lookup[tuple(int(value) for value in symbol)] for symbol in symbols], dtype=int
    )
    probabilities = counts / counts.sum()
    observed = np.flatnonzero(counts)
    top = observed[np.argsort(probabilities[observed])[-min(20, len(observed)) :]]

    fig, axes = plt.subplots(3, 1, figsize=(13, 10))
    axes[0].plot(times, signal, color="0.25", marker="o", markersize=2.5, linewidth=1)
    for sample in range(0, min(len(signal) - span, 8)):
        axes[0].axvspan(
            sample / sfreq,
            (sample + span) / sfreq,
            color="#0072B2" if sample % 2 == 0 else "#D55E00",
            alpha=0.08,
        )
    axes[0].set(
        xlabel="Time from bout onset (s)",
        ylabel="Band-pass amplitude (µV)",
        title="1. Band-limited samples; shaded regions illustrate embedding windows",
    )

    axes[1].step(centers, symbol_indices, where="mid", color="#009E73")
    axes[1].scatter(centers, symbol_indices, s=8, color="#009E73")
    axes[1].set(
        xlabel="Time from bout onset (s)",
        ylabel="Lexicographic pattern index",
        title="2. Ordinal-symbol sequence (each bout encoded independently)",
    )
    axes[1].grid(alpha=0.2)

    labels = ["".join(str(value) for value in ordinal_patterns(dx)[index]) for index in top]
    axes[2].bar(np.arange(len(top)), probabilities[top], color="#CC79A7")
    axes[2].set_xticks(np.arange(len(top)), labels, rotation=90, fontsize=7)
    axes[2].set(
        xlabel="Observed ordinal pattern (top probabilities)",
        ylabel="Probability",
        title="3. Bout ordinal representation",
    )
    fig.suptitle(
        f"{example['subject_id']} — {example['electrode']} — {example['band']} — "
        f"D={dx}, τ={tau}"
    )
    fig.tight_layout()
    _save(fig, path, dpi)


def plot_bout_diagnostics(
    episodes: pd.DataFrame,
    metrics: pd.DataFrame,
    group_order: list[str],
    colors: dict[str, str],
    band_order: list[str],
    band_labels: dict[str, str],
    path: Path,
    dpi: int,
) -> None:
    """Plot bout duration, count, ordinal sample size, and state-space coverage."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    positions = np.arange(len(band_order), dtype=float)
    offsets = np.linspace(-0.17, 0.17, max(1, len(group_order)))
    specifications = (
        (episodes, "duration_s", "Individual bout duration (s)", "Detected bout durations"),
        (metrics, "n_detected_bouts", "Bouts per subject/electrode", "Bout availability"),
        (metrics, "n_ordinal_patterns", "Pooled ordinal patterns", "Ordinal sample size"),
        (
            metrics,
            "ordinal_state_space_coverage",
            "Observed states / D!",
            "Ordinal state-space coverage",
        ),
    )
    for axis, (table, column, ylabel, title) in zip(axes.flat, specifications):
        for group, offset in zip(group_order, offsets):
            for band_index, band in enumerate(band_order):
                selected = table.loc[
                    table["group"].eq(group) & table["band"].eq(band), column
                ].to_numpy(dtype=float)
                _draw_distribution(
                    axis,
                    selected,
                    positions[band_index] + offset,
                    colors[group],
                    width=0.28,
                )
            axis.scatter([], [], color=colors[group], label=group)
        axis.set_xticks(positions, [band_labels[name] for name in band_order])
        axis.set(ylabel=ylabel, title=title)
        axis.grid(axis="y", alpha=0.2)
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Detection and ordinal-representation diagnostics")
    fig.tight_layout()
    _save(fig, path, dpi)


def plot_subject_metric_violins(
    table: pd.DataFrame,
    group_order: list[str],
    colors: dict[str, str],
    band_order: list[str],
    band_labels: dict[str, str],
    path: Path,
    dpi: int,
) -> None:
    """Compare one electrode-averaged value per subject, group, band, and metric."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    positions = np.arange(len(band_order), dtype=float)
    offsets = np.linspace(-0.17, 0.17, max(1, len(group_order)))
    for axis, metric in zip(axes, METRICS):
        for group, offset in zip(group_order, offsets):
            for band_index, band in enumerate(band_order):
                values = table.loc[
                    table["group"].eq(group) & table["band"].eq(band), metric
                ].to_numpy(dtype=float)
                _draw_distribution(
                    axis,
                    values,
                    positions[band_index] + offset,
                    colors[group],
                    width=0.28,
                )
            axis.scatter([], [], color=colors[group], label=group)
        axis.set_xticks(
            positions, [band_labels[name] for name in band_order], rotation=20
        )
        axis.set(ylabel=METRIC_LABELS[metric], title=METRIC_LABELS[metric])
        axis.set_ylim(_finite_limits(table[metric].to_numpy(dtype=float)))
        axis.grid(axis="y", alpha=0.2)
    axes[0].legend(frameon=False)
    fig.suptitle("Ordinal metrics pooled within subjects, then averaged across shared electrodes")
    fig.tight_layout()
    _save(fig, path, dpi)


def plot_ordinal_planes(
    table: pd.DataFrame,
    group_order: list[str],
    colors: dict[str, str],
    band_order: list[str],
    band_labels: dict[str, str],
    path: Path,
    dpi: int,
) -> None:
    """Plot subject-level H×C and H×F planes separately for every band."""
    fig, axes = plt.subplots(len(band_order), 2, figsize=(12, 4 * len(band_order)), squeeze=False)
    pairs = (("entropy", "complexity", "H × C"), ("entropy", "fisher_information", "H × F"))
    for row, band in enumerate(band_order):
        selected_band = table.loc[table["band"].eq(band)]
        for column, (x_metric, y_metric, title) in enumerate(pairs):
            axis = axes[row, column]
            for group in group_order:
                selected = selected_band.loc[selected_band["group"].eq(group)]
                axis.scatter(
                    selected[x_metric],
                    selected[y_metric],
                    color=colors[group],
                    s=24,
                    alpha=0.55,
                    label=group,
                )
            axis.set(
                xlabel=METRIC_LABELS[x_metric],
                ylabel=METRIC_LABELS[y_metric],
                title=f"{band_labels[band]} — {title}",
            )
            axis.grid(alpha=0.2)
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Subject-level bout ordinal planes")
    fig.tight_layout()
    _save(fig, path, dpi)


def plot_electrode_violins(
    table: pd.DataFrame,
    electrode_order: list[str],
    group_order: list[str],
    colors: dict[str, str],
    band_order: list[str],
    band_labels: dict[str, str],
    output_dir: Path,
    dpi: int,
) -> None:
    """Write one three-panel group-distribution figure per frequency band."""
    x = np.arange(len(electrode_order), dtype=float)
    offsets = np.linspace(-0.17, 0.17, max(1, len(group_order)))
    for band in band_order:
        selected_band = table.loc[table["band"].eq(band)]
        fig, axes = plt.subplots(3, 1, figsize=(max(18, 0.34 * len(x)), 16), sharex=True)
        for axis, metric in zip(axes, METRICS):
            for group, offset in zip(group_order, offsets):
                for index, electrode in enumerate(electrode_order):
                    values = selected_band.loc[
                        selected_band["group"].eq(group)
                        & selected_band["electrode"].eq(electrode),
                        metric,
                    ].to_numpy(dtype=float)
                    _draw_distribution(
                        axis, values, x[index] + offset, colors[group], width=0.28
                    )
                axis.scatter([], [], color=colors[group], label=group)
            axis.set(ylabel=METRIC_LABELS[metric], title=METRIC_LABELS[metric])
            axis.grid(axis="y", alpha=0.2)
        axes[-1].set_xticks(x, electrode_order, rotation=90, fontsize=7)
        axes[-1].set_xlabel("Shared electrode")
        axes[0].legend(frameon=False)
        fig.suptitle(f"{band_labels[band]} bout ordinal metrics by electrode")
        fig.tight_layout()
        _save(fig, output_dir / f"{band}_electrode_violins.png", dpi)


def _topomap(axis: Any, values: np.ndarray, info: Any, cmap: str, vlim: tuple[float, float]) -> Any | None:
    finite = np.flatnonzero(np.isfinite(values))
    if len(finite) < 4:
        axis.axis("off")
        axis.text(0.5, 0.5, f"Insufficient finite electrodes\n({len(finite)})", ha="center", va="center")
        return None
    selected_info = mne.pick_info(info, finite.tolist(), copy=True)
    image, _ = mne.viz.plot_topomap(
        values[finite],
        selected_info,
        axes=axis,
        show=False,
        sensors=True,
        contours=6,
        cmap=cmap,
        vlim=vlim,
    )
    return image


def plot_group_topomaps(
    table: pd.DataFrame,
    info: Any,
    group_order: list[str],
    band_order: list[str],
    band_labels: dict[str, str],
    electrode_order: list[str],
    path: Path,
    dpi: int,
) -> None:
    """Plot group means for each band and regular ordinal metric."""
    rows = len(band_order) * len(group_order)
    fig, axes = plt.subplots(rows, len(METRICS), figsize=(12, 3.25 * rows), squeeze=False)
    for column, metric in enumerate(METRICS):
        vlim = _finite_limits(table[metric].to_numpy(dtype=float))
        for band_index, band in enumerate(band_order):
            for group_index, group in enumerate(group_order):
                row = band_index * len(group_order) + group_index
                selected = table.loc[table["band"].eq(band) & table["group"].eq(group)]
                means = selected.groupby("electrode")[metric].mean()
                values = np.asarray([means.get(name, np.nan) for name in electrode_order], dtype=float)
                image = _topomap(axes[row, column], values, info, METRIC_CMAPS[metric], vlim)
                axes[row, column].set_title(f"{band_labels[band]} — {group}\n{METRIC_LABELS[metric]}")
                if image is not None:
                    fig.colorbar(image, ax=axes[row, column], shrink=0.65)
    fig.suptitle("Group-mean bout ordinal topographies")
    fig.tight_layout()
    _save(fig, path, dpi)


def plot_subject_topomaps(
    table: pd.DataFrame,
    infos: dict[str, Any],
    band_order: list[str],
    band_labels: dict[str, str],
    electrode_order: list[str],
    output_dir: Path,
    dpi: int,
) -> None:
    """Write a complete H/C/F × band topographic page for every subject."""
    global_limits = {
        metric: _finite_limits(table[metric].to_numpy(dtype=float)) for metric in METRICS
    }
    for subject_id, selected_subject in table.groupby("subject_id", sort=False):
        fig, axes = plt.subplots(len(band_order), len(METRICS), figsize=(12, 3.2 * len(band_order)), squeeze=False)
        for row, band in enumerate(band_order):
            selected = selected_subject.loc[selected_subject["band"].eq(band)].set_index("electrode")
            for column, metric in enumerate(METRICS):
                values = np.asarray(
                    [selected[metric].get(name, np.nan) for name in electrode_order], dtype=float
                )
                image = _topomap(
                    axes[row, column], values, infos[str(subject_id)], METRIC_CMAPS[metric], global_limits[metric]
                )
                axes[row, column].set_title(f"{band_labels[band]} — {METRIC_LABELS[metric]}")
                if image is not None:
                    fig.colorbar(image, ax=axes[row, column], shrink=0.65)
        group = str(selected_subject["group"].iloc[0])
        fig.suptitle(f"{subject_id} — {group} — ordinal metrics within detected bouts")
        fig.tight_layout()
        _save(fig, output_dir / f"{subject_id}_bout_ordinal_topomaps.png", dpi)
