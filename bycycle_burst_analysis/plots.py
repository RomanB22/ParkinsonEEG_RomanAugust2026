"""Transparent QC and group figures for independent bycycle detections."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.runtime import configure_runtime

configure_runtime()

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt


def _save(fig: Any, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_detection_example(
    signal_uv: np.ndarray,
    masks: dict[str, np.ndarray],
    *,
    sfreq: float,
    bands: dict[str, tuple[float, float]],
    band_labels: dict[str, str],
    subject_id: str,
    electrode: str,
    epoch_index: int,
    path: Path,
    dpi: int,
) -> None:
    """Plot band-filtered waveforms with independently detected burst spans."""
    time = np.arange(len(signal_uv)) / float(sfreq)
    fig, axes = plt.subplots(len(bands), 1, figsize=(12, 2.45 * len(bands)), sharex=True)
    axes = np.atleast_1d(axes)
    for axis, (band, limits) in zip(axes, bands.items()):
        sos = butter(4, limits, btype="bandpass", fs=sfreq, output="sos")
        filtered = sosfiltfilt(sos, signal_uv)
        axis.plot(time, filtered, color="0.18", linewidth=0.8)
        mask = np.asarray(masks[band], dtype=bool)
        edges = np.diff(np.pad(mask.astype(np.int8), (1, 1)))
        for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            axis.axvspan(start / sfreq, stop / sfreq, color="#009E73", alpha=0.24)
        axis.set_ylabel(f"{band_labels[band]}\nµV")
        axis.grid(alpha=0.15)
    axes[-1].set_xlabel("Time within accepted epoch (s)")
    fig.suptitle(
        f"Independent bycycle cycle-consistency bursts — {subject_id}, {electrode}, "
        f"accepted epoch {epoch_index}\n"
        "Green spans are detections; each epoch is analyzed independently"
    )
    fig.tight_layout()
    _save(fig, path, dpi)


def plot_group_metric_violins(
    subject_table: pd.DataFrame,
    *,
    metrics: list[str],
    bands: list[str],
    group_order: list[str],
    colors: dict[str, str],
    band_labels: dict[str, str],
    path: Path,
    dpi: int,
) -> None:
    """Show every primary subject-level burst metric by group and band."""
    fig, axes = plt.subplots(
        len(metrics), len(bands), figsize=(3.35 * len(bands), 3.0 * len(metrics)), squeeze=False
    )
    rng = np.random.default_rng(22)
    labels = {
        "oscillatory_occupancy": "Occupancy",
        "bouts_per_minute": "Bouts/min",
        "bout_duration_mean_s": "Mean duration (s)",
        "bout_cycles_mean": "Cycles/bout",
    }
    for row, metric in enumerate(metrics):
        for column, band in enumerate(bands):
            axis = axes[row, column]
            for group_index, group in enumerate(group_order, start=1):
                values = subject_table.loc[
                    subject_table["group"].eq(group) & subject_table["band"].eq(band), metric
                ].dropna().to_numpy(float)
                if not len(values):
                    continue
                if len(values) > 1 and not np.allclose(values, values[0]):
                    violin = axis.violinplot(
                        [values], positions=[group_index], widths=0.72,
                        showmeans=False, showmedians=True, showextrema=False,
                    )
                    violin["bodies"][0].set_facecolor(colors[group])
                    violin["bodies"][0].set_edgecolor(colors[group])
                    violin["bodies"][0].set_alpha(0.35)
                    violin["cmedians"].set_color("black")
                jitter = rng.uniform(-0.055, 0.055, len(values))
                axis.scatter(
                    group_index + jitter, values, s=9, color=colors[group], alpha=0.65,
                    edgecolors="none",
                )
            axis.set_xticks(range(1, len(group_order) + 1), group_order, rotation=20)
            axis.grid(axis="y", alpha=0.18)
            if row == 0:
                axis.set_title(band_labels[band])
            if column == 0:
                axis.set_ylabel(labels.get(metric, metric))
    fig.suptitle("Independent bycycle burst metrics — subject means across shared electrodes")
    fig.tight_layout()
    _save(fig, path, dpi)


def plot_detection_coverage(
    subject_table: pd.DataFrame,
    *,
    bands: list[str],
    group_order: list[str],
    colors: dict[str, str],
    band_labels: dict[str, str],
    path: Path,
    dpi: int,
) -> None:
    """Plot the fraction of subjects with at least one detected bout."""
    rows = []
    for band in bands:
        for group in group_order:
            selected = subject_table.loc[
                subject_table["band"].eq(band) & subject_table["group"].eq(group)
            ]
            rows.append(
                100.0 * float((selected["n_bouts"] > 0).mean()) if len(selected) else np.nan
            )
    values = np.asarray(rows).reshape(len(bands), len(group_order))
    x = np.arange(len(bands))
    width = 0.78 / len(group_order)
    fig, axis = plt.subplots(figsize=(9.5, 5.2))
    for index, group in enumerate(group_order):
        offset = (index - (len(group_order) - 1) / 2.0) * width
        axis.bar(x + offset, values[:, index], width, color=colors[group], alpha=0.8, label=group)
    axis.set_xticks(x, [band_labels[band] for band in bands], rotation=15)
    axis.set_ylim(0, 105)
    axis.set_ylabel("Subjects with at least one detected bout (%)")
    axis.set_title("Independent detector coverage QC")
    axis.grid(axis="y", alpha=0.2)
    axis.legend(frameon=False)
    fig.tight_layout()
    _save(fig, path, dpi)


def plot_event_agreement(
    agreement: pd.DataFrame,
    *,
    bands: list[str],
    group_order: list[str],
    colors: dict[str, str],
    band_labels: dict[str, str],
    path: Path,
    dpi: int,
) -> None:
    """Plot subject-level time-mask Dice agreement with eBOSC."""
    if agreement.empty:
        return
    subject = agreement.groupby(["subject_id", "group", "band"], sort=False)["dice"].mean().reset_index()
    fig, axes = plt.subplots(1, len(bands), figsize=(3.25 * len(bands), 3.8), squeeze=False)
    rng = np.random.default_rng(8)
    for axis, band in zip(axes.flat, bands):
        for index, group in enumerate(group_order, start=1):
            values = subject.loc[
                subject["band"].eq(band) & subject["group"].eq(group), "dice"
            ].dropna().to_numpy(float)
            if len(values) > 1 and not np.allclose(values, values[0]):
                violin = axis.violinplot([values], positions=[index], widths=0.7,
                                          showmedians=True, showextrema=False)
                violin["bodies"][0].set_facecolor(colors[group])
                violin["bodies"][0].set_alpha(0.35)
                violin["cmedians"].set_color("black")
            axis.scatter(index + rng.uniform(-0.05, 0.05, len(values)), values,
                         s=9, color=colors[group], alpha=0.6)
        axis.set_xticks(range(1, len(group_order) + 1), group_order, rotation=20)
        axis.set_ylim(-0.03, 1.03)
        axis.set_title(band_labels[band])
        axis.grid(axis="y", alpha=0.2)
    axes[0, 0].set_ylabel("Time-mask Dice coefficient")
    fig.suptitle("Event-level agreement: independent bycycle vs aperiodic eBOSC")
    fig.tight_layout()
    _save(fig, path, dpi)


def plot_metric_agreement(
    paired: pd.DataFrame,
    *,
    metrics: list[str],
    bands: list[str],
    band_labels: dict[str, str],
    path: Path,
    dpi: int,
) -> None:
    """Plot eBOSC-versus-bycycle subject summaries without implying identity."""
    if paired.empty:
        return
    fig, axes = plt.subplots(len(metrics), len(bands), figsize=(3.1 * len(bands), 3.0 * len(metrics)), squeeze=False)
    for row, metric in enumerate(metrics):
        for column, band in enumerate(bands):
            axis = axes[row, column]
            selected = paired.loc[paired["band"].eq(band)]
            x = selected[f"{metric}_ebosc"].to_numpy(float)
            y = selected[f"{metric}_bycycle"].to_numpy(float)
            finite = np.isfinite(x) & np.isfinite(y)
            axis.scatter(x[finite], y[finite], s=11, color="0.25", alpha=0.55)
            if finite.sum() >= 3:
                rho = pd.Series(x[finite]).corr(pd.Series(y[finite]), method="spearman")
                axis.text(0.04, 0.94, f"Spearman ρ={rho:.2f}", transform=axis.transAxes,
                          ha="left", va="top", fontsize=8)
            axis.grid(alpha=0.17)
            if row == 0:
                axis.set_title(band_labels[band])
            if column == 0:
                axis.set_ylabel(f"bycycle\n{metric}")
            if row == len(metrics) - 1:
                axis.set_xlabel("eBOSC")
    fig.suptitle("Detector agreement at the subject-summary level")
    fig.tight_layout()
    _save(fig, path, dpi)
