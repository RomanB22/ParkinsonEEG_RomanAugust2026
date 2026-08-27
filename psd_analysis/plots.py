"""PSD confidence-band and group band-power topographic figures."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.runtime import configure_runtime

configure_runtime()

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

from src.plotting import save_figure as _save

from .metrics import to_db


def plot_group_median_psd(
    summary: pd.DataFrame,
    group_order: list[str],
    colors: dict[str, str],
    path: Path,
    dpi: int,
) -> None:
    """Plot group medians and pointwise bootstrap confidence bands."""
    fig, axis = plt.subplots(figsize=(10, 6))
    for group in group_order:
        selected = summary.loc[summary["group"].eq(group)].sort_values("frequency_hz")
        frequencies = selected["frequency_hz"].to_numpy(dtype=float)
        median = to_db(selected["median_psd_uv2_hz"].to_numpy(dtype=float))
        lower = to_db(selected["ci_lower_psd_uv2_hz"].to_numpy(dtype=float))
        upper = to_db(selected["ci_upper_psd_uv2_hz"].to_numpy(dtype=float))
        axis.plot(frequencies, median, color=colors[group], linewidth=2.0, label=group)
        axis.fill_between(frequencies, lower, upper, color=colors[group], alpha=0.22)
    axis.set(
        xlabel="Frequency (Hz)",
        ylabel="PSD (dB µV²/Hz)",
        title="Concatenated accepted epochs — group median PSD with pointwise 95% bootstrap CIs",
    )
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    fig.tight_layout()
    _save(fig, path, dpi)


def plot_group_relative_band_power_violins(
    subject_band_table: pd.DataFrame,
    band_order: list[str],
    band_labels: dict[str, str],
    group_order: list[str],
    colors: dict[str, str],
    path: Path,
    dpi: int,
) -> None:
    """Plot one subject-level relative-power distribution per group and band."""
    n_columns = min(3, len(band_order))
    n_rows = int(np.ceil(len(band_order) / n_columns))
    fig, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(4.3 * n_columns, 4.0 * n_rows),
        squeeze=False,
    )
    jitter_rng = np.random.default_rng(0)
    for axis, band in zip(axes.flat, band_order):
        for group_index, group in enumerate(group_order):
            values = subject_band_table.loc[
                subject_band_table["group"].eq(group)
                & subject_band_table["band"].eq(band),
                "median_relative_band_power_percent",
            ].dropna().to_numpy(dtype=float)
            if not len(values):
                continue
            position = float(group_index + 1)
            if len(values) >= 2 and not np.allclose(values, values[0]):
                violin = axis.violinplot(
                    [values],
                    positions=[position],
                    widths=0.72,
                    showmeans=False,
                    showmedians=True,
                    showextrema=False,
                )
                body = violin["bodies"][0]
                body.set_facecolor(colors[group])
                body.set_edgecolor(colors[group])
                body.set_alpha(0.35)
                violin["cmedians"].set_color("black")
                violin["cmedians"].set_linewidth(1.5)
            else:
                axis.hlines(
                    float(np.median(values)),
                    position - 0.18,
                    position + 0.18,
                    color="black",
                    linewidth=1.5,
                )
            jitter = jitter_rng.uniform(-0.06, 0.06, len(values))
            axis.scatter(
                position + jitter,
                values,
                s=18,
                color=colors[group],
                edgecolor="white",
                linewidth=0.35,
                alpha=0.8,
                zorder=3,
            )
        counts = [
            subject_band_table.loc[
                subject_band_table["group"].eq(group)
                & subject_band_table["band"].eq(band),
                "subject_id",
            ].nunique()
            for group in group_order
        ]
        axis.set_xticks(
            np.arange(1, len(group_order) + 1),
            [f"{group}\nn={count}" for group, count in zip(group_order, counts)],
        )
        axis.set(
            ylabel="Relative power (% of total 1–50 Hz power)",
            title=band_labels[band].replace("\n", " — "),
        )
        axis.grid(axis="y", alpha=0.2)
    for axis in axes.flat[len(band_order) :]:
        axis.set_visible(False)
    fig.suptitle(
        "Subject-level relative band power — PD vs Control",
        fontsize=14,
    )
    fig.tight_layout()
    _save(fig, path, dpi)


def plot_group_band_topomaps(
    group_band_table: pd.DataFrame,
    info: Any,
    band_order: list[str],
    band_labels: dict[str, str],
    group_order: list[str],
    path: Path,
    dpi: int,
) -> dict[str, tuple[float, float]]:
    """Plot group median relative band power on common electrodes."""
    fig = plt.figure(figsize=(3.7 * len(band_order), 3.9 * len(group_order)))
    grid = fig.add_gridspec(
        len(group_order) + 1,
        len(band_order),
        height_ratios=[1.0] * len(group_order) + [0.055],
        hspace=0.28,
        wspace=0.28,
    )
    axes = np.asarray(
        [
            [fig.add_subplot(grid[row, column]) for column in range(len(band_order))]
            for row in range(len(group_order))
        ]
    )
    colorbar_axes = [
        fig.add_subplot(grid[len(group_order), column])
        for column in range(len(band_order))
    ]
    limits: dict[str, tuple[float, float]] = {}
    images = {}
    for column, band in enumerate(band_order):
        band_values = group_band_table.loc[
            group_band_table["band"].eq(band),
            "median_relative_band_power_percent",
        ].to_numpy(dtype=float)
        low, high = float(np.min(band_values)), float(np.max(band_values))
        if np.isclose(low, high):
            padding = max(abs(low) * 0.01, 1e-6)
            low, high = low - padding, high + padding
        limits[band] = (low, high)
        for row, group in enumerate(group_order):
            selected = group_band_table.loc[
                group_band_table["group"].eq(group)
                & group_band_table["band"].eq(band)
            ].set_index("electrode")
            missing = [channel for channel in info.ch_names if channel not in selected.index]
            if missing:
                raise ValueError(f"{group}/{band}: missing common electrodes {missing}")
            values = selected.loc[
                info.ch_names, "median_relative_band_power_percent"
            ].to_numpy(dtype=float)
            image, _ = mne.viz.plot_topomap(
                values,
                info,
                axes=axes[row, column],
                show=False,
                sensors=True,
                contours=6,
                cmap="viridis",
                vlim=(low, high),
            )
            images[column] = image
            axes[row, column].set_title(band_labels[band], fontsize=10)
            if column == 0:
                axes[row, column].text(
                    -0.24,
                    0.5,
                    group,
                    transform=axes[row, column].transAxes,
                    rotation=90,
                    va="center",
                    ha="center",
                    fontsize=12,
                    fontweight="bold",
                )
    for column, band in enumerate(band_order):
        colorbar = fig.colorbar(
            images[column],
            cax=colorbar_axes[column],
            orientation="horizontal",
        )
        colorbar.set_label("Relative power (% of total 1–50 Hz power)", fontsize=8)
        colorbar.ax.tick_params(labelsize=7)
    fig.suptitle(
        "Group median relative band-power topographies — common electrodes"
    )
    fig.subplots_adjust(top=0.90, bottom=0.08)
    _save(fig, path, dpi)
    return limits
