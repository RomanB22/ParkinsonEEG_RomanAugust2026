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

from .metrics import to_db


def _save(fig, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


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
                cmap="magma",
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
