"""Group and spatial plots for ordinal EEG metrics."""

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

from .metrics import METRICS


METRIC_STYLE = {
    "entropy": ("Normalized permutation entropy (H)", "viridis"),
    "complexity": ("Statistical complexity (C)", "magma"),
    "fisher_information": ("Fisher information (F)", "cividis"),
}


def _save(fig, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _limits(values: np.ndarray, *, lower_zero: bool = False) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return 0.0, 1.0
    low = float(finite.min())
    high = float(finite.max())
    if np.isclose(low, high):
        padding = max(abs(low) * 0.05, 1e-6)
    else:
        padding = (high - low) * 0.05
    return (max(0.0, low - padding) if lower_zero else low - padding, high + padding)


def _draw_violin(axis, values: np.ndarray, position: float, color: str, width: float) -> None:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) >= 2 and not np.allclose(values, values[0]):
        parts = axis.violinplot(
            [values],
            positions=[position],
            widths=width,
            showmeans=False,
            showmedians=True,
            showextrema=False,
        )
        for body in parts["bodies"]:
            body.set_facecolor(color)
            body.set_edgecolor(color)
            body.set_alpha(0.55)
        parts["cmedians"].set_color("black")
        parts["cmedians"].set_linewidth(0.7)
    elif len(values):
        axis.scatter(
            np.full(len(values), position), values, s=10, color=color, alpha=0.7, zorder=3
        )


def plot_electrode_violins(
    table: pd.DataFrame,
    electrode_order: list[str],
    group_order: list[str],
    colors: dict[str, str],
    output_dir: Path,
    dpi: int,
    analysis_label: str | None = None,
) -> None:
    """Create one PD/Control violin figure for each metric across electrodes."""
    x = np.arange(len(electrode_order), dtype=float)
    offsets = np.linspace(-0.19, 0.19, max(1, len(group_order)))
    for metric in METRICS:
        label, _ = METRIC_STYLE[metric]
        fig, axis = plt.subplots(figsize=(max(18, 0.34 * len(electrode_order)), 7))
        for group, offset in zip(group_order, offsets):
            for index, electrode in enumerate(electrode_order):
                values = table.loc[
                    table["group"].eq(group) & table["electrode"].eq(electrode), metric
                ].to_numpy(dtype=float)
                _draw_violin(axis, values, x[index] + offset, colors[group], 0.34)
            axis.scatter([], [], color=colors[group], alpha=0.65, label=group)
        axis.set_xticks(x, electrode_order, rotation=90, fontsize=7)
        title = f"{label} by electrode and group"
        if analysis_label:
            title = f"{analysis_label} — {title}"
        axis.set(xlabel="Electrode", ylabel=label, title=title)
        axis.set_ylim(_limits(table[metric].to_numpy(), lower_zero=True))
        axis.grid(axis="y", alpha=0.2)
        axis.legend(frameon=False)
        fig.tight_layout()
        _save(fig, output_dir / f"electrode_{metric}_violins.png", dpi)


def plot_subject_average_violins(
    table: pd.DataFrame,
    group_order: list[str],
    colors: dict[str, str],
    path: Path,
    dpi: int,
    analysis_label: str | None = None,
) -> None:
    """Plot subject-level means across electrode metrics."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    positions = np.arange(len(group_order), dtype=float)
    for axis, metric in zip(axes, METRICS):
        label, _ = METRIC_STYLE[metric]
        for position, group in zip(positions, group_order):
            values = table.loc[table["group"].eq(group), metric].to_numpy(dtype=float)
            _draw_violin(axis, values, position, colors[group], 0.7)
            if len(values):
                jitter = np.linspace(-0.10, 0.10, len(values))
                axis.scatter(
                    np.full(len(values), position) + jitter,
                    np.sort(values),
                    s=8,
                    color=colors[group],
                    alpha=0.35,
                )
        axis.set_xticks(positions, group_order)
        axis.set(ylabel=label, title=label)
        axis.set_ylim(_limits(table[metric].to_numpy(), lower_zero=True))
        axis.grid(axis="y", alpha=0.2)
    title = "Subject means across available electrode-level ordinal metrics"
    if analysis_label:
        title = f"{analysis_label} — {title}"
    fig.suptitle(title)
    fig.tight_layout()
    _save(fig, path, dpi)


def _scatter_groups(
    axis,
    table: pd.DataFrame,
    y_metric: str,
    group_order: list[str],
    colors: dict[str, str],
) -> None:
    for group in group_order:
        selected = table.loc[table["group"].eq(group)]
        axis.scatter(
            selected["entropy"],
            selected[y_metric],
            s=18,
            alpha=0.55,
            color=colors[group],
            edgecolors="none",
            label=group,
        )


def plot_subject_average_planes(
    table: pd.DataFrame,
    group_order: list[str],
    colors: dict[str, str],
    path: Path,
    dpi: int,
    analysis_label: str | None = None,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for axis, metric, title in zip(
        axes,
        ("complexity", "fisher_information"),
        ("H × C plane", "H × F plane"),
    ):
        _scatter_groups(axis, table, metric, group_order, colors)
        axis.set(
            xlabel=METRIC_STYLE["entropy"][0],
            ylabel=METRIC_STYLE[metric][0],
            title=title,
        )
        axis.set_xlim(_limits(table["entropy"].to_numpy(), lower_zero=True))
        axis.set_ylim(_limits(table[metric].to_numpy(), lower_zero=True))
        axis.grid(alpha=0.2)
    axes[0].legend(frameon=False)
    title = "Subject means across available electrode-level metrics"
    if analysis_label:
        title = f"{analysis_label} — {title}"
    fig.suptitle(title)
    fig.tight_layout()
    _save(fig, path, dpi)


def plot_electrode_plane_pages(
    table: pd.DataFrame,
    electrode_order: list[str],
    group_order: list[str],
    colors: dict[str, str],
    output_dir: Path,
    dpi: int,
    channels_per_page: int,
    analysis_label: str | None = None,
) -> None:
    """Plot all subject points on per-electrode HxC and HxF planes."""
    columns = 4
    for y_metric, stem, plane_title in (
        ("complexity", "electrode_hxc", "H × C"),
        ("fisher_information", "electrode_hxf", "H × F"),
    ):
        xlim = _limits(table["entropy"].to_numpy(), lower_zero=True)
        ylim = _limits(table[y_metric].to_numpy(), lower_zero=True)
        for page, start in enumerate(range(0, len(electrode_order), channels_per_page), 1):
            channels = electrode_order[start : start + channels_per_page]
            rows = math.ceil(len(channels) / columns)
            fig, axes = plt.subplots(rows, columns, figsize=(14, 3.2 * rows), squeeze=False)
            for axis, electrode in zip(axes.flat, channels):
                selected = table.loc[table["electrode"].eq(electrode)]
                _scatter_groups(axis, selected, y_metric, group_order, colors)
                axis.set(title=electrode, xlim=xlim, ylim=ylim)
                axis.grid(alpha=0.15)
                axis.tick_params(labelsize=7)
            for axis in axes.flat[len(channels) :]:
                axis.axis("off")
            for axis in axes[-1, :]:
                if axis.axison:
                    axis.set_xlabel("H")
            for axis in axes[:, 0]:
                if axis.axison:
                    axis.set_ylabel("C" if y_metric == "complexity" else "F")
            handles = [
                plt.Line2D([], [], linestyle="", marker="o", color=colors[group], label=group)
                for group in group_order
            ]
            fig.legend(handles=handles, loc="upper right", frameon=False)
            title = f"Electrode-level {plane_title} planes — page {page}"
            if analysis_label:
                title = f"{analysis_label} — {title}"
            fig.suptitle(title)
            fig.tight_layout(rect=(0, 0, 1, 0.97))
            _save(fig, output_dir / f"{stem}_p{page:02d}.png", dpi)


def metric_color_limits(table: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """Use identical full-data scales for every subject and group topomap."""
    limits = {}
    for metric in METRICS:
        values = table[metric].to_numpy(dtype=float)
        low, high = float(np.nanmin(values)), float(np.nanmax(values))
        if np.isclose(low, high):
            padding = max(abs(low) * 0.01, 1e-9)
            low, high = low - padding, high + padding
        limits[metric] = (low, high)
    return limits


def _plot_topomap_row(
    axes,
    values_by_metric: dict[str, np.ndarray],
    info,
    limits: dict[str, tuple[float, float]],
) -> None:
    for axis, metric in zip(axes, METRICS):
        label, cmap = METRIC_STYLE[metric]
        image, _ = mne.viz.plot_topomap(
            values_by_metric[metric],
            info,
            axes=axis,
            show=False,
            sensors=True,
            contours=6,
            cmap=cmap,
            vlim=limits[metric],
        )
        axis.set_title(label, fontsize=9)
        axis.figure.colorbar(image, ax=axis, shrink=0.72, pad=0.04)


def plot_subject_topomaps(
    table: pd.DataFrame,
    subject_infos: dict[str, Any],
    limits: dict[str, tuple[float, float]],
    output_dir: Path,
    dpi: int,
) -> None:
    """Create one comparable three-metric scalp map per participant."""
    for subject_id, info in subject_infos.items():
        selected = table.loc[table["subject_id"].eq(subject_id)].set_index("electrode")
        missing = [channel for channel in info.ch_names if channel not in selected.index]
        if missing:
            raise ValueError(f"{subject_id}: metrics missing for channels {missing}")
        values = {
            metric: selected.loc[info.ch_names, metric].to_numpy(dtype=float)
            for metric in METRICS
        }
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        _plot_topomap_row(axes, values, info, limits)
        group = str(selected["group"].iloc[0])
        fig.suptitle(f"{subject_id} ({group}) — ordinal metrics by electrode")
        fig.tight_layout()
        _save(fig, output_dir / f"{subject_id}_ordinal_topomaps.png", dpi)


def plot_group_topomaps(
    table: pd.DataFrame,
    common_info,
    group_order: list[str],
    limits: dict[str, tuple[float, float]],
    path: Path,
    dpi: int,
) -> None:
    """Plot group means on the electrode set shared by every participant."""
    fig, axes = plt.subplots(len(group_order), 3, figsize=(12, 4 * len(group_order)), squeeze=False)
    for row, group in enumerate(group_order):
        selected = (
            table.loc[
                table["group"].eq(group) & table["electrode"].isin(common_info.ch_names)
            ]
            .groupby("electrode")[list(METRICS)]
            .mean()
        )
        values = {
            metric: selected.loc[common_info.ch_names, metric].to_numpy(dtype=float)
            for metric in METRICS
        }
        _plot_topomap_row(axes[row], values, common_info, limits)
        axes[row, 0].text(
            -0.22,
            0.5,
            group,
            transform=axes[row, 0].transAxes,
            rotation=90,
            va="center",
            ha="center",
            fontsize=12,
            fontweight="bold",
        )
    fig.suptitle("Group-mean ordinal topographies — 60 electrodes shared by all subjects")
    fig.tight_layout()
    _save(fig, path, dpi)


def band_metric_color_limits(
    table: pd.DataFrame,
    band_order: list[str],
) -> dict[str, dict[str, tuple[float, float]]]:
    """Return full-cohort subject-map limits separately for every band/metric."""
    return {
        band: metric_color_limits(table.loc[table["band"].eq(band)])
        for band in band_order
    }


def plot_subject_band_topomaps(
    table: pd.DataFrame,
    subject_infos: dict[str, Any],
    band_order: list[str],
    band_labels: dict[str, str],
    limits: dict[str, dict[str, tuple[float, float]]],
    output_dir: Path,
    dpi: int,
) -> None:
    """Create one six-band by three-metric scalp-map figure per participant."""
    for subject_id, info in subject_infos.items():
        subject_table = table.loc[table["subject_id"].eq(subject_id)]
        fig, axes = plt.subplots(
            len(band_order),
            len(METRICS),
            figsize=(12, 3.6 * len(band_order)),
            squeeze=False,
        )
        for row, band in enumerate(band_order):
            selected = subject_table.loc[subject_table["band"].eq(band)].set_index(
                "electrode"
            )
            missing = [channel for channel in info.ch_names if channel not in selected.index]
            if missing:
                raise ValueError(
                    f"{subject_id}/{band}: metrics missing for channels {missing}"
                )
            values = {
                metric: selected.loc[info.ch_names, metric].to_numpy(dtype=float)
                for metric in METRICS
            }
            _plot_topomap_row(axes[row], values, info, limits[band])
            axes[row, 0].text(
                -0.23,
                0.5,
                band_labels[band],
                transform=axes[row, 0].transAxes,
                rotation=90,
                va="center",
                ha="center",
                fontsize=10,
                fontweight="bold",
            )
        group = str(subject_table["group"].iloc[0])
        fig.suptitle(f"{subject_id} ({group}) — band-resolved ordinal topographies")
        fig.tight_layout()
        _save(fig, output_dir / f"{subject_id}_band_ordinal_topomaps.png", dpi)


def plot_group_band_topomaps(
    table: pd.DataFrame,
    common_info,
    group_order: list[str],
    band_order: list[str],
    band_labels: dict[str, str],
    limits: dict[str, dict[str, tuple[float, float]]],
    output_dir: Path,
    dpi: int,
) -> None:
    """Create one PD/Control three-metric group topomap figure per band."""
    for band in band_order:
        band_table = table.loc[table["band"].eq(band)]
        fig, axes = plt.subplots(
            len(group_order), 3, figsize=(12, 4 * len(group_order)), squeeze=False
        )
        for row, group in enumerate(group_order):
            selected = (
                band_table.loc[
                    band_table["group"].eq(group)
                    & band_table["electrode"].isin(common_info.ch_names)
                ]
                .groupby("electrode")[list(METRICS)]
                .mean()
            )
            values = {
                metric: selected.loc[common_info.ch_names, metric].to_numpy(dtype=float)
                for metric in METRICS
            }
            _plot_topomap_row(axes[row], values, common_info, limits[band])
            axes[row, 0].text(
                -0.22,
                0.5,
                group,
                transform=axes[row, 0].transAxes,
                rotation=90,
                va="center",
                ha="center",
                fontsize=12,
                fontweight="bold",
            )
        fig.suptitle(
            f"{band_labels[band]} — group-mean ordinal topographies "
            f"({len(common_info.ch_names)} shared electrodes)"
        )
        fig.tight_layout()
        _save(fig, output_dir / f"{band}_group_mean_topomaps.png", dpi)
