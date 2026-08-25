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

from .metrics import CORE_METRICS, METRICS


METRIC_STYLE = {
    "entropy": ("Normalized permutation entropy (H)", "viridis"),
    "complexity": ("Statistical complexity (C)", "magma"),
    "fisher_information": ("Fisher information (F)", "cividis"),
    "renyi_entropy_alpha_0_1": ("Rényi entropy (Hα, α=0.1)", "viridis"),
    "renyi_complexity_alpha_0_1": ("Rényi complexity (Cα, α=0.1)", "magma"),
    "renyi_entropy_alpha_0_5": ("Rényi entropy (Hα, α=0.5)", "viridis"),
    "renyi_complexity_alpha_0_5": ("Rényi complexity (Cα, α=0.5)", "magma"),
    "renyi_entropy_alpha_0_9": ("Rényi entropy (Hα, α=0.9)", "viridis"),
    "renyi_complexity_alpha_0_9": ("Rényi complexity (Cα, α=0.9)", "magma"),
    "renyi_entropy_alpha_1_1": ("Rényi entropy (Hα, α=1.1)", "viridis"),
    "renyi_complexity_alpha_1_1": ("Rényi complexity (Cα, α=1.1)", "magma"),
    "renyi_entropy_alpha_2": ("Rényi entropy (Hα, α=2)", "viridis"),
    "renyi_complexity_alpha_2": ("Rényi complexity (Cα, α=2)", "magma"),
    "renyi_entropy_alpha_5": ("Rényi entropy (Hα, α=5)", "viridis"),
    "renyi_complexity_alpha_5": ("Rényi complexity (Cα, α=5)", "magma"),
}

PLANE_PAIRS = (
    ("entropy", "complexity", "electrode_hxc", "H × C"),
    ("entropy", "fisher_information", "electrode_hxf", "H × F"),
    (
        "renyi_entropy_alpha_0_1",
        "renyi_complexity_alpha_0_1",
        "electrode_renyi_hxc_alpha_0_1",
        "Rényi Hα × Cα (α=0.1)",
    ),
    (
        "renyi_entropy_alpha_0_5",
        "renyi_complexity_alpha_0_5",
        "electrode_renyi_hxc_alpha_0_5",
        "Rényi Hα × Cα (α=0.5)",
    ),
    (
        "renyi_entropy_alpha_0_9",
        "renyi_complexity_alpha_0_9",
        "electrode_renyi_hxc_alpha_0_9",
        "Rényi Hα × Cα (α=0.9)",
    ),
    (
        "renyi_entropy_alpha_1_1",
        "renyi_complexity_alpha_1_1",
        "electrode_renyi_hxc_alpha_1_1",
        "Rényi Hα × Cα (α=1.1)",
    ),
    (
        "renyi_entropy_alpha_2",
        "renyi_complexity_alpha_2",
        "electrode_renyi_hxc_alpha_2",
        "Rényi Hα × Cα (α=2)",
    ),
    (
        "renyi_entropy_alpha_5",
        "renyi_complexity_alpha_5",
        "electrode_renyi_hxc_alpha_5",
        "Rényi Hα × Cα (α=5)",
    ),
)


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
    columns = 3
    rows = math.ceil(len(METRICS) / columns)
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(14, 4.5 * rows),
        squeeze=False,
    )
    positions = np.arange(len(group_order), dtype=float)
    for axis, metric in zip(axes.flat, METRICS):
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
    for axis in axes.flat[len(METRICS) :]:
        axis.axis("off")
    title = "Subject means across shared electrode-level ordinal metrics"
    if analysis_label:
        title = f"{analysis_label} — {title}"
    fig.suptitle(title)
    fig.tight_layout()
    _save(fig, path, dpi)


def _scatter_groups(
    axis,
    table: pd.DataFrame,
    x_metric: str,
    y_metric: str,
    group_order: list[str],
    colors: dict[str, str],
) -> None:
    for group in group_order:
        selected = table.loc[table["group"].eq(group)]
        axis.scatter(
            selected[x_metric],
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
    columns = 2
    rows = math.ceil(len(PLANE_PAIRS) / columns)
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(11, 4.7 * rows),
        squeeze=False,
    )
    for axis, (x_metric, y_metric, _, title) in zip(axes.flat, PLANE_PAIRS):
        _scatter_groups(axis, table, x_metric, y_metric, group_order, colors)
        axis.set(
            xlabel=METRIC_STYLE[x_metric][0],
            ylabel=METRIC_STYLE[y_metric][0],
            title=f"{title} plane",
        )
        axis.set_xlim(_limits(table[x_metric].to_numpy(), lower_zero=True))
        axis.set_ylim(_limits(table[y_metric].to_numpy(), lower_zero=True))
        axis.grid(alpha=0.2)
    for axis in axes.flat[len(PLANE_PAIRS) :]:
        axis.axis("off")
    axes.flat[0].legend(frameon=False)
    title = "Subject means across shared electrode-level metrics"
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
    for x_metric, y_metric, stem, plane_title in PLANE_PAIRS:
        if output_dir.exists():
            for stale_page in output_dir.glob(f"{stem}_p*.png"):
                stale_page.unlink()
        xlim = _limits(table[x_metric].to_numpy(), lower_zero=True)
        ylim = _limits(table[y_metric].to_numpy(), lower_zero=True)
        for page, start in enumerate(range(0, len(electrode_order), channels_per_page), 1):
            channels = electrode_order[start : start + channels_per_page]
            rows = math.ceil(len(channels) / columns)
            fig, axes = plt.subplots(rows, columns, figsize=(14, 3.2 * rows), squeeze=False)
            for axis, electrode in zip(axes.flat, channels):
                selected = table.loc[table["electrode"].eq(electrode)]
                _scatter_groups(
                    axis,
                    selected,
                    x_metric,
                    y_metric,
                    group_order,
                    colors,
                )
                axis.set(title=electrode, xlim=xlim, ylim=ylim)
                axis.grid(alpha=0.15)
                axis.tick_params(labelsize=7)
            for axis in axes.flat[len(channels) :]:
                axis.axis("off")
            for axis in axes[-1, :]:
                if axis.axison:
                    axis.set_xlabel(METRIC_STYLE[x_metric][0])
            for axis in axes[:, 0]:
                if axis.axison:
                    axis.set_ylabel(METRIC_STYLE[y_metric][0])
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
    for metric in CORE_METRICS:
        values = table[metric].to_numpy(dtype=float)
        low, high = float(np.nanmin(values)), float(np.nanmax(values))
        if np.isclose(low, high):
            padding = max(abs(low) * 0.01, 1e-9)
            low, high = low - padding, high + padding
        limits[metric] = (low, high)
    return limits


def electrode_metric_zscores(
    table: pd.DataFrame,
    *,
    strata: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Z-score metrics across subjects within each stratum and electrode.

    Groups are deliberately pooled during standardization so that subsequent
    group means retain between-group differences. Constant metric/electrode
    combinations are assigned zero.
    """
    required = {"electrode", *strata, *CORE_METRICS}
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"Missing z-score columns: {missing}")

    keys = [*strata, "electrode"]
    grouped = table.groupby(keys, sort=False, dropna=False)[list(CORE_METRICS)]
    means = grouped.transform("mean")
    standard_deviations = grouped.transform("std", ddof=0)
    valid = standard_deviations.gt(0.0) & np.isfinite(standard_deviations)
    zscores = (table.loc[:, CORE_METRICS] - means) / standard_deviations
    zscores = zscores.where(valid, 0.0)
    zscores = zscores.where(table.loc[:, CORE_METRICS].notna())

    standardized = table.copy()
    standardized.loc[:, CORE_METRICS] = zscores
    return standardized


def group_mean_symmetric_color_limits(
    standardized_table: pd.DataFrame,
    common_channels: list[str],
) -> dict[str, tuple[float, float]]:
    """Return zero-centered limits spanning standardized group means."""
    group_means = (
        standardized_table.loc[
            standardized_table["electrode"].isin(common_channels)
        ]
        .groupby(["group", "electrode"])[list(CORE_METRICS)]
        .mean()
    )
    limits = {}
    for metric in CORE_METRICS:
        maximum = float(np.nanmax(np.abs(group_means[metric].to_numpy(dtype=float))))
        if not np.isfinite(maximum) or np.isclose(maximum, 0.0):
            maximum = 1.0
        limits[metric] = (-maximum, maximum)
    return limits


def _plot_topomap_row(
    axes,
    values_by_metric: dict[str, np.ndarray],
    info,
    limits: dict[str, tuple[float, float]],
) -> None:
    for axis, metric in zip(axes, CORE_METRICS):
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


def _plot_standardized_topomap_row(
    axes,
    values_by_metric: dict[str, np.ndarray],
    info,
    limits: dict[str, tuple[float, float]],
) -> None:
    for axis, metric in zip(axes, CORE_METRICS):
        label, _ = METRIC_STYLE[metric]
        image, _ = mne.viz.plot_topomap(
            values_by_metric[metric],
            info,
            axes=axis,
            show=False,
            sensors=True,
            contours=6,
            cmap="RdBu_r",
            vlim=limits[metric],
        )
        axis.set_title(f"{label}\nMean pooled-cohort z-score", fontsize=9)
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
            for metric in CORE_METRICS
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
            .groupby("electrode")[list(CORE_METRICS)]
            .mean()
        )
        values = {
            metric: selected.loc[common_info.ch_names, metric].to_numpy(dtype=float)
            for metric in CORE_METRICS
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
    fig.suptitle(
        "Group-mean ordinal topographies — "
        f"{len(common_info.ch_names)} electrodes shared by all analyzed subjects"
    )
    fig.tight_layout()
    _save(fig, path, dpi)


def plot_group_standardized_topomaps(
    standardized_table: pd.DataFrame,
    common_info,
    group_order: list[str],
    limits: dict[str, tuple[float, float]],
    path: Path,
    dpi: int,
    analysis_label: str,
) -> None:
    """Plot pooled-cohort, electrode-wise z-score means for each group."""
    fig, axes = plt.subplots(
        len(group_order), 3, figsize=(12, 4 * len(group_order)), squeeze=False
    )
    for row, group in enumerate(group_order):
        selected = (
            standardized_table.loc[
                standardized_table["group"].eq(group)
                & standardized_table["electrode"].isin(common_info.ch_names)
            ]
            .groupby("electrode")[list(CORE_METRICS)]
            .mean()
        )
        values = {
            metric: selected.loc[common_info.ch_names, metric].to_numpy(dtype=float)
            for metric in CORE_METRICS
        }
        _plot_standardized_topomap_row(axes[row], values, common_info, limits)
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
        f"{analysis_label} — group-mean electrode-wise z-scores "
        f"({len(common_info.ch_names)} shared electrodes)"
    )
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
            len(CORE_METRICS),
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
                for metric in CORE_METRICS
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
                .groupby("electrode")[list(CORE_METRICS)]
                .mean()
            )
            values = {
                metric: selected.loc[common_info.ch_names, metric].to_numpy(dtype=float)
                for metric in CORE_METRICS
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
