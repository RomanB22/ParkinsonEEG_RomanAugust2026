"""Complete condition-distribution and association figures for ds002778."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from core.runtime import configure_runtime

configure_runtime()

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CONDITIONS = ("HC", "PD_OFF", "PD_ON")
CONDITION_LABELS = ("HC", "PD OFF", "PD ON")
FAMILY_ORDER = (
    "psd",
    "ordinal",
    "aperiodic",
    "aperiodic_qc",
    "periodic_peak",
    "bouts",
    "within_bout_ordinal",
)


def _safe(value: str) -> str:
    return "".join(character if character.isalnum() or character in "-_" else "_" for character in value)


def _primary(table: pd.DataFrame) -> pd.DataFrame:
    selected = table.loc[
        table["duration_variant"].eq("all_retained")
        & table["sensitivity_cohort"].eq("all_participants")
    ].copy()
    if "analysis_status" in selected:
        selected = selected.loc[selected["analysis_status"].eq("ok")]
    return selected


def _feature_label(row: Any) -> str:
    band = str(row.band)
    metric = str(row.metric).replace("_", " ")
    return f"{band}: {metric}" if band not in {"broadband", "nan"} else metric


def _pages(values: list[str], size: int) -> list[list[str]]:
    return [values[start : start + size] for start in range(0, len(values), size)]


def plot_all_feature_violins(
    features: pd.DataFrame,
    recordings: pd.DataFrame,
    output_dir: str | Path,
    config: dict[str, Any],
) -> list[Path]:
    """Plot every primary subject feature with PD ON/OFF pairing retained."""
    attached = features.loc[features["duration_variant"].eq("all_retained")].merge(
        recordings[["recording_id", "participant_id", "condition"]],
        on="recording_id",
        validate="many_to_one",
    )
    output_dir = Path(output_dir)
    colors = config["plots"]["condition_colors"]
    dpi = int(config["plots"]["dpi"])
    rng = np.random.default_rng(int(config["statistics"]["random_seed"]))
    paths: list[Path] = []
    for family in FAMILY_ORDER:
        family_table = attached.loc[attached["family"].eq(family)]
        feature_ids = family_table["feature_id"].drop_duplicates().astype(str).tolist()
        for page_index, page in enumerate(_pages(feature_ids, 9), start=1):
            figure, axes = plt.subplots(3, 3, figsize=(15, 12), constrained_layout=True)
            for axis, feature_id in zip(axes.flat, page):
                feature_table = family_table.loc[
                    family_table["feature_id"].eq(feature_id)
                ]
                table = feature_table.dropna(subset=["value"])
                first = feature_table.iloc[0]
                if table.empty:
                    axis.text(
                        0.5,
                        0.5,
                        "No finite observations",
                        ha="center",
                        va="center",
                        transform=axis.transAxes,
                    )
                    axis.set_title(_feature_label(first), fontsize=9)
                    axis.set_xticks(range(3), CONDITION_LABELS, fontsize=8)
                    axis.grid(axis="y", alpha=0.18)
                    continue
                paired = table.loc[table["condition"].isin(("PD_OFF", "PD_ON"))].pivot(
                    index="participant_id", columns="condition", values="value"
                ).dropna()
                for row in paired.itertuples():
                    axis.plot((1, 2), (row.PD_OFF, row.PD_ON), color="#777777", alpha=0.22, linewidth=0.7)
                for position, condition in enumerate(CONDITIONS):
                    values = table.loc[table["condition"].eq(condition), "value"].to_numpy(float)
                    if len(values) >= 2 and np.nanmax(values) > np.nanmin(values):
                        violin = axis.violinplot(values, positions=[position], widths=0.72, showmedians=True)
                        violin["bodies"][0].set_facecolor(colors[condition])
                        violin["bodies"][0].set_alpha(0.28)
                        violin["cmedians"].set_color("black")
                    jitter = rng.uniform(-0.08, 0.08, len(values))
                    axis.scatter(position + jitter, values, s=13, color=colors[condition], alpha=0.72, linewidth=0)
                axis.set_title(_feature_label(first), fontsize=9)
                axis.set_xticks(range(3), CONDITION_LABELS, fontsize=8)
                axis.grid(axis="y", alpha=0.18)
            for axis in axes.flat[len(page) :]:
                axis.set_visible(False)
            figure.suptitle(f"ds002778 {family.replace('_', ' ')} subject-feature distributions\nLines connect paired PD OFF/ON recordings")
            path = output_dir / family / f"{family}_subject_violins_page_{page_index:03d}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(path, dpi=dpi)
            plt.close(figure)
            paths.append(path)
    return paths


def _forest_pages(
    table: pd.DataFrame,
    *,
    grouping: tuple[str, ...],
    estimate: str,
    lower: str,
    upper: str,
    output_dir: Path,
    title_prefix: str,
    dpi: int,
) -> list[Path]:
    paths: list[Path] = []
    for keys, group in table.groupby(list(grouping), sort=False, dropna=False):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        ordered = group.sort_values(["band", "metric", "feature_id"])
        for page_index, start in enumerate(range(0, len(ordered), 14), start=1):
            page = ordered.iloc[start : start + 14].copy()
            y = np.arange(len(page))
            estimates = page[estimate].to_numpy(float)
            lows = page[lower].to_numpy(float)
            highs = page[upper].to_numpy(float)
            finite = np.isfinite(estimates) & np.isfinite(lows) & np.isfinite(highs)
            figure, axis = plt.subplots(figsize=(10.5, max(4.2, 0.48 * len(page) + 1.8)), constrained_layout=True)
            axis.errorbar(
                estimates[finite],
                y[finite],
                xerr=np.vstack((estimates[finite] - lows[finite], highs[finite] - estimates[finite])),
                fmt="o",
                color="#0072B2",
                ecolor="#666666",
                capsize=2.5,
                markersize=5,
            )
            significant = page["fdr_reject"].fillna(False).astype(bool).to_numpy() & finite
            axis.scatter(estimates[significant], y[significant], s=72, facecolors="none", edgecolors="#D55E00", linewidths=1.5, label="BH-FDR significant")
            axis.axvline(0.0, color="black", linewidth=0.9, linestyle="--")
            axis.set_yticks(y, [_feature_label(row) for row in page.itertuples()], fontsize=8)
            axis.invert_yaxis()
            axis.grid(axis="x", alpha=0.18)
            if significant.any():
                axis.legend(loc="best", fontsize=8)
            title_keys = " | ".join(str(value).replace("_", " ") for value in key_values)
            axis.set_title(f"{title_prefix}: {title_keys}")
            axis.set_xlabel("Model estimate with 95% confidence interval")
            token = "__".join(_safe(str(value)) for value in key_values)
            path = output_dir / f"{token}_forest_page_{page_index:03d}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(path, dpi=dpi)
            plt.close(figure)
            paths.append(path)
    return paths


def plot_group_statistics(
    condition_statistics: pd.DataFrame,
    output_dir: str | Path,
    config: dict[str, Any],
) -> list[Path]:
    primary = _primary(condition_statistics).dropna(subset=["effect", "ci_lower", "ci_upper"])
    return _forest_pages(
        primary,
        grouping=("family", "contrast"),
        estimate="effect",
        lower="ci_lower",
        upper="ci_upper",
        output_dir=Path(output_dir),
        title_prefix="ds002778 group/paired condition effect",
        dpi=int(config["plots"]["dpi"]),
    )


def plot_mmse_associations(
    mmse_statistics: pd.DataFrame,
    output_dir: str | Path,
    config: dict[str, Any],
) -> list[Path]:
    primary = _primary(mmse_statistics).dropna(subset=["mmse_slope", "ci_lower", "ci_upper"])
    output_dir = Path(output_dir)
    dpi = int(config["plots"]["dpi"])
    paths = _forest_pages(
        primary,
        grouping=("family", "mmse_model"),
        estimate="mmse_slope",
        lower="ci_lower",
        upper="ci_upper",
        output_dir=output_dir / "forests",
        title_prefix="ds002778 adjusted MMSE association",
        dpi=dpi,
    )
    for family, family_table in primary.groupby("family", sort=False):
        heatmap = family_table.pivot_table(
            index="feature_id",
            columns="mmse_model",
            values="standardized_effect",
            aggfunc="first",
        )
        if heatmap.empty:
            continue
        figure, axis = plt.subplots(
            figsize=(8.5, max(4.5, 0.25 * len(heatmap) + 1.8)), constrained_layout=True
        )
        image = axis.imshow(heatmap.to_numpy(float), aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
        axis.set_xticks(range(len(heatmap.columns)), [str(value).replace("_", " ") for value in heatmap.columns], rotation=25, ha="right")
        axis.set_yticks(range(len(heatmap.index)), [str(value).replace("_", " ") for value in heatmap.index], fontsize=7)
        axis.set_title(f"ds002778 {family.replace('_', ' ')} adjusted MMSE associations")
        figure.colorbar(image, ax=axis, label="Standardized association")
        path = output_dir / "heatmaps" / f"{_safe(str(family))}_mmse_correlation_heatmap.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=dpi)
        plt.close(figure)
        paths.append(path)
    return paths


def plot_complete_statistical_battery(
    *,
    subject_features: pd.DataFrame,
    recordings: pd.DataFrame,
    condition_statistics: pd.DataFrame,
    mmse_statistics: pd.DataFrame,
    output_dir: str | Path,
    config: dict[str, Any],
) -> list[Path]:
    root = Path(output_dir)
    paths = plot_all_feature_violins(subject_features, recordings, root / "violins", config)
    paths.extend(plot_group_statistics(condition_statistics, root / "group_statistics", config))
    paths.extend(plot_mmse_associations(mmse_statistics, root / "correlations", config))
    return paths
