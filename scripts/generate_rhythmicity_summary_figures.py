#!/usr/bin/env python
"""Generate compact, publication-friendly summaries of rhythmicity results.

The script only reads finalized participant/statistics tables. It does not rerun
LAVI, ABBA segmentation, or temporal-burst detection.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


BANDS = ["delta", "theta", "alpha", "beta", "gamma"]
ABBA_BANDS = ["theta_1", "alpha_1", "beta_1", "beta_2", "gamma_1"]
DATASETS = ["primary", "ds007526-1.0.2", "ds008768-1.0.0", "medication_state"]
DATASET_LABELS = {
    "primary": "Primary",
    "ds007526-1.0.2": "ds007526",
    "ds008768-1.0.0": "ds008768",
    "medication_state": "Medication",
}
QUANTITIES = {
    "bursts_per_minute": "Burst rate",
    "cycles": "Cycles / burst",
    "duration_s": "Duration",
    "occupancy_percent": "Occupancy",
    "peak_amplitude_uv": "Peak amplitude",
}
GROUP_COLORS = {
    "Control": "#7f7f7f",
    "PD": "#d95f02",
    "PD_OFF": "#7570b3",
    "PD_ON": "#1b9e77",
}
BAND_COLORS = {
    "theta": "#72B7B2",
    "alpha": "#F2CF5B",
    "beta": "#F58518",
    "gamma": "#E45756",
}


def _stars(q_value: float) -> str:
    if not np.isfinite(q_value) or q_value >= 0.05:
        return ""
    if q_value < 0.001:
        return "***"
    if q_value < 0.01:
        return "**"
    return "*"


def _style_axis(axis: plt.Axes) -> None:
    axis.tick_params(length=0, labelsize=9)
    for spine in axis.spines.values():
        spine.set_visible(False)


def _annotate_heatmap(
    axis: plt.Axes,
    values: np.ndarray,
    markers: np.ndarray,
    limit: float,
    digits: int = 2,
) -> None:
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            if not np.isfinite(value):
                axis.text(column, row, "—", ha="center", va="center", color="#777777", fontsize=9)
                continue
            color = "white" if abs(value) >= 0.55 * limit else "#202020"
            label = f"{value:+.{digits}f}"
            marker = markers[row, column]
            if marker and marker != "nan":
                label += f"\n{marker}"
            else:
                marker = ""
            axis.text(
                column,
                row,
                label,
                ha="center",
                va="center",
                color=color,
                fontsize=8.3,
                fontweight="bold" if marker else "normal",
                linespacing=0.75,
            )


def _matrix(
    table: pd.DataFrame,
    row_key: str,
    column_key: str,
    value_key: str,
    rows: list[str],
    columns: list[str],
    dtype: type = float,
) -> np.ndarray:
    pivot = table.pivot(index=row_key, columns=column_key, values=value_key)
    return pivot.reindex(index=rows, columns=columns).to_numpy(dtype=dtype)


def _save(fig: plt.Figure, output_stem: Path) -> None:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _disease_group(dataset: str) -> str:
    return "PD_ON" if dataset == "medication_state" else "PD"


def _violin_at(
    axis: plt.Axes,
    values: np.ndarray,
    position: float,
    color: str,
    rng: np.random.Generator,
) -> None:
    values = values[np.isfinite(values)]
    if len(values) > 1 and np.ptp(values) > 0:
        violin = axis.violinplot(
            [values], positions=[position], widths=0.28, showmedians=True, showextrema=False
        )
        body = violin["bodies"][0]
        body.set_facecolor(color)
        body.set_edgecolor("white")
        body.set_alpha(0.64)
        violin["cmedians"].set_color("#202020")
        violin["cmedians"].set_linewidth(1.1)
    axis.scatter(
        position + rng.uniform(-0.055, 0.055, len(values)),
        values,
        s=9,
        color=color,
        alpha=0.50,
        linewidth=0,
        rasterized=True,
    )


def _add_bracket(
    axis: plt.Axes,
    x_left: float,
    x_right: float,
    y: float,
    height: float,
    label: str,
) -> None:
    if not label:
        return
    axis.plot(
        [x_left, x_left, x_right, x_right],
        [y, y + height, y + height, y],
        color="#303030",
        linewidth=0.9,
        clip_on=False,
    )
    axis.text(
        (x_left + x_right) / 2,
        y + 1.08 * height,
        label,
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
    )


def _participant_violin_panel(
    axis: plt.Axes,
    table: pd.DataFrame,
    value_column: str,
    title: str,
    statistics: pd.DataFrame,
    statistic_band_column: str,
    statistic_band: str,
    require_relevance: bool,
    rng: np.random.Generator,
) -> None:
    offsets = {"Control": -0.17, "disease": 0.17}
    all_values = pd.to_numeric(table[value_column], errors="coerce").dropna().to_numpy(float)
    span = float(np.ptp(all_values)) if len(all_values) else 1.0
    span = max(span, 1e-6)
    for dataset_index, dataset in enumerate(DATASETS):
        disease = _disease_group(dataset)
        subset = table.loc[table["dataset"].eq(dataset)]
        for group, offset_key in (("Control", "Control"), (disease, "disease")):
            values = pd.to_numeric(
                subset.loc[subset["group"].eq(group), value_column], errors="coerce"
            ).dropna().to_numpy(float)
            _violin_at(
                axis,
                values,
                dataset_index + offsets[offset_key],
                GROUP_COLORS[group],
                rng,
            )
        stat = statistics.loc[
            statistics["dataset"].eq(dataset)
            & statistics[statistic_band_column].eq(statistic_band)
        ]
        if "quantity" in statistics.columns:
            stat = stat.loc[statistics.loc[stat.index, "quantity"].eq(value_column)]
        if dataset == "medication_state":
            stat = stat.loc[
                ((stat["group_a"] == "Control") & (stat["group_b"] == "PD_ON"))
                | ((stat["group_a"] == "PD_ON") & (stat["group_b"] == "Control"))
            ]
        else:
            stat = stat.loc[
                ((stat["group_a"] == "Control") & (stat["group_b"] == "PD"))
                | ((stat["group_a"] == "PD") & (stat["group_b"] == "Control"))
            ]
        if len(stat):
            row = stat.iloc[0]
            supported = bool(row["significant_fdr"])
            if require_relevance:
                supported = supported and bool(row["relevant_effect"])
            marker = _stars(float(row["q_fdr_bh"])) if supported else ""
            local = subset.loc[subset["group"].isin(["Control", disease]), value_column]
            local_high = float(pd.to_numeric(local, errors="coerce").max())
            _add_bracket(
                axis,
                dataset_index - 0.17,
                dataset_index + 0.17,
                local_high + 0.035 * span,
                0.018 * span,
                marker,
            )
    axis.set_xticks(range(len(DATASETS)), [DATASET_LABELS[name] for name in DATASETS])
    axis.set_title(title, loc="left", fontsize=12, fontweight="bold")
    axis.grid(axis="y", alpha=0.18)
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(labelsize=9)


def plot_lavi_violins(rhythmicity_root: Path, figure_dir: Path) -> None:
    participant = pd.read_csv(rhythmicity_root / "metrics" / "participant_rhythmicity.csv.gz")
    statistics = pd.read_csv(rhythmicity_root / "statistics" / "lavi_group_comparisons.csv")
    rng = np.random.default_rng(20260914)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
    for panel, (axis, band) in enumerate(zip(axes, ["theta", "beta"])):
        _participant_violin_panel(
            axis,
            participant.loc[participant["band"].eq(band)],
            "lavi_mean",
            f"{chr(65 + panel)}   {band.title()} LAVI",
            statistics,
            "band",
            band,
            True,
            rng,
        )
        axis.set_ylabel("Mean LAVI")
    handles = [
        Patch(facecolor=GROUP_COLORS["Control"], label="Control"),
        Patch(facecolor=GROUP_COLORS["PD"], label="PD"),
        Patch(facecolor=GROUP_COLORS["PD_ON"], label="PD-ON (Medication only)"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.91), ncol=3, frameon=False)
    fig.suptitle("Participant-level LAVI group differences", fontsize=16, fontweight="bold", y=0.99)
    fig.text(
        0.5,
        0.02,
        "Each point is one participant. Stars require BH-FDR q < .05 and |Hedges g| ≥ 0.50. "
        "The medication comparison is PD-ON vs Control.",
        ha="center",
        fontsize=8.7,
        color="#444444",
    )
    fig.subplots_adjust(left=0.07, right=0.98, top=0.78, bottom=0.15, wspace=0.20)
    _save(fig, figure_dir / "rhythmicity_lavi_violins")


def plot_lavi_cognitive_scatters(rhythmicity_root: Path, figure_dir: Path) -> None:
    participant = pd.read_csv(rhythmicity_root / "metrics" / "participant_rhythmicity.csv.gz")
    statistics = pd.read_csv(rhythmicity_root / "statistics" / "lavi_clinical_correlations.csv")
    for band in ["theta", "alpha"]:
        fig, axes = plt.subplots(1, len(DATASETS), figsize=(16.0, 4.8), sharey=True)
        rng = np.random.default_rng(20260914)
        legend_groups: set[str] = set()
        for panel, (axis, dataset) in enumerate(zip(axes, DATASETS)):
            clinical_column = "mmse" if dataset == "medication_state" else "moca"
            clinical_label = clinical_column.upper()
            subset = participant.loc[
                participant["dataset"].eq(dataset) & participant["band"].eq(band)
            ].copy()
            subset[clinical_column] = pd.to_numeric(subset[clinical_column], errors="coerce")
            subset["lavi_mean"] = pd.to_numeric(subset["lavi_mean"], errors="coerce")
            subset = subset.dropna(subset=[clinical_column, "lavi_mean"])
            for group, frame in subset.groupby("group", sort=False):
                if group not in GROUP_COLORS:
                    continue
                legend_groups.add(group)
                axis.scatter(
                    frame[clinical_column] + rng.uniform(-0.06, 0.06, len(frame)),
                    frame["lavi_mean"],
                    s=19,
                    color=GROUP_COLORS[group],
                    alpha=0.60,
                    edgecolor="white",
                    linewidth=0.25,
                )
            if len(subset) >= 2 and subset[clinical_column].nunique() > 1:
                slope, intercept = np.polyfit(subset[clinical_column], subset["lavi_mean"], 1)
                x_line = np.linspace(subset[clinical_column].min(), subset[clinical_column].max(), 100)
                axis.plot(x_line, slope * x_line + intercept, color="#222222", linewidth=1.4)
            stat = statistics.loc[
                statistics["dataset"].eq(dataset)
                & statistics["band"].eq(band)
                & statistics["clinical_column"].eq(clinical_column)
                & statistics["lavi_metric"].eq("lavi_mean")
            ].iloc[0]
            marker = "*" if bool(stat["significant_and_relevant"]) else ""
            axis.text(
                0.04,
                0.96,
                f"n={int(stat['n'])}\nρ={stat['rho']:+.2f}, q={stat['q_fdr_bh']:.3g} {marker}",
                transform=axis.transAxes,
                ha="left",
                va="top",
                fontsize=8.8,
                fontweight="bold" if marker else "normal",
            )
            axis.set_title(
                f"{chr(65 + panel)}   {DATASET_LABELS[dataset]}",
                loc="left",
                fontsize=11,
                fontweight="bold",
            )
            axis.set_xlabel(clinical_label)
            if panel == 0:
                axis.set_ylabel(f"Mean {band.title()} LAVI")
            axis.grid(alpha=0.18)
            axis.spines[["top", "right"]].set_visible(False)
        handles = [
            Patch(facecolor=GROUP_COLORS[group], label=group.replace("_", "-"))
            for group in ["Control", "PD", "PD_OFF", "PD_ON"]
            if group in legend_groups
        ]
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.90), ncol=4, frameon=False)
        fig.suptitle(
            f"{band.title()} LAVI and cognition across datasets",
            fontsize=16,
            fontweight="bold",
            y=0.99,
        )
        fig.text(
            0.5,
            0.018,
            "Lines are pooled least-squares fits; annotations report Spearman ρ and BH-FDR q. "
            "* indicates q < .05 and |ρ| ≥ .30. Medication uses MMSE; other datasets use MoCA.",
            ha="center",
            fontsize=8.5,
            color="#444444",
        )
        fig.subplots_adjust(left=0.055, right=0.99, top=0.76, bottom=0.16, wspace=0.14)
        _save(fig, figure_dir / f"rhythmicity_lavi_cognitive_scatters_{band}")


def plot_medication_on_off_violins(rhythmicity_root: Path, figure_dir: Path) -> None:
    participant = pd.read_csv(rhythmicity_root / "metrics" / "participant_rhythmicity.csv.gz")
    statistics = pd.read_csv(rhythmicity_root / "statistics" / "lavi_group_comparisons.csv")
    medication = participant.loc[
        participant["dataset"].eq("medication_state")
        & participant["group"].isin(["PD_OFF", "PD_ON"])
    ]
    fig, axes = plt.subplots(1, len(BANDS), figsize=(15.5, 4.6))
    for panel, (axis, band) in enumerate(zip(axes, BANDS)):
        subset = medication.loc[medication["band"].eq(band)]
        paired = subset.pivot(index="participant_id", columns="group", values="lavi_mean").dropna(
            subset=["PD_OFF", "PD_ON"]
        )
        values = [paired["PD_OFF"].to_numpy(float), paired["PD_ON"].to_numpy(float)]
        violin = axis.violinplot(values, positions=[0, 1], widths=0.58, showmedians=True, showextrema=False)
        for body, color in zip(violin["bodies"], [GROUP_COLORS["PD_OFF"], GROUP_COLORS["PD_ON"]]):
            body.set_facecolor(color)
            body.set_edgecolor("white")
            body.set_alpha(0.62)
        violin["cmedians"].set_color("#202020")
        for row in paired.itertuples(index=False):
            axis.plot([0, 1], [row.PD_OFF, row.PD_ON], color="#777777", alpha=0.30, linewidth=0.7)
        axis.scatter([0] * len(paired), paired["PD_OFF"], color=GROUP_COLORS["PD_OFF"], s=18, zorder=3)
        axis.scatter([1] * len(paired), paired["PD_ON"], color=GROUP_COLORS["PD_ON"], s=18, zorder=3)
        stat = statistics.loc[
            statistics["dataset"].eq("medication_state")
            & statistics["band"].eq(band)
            & statistics["comparison"].eq("PD_ON - PD_OFF")
        ].iloc[0]
        axis.set_title(
            f"{chr(65 + panel)}   {band.title()}\nΔ={stat['mean_difference']:+.3f}, q={stat['q_fdr_bh']:.3f}",
            fontsize=10.5,
            fontweight="bold",
        )
        axis.set_xticks([0, 1], ["PD-OFF", "PD-ON"])
        if panel == 0:
            axis.set_ylabel("Mean LAVI")
        axis.grid(axis="y", alpha=0.18)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Medication-state LAVI: PD-OFF versus PD-ON", fontsize=16, fontweight="bold", y=0.99)
    fig.text(
        0.5,
        0.018,
        f"Lines connect the same participants (n={len(paired)} pairs per band). Δ is PD-ON − PD-OFF; q values are from the existing Welch comparison table.",
        ha="center",
        fontsize=8.6,
        color="#444444",
    )
    fig.subplots_adjust(left=0.055, right=0.99, top=0.76, bottom=0.15, wspace=0.25)
    _save(fig, figure_dir / "rhythmicity_lavi_medication_on_off_violins")


def plot_abba_medication_on_off_violins(rhythmicity_root: Path, figure_dir: Path) -> None:
    participant = pd.read_csv(
        rhythmicity_root / "metrics" / "abba_burst_participant_metrics.csv.gz"
    )
    statistics = pd.read_csv(
        rhythmicity_root / "statistics" / "abba_burst_focused_group_comparisons.csv"
    )
    medication = participant.loc[
        participant["dataset"].eq("medication_state")
        & participant["group"].isin(["PD_OFF", "PD_ON"])
        & participant["band_name"].isin(ABBA_BANDS)
    ]
    characteristics = {
        "bursts_per_minute": ("Burst rate", "Bursts / minute"),
        "cycles": ("Cycles per burst", "Cycles"),
        "duration_s": ("Burst duration", "Duration (s)"),
        "occupancy_percent": ("Oscillatory occupancy", "Occupancy (%)"),
        "peak_amplitude_uv": ("Peak amplitude", "Peak amplitude (µV)"),
    }
    for quantity, (title, ylabel) in characteristics.items():
        fig, axes = plt.subplots(1, len(ABBA_BANDS), figsize=(15.5, 4.6))
        pair_counts: list[int] = []
        for panel, (axis, band) in enumerate(zip(axes, ABBA_BANDS)):
            subset = medication.loc[medication["band_name"].eq(band)]
            paired = subset.pivot(index="participant_id", columns="group", values=quantity).dropna(
                subset=["PD_OFF", "PD_ON"]
            )
            pair_counts.append(len(paired))
            values = [paired["PD_OFF"].to_numpy(float), paired["PD_ON"].to_numpy(float)]
            violin = axis.violinplot(
                values, positions=[0, 1], widths=0.58, showmedians=True, showextrema=False
            )
            for body, color in zip(
                violin["bodies"], [GROUP_COLORS["PD_OFF"], GROUP_COLORS["PD_ON"]]
            ):
                body.set_facecolor(color)
                body.set_edgecolor("white")
                body.set_alpha(0.62)
            violin["cmedians"].set_color("#202020")
            for row in paired.itertuples(index=False):
                axis.plot(
                    [0, 1],
                    [row.PD_OFF, row.PD_ON],
                    color="#777777",
                    alpha=0.30,
                    linewidth=0.7,
                )
            axis.scatter(
                [0] * len(paired),
                paired["PD_OFF"],
                color=GROUP_COLORS["PD_OFF"],
                s=18,
                zorder=3,
            )
            axis.scatter(
                [1] * len(paired),
                paired["PD_ON"],
                color=GROUP_COLORS["PD_ON"],
                s=18,
                zorder=3,
            )
            stat = statistics.loc[
                statistics["dataset"].eq("medication_state")
                & statistics["band_name"].eq(band)
                & statistics["quantity"].eq(quantity)
                & statistics["group_a"].eq("PD_OFF")
                & statistics["group_b"].eq("PD_ON")
            ].iloc[0]
            difference = float(stat["mean_b"] - stat["mean_a"])
            marker = _stars(float(stat["q_fdr_bh"])) if bool(stat["significant_fdr"]) else ""
            axis.set_title(
                f"{chr(65 + panel)}   {band.replace('_', ' ').title()}\n"
                f"Δ={difference:+.3g}, q={stat['q_fdr_bh']:.3g} {marker}",
                fontsize=10.2,
                fontweight="bold",
            )
            axis.set_xticks([0, 1], ["PD-OFF", "PD-ON"])
            if panel == 0:
                axis.set_ylabel(ylabel)
            axis.grid(axis="y", alpha=0.18)
            axis.spines[["top", "right"]].set_visible(False)
        if len(set(pair_counts)) != 1:
            raise ValueError(f"Unequal medication pair counts for {quantity}: {pair_counts}")
        fig.suptitle(
            f"Medication-state ABBA bursts: {title}",
            fontsize=16,
            fontweight="bold",
            y=0.99,
        )
        fig.text(
            0.5,
            0.018,
            f"Lines connect the same participants (n={pair_counts[0]} pairs per interval). "
            "Δ is PD-ON − PD-OFF; stars mark BH-FDR q < .05 from the existing Welch table. "
            "ABBA intervals are group-specific.",
            ha="center",
            fontsize=8.4,
            color="#444444",
        )
        fig.subplots_adjust(left=0.055, right=0.99, top=0.76, bottom=0.15, wspace=0.25)
        _save(fig, figure_dir / f"rhythmicity_abba_medication_on_off_{quantity}_violins")


def plot_abba_group_band_definitions(rhythmicity_root: Path, figure_dir: Path) -> None:
    definitions = pd.read_csv(
        rhythmicity_root / "statistics" / "abba_group_mean_segments.csv"
    )
    definitions = definitions.loc[definitions["band_name"].isin(ABBA_BANDS)].copy()
    group_order = ["Control", "PD", "PD_OFF", "PD_ON"]
    frequency_ticks = [3.2, 4, 6, 8, 10, 13, 20, 30, 45]
    fig, axes = plt.subplots(2, 2, figsize=(14.0, 8.0), squeeze=False)
    for panel, (axis, dataset) in enumerate(zip(axes.flat, DATASETS)):
        subset = definitions.loc[definitions["dataset"].eq(dataset)]
        groups = [group for group in group_order if group in set(subset["group"])]
        y_positions = {group: len(groups) - index - 1 for index, group in enumerate(groups)}
        for boundary in [4, 8, 13, 30]:
            axis.axvline(boundary, color="#b8b8b8", linestyle="--", linewidth=0.8, zorder=0)
        for row in subset.itertuples(index=False):
            y = y_positions[str(row.group)]
            start = float(row.start_hz)
            end = float(row.end_hz)
            width = end - start
            rectangle = plt.Rectangle(
                (start, y - 0.29),
                width,
                0.58,
                facecolor=BAND_COLORS[str(row.canonical_region)],
                edgecolor="#303030",
                linewidth=0.75,
                alpha=0.88 if row.direction == "high" else 0.55,
                hatch="" if row.direction == "high" else "///",
                zorder=2,
            )
            axis.add_patch(rectangle)
            midpoint = np.sqrt(start * end)
            short_name = str(row.band_name).replace("theta", "θ").replace("alpha", "α")
            short_name = short_name.replace("beta", "β").replace("gamma", "γ").replace("_", "")
            axis.text(
                midpoint,
                y,
                f"{short_name}\n{start:g}–{end:g}",
                ha="center",
                va="center",
                fontsize=7.2,
                fontweight="bold",
                color="#202020",
                zorder=3,
                linespacing=0.9,
            )
        axis.set_xscale("log")
        axis.set_xlim(3.1, 46.5)
        axis.set_ylim(-0.65, len(groups) - 0.35)
        axis.set_xticks(frequency_ticks, [f"{tick:g}" for tick in frequency_ticks])
        axis.minorticks_off()
        axis.set_yticks(
            [y_positions[group] for group in groups],
            [group.replace("_", "-") for group in groups],
        )
        axis.set_xlabel("Frequency (Hz)")
        axis.set_title(
            f"{chr(65 + panel)}   {DATASET_LABELS[dataset]}",
            loc="left",
            fontsize=12,
            fontweight="bold",
        )
        axis.grid(axis="x", alpha=0.12)
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.tick_params(axis="y", length=0)
    band_handles = [
        Patch(facecolor=BAND_COLORS[band], edgecolor="#303030", label=band.title())
        for band in ["theta", "alpha", "beta", "gamma"]
    ]
    direction_handles = [
        Patch(facecolor="#b0b0b0", edgecolor="#303030", label="High LAVI"),
        Patch(facecolor="#b0b0b0", edgecolor="#303030", hatch="///", alpha=0.55, label="Low LAVI"),
    ]
    fig.legend(
        handles=band_handles + direction_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncol=6,
        frameon=False,
    )
    fig.suptitle("Focused ABBA interval definitions by dataset and group", fontsize=16, fontweight="bold", y=0.99)
    fig.text(
        0.5,
        0.018,
        "Numbers inside bars are start–end frequencies in Hz. Dashed lines mark canonical 4, 8, 13, and 30 Hz boundaries. "
        "Labels identify corresponding data-driven segments, not shared fixed passbands.",
        ha="center",
        fontsize=8.6,
        color="#444444",
    )
    fig.subplots_adjust(left=0.08, right=0.985, top=0.83, bottom=0.10, hspace=0.40, wspace=0.20)
    _save(fig, figure_dir / "rhythmicity_abba_group_band_definitions")


def plot_abba_violins(rhythmicity_root: Path, figure_dir: Path) -> None:
    participant = pd.read_csv(rhythmicity_root / "metrics" / "abba_burst_participant_metrics.csv.gz")
    statistics = pd.read_csv(
        rhythmicity_root / "statistics" / "abba_burst_focused_group_comparisons.csv"
    )
    selections = [
        ("alpha_1", "bursts_per_minute", "Alpha 1 · burst rate", "Bursts / minute"),
        ("beta_1", "bursts_per_minute", "Beta 1 · burst rate", "Bursts / minute"),
        ("theta_1", "occupancy_percent", "Theta 1 · occupancy", "Occupancy (%)"),
        ("beta_1", "duration_s", "Beta 1 · duration", "Duration (s)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.3))
    rng = np.random.default_rng(20260914)
    for panel, (axis, (band, quantity, title, ylabel)) in enumerate(zip(axes.flat, selections)):
        _participant_violin_panel(
            axis,
            participant.loc[participant["band_name"].eq(band)],
            quantity,
            f"{chr(65 + panel)}   {title}",
            statistics,
            "band_name",
            band,
            False,
            rng,
        )
        axis.set_ylabel(ylabel)
    handles = [
        Patch(facecolor=GROUP_COLORS["Control"], label="Control"),
        Patch(facecolor=GROUP_COLORS["PD"], label="PD"),
        Patch(facecolor=GROUP_COLORS["PD_ON"], label="PD-ON (Medication only)"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.94), ncol=3, frameon=False)
    fig.suptitle("Selected ABBA temporal-burst group differences", fontsize=16, fontweight="bold", y=0.995)
    fig.text(
        0.5,
        0.018,
        "Each point is one participant; stars mark BH-FDR-supported contrasts. ABBA intervals are group-specific and exploratory. "
        "The medication comparison is PD-ON vs Control.",
        ha="center",
        fontsize=8.7,
        color="#444444",
    )
    fig.subplots_adjust(left=0.075, right=0.98, top=0.86, bottom=0.10, hspace=0.34, wspace=0.20)
    _save(fig, figure_dir / "rhythmicity_abba_burst_violins")


def _topomap_info(channels: list[str]):
    import mne

    info = mne.create_info(channels, sfreq=250.0, ch_types="eeg")
    info.set_montage(mne.channels.make_standard_montage("standard_1020"), on_missing="ignore")
    return info


def _unique_montage_values(values: pd.Series):
    info = _topomap_info([str(channel) for channel in values.index])
    data = values.to_numpy(float)
    keep: list[int] = []
    seen: set[tuple[float, float]] = set()
    for index, channel in enumerate(info["chs"]):
        position = channel["loc"][:3]
        if not np.isfinite(data[index]) or not np.isfinite(position).all() or np.linalg.norm(position) == 0:
            continue
        key = tuple(np.round(position[:2], 8))
        if key in seen:
            continue
        seen.add(key)
        keep.append(index)
    import mne

    return mne.pick_info(info, keep), data[keep]


def plot_theta_topomaps(rhythmicity_root: Path, figure_dir: Path) -> None:
    import mne

    electrode = pd.read_csv(
        rhythmicity_root / "metrics" / "participant_electrode_rhythmicity.csv.gz"
    )
    statistics = pd.read_csv(rhythmicity_root / "statistics" / "lavi_electrode_comparisons.csv")
    electrode = electrode.loc[electrode["band"].eq("theta")]
    contrasts: dict[str, pd.Series] = {}
    significant: dict[str, set[str]] = {}
    for dataset in DATASETS:
        subset = electrode.loc[electrode["dataset"].eq(dataset)]
        disease = _disease_group(dataset)
        disease_mean = subset.loc[subset["group"].eq(disease)].groupby("electrode")["lavi_mean"].mean()
        control_mean = subset.loc[subset["group"].eq("Control")].groupby("electrode")["lavi_mean"].mean()
        contrasts[dataset] = disease_mean.subtract(control_mean, fill_value=np.nan)
        comparison = f"{disease} - Control"
        significant[dataset] = set(
            statistics.loc[
                statistics["dataset"].eq(dataset)
                & statistics["band"].eq("theta")
                & statistics["comparison"].eq(comparison)
                & statistics["significant_fdr"].astype(bool),
                "electrode",
            ].astype(str)
        )
    finite = np.concatenate(
        [values.dropna().to_numpy(float) for values in contrasts.values() if len(values.dropna())]
    )
    limit = max(0.02, float(np.percentile(np.abs(finite), 98)))
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(12.8, 3.5))
    image = None
    for axis, dataset in zip(axes, DATASETS):
        info, data = _unique_montage_values(contrasts[dataset])
        mask = np.asarray([channel in significant[dataset] for channel in info.ch_names], dtype=bool)
        image, _ = mne.viz.plot_topomap(
            data,
            info,
            axes=axis,
            show=False,
            sensors=True,
            contours=0,
            vlim=(-limit, limit),
            cmap="RdBu_r",
            extrapolate="head",
            image_interp="cubic",
            mask=mask,
            mask_params={
                "marker": "o",
                "markerfacecolor": "white",
                "markeredgecolor": "black",
                "linestyle": "None",
                "linewidth": 0.6,
                "markersize": 5,
            },
        )
        axis.set_title(
            f"{DATASET_LABELS[dataset]}\n{len(significant[dataset])} FDR electrodes",
            fontsize=10,
            fontweight="bold",
        )
    colorbar_axis = fig.add_axes((0.92, 0.19, 0.018, 0.58))
    colorbar = fig.colorbar(image, cax=colorbar_axis)
    colorbar.set_label("Theta LAVI difference", fontsize=9)
    colorbar.ax.tick_params(labelsize=8)
    colorbar.outline.set_visible(False)
    fig.suptitle("Spatial distribution of the theta LAVI effect", fontsize=16, fontweight="bold", y=0.98)
    fig.text(
        0.5,
        0.075,
        "Maps show Parkinson state − Control (PD-ON for Medication). White-ringed sensors pass electrode-wise BH-FDR q < .05.",
        ha="center",
        fontsize=8.7,
        color="#444444",
    )
    fig.subplots_adjust(left=0.025, right=0.90, top=0.82, bottom=0.14, wspace=0.12)
    _save(fig, figure_dir / "rhythmicity_theta_topomaps")


def plot_lavi_summary(statistics_dir: Path, figure_dir: Path) -> None:
    group = pd.read_csv(statistics_dir / "lavi_group_comparisons.csv")
    clinical = pd.read_csv(statistics_dir / "lavi_clinical_correlations.csv")

    group = group.loc[
        ((group["dataset"] != "medication_state") & (group["comparison"] == "PD - Control"))
        | ((group["dataset"] == "medication_state") & (group["comparison"] == "PD_ON - Control"))
    ].copy()
    group["marker"] = np.where(
        group["significant_and_relevant"].astype(bool),
        group["q_fdr_bh"].map(_stars),
        "",
    )

    # Each dataset has one primary cognitive scale in these outputs: MoCA for
    # the three standard cohorts and MMSE for the medication-state cohort.
    clinical = clinical.loc[clinical["lavi_metric"].eq("lavi_mean")].copy()
    preferred_measure = {
        "primary": "MoCA",
        "ds007526-1.0.2": "MoCA",
        "ds008768-1.0.0": "MoCA",
        "medication_state": "MMSE",
    }
    clinical = clinical.loc[
        clinical.apply(lambda row: row["clinical_measure"] == preferred_measure.get(row["dataset"]), axis=1)
    ].copy()
    clinical["marker"] = np.where(
        clinical["significant_and_relevant"].astype(bool),
        clinical["q_fdr_bh"].map(_stars),
        "",
    )

    group_values = _matrix(group, "dataset", "band", "hedges_g", DATASETS, BANDS)
    group_markers = _matrix(group, "dataset", "band", "marker", DATASETS, BANDS, dtype=object).astype(str)
    clinical_values = _matrix(clinical, "dataset", "band", "rho", DATASETS, BANDS)
    clinical_markers = _matrix(
        clinical, "dataset", "band", "marker", DATASETS, BANDS, dtype=object
    ).astype(str)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.7), constrained_layout=False)
    specifications = [
        (
            axes[0],
            group_values,
            group_markers,
            1.1,
            "A   Group differences",
            "Hedges g: Parkinson state − Control",
        ),
        (
            axes[1],
            clinical_values,
            clinical_markers,
            0.5,
            "B   Cognitive associations",
            "Spearman ρ: LAVI mean vs cognitive score",
        ),
    ]
    images = []
    for axis, values, markers, limit, title, subtitle in specifications:
        image = axis.imshow(
            values,
            cmap="RdBu_r",
            norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
            aspect="auto",
        )
        images.append(image)
        _annotate_heatmap(axis, values, markers, limit)
        axis.set_xticks(range(len(BANDS)), [band.title() for band in BANDS])
        axis.set_yticks(range(len(DATASETS)), [DATASET_LABELS[name] for name in DATASETS])
        axis.set_title(title, loc="left", fontsize=13, fontweight="bold", pad=28)
        axis.text(0, 1.035, subtitle, transform=axis.transAxes, fontsize=9.5, color="#4b4b4b")
        _style_axis(axis)
        axis.set_xticks(np.arange(-0.5, len(BANDS), 1), minor=True)
        axis.set_yticks(np.arange(-0.5, len(DATASETS), 1), minor=True)
        axis.grid(which="minor", color="white", linewidth=2)
        axis.tick_params(which="minor", bottom=False, left=False)

    for axis, image, label in zip(axes, images, ["Hedges g", "Spearman ρ"]):
        colorbar = fig.colorbar(image, ax=axis, orientation="horizontal", fraction=0.075, pad=0.13)
        colorbar.set_label(label, fontsize=9)
        colorbar.ax.tick_params(labelsize=8, length=2)
        colorbar.outline.set_visible(False)

    fig.suptitle("Rhythmicity summary: LAVI group and cognitive effects", fontsize=16, fontweight="bold", y=0.985)
    fig.text(
        0.5,
        0.018,
        "Medication row uses PD-ON vs Control and MMSE; other rows use PD vs Control and MoCA. "
        "Stars require BH-FDR q < .05 and the prespecified practical-effect threshold (* q<.05, ** q<.01, *** q<.001).",
        ha="center",
        va="bottom",
        fontsize=8.6,
        color="#444444",
    )
    fig.subplots_adjust(left=0.09, right=0.98, top=0.80, bottom=0.20, wspace=0.30)
    _save(fig, figure_dir / "rhythmicity_summary_lavi")


def plot_abba_summary(statistics_dir: Path, figure_dir: Path) -> None:
    table = pd.read_csv(statistics_dir / "abba_burst_focused_group_comparisons.csv")
    selected = table.loc[
        ((table["dataset"] != "medication_state") & (table["group_a"] == "Control") & (table["group_b"] == "PD"))
        | (
            (table["dataset"] == "medication_state")
            & (table["group_a"] == "Control")
            & (table["group_b"] == "PD_ON")
        )
    ].copy()
    if (selected[["mean_a", "mean_b"]] <= 0).any().any():
        raise ValueError("ABBA summary requires positive group means for log2 ratios")
    selected["log2_ratio"] = np.log2(selected["mean_b"] / selected["mean_a"])
    selected["marker"] = np.where(
        selected["significant_fdr"].astype(bool), selected["q_fdr_bh"].map(_stars), ""
    )

    limit = max(1.25, float(np.nanmax(np.abs(selected["log2_ratio"]))))
    fig, axes = plt.subplots(1, len(QUANTITIES), figsize=(18.5, 5.2), sharey=True)
    image = None
    for panel, (axis, (quantity, label)) in enumerate(zip(axes, QUANTITIES.items())):
        subset = selected.loc[selected["quantity"].eq(quantity)]
        values = _matrix(subset, "dataset", "band_name", "log2_ratio", DATASETS, ABBA_BANDS)
        markers = _matrix(
            subset, "dataset", "band_name", "marker", DATASETS, ABBA_BANDS, dtype=object
        ).astype(str)
        image = axis.imshow(
            values,
            cmap="PuOr_r",
            norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
            aspect="auto",
        )
        _annotate_heatmap(axis, values, markers, limit)
        axis.set_xticks(range(len(ABBA_BANDS)), [band.replace("_", " ").title() for band in ABBA_BANDS])
        axis.tick_params(axis="x", labelrotation=45)
        axis.set_yticks(range(len(DATASETS)), [DATASET_LABELS[name] for name in DATASETS])
        axis.set_title(f"{chr(65 + panel)}   {label}", loc="left", fontsize=11.5, fontweight="bold", pad=12)
        _style_axis(axis)
        axis.set_xticks(np.arange(-0.5, len(ABBA_BANDS), 1), minor=True)
        axis.set_yticks(np.arange(-0.5, len(DATASETS), 1), minor=True)
        axis.grid(which="minor", color="white", linewidth=1.8)
        axis.tick_params(which="minor", bottom=False, left=False)

    assert image is not None
    colorbar_axis = fig.add_axes((0.30, 0.145, 0.40, 0.027))
    colorbar = fig.colorbar(image, cax=colorbar_axis, orientation="horizontal")
    colorbar.set_label("log₂(Parkinson-state mean / Control mean)", fontsize=9)
    colorbar.ax.tick_params(labelsize=8, length=2)
    colorbar.outline.set_visible(False)
    fig.suptitle("Rhythmicity summary: ABBA temporal-burst differences", fontsize=16, fontweight="bold", y=0.98)
    fig.text(
        0.5,
        0.02,
        "Positive values indicate a higher mean in PD (PD-ON for Medication); negative values indicate a lower mean. "
        "Stars mark BH-FDR-supported contrasts. ABBA intervals are group-specific, so this is an exploratory directional summary.",
        ha="center",
        va="bottom",
        fontsize=8.6,
        color="#444444",
    )
    fig.subplots_adjust(left=0.07, right=0.992, top=0.82, bottom=0.29, wspace=0.16)
    _save(fig, figure_dir / "rhythmicity_summary_abba_bursts")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rhythmicity-root",
        type=Path,
        default=Path("outputs/rhythmicity"),
        help="Directory containing the finalized rhythmicity outputs",
    )
    args = parser.parse_args()
    statistics_dir = args.rhythmicity_root / "statistics"
    figure_dir = args.rhythmicity_root / "figures" / "summary"
    plot_lavi_summary(statistics_dir, figure_dir)
    plot_abba_summary(statistics_dir, figure_dir)
    plot_lavi_violins(args.rhythmicity_root, figure_dir)
    plot_lavi_cognitive_scatters(args.rhythmicity_root, figure_dir)
    plot_medication_on_off_violins(args.rhythmicity_root, figure_dir)
    plot_abba_medication_on_off_violins(args.rhythmicity_root, figure_dir)
    plot_abba_group_band_definitions(args.rhythmicity_root, figure_dir)
    plot_abba_violins(args.rhythmicity_root, figure_dir)
    plot_theta_topomaps(args.rhythmicity_root, figure_dir)
    print(f"Wrote rhythmicity summary figures to {figure_dir}")


if __name__ == "__main__":
    main()
