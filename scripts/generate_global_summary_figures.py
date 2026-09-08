#!/usr/bin/env python3
"""Create a compact, presentation-oriented figure set from saved global outputs."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib.colors import TwoSlopeNorm, to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from scipy.stats import spearmanr, ttest_ind


DATASETS = ["primary", "ds007526-1.0.2", "ds008768-1.0.0"]
DATASET_LABELS = {
    "primary": "Primary",
    "ds007526-1.0.2": "ds007526",
    "ds008768-1.0.0": "ds008768",
    "medication_state": "Medication state",
}
DATASET_COLORS = {
    "primary": "#2166ac",
    "ds007526-1.0.2": "#4daf4a",
    "ds008768-1.0.0": "#d73027",
}

SUMMARY_FEATURES = [
    ("Theta relative power", "psd__theta__relative_power"),
    ("Theta burst count", "bout__theta__n_bouts"),
    ("Theta burst occupancy", "bout__theta__oscillatory_occupancy"),
    ("Beta relative power", "psd__beta__relative_power"),
    ("Alpha entropy", "entropy__alpha__entropy__D3"),
]

BURST_FEATURES = [
    ("Burst count", "bout__theta__n_bouts"),
    ("Burst rate", "bout__theta__bouts_per_minute"),
    ("Occupancy", "bout__theta__oscillatory_occupancy"),
]

MEDICATION_FEATURES = [
    ("Theta relative power", "psd__theta__relative_power"),
    ("Beta relative power", "psd__beta__relative_power"),
    ("Theta occupancy", "bout__theta__oscillatory_occupancy"),
    ("Aperiodic offset", "aperiodic__broadband__fixed_aperiodic_offset"),
]

GROUP_COLORS = {
    "Control": "#7f7f7f",
    "PD": "#d95f02",
    "PD_OFF": "#7570b3",
    "PD_ON": "#1b9e77",
}


def _groups_for_dataset(dataset: str) -> list[str]:
    return ["Control", "PD_OFF", "PD_ON"] if dataset == "medication_state" else ["Control", "PD"]


def _read(root: Path, dataset: str, kind: str) -> pd.DataFrame:
    path = root / kind / dataset / f"{kind.rstrip('s')}_features.csv.gz"
    # The directory names are metrics and the filenames are singular in the
    # saved outputs; keep the explicit fallback for clarity and robustness.
    if kind == "metrics":
        path = root / "metrics" / dataset / "subject_features.csv.gz"
    return pd.read_csv(path, low_memory=False)


def _read_subject(root: Path, dataset: str) -> pd.DataFrame:
    return pd.read_csv(root / "metrics" / dataset / "subject_features.csv.gz", low_memory=False)


def _read_recording(root: Path, dataset: str) -> pd.DataFrame:
    return pd.read_csv(root / "metrics" / dataset / "recording_features.csv.gz", low_memory=False)


def _read_stats(root: Path, dataset: str) -> pd.DataFrame:
    return pd.read_csv(root / "statistics" / dataset / "group_statistics.csv.gz", low_memory=False)


def _pd_control_rows(stats: pd.DataFrame, feature: str) -> pd.DataFrame:
    selected = stats.loc[stats["feature"].eq(feature)].copy()
    selected = selected.loc[
        selected.apply(
            lambda row: {str(row["group_a"]), str(row["group_b"])} == {"PD", "Control"},
            axis=1,
        )
    ]
    return selected


def _effect_direction(row: pd.Series) -> float:
    difference = float(row["mean_b"] - row["mean_a"])
    if str(row["group_b"]) == "PD":
        return difference
    return -difference


def _standardized_effects(recording: pd.DataFrame, feature: str) -> dict[str, float]:
    """Return electrode-wise Cohen d values oriented as PD minus Control."""
    values: dict[str, float] = {}
    if feature not in recording:
        return values
    selected = recording[["electrode", "group", feature]].copy()
    selected[feature] = pd.to_numeric(selected[feature], errors="coerce")
    selected = selected.dropna(subset=["electrode", "group", feature])
    for electrode, frame in selected.groupby("electrode", sort=False):
        control = frame.loc[frame["group"].astype(str).eq("Control"), feature].to_numpy(float)
        pd_values = frame.loc[frame["group"].astype(str).eq("PD"), feature].to_numpy(float)
        if len(control) < 2 or len(pd_values) < 2:
            continue
        pooled = np.sqrt(
            ((len(control) - 1) * np.var(control, ddof=1)
             + (len(pd_values) - 1) * np.var(pd_values, ddof=1))
            / (len(control) + len(pd_values) - 2)
        )
        if np.isfinite(pooled) and pooled > 0:
            values[str(electrode)] = float((np.mean(pd_values) - np.mean(control)) / pooled)
    return values


def _hedges_g(left: np.ndarray, right: np.ndarray) -> tuple[float, float, float]:
    left = left[np.isfinite(left)]
    right = right[np.isfinite(right)]
    if len(left) < 2 or len(right) < 2:
        return np.nan, np.nan, np.nan
    pooled = np.sqrt(
        ((len(left) - 1) * np.var(left, ddof=1)
         + (len(right) - 1) * np.var(right, ddof=1))
        / (len(left) + len(right) - 2)
    )
    if not np.isfinite(pooled) or pooled == 0:
        return np.nan, np.nan, np.nan
    d = (np.mean(right) - np.mean(left)) / pooled
    correction = 1.0 - 3.0 / max(4.0 * (len(left) + len(right)) - 9.0, 1.0)
    g = correction * d
    se = np.sqrt(
        (len(left) + len(right)) / (len(left) * len(right))
        + (g * g) / max(2.0 * (len(left) + len(right) - 2), 1.0)
    )
    return float(g), float(g - 1.96 * se), float(g + 1.96 * se)


def _electrode_xy(label: str) -> tuple[float, float] | None:
    """Approximate a standard EEG layout for a readable summary dot-map."""
    normalized = str(label).strip()
    match = re.match(r"^([A-Za-z]+)(z|\d+)$", normalized)
    if not match:
        return None
    prefix, number = match.groups()
    prefix = prefix.lower()
    y_rows = {
        "fp": 1.00, "af": 0.82, "f": 0.62, "ft": 0.40,
        "fc": 0.38, "t": 0.08, "c": 0.12, "tp": -0.16,
        "cp": -0.18, "p": -0.47, "po": -0.70, "o": -0.90,
        "i": -1.02,
    }
    if prefix not in y_rows:
        return None
    y = y_rows[prefix]
    if number.lower() == "z":
        return 0.0, y
    index = int(number)
    sign = -1.0 if index % 2 else 1.0
    magnitude = {1: 0.16, 2: 0.16, 3: 0.34, 4: 0.34,
                 5: 0.55, 6: 0.55, 7: 0.78, 8: 0.78,
                 9: 0.96, 10: 0.96}.get(index, 0.96)
    return sign * magnitude, y


def _draw_head(axis: plt.Axes) -> None:
    circle = plt.Circle((0, 0), 1.08, fill=False, color="#333333", linewidth=1.0)
    axis.add_patch(circle)
    axis.plot([-0.14, 0.0, 0.14], [1.08, 1.19, 1.08], color="#333333", linewidth=1.0)
    axis.plot([-1.08, -1.16, -1.08], [0.15, 0.0, -0.15], color="#333333", linewidth=1.0)
    axis.plot([1.08, 1.16, 1.08], [0.15, 0.0, -0.15], color="#333333", linewidth=1.0)
    axis.set(xlim=(-1.25, 1.25), ylim=(-1.25, 1.30), aspect="equal")
    axis.axis("off")


def _summary_values(root: Path, dataset: str, feature: str) -> tuple[float, int, int]:
    all_stats = _read_stats(root, dataset)
    if dataset == "medication_state":
        stats = all_stats.loc[
            all_stats["feature"].eq(feature)
            & all_stats["group_a"].astype(str).eq("Control")
            & all_stats["group_b"].astype(str).str.startswith("PD_")
        ].copy()
        if stats.empty:
            stats = all_stats.loc[
                all_stats["feature"].eq(feature)
                & all_stats["group_b"].astype(str).eq("Control")
                & all_stats["group_a"].astype(str).str.startswith("PD_")
            ].copy()
    else:
        stats = _pd_control_rows(all_stats, feature)
    if stats.empty:
        return np.nan, 0, 0
    if dataset == "medication_state":
        values = [float(row["mean_b"] - row["mean_a"]) if str(row["group_a"]) == "Control" else float(row["mean_a"] - row["mean_b"]) for _, row in stats.iterrows()]
    else:
        recording = _read_recording(root, dataset)
        effects = _standardized_effects(recording, feature)
        values = list(effects.values())
    significant = stats.loc[stats["welch_p_fdr_bh"] < 0.05, "electrode"].astype(str).nunique()
    total = stats["electrode"].astype(str).nunique()
    median_effect = float(np.median(values)) if values else float(np.median([_effect_direction(row) for _, row in stats.iterrows()]))
    return median_effect, int(significant), int(total)


def plot_cross_dataset_summary(root: Path, output: Path) -> None:
    summary_datasets = [*DATASETS, "medication_state"]
    fractions = np.zeros((len(SUMMARY_FEATURES), len(summary_datasets)))
    directions = np.zeros_like(fractions)
    for row_index, (_, feature) in enumerate(SUMMARY_FEATURES):
        for col_index, dataset in enumerate(summary_datasets):
            effect, significant, total = _summary_values(root, dataset, feature)
            fractions[row_index, col_index] = significant / total if total else 0.0
            directions[row_index, col_index] = np.sign(effect)
    fig, (axis, replication_axis) = plt.subplots(1, 2, figsize=(12.0, 6.8), gridspec_kw={"width_ratios": [3.3, 1.3]})
    positive = "#d95f0e"
    negative = "#2b8cbe"
    for i in range(fractions.shape[0]):
        for j in range(fractions.shape[1]):
            fraction = fractions[i, j]
            color = positive if directions[i, j] >= 0 else negative
            rgba = (*to_rgb(color), 0.12 + 0.88 * fraction)
            axis.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=rgba, edgecolor="white", linewidth=2))
            arrow = "↑" if directions[i, j] > 0 else "↓" if directions[i, j] < 0 else "—"
            _, significant, total = _summary_values(root, summary_datasets[j], SUMMARY_FEATURES[i][1])
            axis.text(j, i, f"{arrow} {fraction:.0%}\n{significant}/{total}", ha="center", va="center", fontsize=10)
    axis.set(xlim=(-0.5, len(summary_datasets) - 0.5), ylim=(len(SUMMARY_FEATURES) - 0.5, -0.5))
    axis.set_xticks(range(len(summary_datasets)), ["Medication\nstate" if d == "medication_state" else DATASET_LABELS[d] for d in summary_datasets])
    axis.set_yticks(range(len(SUMMARY_FEATURES)), [name for name, _ in SUMMARY_FEATURES])
    axis.set_title("How much of the scalp replicates?\nArrows: PD vs Control; medication column: PD-OFF/ON vs Control", pad=14)
    broad_replication = (fractions >= 0.50).sum(axis=1)
    y_positions = np.arange(len(SUMMARY_FEATURES))
    replication_axis.barh(y_positions, broad_replication, color="#636363")
    replication_axis.set(xlim=(0, len(summary_datasets) + 0.2), ylim=(len(SUMMARY_FEATURES) - 0.5, -0.5))
    replication_axis.set_xticks(range(len(summary_datasets) + 1))
    replication_axis.set_yticks(y_positions, [])
    replication_axis.set_xlabel("Datasets with ≥50% significant electrodes")
    for y, count in zip(y_positions, broad_replication):
        replication_axis.text(count + 0.08, y, f"{count}/{len(summary_datasets)}", va="center", fontsize=10)
    replication_axis.set_title("Broad replication", pad=14)
    replication_axis.grid(axis="x", alpha=0.2)
    fig.suptitle("Cross-dataset evidence summary", y=0.995, fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_replicated_topomaps(root: Path, output: Path) -> None:
    features = [
        ("Theta relative power", "psd__theta__relative_power", "PD > Control"),
        ("Beta relative power", "psd__beta__relative_power", "PD < Control"),
    ]
    palette = {0: "#e6e6e6", 1: "#fdae61", 2: "#f46d43", 3: "#a50026"}
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.9), squeeze=False)
    all_electrodes = set()
    significant_by_feature: dict[str, dict[str, int]] = {}
    for title, feature, expected_direction in features:
        counts: dict[str, int] = {}
        for dataset in DATASETS:
            rows = _pd_control_rows(_read_stats(root, dataset), feature)
            all_electrodes.update(rows["electrode"].astype(str))
            for _, row in rows.iterrows():
                direction = _effect_direction(row)
                if row["welch_p_fdr_bh"] < 0.05 and ((expected_direction == "PD > Control" and direction > 0) or (expected_direction == "PD < Control" and direction < 0)):
                    electrode = str(row["electrode"])
                    counts[electrode] = counts.get(electrode, 0) + 1
        significant_by_feature[feature] = counts
    for axis, (title, feature, expected_direction) in zip(axes.flat, features):
        _draw_head(axis)
        counts = significant_by_feature[feature]
        for electrode in all_electrodes:
            point = _electrode_xy(electrode)
            if point is None:
                continue
            count = counts.get(electrode, 0)
            axis.scatter([point[0]], [point[1]], s=38 + 32 * count, color=palette[count], edgecolors="#ffffff", linewidths=0.5, zorder=2)
        common = sum(count == 3 for count in counts.values())
        broad = sum(count >= 2 for count in counts.values())
        axis.set_title(f"{title}\n{expected_direction}", pad=10)
        axis.text(0.5, -0.08, f"{common} electrodes significant in all 3 datasets; {broad} in ≥2", transform=axis.transAxes, ha="center", va="top", fontsize=9)
    legend = [Patch(facecolor=palette[count], edgecolor="none", label=f"{count} dataset{'s' if count != 1 else ''}") for count in range(4)]
    fig.legend(handles=legend, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Where do the strongest replicated spectral effects occur?\nColor shows the number of independent datasets with FDR-significant effects", y=1.03, fontsize=14)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.83, bottom=0.18, wspace=0.22)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_theta_bursts(root: Path, output: Path) -> None:
    results: list[dict[str, float | str]] = []
    burst_datasets = [*DATASETS, "medication_state"]
    for dataset in burst_datasets:
        subject = _read_subject(root, dataset)
        for label, feature in BURST_FEATURES:
            if feature not in subject:
                continue
            control = subject.loc[subject["group"].astype(str).eq("Control"), feature].to_numpy(float)
            comparison_groups = ["PD_OFF", "PD_ON"] if dataset == "medication_state" else ["PD"]
            for comparison_group in comparison_groups:
                comparison = subject.loc[subject["group"].astype(str).eq(comparison_group), feature].to_numpy(float)
                estimate, lower, upper = _hedges_g(control, comparison)
                if np.isfinite(estimate):
                    results.append({"metric": label, "dataset": dataset, "group": comparison_group, "estimate": estimate, "lower": lower, "upper": upper})
    frame = pd.DataFrame(results)
    fig, axis = plt.subplots(figsize=(10.2, 5.5))
    offsets = np.linspace(-0.27, 0.27, len(burst_datasets))
    metric_names = [label for label, _ in BURST_FEATURES]
    for metric_index, metric in enumerate(metric_names):
        subset = frame.loc[frame["metric"].eq(metric)]
        for dataset_index, dataset in enumerate(burst_datasets):
            row = subset.loc[subset["dataset"].eq(dataset)]
            for _, item in row.iterrows():
                y = metric_index + offsets[dataset_index]
                color = GROUP_COLORS[item["group"]] if dataset == "medication_state" else DATASET_COLORS[dataset]
                marker = "s" if item["group"] == "PD_ON" else "o"
                axis.errorbar(item["estimate"], y, xerr=[[item["estimate"] - item["lower"]], [item["upper"] - item["estimate"]]], fmt=marker, color=color, capsize=3, markersize=6)
    axis.axvline(0, color="#333333", linewidth=1.0)
    axis.set_yticks(range(len(metric_names)), metric_names)
    axis.set_xlabel("Hedges g (patient group − Control), approximate 95% CI")
    axis.set_title("Theta-burst effects across datasets\nMedication panel: PD-OFF/ON versus Control")
    axis.grid(axis="x", alpha=0.2)
    handles = [Line2D([], [], marker="o", linestyle="none", color=DATASET_COLORS[d], label=DATASET_LABELS[d]) for d in DATASETS]
    handles.extend([
        Line2D([], [], marker="o", linestyle="none", color=GROUP_COLORS["PD_OFF"], label="Medication: PD-OFF"),
        Line2D([], [], marker="s", linestyle="none", color=GROUP_COLORS["PD_ON"], label="Medication: PD-ON"),
    ])
    axis.legend(handles=handles, frameon=False, loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _bh_adjust(values: list[float]) -> list[float]:
    if not values:
        return []
    array = np.asarray(values, dtype=float)
    order = np.argsort(array)
    adjusted = array[order] * len(array) / np.arange(1, len(array) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1.0)
    return result.tolist()


def _corrected_group_pvalues(frame: pd.DataFrame, feature_specs: list[tuple[str, str]], groups: list[str]) -> dict[tuple[str, str, str], float]:
    raw: dict[tuple[str, str, str], float] = {}
    for _, feature in feature_specs:
        for left_index, left_group in enumerate(groups[:-1]):
            left_values = pd.to_numeric(frame.loc[frame["group"].astype(str).eq(left_group), feature], errors="coerce").dropna().to_numpy(float)
            for right_group in groups[left_index + 1:]:
                right_values = pd.to_numeric(frame.loc[frame["group"].astype(str).eq(right_group), feature], errors="coerce").dropna().to_numpy(float)
                if len(left_values) < 2 or len(right_values) < 2:
                    continue
                p_value = float(ttest_ind(left_values, right_values, equal_var=False).pvalue)
                if np.isfinite(p_value):
                    raw[(feature, left_group, right_group)] = p_value
    adjusted = _bh_adjust(list(raw.values()))
    return dict(zip(raw, adjusted))


def _plot_distribution_cell(axis: plt.Axes, frame: pd.DataFrame, feature: str, groups: list[str]) -> None:
    plotted = False
    for position, group in enumerate(groups, start=1):
        values = pd.to_numeric(frame.loc[frame["group"].astype(str).eq(group), feature], errors="coerce").dropna().to_numpy(float)
        if values.size == 0:
            continue
        plotted = True
        if values.size > 1 and not np.isclose(values.min(), values.max()):
            violin = axis.violinplot(values, positions=[position], widths=0.72, showmeans=False, showmedians=True, showextrema=False)
            for body in violin["bodies"]:
                body.set_facecolor(GROUP_COLORS[group])
                body.set_edgecolor(GROUP_COLORS[group])
                body.set_alpha(0.52)
            violin["cmedians"].set_color("#222222")
        jitter = np.linspace(-0.09, 0.09, values.size)
        axis.scatter(np.full(values.size, position) + jitter, values, s=12, color=GROUP_COLORS[group], alpha=0.65, edgecolor="white", linewidth=0.25, zorder=3)
    if plotted:
        axis.set_xticks(range(1, len(groups) + 1), [group.replace("_", " ") for group in groups])
        axis.tick_params(axis="x", labelrotation=35, labelsize=8)
        axis.grid(axis="y", alpha=0.18)


def _add_significance_bars(axis: plt.Axes, significant_pairs: list[tuple[int, int]]) -> None:
    if not significant_pairs:
        return
    lower, upper = axis.get_ylim()
    span = max(upper - lower, np.finfo(float).eps)
    axis.set_ylim(lower, upper + (0.15 + 0.10 * len(significant_pairs)) * span)
    bar_y = upper + 0.07 * span
    cap_height = 0.035 * span
    for index, (left, right) in enumerate(significant_pairs):
        level = bar_y + index * 0.10 * span
        axis.plot([left, left, right, right], [level, level + cap_height, level + cap_height, level], color="#111111", linewidth=1.2, clip_on=False)
        axis.text((left + right) / 2, level + cap_height + 0.012 * span, "★", ha="center", va="bottom", fontsize=13, color="#111111", clip_on=False)


def _set_column_limits(axes: np.ndarray, frames: dict[str, pd.DataFrame], feature_specs: list[tuple[str, str]]) -> None:
    for column, (_, feature) in enumerate(feature_specs):
        values = []
        for frame in frames.values():
            if feature in frame:
                values.extend(pd.to_numeric(frame[feature], errors="coerce").dropna().to_numpy(float))
        if not values:
            continue
        lower, upper = np.nanpercentile(values, [1, 99])
        padding = max((upper - lower) * 0.10, np.finfo(float).eps)
        for row in range(axes.shape[0]):
            axes[row, column].set_ylim(lower - padding, upper + padding)


def plot_cross_dataset_distributions(root: Path, output: Path) -> None:
    feature_specs = [
        ("Theta relative power", "psd__theta__relative_power"),
        ("Theta burst count", "bout__theta__n_bouts"),
        ("Theta burst occupancy", "bout__theta__oscillatory_occupancy"),
        ("Beta relative power", "psd__beta__relative_power"),
        ("Alpha entropy (D=3)", "entropy__alpha__entropy__D3"),
    ]
    distribution_datasets = [*DATASETS, "medication_state"]
    frames = {dataset: _read_subject(root, dataset) for dataset in distribution_datasets}
    fig, axes = plt.subplots(len(distribution_datasets), len(feature_specs), figsize=(16.0, 10.2), squeeze=False)
    significant_cells: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for row, dataset in enumerate(distribution_datasets):
        frame = frames[dataset]
        groups = _groups_for_dataset(dataset)
        q_values = _corrected_group_pvalues(frame, feature_specs, groups)
        for column, (label, feature) in enumerate(feature_specs):
            significant_cells[(row, column)] = [
                (left_index + 1, right_index + 1)
                for left_index, left_group in enumerate(groups[:-1])
                for right_index, right_group in enumerate(groups[left_index + 1:], start=left_index + 1)
                if q_values.get((feature, left_group, right_group), np.nan) < 0.05
            ]
            _plot_distribution_cell(axes[row, column], frame, feature, groups)
            if row == 0:
                axes[row, column].set_title(label, fontsize=10)
            if column == 0:
                axes[row, column].text(-0.34, 0.5, DATASET_LABELS[dataset], transform=axes[row, column].transAxes, rotation=90, va="center", ha="right", fontsize=11, fontweight="bold")
            if row == len(distribution_datasets) - 1:
                axes[row, column].set_xlabel("Group")
            if column == 0:
                axes[row, column].set_ylabel("Value")
    _set_column_limits(axes, frames, feature_specs)
    for (row, column), significant_pairs in significant_cells.items():
        _add_significance_bars(axes[row, column], significant_pairs)
    fig.suptitle("Main quantities: participant-level distributions", y=0.985, fontsize=15)
    fig.text(0.5, 0.935, "★ = participant-level Welch BH-FDR q < 0.05 within each dataset; medication groups are Control, PD-OFF, and PD-ON", ha="center", fontsize=10)
    fig.legend(handles=[Line2D([], [], marker="o", linestyle="none", color=GROUP_COLORS[group], label=group.replace("_", " ")) for group in ["Control", "PD", "PD_OFF", "PD_ON"]], loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0.04, 0.04, 1, 0.89), h_pad=1.8, w_pad=1.0)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_theta_hcf_distributions(root: Path, output: Path) -> None:
    feature_specs = [
        ("H: entropy", "entropy__theta__entropy__D4"),
        ("C: complexity", "entropy__theta__complexity__D4"),
        ("F: Fisher information", "entropy__theta__fisher_information__D4"),
    ]
    distribution_datasets = [*DATASETS, "medication_state"]
    frames = {dataset: _read_subject(root, dataset) for dataset in distribution_datasets}
    fig, axes = plt.subplots(len(distribution_datasets), len(feature_specs), figsize=(10.5, 10.2), squeeze=False)
    significant_cells: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for row, dataset in enumerate(distribution_datasets):
        frame = frames[dataset]
        groups = _groups_for_dataset(dataset)
        q_values = _corrected_group_pvalues(frame, feature_specs, groups)
        for column, (label, feature) in enumerate(feature_specs):
            significant_cells[(row, column)] = [
                (left_index + 1, right_index + 1)
                for left_index, left_group in enumerate(groups[:-1])
                for right_index, right_group in enumerate(groups[left_index + 1:], start=left_index + 1)
                if q_values.get((feature, left_group, right_group), np.nan) < 0.05
            ]
            _plot_distribution_cell(axes[row, column], frame, feature, groups)
            if row == 0:
                axes[row, column].set_title(label, fontsize=11)
            if column == 0:
                axes[row, column].text(-0.28, 0.5, DATASET_LABELS[dataset], transform=axes[row, column].transAxes, rotation=90, va="center", ha="right", fontsize=11, fontweight="bold")
            if row == len(distribution_datasets) - 1:
                axes[row, column].set_xlabel("Group")
            if column == 0:
                axes[row, column].set_ylabel("Value")
    _set_column_limits(axes, frames, feature_specs)
    for (row, column), significant_pairs in significant_cells.items():
        _add_significance_bars(axes[row, column], significant_pairs)
    fig.suptitle("Theta-band H/C/F distributions at embedding dimension D=4", y=0.985, fontsize=15)
    fig.text(0.5, 0.935, "H = entropy, C = complexity, F = Fisher information; ★ = participant-level Welch BH-FDR q < 0.05", ha="center", fontsize=10)
    fig.legend(handles=[Line2D([], [], marker="o", linestyle="none", color=GROUP_COLORS[group], label=group.replace("_", " ")) for group in ["Control", "PD", "PD_OFF", "PD_ON"]], loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0.04, 0.04, 1, 0.89), h_pad=1.8, w_pad=1.2)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_psd_side_by_side(root: Path, output: Path) -> None:
    """Combine the already-rendered full PSD comparisons into one figure."""
    paths = [
        root / "figures" / dataset / "psd_broadband_mean_ci.png"
        for dataset in [*DATASETS, "medication_state"]
    ]
    images = [Image.open(path).convert("RGB") for path in paths if path.exists()]
    if not images:
        return
    width = sum(image.width for image in images)
    height = max(image.height for image in images)
    canvas = Image.new("RGB", (width, height), "white")
    x_offset = 0
    for image in images:
        canvas.paste(image, (x_offset, 0))
        x_offset += image.width
    canvas.save(output)


def plot_age_group_histograms(root: Path, output: Path) -> None:
    age_datasets = [*DATASETS, "medication_state"]
    age_groups = ["Control", "PD", "PD_OFF", "PD_ON"]
    age_colors = {"Control": "#7f7f7f", "PD": "#d95f02", "PD_OFF": "#7570b3", "PD_ON": "#1b9e77"}
    frames = {dataset: _read_subject(root, dataset) for dataset in age_datasets}
    ages = []
    for frame in frames.values():
        ages.extend(pd.to_numeric(frame["age_years"], errors="coerce").dropna().to_numpy(float))
    if not ages:
        return
    minimum = int(np.floor(np.min(ages) / 5.0) * 5)
    maximum = int(np.ceil(np.max(ages) / 5.0) * 5 + 5)
    bins = np.arange(minimum, maximum + 1, 5)
    counts_by_dataset: dict[str, dict[str, np.ndarray]] = {}
    maximum_count = 0
    for dataset, frame in frames.items():
        counts_by_dataset[dataset] = {}
        for group in age_groups:
            values = pd.to_numeric(frame.loc[frame["group"].astype(str).eq(group), "age_years"], errors="coerce").dropna().to_numpy(float)
            counts = np.histogram(values, bins=bins)[0]
            counts_by_dataset[dataset][group] = counts
            maximum_count = max(maximum_count, int(counts.max(initial=0)))
    width = (bins[1] - bins[0]) * 0.38
    centers = bins[:-1] + (bins[1] - bins[0]) / 2
    fig, axes = plt.subplots(1, len(age_datasets), figsize=(16.5, 4.6), sharex=True, sharey=False)
    if len(age_datasets) == 1:
        axes = [axes]
    for axis, dataset in zip(axes, age_datasets):
        frame = frames[dataset]
        present_groups = [group for group in age_groups if counts_by_dataset[dataset][group].sum() > 0]
        group_offsets = np.linspace(-width * (len(present_groups) - 1) / 2, width * (len(present_groups) - 1) / 2, len(present_groups))
        for offset, group in zip(group_offsets, present_groups):
            counts = counts_by_dataset[dataset][group]
            axis.bar(centers + offset, counts, width=width, color=age_colors[group], alpha=0.82, label=f"{group.replace('_', ' ')} (n={int(counts.sum())})", edgecolor="white", linewidth=0.4)
        axis.set_title(DATASET_LABELS[dataset])
        axis.set_xlabel("Age (years)")
        axis.set_xticks(bins[::2])
        local_maximum = max(int(counts_by_dataset[dataset][group].max(initial=0)) for group in present_groups)
        axis.set_ylim(0, local_maximum * 1.18 + 1)
        axis.grid(axis="y", alpha=0.2)
        axis.legend(frameon=False, fontsize=9)
        axis.set_ylabel("Number of participants")
    fig.suptitle("Participant age distributions across all four datasets", y=0.99, fontsize=15)
    fig.text(0.5, 0.93, "Common 5-year bins; each participant contributes once; y-scales adapt to cohort size", ha="center", fontsize=10)
    fig.tight_layout(rect=(0, 0.02, 1, 0.86), w_pad=1.4)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _clinical_subject_frame(root: Path, dataset: str, feature: str, outcome: str, group: str | None = None) -> pd.DataFrame:
    recording = _read_recording(root, dataset)
    group_values = recording["group"].astype(str)
    group_mask = group_values.eq(group) if group is not None else group_values.str.startswith("PD")
    selected = recording.loc[
        group_mask,
        ["participant_id", "group", feature],
    ].copy()
    selected = selected.groupby(["participant_id", "group"], dropna=False)[feature].mean().reset_index()
    metadata = recording.loc[
        group_mask,
        ["participant_id", "group", outcome],
    ].drop_duplicates(["participant_id", "group"])
    selected = selected.merge(metadata, on=["participant_id", "group"], how="left")
    selected[feature] = pd.to_numeric(selected[feature], errors="coerce")
    selected[outcome] = pd.to_numeric(selected[outcome], errors="coerce")
    return selected.dropna(subset=[feature, outcome])


def plot_clinical_replication(root: Path, output: Path) -> None:
    feature = "entropy__theta__fisher_information__D4"
    frames: dict[str, pd.DataFrame] = {}
    for dataset in DATASETS:
        frames[dataset] = _clinical_subject_frame(root, dataset, feature, "moca")
    # The medication-state cohort has MMSE rather than MoCA in the saved
    # clinical table. Keep it as a fourth panel, with its outcome labeled
    # explicitly instead of silently mixing scales.
    frames["medication_state"] = _clinical_subject_frame(root, "medication_state", feature, "mmse")
    outcomes = {dataset: ("mmse" if dataset == "medication_state" else "moca") for dataset in frames}
    all_points = pd.concat([frame.assign(_outcome=outcomes[dataset]) for dataset, frame in frames.items()], ignore_index=True)
    x_limits = np.nanpercentile(all_points[feature], [1, 99])
    y_values = np.concatenate([frame[outcome].to_numpy(float) for dataset, frame in frames.items() for outcome in [outcomes[dataset]]])
    y_limits = np.nanpercentile(y_values, [1, 99])
    x_pad = max((x_limits[1] - x_limits[0]) * 0.08, 1e-9)
    y_pad = max((y_limits[1] - y_limits[0]) * 0.08, 1e-9)
    x_limits = (x_limits[0] - x_pad, x_limits[1] + x_pad)
    y_limits = (y_limits[0] - y_pad, y_limits[1] + y_pad)
    all_datasets = [*DATASETS, "medication_state"]
    fig, axes = plt.subplots(1, len(all_datasets), figsize=(15.8, 3.9), sharex=True, sharey=True)
    if len(DATASETS) == 1:
        axes = [axes]
    for axis, dataset in zip(axes, all_datasets):
        points = frames[dataset]
        outcome = outcomes[dataset]
        x = points[feature].to_numpy(float)
        y = points[outcome].to_numpy(float)
        if dataset == "medication_state":
            for group, marker in [("PD_OFF", "o"), ("PD_ON", "s")]:
                group_points = points.loc[points["group"].eq(group)]
                axis.scatter(group_points[feature], group_points[outcome], s=28, alpha=0.78, color=GROUP_COLORS[group], marker=marker, edgecolor="white", linewidth=0.35, label=group)
        else:
            axis.scatter(x, y, s=28, alpha=0.78, color=DATASET_COLORS[dataset], edgecolor="white", linewidth=0.35)
        if len(points) >= 3 and np.unique(x).size > 1:
            slope, intercept = np.polyfit(x, y, 1)
            grid = np.linspace(x_limits[0], x_limits[1], 80)
            axis.plot(grid, intercept + slope * grid, color="#222222", linewidth=1.2)
            rho, p_value = spearmanr(x, y)
        else:
            rho, p_value = np.nan, np.nan
        clinical = pd.read_csv(root / "statistics" / dataset / "clinical_correlations.csv.gz", low_memory=False)
        row = clinical.loc[(clinical["method"].eq("spearman_unadjusted")) & clinical["outcome"].eq(outcome) & clinical["feature"].eq(feature)]
        q_value = float(row["p_fdr_bh"].min()) if not row.empty else np.nan
        q_text = (f"q={q_value:.4f}" if dataset != "medication_state" else f"min group q={q_value:.4f}") if np.isfinite(q_value) else "q=n/a"
        axis.text(0.04, 0.96, f"n={len(points)}\nρ={rho:.3f}\n{q_text}", transform=axis.transAxes, va="top", fontsize=9, bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"})
        axis.set_title(DATASET_LABELS[dataset])
        axis.set_xlim(x_limits)
        axis.set_ylim(y_limits)
        axis.grid(alpha=0.2)
        axis.set_xlabel("Theta Fisher information (D=4)")
        axis.set_ylabel("MMSE" if outcome == "mmse" else "MoCA")
    axes[0].set_ylabel("MoCA")
    axes[-1].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle("Theta Fisher information and clinical scores\nMoCA in the first three datasets; MMSE in the medication-state cohort", y=1.04, fontsize=14)
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _best_within_bout_result(root: Path, dataset: str, outcome: str) -> pd.Series | None:
    clinical = pd.read_csv(root / "statistics" / dataset / "clinical_correlations.csv.gz", low_memory=False)
    selected = clinical.loc[
        clinical["method"].eq("spearman_unadjusted")
        & clinical["outcome"].eq(outcome)
        & clinical["feature"].astype(str).str.startswith("within_bout__")
    ].dropna(subset=["p_fdr_bh", "rho"])
    if selected.empty:
        return None
    selected = selected.assign(abs_rho=selected["rho"].abs()).sort_values(["p_fdr_bh", "abs_rho"], ascending=[True, False])
    return selected.iloc[0]


def _clinical_feature_label(feature: str) -> str:
    known = {
        "within_bout__theta__fisher_information__D7": "Theta within-bout Fisher information (D=7)",
        "within_bout__delta__complexity__D4": "Delta within-bout complexity (D=4)",
        "within_bout__theta__fisher_information__D6": "Theta within-bout Fisher information (D=6)",
        "within_bout__gamma__fisher_information__D4": "Gamma within-bout Fisher information (D=4)",
    }
    return known.get(feature, feature.replace("within_bout__", "").replace("__", " / ").replace("_", " "))


def plot_within_bout_clinical(root: Path, output: Path) -> None:
    targets = [
        ("primary", "moca"),
        ("ds007526-1.0.2", "moca"),
        ("ds008768-1.0.0", "updrs"),
        ("medication_state", "mmse"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.2), squeeze=False)
    for axis, (dataset, outcome) in zip(axes.flat, targets):
        result = _best_within_bout_result(root, dataset, outcome)
        if result is None:
            axis.axis("off")
            continue
        feature = str(result["feature"])
        group = str(result["group"])
        points = _clinical_subject_frame(root, dataset, feature, outcome, group=group)
        x = points[feature].to_numpy(float)
        y = points[outcome].to_numpy(float)
        significant = float(result["p_fdr_bh"]) < 0.05
        color = "#238b45" if significant else "#969696"
        axis.scatter(x, y, s=28, alpha=0.78, color=color, edgecolor="white", linewidth=0.35)
        if len(points) >= 3 and np.unique(x).size > 1:
            slope, intercept = np.polyfit(x, y, 1)
            grid = np.linspace(np.min(x), np.max(x), 80)
            axis.plot(grid, intercept + slope * grid, color="#222222", linewidth=1.1)
        status = "FDR-significant" if significant else "not FDR-significant"
        axis.text(0.04, 0.96, f"group={group}\nn={len(points)}\nρ={float(result['rho']):.3f}\nq={float(result['p_fdr_bh']):.4f}\n{status}", transform=axis.transAxes, va="top", fontsize=9, bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"})
        axis.set_title(f"{DATASET_LABELS.get(dataset, dataset)} — {outcome.upper()}\n{_clinical_feature_label(feature)}", fontsize=11)
        axis.set_xlabel(_clinical_feature_label(feature))
        axis.set_ylabel(outcome.upper())
        axis.grid(alpha=0.2)
    fig.suptitle("Within-bout clinical associations\nGreen: strongest saved association survives FDR; gray: strongest available but not significant", y=0.995, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_strongest_updrs(root: Path, output: Path) -> None:
    candidates: list[tuple[float, str, str, float, float, int]] = []
    for dataset in DATASETS:
        clinical = pd.read_csv(root / "statistics" / dataset / "clinical_correlations.csv.gz", low_memory=False)
        selected = clinical.loc[
            clinical["method"].eq("spearman_unadjusted")
            & clinical["outcome"].eq("updrs")
        ].dropna(subset=["p_fdr_bh", "rho"])
        for _, row in selected.iterrows():
            candidates.append((float(row["p_fdr_bh"]), dataset, str(row["feature"]), float(row["rho"]), float(row["p_value"]), int(row["n_subjects"])))
    if not candidates:
        return
    q_value, dataset, feature, table_rho, p_value, table_n = min(candidates, key=lambda item: item[0])
    points = _clinical_subject_frame(root, dataset, feature, "updrs")
    x = points[feature].to_numpy(float)
    y = points["updrs"].to_numpy(float)
    rho = float(spearmanr(x, y).statistic)
    feature_label = {
        "bout__beta__oscillatory_occupancy": "Beta burst occupancy",
        "bout__theta__duration_mean_s": "Theta burst duration",
    }.get(feature, feature.replace("__", " / ").replace("_", " "))
    fig, axis = plt.subplots(figsize=(7.0, 5.0))
    axis.scatter(x, y, s=34, alpha=0.8, color="#7b3294", edgecolor="white", linewidth=0.4)
    if len(points) >= 3 and np.unique(x).size > 1:
        slope, intercept = np.polyfit(x, y, 1)
        grid = np.linspace(np.min(x), np.max(x), 80)
        axis.plot(grid, intercept + slope * grid, color="#222222", linewidth=1.2)
    axis.text(0.04, 0.96, f"n={len(points)}\nρ={rho:.3f}\np={p_value:.4g}\nq={q_value:.4f}", transform=axis.transAxes, va="top", fontsize=10, bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"})
    axis.set_xlabel(feature_label)
    axis.set_ylabel("UPDRS-III")
    axis.set_title(f"Strongest saved UPDRS association\n{DATASET_LABELS[dataset]}: {feature_label}")
    axis.text(0.5, -0.19, "Exploratory single-dataset result; not a replicated clinical biomarker", transform=axis.transAxes, ha="center", fontsize=9, color="#555555")
    axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_medication(root: Path, output: Path) -> None:
    subject = _read_subject(root, "medication_state")
    results: list[dict[str, float | str]] = []
    for label, feature in MEDICATION_FEATURES:
        if feature not in subject:
            continue
        off = subject.loc[subject["group"].astype(str).eq("PD_OFF"), feature].to_numpy(float)
        on = subject.loc[subject["group"].astype(str).eq("PD_ON"), feature].to_numpy(float)
        estimate, lower, upper = _hedges_g(off, on)
        if np.isfinite(estimate):
            results.append({"metric": label, "estimate": estimate, "lower": lower, "upper": upper})
    frame = pd.DataFrame(results)
    fig, axis = plt.subplots(figsize=(8.8, 5.0))
    y = np.arange(len(frame))
    for index, row in frame.iterrows():
        axis.errorbar(row["estimate"], index, xerr=[[row["estimate"] - row["lower"]], [row["upper"] - row["estimate"]]], fmt="o", color="#7b3294", capsize=3, markersize=6)
    axis.axvline(0, color="#333333", linewidth=1.0)
    axis.set_yticks(y, frame["metric"])
    axis.set_xlabel("Hedges g (PD-ON − PD-OFF), approximate 95% CI")
    axis.set_title("Medication-state effects are uncertain in the small cohort")
    axis.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/global"))
    args = parser.parse_args()
    root = args.output_root
    output = root / "figures" / "summary"
    output.mkdir(parents=True, exist_ok=True)
    plot_cross_dataset_summary(root, output / "cross_dataset_summary.png")
    plot_cross_dataset_distributions(root, output / "cross_dataset_distributions.png")
    plot_theta_hcf_distributions(root, output / "theta_hcf_D4_distributions.png")
    plot_psd_side_by_side(root, output / "psd_control_pd_side_by_side.png")
    plot_age_group_histograms(root, output / "age_group_histograms.png")
    plot_replicated_topomaps(root, output / "replicated_spectral_topomaps.png")
    plot_theta_bursts(root, output / "theta_burst_effects.png")
    plot_clinical_replication(root, output / "theta_fisher_moca_replication.png")
    plot_within_bout_clinical(root, output / "within_bout_clinical_associations.png")
    plot_strongest_updrs(root, output / "updrs_strongest_association.png")
    plot_medication(root, output / "medication_state_effects.png")
    print(f"Wrote summary figures to {output}")


if __name__ == "__main__":
    main()
