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
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from scipy.stats import spearmanr


DATASETS = ["primary", "ds007526-1.0.2", "ds008768-1.0.0"]
DATASET_LABELS = {
    "primary": "Primary",
    "ds007526-1.0.2": "ds007526",
    "ds008768-1.0.0": "ds008768",
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
    ("Aperiodic offset", "aperiodic__broadband__aperiodic_offset"),
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
    ("Aperiodic offset", "aperiodic__broadband__aperiodic_offset"),
]


def _read(root: Path, dataset: str, kind: str) -> pd.DataFrame:
    path = root / kind / dataset / f"{kind.rstrip('s')}_features.csv.gz"
    # The directory names are metrics and the filenames are singular in the
    # saved outputs; keep the explicit fallback for clarity and robustness.
    if kind == "metrics":
        path = root / "metrics" / dataset / "subject_features.csv.gz"
    return pd.read_csv(path)


def _read_subject(root: Path, dataset: str) -> pd.DataFrame:
    return pd.read_csv(root / "metrics" / dataset / "subject_features.csv.gz")


def _read_recording(root: Path, dataset: str) -> pd.DataFrame:
    return pd.read_csv(root / "metrics" / dataset / "recording_features.csv.gz")


def _read_stats(root: Path, dataset: str) -> pd.DataFrame:
    return pd.read_csv(root / "statistics" / dataset / "group_statistics.csv.gz")


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
    stats = _pd_control_rows(_read_stats(root, dataset), feature)
    if stats.empty:
        return np.nan, 0, 0
    recording = _read_recording(root, dataset)
    effects = _standardized_effects(recording, feature)
    values = list(effects.values())
    significant = stats.loc[stats["welch_p_fdr_bh"] < 0.05, "electrode"].astype(str).nunique()
    total = stats["electrode"].astype(str).nunique()
    median_effect = float(np.median(values)) if values else float(np.median([_effect_direction(row) for _, row in stats.iterrows()]))
    return median_effect, int(significant), int(total)


def plot_cross_dataset_summary(root: Path, output: Path) -> None:
    values = np.full((len(SUMMARY_FEATURES), len(DATASETS)), np.nan)
    labels: list[list[str]] = [["" for _ in DATASETS] for _ in SUMMARY_FEATURES]
    for row_index, (_, feature) in enumerate(SUMMARY_FEATURES):
        for col_index, dataset in enumerate(DATASETS):
            effect, significant, total = _summary_values(root, dataset, feature)
            values[row_index, col_index] = effect
            labels[row_index][col_index] = f"{effect:+.2f}\n{significant}/{total}"
    fig, axis = plt.subplots(figsize=(9.4, 6.8))
    norm = TwoSlopeNorm(vmin=-1.5, vcenter=0.0, vmax=1.5)
    image = axis.imshow(values, cmap="RdBu_r", norm=norm, aspect="auto")
    axis.set_xticks(range(len(DATASETS)), [DATASET_LABELS[d] for d in DATASETS])
    axis.set_yticks(range(len(SUMMARY_FEATURES)), [name for name, _ in SUMMARY_FEATURES])
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            text_color = "white" if np.isfinite(values[i, j]) and abs(values[i, j]) > 0.65 else "black"
            axis.text(j, i, labels[i][j], ha="center", va="center", color=text_color, fontsize=9)
    axis.set_title("Cross-dataset replication summary\nColor: median standardized PD–Control effect; text: FDR-significant electrodes / tested electrodes", pad=14)
    colorbar = fig.colorbar(image, ax=axis, fraction=0.045, pad=0.04)
    colorbar.set_label("Standardized effect (PD − Control)")
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_replicated_topomaps(root: Path, output: Path) -> None:
    features = [("Theta relative power", "psd__theta__relative_power"), ("Beta relative power", "psd__beta__relative_power")]
    effects: dict[tuple[str, str], dict[str, float]] = {}
    significant: dict[tuple[str, str], set[str]] = {}
    max_value = 0.0
    for dataset in DATASETS:
        stats = _read_stats(root, dataset)
        recording = _read_recording(root, dataset)
        for _, feature in features:
            effects[(dataset, feature)] = _standardized_effects(recording, feature)
            rows = _pd_control_rows(stats, feature)
            significant[(dataset, feature)] = set(rows.loc[rows["welch_p_fdr_bh"] < 0.05, "electrode"].astype(str))
            if effects[(dataset, feature)]:
                max_value = max(max_value, max(abs(value) for value in effects[(dataset, feature)].values()))
    max_value = max(0.6, min(2.0, max_value))
    norm = TwoSlopeNorm(vmin=-max_value, vcenter=0.0, vmax=max_value)
    fig, axes = plt.subplots(len(DATASETS), 2, figsize=(8.0, 9.0), squeeze=False)
    for row, dataset in enumerate(DATASETS):
        for col, (title, feature) in enumerate(features):
            axis = axes[row, col]
            _draw_head(axis)
            data = effects[(dataset, feature)]
            xy = [(point[0], point[1]) for electrode in data if (point := _electrode_xy(electrode)) is not None]
            colors = [data[electrode] for electrode in data if _electrode_xy(electrode) is not None]
            if xy:
                axis.scatter([point[0] for point in xy], [point[1] for point in xy], c=colors, cmap="RdBu_r", norm=norm, s=105, edgecolors="#ffffff", linewidths=0.45, zorder=2)
            for electrode in significant[(dataset, feature)]:
                point = _electrode_xy(electrode)
                if point is not None:
                    axis.scatter([point[0]], [point[1]], s=145, facecolors="none", edgecolors="#111111", linewidths=1.25, zorder=3)
            if row == 0:
                axis.set_title(title, pad=10)
            if col == 0:
                axis.text(-0.08, 0.5, DATASET_LABELS[dataset], transform=axis.transAxes, rotation=90, va="center", ha="right", fontsize=11, fontweight="bold")
    scalar = plt.cm.ScalarMappable(norm=norm, cmap="RdBu_r")
    scalar.set_array([])
    fig.colorbar(scalar, ax=axes, fraction=0.025, pad=0.02, label="Standardized PD − Control effect")
    fig.suptitle("Replicated spectral topographies\nBlack outlines mark electrode-wise Welch BH-FDR q < 0.05", y=0.995, fontsize=14)
    fig.tight_layout(rect=(0.04, 0.02, 0.95, 0.96), h_pad=1.8, w_pad=1.3)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_theta_bursts(root: Path, output: Path) -> None:
    results: list[dict[str, float | str]] = []
    for dataset in DATASETS:
        subject = _read_subject(root, dataset)
        for label, feature in BURST_FEATURES:
            if feature not in subject:
                continue
            control = subject.loc[subject["group"].astype(str).eq("Control"), feature].to_numpy(float)
            pd_values = subject.loc[subject["group"].astype(str).eq("PD"), feature].to_numpy(float)
            estimate, lower, upper = _hedges_g(control, pd_values)
            if np.isfinite(estimate):
                results.append({"metric": label, "dataset": dataset, "estimate": estimate, "lower": lower, "upper": upper})
    frame = pd.DataFrame(results)
    fig, axis = plt.subplots(figsize=(9.0, 5.5))
    offsets = np.linspace(-0.24, 0.24, len(DATASETS))
    metric_names = [label for label, _ in BURST_FEATURES]
    for metric_index, metric in enumerate(metric_names):
        subset = frame.loc[frame["metric"].eq(metric)]
        for dataset_index, dataset in enumerate(DATASETS):
            row = subset.loc[subset["dataset"].eq(dataset)]
            if row.empty:
                continue
            item = row.iloc[0]
            y = metric_index + offsets[dataset_index]
            axis.errorbar(item["estimate"], y, xerr=[[item["estimate"] - item["lower"]], [item["upper"] - item["estimate"]]], fmt="o", color=DATASET_COLORS[dataset], capsize=3, markersize=6)
    axis.axvline(0, color="#333333", linewidth=1.0)
    axis.set_yticks(range(len(metric_names)), metric_names)
    axis.set_xlabel("Hedges g (PD − Control), approximate 95% CI")
    axis.set_title("Theta-burst effects are directionally consistent across datasets")
    axis.grid(axis="x", alpha=0.2)
    axis.legend(handles=[Line2D([], [], marker="o", linestyle="none", color=DATASET_COLORS[d], label=DATASET_LABELS[d]) for d in DATASETS], frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_clinical_replication(root: Path, output: Path) -> None:
    feature = "entropy__theta__fisher_information__D4"
    frames: dict[str, pd.DataFrame] = {}
    for dataset in DATASETS:
        subject = _read_subject(root, dataset)
        selected = subject.loc[subject["group"].astype(str).str.startswith("PD")].copy()
        selected[feature] = pd.to_numeric(selected[feature], errors="coerce")
        selected["moca"] = pd.to_numeric(selected["moca"], errors="coerce")
        frames[dataset] = selected.dropna(subset=[feature, "moca"])
    all_points = pd.concat(frames.values(), ignore_index=True)
    x_limits = np.nanpercentile(all_points[feature], [1, 99])
    y_limits = np.nanpercentile(all_points["moca"], [1, 99])
    x_pad = max((x_limits[1] - x_limits[0]) * 0.08, 1e-9)
    y_pad = max((y_limits[1] - y_limits[0]) * 0.08, 1e-9)
    x_limits = (x_limits[0] - x_pad, x_limits[1] + x_pad)
    y_limits = (y_limits[0] - y_pad, y_limits[1] + y_pad)
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(12.2, 3.9), sharex=True, sharey=True)
    if len(DATASETS) == 1:
        axes = [axes]
    for axis, dataset in zip(axes, DATASETS):
        points = frames[dataset]
        x = points[feature].to_numpy(float)
        y = points["moca"].to_numpy(float)
        axis.scatter(x, y, s=28, alpha=0.78, color=DATASET_COLORS[dataset], edgecolor="white", linewidth=0.35)
        if len(points) >= 3 and np.unique(x).size > 1:
            slope, intercept = np.polyfit(x, y, 1)
            grid = np.linspace(x_limits[0], x_limits[1], 80)
            axis.plot(grid, intercept + slope * grid, color="#222222", linewidth=1.2)
            rho, p_value = spearmanr(x, y)
        else:
            rho, p_value = np.nan, np.nan
        clinical = pd.read_csv(root / "statistics" / dataset / "clinical_correlations.csv.gz")
        row = clinical.loc[(clinical["method"].eq("spearman_unadjusted")) & clinical["outcome"].eq("moca") & clinical["feature"].eq(feature)]
        q_value = float(row["p_fdr_bh"].iloc[0]) if not row.empty else np.nan
        axis.text(0.04, 0.96, f"n={len(points)}\nρ={rho:.3f}\nq={q_value:.4f}", transform=axis.transAxes, va="top", fontsize=9, bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"})
        axis.set_title(DATASET_LABELS[dataset])
        axis.set_xlim(x_limits)
        axis.set_ylim(y_limits)
        axis.grid(alpha=0.2)
        axis.set_xlabel("Theta Fisher information (D=4)")
    axes[0].set_ylabel("MoCA")
    fig.suptitle("Replicated clinical association in PD participants", y=1.02, fontsize=14)
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
    plot_replicated_topomaps(root, output / "replicated_spectral_topomaps.png")
    plot_theta_bursts(root, output / "theta_burst_effects.png")
    plot_clinical_replication(root, output / "theta_fisher_moca_replication.png")
    plot_medication(root, output / "medication_state_effects.png")
    print(f"Wrote summary figures to {output}")


if __name__ == "__main__":
    main()
