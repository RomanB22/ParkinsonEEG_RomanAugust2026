#!/usr/bin/env python
"""Pool Control versus PD/PD-ON LAVI and eBOSC participant summaries.

PD-OFF is deliberately excluded so medication participants do not enter the
pooled case group twice.  Figures show raw participant values, while inference
uses an OLS group coefficient with dataset fixed effects and HC3 standard
errors.  Expensive LAVI and eBOSC calculations are not repeated: this script
uses the completed participant tables and saved per-recording LAVI profiles.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from scipy.stats import t as student_t, ttest_ind


BANDS = ["delta", "theta", "alpha", "beta", "gamma"]
PLOT_BANDS = ["theta", "alpha", "beta", "gamma"]
GROUPS = ["Control", "PD + PD-ON"]
COLORS = {"Control": "#7f7f7f", "PD + PD-ON": "#d95f02"}

LAVI_FEATURES = {
    "lavi_mean": "Mean LAVI",
    "lavi_median": "Median LAVI",
    "lavi_peak": "Peak LAVI",
    "lavi_peak_frequency_hz": "Peak frequency (Hz)",
    "high_rhythmicity_fraction": "High-LAVI fraction",
    "low_rhythmicity_fraction": "Low-LAVI fraction",
}
LAVI_BURST_FEATURES = {
    "lavi_high_bout_count": "Frequency-bout count",
    "lavi_high_occupancy": "Frequency-bout occupancy",
    "lavi_high_bout_density_hz": "Frequency bouts / Hz",
    "lavi_high_bout_width_hz_mean": "Mean width (Hz)",
    "lavi_high_bout_peak_excess_mean": "Mean peak excess (LAVI)",
    "lavi_high_bout_peak_frequency_hz_mean": "Mean peak frequency (Hz)",
}
EBOSC_FEATURES = {
    "n_bouts": "eBOSC bout count",
    "oscillatory_occupancy": "eBOSC occupancy",
    "bouts_per_minute": "eBOSC bouts / minute",
    "duration_mean_s": "Mean duration (s)",
    "amplitude_mean": "Mean amplitude",
    "cycles_mean": "Mean cycles",
}


def _pool_groups(frame: pd.DataFrame, dataset_column: str = "dataset") -> pd.DataFrame:
    result = frame.loc[frame["group"].isin(["Control", "PD", "PD_ON"])].copy()
    result["original_group"] = result["group"]
    result["group"] = result["group"].map(
        {"Control": "Control", "PD": "PD + PD-ON", "PD_ON": "PD + PD-ON"}
    )
    result["observation_id"] = result[dataset_column].astype(str) + "::" + result["participant_id"].astype(str)
    return result


def _fdr_bh(values: pd.Series | np.ndarray) -> np.ndarray:
    p = np.asarray(values, dtype=float)
    q = np.full(p.shape, np.nan)
    valid = np.isfinite(p)
    if not valid.any():
        return q
    indices = np.flatnonzero(valid)
    order = np.argsort(p[valid])
    ranked = p[valid][order] * valid.sum() / np.arange(1, valid.sum() + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q[indices[order]] = np.minimum(ranked, 1.0)
    return q


def _hedges_g(case: np.ndarray, control: np.ndarray) -> float:
    if len(case) < 2 or len(control) < 2:
        return np.nan
    pooled_var = (
        (len(case) - 1) * np.var(case, ddof=1) + (len(control) - 1) * np.var(control, ddof=1)
    ) / (len(case) + len(control) - 2)
    if not np.isfinite(pooled_var) or pooled_var <= 0:
        return np.nan
    correction = 1.0 - 3.0 / (4.0 * (len(case) + len(control)) - 9.0)
    return float(correction * (np.mean(case) - np.mean(control)) / np.sqrt(pooled_var))


def _adjusted_group_test(frame: pd.DataFrame, feature: str) -> dict[str, Any]:
    data = frame[["group", "dataset", feature]].copy()
    data[feature] = pd.to_numeric(data[feature], errors="coerce")
    data = data.dropna()
    control = data.loc[data["group"].eq("Control"), feature].to_numpy(float)
    case = data.loc[data["group"].eq("PD + PD-ON"), feature].to_numpy(float)
    output: dict[str, Any] = {
        "n_control": len(control),
        "n_pd_plus_pd_on": len(case),
        "mean_control": float(np.mean(control)) if len(control) else np.nan,
        "mean_pd_plus_pd_on": float(np.mean(case)) if len(case) else np.nan,
        "median_control": float(np.median(control)) if len(control) else np.nan,
        "median_pd_plus_pd_on": float(np.median(case)) if len(case) else np.nan,
        "raw_mean_difference": float(np.mean(case) - np.mean(control)) if len(control) and len(case) else np.nan,
        "hedges_g": _hedges_g(case, control),
    }
    if len(control) >= 2 and len(case) >= 2 and np.unique(np.r_[control, case]).size > 1:
        welch = ttest_ind(case, control, equal_var=False, nan_policy="omit")
        output.update(raw_welch_t=float(welch.statistic), raw_welch_p=float(welch.pvalue))
    else:
        output.update(raw_welch_t=np.nan, raw_welch_p=np.nan)

    # Intercept + case indicator + k-1 dataset indicators.  HC3 protects the
    # group coefficient against unequal residual variance across cohorts.
    group_indicator = data["group"].eq("PD + PD-ON").astype(float).to_numpy()
    dataset_dummies = pd.get_dummies(data["dataset"].astype(str), drop_first=True, dtype=float)
    design = np.column_stack([np.ones(len(data)), group_indicator, dataset_dummies.to_numpy(float)])
    response = data[feature].to_numpy(float)
    beta, _, rank, _ = np.linalg.lstsq(design, response, rcond=None)
    residual = response - design @ beta
    bread = np.linalg.pinv(design.T @ design)
    leverage = np.sum((design @ bread) * design, axis=1)
    scaled_residual_sq = (residual / np.clip(1.0 - leverage, 1e-8, None)) ** 2
    meat = design.T @ (design * scaled_residual_sq[:, None])
    covariance = bread @ meat @ bread
    standard_error = float(np.sqrt(max(covariance[1, 1], 0.0)))
    statistic = float(beta[1] / standard_error) if standard_error > 0 else np.nan
    degrees_freedom = int(len(data) - rank)
    p_value = float(2.0 * student_t.sf(abs(statistic), degrees_freedom)) if degrees_freedom > 0 and np.isfinite(statistic) else np.nan
    critical = float(student_t.ppf(0.975, degrees_freedom)) if degrees_freedom > 0 else np.nan
    output.update(
        adjusted_difference=float(beta[1]),
        adjusted_hc3_se=standard_error,
        adjusted_t=statistic,
        adjusted_df=degrees_freedom,
        adjusted_p=p_value,
        adjusted_ci95_low=float(beta[1] - critical * standard_error),
        adjusted_ci95_high=float(beta[1] + critical * standard_error),
        adjustment="OLS: value ~ pooled_group + C(dataset), HC3 SE",
    )
    return output


def _family_statistics(
    frame: pd.DataFrame,
    features: dict[str, str],
    family: str,
    bands: list[str] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for band in bands or BANDS:
        band_frame = frame.loc[frame["band"].eq(band)]
        for feature, label in features.items():
            rows.append({"family": family, "band": band, "feature": feature, "feature_label": label, **_adjusted_group_test(band_frame, feature)})
    result = pd.DataFrame(rows)
    result["q_fdr_bh"] = _fdr_bh(result["adjusted_p"])
    result["significant_fdr"] = result["q_fdr_bh"] < 0.05
    return result


def _stars(value: float) -> str:
    if not np.isfinite(value) or value >= 0.05:
        return ""
    return "***" if value < 0.001 else "**" if value < 0.01 else "*"


def _plot_family(
    frame: pd.DataFrame,
    statistics: pd.DataFrame,
    features: dict[str, str],
    title: str,
    output: Path,
    bands: list[str] | None = None,
) -> None:
    bands = bands or BANDS
    fig, axes = plt.subplots(2, 3, figsize=(17, 9), squeeze=False)
    rng = np.random.default_rng(20260913)
    offsets = {"Control": -0.18, "PD + PD-ON": 0.18}
    handles = [Patch(facecolor=COLORS[group], edgecolor="white", label=group) for group in GROUPS]
    for axis, (feature, label) in zip(axes.flat, features.items()):
        for band_index, band in enumerate(bands):
            for group in GROUPS:
                values = pd.to_numeric(
                    frame.loc[frame["band"].eq(band) & frame["group"].eq(group), feature], errors="coerce"
                ).dropna().to_numpy(float)
                position = band_index + offsets[group]
                if len(values) > 1 and np.ptp(values) > 0:
                    violin = axis.violinplot([values], positions=[position], widths=0.32, showmedians=True, showextrema=False)
                    body = violin["bodies"][0]
                    body.set_facecolor(COLORS[group]); body.set_edgecolor("white"); body.set_alpha(0.72)
                    violin["cmedians"].set_color("#202020"); violin["cmedians"].set_linewidth(1.0)
                if len(values):
                    keep = np.arange(len(values)) if len(values) <= 180 else rng.choice(len(values), 180, replace=False)
                    jitter = rng.uniform(-0.055, 0.055, len(keep))
                    axis.scatter(position + jitter, values[keep], s=7, color=COLORS[group], alpha=0.42, linewidth=0, rasterized=True)
            test = statistics.loc[statistics["band"].eq(band) & statistics["feature"].eq(feature)]
            if not test.empty:
                stars = _stars(float(test.iloc[0]["q_fdr_bh"]))
                if stars:
                    axis.text(band_index, 0.98, stars, transform=axis.get_xaxis_transform(), ha="center", va="top", fontsize=12, fontweight="bold")
        axis.set_title(label, fontsize=11, fontweight="bold")
        axis.set_xticks(range(len(bands)), [band.replace("_", " ").title() for band in bands], rotation=20, ha="right")
        axis.grid(axis="y", alpha=0.20)
        axis.spines[["top", "right"]].set_visible(False)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.952), ncol=2, frameon=False)
    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.995)
    fig.text(0.5, 0.902, "Each point is one participant-dataset observation; stars use dataset-adjusted HC3 tests with BH-FDR across the analysis family", ha="center", fontsize=9, color="#444444")
    fig.tight_layout(rect=(0.02, 0.02, 0.99, 0.86))
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _frequency_edges(frequencies: np.ndarray) -> np.ndarray:
    edges = np.empty(len(frequencies) + 1, dtype=float)
    edges[1:-1] = (frequencies[:-1] + frequencies[1:]) / 2.0
    edges[0] = frequencies[0] - (frequencies[1] - frequencies[0]) / 2.0
    edges[-1] = frequencies[-1] + (frequencies[-1] - frequencies[-2]) / 2.0
    return edges


def _contiguous_signed_segments(status: np.ndarray) -> list[tuple[int, int, int]]:
    values = np.sign(np.asarray(status, dtype=float)).astype(int)
    rows: list[tuple[int, int, int]] = []
    start = 0
    for index in range(1, len(values) + 1):
        if index == len(values) or values[index] != values[start]:
            if values[start] != 0:
                rows.append((start, index - 1, int(values[start])))
            start = index
    return rows


def _pooled_group_profiles(rhythmicity_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    from lavi import abba

    electrode = pd.read_csv(rhythmicity_root / "metrics" / "electrode_rhythmicity.csv.gz", low_memory=False)
    metadata = _pool_groups(electrode)[["dataset", "recording_id", "participant_id", "group"]].drop_duplicates()
    profile_rows: list[dict[str, Any]] = []
    expected_frequencies: np.ndarray | None = None
    for row in metadata.itertuples(index=False):
        path = rhythmicity_root / "profiles" / str(row.dataset) / f"{row.recording_id}.npz"
        if not path.is_file():
            continue
        with np.load(path, allow_pickle=False) as saved:
            frequencies = np.asarray(saved["foi"], dtype=float)
            profile = np.nanmean(np.asarray(saved["lavi"], dtype=float), axis=0)
        if expected_frequencies is None:
            expected_frequencies = frequencies
        elif not np.allclose(frequencies, expected_frequencies):
            raise ValueError(f"Incompatible LAVI frequency grid: {path}")
        for frequency, value in zip(frequencies, profile):
            profile_rows.append({"dataset": row.dataset, "recording_id": row.recording_id, "participant_id": row.participant_id, "group": row.group, "frequency_hz": frequency, "lavi": value})
    profiles = pd.DataFrame(profile_rows)
    participant = profiles.groupby(["dataset", "participant_id", "group", "frequency_hz"], as_index=False)["lavi"].mean()
    group_profiles = participant.groupby(["group", "frequency_hz"], as_index=False).agg(lavi_mean=("lavi", "mean"), lavi_sd=("lavi", "std"), n=("lavi", "count"))
    frequencies = np.sort(group_profiles["frequency_hz"].unique())
    matrix = np.vstack([group_profiles.loc[group_profiles["group"].eq(group)].sort_values("frequency_hz")["lavi_mean"].to_numpy(float) for group in GROUPS])
    _, _, sigvect = abba(matrix, frequencies)
    edges = _frequency_edges(frequencies)
    segment_rows: list[dict[str, Any]] = []
    for group, profile, status in zip(GROUPS, matrix, sigvect):
        baseline = float(np.nanmedian(profile))
        for segment_index, (start, stop, sign) in enumerate(_contiguous_signed_segments(status), start=1):
            values = profile[start : stop + 1]
            local_peak = int(np.nanargmax(values) if sign > 0 else np.nanargmin(values))
            segment_rows.append({
                "group": group,
                "segment_index": segment_index,
                "direction": "high" if sign > 0 else "low",
                "start_frequency_hz": float(edges[start]),
                "end_frequency_hz": float(edges[stop + 1]),
                "start_bin_center_hz": float(frequencies[start]),
                "end_bin_center_hz": float(frequencies[stop]),
                "extreme_frequency_hz": float(frequencies[start + local_peak]),
                "extreme_lavi": float(values[local_peak]),
                "median_baseline_lavi": baseline,
                "n_frequency_bins": int(stop - start + 1),
            })
    return group_profiles, pd.DataFrame(segment_rows)


def _plot_group_profiles(profiles: pd.DataFrame, segments: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for axis, group in zip(axes, GROUPS):
        frame = profiles.loc[profiles["group"].eq(group)].sort_values("frequency_hz")
        x = frame["frequency_hz"].to_numpy(float); y = frame["lavi_mean"].to_numpy(float)
        sem = frame["lavi_sd"].to_numpy(float) / np.sqrt(frame["n"].to_numpy(float))
        axis.fill_between(x, y - 1.96 * sem, y + 1.96 * sem, color=COLORS[group], alpha=0.18, linewidth=0)
        axis.plot(x, y, color=COLORS[group], linewidth=2.2)
        for row in segments.loc[segments["group"].eq(group)].itertuples(index=False):
            color = "#238b45" if row.direction == "high" else "#756bb1"
            axis.axvspan(row.start_frequency_hz, row.end_frequency_hz, color=color, alpha=0.13)
            axis.text(np.sqrt(row.start_frequency_hz * row.end_frequency_hz), 0.97, f"{row.direction}\n{row.start_frequency_hz:.1f}–{row.end_frequency_hz:.1f}", transform=axis.get_xaxis_transform(), ha="center", va="top", fontsize=7, color=color)
        axis.axhline(np.median(y), color="#333333", linestyle="--", linewidth=0.8)
        axis.set_ylabel("LAVI")
        axis.set_title(f"{group} (n={int(frame['n'].max())})", loc="left", fontweight="bold")
        axis.grid(axis="y", alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    axes[-1].set_xscale("log"); axes[-1].set_xlabel("Frequency (Hz)")
    axes[-1].set_xticks([4, 6, 8, 10, 13, 20, 30, 40], ["4", "6", "8", "10", "13", "20", "30", "40"])
    fig.suptitle("Pooled participant-level LAVI profiles and data-driven ABBA intervals", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run(rhythmicity_root: Path, output_root: Path) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    figures = output_root / "figures"; figures.mkdir(exist_ok=True)
    metrics = output_root / "metrics"; metrics.mkdir(exist_ok=True)
    statistics_root = output_root / "statistics"; statistics_root.mkdir(exist_ok=True)

    rhythmicity = _pool_groups(pd.read_csv(rhythmicity_root / "metrics" / "participant_rhythmicity.csv.gz"))
    lavi_bursts = _pool_groups(pd.read_csv(rhythmicity_root / "metrics" / "lavi_burst_features_participant.csv.gz"))
    key = ["dataset", "participant_id", "group", "band"]
    if rhythmicity.duplicated(key).any() or lavi_bursts.duplicated(key).any():
        raise ValueError("Participant tables are not unique after repeated recordings should have been averaged")
    rhythmicity.to_csv(metrics / "pooled_lavi_and_ebosc_participants.csv.gz", index=False)
    lavi_bursts.to_csv(metrics / "pooled_lavi_frequency_burst_participants.csv.gz", index=False)

    lavi_stats = _family_statistics(rhythmicity, LAVI_FEATURES, "LAVI")
    lavi_burst_stats = _family_statistics(lavi_bursts, LAVI_BURST_FEATURES, "LAVI frequency bouts")
    ebosc_stats = _family_statistics(rhythmicity, EBOSC_FEATURES, "eBOSC temporal bouts")
    all_statistics = pd.concat([lavi_stats, lavi_burst_stats, ebosc_stats], ignore_index=True)
    all_statistics.to_csv(statistics_root / "pooled_group_comparisons.csv", index=False)

    _plot_family(rhythmicity, lavi_stats, LAVI_FEATURES, "Pooled LAVI by canonical frequency band", figures / "pooled_lavi_band_violins.png", PLOT_BANDS)
    _plot_family(lavi_bursts, lavi_burst_stats, LAVI_BURST_FEATURES, "Pooled LAVI frequency-domain burst characteristics", figures / "pooled_lavi_frequency_burst_violins.png", PLOT_BANDS)
    _plot_family(rhythmicity, ebosc_stats, EBOSC_FEATURES, "Pooled aperiodic-relative eBOSC temporal burst characteristics", figures / "pooled_ebosc_burst_violins.png", PLOT_BANDS)

    profiles, segments = _pooled_group_profiles(rhythmicity_root)
    profiles.to_csv(metrics / "pooled_lavi_group_profiles.csv.gz", index=False)
    segments.to_csv(statistics_root / "pooled_lavi_abba_intervals.csv", index=False)
    _plot_group_profiles(profiles, segments, figures / "pooled_lavi_profiles_abba_intervals.png")

    cohort = rhythmicity[["dataset", "participant_id", "original_group", "group"]].drop_duplicates()
    cohort_counts = cohort.groupby(["dataset", "original_group"]).size().rename("n").reset_index()
    cohort_counts.to_csv(metrics / "pooled_cohort_counts.csv", index=False)
    summary = {
        "n_control": int(cohort["group"].eq("Control").sum()),
        "n_pd_plus_pd_on": int(cohort["group"].eq("PD + PD-ON").sum()),
        "n_total": int(len(cohort)),
        "pd_off_excluded": True,
        "statistics_rows": int(len(all_statistics)),
        "abba_intervals": int(len(segments)),
    }
    (output_root / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rhythmicity-root", type=Path, default=Path("outputs/rhythmicity"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/rhythmicity/pooled_control_vs_pd_on"))
    args = parser.parse_args()
    print(json.dumps(run(args.rhythmicity_root, args.output_root), indent=2))


if __name__ == "__main__":
    main()
