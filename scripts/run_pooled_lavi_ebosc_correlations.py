#!/usr/bin/env python
"""Relate LAVI frequency-bout quantities to eBOSC temporal-burst quantities."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr, t as student_t


BANDS = ["theta_low", "alpha_high", "beta_low", "beta_high", "gamma_low"]
GROUPS = ["Control", "PD + PD-ON"]
LAVI_FEATURES = {
    "lavi_high_bout_count": "Count",
    "lavi_high_occupancy": "Occupancy",
    "lavi_high_bout_density_hz": "Bouts / Hz",
    "lavi_high_bout_width_hz_mean": "Width",
    "lavi_high_bout_peak_excess_mean": "Peak excess",
    "lavi_high_bout_peak_frequency_hz_mean": "Peak frequency",
}
EBOSC_FEATURES = {
    "n_bouts": "Count",
    "oscillatory_occupancy": "Occupancy",
    "bouts_per_minute": "Bouts / min",
    "duration_mean_s": "Duration",
    "amplitude_mean": "Amplitude",
    "cycles_mean": "Cycles",
}
PRESPECIFIED_PAIRS = {
    ("lavi_high_occupancy", "oscillatory_occupancy"),
    ("lavi_high_bout_width_hz_mean", "duration_mean_s"),
    ("lavi_high_bout_peak_excess_mean", "amplitude_mean"),
    ("lavi_high_bout_count", "n_bouts"),
    ("lavi_high_bout_count", "bouts_per_minute"),
    ("lavi_high_bout_peak_frequency_hz_mean", "cycles_mean"),
}


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


def _partial_spearman(x: np.ndarray, y: np.ndarray, datasets: np.ndarray) -> tuple[float, float, int]:
    valid = pd.notna(x) & pd.notna(y) & pd.notna(datasets)
    x = np.asarray(x[valid], dtype=float)
    y = np.asarray(y[valid], dtype=float)
    datasets = np.asarray(datasets[valid], dtype=str)
    n = len(x)
    if n < 5 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return np.nan, np.nan, n
    ranked_x = rankdata(x, method="average")
    ranked_y = rankdata(y, method="average")
    dummies = pd.get_dummies(pd.Series(datasets), drop_first=True, dtype=float).to_numpy()
    design = np.column_stack([np.ones(n), dummies])
    residual_x = ranked_x - design @ np.linalg.lstsq(design, ranked_x, rcond=None)[0]
    residual_y = ranked_y - design @ np.linalg.lstsq(design, ranked_y, rcond=None)[0]
    if np.ptp(residual_x) == 0 or np.ptp(residual_y) == 0:
        return np.nan, np.nan, n
    rho = float(np.corrcoef(residual_x, residual_y)[0, 1])
    degrees_freedom = int(n - np.linalg.matrix_rank(design) - 1)
    if degrees_freedom <= 0 or abs(rho) >= 1:
        p_value = 0.0 if abs(rho) >= 1 else np.nan
    else:
        statistic = rho * np.sqrt(degrees_freedom / max(1.0 - rho**2, np.finfo(float).eps))
        p_value = float(2.0 * student_t.sf(abs(statistic), degrees_freedom))
    return rho, p_value, n


def _correlations(merged: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for group in GROUPS:
        for band in BANDS:
            selected = merged.loc[merged["group"].eq(group) & merged["band"].eq(band)]
            for lavi_feature, lavi_label in LAVI_FEATURES.items():
                for ebosc_feature, ebosc_label in EBOSC_FEATURES.items():
                    subset = selected[["dataset", lavi_feature, ebosc_feature]].dropna()
                    if len(subset) >= 3 and subset[lavi_feature].nunique() > 1 and subset[ebosc_feature].nunique() > 1:
                        unadjusted = spearmanr(subset[lavi_feature], subset[ebosc_feature])
                        raw_rho, raw_p = float(unadjusted.statistic), float(unadjusted.pvalue)
                    else:
                        raw_rho = raw_p = np.nan
                    rho, p_value, n = _partial_spearman(
                        selected[lavi_feature].to_numpy(),
                        selected[ebosc_feature].to_numpy(),
                        selected["dataset"].to_numpy(),
                    )
                    rows.append({
                        "group": group,
                        "band": band,
                        "lavi_feature": lavi_feature,
                        "lavi_label": lavi_label,
                        "ebosc_feature": ebosc_feature,
                        "ebosc_label": ebosc_label,
                        "n": n,
                        "spearman_rho_unadjusted": raw_rho,
                        "spearman_p_unadjusted": raw_p,
                        "partial_spearman_rho_dataset_adjusted": rho,
                        "partial_spearman_p_dataset_adjusted": p_value,
                        "prespecified_pair": (lavi_feature, ebosc_feature) in PRESPECIFIED_PAIRS,
                    })
    result = pd.DataFrame(rows)
    result["q_fdr_group_180"] = np.nan
    for group, indices in result.groupby("group").groups.items():
        result.loc[indices, "q_fdr_group_180"] = _fdr_bh(
            result.loc[indices, "partial_spearman_p_dataset_adjusted"]
        )
    result["significant_fdr_group_180"] = result["q_fdr_group_180"] < 0.05
    result["q_fdr_prespecified_30"] = np.nan
    for group, indices in result.loc[result["prespecified_pair"]].groupby("group").groups.items():
        result.loc[indices, "q_fdr_prespecified_30"] = _fdr_bh(
            result.loc[indices, "partial_spearman_p_dataset_adjusted"]
        )
    result["significant_fdr_prespecified_30"] = result["q_fdr_prespecified_30"] < 0.05
    return result


def _stars(q_value: float) -> str:
    if not np.isfinite(q_value) or q_value >= 0.05:
        return ""
    return "***" if q_value < 0.001 else "**" if q_value < 0.01 else "*"


def _plot_group(result: pd.DataFrame, group: str, output: Path) -> None:
    fig, axes = plt.subplots(1, len(BANDS), figsize=(21, 5.2), sharex=True, sharey=True)
    image = None
    for axis, band in zip(axes, BANDS):
        selected = result.loc[result["group"].eq(group) & result["band"].eq(band)]
        matrix = selected.pivot(index="lavi_feature", columns="ebosc_feature", values="partial_spearman_rho_dataset_adjusted")
        matrix = matrix.reindex(index=LAVI_FEATURES, columns=EBOSC_FEATURES)
        image = axis.imshow(matrix.to_numpy(float), vmin=-1, vmax=1, cmap="RdBu_r", aspect="equal")
        q_matrix = selected.pivot(index="lavi_feature", columns="ebosc_feature", values="q_fdr_group_180")
        q_matrix = q_matrix.reindex(index=LAVI_FEATURES, columns=EBOSC_FEATURES)
        for row in range(len(LAVI_FEATURES)):
            for column in range(len(EBOSC_FEATURES)):
                rho = matrix.iloc[row, column]
                if not np.isfinite(rho):
                    continue
                text = f"{rho:.2f}{_stars(float(q_matrix.iloc[row, column]))}"
                color = "white" if abs(rho) >= 0.55 else "#202020"
                axis.text(column, row, text, ha="center", va="center", fontsize=7, color=color)
        axis.set_title(band.replace("_", " ").title(), fontweight="bold")
        axis.set_xticks(range(len(EBOSC_FEATURES)), EBOSC_FEATURES.values(), rotation=45, ha="right")
        axis.set_yticks(range(len(LAVI_FEATURES)), LAVI_FEATURES.values())
        axis.set_xlabel("eBOSC temporal-burst metric")
    axes[0].set_ylabel("LAVI frequency-bout metric")
    fig.suptitle(f"LAVI frequency bouts vs eBOSC temporal bursts — {group}", fontsize=15, fontweight="bold")
    fig.text(
        0.5, 0.92,
        "Cell: dataset-adjusted partial Spearman rho; stars use BH-FDR across all 180 cross-method tests in this group",
        ha="center", fontsize=9, color="#444444",
    )
    if image is not None:
        colorbar_axis = fig.add_axes([0.955, 0.25, 0.012, 0.52])
        colorbar = fig.colorbar(image, cax=colorbar_axis)
        colorbar.set_label("Partial Spearman rho")
    fig.subplots_adjust(left=0.08, right=0.94, bottom=0.23, top=0.82, wspace=0.12)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _write_summary(result: pd.DataFrame, output: Path) -> None:
    lines = [
        "# LAVI frequency-bout and eBOSC temporal-burst correlations",
        "",
        "This exploratory analysis relates participant-level quantities from the two",
        "LAVI-determined-band figures. Correlations are calculated separately within",
        "Control and pooled PD + PD-ON. Ranks of both variables are residualized for",
        "dataset, and their residual correlation is reported as a dataset-adjusted",
        "partial Spearman rho.",
        "",
        "The primary multiplicity correction is BH-FDR across all 180 cross-method",
        "tests (5 bands × 6 LAVI quantities × 6 eBOSC quantities) within each group.",
        "The CSV also contains a secondary FDR correction across 30 prespecified tests",
        "per group (5 bands × 6 conceptually paired metric combinations).",
        "",
    ]
    for group in GROUPS:
        selected = result.loc[result["group"].eq(group)]
        significant = selected.loc[selected["significant_fdr_group_180"]].sort_values("q_fdr_group_180")
        prespecified = selected.loc[
            selected["prespecified_pair"] & selected["significant_fdr_prespecified_30"]
        ].sort_values("q_fdr_prespecified_30")
        lines.extend([
            f"## {group}",
            "",
            f"FDR-significant exhaustive correlations: **{len(significant)} / 180**. "
            f"FDR-significant prespecified correlations: **{len(prespecified)} / 30**.",
            "",
        ])
        if significant.empty:
            lines.append("No correlation survived the exhaustive correction.")
            lines.append("")
            continue
        lines.extend([
            "### Fifteen strongest exhaustive associations",
            "",
            "| Band | LAVI quantity | eBOSC quantity | n | Partial rho | q |",
            "|---|---|---|---:|---:|---:|",
        ])
        for row in significant.head(15).itertuples(index=False):
            lines.append(
                f"| {str(row.band).replace('_', ' ').title()} | {row.lavi_label} | {row.ebosc_label} | "
                f"{row.n} | {row.partial_spearman_rho_dataset_adjusted:.3f} | {row.q_fdr_group_180:.3g} |"
            )
        lines.append("")
        lines.extend([
            "The complete set of significant and nonsignificant associations is in",
            "`../statistics/lavi_determined_band_lavi_ebosc_correlations.csv`.",
            "",
        ])
    lines.extend([
        "## Interpretation limits",
        "",
        "These are cross-sectional associations, not evidence that one type of bout",
        "causes the other. LAVI frequency bouts and eBOSC bursts remain different",
        "spectral and temporal constructs. Shared, group-specific band boundaries can",
        "also induce association through band width or placement. The bands were",
        "selected and evaluated in the same cohort, so the results are exploratory.",
        "The 180 tests are also not 180 independent biological signals: within a fixed",
        "group-band, LAVI count and bouts/Hz are proportional, and several eBOSC event",
        "summaries are intrinsically related. Discovery counts should therefore not be",
        "interpreted as counts of distinct mechanisms.",
        "",
        "See `lavi_determined_band_burst_methods.md` for the detector definitions and",
        "the complete CSV for nonsignificant and unadjusted results.",
        "",
    ])
    output.write_text("\n".join(lines), encoding="utf-8")


def run(output_root: Path) -> None:
    metrics = output_root / "metrics"
    statistics = output_root / "statistics"
    figures = output_root / "figures"
    lavi = pd.read_csv(metrics / "lavi_determined_band_participant_metrics.csv.gz", low_memory=False)
    ebosc = pd.read_csv(metrics / "lavi_determined_band_ebosc_participant_metrics.csv.gz", low_memory=False)
    keys = ["dataset", "participant_id", "group", "band"]
    if lavi.duplicated(keys).any() or ebosc.duplicated(keys).any():
        raise ValueError("Input participant tables must be unique by dataset, participant, group, and band")
    merged = lavi[keys + list(LAVI_FEATURES)].merge(
        ebosc[keys + list(EBOSC_FEATURES)], on=keys, how="inner", validate="one_to_one"
    )
    expected_rows = len(GROUPS) * len(BANDS)
    if merged.groupby(["group", "band"]).ngroups != expected_rows:
        raise ValueError("Merged table is missing a requested group-band combination")
    result = _correlations(merged)
    result.to_csv(statistics / "lavi_determined_band_lavi_ebosc_correlations.csv", index=False)
    for group, suffix in [("Control", "control"), ("PD + PD-ON", "pd_plus_pd_on")]:
        _plot_group(
            result,
            group,
            figures / f"lavi_determined_band_lavi_ebosc_correlations_{suffix}.png",
        )
    _write_summary(result, figures / "lavi_determined_band_lavi_ebosc_correlations.md")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/rhythmicity/pooled_control_vs_pd_on"),
    )
    args = parser.parse_args()
    run(args.output_root)
    print(args.output_root / "statistics" / "lavi_determined_band_lavi_ebosc_correlations.csv")


if __name__ == "__main__":
    main()
