#!/usr/bin/env python3
"""Plot clinical/behavioral scores against temporal bursts in ABBA bands.

The script consumes the saved participant-level ABBA burst metrics, so it does
not rerun EEG processing.  It creates one figure per selected ABBA interval
and outcome family (cognitive score and UPDRS), with burst quantities in
columns and datasets in rows.  MoCA is used for the three standard datasets
and MMSE for the medication-state cohort.  UPDRS plots are restricted to PD
groups because UPDRS is a motor-severity measure and is not meaningful for
healthy controls.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


DATASET_ORDER = ["primary", "ds007526-1.0.2", "ds008768-1.0.0", "medication_state"]
DATASET_LABELS = {
    "primary": "Primary",
    "ds007526-1.0.2": "ds007526",
    "ds008768-1.0.0": "ds008768",
    "medication_state": "Medication state",
}
GROUP_COLORS = {
    "Control": "#0072B2",
    "PD": "#D55E00",
    "PD_OFF": "#7570B1",
    "PD_ON": "#009E73",
}
QUANTITIES = [
    "duration_s",
    "cycles",
    "bursts_per_minute",
    "occupancy_percent",
    "peak_amplitude_uv",
]
QUANTITY_LABELS = {
    "duration_s": "Burst duration (s)",
    "cycles": "Burst cycles",
    "bursts_per_minute": "Bursts per minute",
    "occupancy_percent": "Burst occupancy (%)",
    "peak_amplitude_uv": "Peak EEG amplitude (µV)",
}
SELECTED_BANDS = ("theta_1", "alpha_1", "beta_1", "beta_2", "gamma_1")
OUTLIER_SD = 3.0


def _stars(q_value: float) -> str:
    if not np.isfinite(q_value) or q_value >= 0.05:
        return ""
    if q_value < 0.001:
        return "***"
    if q_value < 0.01:
        return "**"
    return "*"


def _fdr_bh(values: pd.Series) -> np.ndarray:
    result = np.full(len(values), np.nan, dtype=float)
    finite = np.isfinite(values.to_numpy(float))
    if not finite.any():
        return result
    indices = np.flatnonzero(finite)
    order = np.argsort(values.to_numpy(float)[indices])
    p_sorted = values.to_numpy(float)[indices][order]
    adjusted = p_sorted * len(p_sorted) / np.arange(1, len(p_sorted) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result[indices[order]] = np.minimum(adjusted, 1.0)
    return result


def _load_clinical(global_root: Path) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for path in sorted((global_root / "canonical").glob("*/recordings.csv.gz")):
        frame = pd.read_csv(path, low_memory=False)
        if "dataset_id" not in frame:
            continue
        frame = frame.rename(columns={"dataset_id": "dataset"})
        columns = [
            column
            for column in ("dataset", "participant_id", "group", "moca", "mmse", "updrs")
            if column in frame
        ]
        frame = frame[columns].copy()
        for column in ("moca", "mmse", "updrs"):
            if column in frame:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
        rows.append(frame)
    if not rows:
        return pd.DataFrame(columns=["dataset", "participant_id", "group"])
    clinical = pd.concat(rows, ignore_index=True)
    keys = ["dataset", "participant_id", "group"]
    value_columns = [column for column in ("moca", "mmse", "updrs") if column in clinical]
    return clinical.groupby(keys, as_index=False)[value_columns].mean(numeric_only=True)


def _focused_view(table: pd.DataFrame, band_name: str) -> pd.DataFrame:
    """Apply the same medication theta alignment as the focused ABBA figures."""
    result = table.copy()
    if band_name != "theta_1":
        return result
    medication_control = result["dataset"].eq("medication_state") & result["group"].eq("Control")
    result = result.loc[~(medication_control & result["band_name"].eq("theta_1"))].copy()
    remap = medication_control.loc[result.index] & result["band_name"].eq("theta_2")
    result.loc[remap, "band_name"] = "theta_1"
    return result


def _clinical_outcome(dataset: str, frame: pd.DataFrame) -> tuple[str, str]:
    preferred = ("mmse", "MMSE") if dataset == "medication_state" else ("moca", "MoCA")
    fallback = ("moca", "MoCA") if preferred[0] == "mmse" else ("mmse", "MMSE")
    for column, label in (preferred, fallback):
        if column in frame and pd.to_numeric(frame[column], errors="coerce").notna().any():
            return column, label
    return preferred


def _trim_x_outliers(
    paired: pd.DataFrame,
    quantity: str,
    *,
    threshold_sd: float | None = OUTLIER_SD,
) -> tuple[pd.DataFrame, pd.DataFrame, float, float]:
    """Remove x values beyond ``threshold_sd`` pooled standard deviations.

    Outlier detection is performed separately for each dataset/band/outcome/
    quantity panel and only on the burst-property (x-axis) values. Clinical
    scores are never removed by this rule. The returned second frame is an
    audit table containing every removed participant.
    """
    if paired.empty or quantity not in paired:
        return paired, pd.DataFrame(), np.nan, np.nan
    values = pd.to_numeric(paired[quantity], errors="coerce")
    mean = float(values.mean()) if values.notna().any() else np.nan
    sd = float(values.std(ddof=1)) if values.notna().sum() > 1 else np.nan
    if threshold_sd is None or not np.isfinite(mean) or not np.isfinite(sd) or sd <= 0:
        return paired, pd.DataFrame(), mean, sd
    excluded = (values - mean).abs() > float(threshold_sd) * sd
    removed = paired.loc[excluded].copy()
    return paired.loc[~excluded].copy(), removed, mean, sd


def _correlations(table: pd.DataFrame, *, outlier_sd: float | None = OUTLIER_SD) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    excluded_rows: list[dict[str, object]] = []
    for band in SELECTED_BANDS:
        band_table = _focused_view(table, band)
        band_table = band_table.loc[band_table["band_name"].eq(band)]
        for dataset in DATASET_ORDER:
            frame = band_table.loc[band_table["dataset"].eq(dataset)].copy()
            if frame.empty:
                continue
            cognitive_column, cognitive_label = _clinical_outcome(dataset, frame)
            outcomes = [(cognitive_column, cognitive_label, "all groups")]
            if "updrs" in frame:
                outcomes.append(("updrs", "UPDRS", "PD groups only"))
            for clinical_column, clinical_label, population in outcomes:
                analysis = frame.copy()
                if clinical_label == "UPDRS":
                    analysis = analysis.loc[analysis["group"].astype(str).str.startswith("PD")]
                if clinical_column not in analysis:
                    continue
                for quantity in QUANTITIES:
                    if quantity not in analysis:
                        continue
                    pair_columns = [quantity, clinical_column, "participant_id", "group"]
                    paired = analysis[pair_columns].copy()
                    paired[quantity] = pd.to_numeric(paired[quantity], errors="coerce")
                    paired[clinical_column] = pd.to_numeric(paired[clinical_column], errors="coerce")
                    paired = paired.dropna(subset=[quantity, clinical_column])
                    n_raw = len(paired)
                    paired, removed, x_mean, x_sd = _trim_x_outliers(paired, quantity, threshold_sd=outlier_sd)
                    if not removed.empty:
                        for _, removed_row in removed.iterrows():
                            excluded_rows.append(
                                {
                                    "band_name": band,
                                    "dataset": dataset,
                                    "quantity": quantity,
                                    "clinical_measure": clinical_label,
                                    "clinical_column": clinical_column,
                                    "population": population,
                                    "participant_id": removed_row["participant_id"],
                                    "group": removed_row["group"],
                                    "x_value": removed_row[quantity],
                                    "clinical_value": removed_row[clinical_column],
                                    "x_mean": x_mean,
                                    "x_sd": x_sd,
                                    "threshold_sd": outlier_sd,
                                    "lower_cutoff": x_mean - float(outlier_sd) * x_sd,
                                    "upper_cutoff": x_mean + float(outlier_sd) * x_sd,
                                }
                            )
                    if len(paired) >= 5 and paired[[quantity, clinical_column]].nunique().min() > 1:
                        rho, p_value = spearmanr(paired[quantity], paired[clinical_column])
                    else:
                        rho, p_value = np.nan, np.nan
                    rows.append(
                        {
                            "band_name": band,
                            "dataset": dataset,
                            "quantity": quantity,
                            "clinical_measure": clinical_label,
                            "clinical_column": clinical_column,
                            "population": population,
                            "n": len(paired),
                            "n_raw": n_raw,
                            "n_excluded_x_outlier": n_raw - len(paired),
                            "x_mean_raw": x_mean,
                            "x_sd_raw": x_sd,
                            "outlier_rule": "none" if outlier_sd is None else f"abs(x - mean) <= {outlier_sd:g} SD",
                            "rho": float(rho) if np.isfinite(rho) else np.nan,
                            "p_value": float(p_value) if np.isfinite(p_value) else np.nan,
                        }
                    )
    result = pd.DataFrame(rows)
    if result.empty:
        return result, pd.DataFrame(excluded_rows)
    result["q_fdr_bh"] = np.nan
    for measure, indices in result.groupby("clinical_measure").groups.items():
        result.loc[indices, "q_fdr_bh"] = _fdr_bh(result.loc[indices, "p_value"])
    result["significant_fdr"] = result["q_fdr_bh"] < 0.05
    result["significance"] = result["q_fdr_bh"].map(_stars)
    return result, pd.DataFrame(excluded_rows)


def _plot_band(
    table: pd.DataFrame,
    stats: pd.DataFrame,
    band: str,
    outcome: str,
    output: Path,
    *,
    outlier_sd: float | None = OUTLIER_SD,
) -> None:
    datasets = [dataset for dataset in DATASET_ORDER if dataset in set(table["dataset"])]
    if not datasets:
        return
    fig, axes = plt.subplots(
        len(datasets), len(QUANTITIES),
        figsize=(4.0 * len(QUANTITIES), 3.1 * len(datasets)),
        squeeze=False,
        constrained_layout=False,
    )
    for row, dataset in enumerate(datasets):
        source = _focused_view(table, band)
        frame = source.loc[(source["dataset"].eq(dataset)) & source["band_name"].eq(band)].copy()
        if outcome == "cognitive":
            clinical_column, clinical_label = _clinical_outcome(dataset, frame)
            population = "all groups"
        else:
            clinical_column, clinical_label, population = "updrs", "UPDRS", "PD groups only"
            if not frame.empty:
                frame = frame.loc[frame["group"].astype(str).str.startswith("PD")]
        if clinical_column in frame:
            frame[clinical_column] = pd.to_numeric(frame[clinical_column], errors="coerce")
        groups = [group for group in ("Control", "PD", "PD_OFF", "PD_ON") if group in set(frame.get("group", []))]
        for col, quantity in enumerate(QUANTITIES):
            axis = axes[row, col]
            if clinical_column not in frame or quantity not in frame:
                paired = pd.DataFrame()
            else:
                paired = frame[[quantity, clinical_column, "participant_id", "group"]].copy()
                paired[quantity] = pd.to_numeric(paired[quantity], errors="coerce")
                paired = paired.dropna(subset=[quantity, clinical_column])
                paired, _, _, _ = _trim_x_outliers(paired, quantity, threshold_sd=outlier_sd)
            for group in groups:
                points = paired.loc[paired["group"].eq(group)] if not paired.empty else paired
                if not points.empty:
                    axis.scatter(
                        points[quantity], points[clinical_column],
                        s=21, alpha=0.78, color=GROUP_COLORS[group],
                        edgecolor="white", linewidth=0.35,
                    )
            if not paired.empty and len(paired) >= 5 and paired[quantity].nunique() > 1 and paired[clinical_column].nunique() > 1:
                x = paired[quantity].to_numpy(float)
                y = paired[clinical_column].to_numpy(float)
                slope, intercept = np.polyfit(x, y, 1)
                grid = np.linspace(float(x.min()), float(x.max()), 80)
                axis.plot(grid, intercept + slope * grid, color="#222222", linewidth=1.05)
            match = stats.loc[
                (stats["band_name"].eq(band))
                & (stats["dataset"].eq(dataset))
                & (stats["quantity"].eq(quantity))
                & (stats["clinical_measure"].eq(clinical_label))
            ]
            if not match.empty and int(match.iloc[0]["n"]) > 0:
                item = match.iloc[0]
                annotation = f"n={int(item['n'])}\nρ={item['rho']:.2f}" if np.isfinite(item["rho"]) else f"n={int(item['n'])}"
                if np.isfinite(item["q_fdr_bh"]):
                    annotation += f"{_stars(item['q_fdr_bh'])}\nq={item['q_fdr_bh']:.2g}"
                axis.text(0.04, 0.96, annotation, transform=axis.transAxes, va="top", fontsize=7.5,
                          bbox={"facecolor": "white", "alpha": 0.80, "edgecolor": "none", "pad": 1.5})
            elif paired.empty or match.empty or int(match.iloc[0]["n"]) == 0:
                axis.text(0.5, 0.5, f"No {clinical_label}\ndata", transform=axis.transAxes,
                          ha="center", va="center", fontsize=8, color="#777777")
            if row == 0:
                axis.set_title(QUANTITY_LABELS[quantity], fontsize=9.5, fontweight="bold")
            if col == 0:
                axis.set_ylabel(f"{DATASET_LABELS[dataset]}\n{clinical_label}", fontsize=9, fontweight="bold")
            else:
                axis.set_ylabel("")
            axis.set_xlabel(QUANTITY_LABELS[quantity] if row == len(datasets) - 1 else "")
            axis.grid(alpha=0.18)
    handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=color,
                      markeredgecolor="white", markersize=6, label=group.replace("PD_", "PD-"))
               for group, color in GROUP_COLORS.items()]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.965),
               ncol=4, frameon=False, fontsize=8)
    outcome_title = "cognitive score (MoCA; MMSE in medication state)" if outcome == "cognitive" else "UPDRS motor severity"
    population_note = "all available groups" if outcome == "cognitive" else "PD groups only"
    alignment_note = "; medication Control theta_2 aligned to theta_1" if band == "theta_1" else ""
    outlier_note = f"; x outliers >{outlier_sd:g} SD excluded" if outlier_sd is not None else "; no x-axis outlier exclusion"
    fig.suptitle(f"ABBA {band} burst properties versus {outcome_title}", fontsize=15, fontweight="bold", y=0.995)
    fig.text(0.5, 0.945, f"Participant-level Spearman associations; {population_note}{alignment_note}{outlier_note}; stars mark FDR q<0.05",
             ha="center", va="top", fontsize=8.5, color="#444444")
    fig.subplots_adjust(left=0.055, right=0.995, bottom=0.055, top=0.89, wspace=0.22, hspace=0.28)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=320, bbox_inches="tight")
    plt.close(fig)


def run(output_root: Path, global_root: Path, *, outlier_sd: float | None = OUTLIER_SD) -> dict[str, int]:
    metrics_path = output_root / "metrics" / "abba_burst_participant_metrics.csv.gz"
    if not metrics_path.is_file():
        raise FileNotFoundError(metrics_path)
    table = pd.read_csv(metrics_path, low_memory=False)
    table["dataset"] = table["dataset"].astype(str)
    table["participant_id"] = table["participant_id"].astype(str)
    clinical = _load_clinical(global_root)
    for key in ("dataset", "participant_id", "group"):
        table[key] = table[key].astype(str)
        clinical[key] = clinical[key].astype(str)
    table = table.merge(clinical, on=["dataset", "participant_id", "group"], how="left")
    raw_stats, _ = _correlations(table, outlier_sd=None)
    stats, excluded = _correlations(table, outlier_sd=outlier_sd)
    statistics_path = output_root / "statistics" / "abba_burst_clinical_correlations.csv"
    statistics_path.parent.mkdir(parents=True, exist_ok=True)
    stats.to_csv(statistics_path, index=False)
    raw_stats.to_csv(output_root / "statistics" / "abba_burst_clinical_correlations_untrimmed.csv", index=False)
    excluded.to_csv(output_root / "statistics" / "abba_burst_clinical_x_outlier_exclusions.csv", index=False)
    figures = output_root / "figures"
    for band in SELECTED_BANDS:
        for outcome in ("cognitive", "updrs"):
            _plot_band(table, stats, band, outcome, figures / f"abba_burst_clinical_{outcome}_{band}.png", outlier_sd=outlier_sd)
    return {
        "participant_rows": len(table),
        "correlation_rows": len(stats),
        "x_outliers_excluded": len(excluded),
        "figures": len(SELECTED_BANDS) * 2,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/rhythmicity"))
    parser.add_argument("--global-root", type=Path, default=Path("outputs/global"))
    parser.add_argument("--outlier-sd", type=float, default=OUTLIER_SD,
                        help="Exclude x-axis burst values beyond this many SDs within each panel (default: 3)")
    args = parser.parse_args()
    result = run(args.output_root, args.global_root, outlier_sd=args.outlier_sd)
    print(result)


if __name__ == "__main__":
    main()
