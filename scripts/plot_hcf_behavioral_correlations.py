#!/usr/bin/env python3
"""Plot full-signal ABBA-band H/C/F associations with clinical scores.

The input is the participant-level ABBA ordinal table.  ``full_signal`` here
means the complete epoch-concatenated trace after filtering to an ABBA
frequency interval; it is distinct from the ``within_bout`` estimate.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


METRICS = {
    "entropy": "H (permutation entropy)",
    "complexity": "C (statistical complexity)",
    "fisher_information": "F (Fisher information)",
}
METRIC_SHORT = {"entropy": "H", "complexity": "C", "fisher_information": "F"}
DATASET_LABELS = {
    "primary": "Primary",
    "ds007526-1.0.2": "ds007526",
    "ds008768-1.0.0": "ds008768",
    "medication_state": "Medication state",
}
GROUP_COLORS = {
    "Control": "#0072B2",
    "PD": "#D55E00",
    "PD_OFF": "#7570B3",
    "PD_ON": "#009E73",
    "PD_combined": "#D55E00",
}
MINIMUM_N = 5


def _fdr_bh(values: pd.Series) -> np.ndarray:
    p = pd.to_numeric(values, errors="coerce").to_numpy(float)
    adjusted = np.full(len(p), np.nan)
    valid = np.isfinite(p)
    if not valid.any():
        return adjusted
    selected = p[valid]
    order = np.argsort(selected)
    ranked = selected[order] * len(selected) / np.arange(1, len(selected) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    restored = np.empty_like(ranked)
    restored[order] = np.minimum(ranked, 1.0)
    adjusted[valid] = restored
    return adjusted


def _correlation_row(
    frame: pd.DataFrame,
    *,
    dataset: str,
    band: str,
    outcome_family: str,
    outcome: str,
    cohort: str,
    metric: str,
) -> dict[str, object]:
    paired = frame[["participant_id", outcome, metric]].copy()
    paired[outcome] = pd.to_numeric(paired[outcome], errors="coerce")
    paired[metric] = pd.to_numeric(paired[metric], errors="coerce")
    paired = paired.dropna(subset=[outcome, metric])
    if len(paired) >= MINIMUM_N and paired[[outcome, metric]].nunique().min() > 1:
        result = spearmanr(paired[outcome], paired[metric])
        rho, p_value = float(result.statistic), float(result.pvalue)
    else:
        rho, p_value = np.nan, np.nan
    return {
        "dataset": dataset,
        "band": band,
        "scope": "full_signal",
        "outcome_family": outcome_family,
        "outcome": outcome,
        "cohort": cohort,
        "metric": metric,
        "n": int(len(paired)),
        "spearman_rho": rho,
        "p_value": p_value,
    }


def _subject_mean(frame: pd.DataFrame, outcome: str) -> pd.DataFrame:
    """Return one row per participant, preventing ON/OFF pseudoreplication."""
    numeric = [outcome, *METRICS]
    values = frame[["participant_id", *numeric]].copy()
    for column in numeric:
        values[column] = pd.to_numeric(values[column], errors="coerce")
    return values.groupby("participant_id", as_index=False)[numeric].mean()


def compute_correlations(table: pd.DataFrame, embedding_dimension: int | None = None) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    full = table.loc[table["scope"].eq("full_signal")].copy()
    for (dataset, band), frame in full.groupby(["dataset", "comparison_band_name"], sort=False):
        outcome = "moca" if frame["moca"].notna().any() else "mmse"
        cognitive_all = _subject_mean(frame, outcome)
        cognitive_pd = _subject_mean(
            frame.loc[frame["group"].astype(str).str.startswith("PD")], outcome
        )
        updrs_pd_rows = frame.loc[frame["group"].astype(str).str.startswith("PD")]
        updrs_pd = _subject_mean(updrs_pd_rows, "updrs")
        for metric in METRICS:
            rows.append(_correlation_row(
                cognitive_all, dataset=str(dataset), band=str(band),
                outcome_family="cognitive", outcome=outcome,
                cohort="all_participants", metric=metric,
            ))
            rows.append(_correlation_row(
                cognitive_pd, dataset=str(dataset), band=str(band),
                outcome_family="cognitive", outcome=outcome,
                cohort="pd_only", metric=metric,
            ))
            rows.append(_correlation_row(
                updrs_pd, dataset=str(dataset), band=str(band),
                outcome_family="motor", outcome="updrs",
                cohort="pd_only_state_mean", metric=metric,
            ))

            # Medication-state sensitivity analyses retain state-specific rows.
            if str(dataset) == "medication_state":
                for group, cohort in (("PD_OFF", "pd_off"), ("PD_ON", "pd_on")):
                    state = _subject_mean(frame.loc[frame["group"].eq(group)], "updrs")
                    rows.append(_correlation_row(
                        state, dataset=str(dataset), band=str(band),
                        outcome_family="motor", outcome="updrs",
                        cohort=cohort, metric=metric,
                    ))
                off = frame.loc[frame["group"].eq("PD_OFF"), ["participant_id", "updrs", metric]]
                on = frame.loc[frame["group"].eq("PD_ON"), ["participant_id", "updrs", metric]]
                paired = off.merge(on, on="participant_id", suffixes=("_off", "_on"))
                paired["updrs_change"] = paired["updrs_on"] - paired["updrs_off"]
                paired[metric] = paired[f"{metric}_on"] - paired[f"{metric}_off"]
                rows.append(_correlation_row(
                    paired, dataset=str(dataset), band=str(band),
                    outcome_family="motor_change", outcome="updrs_change",
                    cohort="pd_on_minus_off", metric=metric,
                ))

    result = pd.DataFrame(rows)
    result.insert(0, "embedding_dimension", embedding_dimension)
    result["q_fdr_bh"] = np.nan
    family = ["dataset", "outcome_family", "cohort", "scope"]
    for _, indices in result.groupby(family, sort=False).groups.items():
        result.loc[indices, "q_fdr_bh"] = _fdr_bh(result.loc[indices, "p_value"])
    result["significant_fdr"] = result["q_fdr_bh"] < 0.05
    result["relevant_abs_rho_ge_0_30"] = result["spearman_rho"].abs() >= 0.30
    return result.sort_values(["outcome_family", "dataset", "band", "cohort", "metric"])


def _plot_heatmap(
    stats: pd.DataFrame, family: str, cohort: str, output: Path,
    embedding_dimension: int | None = None,
) -> None:
    selected = stats.loc[
        stats["outcome_family"].eq(family) & stats["cohort"].eq(cohort)
    ].copy()
    datasets = [value for value in DATASET_LABELS if value in set(selected["dataset"])]
    bands_by_dataset = {
        dataset: list(dict.fromkeys(selected.loc[selected["dataset"].eq(dataset), "band"]))
        for dataset in datasets
    }
    rows = [(dataset, band) for dataset in datasets for band in bands_by_dataset[dataset]]
    matrix = np.full((len(rows), len(METRICS)), np.nan)
    q_values = np.full_like(matrix, np.nan)
    n_values = np.zeros_like(matrix)
    for row_index, (dataset, band) in enumerate(rows):
        for column_index, metric in enumerate(METRICS):
            match = selected.loc[
                selected["dataset"].eq(dataset)
                & selected["band"].eq(band)
                & selected["metric"].eq(metric)
            ]
            if not match.empty:
                matrix[row_index, column_index] = match.iloc[0]["spearman_rho"]
                q_values[row_index, column_index] = match.iloc[0]["q_fdr_bh"]
                n_values[row_index, column_index] = match.iloc[0]["n"]

    figure, axis = plt.subplots(figsize=(9.0, max(7.0, 0.42 * len(rows))))
    image = axis.imshow(matrix, cmap="RdBu_r", vmin=-0.7, vmax=0.7, aspect="auto")
    for row_index in range(len(rows)):
        for column_index in range(len(METRICS)):
            rho = matrix[row_index, column_index]
            if np.isfinite(rho):
                star = "*" if q_values[row_index, column_index] < 0.05 else ""
                axis.text(column_index, row_index, f"{rho:+.2f}{star}\nn={int(n_values[row_index, column_index])}",
                          ha="center", va="center", fontsize=8,
                          color="white" if abs(rho) > 0.42 else "black")
            else:
                axis.text(column_index, row_index, "NA", ha="center", va="center", fontsize=8, color="0.45")
    axis.set_xticks(range(len(METRICS)), ["H", "C", "F"])
    axis.set_yticks(
        range(len(rows)),
        [f"{DATASET_LABELS.get(dataset, dataset)}  |  {band}" for dataset, band in rows],
        fontsize=8,
    )
    previous = None
    for row_index, (dataset, _) in enumerate(rows):
        if previous is not None and dataset != previous:
            axis.axhline(row_index - 0.5, color="black", linewidth=1.2)
        previous = dataset
    label = "MoCA / MMSE" if family == "cognitive" else "UPDRS"
    population = "all participants" if family == "cognitive" else "PD only; ON/OFF averaged"
    dimension = f" — D={embedding_dimension}" if embedding_dimension is not None else ""
    axis.set_title(f"Full-signal ABBA-band H/C/F vs {label}{dimension}\nSpearman correlation ({population})", fontweight="bold")
    colorbar = figure.colorbar(image, ax=axis, shrink=0.72, pad=0.03)
    colorbar.set_label("Spearman ρ")
    figure.text(0.5, 0.01, "* BH-FDR q < 0.05 within dataset and analysis family", ha="center", fontsize=8)
    figure.tight_layout(rect=(0, 0.035, 1, 1))
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _plot_scatter_grid(
    table: pd.DataFrame, stats: pd.DataFrame, dataset: str, family: str, output: Path,
    embedding_dimension: int | None = None,
) -> None:
    data = table.loc[table["scope"].eq("full_signal") & table["dataset"].eq(dataset)].copy()
    outcome = "moca" if data["moca"].notna().any() else "mmse"
    if family == "motor":
        outcome = "updrs"
        data = data.loc[data["group"].astype(str).str.startswith("PD")]
        cohort = "pd_only_state_mean"
    else:
        cohort = "all_participants"
    bands = list(dict.fromkeys(data["comparison_band_name"]))
    figure, axes = plt.subplots(len(bands), 3, figsize=(12.5, 3.0 * len(bands)), squeeze=False)
    for row_index, band in enumerate(bands):
        band_frame = data.loc[data["comparison_band_name"].eq(band)]
        for column_index, metric in enumerate(METRICS):
            axis = axes[row_index, column_index]
            for group, group_frame in band_frame.groupby("group", sort=False):
                points = group_frame[["participant_id", outcome, metric]].copy()
                points[[outcome, metric]] = points[[outcome, metric]].apply(pd.to_numeric, errors="coerce")
                points = points.dropna(subset=[outcome, metric])
                axis.scatter(points[metric], points[outcome], s=22, alpha=0.70,
                             color=GROUP_COLORS.get(str(group), "#666666"),
                             label=str(group).replace("PD_", "PD-"))
            # A single descriptive line is fit to one-row-per-participant values.
            collapsed = _subject_mean(band_frame, outcome).dropna(subset=[outcome, metric])
            if len(collapsed) >= 3 and collapsed[metric].nunique() > 1:
                coefficients = np.polyfit(collapsed[metric], collapsed[outcome], 1)
                grid = np.linspace(collapsed[metric].min(), collapsed[metric].max(), 50)
                axis.plot(grid, np.polyval(coefficients, grid), color="0.15", linewidth=1.1)
            match = stats.loc[
                stats["dataset"].eq(dataset) & stats["band"].eq(band)
                & stats["outcome_family"].eq(family) & stats["cohort"].eq(cohort)
                & stats["metric"].eq(metric)
            ]
            if not match.empty:
                result = match.iloc[0]
                annotation = f"ρ={result.spearman_rho:+.2f}, q={result.q_fdr_bh:.3g}, n={int(result.n)}"
                axis.text(0.02, 0.97, annotation, transform=axis.transAxes, va="top", fontsize=8)
            axis.grid(alpha=0.18)
            axis.set_xlabel(METRICS[metric] if row_index == len(bands) - 1 else "")
            axis.set_ylabel(f"{band}\n{outcome.upper()}" if column_index == 0 else outcome.upper())
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        unique = dict(zip(labels, handles))
        figure.legend(unique.values(), unique.keys(), loc="lower center", ncol=len(unique), frameon=False)
    population = "all participants" if family == "cognitive" else "PD participants only"
    dimension = f" (D={embedding_dimension})" if embedding_dimension is not None else ""
    figure.suptitle(
        f"{DATASET_LABELS.get(dataset, dataset)}: full-signal ABBA-band H/C/F{dimension} vs {outcome.upper()}\n{population}",
        fontsize=14, fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0.025, 1, 0.975))
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)


def _write_readme(stats: pd.DataFrame, output: Path, embedding_dimension: int | None = None) -> None:
    primary = stats.loc[
        ((stats["outcome_family"].eq("cognitive") & stats["cohort"].eq("all_participants"))
         | (stats["outcome_family"].eq("motor") & stats["cohort"].eq("pd_only_state_mean")))
        & stats["significant_fdr"]
    ].sort_values("q_fdr_bh")
    lines = [
        f"# H/C/F behavioral associations{f' (D={embedding_dimension})' if embedding_dimension is not None else ''}",
        "",
        "Spearman correlations use participant-level `full_signal` ordinal metrics: the entire",
        "epoch-concatenated EEG trace after filtering to each group-specific ABBA interval.",
        "These are not broadband estimates and are not the `within_bout` metrics.",
        "",
        "Cognitive plots use MoCA in primary, ds007526, and ds008768, and MMSE in",
        "medication-state. Primary cognitive estimates include all participants; the CSV also",
        "contains PD-only sensitivity estimates. UPDRS estimates exclude controls. Repeated",
        "PD-OFF/PD-ON measurements are averaged by participant for the medication-state",
        "cross-sectional estimate; state-specific and paired-change results are also in the CSV.",
        "BH-FDR is applied within each dataset, outcome family, cohort, and signal scope across",
        "all available ABBA intervals and H/C/F tests.",
        "",
        f"Primary plotted correlations surviving BH-FDR: {len(primary)}.",
    ]
    if len(primary):
        lines.extend(["", "| Dataset | Band | Outcome | Metric | n | rho | q |", "|---|---|---|---:|---:|---:|---:|"])
        for row in primary.itertuples():
            lines.append(
                f"| {row.dataset} | {row.band} | {row.outcome.upper()} | {row.metric} | "
                f"{row.n} | {row.spearman_rho:.3f} | {row.q_fdr_bh:.3g} |"
            )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _combine_dimension_outputs(output_dir: Path) -> None:
    paths = sorted(output_dir.glob("D*/hcf_behavioral_correlations.csv"))
    if not paths:
        raise FileNotFoundError(f"No D*/hcf_behavioral_correlations.csv files under {output_dir}")
    combined = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    combined.to_csv(
        output_dir / "hcf_behavioral_correlations_all_dimensions.csv",
        index=False,
        float_format="%.10g",
    )

    selected = combined.loc[
        combined["band"].eq("alpha_1")
        & (
            (combined["outcome_family"].eq("cognitive") & combined["cohort"].eq("all_participants"))
            | (combined["outcome_family"].eq("motor") & combined["cohort"].eq("pd_only_state_mean"))
        )
    ].copy()
    dimensions = sorted(selected["embedding_dimension"].dropna().astype(int).unique())
    datasets = [dataset for dataset in DATASET_LABELS if dataset in set(selected["dataset"])]
    rows = [
        (family, dataset, metric)
        for family in ("cognitive", "motor")
        for dataset in datasets
        for metric in METRICS
    ]
    matrix = np.full((len(rows), len(dimensions)), np.nan)
    q_values = np.full_like(matrix, np.nan)
    for row_index, (family, dataset, metric) in enumerate(rows):
        for column_index, dimension in enumerate(dimensions):
            match = selected.loc[
                selected["outcome_family"].eq(family)
                & selected["dataset"].eq(dataset)
                & selected["metric"].eq(metric)
                & selected["embedding_dimension"].eq(dimension)
            ]
            if not match.empty:
                matrix[row_index, column_index] = match.iloc[0]["spearman_rho"]
                q_values[row_index, column_index] = match.iloc[0]["q_fdr_bh"]
    figure, axis = plt.subplots(figsize=(7.5, 0.38 * len(rows) + 2.2))
    image = axis.imshow(matrix, cmap="RdBu_r", vmin=-0.55, vmax=0.55, aspect="auto")
    for row_index in range(len(rows)):
        for column_index in range(len(dimensions)):
            rho = matrix[row_index, column_index]
            if np.isfinite(rho):
                star = "*" if q_values[row_index, column_index] < 0.05 else ""
                axis.text(column_index, row_index, f"{rho:+.2f}{star}", ha="center", va="center", fontsize=8,
                          color="white" if abs(rho) > 0.38 else "black")
    axis.set_xticks(range(len(dimensions)), [f"D={dimension}" for dimension in dimensions])
    axis.set_yticks(
        range(len(rows)),
        [f"{'MoCA/MMSE' if family == 'cognitive' else 'UPDRS'} | {DATASET_LABELS[dataset]} | {METRIC_SHORT[metric]}"
         for family, dataset, metric in rows],
        fontsize=8,
    )
    axis.axhline(len(datasets) * len(METRICS) - 0.5, color="black", linewidth=1.3)
    axis.set_title("Alpha₁ full-signal H/C/F behavioral correlations across embedding dimensions", fontweight="bold")
    colorbar = figure.colorbar(image, ax=axis, shrink=0.72, pad=0.03)
    colorbar.set_label("Spearman ρ")
    figure.text(0.5, 0.01, "* BH-FDR q < 0.05 in the corresponding dimension-specific analysis", ha="center", fontsize=8)
    figure.tight_layout(rect=(0, 0.035, 1, 1))
    figure.savefig(output_dir / "hcf_alpha1_dimension_robustness.png", dpi=220, bbox_inches="tight")
    plt.close(figure)

    lines = [
        "# H/C/F behavioral associations across embedding dimensions",
        "",
        f"Included dimensions: {', '.join(f'D={value}' for value in dimensions)} (delay tau=1).",
        "",
        "Each dimension subfolder contains its complete correlation table, cognitive and UPDRS",
        "heatmaps, dataset-level scatterplot grids, and methods/results summary. The combined CSV",
        "contains every dimension-specific estimate. The robustness figure compares the alpha_1",
        "results directly because that is the main replicated behavioral association.",
        "",
        "The authoritative dimension-specific outputs are in `D4/`, `D5/`, and `D6/`.",
        "Any older unlabelled plots at this directory's top level correspond to the initial D=5 run.",
    ]
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Combined {len(paths)} dimensions into {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("outputs/rhythmicity/metrics/abba_ordinal_participant_metrics.csv.gz"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/rhythmicity/behavioral_information"),
    )
    parser.add_argument("--embedding-dimension", type=int)
    parser.add_argument(
        "--combine-dimensions",
        action="store_true",
        help="Combine existing D* subfolder tables and make a cross-dimension alpha_1 plot.",
    )
    args = parser.parse_args()
    if args.combine_dimensions:
        _combine_dimension_outputs(args.output_dir)
        return
    table = pd.read_csv(args.input, low_memory=False)
    required = {"dataset", "participant_id", "group", "comparison_band_name", "scope", "moca", "mmse", "updrs", *METRICS}
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"Input is missing required columns: {missing}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scatter_dir = args.output_dir / "scatterplots"
    scatter_dir.mkdir(parents=True, exist_ok=True)

    stats = compute_correlations(table, args.embedding_dimension)
    stats.to_csv(args.output_dir / "hcf_behavioral_correlations.csv", index=False, float_format="%.10g")
    _plot_heatmap(stats, "cognitive", "all_participants", args.output_dir / "hcf_cognitive_full_signal_heatmap.png", args.embedding_dimension)
    _plot_heatmap(stats, "motor", "pd_only_state_mean", args.output_dir / "hcf_updrs_full_signal_heatmap.png", args.embedding_dimension)
    for dataset in DATASET_LABELS:
        if dataset not in set(table["dataset"]):
            continue
        _plot_scatter_grid(table, stats, dataset, "cognitive", scatter_dir / f"{dataset}__cognitive.png", args.embedding_dimension)
        _plot_scatter_grid(table, stats, dataset, "motor", scatter_dir / f"{dataset}__updrs.png", args.embedding_dimension)
    _write_readme(stats, args.output_dir / "README.md", args.embedding_dimension)
    print(f"Wrote {len(stats)} correlations and plots to {args.output_dir}")


if __name__ == "__main__":
    main()
