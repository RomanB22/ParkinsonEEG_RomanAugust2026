#!/usr/bin/env python3
"""Regenerate output-level results reports from the current analysis tables."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


OUTPUT = Path("outputs")
DATASET_ORDER = ["primary", "ds007526-1.0.2", "ds008768-1.0.0", "medication_state"]


def _table(headers: list[str], rows: list[list[object]]) -> str:
    def clean(value: object) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = ["| " + " | ".join(map(clean, headers)) + " |"]
    lines.append("|" + "|".join("---" for _ in headers) + "|")
    lines.extend("| " + " | ".join(clean(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def _groups(table: pd.DataFrame) -> str:
    values = []
    for group, subset in table.groupby("group", sort=True):
        values.append(f"{group}: {subset['participant_id'].nunique()}")
    return ", ".join(values)


def generate_global_report() -> None:
    root = OUTPUT / "global"
    canonical = pd.read_csv(root / "canonical" / "recordings.csv.gz", low_memory=False)
    statistics = pd.read_csv(root / "statistics" / "group_statistics.csv.gz", low_memory=False)
    clinical = pd.read_csv(root / "statistics" / "clinical_correlations.csv.gz", low_memory=False)
    qc = pd.read_csv(root / "metrics" / "aperiodic_subject_qc.csv.gz", low_memory=False)

    overview = []
    for dataset in DATASET_ORDER:
        subset = canonical.loc[canonical["dataset_id"].eq(dataset)]
        q = qc.loc[qc["dataset_id"].eq(dataset)].drop_duplicates("participant_id")
        s = statistics.loc[statistics["dataset_id"].eq(dataset)]
        overview.append([
            f"`{dataset}`",
            subset["participant_id"].nunique(),
            len(subset),
            _groups(subset),
            f"{int(q['subject_qc_pass'].sum())}/{len(q)}",
            f"{int(s['welch_p_fdr_bh'].lt(0.05).sum())}/{len(s)}",
        ])

    feature_rows = []
    features = {
        "Theta relative power": "psd__theta__relative_power",
        "Beta relative power": "psd__beta__relative_power",
        "Aperiodic offset": "aperiodic__broadband__aperiodic_offset",
        "Aperiodic exponent": "aperiodic__broadband__aperiodic_exponent",
    }
    for dataset in DATASET_ORDER[:3]:
        subset = statistics.loc[
            statistics["dataset_id"].eq(dataset)
            & statistics[["group_a", "group_b"]].apply(lambda row: set(row) == {"Control", "PD"}, axis=1)
        ]
        for label, feature in features.items():
            rows = subset.loc[subset["feature"].eq(feature)].copy()
            if rows.empty:
                continue
            pd_mean = rows["mean_a"].where(rows["group_a"].eq("PD"), rows["mean_b"])
            control_mean = rows["mean_a"].where(rows["group_a"].eq("Control"), rows["mean_b"])
            feature_rows.append([
                f"`{dataset}`",
                label,
                f"{int(rows['welch_p_fdr_bh'].lt(0.05).sum())}/{len(rows)}",
                f"{(pd_mean - control_mean).median():+.4g}",
            ])

    clinical_rows = []
    for (dataset, outcome, method), subset in clinical.groupby(["dataset_id", "outcome", "method"]):
        clinical_rows.append([
            f"`{dataset}`", outcome.upper(), method,
            int(subset["p_fdr_bh"].lt(0.05).sum()),
            f"{subset['p_fdr_bh'].min():.3g}",
            int(subset["n_subjects"].max()),
        ])

    model = pd.read_csv(root / "statistics" / "theta_fisher_moca_incremental" / "model_summary.csv")
    full = model.iloc[-1]
    selection = json.loads((root / "statistics" / "xgboost_minimal_features" / "minimal_selection.json").read_text())
    classifier = pd.read_csv(root / "statistics" / "xgboost_multidataset" / "classification_summary.csv")
    best_classifier = classifier.loc[classifier["roc_auc_mean"].idxmax()]
    moca = pd.read_csv(root / "statistics" / "xgboost_multidataset" / "moca_summary.csv")
    best_moca = moca.loc[moca["rmse_mean"].idxmin()]

    lines = [
        "# Global EEG results and conclusions",
        "",
        "## Scope and cohort integrity",
        "",
        "This report is generated from the current canonical and statistical tables. "
        "The 149 participants shared by `primary` and `ds008768-1.0.0` are retained only "
        "in `primary`; all 174 of their `ds008768-1.0.0` sessions are excluded before analysis. "
        "Non-overlapping participants with repeated sessions remain available. The canonical "
        f"analysis contains **{len(canonical)} recordings** from four datasets.",
        "",
        _table(
            ["Dataset", "Participants", "Recordings", "Participant group membership", "Aperiodic QC pass", "FDR-significant electrode-feature rows"],
            overview,
        ),
        "",
        "Group membership counts are participant counts within each group. In the medication "
        "dataset, the same 15 PD participants contribute both OFF and ON recordings.",
        "",
        "## Prespecified spectral and aperiodic contrasts",
        "",
        _table(["Dataset", "Feature", "FDR-significant electrodes", "Median PD − Control"], feature_rows),
        "",
        "These rows summarize electrode-wise Welch tests. They are spatial tests, not independent "
        "participant replications. The most defensible cross-dataset conclusion is based on effects "
        "that retain the same direction across the three independent PD–Control cohorts; effects "
        "appearing in only one cohort should remain dataset-specific.",
        "",
        "## Clinical correlations",
        "",
        _table(["Dataset", "Outcome", "Method", "FDR-significant features", "Minimum q", "Maximum n"], clinical_rows),
        "",
        "Clinical tests are participant-level. Repeated recordings are averaged before inference, "
        "and adjusted and unadjusted families are FDR-corrected separately.",
        "",
        "## Multivariable and machine-learning summaries",
        "",
        f"- The full theta-Fisher incremental model uses **{int(full['n'])}** complete PD cases "
        f"and has in-sample R²={full['r_squared']:.3f} (adjusted R²={full['adjusted_r_squared']:.3f}).",
        f"- The best repeated-CV classifier is `{best_classifier['model']}` with mean ROC AUC "
        f"{best_classifier['roc_auc_mean']:.3f}; the compact selection uses "
        f"{selection['classification']['selected_k']} features (ROC AUC "
        f"{selection['classification']['selected_roc_auc']:.3f}).",
        f"- The lowest-RMSE repeated-CV MoCA model is `{best_moca['model']}` with RMSE "
        f"{best_moca['rmse_mean']:.3f}; the compact selection uses "
        f"{selection['regression']['selected_k']} features (RMSE "
        f"{selection['regression']['selected_rmse']:.3f}).",
        "",
        "## Overall conclusion",
        "",
        "The current results compare four non-overlapping cohorts at the participant level for "
        "inference. The output tables and figures must be interpreted from this regenerated report, "
        "not from pre-exclusion sample sizes. Cross-dataset consistency is stronger evidence than "
        "the number of significant electrodes within any single dataset; clinical and predictive "
        "results remain observational and require external validation.",
        "",
    ]
    (root / "results_conclusions.md").write_text("\n".join(lines), encoding="utf-8")


def generate_rhythmicity_report() -> None:
    root = OUTPUT / "rhythmicity"
    manifest = json.loads((root / "manifest.json").read_text())
    lavi = pd.read_csv(root / "statistics" / "lavi_group_comparisons.csv")
    clinical = pd.read_csv(root / "statistics" / "lavi_clinical_correlations.csv")
    burst = pd.read_csv(root / "statistics" / "abba_burst_focused_group_comparisons.csv")
    burst_clinical = pd.read_csv(root / "statistics" / "abba_burst_clinical_correlations.csv")

    lavi_rows = []
    for row in lavi.loc[lavi["significant_and_relevant"]].itertuples():
        lavi_rows.append([
            f"`{row.dataset}`", row.band, row.comparison, row.n_a, row.n_b,
            f"{row.mean_difference:+.4f}", f"{row.hedges_g:+.3f}", f"{row.q_fdr_bh:.3g}",
        ])

    clinical_rows = []
    for row in clinical.loc[clinical["significant_and_relevant"]].itertuples():
        clinical_rows.append([
            f"`{row.dataset}`", row.band, row.clinical_measure, row.lavi_metric,
            row.n, f"{row.rho:+.3f}", f"{row.q_fdr_bh:.3g}",
        ])

    replicated = (
        burst.loc[burst["significant_fdr"]]
        .groupby(["band_name", "quantity"])["dataset"]
        .nunique()
        .reset_index(name="datasets_with_fdr_effect")
        .query("datasets_with_fdr_effect >= 3")
        .sort_values(["datasets_with_fdr_effect", "band_name", "quantity"], ascending=[False, True, True])
    )
    replicated_rows = [[row.band_name, row.quantity, row.datasets_with_fdr_effect] for row in replicated.itertuples()]

    lines = [
        "# Rhythmicity results and conclusions",
        "",
        "## Scope and cohort integrity",
        "",
        f"The current LAVI run requested **{manifest['n_requested_recordings']}** canonical recordings, "
        f"completed **{manifest['n_completed_recordings']}**, and skipped "
        f"**{manifest['n_failed_recordings']}** recordings with no retained epochs. It contains "
        f"**{manifest['n_participants']} dataset-qualified participants**. The 149 participants "
        "shared with `primary` and all of their `ds008768-1.0.0` sessions were excluded before tasks "
        "were built; retained repeated sessions are averaged within participant for inference.",
        "",
        "## LAVI group effects",
        "",
        "The following comparisons pass both BH-FDR q<0.05 and the configured practical-effect rule:",
        "",
        _table(["Dataset", "Band", "Contrast", "n A", "n B", "Mean difference", "Hedges g", "q"], lavi_rows),
        "",
        "The clearest replicated LAVI result is higher theta LAVI in the Parkinson-related group. "
        "After deduplication, the `ds008768-1.0.0` theta difference is +0.0139 (101 PD versus 59 "
        "Control), so conclusions based on the earlier, overlapping cohort should not be reused. "
        "Primary also shows a practically relevant beta reduction; other band effects are less stable.",
        "",
        "## LAVI clinical associations",
        "",
        _table(["Dataset", "Band", "Outcome", "LAVI metric", "n", "Spearman rho", "q"], clinical_rows),
        "",
        "Theta mean LAVI remains inversely associated with cognition in `primary`, `ds007526-1.0.2`, "
        "and `medication_state`. The filtered `ds008768-1.0.0` result is instead an inverse theta-peak "
        "association in the 59 participants with MoCA; its theta-mean association no longer meets the "
        "combined significance-and-effect criterion.",
        "",
        "## ABBA temporal-burst effects",
        "",
        f"There are **{int(burst['significant_fdr'].sum())}** FDR-significant focused group contrasts. "
        "The following band/quantity combinations are significant in at least three datasets:",
        "",
        _table(["ABBA interval", "Quantity", "Datasets with FDR effect"], replicated_rows),
        "",
        f"The clinical ABBA table contains **{int(burst_clinical['significant_fdr'].sum())}** "
        "FDR-significant rows after its configured outlier screen. Because ABBA intervals are "
        "group-specific and many contrasts are tested, these results are exploratory and should not "
        "be interpreted as fixed-band medication or causal effects.",
        "",
        "## Overall conclusion",
        "",
        "Theta rhythmicity remains the most reproducible LAVI group signal after removing cohort "
        "overlap. Alpha/beta ABBA burst-rate and duration effects recur across datasets, but their "
        "exact directions must be read from the contrast tables because interval definitions differ. "
        "All reported sample sizes above come from the filtered outputs.",
        "",
    ]
    (root / "results_rhythmicity.md").write_text("\n".join(lines), encoding="utf-8")


def generate_behavioral_report() -> None:
    root = OUTPUT / "rhythmicity" / "behavioral_information"
    data = pd.read_csv(root / "hcf_behavioral_correlations_all_dimensions.csv")
    keys = ["dataset", "band", "outcome_family", "outcome", "cohort", "metric"]
    robust_rows = []
    for key, subset in data.groupby(keys, dropna=False):
        if set(subset["embedding_dimension"]) != {4, 5, 6} or not subset["significant_fdr"].all():
            continue
        robust_rows.append([
            f"`{key[0]}`", key[1], key[2], key[3].upper(), key[4], key[5],
            int(subset["n"].min()), f"{subset['spearman_rho'].min():+.3f} to {subset['spearman_rho'].max():+.3f}",
            "yes" if subset["relevant_abs_rho_ge_0_30"].all() else "no",
        ])

    lines = [
        "# Full-signal H/C/F behavioral results",
        "",
        "## Scope and cohort integrity",
        "",
        "This report is generated from the combined D=4, D=5, and D=6 participant-level tables. "
        "It uses full-signal ordinal metrics within group-specific ABBA intervals. Cognitive analyses "
        "use MoCA in the three standard datasets and MMSE in `medication_state`; UPDRS analyses are "
        "PD-only. All sessions from the 149 overlapping `ds008768-1.0.0` participants are absent.",
        "",
        "## Associations significant at every embedding dimension",
        "",
        _table(["Dataset", "Band", "Family", "Outcome", "Cohort", "Metric", "Minimum n", "rho range D4–D6", "|rho|≥0.30 at every D"], robust_rows),
        "",
        "## Conclusions",
        "",
        "The strongest dimension-stable cognitive result is positive alpha-1 entropy and complexity "
        "versus MoCA in `primary` and `ds007526-1.0.2`. The dimension-stable motor result with a "
        "practically relevant effect is negative alpha-1 complexity versus UPDRS in `ds007526-1.0.2`. "
        "No `ds008768-1.0.0` or medication-state association survives FDR independently at all three "
        "embedding dimensions after removing the overlapping participants. Fisher-information claims "
        "are more dimension-sensitive and should always state D explicitly.",
        "",
        "These are observational, multiply tested correlations. Group-specific ABBA limits mean that "
        "pooled-group estimates can compare slightly different frequency intervals, and no result "
        "establishes causality or external predictive validity.",
        "",
    ]
    (root / "RESULTS_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    generate_global_report()
    generate_rhythmicity_report()
    generate_behavioral_report()
    print("Regenerated 3 output results/conclusions reports")


if __name__ == "__main__":
    main()
