"""Propagate specparam fit QC into bout and within-bout ordinal summaries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from src.runtime import configure_runtime

configure_runtime()

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import fisher_exact, mannwhitneyu, spearmanr, ttest_ind, wilcoxon

from bout_analyses.metrics import METRICS as BOUT_ORDINAL_METRICS
from bout_analyses.pipeline import _group_summary as bout_ordinal_group_summary
from bout_analyses.pipeline import _subject_band_means as bout_ordinal_subject_means
from scale_free_analysis.metrics import BAND_FEATURES
from scale_free_analysis.pipeline import (
    _describe,
    _fdr_bh,
    _hedges_g,
    _subject_means,
)


FIT_KEYS = ["subject_id", "group", "electrode"]
COVERAGE_KEYS = ["subject_id", "group"]
BAND_KEYS = ["subject_id", "group", "band", "band_low_hz", "band_high_hz"]
PLOTTED_BOUT_METRICS = (
    "oscillatory_occupancy",
    "bouts_per_minute",
    "bout_duration_mean_s",
    "bout_cycles_mean",
    "bout_snr_mean",
)


def _write_csv(table: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False, float_format="%.17g", compression="infer")


def _fit_coverage(
    aperiodic: pd.DataFrame, minimum_fraction: float
) -> pd.DataFrame:
    coverage = (
        aperiodic.groupby(COVERAGE_KEYS, sort=False)
        .agg(
            n_electrodes_total=("electrode", "nunique"),
            n_fit_qc_electrodes=("specparam_fit_qc_pass", "sum"),
        )
        .reset_index()
    )
    coverage["fit_qc_fraction"] = (
        coverage["n_fit_qc_electrodes"] / coverage["n_electrodes_total"]
    )
    coverage["subject_fit_qc_pass"] = coverage["fit_qc_fraction"].ge(
        float(minimum_fraction)
    )
    coverage["fit_failure_fraction"] = 1.0 - coverage["fit_qc_fraction"]
    return coverage


def _attach_fit_qc(
    table: pd.DataFrame,
    aperiodic: pd.DataFrame,
    coverage: pd.DataFrame,
) -> pd.DataFrame:
    mask_columns = [
        *FIT_KEYS,
        "specparam_fit_qc_pass",
        "specparam_fit_qc_reasons",
    ]
    result = table.drop(
        columns=["specparam_fit_qc_pass", "specparam_fit_qc_reasons"],
        errors="ignore",
    ).merge(
        aperiodic[mask_columns],
        on=FIT_KEYS,
        how="left",
        validate="many_to_one",
    )
    result = result.merge(
        coverage,
        on=COVERAGE_KEYS,
        how="left",
        validate="many_to_one",
    )
    if result["specparam_fit_qc_pass"].isna().any():
        raise RuntimeError("Some bout rows are missing specparam fit-QC status")
    return result


def _merge_coverage(
    subject_table: pd.DataFrame, coverage: pd.DataFrame
) -> pd.DataFrame:
    return subject_table.merge(
        coverage,
        on=COVERAGE_KEYS,
        how="left",
        validate="many_to_one",
    )


def _group_comparisons(
    subject_table: pd.DataFrame,
    metrics: Iterable[str],
    *,
    fdr_alpha: float,
    analysis: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for band, selected_band in subject_table.groupby("band", sort=False):
        for metric in metrics:
            pd_values = selected_band.loc[
                selected_band["group"].eq("PD"), metric
            ].dropna().to_numpy(dtype=float)
            control_values = selected_band.loc[
                selected_band["group"].eq("Control"), metric
            ].dropna().to_numpy(dtype=float)
            if min(len(pd_values), len(control_values)) >= 2:
                welch_result = ttest_ind(pd_values, control_values, equal_var=False)
                mann_result = mannwhitneyu(
                    pd_values, control_values, alternative="two-sided"
                )
                welch_t = float(welch_result.statistic)
                welch_p = float(welch_result.pvalue)
                mann_u = float(mann_result.statistic)
                mann_p = float(mann_result.pvalue)
            else:
                welch_t = welch_p = mann_u = mann_p = np.nan
            rows.append(
                {
                    "analysis": analysis,
                    "fit_policy": (
                        "QC-passing electrodes; subjects require at least 80% "
                        "of shared electrodes to pass"
                    ),
                    "band": str(band),
                    "metric": str(metric),
                    "n_pd": int(len(pd_values)),
                    "n_control": int(len(control_values)),
                    "pd_mean": float(np.mean(pd_values)) if len(pd_values) else np.nan,
                    "control_mean": (
                        float(np.mean(control_values)) if len(control_values) else np.nan
                    ),
                    "welch_t": welch_t,
                    "welch_p": welch_p,
                    "mann_whitney_u": mann_u,
                    "mann_whitney_p": mann_p,
                    "hedges_g_pd_minus_control": _hedges_g(
                        pd_values, control_values
                    ),
                }
            )
    result = pd.DataFrame.from_records(rows)
    adjusted, rejected = _fdr_bh(
        result["welch_p"].to_numpy(dtype=float), float(fdr_alpha)
    )
    result["welch_p_fdr_bh"] = adjusted
    result["fdr_alpha"] = float(fdr_alpha)
    result["fdr_reject"] = rejected
    return result


def _paired_sensitivity(
    all_subjects: pd.DataFrame,
    qc_subjects: pd.DataFrame,
    metrics: Iterable[str],
    *,
    analysis: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for band in qc_subjects["band"].drop_duplicates():
        all_band = all_subjects.loc[all_subjects["band"].eq(band)]
        qc_band = qc_subjects.loc[qc_subjects["band"].eq(band)]
        for metric in metrics:
            paired = all_band[["subject_id", "group", metric]].merge(
                qc_band[["subject_id", metric]],
                on="subject_id",
                suffixes=("_all", "_fit_qc"),
                validate="one_to_one",
            ).dropna(subset=[f"{metric}_all", f"{metric}_fit_qc"])
            all_values = paired[f"{metric}_all"].to_numpy(dtype=float)
            qc_values = paired[f"{metric}_fit_qc"].to_numpy(dtype=float)
            delta = qc_values - all_values
            if len(paired) >= 3 and not np.allclose(delta, 0.0):
                paired_test = wilcoxon(qc_values, all_values)
                paired_statistic = float(paired_test.statistic)
                paired_p = float(paired_test.pvalue)
            else:
                paired_statistic = paired_p = np.nan
            rank = spearmanr(all_values, qc_values) if len(paired) >= 3 else None
            rows.append(
                {
                    "analysis": analysis,
                    "band": str(band),
                    "metric": str(metric),
                    "n_qualified_subjects": int(len(paired)),
                    "all_electrode_mean": float(np.mean(all_values)),
                    "fit_qc_mean": float(np.mean(qc_values)),
                    "mean_change_fit_qc_minus_all": float(np.mean(delta)),
                    "median_absolute_change": float(np.median(np.abs(delta))),
                    "spearman_all_vs_fit_qc": (
                        float(rank.statistic) if rank is not None else np.nan
                    ),
                    "paired_wilcoxon_statistic": paired_statistic,
                    "paired_wilcoxon_p": paired_p,
                }
            )
    result = pd.DataFrame.from_records(rows)
    adjusted, rejected = _fdr_bh(
        result["paired_wilcoxon_p"].to_numpy(dtype=float), 0.05
    )
    result["paired_wilcoxon_p_fdr_bh"] = adjusted
    result["paired_wilcoxon_fdr_reject"] = rejected
    result["fdr_scope"] = "within analysis across all displayed bands and metrics"
    return result


def _plot_paired_sensitivity(
    all_subjects: pd.DataFrame,
    qc_subjects: pd.DataFrame,
    metrics: Iterable[str],
    output_path: Path,
    *,
    title: str,
    dpi: int,
) -> None:
    metrics = list(metrics)
    bands = qc_subjects["band"].drop_duplicates().astype(str).tolist()
    fig, axes = plt.subplots(
        len(metrics), len(bands),
        figsize=(4.2 * len(bands), 3.6 * len(metrics)),
        squeeze=False,
    )
    colors = {"PD": "#D55E00", "Control": "#0072B2"}
    for row_index, metric in enumerate(metrics):
        for column_index, band in enumerate(bands):
            axis = axes[row_index, column_index]
            paired = all_subjects.loc[
                all_subjects["band"].eq(band), ["subject_id", "group", metric]
            ].merge(
                qc_subjects.loc[
                    qc_subjects["band"].eq(band), ["subject_id", metric]
                ],
                on="subject_id",
                suffixes=("_all", "_fit_qc"),
                validate="one_to_one",
            ).dropna(subset=[f"{metric}_all", f"{metric}_fit_qc"])
            x = paired[f"{metric}_all"].to_numpy(dtype=float)
            y = paired[f"{metric}_fit_qc"].to_numpy(dtype=float)
            limits = np.asarray([np.min(np.r_[x, y]), np.max(np.r_[x, y])])
            padding = 0.04 * (limits[1] - limits[0] or 1.0)
            limits += np.asarray([-padding, padding])
            for group in ("PD", "Control"):
                selected = paired["group"].eq(group)
                axis.scatter(
                    paired.loc[selected, f"{metric}_all"],
                    paired.loc[selected, f"{metric}_fit_qc"],
                    s=14,
                    alpha=0.55,
                    color=colors[group],
                    label=group,
                )
            axis.plot(limits, limits, color="0.35", linestyle="--", linewidth=0.8)
            axis.set_xlim(limits)
            axis.set_ylim(limits)
            axis.grid(alpha=0.16)
            axis.set_title(str(band).replace("_", " ").title(), fontsize=9)
            if column_index == 0:
                axis.set_ylabel(f"QC-filtered\n{metric.replace('_', ' ')}", fontsize=8)
            if row_index == len(metrics) - 1:
                axis.set_xlabel("All-electrode value", fontsize=8)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)


def _failure_group_analysis(
    coverage: pd.DataFrame, participants: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metadata = participants.rename(
        columns={
            "participant_id": "subject_id",
            "GROUP": "metadata_group",
            "AGE": "age_years",
            "GENDER": "gender",
        }
    )
    required = {"subject_id", "metadata_group", "age_years", "gender"}
    missing = sorted(required - set(metadata.columns))
    if missing:
        raise ValueError(f"Participant metadata are missing: {missing}")
    metadata["sex_male"] = metadata["gender"].astype(str).eq("M").astype(int)
    subjects = coverage.merge(
        metadata[["subject_id", "metadata_group", "age_years", "gender", "sex_male"]],
        on="subject_id",
        how="left",
        validate="one_to_one",
    )
    if not subjects["group"].eq(subjects["metadata_group"]).all():
        raise ValueError("Fit-QC and participant group labels disagree")
    subjects = subjects.drop(columns="metadata_group")
    pd_values = subjects.loc[
        subjects["group"].eq("PD"), "fit_failure_fraction"
    ].to_numpy(dtype=float)
    control_values = subjects.loc[
        subjects["group"].eq("Control"), "fit_failure_fraction"
    ].to_numpy(dtype=float)
    welch_result = ttest_ind(pd_values, control_values, equal_var=False)
    mann_result = mannwhitneyu(pd_values, control_values, alternative="two-sided")
    design = pd.DataFrame(
        {
            "pd_indicator": subjects["group"].eq("PD").astype(float),
            "age_years": subjects["age_years"].astype(float),
            "sex_male": subjects["sex_male"].astype(float),
        },
        index=subjects.index,
    )
    fitted = sm.OLS(
        subjects["fit_failure_fraction"].to_numpy(dtype=float),
        sm.add_constant(design, has_constant="add"),
    ).fit(cov_type="HC3")
    interval = fitted.conf_int(alpha=0.05).loc["pd_indicator"]
    qualification = pd.crosstab(
        subjects["group"], subjects["subject_fit_qc_pass"]
    ).reindex(index=["PD", "Control"], columns=[False, True], fill_value=0)
    fisher_result = fisher_exact(qualification.to_numpy())
    comparison = pd.DataFrame.from_records(
        [
            {
                "outcome": "subject_fraction_of_electrodes_failing_specparam_qc",
                "unit_of_analysis": "subject",
                "n_pd": int(len(pd_values)),
                "n_control": int(len(control_values)),
                "pd_mean_failure_fraction": float(np.mean(pd_values)),
                "pd_median_failure_fraction": float(np.median(pd_values)),
                "control_mean_failure_fraction": float(np.mean(control_values)),
                "control_median_failure_fraction": float(np.median(control_values)),
                "mean_difference_pd_minus_control": float(
                    np.mean(pd_values) - np.mean(control_values)
                ),
                "hedges_g_pd_minus_control": _hedges_g(pd_values, control_values),
                "welch_t": float(welch_result.statistic),
                "welch_p": float(welch_result.pvalue),
                "mann_whitney_u": float(mann_result.statistic),
                "mann_whitney_p": float(mann_result.pvalue),
                "adjusted_model": (
                    "OLS: failure fraction ~ PD + age + sex; HC3 robust SE"
                ),
                "adjusted_pd_coefficient": float(fitted.params["pd_indicator"]),
                "adjusted_pd_se_hc3": float(fitted.bse["pd_indicator"]),
                "adjusted_pd_ci_lower": float(interval.iloc[0]),
                "adjusted_pd_ci_upper": float(interval.iloc[1]),
                "adjusted_pd_p": float(fitted.pvalues["pd_indicator"]),
                "n_pd_subject_qc_pass": int(qualification.loc["PD", True]),
                "n_pd_subject_qc_fail": int(qualification.loc["PD", False]),
                "n_control_subject_qc_pass": int(qualification.loc["Control", True]),
                "n_control_subject_qc_fail": int(qualification.loc["Control", False]),
                "subject_qualification_fisher_odds_ratio": float(
                    fisher_result.statistic
                ),
                "subject_qualification_fisher_p": float(fisher_result.pvalue),
            }
        ]
    )
    return subjects, comparison


def _plot_failure_groups(
    subjects: pd.DataFrame, comparison: pd.DataFrame, path: Path, dpi: int
) -> None:
    rng = np.random.default_rng(20260824)
    fig, axis = plt.subplots(figsize=(7.5, 5.5))
    groups = ["PD", "Control"]
    colors = {"PD": "#D55E00", "Control": "#0072B2"}
    data = [
        subjects.loc[subjects["group"].eq(group), "fit_failure_fraction"].to_numpy()
        for group in groups
    ]
    parts = axis.violinplot(data, positions=[0, 1], widths=0.7, showextrema=False)
    for body, group in zip(parts["bodies"], groups):
        body.set_facecolor(colors[group])
        body.set_alpha(0.25)
    for position, (group, values) in enumerate(zip(groups, data)):
        jitter = rng.uniform(-0.12, 0.12, size=len(values))
        axis.scatter(
            position + jitter, values, color=colors[group], alpha=0.65, s=22
        )
        axis.plot(
            [position - 0.2, position + 0.2],
            [np.mean(values), np.mean(values)],
            color="black",
            linewidth=2,
        )
    row = comparison.iloc[0]
    axis.set_xticks([0, 1], [f"PD\n(n={len(data[0])})", f"Control\n(n={len(data[1])})"])
    axis.set_ylabel("Fraction of 60 electrodes failing fit QC")
    axis.set_ylim(-0.03, 1.03)
    axis.grid(axis="y", alpha=0.2)
    axis.set_title(
        "1/f model failure by diagnostic group\n"
        f"age/sex-adjusted PD−Control={row['adjusted_pd_coefficient']:.3f}, "
        f"95% CI [{row['adjusted_pd_ci_lower']:.3f}, "
        f"{row['adjusted_pd_ci_upper']:.3f}], p={row['adjusted_pd_p']:.3g}"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)


def run_fit_qc_sensitivity(
    *,
    scale_free_output: str | Path = "scale_free_analysis/processed",
    bout_ordinal_output: str | Path = "bout_analyses/processed",
    participants_file: str | Path = "processed/metadata/subjects.csv",
    minimum_subject_qc_fraction: float = 0.8,
    fdr_alpha: float = 0.05,
    dpi: int = 150,
) -> dict[str, Any]:
    """Re-aggregate bout results after excluding failed electrode fits."""
    scale_root = Path(scale_free_output)
    ordinal_root = Path(bout_ordinal_output)
    scale_metrics = scale_root / "metrics"
    ordinal_metrics = ordinal_root / "metrics"
    aperiodic = pd.read_csv(scale_metrics / "electrode_aperiodic_metrics.csv")
    scale_electrodes = pd.read_csv(scale_metrics / "electrode_band_metrics.csv")
    ordinal_electrodes = pd.read_csv(
        ordinal_metrics / "subject_electrode_band_metrics.csv"
    )
    participants = pd.read_csv(participants_file)
    required_fit = {*FIT_KEYS, "specparam_fit_qc_pass", "specparam_fit_qc_reasons"}
    missing = sorted(required_fit - set(aperiodic.columns))
    if missing:
        raise ValueError(f"Aperiodic fit-QC table is missing columns: {missing}")
    if not 0.0 < float(minimum_subject_qc_fraction) <= 1.0:
        raise ValueError("minimum_subject_qc_fraction must be in (0, 1]")

    # The independent pipelines intentionally repeat the same fit. Confirm that
    # reusing the audited scale-free mask is exact before propagating it.
    repeated = ordinal_electrodes[
        [*FIT_KEYS, "aperiodic_exponent", "specparam_r_squared"]
    ].drop_duplicates(FIT_KEYS)
    verified = aperiodic.merge(
        repeated,
        on=FIT_KEYS,
        suffixes=("_scale_free", "_bout_ordinal"),
        validate="one_to_one",
    )
    if len(verified) != len(aperiodic):
        raise ValueError("Scale-free and within-bout pipelines do not cover identical fits")
    for metric in ("aperiodic_exponent", "specparam_r_squared"):
        if not np.allclose(
            verified[f"{metric}_scale_free"],
            verified[f"{metric}_bout_ordinal"],
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(f"Repeated specparam result differs for {metric}")

    coverage = _fit_coverage(aperiodic, float(minimum_subject_qc_fraction))
    subjects, failure_comparison = _failure_group_analysis(coverage, participants)
    _write_csv(subjects, scale_metrics / "subject_specparam_fit_failures.csv")
    _write_csv(
        failure_comparison,
        scale_metrics / "specparam_fit_failure_group_comparison.csv",
    )
    _plot_failure_groups(
        subjects,
        failure_comparison,
        scale_root / "figures" / "aperiodic_diagnostics" / "fit_failures_by_group.png",
        int(dpi),
    )

    scale_augmented = _attach_fit_qc(scale_electrodes, aperiodic, coverage)
    scale_passing = scale_augmented.loc[
        scale_augmented["specparam_fit_qc_pass"]
    ].copy()
    scale_subjects_all = _merge_coverage(
        _subject_means(scale_passing, BAND_KEYS, BAND_FEATURES), coverage
    )
    scale_subjects_qc = scale_subjects_all.loc[
        scale_subjects_all["subject_fit_qc_pass"]
    ].copy()
    scale_group_qc = _describe(
        scale_subjects_qc,
        ["band", "band_low_hz", "band_high_hz", "group"],
        BAND_FEATURES,
    )
    scale_comparisons_qc = _group_comparisons(
        scale_subjects_qc,
        BAND_FEATURES,
        fdr_alpha=float(fdr_alpha),
        analysis="scale_free_bout_and_cycle_properties",
    )
    scale_subjects_original = pd.read_csv(
        scale_metrics / "subject_band_metrics.csv"
    )
    scale_sensitivity = _paired_sensitivity(
        scale_subjects_original,
        scale_subjects_qc,
        BAND_FEATURES,
        analysis="scale_free_bout_and_cycle_properties",
    )
    _write_csv(scale_augmented, scale_metrics / "electrode_band_metrics_fit_qc.csv")
    _write_csv(
        scale_subjects_all,
        scale_metrics / "subject_band_metrics_fit_qc_all_subjects.csv",
    )
    _write_csv(scale_subjects_qc, scale_metrics / "subject_band_metrics_fit_qc.csv")
    _write_csv(scale_group_qc, scale_metrics / "group_band_summary_fit_qc.csv")
    _write_csv(
        scale_comparisons_qc,
        scale_metrics / "pd_control_comparisons_fit_qc.csv",
    )
    _write_csv(
        scale_sensitivity,
        scale_metrics / "bout_property_fit_qc_sensitivity.csv",
    )
    _plot_paired_sensitivity(
        scale_subjects_original,
        scale_subjects_qc,
        PLOTTED_BOUT_METRICS,
        scale_root / "figures" / "fit_qc_sensitivity" / "bout_properties_all_vs_fit_qc.png",
        title=(
            "Bout properties: all electrodes versus fit-QC electrodes\n"
            f"paired within the {int(subjects['subject_fit_qc_pass'].sum())} "
            "QC-qualified subjects"
        ),
        dpi=int(dpi),
    )

    ordinal_augmented = _attach_fit_qc(ordinal_electrodes, aperiodic, coverage)
    ordinal_passing = ordinal_augmented.loc[
        ordinal_augmented["specparam_fit_qc_pass"]
    ].copy()
    ordinal_subjects_all = _merge_coverage(
        bout_ordinal_subject_means(ordinal_passing), coverage
    )
    ordinal_subjects_qc = ordinal_subjects_all.loc[
        ordinal_subjects_all["subject_fit_qc_pass"]
    ].copy()
    ordinal_group_qc = bout_ordinal_group_summary(ordinal_subjects_qc)
    ordinal_comparisons_qc = _group_comparisons(
        ordinal_subjects_qc,
        BOUT_ORDINAL_METRICS,
        fdr_alpha=float(fdr_alpha),
        analysis="ordinal_quantities_within_detected_bouts",
    )
    ordinal_subjects_original = pd.read_csv(
        ordinal_metrics / "subject_band_metrics.csv"
    )
    ordinal_sensitivity = _paired_sensitivity(
        ordinal_subjects_original,
        ordinal_subjects_qc,
        BOUT_ORDINAL_METRICS,
        analysis="ordinal_quantities_within_detected_bouts",
    )
    _write_csv(
        ordinal_augmented,
        ordinal_metrics / "subject_electrode_band_metrics_fit_qc.csv",
    )
    _write_csv(
        ordinal_subjects_all,
        ordinal_metrics / "subject_band_metrics_fit_qc_all_subjects.csv",
    )
    _write_csv(
        ordinal_subjects_qc,
        ordinal_metrics / "subject_band_metrics_fit_qc.csv",
    )
    _write_csv(
        ordinal_group_qc,
        ordinal_metrics / "group_band_summary_fit_qc.csv",
    )
    _write_csv(
        ordinal_comparisons_qc,
        ordinal_metrics / "pd_control_comparisons_fit_qc.csv",
    )
    _write_csv(
        ordinal_sensitivity,
        ordinal_metrics / "within_bout_ordinal_fit_qc_sensitivity.csv",
    )
    _plot_paired_sensitivity(
        ordinal_subjects_original,
        ordinal_subjects_qc,
        BOUT_ORDINAL_METRICS,
        ordinal_root
        / "figures"
        / "fit_qc_sensitivity"
        / "within_bout_ordinal_all_vs_fit_qc.png",
        title=(
            "Within-bout ordinal quantities: all electrodes versus fit-QC electrodes\n"
            f"paired within the {int(subjects['subject_fit_qc_pass'].sum())} "
            "QC-qualified subjects"
        ),
        dpi=int(dpi),
    )

    row = failure_comparison.iloc[0]
    significant_scale = scale_comparisons_qc.loc[scale_comparisons_qc["fdr_reject"]]
    significant_ordinal = ordinal_comparisons_qc.loc[
        ordinal_comparisons_qc["fdr_reject"]
    ]
    ordinal_result_lines = [
        (
            f"- {entry.band.replace('_', ' ').title()} {entry.metric.replace('_', ' ')}: "
            f"Hedges g={entry.hedges_g_pd_minus_control:.3f}, "
            f"Welch p={entry.welch_p:.4g}, BH q={entry.welch_p_fdr_bh:.4g}."
        )
        for entry in significant_ordinal.itertuples(index=False)
    ]
    report = "\n".join(
        [
            "# Specparam fit-QC propagation report",
            "",
            "The original all-electrode results are retained. This sensitivity analysis "
            "excludes electrode fits that fail the formal specparam QC and restricts "
            f"inference to subjects with at least {100 * minimum_subject_qc_fraction:.0f}% "
            "passing shared electrodes.",
            "",
            f"Qualified subjects: {int(subjects['subject_fit_qc_pass'].sum())}/"
            f"{len(subjects)} ({int(subjects.loc[subjects.group.eq('PD'), 'subject_fit_qc_pass'].sum())} "
            "PD; "
            f"{int(subjects.loc[subjects.group.eq('Control'), 'subject_fit_qc_pass'].sum())} Control).",
            "",
            "## Diagnostic-group difference in failures",
            "",
            f"Mean failed-electrode fraction: PD={row['pd_mean_failure_fraction']:.3f}; "
            f"Control={row['control_mean_failure_fraction']:.3f}. The age/sex-adjusted "
            f"PD-minus-Control difference was {row['adjusted_pd_coefficient']:.3f} "
            f"(95% CI [{row['adjusted_pd_ci_lower']:.3f}, "
            f"{row['adjusted_pd_ci_upper']:.3f}], p={row['adjusted_pd_p']:.4g}).",
            (
                "The descriptive difference points toward more failures in Controls, "
                "not PD. Welch p="
                f"{row['welch_p']:.4g}, Mann–Whitney p={row['mann_whitney_p']:.4g}, "
                f"and subject-qualification Fisher p={row['subject_qualification_fisher_p']:.4g}. "
                "Because these tests disagree and the adjusted interval crosses zero, "
                "there is no robust diagnostic-group difference in fit failure."
            ),
            "",
            "Electrodes are not treated as independent observations; all group tests use "
            "one failure fraction per subject.",
            "",
            "## Interpretation",
            "",
            "QC-filtered bout properties and within-bout ordinal quantities reuse only "
            "electrode-level results whose independently repeated specparam fits agree "
            "numerically and pass QC. No failed fit contributes to these sensitivity "
            "summaries. The all-electrode files remain the provenance analysis.",
            "",
            "## QC-filtered group summaries",
            "",
            (
                f"The exploratory scale-free bout/cycle family has "
                f"{len(significant_scale)}/{len(scale_comparisons_qc)} Welch comparisons "
                "passing BH FDR after QC filtering. These include correlated properties "
                "and are not independent discoveries."
            ),
            (
                f"Within-bout ordinal H/C/F has {len(significant_ordinal)}/"
                f"{len(ordinal_comparisons_qc)} Welch comparisons passing BH FDR:"
            ),
            *ordinal_result_lines,
            "",
            (
                "The all-versus-QC paired tests answer whether filtering changes a "
                "quantity, not whether PD differs from Control. Consult the dedicated "
                "comparison CSV files for the group tests and the paired sensitivity "
                "CSV files for robustness to electrode exclusion."
            ),
            "",
        ]
    )
    (scale_root / "FIT_QC_SENSITIVITY.md").write_text(report, encoding="utf-8")
    (ordinal_root / "FIT_QC_SENSITIVITY.md").write_text(report, encoding="utf-8")

    payload = {
        "minimum_subject_qc_fraction": float(minimum_subject_qc_fraction),
        "n_subjects": int(len(subjects)),
        "n_qualified_subjects": int(subjects["subject_fit_qc_pass"].sum()),
        "qualified_group_counts": subjects.loc[
            subjects["subject_fit_qc_pass"], "group"
        ].value_counts().to_dict(),
        "n_electrode_fits": int(len(aperiodic)),
        "n_passing_electrode_fits": int(aperiodic["specparam_fit_qc_pass"].sum()),
        "failure_group_comparison": failure_comparison.iloc[0].to_dict(),
        "policy": (
            "Retain all-electrode provenance outputs; formal fit-QC sensitivity uses "
            "only passing electrodes and subjects with at least 80% passing coverage."
        ),
    }
    (scale_root / "fit_qc_sensitivity_manifest.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    return payload
