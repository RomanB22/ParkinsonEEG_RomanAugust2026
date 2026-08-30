"""Repeated-measures condition contrasts and MMSE associations."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import t as student_t
from scipy.stats import ttest_rel, wilcoxon

from core.group_statistics import fdr_bh
from analyses.behavioral.statistics import partial_spearman, unadjusted_spearman


def _hedges_g(treated: np.ndarray, reference: np.ndarray) -> float:
    n_treated, n_reference = len(treated), len(reference)
    if min(n_treated, n_reference) < 2:
        return np.nan
    degrees = n_treated + n_reference - 2
    pooled = (
        (n_treated - 1) * np.var(treated, ddof=1)
        + (n_reference - 1) * np.var(reference, ddof=1)
    ) / degrees
    if pooled <= 0.0 or not np.isfinite(pooled):
        return np.nan
    correction = 1.0 - 3.0 / (4.0 * (n_treated + n_reference) - 9.0)
    return float(correction * (np.mean(treated) - np.mean(reference)) / np.sqrt(pooled))


def _between_contrast(
    table: pd.DataFrame,
    *,
    reference: str,
    treated: str,
    minimum_per_condition: int,
    confidence_level: float,
) -> dict[str, Any]:
    selected = table.loc[table["condition"].isin([reference, treated])].dropna(
        subset=["value", "age_years", "sex_male"]
    )
    if selected["participant_id"].duplicated().any():
        raise ValueError(f"{reference}/{treated} contrast has repeated participants")
    reference_values = selected.loc[
        selected["condition"].eq(reference), "value"
    ].to_numpy(dtype=float)
    treated_values = selected.loc[
        selected["condition"].eq(treated), "value"
    ].to_numpy(dtype=float)
    base = {
        "reference_condition": reference,
        "treated_condition": treated,
        "effect_direction": f"{treated}_minus_{reference}",
        "n_reference": int(len(reference_values)),
        "n_treated": int(len(treated_values)),
        "reference_mean": (
            float(np.mean(reference_values)) if len(reference_values) else np.nan
        ),
        "treated_mean": (
            float(np.mean(treated_values)) if len(treated_values) else np.nan
        ),
        "unadjusted_mean_difference": (
            float(np.mean(treated_values) - np.mean(reference_values))
            if len(reference_values) and len(treated_values)
            else np.nan
        ),
        "hedges_g": _hedges_g(treated_values, reference_values),
        "primary_model": "OLS: feature ~ condition + age + sex; HC3 robust SE",
    }
    if min(len(reference_values), len(treated_values)) < minimum_per_condition:
        return {
            **base,
            "analysis_status": "insufficient_complete_participants",
            "effect": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "statistic": np.nan,
            "primary_p_value": np.nan,
            "standardized_effect": np.nan,
        }
    design = pd.DataFrame(
        {
            "treated_indicator": selected["condition"].eq(treated).astype(float),
            "age_centered": selected["age_years"] - selected["age_years"].mean(),
            "sex_male": selected["sex_male"].astype(float),
        },
        index=selected.index,
    )
    fitted = sm.OLS(
        selected["value"].to_numpy(dtype=float),
        sm.add_constant(design, has_constant="add"),
    ).fit(cov_type="HC3")
    alpha = 1.0 - confidence_level
    interval = fitted.conf_int(alpha=alpha).loc["treated_indicator"]
    effect = float(fitted.params["treated_indicator"])
    outcome_sd = float(selected["value"].std(ddof=1))
    return {
        **base,
        "analysis_status": "ok",
        "effect": effect,
        "ci_lower": float(interval.iloc[0]),
        "ci_upper": float(interval.iloc[1]),
        "statistic": float(fitted.tvalues["treated_indicator"]),
        "primary_p_value": float(fitted.pvalues["treated_indicator"]),
        "standardized_effect": effect / outcome_sd if outcome_sd > 0 else np.nan,
    }


def _paired_contrast(
    table: pd.DataFrame,
    *,
    minimum_pairs: int,
    confidence_level: float,
    bootstrap_resamples: int,
    seed: int,
) -> dict[str, Any]:
    selected = table.loc[table["condition"].isin(["PD_OFF", "PD_ON"])].dropna(
        subset=["value"]
    )
    if selected.duplicated(["participant_id", "condition"]).any():
        raise ValueError("PD ON/OFF contrast has duplicate participant-condition rows")
    paired = selected.pivot(
        index="participant_id", columns="condition", values="value"
    )
    for condition in ("PD_OFF", "PD_ON"):
        if condition not in paired:
            paired[condition] = np.nan
    paired = paired.dropna(subset=["PD_OFF", "PD_ON"])
    differences = paired["PD_ON"].to_numpy(float) - paired["PD_OFF"].to_numpy(float)
    base = {
        "reference_condition": "PD_OFF",
        "treated_condition": "PD_ON",
        "effect_direction": "PD_ON_minus_PD_OFF",
        "n_reference": int(len(paired)),
        "n_treated": int(len(paired)),
        "n_pairs": int(len(paired)),
        "reference_mean": float(paired["PD_OFF"].mean()) if len(paired) else np.nan,
        "treated_mean": float(paired["PD_ON"].mean()) if len(paired) else np.nan,
        "unadjusted_mean_difference": float(np.mean(differences)) if len(paired) else np.nan,
        "hedges_g": np.nan,
        "primary_model": "paired t test: PD ON minus PD OFF",
    }
    if len(paired) < minimum_pairs:
        return {
            **base,
            "analysis_status": "insufficient_complete_pairs",
            "effect": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "statistic": np.nan,
            "primary_p_value": np.nan,
            "standardized_effect": np.nan,
            "wilcoxon_statistic": np.nan,
            "wilcoxon_p_value": np.nan,
            "bootstrap_ci_lower": np.nan,
            "bootstrap_ci_upper": np.nan,
        }
    mean_difference = float(np.mean(differences))
    difference_sd = float(np.std(differences, ddof=1))
    standard_error = difference_sd / np.sqrt(len(differences))
    critical = float(
        student_t.ppf(0.5 + confidence_level / 2.0, len(differences) - 1)
    )
    # SciPy's asymptotic Wilcoxon implementation divides by a zero standard
    # error when every paired difference is exactly zero.  This is a valid
    # null result for our purposes, but calling scipy.stats.wilcoxon emits one
    # RuntimeWarning per feature and returns NaN on recent SciPy versions.
    # Handle the degenerate null explicitly and retain it in FDR correction as
    # p=1.  The paired t statistic is likewise represented as 0 rather than
    # the undefined 0/0 returned by scipy.stats.ttest_rel.
    if np.count_nonzero(differences) == 0:
        paired_statistic, paired_p = 0.0, 1.0
        wilcoxon_statistic, wilcoxon_p = 0.0, 1.0
    else:
        paired_test = ttest_rel(paired["PD_ON"], paired["PD_OFF"])
        paired_statistic = float(paired_test.statistic)
        paired_p = float(paired_test.pvalue)
        try:
            signed_rank = wilcoxon(differences, alternative="two-sided")
            wilcoxon_statistic = float(signed_rank.statistic)
            wilcoxon_p = float(signed_rank.pvalue)
        except ValueError:
            wilcoxon_statistic, wilcoxon_p = 0.0, 1.0
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0, len(differences), size=(bootstrap_resamples, len(differences))
    )
    bootstrapped = np.mean(differences[indices], axis=1)
    tail = (1.0 - confidence_level) / 2.0
    bootstrap_ci = np.quantile(bootstrapped, [tail, 1.0 - tail])
    return {
        **base,
        "analysis_status": "ok",
        "effect": mean_difference,
        "ci_lower": mean_difference - critical * standard_error,
        "ci_upper": mean_difference + critical * standard_error,
        "statistic": paired_statistic,
        "primary_p_value": paired_p,
        "standardized_effect": (
            mean_difference / difference_sd if difference_sd > 0 else np.nan
        ),
        "wilcoxon_statistic": wilcoxon_statistic,
        "wilcoxon_p_value": wilcoxon_p,
        "bootstrap_ci_lower": float(bootstrap_ci[0]),
        "bootstrap_ci_upper": float(bootstrap_ci[1]),
    }


def _mmse_model(
    table: pd.DataFrame,
    *,
    minimum_participants: int,
    confidence_level: float,
) -> dict[str, Any]:
    selected = table.dropna(subset=["value", "mmse", "age_years", "sex_male"]).copy()
    if selected["participant_id"].duplicated().any():
        raise ValueError("MMSE model requires one row per participant")
    base = {
        "n_participants": int(len(selected)),
        "mmse_min": float(selected["mmse"].min()) if len(selected) else np.nan,
        "mmse_max": float(selected["mmse"].max()) if len(selected) else np.nan,
        "primary_model": "partial Spearman: EEG feature vs MMSE, adjusted for age and sex",
    }
    if len(selected) < minimum_participants or selected["mmse"].nunique() < 3:
        return {
            **base,
            "analysis_status": "insufficient_mmse_information",
            "mmse_slope": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "statistic": np.nan,
            "primary_p_value": np.nan,
            "standardized_effect": np.nan,
            "unadjusted_spearman_rho": np.nan,
            "unadjusted_spearman_p_value": np.nan,
            "ols_hc3_statistic": np.nan,
            "ols_hc3_p_value": np.nan,
        }
    design = pd.DataFrame(
        {
            "mmse_centered": selected["mmse"] - selected["mmse"].mean(),
            "age_centered": selected["age_years"] - selected["age_years"].mean(),
            "sex_male": selected["sex_male"].astype(float),
        },
        index=selected.index,
    )
    fitted = sm.OLS(
        selected["value"].to_numpy(float),
        sm.add_constant(design, has_constant="add"),
    ).fit(cov_type="HC3")
    interval = fitted.conf_int(alpha=1.0 - confidence_level).loc["mmse_centered"]
    slope = float(fitted.params["mmse_centered"])
    outcome_sd = float(selected["value"].std(ddof=1))
    mmse_sd = float(selected["mmse"].std(ddof=1))
    partial_rho, partial_p = partial_spearman(
        selected["value"].to_numpy(float),
        selected["mmse"].to_numpy(float),
        selected[["age_years", "sex_male"]].to_numpy(float),
    )
    unadjusted_rho, unadjusted_p = unadjusted_spearman(
        selected["value"].to_numpy(float), selected["mmse"].to_numpy(float)
    )
    return {
        **base,
        "analysis_status": "ok",
        "mmse_slope": slope,
        "ci_lower": float(interval.iloc[0]),
        "ci_upper": float(interval.iloc[1]),
        "statistic": partial_rho,
        "primary_p_value": partial_p,
        "standardized_effect": partial_rho,
        "unadjusted_spearman_rho": unadjusted_rho,
        "unadjusted_spearman_p_value": unadjusted_p,
        "ols_hc3_statistic": float(fitted.tvalues["mmse_centered"]),
        "ols_hc3_p_value": float(fitted.pvalues["mmse_centered"]),
        "ols_standardized_slope": (
            slope * mmse_sd / outcome_sd if outcome_sd > 0 and mmse_sd > 0 else np.nan
        ),
    }


def _attach_metadata(features: pd.DataFrame, recordings: pd.DataFrame) -> pd.DataFrame:
    uniqueness = ["recording_id", "duration_variant", "feature_id"]
    if "electrode" in features:
        uniqueness.append("electrode")
    if features.duplicated(uniqueness).any():
        raise ValueError("Feature table contains duplicate recording analysis units")
    columns = [
        "recording_id",
        "participant_id",
        "condition",
        "age_years",
        "sex_male",
        "mmse",
        "provenance_sensitivity_exclusion",
    ]
    attached = features.merge(
        recordings[columns], on="recording_id", how="left", validate="many_to_one"
    )
    if attached["condition"].isna().any():
        raise ValueError("An EEG feature lacks recording metadata")
    return attached


def _apply_fdr(
    table: pd.DataFrame,
    *,
    group_columns: list[str],
    alpha: float,
) -> pd.DataFrame:
    result = table.copy()
    result["p_fdr_bh"] = np.nan
    result["fdr_reject"] = False
    result["fdr_alpha"] = float(alpha)
    if result.empty:
        return result
    for _, indices in result.groupby(group_columns, sort=False, dropna=False).groups.items():
        adjusted, rejected = fdr_bh(
            result.loc[indices, "primary_p_value"].to_numpy(float), alpha
        )
        result.loc[indices, "p_fdr_bh"] = adjusted
        result.loc[indices, "fdr_reject"] = rejected
    return result


def compute_condition_statistics(
    features: pd.DataFrame,
    recordings: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    attached = _attach_metadata(features, recordings)
    settings = config["statistics"]
    rows: list[dict[str, Any]] = []
    cohorts = {
        "all_participants": attached,
        "exclude_preprocessed_on_provenance": attached.loc[
            ~attached["provenance_sensitivity_exclusion"]
        ],
    }
    contrasts = (("HC", "PD_OFF"), ("HC", "PD_ON"))
    feature_columns = [
        "duration_variant",
        "feature_id",
        "family",
        "domain",
        "band",
        "metric",
    ]
    if "electrode" in attached:
        feature_columns.append("electrode")
    seed = int(settings["random_seed"])
    comparison_index = 0
    for cohort_name, cohort in cohorts.items():
        for keys, table in cohort.groupby(feature_columns, sort=False, dropna=False):
            specification = dict(zip(feature_columns, keys))
            for reference, treated in contrasts:
                result = _between_contrast(
                    table,
                    reference=reference,
                    treated=treated,
                    minimum_per_condition=int(settings["minimum_per_condition"]),
                    confidence_level=float(settings["confidence_level"]),
                )
                rows.append({"sensitivity_cohort": cohort_name, **specification, **result})
            result = _paired_contrast(
                table,
                minimum_pairs=int(settings["minimum_pairs"]),
                confidence_level=float(settings["confidence_level"]),
                bootstrap_resamples=int(settings["bootstrap_resamples"]),
                seed=seed + comparison_index,
            )
            rows.append({"sensitivity_cohort": cohort_name, **specification, **result})
            comparison_index += 1
    result = pd.DataFrame.from_records(rows)
    result["contrast"] = result["effect_direction"]
    return _apply_fdr(
        result,
        group_columns=[
            "sensitivity_cohort",
            "duration_variant",
            "family",
            "contrast",
        ],
        alpha=float(settings["fdr_alpha"]),
    )


def compute_mmse_statistics(
    features: pd.DataFrame,
    recordings: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    attached = _attach_metadata(features, recordings)
    settings = config["statistics"]
    feature_columns = [
        "duration_variant",
        "feature_id",
        "family",
        "domain",
        "band",
        "metric",
    ]
    if "electrode" in attached:
        feature_columns.append("electrode")
    rows: list[dict[str, Any]] = []
    cohorts = {
        "all_participants": attached,
        "exclude_preprocessed_on_provenance": attached.loc[
            ~attached["provenance_sensitivity_exclusion"]
        ],
    }
    for cohort_name, cohort in cohorts.items():
        for keys, table in cohort.groupby(feature_columns, sort=False, dropna=False):
            specification = dict(zip(feature_columns, keys))
            for condition in ("HC", "PD_OFF", "PD_ON"):
                result = _mmse_model(
                    table.loc[table["condition"].eq(condition)],
                    minimum_participants=int(settings["minimum_mmse_participants"]),
                    confidence_level=float(settings["confidence_level"]),
                )
                rows.append(
                    {
                        "sensitivity_cohort": cohort_name,
                        **specification,
                        "mmse_model": condition,
                        "outcome_definition": f"{condition} EEG feature",
                        **result,
                    }
                )
            pd_table = table.loc[table["condition"].isin(["PD_OFF", "PD_ON"])].copy()
            pivot = pd_table.pivot(
                index="participant_id", columns="condition", values="value"
            )
            for condition in ("PD_OFF", "PD_ON"):
                if condition not in pivot:
                    pivot[condition] = np.nan
            pivot = pivot.dropna(subset=["PD_OFF", "PD_ON"])
            metadata = (
                pd_table[["participant_id", "mmse", "age_years", "sex_male"]]
                .drop_duplicates("participant_id")
                .set_index("participant_id")
            )
            delta = pivot.join(metadata, how="inner").reset_index()
            delta["value"] = delta["PD_ON"] - delta["PD_OFF"]
            result = _mmse_model(
                delta,
                minimum_participants=int(settings["minimum_mmse_participants"]),
                confidence_level=float(settings["confidence_level"]),
            )
            rows.append(
                {
                    "sensitivity_cohort": cohort_name,
                    **specification,
                    "mmse_model": "PD_ON_minus_PD_OFF",
                    "outcome_definition": "within-participant medication EEG change",
                    **result,
                }
            )
    result = pd.DataFrame.from_records(rows)
    return _apply_fdr(
        result,
        group_columns=[
            "sensitivity_cohort",
            "duration_variant",
            "family",
            "mmse_model",
        ],
        alpha=float(settings["fdr_alpha"]),
    )
