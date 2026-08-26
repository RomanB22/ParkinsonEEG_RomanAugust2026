"""Consistent subject-level and exploratory electrode-level group inference."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import mannwhitneyu, t as student_t, ttest_ind, ttest_rel, wilcoxon


def fdr_bh(p_values: np.ndarray, alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Benjamini-Hochberg correction preserving missing values and row order."""
    values = np.asarray(p_values, dtype=float)
    adjusted = np.full(values.shape, np.nan, dtype=float)
    rejected = np.zeros(values.shape, dtype=bool)
    finite_indices = np.flatnonzero(np.isfinite(values))
    if not len(finite_indices):
        return adjusted, rejected
    finite = values[finite_indices]
    order = np.argsort(finite)
    ranked = finite[order]
    count = len(ranked)
    corrected = ranked * count / np.arange(1, count + 1)
    corrected = np.minimum.accumulate(corrected[::-1])[::-1]
    corrected = np.clip(corrected, 0.0, 1.0)
    restored = np.empty_like(corrected)
    restored[order] = corrected
    adjusted[finite_indices] = restored
    rejected[finite_indices] = restored <= float(alpha)
    return adjusted, rejected


def _participant_metadata(
    participants: pd.DataFrame,
    analyzed_subjects: Iterable[str],
) -> pd.DataFrame:
    required = {"participant_id", "GROUP", "AGE", "GENDER"}
    missing = sorted(required - set(participants))
    if missing:
        raise ValueError(f"Group statistics participant metadata is missing: {missing}")
    if participants["participant_id"].duplicated().any():
        raise ValueError("Participant IDs must be unique for group statistics")
    metadata = participants.copy()
    metadata["subject_id"] = metadata["participant_id"].astype(str)
    expected = {str(value) for value in analyzed_subjects}
    metadata = metadata.loc[metadata["subject_id"].isin(expected)].copy()
    missing_subjects = sorted(expected - set(metadata["subject_id"]))
    if missing_subjects:
        raise ValueError(
            f"Group statistics lack participant metadata for: {missing_subjects}"
        )
    metadata["group"] = metadata["GROUP"].astype(str)
    if set(metadata["group"]) != {"PD", "Control"}:
        raise ValueError("Group statistics require PD and Control participants")
    metadata["target_pd"] = metadata["group"].eq("PD").astype(int)
    metadata["age_years"] = pd.to_numeric(metadata["AGE"], errors="raise")
    if not set(metadata["GENDER"].astype(str)).issubset({"M", "F"}):
        raise ValueError("GENDER must contain only M and F")
    metadata["sex_male"] = metadata["GENDER"].astype(str).eq("M").astype(int)
    columns = [
        "subject_id",
        "group",
        "target_pd",
        "age_years",
        "sex_male",
    ]
    if "match_pair_id" in metadata:
        if metadata["match_pair_id"].isna().any():
            raise ValueError("Matched participant rows cannot have missing pair IDs")
        metadata["match_pair_id"] = metadata["match_pair_id"].astype(str)
        if not (metadata.groupby("match_pair_id")["target_pd"].nunique() == 2).all():
            raise ValueError("Each matched pair must contain one PD and one Control")
        columns.append("match_pair_id")
    values = metadata[["age_years", "sex_male"]].to_numpy(dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("Age and sex must be complete for group statistics")
    return metadata[columns].sort_values("subject_id").reset_index(drop=True)


def _hedges_g(pd_values: np.ndarray, control_values: np.ndarray) -> float:
    n_pd, n_control = len(pd_values), len(control_values)
    degrees = n_pd + n_control - 2
    if n_pd < 2 or n_control < 2 or degrees <= 0:
        return np.nan
    pooled_variance = (
        (n_pd - 1) * np.var(pd_values, ddof=1)
        + (n_control - 1) * np.var(control_values, ddof=1)
    ) / degrees
    if not np.isfinite(pooled_variance) or pooled_variance <= 0.0:
        return 0.0 if np.isclose(np.mean(pd_values), np.mean(control_values)) else np.nan
    correction = 1.0 - 3.0 / (4.0 * (n_pd + n_control) - 9.0)
    return float(
        correction
        * (np.mean(pd_values) - np.mean(control_values))
        / np.sqrt(pooled_variance)
    )


def _unadjusted_statistics(
    pd_values: np.ndarray,
    control_values: np.ndarray,
) -> dict[str, float]:
    try:
        welch = ttest_ind(pd_values, control_values, equal_var=False)
        welch_t, welch_p = float(welch.statistic), float(welch.pvalue)
    except (FloatingPointError, ValueError):
        welch_t = welch_p = np.nan
    try:
        mann = mannwhitneyu(pd_values, control_values, alternative="two-sided")
        mann_u, mann_p = float(mann.statistic), float(mann.pvalue)
    except ValueError:
        mann_u = mann_p = np.nan
    return {
        "pd_mean": float(np.mean(pd_values)),
        "pd_std": float(np.std(pd_values, ddof=1)),
        "pd_median": float(np.median(pd_values)),
        "control_mean": float(np.mean(control_values)),
        "control_std": float(np.std(control_values, ddof=1)),
        "control_median": float(np.median(control_values)),
        "mean_difference_pd_minus_control": float(
            np.mean(pd_values) - np.mean(control_values)
        ),
        "hedges_g_pd_minus_control": _hedges_g(pd_values, control_values),
        "welch_t": welch_t,
        "welch_p_value": welch_p,
        "mann_whitney_u": mann_u,
        "mann_whitney_p_value": mann_p,
    }


def _compare_one(
    table: pd.DataFrame,
    *,
    confidence_level: float,
) -> dict[str, Any]:
    complete_columns = ["value", "group", "age_years", "sex_male", "target_pd"]
    if "match_pair_id" in table:
        complete_columns.append("match_pair_id")
    selected = table.dropna(subset=complete_columns).copy()
    if selected["subject_id"].duplicated().any():
        raise ValueError("Every group comparison requires one row per subject")
    pd_values = selected.loc[selected["group"].eq("PD"), "value"].to_numpy(float)
    control_values = selected.loc[
        selected["group"].eq("Control"), "value"
    ].to_numpy(float)
    if min(len(pd_values), len(control_values)) < 3:
        matched = "match_pair_id" in selected
        return {
            "analysis_status": "insufficient_complete_subjects",
            "inference_design": (
                "demographic_matched_pairs"
                if matched
                else "full_cohort_age_sex_adjusted"
            ),
            "primary_model": (
                "paired t test (PD minus Control within pair)"
                if matched
                else "OLS: value ~ PD + age + sex; HC3 robust SE"
            ),
            "n_pd": int(len(pd_values)),
            "n_control": int(len(control_values)),
            "n_pairs": 0 if matched else np.nan,
            **{
                name: np.nan
                for name in (
                    "pd_mean",
                    "pd_std",
                    "pd_median",
                    "control_mean",
                    "control_std",
                    "control_median",
                    "mean_difference_pd_minus_control",
                    "hedges_g_pd_minus_control",
                    "welch_t",
                    "welch_p_value",
                    "mann_whitney_u",
                    "mann_whitney_p_value",
                    "primary_effect_pd_minus_control",
                    "primary_effect_ci_lower",
                    "primary_effect_ci_upper",
                    "primary_statistic",
                    "primary_p_value",
                    "standardized_effect_pd_minus_control",
                    "paired_wilcoxon_statistic",
                    "paired_wilcoxon_p_value",
                )
            },
            "standardized_effect_definition": (
                "paired Cohen dz"
                if matched
                else "age/sex-adjusted PD coefficient divided by outcome SD"
            ),
        }
    result: dict[str, Any] = {
        "n_pd": int(len(pd_values)),
        "n_control": int(len(control_values)),
        **_unadjusted_statistics(pd_values, control_values),
    }
    tail = (1.0 - float(confidence_level)) / 2.0
    if "match_pair_id" in selected:
        paired = selected.pivot(
            index="match_pair_id", columns="group", values="value"
        ).dropna()
        if len(paired) < 3:
            result.update(
                {
                    "analysis_status": "insufficient_complete_pairs",
                    "inference_design": "demographic_matched_pairs",
                    "primary_model": "paired t test (PD minus Control within pair)",
                    "n_pairs": int(len(paired)),
                    "primary_effect_pd_minus_control": np.nan,
                    "primary_effect_ci_lower": np.nan,
                    "primary_effect_ci_upper": np.nan,
                    "primary_statistic": np.nan,
                    "primary_p_value": np.nan,
                    "standardized_effect_pd_minus_control": np.nan,
                    "standardized_effect_definition": "paired Cohen dz",
                    "paired_wilcoxon_statistic": np.nan,
                    "paired_wilcoxon_p_value": np.nan,
                }
            )
            return result
        differences = paired["PD"].to_numpy(float) - paired["Control"].to_numpy(float)
        paired_test = ttest_rel(paired["PD"], paired["Control"])
        standard_error = float(
            np.std(differences, ddof=1) / np.sqrt(len(differences))
        )
        critical = float(student_t.ppf(1.0 - tail, len(differences) - 1))
        mean_difference = float(np.mean(differences))
        difference_sd = float(np.std(differences, ddof=1))
        try:
            signed_rank = wilcoxon(differences, alternative="two-sided")
            signed_rank_statistic = float(signed_rank.statistic)
            signed_rank_p = float(signed_rank.pvalue)
        except ValueError:
            signed_rank_statistic, signed_rank_p = 0.0, 1.0
        result.update(
            {
                "analysis_status": "ok",
                "inference_design": "demographic_matched_pairs",
                "primary_model": "paired t test (PD minus Control within pair)",
                "n_pairs": int(len(paired)),
                "primary_effect_pd_minus_control": mean_difference,
                "primary_effect_ci_lower": mean_difference - critical * standard_error,
                "primary_effect_ci_upper": mean_difference + critical * standard_error,
                "primary_statistic": float(paired_test.statistic),
                "primary_p_value": float(paired_test.pvalue),
                "standardized_effect_pd_minus_control": (
                    mean_difference / difference_sd if difference_sd > 0.0 else 0.0
                ),
                "standardized_effect_definition": "paired Cohen dz",
                "paired_wilcoxon_statistic": signed_rank_statistic,
                "paired_wilcoxon_p_value": signed_rank_p,
            }
        )
    else:
        design = pd.DataFrame(
            {
                "pd_indicator": selected["target_pd"].astype(float),
                "age_years": selected["age_years"].astype(float),
                "sex_male": selected["sex_male"].astype(float),
            },
            index=selected.index,
        )
        fitted = sm.OLS(
            selected["value"].to_numpy(float),
            sm.add_constant(design, has_constant="add"),
        ).fit(cov_type="HC3")
        interval = fitted.conf_int(alpha=1.0 - float(confidence_level)).loc[
            "pd_indicator"
        ]
        outcome_sd = float(np.std(selected["value"].to_numpy(float), ddof=1))
        adjusted_effect = float(fitted.params["pd_indicator"])
        result.update(
            {
                "analysis_status": "ok",
                "inference_design": "full_cohort_age_sex_adjusted",
                "primary_model": "OLS: value ~ PD + age + sex; HC3 robust SE",
                "n_pairs": np.nan,
                "primary_effect_pd_minus_control": adjusted_effect,
                "primary_effect_ci_lower": float(interval.iloc[0]),
                "primary_effect_ci_upper": float(interval.iloc[1]),
                "primary_statistic": float(fitted.tvalues["pd_indicator"]),
                "primary_p_value": float(fitted.pvalues["pd_indicator"]),
                "standardized_effect_pd_minus_control": (
                    adjusted_effect / outcome_sd if outcome_sd > 0.0 else 0.0
                ),
                "standardized_effect_definition": (
                    "age/sex-adjusted PD coefficient divided by outcome SD"
                ),
                "paired_wilcoxon_statistic": np.nan,
                "paired_wilcoxon_p_value": np.nan,
            }
        )
    return result


def _group_iterator(
    table: pd.DataFrame,
    strata: Sequence[str],
):
    if not strata:
        yield (), table
        return
    grouping: str | list[str] = strata[0] if len(strata) == 1 else list(strata)
    for keys, selected in table.groupby(grouping, sort=False, dropna=False):
        yield keys if isinstance(keys, tuple) else (keys,), selected


def _attach_metadata(table: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    source = table.copy()
    source["subject_id"] = source["subject_id"].astype(str)
    if "group" in source:
        source = source.rename(columns={"group": "source_group"})
    result = source.merge(metadata, on="subject_id", how="left", validate="many_to_one")
    if result["group"].isna().any():
        raise ValueError("A metric row lacks participant metadata")
    if "source_group" in result and not result["source_group"].astype(str).eq(
        result["group"]
    ).all():
        raise ValueError("Metric and participant group labels disagree")
    return result.drop(columns="source_group", errors="ignore")


def compute_group_statistics(
    electrode_table: pd.DataFrame,
    participants: pd.DataFrame,
    *,
    metrics: Sequence[str],
    strata: Sequence[str] = (),
    domain: str,
    subject_aggregation: str = "mean",
    confidence_level: float = 0.95,
    fdr_alpha: float = 0.05,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return primary subject summaries and exploratory electrode inference.

    Subject-level tests correct across every metric/stratum combination in the
    domain. Electrode tests expose both within-feature BH and a stricter BH
    correction across every electrode test in the complete domain.
    """
    if subject_aggregation not in {"mean", "median"}:
        raise ValueError("subject_aggregation must be mean or median")
    if not 0.0 < float(confidence_level) < 1.0:
        raise ValueError("confidence_level must be between zero and one")
    if not 0.0 < float(fdr_alpha) < 1.0:
        raise ValueError("fdr_alpha must be between zero and one")
    required = {"subject_id", "group", "electrode", *metrics, *strata}
    missing = sorted(required - set(electrode_table))
    if missing:
        raise ValueError(f"{domain} electrode table is missing columns: {missing}")
    duplicate_keys = ["subject_id", "electrode", *strata]
    if electrode_table.duplicated(duplicate_keys).any():
        raise ValueError(f"{domain} contains duplicate subject/electrode rows")
    metadata = _participant_metadata(participants, electrode_table["subject_id"].unique())
    attached = _attach_metadata(electrode_table, metadata)

    subject_keys = ["subject_id", *strata]
    aggregation = "mean" if subject_aggregation == "mean" else "median"
    subject_values = (
        attached.groupby(subject_keys, sort=False, dropna=False)[list(metrics)]
        .agg(aggregation)
        .reset_index()
    )
    subject_values = _attach_metadata(
        subject_values.assign(
            group=subject_values["subject_id"].map(
                metadata.set_index("subject_id")["group"]
            )
        ),
        metadata,
    )

    subject_rows: list[dict[str, Any]] = []
    for keys, selected in _group_iterator(subject_values, strata):
        stratum = dict(zip(strata, keys))
        for metric in metrics:
            comparison = _compare_one(
                selected[
                    [
                        "subject_id",
                        "group",
                        "target_pd",
                        "age_years",
                        "sex_male",
                        *(
                            ["match_pair_id"]
                            if "match_pair_id" in selected
                            else []
                        ),
                        metric,
                    ]
                ].rename(columns={metric: "value"}),
                confidence_level=confidence_level,
            )
            if comparison is None:
                continue
            subject_rows.append(
                {
                    "domain": domain,
                    "analysis_level": "subject_shared_electrode_aggregate",
                    "subject_aggregation": subject_aggregation,
                    **stratum,
                    "metric": metric,
                    **comparison,
                }
            )
    subject_statistics = pd.DataFrame.from_records(subject_rows)
    if subject_statistics.empty:
        raise ValueError(f"{domain} produced no analyzable subject-level tests")
    subject_adjusted, subject_rejected = fdr_bh(
        subject_statistics["primary_p_value"].to_numpy(float), fdr_alpha
    )
    subject_statistics["primary_p_fdr_bh_domain"] = subject_adjusted
    subject_statistics["primary_fdr_reject_domain"] = subject_rejected
    subject_statistics["fdr_alpha"] = float(fdr_alpha)
    subject_statistics["fdr_scope"] = (
        "all subject-level metric-by-stratum tests within analysis domain"
    )

    electrode_rows: list[dict[str, Any]] = []
    electrode_grouping = [*strata, "electrode"]
    for keys, selected in _group_iterator(attached, electrode_grouping):
        stratum = dict(zip(electrode_grouping, keys))
        for metric in metrics:
            comparison = _compare_one(
                selected[
                    [
                        "subject_id",
                        "group",
                        "target_pd",
                        "age_years",
                        "sex_male",
                        *(
                            ["match_pair_id"]
                            if "match_pair_id" in selected
                            else []
                        ),
                        metric,
                    ]
                ].rename(columns={metric: "value"}),
                confidence_level=confidence_level,
            )
            if comparison is None:
                continue
            electrode_rows.append(
                {
                    "domain": domain,
                    "analysis_level": "electrode_exploratory",
                    **stratum,
                    "metric": metric,
                    **comparison,
                }
            )
    electrode_statistics = pd.DataFrame.from_records(electrode_rows)
    if electrode_statistics.empty:
        raise ValueError(f"{domain} produced no analyzable electrode tests")
    electrode_statistics["primary_p_fdr_bh_within_feature"] = np.nan
    electrode_statistics["primary_fdr_reject_within_feature"] = False
    feature_grouping = [*strata, "metric"]
    grouping: str | list[str] = (
        feature_grouping[0] if len(feature_grouping) == 1 else feature_grouping
    )
    for _, indices in electrode_statistics.groupby(grouping, sort=False).groups.items():
        adjusted, rejected = fdr_bh(
            electrode_statistics.loc[indices, "primary_p_value"].to_numpy(float),
            fdr_alpha,
        )
        electrode_statistics.loc[
            indices, "primary_p_fdr_bh_within_feature"
        ] = adjusted
        electrode_statistics.loc[
            indices, "primary_fdr_reject_within_feature"
        ] = rejected
    domain_adjusted, domain_rejected = fdr_bh(
        electrode_statistics["primary_p_value"].to_numpy(float), fdr_alpha
    )
    electrode_statistics["primary_p_fdr_bh_domain"] = domain_adjusted
    electrode_statistics["primary_fdr_reject_domain"] = domain_rejected
    electrode_statistics["fdr_alpha"] = float(fdr_alpha)
    electrode_statistics["formal_spatial_inference"] = (
        "primary_fdr_reject_domain; within-feature FDR is secondary localization"
    )
    return (
        subject_statistics.sort_values([*strata, "metric"]).reset_index(drop=True),
        electrode_statistics.sort_values([*strata, "metric", "electrode"]).reset_index(
            drop=True
        ),
    )
