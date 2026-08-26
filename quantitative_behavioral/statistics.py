"""Conservative subject-level and spatial MOCA association statistics."""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import (
    mannwhitneyu,
    pearsonr,
    rankdata,
    spearmanr,
    t as student_t,
    ttest_ind,
)


def compare_aperiodic_exponent_groups(
    features: pd.DataFrame,
    *,
    confidence_level: float = 0.95,
    fdr_alpha: float = 0.05,
) -> pd.DataFrame:
    """Compare subject-mean aperiodic exponent between PD and Control.

    The primary group estimate is an OLS PD indicator coefficient
    adjusted for age and sex, with HC3 heteroskedasticity-robust uncertainty.
    Welch's t test, Mann–Whitney U, and Hedges' g are unadjusted sensitivity and
    effect-size summaries.
    """
    required = {
        "subject_id",
        "group",
        "feature_id",
        "value",
        "age_years",
        "sex_male",
    }
    missing = sorted(required - set(features.columns))
    if missing:
        raise ValueError(f"Aperiodic comparison input is missing columns: {missing}")
    specifications = (
        (
            "aperiodic_exponent",
            "BIC-selected aperiodic exponent (1–50 Hz; all 60 electrode fits)",
            "all shared-electrode fits",
        ),
        (
            "aperiodic_exponent_qc",
            "QC-qualified BIC-selected aperiodic exponent (1–50 Hz)",
            "QC-passing electrodes in subjects with at least 80% coverage",
        ),
    )
    rows = []
    for feature_id, feature_label, aggregation_policy in specifications:
        table = features.loc[features["feature_id"].eq(feature_id)].dropna(
            subset=["value", "age_years", "sex_male", "group"]
        )
        if table["subject_id"].duplicated().any():
            raise ValueError(
                f"{feature_id} comparison must have one row per subject"
            )
        if set(table["group"]) != {"PD", "Control"}:
            raise ValueError(f"{feature_id} comparison requires PD and Control groups")
        pd_values = table.loc[
            table["group"].eq("PD"), "value"
        ].to_numpy(dtype=float)
        control_values = table.loc[
            table["group"].eq("Control"), "value"
        ].to_numpy(dtype=float)
        if min(len(pd_values), len(control_values)) < 3:
            raise ValueError(f"{feature_id} requires at least 3 subjects per group")
        welch = ttest_ind(pd_values, control_values, equal_var=False)
        mann_whitney = mannwhitneyu(
            pd_values, control_values, alternative="two-sided"
        )
        pooled_degrees = len(pd_values) + len(control_values) - 2
        pooled_sd = np.sqrt(
            (
                (len(pd_values) - 1) * np.var(pd_values, ddof=1)
                + (len(control_values) - 1) * np.var(control_values, ddof=1)
            )
            / pooled_degrees
        )
        cohen_d = (np.mean(pd_values) - np.mean(control_values)) / pooled_sd
        hedges_correction = 1.0 - 3.0 / (4.0 * pooled_degrees - 1.0)
        hedges_g = hedges_correction * cohen_d
        design = pd.DataFrame(
            {
                "pd_indicator": table["group"].eq("PD").astype(float),
                "age_years": table["age_years"].astype(float),
                "sex_male": table["sex_male"].astype(float),
            },
            index=table.index,
        )
        fitted = sm.OLS(
            table["value"].to_numpy(dtype=float),
            sm.add_constant(design, has_constant="add"),
        ).fit(cov_type="HC3")
        alpha = 1.0 - float(confidence_level)
        interval = fitted.conf_int(alpha=alpha).loc["pd_indicator"]
        rows.append(
            {
                "feature_id": feature_id,
                "feature_label": feature_label,
                "unit_of_analysis": "subject mean across shared electrodes",
                "aggregation_policy": aggregation_policy,
                "n_pd": int(len(pd_values)),
                "n_control": int(len(control_values)),
                "pd_mean": float(np.mean(pd_values)),
                "pd_std": float(np.std(pd_values, ddof=1)),
                "pd_median": float(np.median(pd_values)),
                "control_mean": float(np.mean(control_values)),
                "control_std": float(np.std(control_values, ddof=1)),
                "control_median": float(np.median(control_values)),
                "mean_difference_pd_minus_control": float(
                    np.mean(pd_values) - np.mean(control_values)
                ),
                "hedges_g_pd_minus_control": float(hedges_g),
                "welch_t": float(welch.statistic),
                "welch_p_value": float(welch.pvalue),
                "mann_whitney_u": float(mann_whitney.statistic),
                "mann_whitney_p_value": float(mann_whitney.pvalue),
                "adjusted_model": "OLS: exponent ~ PD + age + sex; HC3 robust SE",
                "adjusted_pd_coefficient": float(fitted.params["pd_indicator"]),
                "adjusted_pd_se_hc3": float(fitted.bse["pd_indicator"]),
                "adjusted_pd_ci_lower": float(interval.iloc[0]),
                "adjusted_pd_ci_upper": float(interval.iloc[1]),
                "adjusted_pd_p_value": float(fitted.pvalues["pd_indicator"]),
                "confidence_level": float(confidence_level),
                "multiplicity_scope": (
                    "primary all-fit estimate plus one formal QC sensitivity"
                ),
            }
        )
    result = pd.DataFrame.from_records(rows)
    adjusted, rejected = fdr_bh(
        result["adjusted_pd_p_value"].to_numpy(dtype=float), alpha=float(fdr_alpha)
    )
    result["adjusted_pd_p_fdr_bh"] = adjusted
    result["adjusted_pd_fdr_reject"] = rejected
    result["fdr_alpha"] = float(fdr_alpha)
    return result


def fdr_bh(p_values: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """Benjamini–Hochberg correction that preserves missing p-values."""
    values = np.asarray(p_values, dtype=float)
    adjusted = np.full(values.shape, np.nan, dtype=float)
    rejected = np.zeros(values.shape, dtype=bool)
    finite_indices = np.flatnonzero(np.isfinite(values))
    if not len(finite_indices):
        return adjusted, rejected
    finite = values[finite_indices]
    order = np.argsort(finite)
    ranked = finite[order]
    m = len(ranked)
    corrected = ranked * m / np.arange(1, m + 1)
    corrected = np.minimum.accumulate(corrected[::-1])[::-1]
    corrected = np.clip(corrected, 0.0, 1.0)
    restored = np.empty_like(corrected)
    restored[order] = corrected
    adjusted[finite_indices] = restored
    rejected[finite_indices] = restored <= float(alpha)
    return adjusted, rejected


def _validate_vectors(x: np.ndarray, y: np.ndarray, covariates: np.ndarray | None) -> None:
    if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape:
        raise ValueError("x and y must be matching one-dimensional vectors")
    if covariates is not None and (
        covariates.ndim != 2 or covariates.shape[0] != len(x)
    ):
        raise ValueError("covariates must have shape (subjects, covariates)")
    arrays = [x, y] + ([] if covariates is None else [covariates])
    if not all(np.all(np.isfinite(values)) for values in arrays):
        raise ValueError("Correlation inputs must be finite after complete-case selection")


def unadjusted_spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    values_x = np.asarray(x, dtype=float)
    values_y = np.asarray(y, dtype=float)
    _validate_vectors(values_x, values_y, None)
    if len(values_x) < 3 or np.allclose(values_x, values_x[0]) or np.allclose(values_y, values_y[0]):
        return np.nan, np.nan
    result = spearmanr(values_x, values_y)
    return float(result.statistic), float(result.pvalue)


def partial_spearman(
    x: np.ndarray, y: np.ndarray, covariates: np.ndarray
) -> tuple[float, float]:
    """Residual correlation after rank-transforming outcome, feature, and covariates."""
    values_x = np.asarray(x, dtype=float)
    values_y = np.asarray(y, dtype=float)
    values_covariates = np.asarray(covariates, dtype=float)
    _validate_vectors(values_x, values_y, values_covariates)
    if len(values_x) < values_covariates.shape[1] + 4:
        return np.nan, np.nan
    if np.allclose(values_x, values_x[0]) or np.allclose(values_y, values_y[0]):
        return np.nan, np.nan
    ranked_x = rankdata(values_x, method="average")
    ranked_y = rankdata(values_y, method="average")
    ranked_covariates = np.column_stack(
        [rankdata(values_covariates[:, index], method="average") for index in range(values_covariates.shape[1])]
    )
    design = np.column_stack([np.ones(len(values_x)), ranked_covariates])
    residual_x = ranked_x - design @ np.linalg.lstsq(design, ranked_x, rcond=None)[0]
    residual_y = ranked_y - design @ np.linalg.lstsq(design, ranked_y, rcond=None)[0]
    if np.allclose(residual_x, 0.0) or np.allclose(residual_y, 0.0):
        return np.nan, np.nan
    correlation = float(pearsonr(residual_x, residual_y).statistic)
    covariate_rank = int(np.linalg.matrix_rank(design) - 1)
    degrees_freedom = len(values_x) - covariate_rank - 2
    if degrees_freedom <= 0 or abs(correlation) >= 1.0:
        p_value = 0.0 if abs(correlation) >= 1.0 else np.nan
    else:
        statistic = correlation * np.sqrt(
            degrees_freedom / max(1.0 - correlation**2, np.finfo(float).tiny)
        )
        p_value = float(2.0 * student_t.sf(abs(statistic), degrees_freedom))
    return correlation, p_value


def _bootstrap_interval(
    x: np.ndarray,
    y: np.ndarray,
    covariates: np.ndarray | None,
    estimator: Callable[..., tuple[float, float]],
    *,
    n_resamples: int,
    confidence_level: float,
    seed: int,
) -> tuple[float, float, int]:
    rng = np.random.default_rng(int(seed))
    estimates: list[float] = []
    for _ in range(int(n_resamples)):
        indices = rng.integers(0, len(x), size=len(x))
        if covariates is None:
            estimate, _ = estimator(x[indices], y[indices])
        else:
            estimate, _ = estimator(x[indices], y[indices], covariates[indices])
        if np.isfinite(estimate):
            estimates.append(float(estimate))
    minimum_valid = max(20, int(0.8 * n_resamples))
    if len(estimates) < minimum_valid:
        return np.nan, np.nan, len(estimates)
    tail = (1.0 - float(confidence_level)) / 2.0
    lower, upper = np.quantile(estimates, [tail, 1.0 - tail])
    return float(lower), float(upper), len(estimates)


def correlate_subject_features(
    features: pd.DataFrame,
    dictionary: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """Calculate primary adjusted and sensitivity unadjusted subject correlations."""
    settings = config["analysis"]
    selected_group = str(settings["primary_group"])
    selected = features.loc[features["group"].eq(selected_group)].copy()
    minimum = int(settings["minimum_subjects"])
    n_resamples = int(settings["bootstrap_resamples"])
    confidence_level = float(settings["bootstrap_confidence_level"])
    seed = int(settings["random_seed"])
    rows: list[dict] = []
    for feature_index, specification in dictionary.reset_index(drop=True).iterrows():
        feature_id = str(specification["feature_id"])
        table = selected.loc[selected["feature_id"].eq(feature_id)].dropna(
            subset=["value", "moca", "age_years", "sex_male"]
        )
        if table["subject_id"].duplicated().any():
            raise ValueError(f"{feature_id}: more than one row per subject")
        x = table["value"].to_numpy(dtype=float)
        y = table["moca"].to_numpy(dtype=float)
        covariates = table[["age_years", "sex_male"]].to_numpy(dtype=float)
        for method_index, (method, estimator, adjustment, covariate_values) in enumerate(
            (
                (
                    "partial_spearman_age_sex",
                    partial_spearman,
                    "rank-residualized for age and sex",
                    covariates,
                ),
                (
                    "spearman_unadjusted",
                    unadjusted_spearman,
                    "unadjusted sensitivity analysis",
                    None,
                ),
            )
        ):
            if len(table) < minimum:
                estimate = p_value = ci_lower = ci_upper = np.nan
                valid_bootstraps = 0
            else:
                if covariate_values is None:
                    estimate, p_value = estimator(x, y)
                else:
                    estimate, p_value = estimator(x, y, covariate_values)
                ci_lower, ci_upper, valid_bootstraps = _bootstrap_interval(
                    x,
                    y,
                    covariate_values,
                    estimator,
                    n_resamples=n_resamples,
                    confidence_level=confidence_level,
                    seed=seed + feature_index * 17 + method_index * 100003,
                )
            rows.append(
                {
                    **specification.to_dict(),
                    "cohort": selected_group,
                    "method": method,
                    "adjustment": adjustment,
                    "n_subjects": int(len(table)),
                    "estimate": estimate,
                    "p_value": p_value,
                    "ci_lower": ci_lower,
                    "ci_upper": ci_upper,
                    "bootstrap_resamples_requested": n_resamples,
                    "bootstrap_resamples_valid": int(valid_bootstraps),
                    "confidence_level": confidence_level,
                }
            )
    result = pd.DataFrame.from_records(rows)
    result["p_fdr_bh"] = np.nan
    result["fdr_reject"] = False
    result["fdr_alpha"] = float(settings["fdr_alpha"])
    for _, indices in result.groupby(["family", "method"], sort=False).groups.items():
        adjusted, rejected = fdr_bh(
            result.loc[indices, "p_value"].to_numpy(dtype=float),
            float(settings["fdr_alpha"]),
        )
        result.loc[indices, "p_fdr_bh"] = adjusted
        result.loc[indices, "fdr_reject"] = rejected
    return result


def correlate_electrodes(
    electrode_features: pd.DataFrame,
    dictionary: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """Calculate secondary age/sex-adjusted spatial correlations within PD."""
    settings = config["analysis"]
    selected_group = str(settings["primary_group"])
    selected = electrode_features.loc[electrode_features["group"].eq(selected_group)]
    minimum = int(settings["minimum_subjects"])
    rows: list[dict] = []
    lookup = dictionary.set_index("feature_id").to_dict(orient="index")
    for (feature_id, electrode), table in selected.groupby(
        ["feature_id", "electrode"], sort=False
    ):
        complete = table.dropna(subset=["value", "moca", "age_years", "sex_male"])
        if complete["subject_id"].duplicated().any():
            raise ValueError(f"{feature_id}/{electrode}: duplicated subjects")
        if len(complete) < minimum:
            estimate = p_value = np.nan
        else:
            estimate, p_value = partial_spearman(
                complete["value"].to_numpy(dtype=float),
                complete["moca"].to_numpy(dtype=float),
                complete[["age_years", "sex_male"]].to_numpy(dtype=float),
            )
        rows.append(
            {
                "feature_id": feature_id,
                **lookup[str(feature_id)],
                "electrode": electrode,
                "cohort": selected_group,
                "method": "partial_spearman_age_sex",
                "n_subjects": int(len(complete)),
                "estimate": estimate,
                "p_value": p_value,
            }
        )
    result = pd.DataFrame.from_records(rows)
    result["p_fdr_bh_within_feature"] = np.nan
    result["fdr_reject_within_feature"] = False
    result["fdr_alpha"] = float(settings["fdr_alpha"])
    for _, indices in result.groupby("feature_id", sort=False).groups.items():
        adjusted, rejected = fdr_bh(
            result.loc[indices, "p_value"].to_numpy(dtype=float),
            float(settings["fdr_alpha"]),
        )
        result.loc[indices, "p_fdr_bh_within_feature"] = adjusted
        result.loc[indices, "fdr_reject_within_feature"] = rejected
    return result
