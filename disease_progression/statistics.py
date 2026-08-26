"""Age/sex-adjusted cross-sectional severity-axis association statistics."""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from quantitative_behavioral.statistics import (
    fdr_bh,
    partial_spearman,
    unadjusted_spearman,
)


def _bootstrap_interval(
    x: np.ndarray,
    y: np.ndarray,
    covariates: np.ndarray | None,
    estimator: Callable,
    *,
    n_resamples: int,
    confidence_level: float,
    seed: int,
) -> tuple[float, float, int]:
    rng = np.random.default_rng(seed)
    estimates: list[float] = []
    for _ in range(n_resamples):
        indices = rng.integers(0, len(x), size=len(x))
        if covariates is None:
            estimate, _ = estimator(x[indices], y[indices])
        else:
            estimate, _ = estimator(x[indices], y[indices], covariates[indices])
        if np.isfinite(estimate):
            estimates.append(float(estimate))
    if len(estimates) < max(20, int(0.8 * n_resamples)):
        return np.nan, np.nan, len(estimates)
    tail = (1.0 - confidence_level) / 2.0
    lower, upper = np.quantile(estimates, [tail, 1.0 - tail])
    return float(lower), float(upper), len(estimates)


def correlate_progression_features(
    features: pd.DataFrame,
    dictionary: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """Correlate every subject feature with UPDRS and complementary MOCA."""
    settings = config["analysis"]
    outcomes = [str(settings["primary_outcome"]), *map(str, settings["secondary_outcomes"])]
    minimum = int(settings["minimum_subjects"])
    n_resamples = int(settings["bootstrap_resamples"])
    confidence_level = float(settings["bootstrap_confidence_level"])
    base_seed = int(settings["random_seed"])
    rows: list[dict] = []
    for outcome_index, outcome in enumerate(outcomes):
        worse_direction = 1.0 if outcome == "updrs" else -1.0
        for feature_index, specification in dictionary.reset_index(drop=True).iterrows():
            feature_id = str(specification["feature_id"])
            table = features.loc[features["feature_id"].eq(feature_id)].dropna(
                subset=["value", outcome, "age_years", "sex_male"]
            )
            if table["subject_id"].duplicated().any():
                raise ValueError(f"{feature_id}/{outcome}: duplicated subjects")
            x = table["value"].to_numpy(float)
            y = table[outcome].to_numpy(float)
            covariates = table[["age_years", "sex_male"]].to_numpy(float)
            methods = (
                (
                    "partial_spearman_age_sex",
                    partial_spearman,
                    covariates,
                    "rank-residualized for age and sex",
                ),
                (
                    "spearman_unadjusted",
                    unadjusted_spearman,
                    None,
                    "unadjusted sensitivity analysis",
                ),
            )
            for method_index, (method, estimator, covariate_values, adjustment) in enumerate(methods):
                if len(table) < minimum:
                    estimate = p_value = ci_lower = ci_upper = np.nan
                    valid_bootstraps = 0
                    status = "insufficient_complete_subjects"
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
                        seed=(
                            base_seed
                            + outcome_index * 1_000_003
                            + feature_index * 101
                            + method_index * 10_007
                        ),
                    )
                    status = "ok" if np.isfinite(estimate) else "non_estimable"
                rows.append(
                    {
                        **specification.to_dict(),
                        "outcome": outcome,
                        "outcome_role": "primary" if outcome == settings["primary_outcome"] else "secondary",
                        "outcome_higher_means": "worse" if outcome == "updrs" else "better",
                        "method": method,
                        "adjustment": adjustment,
                        "analysis_status": status,
                        "n_subjects": int(len(table)),
                        "estimate": estimate,
                        "progression_aligned_estimate": (
                            float(worse_direction * estimate) if np.isfinite(estimate) else np.nan
                        ),
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
    for _, indices in result.groupby(["outcome", "family", "method"], sort=False).groups.items():
        adjusted, rejected = fdr_bh(
            result.loc[indices, "p_value"].to_numpy(float),
            alpha=float(settings["fdr_alpha"]),
        )
        result.loc[indices, "p_fdr_bh"] = adjusted
        result.loc[indices, "fdr_reject"] = rejected
    result["fdr_scope"] = "within outcome, feature family, and correlation method"
    return result.sort_values(["outcome", "family", "feature_id", "method"]).reset_index(drop=True)


def clinical_axis_association(cohort: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Quantify how UPDRS and MOCA relate after age/sex adjustment."""
    complete = cohort.dropna(subset=["updrs", "moca", "age_years", "sex_male"])
    x = complete["updrs"].to_numpy(float)
    y = complete["moca"].to_numpy(float)
    covariates = complete[["age_years", "sex_male"]].to_numpy(float)
    adjusted, adjusted_p = partial_spearman(x, y, covariates)
    unadjusted, unadjusted_p = unadjusted_spearman(x, y)
    return pd.DataFrame.from_records(
        [
            {
                "axis_x": "updrs",
                "axis_y": "moca",
                "n_subjects": int(len(complete)),
                "method": "partial_spearman_age_sex",
                "estimate": adjusted,
                "p_value": adjusted_p,
            },
            {
                "axis_x": "updrs",
                "axis_y": "moca",
                "n_subjects": int(len(complete)),
                "method": "spearman_unadjusted",
                "estimate": unadjusted,
                "p_value": unadjusted_p,
            },
        ]
    )
