"""Sensitivity analyses requiring at least 60 seconds of accepted EEG."""

from __future__ import annotations

import copy
import json
import logging
import platform
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.runtime import configure_runtime

configure_runtime()

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
import statsmodels
import statsmodels.api as sm
from scipy.stats import mannwhitneyu, ttest_ind, ttest_rel, wilcoxon

from exploration.matching import remove_demographic_predictors
from exploration.modeling import (
    average_repeated_predictions,
    bootstrap_performance,
    run_nested_validation,
)
from exploration.pipeline import load_exploration_config
from quantitative_behavioral.pipeline import load_analysis_config as load_behavioral_config
from quantitative_behavioral.statistics import correlate_subject_features, fdr_bh


IDENTIFIER_COLUMNS = {
    "subject_id",
    "participant_id",
    "group",
    "target_pd",
    "match_pair_id",
    "cv_group",
}
NON_EEG_COLUMNS = {"age_years", "sex_male", "moca"}


def load_duration_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as stream:
        config = json.load(stream)
    required = {
        "minimum_accepted_duration_seconds",
        "preprocessing_epoch_duration_seconds",
        "input",
        "output_dirs",
        "statistics",
        "plots",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Missing duration-QC config sections: {missing}")
    threshold = float(config["minimum_accepted_duration_seconds"])
    epoch_duration = float(config["preprocessing_epoch_duration_seconds"])
    if threshold != 60.0:
        raise ValueError("The prespecified accepted-duration threshold must be 60 seconds")
    if epoch_duration != 4.0:
        raise ValueError("The primary preprocessing epoch duration must remain 4 seconds")
    if threshold / epoch_duration != int(threshold / epoch_duration):
        raise ValueError("The duration threshold must equal a whole number of epochs")
    if config["statistics"].get("group_fdr_scope") != "within_feature_family":
        raise ValueError("Group-comparison FDR must remain within feature family")
    alpha = float(config["statistics"]["fdr_alpha"])
    if not 0.0 < alpha < 1.0:
        raise ValueError("statistics.fdr_alpha must lie between zero and one")
    return config


def _read_csv(path: str | Path, required: set[str]) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Required duration-QC input does not exist: {path}")
    table = pd.read_csv(path)
    missing = sorted(required - set(table))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    return table


def _feature_family(feature: str) -> str:
    if feature.startswith("bout_ordinal_"):
        return "bout_ordinal"
    if feature.startswith("bout_"):
        return "bout_properties"
    if feature.startswith("typical_"):
        return "typical_bout"
    if feature.startswith("ordinal_"):
        return "ordinal"
    if feature.startswith("psd_"):
        return "psd"
    if feature.startswith("aperiodic_"):
        return "aperiodic"
    return "other"


def select_duration_cohort(
    modeling_table: pd.DataFrame,
    preprocessing_qc: pd.DataFrame,
    *,
    minimum_seconds: float,
    pair_table: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Filter subjects by accepted duration, preserving complete matched pairs."""
    required_modeling = {"subject_id", "group", "target_pd"}
    missing = sorted(required_modeling - set(modeling_table))
    if missing:
        raise ValueError(f"Modeling table is missing columns: {missing}")
    required_qc = {
        "subject_id",
        "group",
        "usable_duration_sec",
        "n_epochs_retained",
        "percent_epochs_retained",
    }
    missing = sorted(required_qc - set(preprocessing_qc))
    if missing:
        raise ValueError(f"Preprocessing QC is missing columns: {missing}")
    if modeling_table["subject_id"].duplicated().any():
        raise ValueError("Modeling table must contain one row per subject")
    if preprocessing_qc["subject_id"].duplicated().any():
        raise ValueError("Preprocessing QC must contain one row per subject")

    model = modeling_table.copy()
    model["subject_id"] = model["subject_id"].astype(str)
    qc = preprocessing_qc.copy()
    qc["subject_id"] = qc["subject_id"].astype(str)
    audit = model[["subject_id", "group", "target_pd"]].merge(
        qc[
            [
                "subject_id",
                "group",
                "usable_duration_sec",
                "n_epochs_retained",
                "percent_epochs_retained",
            ]
        ],
        on="subject_id",
        how="left",
        suffixes=("", "_qc"),
        validate="one_to_one",
    )
    if audit["usable_duration_sec"].isna().any():
        missing_subjects = audit.loc[
            audit["usable_duration_sec"].isna(), "subject_id"
        ].tolist()
        raise ValueError(f"Subjects lack preprocessing duration QC: {missing_subjects}")
    if not audit["group"].eq(audit["group_qc"]).all():
        raise ValueError("Group labels disagree between modeling and preprocessing QC")
    audit = audit.drop(columns="group_qc")
    audit["minimum_accepted_duration_seconds"] = float(minimum_seconds)
    audit["individual_duration_qualified"] = audit["usable_duration_sec"].ge(
        float(minimum_seconds)
    )
    audit["analysis_included"] = audit["individual_duration_qualified"]
    audit["exclusion_reason"] = np.where(
        audit["individual_duration_qualified"],
        "",
        "accepted_duration_below_60_seconds",
    )
    retained_pairs = pd.DataFrame()

    if pair_table is not None:
        required_pairs = {"match_pair_id", "control_subject_id", "pd_subject_id"}
        missing = sorted(required_pairs - set(pair_table))
        if missing:
            raise ValueError(f"Matched pair table is missing columns: {missing}")
        pairs = pair_table.copy()
        pairs["control_subject_id"] = pairs["control_subject_id"].astype(str)
        pairs["pd_subject_id"] = pairs["pd_subject_id"].astype(str)
        pairs["match_pair_id"] = pairs["match_pair_id"].astype(str)
        if pairs["match_pair_id"].duplicated().any():
            raise ValueError("Matched pair IDs must be unique")
        qualification = audit.set_index("subject_id")[
            "individual_duration_qualified"
        ].to_dict()
        for column in ("control_subject_id", "pd_subject_id"):
            missing_subjects = sorted(set(pairs[column].astype(str)) - set(qualification))
            if missing_subjects:
                raise ValueError(
                    f"Pair table contains subjects absent from modeling table: {missing_subjects}"
                )
        pairs["control_duration_qualified"] = pairs["control_subject_id"].astype(
            str
        ).map(qualification)
        pairs["pd_duration_qualified"] = pairs["pd_subject_id"].astype(str).map(
            qualification
        )
        pairs["pair_duration_qualified"] = pairs[
            ["control_duration_qualified", "pd_duration_qualified"]
        ].all(axis=1)
        retained_pairs = pairs.loc[pairs["pair_duration_qualified"]].copy()
        retained_ids = set(
            retained_pairs[["control_subject_id", "pd_subject_id"]]
            .astype(str)
            .to_numpy()
            .ravel()
        )
        audit["analysis_included"] = audit["subject_id"].isin(retained_ids)
        partner_excluded = (
            audit["individual_duration_qualified"] & ~audit["analysis_included"]
        )
        audit.loc[
            partner_excluded, "exclusion_reason"
        ] = "matched_partner_below_60_seconds"
        pair_lookup = pd.concat(
            [
                retained_pairs[["match_pair_id", "control_subject_id"]].rename(
                    columns={"control_subject_id": "subject_id"}
                ),
                retained_pairs[["match_pair_id", "pd_subject_id"]].rename(
                    columns={"pd_subject_id": "subject_id"}
                ),
            ],
            ignore_index=True,
        )
        model = model.drop(columns=["match_pair_id", "cv_group"], errors="ignore")
        model = model.merge(pair_lookup, on="subject_id", how="inner", validate="one_to_one")
        model["cv_group"] = model["match_pair_id"]
    else:
        retained_ids = set(audit.loc[audit["analysis_included"], "subject_id"])
        model = model.loc[model["subject_id"].isin(retained_ids)].copy()

    model = model.sort_values("subject_id").reset_index(drop=True)
    audit = audit.sort_values("subject_id").reset_index(drop=True)
    if set(model["group"]) != {"PD", "Control"}:
        raise ValueError("Duration-qualified cohort must retain PD and Control subjects")
    if pair_table is not None:
        if not (model.groupby("match_pair_id")["target_pd"].nunique() == 2).all():
            raise RuntimeError("Duration-qualified matched pairs are incomplete")
    return model, audit, retained_pairs


def _hedges_g(pd_values: np.ndarray, control_values: np.ndarray) -> float:
    degrees = len(pd_values) + len(control_values) - 2
    pooled_variance = (
        (len(pd_values) - 1) * np.var(pd_values, ddof=1)
        + (len(control_values) - 1) * np.var(control_values, ddof=1)
    ) / degrees
    if pooled_variance <= 0.0:
        return np.nan
    correction = 1.0 - 3.0 / (4.0 * degrees - 1.0)
    return float(
        correction
        * (np.mean(pd_values) - np.mean(control_values))
        / np.sqrt(pooled_variance)
    )


def compare_group_features(
    table: pd.DataFrame,
    features: list[str],
    *,
    matched: bool,
    confidence_level: float,
    fdr_alpha: float,
) -> pd.DataFrame:
    """Recompute transparent PD-Control comparisons after duration filtering."""
    rows: list[dict[str, Any]] = []
    alpha = 1.0 - float(confidence_level)
    for feature in features:
        columns = ["subject_id", "group", "target_pd", feature]
        if matched:
            columns.append("match_pair_id")
        else:
            columns.extend(["age_years", "sex_male"])
        selected = table[columns].dropna().copy()
        pd_values = selected.loc[selected["group"].eq("PD"), feature].to_numpy(float)
        control_values = selected.loc[
            selected["group"].eq("Control"), feature
        ].to_numpy(float)
        if min(len(pd_values), len(control_values)) < 3:
            continue
        welch = ttest_ind(pd_values, control_values, equal_var=False)
        mann = mannwhitneyu(pd_values, control_values, alternative="two-sided")
        row: dict[str, Any] = {
            "feature": feature,
            "family": _feature_family(feature),
            "n_pd": len(pd_values),
            "n_control": len(control_values),
            "pd_mean": float(np.mean(pd_values)),
            "control_mean": float(np.mean(control_values)),
            "mean_difference_pd_minus_control": float(
                np.mean(pd_values) - np.mean(control_values)
            ),
            "hedges_g_pd_minus_control": _hedges_g(pd_values, control_values),
            "welch_t": float(welch.statistic),
            "welch_p_value": float(welch.pvalue),
            "mann_whitney_u": float(mann.statistic),
            "mann_whitney_p_value": float(mann.pvalue),
            "confidence_level": float(confidence_level),
        }
        if matched:
            paired = selected.pivot(
                index="match_pair_id", columns="group", values=feature
            ).dropna()
            differences = paired["PD"].to_numpy(float) - paired["Control"].to_numpy(
                float
            )
            paired_t = ttest_rel(paired["PD"], paired["Control"])
            try:
                signed_rank = wilcoxon(differences, alternative="two-sided")
                signed_rank_statistic = float(signed_rank.statistic)
                signed_rank_p = float(signed_rank.pvalue)
            except ValueError:
                signed_rank_statistic = 0.0
                signed_rank_p = 1.0
            standard_deviation = float(np.std(differences, ddof=1))
            row.update(
                {
                    "primary_model": "paired t test within retained demographic pairs",
                    "n_pairs": len(paired),
                    "primary_effect": float(np.mean(differences)),
                    "primary_ci_lower": float(
                        np.mean(differences)
                        - scipy.stats.t.ppf(1.0 - alpha / 2.0, len(differences) - 1)
                        * scipy.stats.sem(differences)
                    ),
                    "primary_ci_upper": float(
                        np.mean(differences)
                        + scipy.stats.t.ppf(1.0 - alpha / 2.0, len(differences) - 1)
                        * scipy.stats.sem(differences)
                    ),
                    "primary_p_value": float(paired_t.pvalue),
                    "paired_standardized_effect_dz": (
                        float(np.mean(differences) / standard_deviation)
                        if standard_deviation > 0.0
                        else np.nan
                    ),
                    "wilcoxon_signed_rank": signed_rank_statistic,
                    "wilcoxon_signed_rank_p_value": signed_rank_p,
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
                selected[feature].to_numpy(float),
                sm.add_constant(design, has_constant="add"),
            ).fit(cov_type="HC3")
            interval = fitted.conf_int(alpha=alpha).loc["pd_indicator"]
            row.update(
                {
                    "primary_model": "OLS: feature ~ PD + age + sex; HC3 robust SE",
                    "n_pairs": np.nan,
                    "primary_effect": float(fitted.params["pd_indicator"]),
                    "primary_ci_lower": float(interval.iloc[0]),
                    "primary_ci_upper": float(interval.iloc[1]),
                    "primary_p_value": float(fitted.pvalues["pd_indicator"]),
                    "paired_standardized_effect_dz": np.nan,
                    "wilcoxon_signed_rank": np.nan,
                    "wilcoxon_signed_rank_p_value": np.nan,
                }
            )
        rows.append(row)
    result = pd.DataFrame.from_records(rows)
    if result.empty:
        raise ValueError("No group-comparison features were analyzable")
    result["primary_p_fdr_bh"] = np.nan
    result["primary_fdr_reject"] = False
    result["fdr_alpha"] = float(fdr_alpha)
    for _, indices in result.groupby("family", sort=False).groups.items():
        adjusted, rejected = fdr_bh(
            result.loc[indices, "primary_p_value"].to_numpy(float), float(fdr_alpha)
        )
        result.loc[indices, "primary_p_fdr_bh"] = adjusted
        result.loc[indices, "primary_fdr_reject"] = rejected
    return result.sort_values(["family", "primary_p_value", "feature"]).reset_index(
        drop=True
    )


def _write_csv(table: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False, float_format="%.17g")


def _save_figure(fig: Any, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _plot_duration_audit(audit: pd.DataFrame, path: Path, dpi: int) -> None:
    colors = {"PD": "#D55E00", "Control": "#0072B2"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    rng = np.random.default_rng(60)
    maximum = max(360.0, float(audit["usable_duration_sec"].max()) + 20.0)
    for index, group in enumerate(("Control", "PD"), start=1):
        values = audit.loc[audit["group"].eq(group), "usable_duration_sec"].to_numpy(
            float
        )
        axes[0].scatter(
            index + rng.uniform(-0.08, 0.08, len(values)),
            values,
            color=colors[group],
            alpha=0.7,
            s=22,
            label=group,
        )
        axes[1].hist(
            values,
            bins=np.arange(0, maximum, 20.0),
            histtype="step",
            linewidth=2,
            color=colors[group],
            label=group,
        )
    for axis in axes:
        axis.grid(alpha=0.2)
    axes[0].set(
        xticks=[1, 2],
        xticklabels=["Control", "PD"],
        ylabel="Accepted EEG duration (s)",
        title="Subject-level accepted duration",
    )
    axes[0].axhline(60.0, color="black", linestyle="--", linewidth=1.2)
    axes[1].axvline(60.0, color="black", linestyle="--", linewidth=1.2)
    axes[1].set(
        xlabel="Accepted EEG duration (s)",
        ylabel="Subjects",
        title="Duration distribution",
    )
    axes[1].legend(frameon=False)
    fig.suptitle("Accepted-duration QC sensitivity threshold: 60 seconds")
    fig.tight_layout()
    _save_figure(fig, path, dpi)


def _plot_group_effects(comparisons: pd.DataFrame, path: Path, dpi: int) -> None:
    selected = comparisons.assign(
        magnitude=comparisons["hedges_g_pd_minus_control"].abs()
    ).nlargest(30, "magnitude")
    selected = selected.sort_values("hedges_g_pd_minus_control")
    colors = np.where(selected["primary_fdr_reject"], "#009E73", "0.55")
    fig, axis = plt.subplots(figsize=(10, max(5.5, 0.27 * len(selected))))
    axis.barh(
        np.arange(len(selected)),
        selected["hedges_g_pd_minus_control"],
        color=colors,
    )
    axis.axvline(0.0, color="black", linewidth=1)
    axis.set_yticks(np.arange(len(selected)), selected["feature"], fontsize=7)
    axis.set(
        xlabel="Hedges g (PD minus Control)",
        title="Largest standardized group differences after duration QC",
    )
    axis.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    _save_figure(fig, path, dpi)


def _plot_estimate_stability(
    comparison: pd.DataFrame,
    *,
    original_column: str,
    sensitivity_column: str,
    title: str,
    xlabel: str,
    ylabel: str,
    path: Path,
    dpi: int,
) -> None:
    selected = comparison.dropna(subset=[original_column, sensitivity_column])
    if selected.empty:
        return
    limits = np.asarray(
        [
            min(selected[original_column].min(), selected[sensitivity_column].min()),
            max(selected[original_column].max(), selected[sensitivity_column].max()),
        ]
    )
    padding = max(0.02, 0.05 * np.ptp(limits))
    limits = limits + np.asarray([-padding, padding])
    fig, axis = plt.subplots(figsize=(6.2, 6.0))
    axis.scatter(
        selected[original_column],
        selected[sensitivity_column],
        s=22,
        alpha=0.65,
        color="#0072B2",
    )
    axis.plot(limits, limits, color="0.45", linestyle="--")
    axis.set(xlim=limits, ylim=limits, xlabel=xlabel, ylabel=ylabel, title=title)
    axis.grid(alpha=0.2)
    fig.tight_layout()
    _save_figure(fig, path, dpi)


def _report(
    path: Path,
    *,
    mode: str,
    audit: pd.DataFrame,
    retained_pairs: pd.DataFrame,
    group_comparisons: pd.DataFrame,
    correlation_comparison: pd.DataFrame,
    performance_comparison: pd.DataFrame,
) -> None:
    included = audit.loc[audit["analysis_included"]]
    excluded = audit.loc[~audit["analysis_included"]]
    lines = [
        "# Accepted-duration QC sensitivity",
        "",
        "The primary pipeline remains based on non-overlapping 4-second epochs. This "
        "sensitivity analysis retains only participants with at least 60 seconds (15 "
        "epochs) of accepted EEG. Subject-level EEG quantities are reused unchanged, so "
        "the comparison isolates cohort exclusion rather than recomputing electrodes, "
        "filters, decompositions, or features.",
        "",
        f"Cohort mode: **{mode}**.",
        f"Included: **{len(included)}** subjects "
        f"({int(included['group'].eq('PD').sum())} PD; "
        f"{int(included['group'].eq('Control').sum())} Control).",
        f"Excluded: **{len(excluded)}** subjects.",
    ]
    if mode == "matched":
        lines.append(
            f"Retained complete demographic pairs: **{len(retained_pairs)}**. If either "
            "member failed duration QC, both members were excluded."
        )
    if len(excluded):
        lines.extend(
            [
                "",
                "Excluded IDs: "
                + ", ".join(
                    f"{row.subject_id} ({row.usable_duration_sec:g} s; {row.exclusion_reason})"
                    for row in excluded.itertuples(index=False)
                ),
            ]
        )
    significant_groups = group_comparisons.loc[
        group_comparisons["primary_fdr_reject"]
    ]
    lines.extend(
        [
            "",
            "## PD versus Control quantities",
            "",
            f"{len(significant_groups)} of {len(group_comparisons)} conservative modeled "
            "EEG features pass BH FDR within feature family after duration QC.",
        ]
    )
    if len(significant_groups):
        lines.extend(["", "| Feature | Effect | q |", "|---|---:|---:|"])
        for row in significant_groups.itertuples(index=False):
            lines.append(
                f"| {row.feature} | {row.primary_effect:.4g} | "
                f"{row.primary_p_fdr_bh:.4g} |"
            )
    partial = correlation_comparison.loc[
        correlation_comparison["method"].eq("partial_spearman_age_sex")
    ]
    primary_significant = set(
        partial.loc[partial["primary_fdr_reject"].fillna(False), "feature_id"]
    )
    sensitivity_significant = set(
        partial.loc[partial["duration_qc_fdr_reject"].fillna(False), "feature_id"]
    )
    lines.extend(
        [
            "",
            "## MOCA associations",
            "",
            f"Age/sex-adjusted partial Spearman associations passing their original "
            f"within-family BH correction: primary={len(primary_significant)}, "
            f"duration-QC={len(sensitivity_significant)}.",
            "Lost after duration QC: "
            + (", ".join(sorted(primary_significant - sensitivity_significant)) or "none"),
            "New after duration QC: "
            + (", ".join(sorted(sensitivity_significant - primary_significant)) or "none"),
        ]
    )
    if not performance_comparison.empty:
        auc = performance_comparison.loc[performance_comparison["metric"].eq("roc_auc")]
        lines.extend(["", "## Prediction models", "", "| Model | Primary AUC | Duration-QC AUC |", "|---|---:|---:|"])
        for row in auc.sort_values("duration_qc_estimate", ascending=False).itertuples(
            index=False
        ):
            lines.append(
                f"| {row.model} | {row.primary_estimate:.3f} | "
                f"{row.duration_qc_estimate:.3f} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This is a prespecified robustness analysis, not a replacement primary cohort. "
            "A result that loses significance may reflect reduced sample size rather than "
            "bias from short recordings; compare effect direction and uncertainty as well as "
            "the FDR decision.",
            "",
            "Complete machine-readable tables are under `metrics/`.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_duration_sensitivity(
    config_path: str | Path = "duration_qc_analysis/config.json",
    *,
    matched: bool = False,
    overwrite: bool = False,
    quick: bool = False,
    skip_models: bool = False,
) -> dict[str, Any]:
    config_path = Path(config_path)
    config = load_duration_config(config_path)
    mode = "matched" if matched else "full"
    output_dir = Path(config["output_dirs"][mode])
    sentinel = output_dir / "manifest.json"
    if sentinel.exists() and not overwrite:
        raise FileExistsError(
            f"Duration-QC output exists at {sentinel}; rerun with --overwrite"
        )
    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"duration_qc_analysis.{mode}")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (
        logging.StreamHandler(),
        logging.FileHandler(output_dir / "duration_qc_analysis.log", mode="w"),
    ):
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    source = config["input"][mode]
    modeling = _read_csv(
        source["modeling_table"], {"subject_id", "group", "target_pd", "age_years", "sex_male"}
    )
    qc = _read_csv(
        config["input"]["preprocessing_qc_file"],
        {
            "subject_id",
            "group",
            "usable_duration_sec",
            "n_epochs_retained",
            "percent_epochs_retained",
        },
    )
    pair_table = (
        _read_csv(
            config["input"]["matched_pair_file"],
            {"match_pair_id", "control_subject_id", "pd_subject_id"},
        )
        if matched
        else None
    )
    qualified, audit, retained_pairs = select_duration_cohort(
        modeling,
        qc,
        minimum_seconds=float(config["minimum_accepted_duration_seconds"]),
        pair_table=pair_table,
    )
    logger.info(
        "Duration-qualified %s cohort: %d/%d subjects",
        mode,
        len(qualified),
        len(modeling),
    )

    exploration_config = load_exploration_config(
        config["input"]["exploration_config_file"]
    )
    models = (
        remove_demographic_predictors(exploration_config["models"])
        if matched
        else exploration_config["models"]
    )
    modeled_features = list(
        dict.fromkeys(
            feature
            for specification in models.values()
            for feature in specification["features"]
            if feature not in IDENTIFIER_COLUMNS | NON_EEG_COLUMNS
        )
    )
    missing = sorted(set(modeled_features) - set(qualified))
    if missing:
        raise ValueError(f"Duration-QC modeling table is missing features: {missing}")
    group_comparisons = compare_group_features(
        qualified,
        modeled_features,
        matched=matched,
        confidence_level=float(config["statistics"]["confidence_level"]),
        fdr_alpha=float(config["statistics"]["fdr_alpha"]),
    )

    behavioral_features = _read_csv(
        source["behavioral_features"],
        {"subject_id", "group", "feature_id", "value", "moca", "age_years", "sex_male"},
    )
    dictionary = _read_csv(
        source["behavioral_dictionary"], {"feature_id", "family", "feature_label"}
    )
    qualified_ids = set(qualified["subject_id"].astype(str))
    behavioral_features = behavioral_features.loc[
        behavioral_features["subject_id"].astype(str).isin(qualified_ids)
    ].copy()
    behavioral_config = load_behavioral_config(
        config["input"]["quantitative_config_file"]
    )
    if quick:
        behavioral_config = copy.deepcopy(behavioral_config)
        behavioral_config["analysis"]["bootstrap_resamples"] = 100
    duration_correlations = correlate_subject_features(
        behavioral_features, dictionary, behavioral_config
    )
    primary_correlations = _read_csv(
        source["behavioral_correlations"],
        {"feature_id", "method", "estimate", "p_value", "p_fdr_bh", "fdr_reject"},
    )
    correlation_comparison = primary_correlations[
        ["feature_id", "method", "estimate", "p_value", "p_fdr_bh", "fdr_reject"]
    ].merge(
        duration_correlations[
            ["feature_id", "method", "estimate", "p_value", "p_fdr_bh", "fdr_reject", "n_subjects"]
        ],
        on=["feature_id", "method"],
        how="outer",
        suffixes=("_primary", "_duration_qc"),
        validate="one_to_one",
    ).rename(
        columns={
            "estimate_primary": "primary_estimate",
            "p_value_primary": "primary_p_value",
            "p_fdr_bh_primary": "primary_p_fdr_bh",
            "fdr_reject_primary": "primary_fdr_reject",
            "estimate_duration_qc": "duration_qc_estimate",
            "p_value_duration_qc": "duration_qc_p_value",
            "p_fdr_bh_duration_qc": "duration_qc_p_fdr_bh",
            "fdr_reject_duration_qc": "duration_qc_fdr_reject",
        }
    )

    performance = pd.DataFrame()
    performance_comparison = pd.DataFrame()
    if not skip_models:
        validation = copy.deepcopy(exploration_config["validation"])
        if quick:
            validation["outer_repeats"] = 2
            validation["bootstrap_resamples"] = 100
        logger.info(
            "Running duration-QC nested validation: models=%d, repeats=%d",
            len(models),
            int(validation["outer_repeats"]),
        )
        predictions, fold_metrics, coefficients = run_nested_validation(
            qualified, models, validation
        )
        averaged = average_repeated_predictions(predictions)
        performance = bootstrap_performance(
            averaged,
            n_resamples=int(validation["bootstrap_resamples"]),
            seed=int(validation["random_seed"]) + 60,
        )
        primary_performance = _read_csv(
            source["model_performance"], {"model", "metric", "estimate"}
        )
        performance_comparison = primary_performance.merge(
            performance,
            on=["model", "metric"],
            how="inner",
            suffixes=("_primary", "_duration_qc"),
            validate="one_to_one",
        ).rename(
            columns={
                "estimate_primary": "primary_estimate",
                "estimate_duration_qc": "duration_qc_estimate",
            }
        )
        _write_csv(predictions, output_dir / "predictions" / "repeated_outer_predictions.csv")
        _write_csv(averaged, output_dir / "predictions" / "subject_out_of_fold_predictions.csv")
        _write_csv(fold_metrics, output_dir / "cross_validation" / "outer_fold_metrics.csv")
        _write_csv(coefficients, output_dir / "cross_validation" / "outer_fold_coefficients.csv")
        _write_csv(performance, output_dir / "metrics" / "model_performance.csv")
        _write_csv(
            performance_comparison,
            output_dir / "metrics" / "model_performance_primary_vs_duration_qc.csv",
        )

    _write_csv(audit, output_dir / "metrics" / "duration_cohort_audit.csv")
    _write_csv(
        audit.loc[~audit["analysis_included"]],
        output_dir / "metrics" / "excluded_subjects.csv",
    )
    if matched:
        _write_csv(retained_pairs, output_dir / "metrics" / "retained_match_pairs.csv")
    _write_csv(qualified, output_dir / "features" / "qualified_modeling_table.csv")
    _write_csv(group_comparisons, output_dir / "metrics" / "group_comparisons.csv")
    _write_csv(
        duration_correlations,
        output_dir / "metrics" / "moca_correlations.csv",
    )
    _write_csv(
        correlation_comparison,
        output_dir / "metrics" / "moca_correlations_primary_vs_duration_qc.csv",
    )

    dpi = int(config["plots"]["dpi"])
    _plot_duration_audit(
        audit, output_dir / "figures" / "accepted_duration_audit.png", dpi
    )
    _plot_group_effects(
        group_comparisons,
        output_dir / "figures" / "group_effect_sizes.png",
        dpi,
    )
    partial = correlation_comparison.loc[
        correlation_comparison["method"].eq("partial_spearman_age_sex")
    ]
    _plot_estimate_stability(
        partial,
        original_column="primary_estimate",
        sensitivity_column="duration_qc_estimate",
        title="MOCA association stability after accepted-duration QC",
        xlabel="Primary partial Spearman rho",
        ylabel="Duration-QC partial Spearman rho",
        path=output_dir / "figures" / "moca_correlation_stability.png",
        dpi=dpi,
    )
    if not performance_comparison.empty:
        _plot_estimate_stability(
            performance_comparison.loc[
                performance_comparison["metric"].eq("roc_auc")
            ],
            original_column="primary_estimate",
            sensitivity_column="duration_qc_estimate",
            title="Prediction stability after accepted-duration QC",
            xlabel="Primary nested-CV ROC AUC",
            ylabel="Duration-QC nested-CV ROC AUC",
            path=output_dir / "figures" / "model_auc_stability.png",
            dpi=dpi,
        )
    _report(
        output_dir / "REPORT.md",
        mode=mode,
        audit=audit,
        retained_pairs=retained_pairs,
        group_comparisons=group_comparisons,
        correlation_comparison=correlation_comparison,
        performance_comparison=performance_comparison,
    )

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config_file": str(config_path.resolve()),
        "analysis_config": config,
        "cohort_mode": mode,
        "minimum_accepted_duration_seconds": float(
            config["minimum_accepted_duration_seconds"]
        ),
        "minimum_retained_epochs": int(
            config["minimum_accepted_duration_seconds"]
            / config["preprocessing_epoch_duration_seconds"]
        ),
        "n_input_subjects": len(audit),
        "n_included_subjects": int(audit["analysis_included"].sum()),
        "n_excluded_subjects": int((~audit["analysis_included"]).sum()),
        "included_group_counts": audit.loc[audit["analysis_included"], "group"]
        .value_counts()
        .to_dict(),
        "n_retained_pairs": int(len(retained_pairs)) if matched else None,
        "models_recomputed": not skip_models,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "statsmodels": statsmodels.__version__,
            "sklearn": sklearn.__version__,
            "matplotlib": matplotlib.__version__,
        },
    }
    sentinel.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    logger.info("Duration-QC sensitivity complete: %s", output_dir)
    return manifest
