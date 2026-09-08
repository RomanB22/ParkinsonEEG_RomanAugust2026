#!/usr/bin/env python3
"""Test the incremental value of theta Fisher information for MoCA.

The primary analysis is PD-only and pools the three datasets with MoCA. Dataset
is included as a fixed effect. Nested models separate demographic/contextual
covariates, conventional spectral slowing, aperiodic-aware EEG, and theta
Fisher information. The script also reports repeated stratified
cross-validated performance for the same model blocks.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import f as fisher_f
from scipy.stats import t as student_t
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATASETS = ["primary", "ds007526-1.0.2", "ds008768-1.0.0"]
DATASET_LABELS = {
    "primary": "Primary",
    "ds007526-1.0.2": "ds007526",
    "ds008768-1.0.0": "ds008768",
}

FEATURES = {
    "theta_fisher": "entropy__theta__fisher_information__D4",
    "theta_power": "psd__theta__relative_power",
    "beta_power": "psd__beta__relative_power",
    "alpha_power": "psd__alpha__relative_power",
    "alpha_theta_ratio": "alpha_theta_ratio",
    "aperiodic_offset": "aperiodic__broadband__aperiodic_offset",
    "aperiodic_exponent": "aperiodic__broadband__aperiodic_exponent",
}

MODEL_BLOCKS = {
    "covariates": [],
    "conventional_eeg": ["theta_power", "beta_power", "alpha_theta_ratio"],
    "conventional_plus_aperiodic": [
        "theta_power",
        "beta_power",
        "alpha_theta_ratio",
        "aperiodic_offset",
        "aperiodic_exponent",
    ],
    "conventional_plus_aperiodic_plus_fisher": [
        "theta_power",
        "beta_power",
        "alpha_theta_ratio",
        "aperiodic_offset",
        "aperiodic_exponent",
        "theta_fisher",
    ],
}


def _read_subject(root: Path, dataset: str) -> pd.DataFrame:
    return pd.read_csv(root / "metrics" / dataset / "subject_features.csv.gz", low_memory=False)


def _read_sex(root: Path, dataset: str) -> pd.DataFrame:
    recording = pd.read_csv(
        root / "metrics" / dataset / "recording_features.csv.gz",
        usecols=["participant_id", "sex"],
        low_memory=False,
    )
    recording["sex"] = recording["sex"].astype("string").str.strip()
    recording.loc[recording["sex"].isin(["", "nan", "<NA>"]), "sex"] = pd.NA
    conflicts = (
        recording.dropna(subset=["sex"])
        .groupby("participant_id")["sex"]
        .nunique()
    )
    if (conflicts > 1).any():
        raise ValueError(f"{dataset}: conflicting sex values within participant")
    return recording.drop_duplicates("participant_id")


def _load_analysis_table(root: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    frames = []
    education_column: str | None = None
    for dataset in DATASETS:
        subject = _read_subject(root, dataset)
        sex = _read_sex(root, dataset)
        frame = subject.merge(sex, on="participant_id", how="left", validate="one_to_one").copy()
        frame["dataset"] = dataset
        frames.append(frame)
        for candidate in ("education_years", "education"):
            if candidate in frame.columns:
                education_column = candidate
                break
    table = pd.concat(frames, ignore_index=True)
    table = table.loc[table["group"].astype(str).eq("PD")].copy()
    table["sex"] = table["sex"].astype("string")
    table["moca"] = pd.to_numeric(table["moca"], errors="coerce")
    raw_feature_columns = [
        column for alias, column in FEATURES.items() if alias != "alpha_theta_ratio"
    ]
    required = ["moca", "age_years", "sex", *raw_feature_columns]
    missing_columns = [column for column in required if column not in table.columns]
    if missing_columns:
        raise KeyError(f"Missing required analysis columns: {missing_columns}")
    if education_column is not None:
        table[education_column] = pd.to_numeric(table[education_column], errors="coerce")
        required.append(education_column)
    complete = table.dropna(subset=required).copy()
    complete["alpha_theta_ratio"] = (
        complete[FEATURES["alpha_power"]] / complete[FEATURES["theta_power"]]
    )
    complete = complete.replace([np.inf, -np.inf], np.nan).dropna(subset=["alpha_theta_ratio"])
    complete["dataset_label"] = complete["dataset"].map(DATASET_LABELS)
    notes = {
        "analysis_population": "PD participants from the three MoCA datasets",
        "datasets": DATASETS,
        "excluded_medication_state": "No MoCA is available in the medication-state cohort",
        "education_column": education_column,
        "education_status": "included" if education_column is not None else "unavailable in saved outputs",
        "excluded_rows_before_modeling": int(len(table) - len(complete)),
        "n_complete": int(len(complete)),
        "n_by_dataset": complete["dataset"].value_counts().sort_index().astype(int).to_dict(),
        "n_by_group": complete["group"].value_counts().sort_index().astype(int).to_dict(),
        "conventional_eeg_features": [
            "theta relative power",
            "beta relative power",
            "alpha/theta relative-power ratio",
        ],
        "aperiodic_features": ["broadband aperiodic offset", "broadband aperiodic exponent"],
    }
    if education_column is not None:
        complete["education"] = complete[education_column]
    return complete, notes


def _zscore(table: pd.DataFrame, column: str, output_column: str | None = None) -> str:
    name = output_column or f"{column}_z"
    mean = float(table[column].mean())
    standard_deviation = float(table[column].std(ddof=1))
    if not np.isfinite(standard_deviation) or standard_deviation == 0:
        raise ValueError(f"Cannot standardize {column}: zero or invalid standard deviation")
    table[name] = (table[column] - mean) / standard_deviation
    return name


def _model_formula(continuous_terms: list[str], education_column: str | None) -> str:
    terms = ["age_z", "C(sex)", "C(dataset)"]
    if education_column is not None:
        terms.append("education_z")
    terms.extend(f"{term}_z" for term in continuous_terms)
    return "moca ~ " + " + ".join(terms)


def _fit_nested_models(
    table: pd.DataFrame,
    education_column: str | None,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    _zscore(table, "age_years", "age_z")
    for alias, column in FEATURES.items():
        _zscore(table, column, f"{alias}_z")
    if education_column is not None:
        _zscore(table, "education")
    fits: dict[str, object] = {}
    model_rows = []
    coefficient_rows = []
    for model_name, block in MODEL_BLOCKS.items():
        formula = _model_formula(block, education_column)
        fit = smf.ols(formula, data=table).fit(cov_type="HC3")
        raw_fit = smf.ols(formula, data=table).fit()
        fits[model_name] = raw_fit
        model_rows.append(
            {
                "model": model_name,
                "formula": formula,
                "n": int(fit.nobs),
                "r_squared": float(fit.rsquared),
                "adjusted_r_squared": float(fit.rsquared_adj),
                "aic": float(fit.aic),
                "bic": float(fit.bic),
                "rss": float(np.sum(raw_fit.resid**2)),
            }
        )
        for term in fit.params.index:
            coefficient_rows.append(
                {
                    "model": model_name,
                    "term": term,
                    "estimate": float(fit.params[term]),
                    "standard_error_hc3": float(fit.bse[term]),
                    "ci95_lower": float(fit.conf_int().loc[term, 0]),
                    "ci95_upper": float(fit.conf_int().loc[term, 1]),
                    "p_value_hc3": float(fit.pvalues[term]),
                }
            )
    nested_rows = []
    for reduced_name, full_name in [
        ("covariates", "conventional_eeg"),
        ("conventional_eeg", "conventional_plus_aperiodic"),
        ("conventional_plus_aperiodic", "conventional_plus_aperiodic_plus_fisher"),
    ]:
        reduced = fits[reduced_name]
        full = fits[full_name]
        df_difference = int(full.df_model - reduced.df_model)
        rss_difference = float(np.sum(reduced.resid**2) - np.sum(full.resid**2))
        f_statistic = max(rss_difference / max(df_difference, 1), 0.0) / max(
            float(np.sum(full.resid**2) / full.df_resid), np.finfo(float).eps
        )
        nested_rows.append(
            {
                "reduced_model": reduced_name,
                "full_model": full_name,
                "df_difference": df_difference,
                "f_statistic": float(f_statistic),
                "p_value": float(fisher_f.sf(f_statistic, df_difference, full.df_resid)),
                "delta_r_squared": float(full.rsquared - reduced.rsquared),
                "partial_r_squared": float(rss_difference / max(np.sum(reduced.resid**2), np.finfo(float).eps)),
            }
        )
    return fits, pd.DataFrame(model_rows), pd.DataFrame(coefficient_rows), pd.DataFrame(nested_rows)


def _pipeline(table: pd.DataFrame, continuous_terms: list[str]) -> Pipeline:
    numeric = ["age_years", *continuous_terms]
    if "education" in table:
        numeric.append("education")
    preprocessor = ColumnTransformer(
        [
            ("numeric", StandardScaler(), numeric),
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["sex", "dataset"]),
        ],
        remainder="drop",
    )
    return Pipeline([("preprocess", preprocessor), ("model", LinearRegression())])


def _cross_validated_performance(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_features = {name: list(block) for name, block in MODEL_BLOCKS.items()}
    metrics = []
    stratification = table["dataset"].to_numpy()
    y = table["moca"].to_numpy(float)
    for repeat in range(30):
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=20260907 + repeat)
        for model_name, feature_names in model_features.items():
            predictors = ["age_years", "sex", "dataset", *[FEATURES[name] for name in feature_names]]
            if "education" in table:
                predictors.append("education")
            predictions = np.full(len(table), np.nan)
            for train, test in splitter.split(table, stratification):
                fitted = _pipeline(table, [FEATURES[name] for name in feature_names])
                fitted.fit(table.iloc[train][predictors], y[train])
                predictions[test] = fitted.predict(table.iloc[test][predictors])
            metrics.append(
                {
                    "repeat": repeat + 1,
                    "model": model_name,
                    "r_squared": float(r2_score(y, predictions)),
                    "rmse": float(np.sqrt(mean_squared_error(y, predictions))),
                    "mae": float(np.mean(np.abs(y - predictions))),
                }
            )
    metrics_frame = pd.DataFrame(metrics)
    summary_rows = []
    for model, frame in metrics_frame.groupby("model", sort=False):
        row = {"model": model}
        for metric in ("r_squared", "rmse", "mae"):
            values = frame[metric].to_numpy(float)
            row.update(
                {
                    f"{metric}_mean": float(np.mean(values)),
                    f"{metric}_std": float(np.std(values, ddof=1)),
                    f"{metric}_ci025": float(np.percentile(values, 2.5)),
                    f"{metric}_ci975": float(np.percentile(values, 97.5)),
                }
            )
        summary_rows.append(row)
    return metrics_frame, pd.DataFrame(summary_rows)


def _plot_results(
    model_summary: pd.DataFrame,
    coefficients: pd.DataFrame,
    cv_summary: pd.DataFrame,
    nested: pd.DataFrame,
    output: Path,
) -> None:
    model_order = [
        "covariates",
        "conventional_eeg",
        "conventional_plus_aperiodic",
        "conventional_plus_aperiodic_plus_fisher",
    ]
    labels = ["Covariates", "Conventional EEG", "+ aperiodic", "+ Theta Fisher"]
    colors = ["#999999", "#4daf4a", "#e69f00", "#2166ac"]
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.8))

    in_sample = model_summary.set_index("model").loc[model_order]
    axes[0].bar(labels, in_sample["r_squared"], color=colors)
    axes[0].set_ylabel("In-sample R²")
    axes[0].set_title("Nested explanatory models")
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].grid(axis="y", alpha=0.2)
    for index, value in enumerate(in_sample["r_squared"]):
        axes[0].text(index, value, f"{value:.3f}", ha="center", va="bottom", fontsize=9)

    cv = cv_summary.set_index("model").loc[model_order]
    x = np.arange(len(labels))
    axes[1].errorbar(
        x,
        cv["r_squared_mean"],
        yerr=[cv["r_squared_mean"] - cv["r_squared_ci025"], cv["r_squared_ci975"] - cv["r_squared_mean"]],
        fmt="o",
        color="#2166ac",
        capsize=4,
    )
    axes[1].set_xticks(x, labels, rotation=25)
    axes[1].set_ylabel("Repeated 5-fold CV R²")
    axes[1].set_title("Out-of-sample performance")
    axes[1].axhline(0, color="#333333", linewidth=0.8)
    axes[1].grid(axis="y", alpha=0.2)

    fisher_row = coefficients.loc[
        (coefficients["model"] == "conventional_plus_aperiodic_plus_fisher")
        & (coefficients["term"] == "theta_fisher_z")
    ]
    if fisher_row.empty:
        axes[2].axis("off")
    else:
        row = fisher_row.iloc[0]
        estimate = row["estimate"]
        axes[2].errorbar(
            estimate,
            0,
            xerr=[[estimate - row["ci95_lower"]], [row["ci95_upper"] - estimate]],
            fmt="o",
            color="#2166ac",
            capsize=4,
        )
        axes[2].axvline(0, color="#333333", linewidth=0.8)
        axes[2].set_yticks([0], ["Theta Fisher"])
        axes[2].set_xlabel("Standardized coefficient for MoCA\n(HC3 95% CI)")
        axes[2].set_title("Incremental Fisher coefficient")
        axes[2].grid(axis="x", alpha=0.2)
    fisher_nested = nested.loc[
        nested["full_model"] == "conventional_plus_aperiodic_plus_fisher"
    ].iloc[0]
    fig.suptitle(
        "Theta Fisher information and MoCA: incremental-value analysis\n"
        f"PD-only, n={int(in_sample.iloc[0]['n'])}; ΔR² after Fisher = {fisher_nested['delta_r_squared']:.3f}, p={fisher_nested['p_value']:.3g}",
        y=1.04,
        fontsize=14,
    )
    fig.text(0.5, -0.02, "Education was unavailable in the saved outputs; dataset is modeled as a fixed effect.", ha="center", fontsize=9, color="#555555")
    fig.tight_layout(rect=(0, 0.04, 1, 0.91), w_pad=2.0)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/global"))
    args = parser.parse_args()
    root = args.output_root
    output_dir = root / "statistics" / "theta_fisher_moca_incremental"
    figure = root / "figures" / "summary" / "theta_fisher_moca_incremental_value.png"
    table, notes = _load_analysis_table(root)
    _, model_summary, coefficients, nested = _fit_nested_models(
        table, notes["education_column"]
    )
    cv_metrics, cv_summary = _cross_validated_performance(table)
    output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_dir / "analysis_dataset.csv.gz", index=False, compression="gzip")
    model_summary.to_csv(output_dir / "model_summary.csv", index=False)
    coefficients.to_csv(output_dir / "coefficients.csv", index=False)
    nested.to_csv(output_dir / "nested_model_comparisons.csv", index=False)
    cv_metrics.to_csv(output_dir / "cross_validated_metrics.csv", index=False)
    cv_summary.to_csv(output_dir / "cross_validated_summary.csv", index=False)
    (output_dir / "analysis_notes.json").write_text(json.dumps(notes, indent=2, default=str) + "\n")
    _plot_results(model_summary, coefficients, cv_summary, nested, figure)
    fisher_row = coefficients.loc[
        (coefficients["model"] == "conventional_plus_aperiodic_plus_fisher")
        & (coefficients["term"] == "theta_fisher_z")
    ].iloc[0]
    fisher_nested = nested.loc[
        nested["full_model"] == "conventional_plus_aperiodic_plus_fisher"
    ].iloc[0]
    print(f"Complete PD MoCA cases: {len(table)}")
    print(f"Theta Fisher standardized coefficient: {fisher_row['estimate']:.4f} (p={fisher_row['p_value_hc3']:.4g})")
    print(f"Incremental R2: {fisher_nested['delta_r_squared']:.4f} (p={fisher_nested['p_value']:.4g})")
    print(f"Wrote results to {output_dir}")
    print(f"Wrote figure to {figure}")


if __name__ == "__main__":
    main()
