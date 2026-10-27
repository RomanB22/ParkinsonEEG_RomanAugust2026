#!/usr/bin/env python3
"""Train leakage-safe XGBoost models across the three large EEG datasets.

Two related analyses are run from the subject-level global metric tables:

* PD versus Control classification across ``primary``, ``ds007526-1.0.2``,
  and ``ds008768-1.0.0``;
* MoCA prediction among PD participants with an available MoCA score.

The analysis deliberately keeps dataset out of the predictors.  Dataset is
used for stratification and a leave-one-dataset-out sensitivity analysis so a
model cannot obtain a high score merely by learning cohort membership.  The
outer cross-validation predictions are the primary performance outputs;
feature screening and hyperparameter tuning occur only inside training folds.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, KFold, RepeatedStratifiedKFold, StratifiedKFold
from xgboost import XGBClassifier, XGBRegressor


DATASETS = ["primary", "ds007526-1.0.2", "ds008768-1.0.0"]
BANDS = ["delta", "theta", "alpha", "beta", "gamma"]
BOUT_BANDS = ["theta", "alpha", "beta", "gamma"]
D4_METRICS = ["entropy", "complexity", "fisher_information", "weighted_permutation_entropy"]
BOUT_METRICS = [
    "n_bouts",
    "oscillatory_occupancy",
    "bouts_per_minute",
    "duration_mean_s",
    "amplitude_mean",
    "cycles_mean",
]

CLASSIFIER_TARGET = "target_pd"
REGRESSION_TARGET = "moca"
SEED = 20260908


def _read_subject(root: Path, dataset: str) -> pd.DataFrame:
    path = root / "metrics" / dataset / "subject_features.csv.gz"
    table = pd.read_csv(path, low_memory=False)
    metadata = pd.DataFrame(
        {"dataset": [dataset] * len(table), "dataset_label": [dataset] * len(table)},
        index=table.index,
    )
    return pd.concat([table, metadata], axis=1)


def _read_subject_sex(root: Path, dataset: str) -> pd.DataFrame:
    path = root / "metrics" / dataset / "recording_features.csv.gz"
    recording = pd.read_csv(path, usecols=["participant_id", "sex"], low_memory=False)
    recording["sex"] = recording["sex"].astype("string").str.strip().replace({"": pd.NA})
    conflicts = recording.dropna(subset=["sex"]).groupby("participant_id")["sex"].nunique()
    if (conflicts > 1).any():
        raise ValueError(f"{dataset}: participant has conflicting sex values")
    recording = recording.dropna(subset=["sex"]).drop_duplicates("participant_id")
    recording["sex_male"] = recording["sex"].eq("M").astype(float)
    return recording[["participant_id", "sex_male"]]


def _add_derived_features(table: pd.DataFrame) -> pd.DataFrame:
    table = table.copy()
    ratios = {
        "alpha_theta_ratio": ("psd__alpha__relative_power", "psd__theta__relative_power"),
        "theta_beta_ratio": ("psd__theta__relative_power", "psd__beta__relative_power"),
        "alpha_beta_ratio": ("psd__alpha__relative_power", "psd__beta__relative_power"),
    }
    for name, (numerator, denominator) in ratios.items():
        table[name] = table[numerator] / table[denominator].replace(0, np.nan)
    return table.replace([np.inf, -np.inf], np.nan)


def _load_table(root: Path) -> pd.DataFrame:
    frames = []
    for dataset in DATASETS:
        frame = _read_subject(root, dataset)
        sex = _read_subject_sex(root, dataset)
        frame = frame.merge(sex, on="participant_id", how="left", validate="one_to_one")
        frames.append(frame)
    table = _add_derived_features(pd.concat(frames, ignore_index=True))
    table["group"] = table["group"].astype(str)
    table["target_pd"] = table["group"].eq("PD").astype(int)
    table["moca"] = pd.to_numeric(table["moca"], errors="coerce")
    table["age_years"] = pd.to_numeric(table["age_years"], errors="coerce")
    return table


def _feature_candidates(table: pd.DataFrame) -> list[str]:
    """Return a prespecified, harmonized EEG candidate set.

    The selection is based on feature semantics and availability only; it does
    not inspect the outcome.  This keeps the feature definition independent of
    every outer validation split.
    """
    candidates: list[str] = []
    candidates.extend(f"psd__{band}__relative_power" for band in BANDS)
    candidates.extend(
        f"entropy__{band}__{metric}__D4" for band in BANDS for metric in D4_METRICS
    )
    candidates.extend(
        f"within_bout__{band}__{metric}__D4"
        for band in BOUT_BANDS
        for metric in D4_METRICS
    )
    candidates.extend(
        f"bout__{band}__{metric}" for band in BOUT_BANDS for metric in BOUT_METRICS
    )
    candidates.extend(
        [
            "aperiodic__broadband__aperiodic_offset",
            "aperiodic__broadband__aperiodic_exponent",
            "alpha_theta_ratio",
            "theta_beta_ratio",
            "alpha_beta_ratio",
        ]
    )
    available = []
    for feature in candidates:
        if feature not in table.columns:
            continue
        values = pd.to_numeric(table[feature], errors="coerce")
        if values.notna().sum() < 10 or values.nunique(dropna=True) < 3:
            continue
        if values.notna().mean() < 0.85:
            continue
        available.append(feature)
    if not available:
        raise ValueError("No harmonized EEG candidate features were available")
    return available


def _feature_blocks(candidates: list[str]) -> dict[str, list[str]]:
    def present(names: Iterable[str]) -> list[str]:
        return [name for name in names if name in candidates]

    core = [
        "psd__delta__relative_power",
        "psd__theta__relative_power",
        "psd__alpha__relative_power",
        "psd__beta__relative_power",
        "entropy__theta__entropy__D4",
        "entropy__theta__complexity__D4",
        "entropy__theta__fisher_information__D4",
        "entropy__alpha__entropy__D4",
        "entropy__alpha__complexity__D4",
        "entropy__alpha__fisher_information__D4",
        "within_bout__theta__entropy__D4",
        "within_bout__theta__complexity__D4",
        "within_bout__theta__fisher_information__D4",
        "within_bout__alpha__entropy__D4",
        "within_bout__alpha__complexity__D4",
        "within_bout__alpha__fisher_information__D4",
        *[f"bout__{band}__{metric}" for band in BOUT_BANDS for metric in ["oscillatory_occupancy", "bouts_per_minute", "duration_mean_s"]],
        "aperiodic__broadband__aperiodic_offset",
        "aperiodic__broadband__aperiodic_exponent",
        "alpha_theta_ratio",
        "theta_beta_ratio",
    ]
    core_features = present(core)
    expanded_features = list(candidates)
    demographics = ["age_years", "sex_male"]
    return {
        "demographics": demographics,
        "core": core_features,
        "core_plus_demographics": core_features + demographics,
        "expanded": expanded_features,
        "expanded_plus_demographics": expanded_features + demographics,
    }


def _screen_features(table: pd.DataFrame, candidates: list[str]) -> pd.DataFrame:
    """Create descriptive group-effect and PD-only MoCA association results."""
    rows: list[dict[str, Any]] = []
    for feature in candidates:
        values = pd.to_numeric(table[feature], errors="coerce")
        pd_values = values[table["target_pd"].eq(1)]
        control_values = values[table["target_pd"].eq(0)]
        pooled_sd = np.sqrt(
            ((pd_values.count() - 1) * pd_values.var() + (control_values.count() - 1) * control_values.var())
            / max(pd_values.count() + control_values.count() - 2, 1)
        )
        effect = (pd_values.mean() - control_values.mean()) / pooled_sd if pooled_sd > 0 else np.nan
        pd_table = table.loc[table["target_pd"].eq(1), [feature, "moca"]].dropna()
        rho = spearmanr(pd_table[feature], pd_table["moca"]).statistic if len(pd_table) >= 10 else np.nan
        rows.append(
            {
                "feature": feature,
                "n_total": int(values.notna().sum()),
                "missing_fraction": float(values.isna().mean()),
                "control_mean": float(control_values.mean()),
                "pd_mean": float(pd_values.mean()),
                "pd_minus_control_smd": float(effect),
                "pd_moca_n": int(len(pd_table)),
                "pd_moca_spearman_rho": float(rho) if np.isfinite(rho) else np.nan,
            }
        )
    result = pd.DataFrame(rows)
    result["abs_group_effect"] = result["pd_minus_control_smd"].abs()
    result["abs_pd_moca_rho"] = result["pd_moca_spearman_rho"].abs()
    return result.sort_values(["abs_group_effect", "abs_pd_moca_rho"], ascending=False).drop(
        columns=["abs_group_effect", "abs_pd_moca_rho"]
    )


CLASSIFIER_GRID = {
    "max_depth": [2, 3],
    "learning_rate": [0.03, 0.08],
    "n_estimators": [150, 300],
    "min_child_weight": [3, 8],
    "subsample": [0.8],
    "colsample_bytree": [0.8],
    "reg_lambda": [5.0],
}

REGRESSOR_GRID = {
    "max_depth": [1, 2],
    "learning_rate": [0.03, 0.08],
    "n_estimators": [100, 250],
    "min_child_weight": [3, 8],
    "subsample": [0.8],
    "colsample_bytree": [0.8],
    "reg_lambda": [10.0],
}


def _classifier(seed: int) -> XGBClassifier:
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=1,
        random_state=int(seed),
        verbosity=0,
    )


def _regressor(seed: int) -> XGBRegressor:
    return XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        tree_method="hist",
        n_jobs=1,
        random_state=int(seed),
        verbosity=0,
    )


def _classification_metrics(truth: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    predicted = (probability >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(truth, predicted, labels=[0, 1]).ravel()
    return {
        "roc_auc": float(roc_auc_score(truth, probability)),
        "average_precision": float(average_precision_score(truth, probability)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predicted)),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else np.nan,
        "specificity": float(tn / (tn + fp)) if tn + fp else np.nan,
        "brier_score": float(brier_score_loss(truth, probability)),
        "log_loss": float(log_loss(truth, probability, labels=[0, 1])),
    }


def _regression_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    pearson = pearsonr(truth, prediction).statistic if len(truth) > 2 else np.nan
    spearman = spearmanr(truth, prediction).statistic if len(truth) > 2 else np.nan
    return {
        "r_squared": float(r2_score(truth, prediction)),
        "rmse": float(np.sqrt(mean_squared_error(truth, prediction))),
        "mae": float(mean_absolute_error(truth, prediction)),
        "pearson_r": float(pearson) if np.isfinite(pearson) else np.nan,
        "spearman_rho": float(spearman) if np.isfinite(spearman) else np.nan,
    }


def _summary(metrics: pd.DataFrame, metric_names: list[str], group_columns: list[str]) -> pd.DataFrame:
    rows = []
    for keys, frame in metrics.groupby(group_columns, sort=False, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_columns, keys))
        for metric in metric_names:
            values = frame[metric].dropna().to_numpy(float)
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            row[f"{metric}_ci025"] = float(np.percentile(values, 2.5))
            row[f"{metric}_ci975"] = float(np.percentile(values, 97.5))
        rows.append(row)
    return pd.DataFrame(rows)


def _inner_cv(y: np.ndarray, stratification: np.ndarray, folds: int, seed: int) -> Any:
    if len(np.unique(stratification)) > 1 and min(np.bincount(pd.Categorical(stratification).codes)) >= folds:
        return StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed).split(
            np.zeros(len(y)), stratification
        )
    return KFold(n_splits=folds, shuffle=True, random_state=seed).split(np.zeros(len(y)))


def _fit_search(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    kind: str,
    seed: int,
    inner_folds: int,
    stratification: np.ndarray,
) -> GridSearchCV:
    if kind == "classification":
        estimator = _classifier(seed)
        grid = CLASSIFIER_GRID
        scoring = "roc_auc"
    else:
        estimator = _regressor(seed)
        grid = REGRESSOR_GRID
        scoring = "neg_root_mean_squared_error"
    cv = list(_inner_cv(y_train, stratification, inner_folds, seed))
    search = GridSearchCV(estimator, grid, scoring=scoring, cv=cv, n_jobs=1, refit=True)
    search.fit(x_train, y_train)
    return search


def _run_cv(
    table: pd.DataFrame,
    blocks: dict[str, list[str]],
    kind: str,
    outer_folds: int,
    outer_repeats: int,
    inner_folds: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if kind == "classification":
        analysis = table.dropna(subset=["target_pd"]).copy()
        y = analysis["target_pd"].to_numpy(int)
        outer = list(
            RepeatedStratifiedKFold(
                n_splits=outer_folds, n_repeats=outer_repeats, random_state=seed
            ).split(analysis, y)
        )
    else:
        analysis = table.loc[table["target_pd"].eq(1)].dropna(subset=["moca"]).copy()
        y = analysis["moca"].to_numpy(float)
        stratification = analysis["dataset"].to_numpy(str)
        outer = []
        for repeat in range(outer_repeats):
            splitter = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=seed + repeat)
            outer.extend(splitter.split(analysis, stratification))

    prediction_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    for model_name, features in blocks.items():
        x = analysis[features]
        for split_index, (train_indices, test_indices) in enumerate(outer):
            repeat = split_index // outer_folds + 1
            fold = split_index % outer_folds + 1
            split_seed = seed + 1000 * list(blocks).index(model_name) + split_index
            if kind == "classification":
                inner_stratification = analysis.iloc[train_indices]["dataset"].astype(str).to_numpy()
                # Preserve class balance in tuning; dataset remains a useful blocking variable.
                inner_stratification = np.array(
                    [f"{d}__{label}" for d, label in zip(inner_stratification, y[train_indices])]
                )
                if min(pd.Series(inner_stratification).value_counts()) < inner_folds:
                    inner_stratification = y[train_indices]
            else:
                inner_stratification = analysis.iloc[train_indices]["dataset"].astype(str).to_numpy()
            search = _fit_search(
                x.iloc[train_indices],
                y[train_indices],
                kind,
                split_seed,
                inner_folds,
                inner_stratification,
            )
            if kind == "classification":
                prediction = search.predict_proba(x.iloc[test_indices])[:, 1]
                row_metrics = _classification_metrics(y[test_indices], prediction)
            else:
                prediction = search.predict(x.iloc[test_indices])
                row_metrics = _regression_metrics(y[test_indices], prediction)
            metric_rows.append(
                {
                    "model": model_name,
                    "repeat": repeat,
                    "fold": fold,
                    "n_test": len(test_indices),
                    "best_params": json.dumps(search.best_params_, sort_keys=True),
                    **row_metrics,
                }
            )
            for local_index, table_index in enumerate(test_indices):
                row = {
                    "model": model_name,
                    "repeat": repeat,
                    "fold": fold,
                    "participant_id": analysis.iloc[table_index]["participant_id"],
                    "dataset": analysis.iloc[table_index]["dataset"],
                    "group": analysis.iloc[table_index]["group"],
                    "truth": float(y[table_index]),
                    "prediction": float(prediction[local_index]),
                }
                prediction_rows.append(row)
    metrics = pd.DataFrame(metric_rows)
    predictions = pd.DataFrame(prediction_rows)
    metric_names = (
        ["roc_auc", "average_precision", "balanced_accuracy", "sensitivity", "specificity", "brier_score", "log_loss"]
        if kind == "classification"
        else ["r_squared", "rmse", "mae", "pearson_r", "spearman_rho"]
    )
    summary = _summary(metrics, metric_names, ["model"])
    return predictions, metrics, summary


def _run_leave_one_dataset_out(
    table: pd.DataFrame,
    blocks: dict[str, list[str]],
    kind: str,
    inner_folds: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prediction_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    for model_index, (model_name, features) in enumerate(blocks.items()):
        for dataset_index, held_out in enumerate(DATASETS):
            if kind == "classification":
                analysis = table.copy()
                train = analysis[analysis["dataset"] != held_out]
                test = analysis[analysis["dataset"] == held_out]
                y_train = train["target_pd"].to_numpy(int)
                y_test = test["target_pd"].to_numpy(int)
                inner_stratification = np.array(
                    [f"{d}__{label}" for d, label in zip(train["dataset"], y_train)]
                )
                if min(pd.Series(inner_stratification).value_counts()) < inner_folds:
                    inner_stratification = y_train
            else:
                analysis = table.loc[table["target_pd"].eq(1) & table["moca"].notna()].copy()
                train = analysis[analysis["dataset"] != held_out]
                test = analysis[analysis["dataset"] == held_out]
                y_train = train["moca"].to_numpy(float)
                y_test = test["moca"].to_numpy(float)
                inner_stratification = train["dataset"].to_numpy(str)
            if len(np.unique(y_test)) < 2 and kind == "classification":
                raise ValueError(f"{held_out} has only one outcome class in the holdout set")
            search = _fit_search(
                train[features],
                y_train,
                kind,
                seed + 5000 + 100 * model_index + dataset_index,
                inner_folds,
                inner_stratification,
            )
            if kind == "classification":
                prediction = search.predict_proba(test[features])[:, 1]
                row_metrics = _classification_metrics(y_test, prediction)
            else:
                prediction = search.predict(test[features])
                row_metrics = _regression_metrics(y_test, prediction)
            metric_rows.append(
                {
                    "model": model_name,
                    "held_out_dataset": held_out,
                    "n_train": len(train),
                    "n_test": len(test),
                    "best_params": json.dumps(search.best_params_, sort_keys=True),
                    **row_metrics,
                }
            )
            for local_index, (_, source_row) in enumerate(test.iterrows()):
                prediction_rows.append(
                    {
                        "model": model_name,
                        "held_out_dataset": held_out,
                        "participant_id": source_row["participant_id"],
                        "truth": float(y_test[local_index]),
                        "prediction": float(prediction[local_index]),
                    }
                )
    return pd.DataFrame(prediction_rows), pd.DataFrame(metric_rows)


def _final_models(
    table: pd.DataFrame,
    blocks: dict[str, list[str]],
    kind: str,
    metrics: pd.DataFrame,
    inner_folds: int,
    seed: int,
    output_dir: Path,
) -> pd.DataFrame:
    rows = []
    for model_index, (model_name, features) in enumerate(blocks.items()):
        if kind == "classification":
            analysis = table
            y = analysis["target_pd"].to_numpy(int)
            stratification = np.array([f"{d}__{label}" for d, label in zip(analysis["dataset"], y)])
            if min(pd.Series(stratification).value_counts()) < inner_folds:
                stratification = y
        else:
            analysis = table.loc[table["target_pd"].eq(1) & table["moca"].notna()]
            y = analysis["moca"].to_numpy(float)
            stratification = analysis["dataset"].to_numpy(str)
        search = _fit_search(
            analysis[features], y, kind, seed + 7000 + model_index, inner_folds, stratification
        )
        model_path = output_dir / f"{kind}_{model_name}.json"
        search.best_estimator_.save_model(model_path)
        importance = pd.DataFrame(
            {
                "model": model_name,
                "feature": features,
                "gain": search.best_estimator_.feature_importances_,
                "kind": kind,
            }
        ).sort_values("gain", ascending=False)
        importance.to_csv(output_dir / f"{kind}_{model_name}_feature_importance.csv", index=False)
        best_params = json.dumps(search.best_params_, sort_keys=True)
        rows.append(
            {
                "model": model_name,
                "kind": kind,
                "n": len(analysis),
                "n_features": len(features),
                "best_params_full_data": best_params,
                "mean_outer_primary_metric": float(
                    metrics.loc[metrics["model"].eq(model_name), "roc_auc" if kind == "classification" else "rmse"].mean()
                ),
                "saved_model": str(model_path),
            }
        )
    return pd.DataFrame(rows)


def _plot_cv(summary: pd.DataFrame, kind: str, output: Path) -> None:
    metric = "roc_auc" if kind == "classification" else "rmse"
    direction = 1 if kind == "classification" else -1
    ordered = summary.sort_values(f"{metric}_mean", ascending=direction > 0)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    positions = np.arange(len(ordered))
    means = ordered[f"{metric}_mean"].to_numpy(float)
    lower = means - ordered[f"{metric}_ci025"].to_numpy(float)
    upper = ordered[f"{metric}_ci975"].to_numpy(float) - means
    ax.errorbar(positions, means, yerr=[lower, upper], fmt="o", color="#2166ac", capsize=4)
    ax.set_xticks(positions, ordered["model"], rotation=20, ha="right")
    ax.set_ylabel("ROC AUC" if kind == "classification" else "RMSE (MoCA points)")
    ax.set_title("Repeated outer cross-validation")
    ax.grid(axis="y", alpha=0.2)
    if kind == "classification":
        ax.set_ylim(0.45, 1.0)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/global"))
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--outer-repeats", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    root = args.output_root
    output_dir = root / "statistics" / "xgboost_multidataset"
    output_dir.mkdir(parents=True, exist_ok=True)
    table = _load_table(root)
    candidates = _feature_candidates(table)
    blocks = _feature_blocks(candidates)
    blocks = {name: features for name, features in blocks.items() if features}
    screen = _screen_features(table, candidates)

    classification_predictions, classification_metrics, classification_summary = _run_cv(
        table, blocks, "classification", args.outer_folds, args.outer_repeats, args.inner_folds, args.seed
    )
    classification_loso_predictions, classification_loso_metrics = _run_leave_one_dataset_out(
        table, blocks, "classification", args.inner_folds, args.seed
    )
    regression_predictions, regression_metrics, regression_summary = _run_cv(
        table, blocks, "regression", args.outer_folds, args.outer_repeats, args.inner_folds, args.seed + 100
    )
    regression_loso_predictions, regression_loso_metrics = _run_leave_one_dataset_out(
        table, blocks, "regression", args.inner_folds, args.seed + 100
    )
    classification_loso_summary = _summary(
        classification_loso_metrics,
        ["roc_auc", "average_precision", "balanced_accuracy", "sensitivity", "specificity", "brier_score", "log_loss"],
        ["model"],
    )
    regression_loso_summary = _summary(
        regression_loso_metrics,
        ["r_squared", "rmse", "mae", "pearson_r", "spearman_rho"],
        ["model"],
    )

    final_classification = _final_models(
        table, blocks, "classification", classification_metrics, args.inner_folds, args.seed, output_dir
    )
    final_regression = _final_models(
        table, blocks, "regression", regression_metrics, args.inner_folds, args.seed + 100, output_dir
    )

    table.to_csv(output_dir / "analysis_table.csv.gz", index=False, compression="gzip")
    screen.to_csv(output_dir / "feature_screening.csv", index=False)
    classification_predictions.to_csv(output_dir / "classification_outer_predictions.csv.gz", index=False, compression="gzip")
    classification_metrics.to_csv(output_dir / "classification_outer_metrics.csv", index=False)
    classification_summary.to_csv(output_dir / "classification_summary.csv", index=False)
    classification_loso_predictions.to_csv(output_dir / "classification_loso_predictions.csv.gz", index=False, compression="gzip")
    classification_loso_metrics.to_csv(output_dir / "classification_loso_metrics.csv", index=False)
    classification_loso_summary.to_csv(output_dir / "classification_loso_summary.csv", index=False)
    regression_predictions.to_csv(output_dir / "moca_outer_predictions.csv.gz", index=False, compression="gzip")
    regression_metrics.to_csv(output_dir / "moca_outer_metrics.csv", index=False)
    regression_summary.to_csv(output_dir / "moca_summary.csv", index=False)
    regression_loso_predictions.to_csv(output_dir / "moca_loso_predictions.csv.gz", index=False, compression="gzip")
    regression_loso_metrics.to_csv(output_dir / "moca_loso_metrics.csv", index=False)
    regression_loso_summary.to_csv(output_dir / "moca_loso_summary.csv", index=False)
    final_classification.to_csv(output_dir / "final_classification_models.csv", index=False)
    final_regression.to_csv(output_dir / "final_moca_models.csv", index=False)
    _plot_cv(classification_summary, "classification", root / "figures" / "summary" / "xgboost_classification_cv.png")
    _plot_cv(regression_summary, "regression", root / "figures" / "summary" / "xgboost_moca_cv.png")

    notes = {
        "datasets": DATASETS,
        "classification_population": table["group"].value_counts().astype(int).to_dict(),
        "moca_population": int(table["target_pd"].eq(1).mul(table["moca"].notna()).sum()),
        "candidate_feature_count": len(candidates),
        "feature_blocks": {name: features for name, features in blocks.items()},
        "excluded_from_predictors": [
            "dataset (used only for stratification and held-out-cohort validation)",
            "group/target_pd",
            "moca (classification target and unavailable in many Controls)",
            "UPDRS/MMSE/medication state/identifiers",
        ],
        "validation": {
            "outer_folds": args.outer_folds,
            "outer_repeats": args.outer_repeats,
            "inner_folds": args.inner_folds,
            "random_seed": args.seed,
            "classification_primary_metric": "ROC AUC",
            "moca_primary_metric": "RMSE",
        },
    }
    (output_dir / "analysis_notes.json").write_text(json.dumps(notes, indent=2) + "\n")

    best_classifier = classification_summary.sort_values("roc_auc_mean", ascending=False).iloc[0]
    best_regressor = regression_summary.sort_values("rmse_mean", ascending=True).iloc[0]
    print(f"Classification population: {len(table)} ({table.target_pd.sum()} PD, {(1-table.target_pd).sum()} Control)")
    print(f"Best repeated-CV classifier: {best_classifier['model']} ROC AUC={best_classifier['roc_auc_mean']:.3f}")
    print(f"MoCA population: {int(table.target_pd.eq(1).mul(table.moca.notna()).sum())} PD participants")
    print(f"Best repeated-CV MoCA model: {best_regressor['model']} RMSE={best_regressor['rmse_mean']:.3f}, R2={best_regressor['r_squared_mean']:.3f}")
    print(f"Wrote XGBoost analysis to {output_dir}")


if __name__ == "__main__":
    main()
