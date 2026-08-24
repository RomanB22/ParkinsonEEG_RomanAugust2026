"""Leakage-safe nested validation for transparent logistic models."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_predict,
    permutation_test_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve


METRICS: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "roc_auc": roc_auc_score,
    "average_precision": average_precision_score,
    "brier_score": brier_score_loss,
    "log_loss": lambda truth, probability: log_loss(
        truth, probability, labels=[0, 1]
    ),
}


def _pipeline(seed: int, c_value: float = 1.0) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "logistic",
                LogisticRegression(
                    C=float(c_value),
                    l1_ratio=0.0,
                    solver="lbfgs",
                    max_iter=5000,
                    random_state=int(seed),
                ),
            ),
        ]
    )


def _classification_metrics(
    truth: np.ndarray,
    probability: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    predicted = (probability >= float(threshold)).astype(int)
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


def _training_balanced_threshold(
    truth: np.ndarray,
    probability: np.ndarray,
) -> float:
    """Choose the training-only threshold maximizing Youden's J statistic."""
    false_positive, true_positive, thresholds = roc_curve(truth, probability)
    finite = np.isfinite(thresholds) & (thresholds > 0.0) & (thresholds < 1.0)
    if not finite.any():
        return 0.5
    candidate_thresholds = thresholds[finite]
    candidate_scores = (true_positive - false_positive)[finite]
    best_score = np.max(candidate_scores)
    tied = candidate_thresholds[np.isclose(candidate_scores, best_score)]
    return float(tied[np.argmin(np.abs(tied - 0.5))])


def _coefficient_rows(
    fitted: Pipeline,
    model_name: str,
    features: list[str],
    **identifiers: Any,
) -> list[dict[str, Any]]:
    scaler = fitted.named_steps["scale"]
    logistic = fitted.named_steps["logistic"]
    standardized = logistic.coef_[0]
    native = standardized / scaler.scale_
    rows = []
    for index, feature in enumerate(features):
        standard_beta = float(standardized[index])
        native_beta = float(native[index])
        rows.append(
            {
                "model": model_name,
                "feature": feature,
                "coefficient_per_sd": standard_beta,
                "odds_ratio_per_sd": float(np.exp(np.clip(standard_beta, -700, 700))),
                "coefficient_native_unit": native_beta,
                "odds_ratio_native_unit": float(np.exp(np.clip(native_beta, -700, 700))),
                "training_mean": float(scaler.mean_[index]),
                "training_scale": float(scaler.scale_[index]),
                **identifiers,
            }
        )
    return rows


def run_nested_validation(
    feature_table: pd.DataFrame,
    models: dict[str, dict[str, Any]],
    validation: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run repeated nested CV and return predictions, metrics, and coefficients."""
    y = feature_table["target_pd"].to_numpy(dtype=int)
    subject_ids = feature_table["subject_id"].astype(str).to_numpy()
    n_splits = int(validation["outer_folds"])
    n_repeats = int(validation["outer_repeats"])
    inner_folds = int(validation["inner_folds"])
    fixed_threshold = float(validation["classification_threshold"])
    threshold_policy = str(validation.get("threshold_policy", "fixed"))
    seed = int(validation["random_seed"])
    c_grid = [float(value) for value in validation["c_grid"]]
    if min(np.bincount(y)) < n_splits:
        raise ValueError("Each outcome class must contain at least outer_folds subjects")

    prediction_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    coefficient_rows: list[dict[str, Any]] = []
    for model_index, (model_name, specification) in enumerate(models.items()):
        features = [str(value) for value in specification["features"]]
        x = feature_table[features].to_numpy(dtype=float)
        outer = RepeatedStratifiedKFold(
            n_splits=n_splits,
            n_repeats=n_repeats,
            random_state=seed,
        )
        for split_index, (train_indices, test_indices) in enumerate(outer.split(x, y)):
            repeat = split_index // n_splits
            fold = split_index % n_splits
            split_seed = seed + 1000 * model_index + split_index
            inner = StratifiedKFold(
                n_splits=inner_folds,
                shuffle=True,
                random_state=split_seed,
            )
            search = GridSearchCV(
                _pipeline(split_seed),
                {"logistic__C": c_grid},
                scoring=str(validation["primary_metric"]),
                cv=inner,
                n_jobs=1,
                refit=True,
            )
            search.fit(x[train_indices], y[train_indices])
            if threshold_policy == "inner_youden":
                training_probability = cross_val_predict(
                    clone(search.best_estimator_),
                    x[train_indices],
                    y[train_indices],
                    cv=inner,
                    method="predict_proba",
                    n_jobs=1,
                )[:, 1]
                threshold = _training_balanced_threshold(
                    y[train_indices], training_probability
                )
            elif threshold_policy == "fixed":
                threshold = fixed_threshold
            else:
                raise ValueError(f"Unknown threshold policy: {threshold_policy}")
            probability = search.predict_proba(x[test_indices])[:, 1]
            predicted = (probability >= threshold).astype(int)
            best_c = float(search.best_params_["logistic__C"])
            for row_index, subject_index in enumerate(test_indices):
                prediction_rows.append(
                    {
                        "model": model_name,
                        "model_label": specification["label"],
                        "model_role": specification["role"],
                        "repeat": int(repeat),
                        "fold": int(fold),
                        "subject_id": subject_ids[subject_index],
                        "target_pd": int(y[subject_index]),
                        "predicted_probability_pd": float(probability[row_index]),
                        "predicted_class_pd": int(predicted[row_index]),
                        "classification_threshold": threshold,
                        "best_c": best_c,
                    }
                )
            fold_metrics = _classification_metrics(
                y[test_indices], probability, threshold
            )
            for metric, value in fold_metrics.items():
                metric_rows.append(
                    {
                        "model": model_name,
                        "model_label": specification["label"],
                        "model_role": specification["role"],
                        "repeat": int(repeat),
                        "fold": int(fold),
                        "metric": metric,
                        "value": value,
                        "n_test_subjects": int(len(test_indices)),
                        "best_c": best_c,
                    }
                )
            coefficient_rows.extend(
                _coefficient_rows(
                    search.best_estimator_,
                    model_name,
                    features,
                    model_label=specification["label"],
                    model_role=specification["role"],
                    repeat=int(repeat),
                    fold=int(fold),
                    best_c=best_c,
                )
            )
    return (
        pd.DataFrame.from_records(prediction_rows),
        pd.DataFrame.from_records(metric_rows),
        pd.DataFrame.from_records(coefficient_rows),
    )


def average_repeated_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """Average repeated out-of-fold probabilities to one prediction per subject/model."""
    result = (
        predictions.groupby(
            ["model", "model_label", "model_role", "subject_id", "target_pd"],
            sort=False,
            as_index=False,
        )
        .agg(
            predicted_probability_pd=("predicted_probability_pd", "mean"),
            prediction_sd=("predicted_probability_pd", "std"),
            n_outer_predictions=("predicted_probability_pd", "size"),
            classification_threshold=("classification_threshold", "mean"),
        )
    )
    result["predicted_class_pd"] = (
        result["predicted_probability_pd"] >= result["classification_threshold"]
    ).astype(int)
    return result


def bootstrap_performance(
    averaged_predictions: pd.DataFrame,
    *,
    n_resamples: int,
    seed: int,
) -> pd.DataFrame:
    """Calculate subject-stratified bootstrap intervals for model performance."""
    rows = []
    rng = np.random.default_rng(int(seed))
    for model_name, selected in averaged_predictions.groupby("model", sort=False):
        truth = selected["target_pd"].to_numpy(dtype=int)
        probability = selected["predicted_probability_pd"].to_numpy(dtype=float)
        threshold = float(selected["classification_threshold"].iloc[0])
        point = _classification_metrics(truth, probability, threshold)
        class_indices = [np.flatnonzero(truth == outcome) for outcome in (0, 1)]
        bootstrap_values = {metric: [] for metric in point}
        for _ in range(int(n_resamples)):
            indices = np.concatenate(
                [rng.choice(values, size=len(values), replace=True) for values in class_indices]
            )
            values = _classification_metrics(
                truth[indices], probability[indices], threshold
            )
            for metric, value in values.items():
                bootstrap_values[metric].append(value)
        for metric, estimate in point.items():
            distribution = np.asarray(bootstrap_values[metric], dtype=float)
            rows.append(
                {
                    "model": model_name,
                    "model_label": selected["model_label"].iloc[0],
                    "model_role": selected["model_role"].iloc[0],
                    "metric": metric,
                    "estimate": estimate,
                    "ci_lower": float(np.nanquantile(distribution, 0.025)),
                    "ci_upper": float(np.nanquantile(distribution, 0.975)),
                    "n_subjects": int(len(selected)),
                    "bootstrap_resamples": int(n_resamples),
                }
            )
    return pd.DataFrame.from_records(rows)


def bootstrap_auc_differences(
    averaged_predictions: pd.DataFrame,
    *,
    reference_model: str,
    n_resamples: int,
    seed: int,
) -> pd.DataFrame:
    """Return paired subject-bootstrap AUC differences versus a reference model."""
    wide = averaged_predictions.pivot(
        index=["subject_id", "target_pd"],
        columns="model",
        values="predicted_probability_pd",
    ).reset_index()
    if reference_model not in wide:
        raise ValueError(f"Reference model is unavailable: {reference_model}")
    truth = wide["target_pd"].to_numpy(dtype=int)
    class_indices = [np.flatnonzero(truth == outcome) for outcome in (0, 1)]
    reference = wide[reference_model].to_numpy(dtype=float)
    reference_auc = float(roc_auc_score(truth, reference))
    rng = np.random.default_rng(int(seed))
    rows = []
    for model_name in sorted(set(wide.columns) - {"subject_id", "target_pd", reference_model}):
        candidate = wide[model_name].to_numpy(dtype=float)
        observed = float(roc_auc_score(truth, candidate) - reference_auc)
        differences = []
        for _ in range(int(n_resamples)):
            indices = np.concatenate(
                [rng.choice(values, size=len(values), replace=True) for values in class_indices]
            )
            differences.append(
                roc_auc_score(truth[indices], candidate[indices])
                - roc_auc_score(truth[indices], reference[indices])
            )
        rows.append(
            {
                "model": model_name,
                "reference_model": reference_model,
                "auc_difference": observed,
                "ci_lower": float(np.quantile(differences, 0.025)),
                "ci_upper": float(np.quantile(differences, 0.975)),
                "bootstrap_resamples": int(n_resamples),
            }
        )
    return pd.DataFrame.from_records(rows)


def fit_final_models(
    feature_table: pd.DataFrame,
    models: dict[str, dict[str, Any]],
    validation: dict[str, Any],
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Tune on all subjects, save descriptive final models, and run permutation tests."""
    output_dir.mkdir(parents=True, exist_ok=True)
    y = feature_table["target_pd"].to_numpy(dtype=int)
    seed = int(validation["random_seed"])
    inner_folds = int(validation["inner_folds"])
    c_grid = [float(value) for value in validation["c_grid"]]
    n_permutations = int(validation["permutation_resamples"])
    coefficient_rows = []
    permutation_rows = []
    for model_index, (model_name, specification) in enumerate(models.items()):
        features = [str(value) for value in specification["features"]]
        x = feature_table[features].to_numpy(dtype=float)
        model_seed = seed + 10000 + model_index
        inner = StratifiedKFold(
            n_splits=inner_folds,
            shuffle=True,
            random_state=model_seed,
        )
        search = GridSearchCV(
            _pipeline(model_seed),
            {"logistic__C": c_grid},
            scoring=str(validation["primary_metric"]),
            cv=inner,
            n_jobs=1,
            refit=True,
        )
        search.fit(x, y)
        best_c = float(search.best_params_["logistic__C"])
        fitted = search.best_estimator_
        if str(validation.get("threshold_policy", "fixed")) == "inner_youden":
            training_probability = cross_val_predict(
                clone(fitted),
                x,
                y,
                cv=inner,
                method="predict_proba",
                n_jobs=1,
            )[:, 1]
            final_threshold = _training_balanced_threshold(y, training_probability)
        else:
            final_threshold = float(validation["classification_threshold"])
        joblib.dump(fitted, output_dir / f"{model_name}.joblib")
        coefficient_rows.extend(
            _coefficient_rows(
                fitted,
                model_name,
                features,
                model_label=specification["label"],
                model_role=specification["role"],
                best_c=best_c,
                decision_threshold_full_training=final_threshold,
                full_data_fit=True,
            )
        )
        if n_permutations > 0:
            permutation_cv = StratifiedKFold(
                n_splits=int(validation["outer_folds"]),
                shuffle=True,
                random_state=model_seed,
            )
            score, permutation_scores, p_value = permutation_test_score(
                _pipeline(model_seed, best_c),
                x,
                y,
                scoring=str(validation["primary_metric"]),
                cv=permutation_cv,
                n_permutations=n_permutations,
                random_state=model_seed,
                n_jobs=1,
            )
            permutation_rows.append(
                {
                    "model": model_name,
                    "model_label": specification["label"],
                    "observed_cv_score": float(score),
                    "permutation_mean": float(np.mean(permutation_scores)),
                    "permutation_std": float(np.std(permutation_scores, ddof=1)),
                    "permutation_p": float(p_value),
                    "n_permutations": n_permutations,
                    "best_c_full_data": best_c,
                    "decision_threshold_full_training": final_threshold,
                }
            )
    return (
        pd.DataFrame.from_records(coefficient_rows),
        pd.DataFrame.from_records(permutation_rows),
    )
