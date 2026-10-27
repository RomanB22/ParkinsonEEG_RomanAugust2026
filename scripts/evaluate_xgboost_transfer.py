#!/usr/bin/env python3
"""Evaluate the saved XGBoost models on an independent smaller dataset.

The existing multidataset XGBoost workflow trains on the three datasets with
MoCA information (``primary``, ``ds007526-1.0.2``, and ``ds008768-1.0.0``).
This script applies the saved models to a separate dataset without refitting
or using its labels.  The default target is ``medication_state`` (the
ds002778 cohort), restricted to Controls and PD_ON recordings so it matches
the medication-on state of the source studies.

The classification result is reported at both recording and participant
level.  Participant aggregation is important here because the medication
cohort contains paired ON/OFF recordings for the same PD participant.

MoCA and MMSE are reported as a compatibility check rather than silently
treated as interchangeable outcomes.  The source datasets contain MoCA but
not MMSE, while the default target contains MMSE but not MoCA; therefore no
valid external cognitive-score transfer can be computed for that target.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
from scipy.stats import pearsonr, spearmanr
from xgboost import XGBClassifier, XGBRegressor


SOURCE_DATASETS = ["primary", "ds007526-1.0.2", "ds008768-1.0.0"]
DEFAULT_TARGET_DATASET = "medication_state"
SEED = 20260908


def _add_derived_features(table: pd.DataFrame) -> pd.DataFrame:
    table = table.copy()
    ratios = {
        "alpha_theta_ratio": (
            "psd__alpha__relative_power",
            "psd__theta__relative_power",
        ),
        "theta_beta_ratio": (
            "psd__theta__relative_power",
            "psd__beta__relative_power",
        ),
        "alpha_beta_ratio": (
            "psd__alpha__relative_power",
            "psd__beta__relative_power",
        ),
    }
    for name, (numerator, denominator) in ratios.items():
        if numerator in table.columns and denominator in table.columns:
            table[name] = table[numerator] / table[denominator].replace(0, np.nan)
    return table.replace([np.inf, -np.inf], np.nan)


def _read_subject(root: Path, dataset: str) -> pd.DataFrame:
    path = root / "metrics" / dataset / "subject_features.csv.gz"
    if not path.exists():
        raise FileNotFoundError(f"Missing subject feature table: {path}")
    table = pd.read_csv(path, low_memory=False)
    table["group"] = table["group"].astype("string").str.strip()
    table["age_years"] = pd.to_numeric(table["age_years"], errors="coerce")
    table["moca"] = pd.to_numeric(table["moca"], errors="coerce")
    table["mmse"] = pd.to_numeric(table["mmse"], errors="coerce")
    table = pd.concat(
        [
            table,
            pd.DataFrame(
                {
                    "dataset": dataset,
                    "target_pd": table["group"].str.upper().str.startswith("PD").astype(int),
                },
                index=table.index,
            ),
        ],
        axis=1,
    )
    return _add_derived_features(table)


def _load_tables(root: Path, datasets: list[str]) -> pd.DataFrame:
    return pd.concat([_read_subject(root, dataset) for dataset in datasets], ignore_index=True)


def _load_minimal_model(root: Path, kind: str) -> tuple[Any, list[str], Path]:
    model_dir = root / "statistics" / "xgboost_minimal_features"
    final_models = pd.read_csv(model_dir / "minimal_final_models.csv")
    row = final_models.loc[final_models["kind"].eq(kind)]
    if len(row) != 1:
        raise ValueError(f"Expected one saved minimal {kind} model, found {len(row)}")
    row = row.iloc[0]
    model_path = Path(str(row["model_path"]))
    if not model_path.is_absolute():
        candidates = [Path.cwd() / model_path, root.parent.parent / model_path]
        model_path = next((candidate for candidate in candidates if candidate.exists()), model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Missing saved XGBoost model: {model_path}")
    features = json.loads(row["features"])
    model = XGBClassifier() if kind == "classification" else XGBRegressor()
    model.load_model(model_path)
    if int(model.n_features_in_) != len(features):
        raise ValueError(
            f"Saved {kind} model expects {model.n_features_in_} features, but metadata lists {len(features)}"
        )
    return model, [str(feature) for feature in features], model_path


def _validate_features(table: pd.DataFrame, features: list[str], label: str) -> None:
    missing = sorted(set(features) - set(table.columns))
    if missing:
        raise ValueError(f"{label} is missing model features: {missing}")


def _classification_metrics(truth: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    predicted = (probability >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(truth, predicted, labels=[0, 1]).ravel()
    accuracy = float(np.mean(predicted == truth))
    metrics = {
        "n": int(len(truth)),
        "n_pd": int(np.sum(truth == 1)),
        "n_control": int(np.sum(truth == 0)),
        "roc_auc": float(roc_auc_score(truth, probability)),
        "average_precision": float(average_precision_score(truth, probability)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predicted)),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else np.nan,
        "specificity": float(tn / (tn + fp)) if tn + fp else np.nan,
        "brier_score": float(brier_score_loss(truth, probability)),
        "log_loss": float(log_loss(truth, probability, labels=[0, 1])),
        "n_correct_at_0_5": int(np.sum(predicted == truth)),
    }
    metrics["accuracy"] = accuracy
    for name in [
        "roc_auc",
        "average_precision",
        "balanced_accuracy",
        "sensitivity",
        "specificity",
        "accuracy",
    ]:
        metrics[f"{name}_percent"] = metrics[name] * 100.0
    return metrics


def _regression_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    pearson = pearsonr(truth, prediction).statistic if len(truth) > 2 else np.nan
    spearman = spearmanr(truth, prediction).statistic if len(truth) > 2 else np.nan
    baseline = np.full_like(truth, np.mean(truth), dtype=float)
    return {
        "n": int(len(truth)),
        "rmse": float(np.sqrt(mean_squared_error(truth, prediction))),
        "mae": float(mean_absolute_error(truth, prediction)),
        "r_squared": float(r2_score(truth, prediction)),
        "pearson_r": float(pearson) if np.isfinite(pearson) else np.nan,
        "spearman_rho": float(spearman) if np.isfinite(spearman) else np.nan,
        "mean_truth": float(np.mean(truth)),
        "mean_prediction": float(np.mean(prediction)),
        "mean_prediction_minus_truth": float(np.mean(prediction - truth)),
        "baseline_mean_rmse": float(np.sqrt(mean_squared_error(truth, baseline))),
        "baseline_mean_mae": float(mean_absolute_error(truth, baseline)),
    }


def _run_classification_transfer(
    source: pd.DataFrame,
    target: pd.DataFrame,
    model: XGBClassifier,
    features: list[str],
    target_dataset: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    _validate_features(source, features, "Source table")
    _validate_features(target, features, "Target table")
    probabilities = model.predict_proba(target[features])[:, 1]
    predictions = target[
        ["dataset", "participant_id", "group", "target_pd", "medication_state"]
    ].copy() if "medication_state" in target.columns else target[
        ["dataset", "participant_id", "group", "target_pd"]
    ].copy()
    predictions["probability_pd"] = probabilities
    predictions["predicted_pd_at_0_5"] = (probabilities >= 0.5).astype(int)
    predictions["prediction_correct_at_0_5"] = (
        predictions["predicted_pd_at_0_5"] == predictions["target_pd"]
    )

    participant = (
        predictions.groupby(["dataset", "participant_id"], as_index=False)
        .agg(
            group=("group", "first"),
            target_pd=("target_pd", "first"),
            probability_pd=("probability_pd", "mean"),
            n_recordings=("probability_pd", "size"),
        )
    )
    participant["predicted_pd_at_0_5"] = (participant["probability_pd"] >= 0.5).astype(int)
    participant["prediction_correct_at_0_5"] = (
        participant["predicted_pd_at_0_5"] == participant["target_pd"]
    )

    metrics = []
    metrics.append({"unit": "recording", **_classification_metrics(
        predictions["target_pd"].to_numpy(int), predictions["probability_pd"].to_numpy(float)
    )})
    metrics.append({"unit": "participant_mean_probability", **_classification_metrics(
        participant["target_pd"].to_numpy(int), participant["probability_pd"].to_numpy(float)
    )})
    metrics_table = pd.DataFrame(metrics)
    metadata = {
        "source_datasets": sorted(source["dataset"].unique().tolist()),
        "target_dataset": target_dataset,
        "n_source_recordings": int(len(source)),
        "n_target_recordings": int(len(target)),
        "n_target_participants": int(target["participant_id"].nunique()),
        "target_group_counts_recording": target["target_pd"].value_counts().astype(int).to_dict(),
        "target_group_counts_participant": participant["target_pd"].value_counts().astype(int).to_dict(),
    }
    return predictions, participant, {"metrics": metrics_table, "metadata": metadata}


def _run_cognitive_transfer(
    target: pd.DataFrame,
    model: XGBRegressor,
    features: list[str],
    target_outcome: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    analysis = target.loc[target["group"].eq("PD_ON") & target[target_outcome].notna()].copy()
    if analysis.empty:
        raise ValueError(f"No PD_ON rows with {target_outcome} are available for transfer")
    _validate_features(analysis, features, "Cognitive target table")
    prediction = model.predict(analysis[features])
    predictions = analysis[["dataset", "participant_id", "group", target_outcome]].copy()
    predictions = predictions.rename(columns={target_outcome: "truth_mmse"})
    predictions["prediction_from_moca_model"] = prediction
    predictions["residual_prediction_minus_mmse"] = prediction - predictions["truth_mmse"]
    metrics = pd.DataFrame(
        [
            {
                "source_model_outcome": "moca",
                "target_outcome": target_outcome,
                "interpretation": "zero-shot numeric transfer; MoCA and MMSE are not interchangeable scales",
                **_regression_metrics(
                    predictions["truth_mmse"].to_numpy(float),
                    predictions["prediction_from_moca_model"].to_numpy(float),
                ),
            }
        ]
    )
    return predictions, metrics


def _cognitive_compatibility(
    source: pd.DataFrame, target: pd.DataFrame, target_dataset: str
) -> pd.DataFrame:
    rows = []
    for outcome in ["moca", "mmse"]:
        for source_dataset, source_table in source.groupby("dataset", sort=False):
            source_n = int(source_table[outcome].notna().sum())
            target_n = int(target[outcome].notna().sum())
            if source_n == 0:
                status = "not_trainable_in_source"
            elif target_n == 0:
                status = "not_evaluable_target_missing_outcome"
            elif outcome != "moca":
                status = "target_outcome_available_but_no_saved_source_model"
            else:
                status = "externally_evaluable"
            rows.append(
                {
                    "outcome": outcome,
                    "source_dataset": source_dataset,
                    "target_dataset": target_dataset,
                    "source_n_recordings": source_n,
                    "target_n_recordings": target_n,
                    "source_mean": float(source_table[outcome].mean()) if source_n else np.nan,
                    "target_mean": float(target[outcome].mean()) if target_n else np.nan,
                    "status": status,
                }
            )
    return pd.DataFrame(rows)


def _source_benchmark(root: Path, n_features: int) -> dict[str, float]:
    path = root / "statistics" / "xgboost_minimal_features" / "minimal_feature_summary.csv"
    if not path.exists():
        return {}
    summary = pd.read_csv(path)
    row = summary.loc[
        summary["kind"].eq("classification") & summary["k"].eq(n_features)
    ]
    if len(row) != 1:
        return {}
    row = row.iloc[0]
    return {
        "n_features": int(n_features),
        "nested_cv_roc_auc_mean": float(row["roc_auc_mean"]),
        "nested_cv_balanced_accuracy_mean": float(row["balanced_accuracy_mean"]),
        "nested_cv_sensitivity_mean": float(row["sensitivity_mean"]),
        "nested_cv_specificity_mean": float(row["specificity_mean"]),
    }


def _plot_transfer(predictions: pd.DataFrame, participant: pd.DataFrame, output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    colors = {0: "#0072B2", 1: "#D55E00"}
    for group, frame in predictions.groupby("target_pd", sort=True):
        axes[0].scatter(
            np.arange(len(frame)),
            frame["probability_pd"],
            s=38,
            alpha=0.75,
            color=colors[int(group)],
            label="PD" if group else "Control",
        )
    axes[0].axhline(0.5, color="#333333", linestyle="--", linewidth=1)
    axes[0].set_xlabel("Target recordings (arbitrary order)")
    axes[0].set_ylabel("Predicted probability of PD")
    axes[0].set_title("Recording-level transfer")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.2)

    for group, frame in participant.groupby("target_pd", sort=True):
        axes[1].scatter(
            np.arange(len(frame)),
            frame["probability_pd"],
            s=48,
            alpha=0.8,
            color=colors[int(group)],
            label="PD" if group else "Control",
        )
    axes[1].axhline(0.5, color="#333333", linestyle="--", linewidth=1)
    axes[1].set_xlabel("Target participants (arbitrary order)")
    axes[1].set_ylabel("Mean predicted probability of PD")
    axes[1].set_title("Participant-level transfer")
    axes[1].grid(alpha=0.2)
    figure.suptitle("External XGBoost PD/control transfer")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _plot_confusion_matrix(participant: pd.DataFrame, output: Path) -> None:
    truth = participant["target_pd"].to_numpy(int)
    predicted = participant["predicted_pd_at_0_5"].to_numpy(int)
    matrix = confusion_matrix(truth, predicted, labels=[0, 1])
    row_percent = matrix / matrix.sum(axis=1, keepdims=True) * 100.0
    figure, axis = plt.subplots(figsize=(6.6, 5.4))
    image = axis.imshow(matrix, cmap="Blues", vmin=0)
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="Participants")
    axis.set_xticks([0, 1], ["Predicted Control", "Predicted PD"])
    axis.set_yticks([0, 1], ["True Control", "True PD"])
    axis.set_xlabel("Model prediction at probability threshold 0.5")
    axis.set_ylabel("Observed diagnosis")
    for row in range(2):
        for column in range(2):
            axis.text(
                column,
                row,
                f"{matrix[row, column]}\n({row_percent[row, column]:.1f}%)",
                ha="center",
                va="center",
                color="white" if matrix[row, column] > matrix.max() / 2 else "black",
                fontsize=13,
                fontweight="bold",
            )
    axis.set_title("PD/control transfer in medication-on participants\nParticipant-level confusion matrix")
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/global"))
    parser.add_argument("--target-dataset", default=DEFAULT_TARGET_DATASET)
    parser.add_argument(
        "--target-groups",
        default="Control,PD_ON",
        help="Comma-separated target group labels. Default keeps Controls and PD_ON only.",
    )
    parser.add_argument(
        "--source-datasets",
        default=",".join(SOURCE_DATASETS),
        help="Comma-separated source dataset IDs used to train the saved model.",
    )
    args = parser.parse_args()

    root = args.output_root
    source_datasets = [value.strip() for value in args.source_datasets.split(",") if value.strip()]
    if args.target_dataset in source_datasets:
        raise ValueError("The target dataset must not also be listed as a source dataset")

    output_dir = root / "statistics" / "xgboost_transfer"
    output_dir.mkdir(parents=True, exist_ok=True)
    source = _load_tables(root, source_datasets)
    target_all = _load_tables(root, [args.target_dataset])
    target_groups = [value.strip() for value in args.target_groups.split(",") if value.strip()]
    target = target_all.loc[target_all["group"].isin(target_groups)].copy()
    if target.empty:
        raise ValueError(f"No target rows remain after --target-groups={target_groups}")
    model, features, model_path = _load_minimal_model(root, "classification")
    recording, participant, result = _run_classification_transfer(
        source, target, model, features, args.target_dataset
    )
    compatibility = _cognitive_compatibility(source, target, args.target_dataset)
    source_benchmark = _source_benchmark(root, len(features))
    cognitive_model, cognitive_features, cognitive_model_path = _load_minimal_model(root, "regression")
    cognitive_predictions, cognitive_metrics = _run_cognitive_transfer(
        target, cognitive_model, cognitive_features, "mmse"
    )

    recording.to_csv(output_dir / "classification_transfer_recordings.csv.gz", index=False, compression="gzip")
    participant.to_csv(output_dir / "classification_transfer_participants.csv", index=False)
    result["metrics"].to_csv(output_dir / "classification_transfer_metrics.csv", index=False)
    compatibility.to_csv(output_dir / "cognitive_transfer_compatibility.csv", index=False)
    cognitive_predictions.to_csv(output_dir / "moca_model_to_mmse_predictions.csv", index=False)
    cognitive_metrics.to_csv(output_dir / "moca_model_to_mmse_metrics.csv", index=False)
    _plot_transfer(
        recording,
        participant,
        root / "figures" / "summary" / "xgboost_transfer_pd_control.png",
    )
    _plot_confusion_matrix(
        participant,
        root / "figures" / "summary" / "xgboost_transfer_confusion_matrix.png",
    )

    notes = {
        "source_datasets": source_datasets,
        "target_dataset": args.target_dataset,
        "target_groups": target_groups,
        "classifier_model": str(model_path),
        "classifier_features": features,
        "cognitive_model": str(cognitive_model_path),
        "cognitive_features": cognitive_features,
        "target_pd_mapping": "group values beginning with PD (PD, PD_ON, PD_OFF) map to 1; Control maps to 0",
        "participant_aggregation": "mean recording-level PD probability within participant",
        "cognitive_outcome_rule": "MoCA and MMSE are distinct scales; an external transfer requires the same outcome to be observed in source and target",
        "source_nested_cv_benchmark": source_benchmark,
        **result["metadata"],
    }
    (output_dir / "analysis_notes.json").write_text(json.dumps(notes, indent=2) + "\n")

    print(f"Source datasets: {', '.join(source_datasets)} ({len(source)} recordings)")
    print(f"Target dataset: {args.target_dataset} ({len(target)} recordings; {target['participant_id'].nunique()} participants)")
    print(result["metrics"].to_string(index=False))
    if source_benchmark:
        participant_auc = float(
            result["metrics"].loc[
                result["metrics"]["unit"].eq("participant_mean_probability"), "roc_auc"
            ].iloc[0]
        )
        print(
            f"\nSource nested-CV ROC AUC: {source_benchmark['nested_cv_roc_auc_mean']:.3f}; "
            f"target participant-level ROC AUC: {participant_auc:.3f}; "
            f"difference: {participant_auc - source_benchmark['nested_cv_roc_auc_mean']:+.3f}"
        )
    print("\nMoCA-model to PD_ON MMSE zero-shot transfer:")
    print(cognitive_metrics.to_string(index=False))
    print("\nCognitive compatibility:")
    print(compatibility[["outcome", "source_dataset", "source_n_recordings", "target_n_recordings", "status"]].to_string(index=False))
    print(f"\nWrote transfer outputs to {output_dir}")


if __name__ == "__main__":
    main()
