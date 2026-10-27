#!/usr/bin/env python3
"""Find compact XGBoost feature sets and make model interpretation figures.

Feature rankings are recomputed inside every outer training fold.  The
resulting feature-count curves therefore provide an honest comparison of
compact models against the expanded model, while the final saved compact
models are refit on all available participants for downstream use.
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
from matplotlib.colors import ListedColormap
from sklearn.base import clone
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold

import train_xgboost_multidataset as xgb_workflow


DEFAULT_K = [2, 3, 4, 5, 6, 8, 10, 12, 16, 24, 32, 48, 72]
MODEL_NAME = "expanded_plus_demographics"


def _outer_analysis_and_splits(
    table: pd.DataFrame,
    kind: str,
    outer_folds: int,
    outer_repeats: int,
    seed: int,
) -> tuple[pd.DataFrame, np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    if kind == "classification":
        analysis = table.copy()
        y = analysis["target_pd"].to_numpy(int)
        splits = list(
            RepeatedStratifiedKFold(
                n_splits=outer_folds,
                n_repeats=outer_repeats,
                random_state=seed,
            ).split(analysis, y)
        )
    else:
        analysis = table.loc[table["target_pd"].eq(1) & table["moca"].notna()].copy()
        y = analysis["moca"].to_numpy(float)
        splits = []
        dataset_labels = analysis["dataset"].astype(str).to_numpy()
        for repeat in range(outer_repeats):
            splitter = StratifiedKFold(
                n_splits=outer_folds,
                shuffle=True,
                random_state=seed + repeat,
            )
            splits.extend(splitter.split(analysis, dataset_labels))
    return analysis, y, splits


def _inner_stratification(
    analysis: pd.DataFrame,
    train_indices: np.ndarray,
    y: np.ndarray,
    kind: str,
    inner_folds: int,
) -> np.ndarray:
    if kind == "regression":
        labels = analysis.iloc[train_indices]["dataset"].astype(str).to_numpy()
    else:
        datasets = analysis.iloc[train_indices]["dataset"].astype(str).to_numpy()
        labels = np.array([f"{dataset}__{label}" for dataset, label in zip(datasets, y[train_indices])])
        if min(pd.Series(labels).value_counts()) < inner_folds:
            labels = y[train_indices]
    return labels


def _feature_count_rows(k_values: list[int], n_features: int) -> list[int]:
    return sorted({int(k) for k in k_values if 1 <= int(k) <= n_features})


def _run_minimal_curve(
    table: pd.DataFrame,
    features: list[str],
    kind: str,
    k_values: list[int],
    outer_folds: int,
    outer_repeats: int,
    inner_folds: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    analysis, y, splits = _outer_analysis_and_splits(
        table, kind, outer_folds, outer_repeats, seed
    )
    k_values = _feature_count_rows(k_values, len(features))
    metric_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    ranking_rows: list[dict[str, object]] = []
    for split_index, (train_indices, test_indices) in enumerate(splits):
        repeat = split_index // outer_folds + 1
        fold = split_index % outer_folds + 1
        split_seed = seed + split_index
        inner_labels = _inner_stratification(
            analysis, train_indices, y, kind, inner_folds
        )
        search = xgb_workflow._fit_search(
            analysis.iloc[train_indices][features],
            y[train_indices],
            kind,
            split_seed,
            inner_folds,
            inner_labels,
        )
        importance = pd.Series(
            search.best_estimator_.feature_importances_, index=features
        ).sort_values(ascending=False, kind="stable")
        for rank, (feature, value) in enumerate(importance.items(), start=1):
            ranking_rows.append(
                {
                    "kind": kind,
                    "repeat": repeat,
                    "fold": fold,
                    "rank": rank,
                    "feature": feature,
                    "importance": float(value),
                }
            )
        for k in k_values:
            selected = importance.index[:k].tolist()
            estimator = clone(search.best_estimator_)
            estimator.fit(
                analysis.iloc[train_indices][selected], y[train_indices]
            )
            if kind == "classification":
                prediction = estimator.predict_proba(
                    analysis.iloc[test_indices][selected]
                )[:, 1]
                row_metrics = xgb_workflow._classification_metrics(
                    y[test_indices], prediction
                )
            else:
                prediction = estimator.predict(
                    analysis.iloc[test_indices][selected]
                )
                row_metrics = xgb_workflow._regression_metrics(
                    y[test_indices], prediction
                )
            metric_rows.append(
                {
                    "kind": kind,
                    "k": k,
                    "repeat": repeat,
                    "fold": fold,
                    "n_test": len(test_indices),
                    **row_metrics,
                }
            )
            for local_index, source_index in enumerate(test_indices):
                source = analysis.iloc[source_index]
                prediction_rows.append(
                    {
                        "kind": kind,
                        "k": k,
                        "repeat": repeat,
                        "fold": fold,
                        "participant_id": source["participant_id"],
                        "dataset": source["dataset"],
                        "group": source["group"],
                        "truth": float(y[source_index]),
                        "prediction": float(prediction[local_index]),
                    }
                )
    metrics = pd.DataFrame(metric_rows)
    predictions = pd.DataFrame(prediction_rows)
    if kind == "classification":
        metric_names = [
            "roc_auc",
            "average_precision",
            "balanced_accuracy",
            "sensitivity",
            "specificity",
            "brier_score",
            "log_loss",
        ]
    else:
        metric_names = ["r_squared", "rmse", "mae", "pearson_r", "spearman_rho"]
    summary = xgb_workflow._summary(metrics, metric_names, ["kind", "k"])
    return metrics, predictions, summary, pd.DataFrame(ranking_rows)


def _category(feature: str) -> str:
    if feature in {"age_years", "sex_male"}:
        return "demographics"
    if feature.startswith("psd__"):
        return "relative power"
    if feature.startswith("entropy__"):
        return "entropy / Fisher"
    if feature.startswith("within_bout__"):
        return "within-bout"
    if feature.startswith("bout__"):
        return "bout dynamics"
    if feature.startswith("aperiodic__"):
        return "aperiodic"
    return "ratios"


def _short_label(feature: str) -> str:
    replacements = {
        "psd__": "",
        "entropy__": "H/F ",
        "within_bout__": "WB ",
        "bout__": "B ",
        "__relative_power": " relP",
        "__fisher_information__D4": " F D4",
        "__weighted_permutation_entropy__D4": " WPE D4",
        "__complexity__D4": " C D4",
        "__entropy__D4": " H D4",
        "__oscillatory_occupancy": " occupancy",
        "__bouts_per_minute": " /min",
        "__duration_mean_s": " duration",
        "__amplitude_mean": " amplitude",
        "__cycles_mean": " cycles",
        "__n_bouts": " count",
        "__": " ",
        "aperiodic__broadband__": "",
        "aperiodic_": "",
        "_": " ",
    }
    label = feature
    for old, new in replacements.items():
        label = label.replace(old, new)
    return label.strip()


def _plot_feature_sets(blocks: dict[str, list[str]], output: Path) -> None:
    feature_order = blocks["expanded_plus_demographics"]
    model_order = list(blocks)
    matrix = np.array(
        [[int(feature in blocks[model]) for feature in feature_order] for model in model_order]
    )
    fig, ax = plt.subplots(figsize=(24, 5.8))
    ax.imshow(matrix, aspect="auto", interpolation="none", cmap=ListedColormap(["#f1f1f1", "#2166ac"]))
    ax.set_yticks(np.arange(len(model_order)), model_order)
    ax.set_xticks(np.arange(len(feature_order)), [_short_label(f) for f in feature_order], rotation=75, ha="right", fontsize=7)
    ax.set_ylabel("Model feature set")
    ax.set_title("XGBoost model feature sets\nBlue = included feature")
    ax.set_xlim(-0.5, len(feature_order) - 0.5)
    ax.set_ylim(len(model_order) - 0.5, -0.5)
    ax.grid(which="major", axis="y", color="white", linewidth=0.8)
    for index, feature in enumerate(feature_order):
        ax.get_xticklabels()[index].set_color(
            {"demographics": "#7f3c8d", "relative power": "#1b9e77", "entropy / Fisher": "#d95f02", "within-bout": "#7570b3", "bout dynamics": "#e7298a", "aperiodic": "#66a61e", "ratios": "#a6761d"}[_category(feature)]
        )
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_minimal_curve(summary: pd.DataFrame, selection: dict[str, dict[str, float]], output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.7))
    class_summary = summary.loc[summary["kind"].eq("classification")]
    axes[0].errorbar(
        class_summary["k"], class_summary["roc_auc_mean"],
        yerr=[class_summary["roc_auc_mean"] - class_summary["roc_auc_ci025"], class_summary["roc_auc_ci975"] - class_summary["roc_auc_mean"]],
        fmt="o-", color="#2166ac", capsize=3,
    )
    class_k = int(selection["classification"]["selected_k"])
    axes[0].axvline(class_k, color="#d55e00", linestyle="--")
    axes[0].set_xlabel("Number of features")
    axes[0].set_ylabel("ROC AUC")
    axes[0].set_title(f"PD/Control classifier\nminimal k={class_k}")
    axes[0].set_ylim(0.45, 1.0)
    axes[0].grid(alpha=0.2)

    regression_summary = summary.loc[summary["kind"].eq("regression")]
    axes[1].errorbar(
        regression_summary["k"], regression_summary["rmse_mean"],
        yerr=[regression_summary["rmse_mean"] - regression_summary["rmse_ci025"], regression_summary["rmse_ci975"] - regression_summary["rmse_mean"]],
        fmt="o-", color="#009e73", capsize=3,
    )
    moca_k = int(selection["regression"]["selected_k"])
    axes[1].axvline(moca_k, color="#d55e00", linestyle="--")
    axes[1].set_xlabel("Number of features")
    axes[1].set_ylabel("RMSE (MoCA points)")
    axes[1].set_title(f"MoCA predictor\nminimal k={moca_k}")
    axes[1].grid(alpha=0.2)

    axes[2].errorbar(
        regression_summary["k"], regression_summary["r_squared_mean"],
        yerr=[regression_summary["r_squared_mean"] - regression_summary["r_squared_ci025"], regression_summary["r_squared_ci975"] - regression_summary["r_squared_mean"]],
        fmt="o-", color="#e69f00", capsize=3,
    )
    axes[2].axvline(moca_k, color="#d55e00", linestyle="--")
    axes[2].set_xlabel("Number of features")
    axes[2].set_ylabel("R²")
    axes[2].set_title("MoCA explained variance")
    axes[2].axhline(0, color="#333333", linewidth=0.8)
    axes[2].grid(alpha=0.2)
    fig.suptitle("Compact-feature performance curves\nError bars show the 2.5–97.5 percentile across outer folds", y=1.03)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_moca_predictions(predictions: pd.DataFrame, selected_k: int, summary: pd.DataFrame, output: Path) -> None:
    selected = predictions.loc[(predictions["kind"].eq("regression")) & (predictions["k"].eq(selected_k))].copy()
    averaged = selected.groupby(["participant_id", "dataset"], as_index=False).agg(
        truth=("truth", "first"), prediction=("prediction", "mean")
    )
    colors = {"primary": "#0072B2", "ds007526-1.0.2": "#D55E00", "ds008768-1.0.0": "#009E73"}
    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    for dataset, group in averaged.groupby("dataset", sort=False):
        ax.scatter(group["truth"], group["prediction"], s=34, alpha=0.72, label=dataset, color=colors.get(dataset, "#555555"))
    limits = [float(min(averaged["truth"].min(), averaged["prediction"].min()) - 0.5), float(max(averaged["truth"].max(), averaged["prediction"].max()) + 0.5)]
    ax.plot(limits, limits, color="#333333", linestyle="--", linewidth=1.0, label="identity")
    metrics = xgb_workflow._regression_metrics(averaged["truth"].to_numpy(), averaged["prediction"].to_numpy())
    fold_row = summary.loc[(summary["kind"].eq("regression")) & (summary["k"].eq(selected_k))].iloc[0]
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_xlabel("Observed MoCA")
    ax.set_ylabel("Predicted MoCA")
    ax.set_title(f"Held-out MoCA predictions\n{selected_k} features; fold-mean RMSE={fold_row['rmse_mean']:.2f}, R²={fold_row['r_squared_mean']:.2f}\nParticipant-averaged points: RMSE={metrics['rmse']:.2f}, R²={metrics['r_squared']:.2f}")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _select_minimal(summary: pd.DataFrame) -> dict[str, dict[str, float]]:
    selected: dict[str, dict[str, float]] = {}
    class_summary = summary.loc[summary["kind"].eq("classification")].sort_values("k")
    full_class = class_summary.iloc[-1]
    class_target = float(full_class["roc_auc_mean"] * 0.98)
    class_eligible = class_summary.loc[class_summary["roc_auc_mean"] >= class_target]
    selected_class = class_eligible.iloc[0] if not class_eligible.empty else full_class
    selected["classification"] = {
        "selected_k": int(selected_class["k"]),
        "reference_roc_auc": float(full_class["roc_auc_mean"]),
        "target_roc_auc": class_target,
        "selected_roc_auc": float(selected_class["roc_auc_mean"]),
        "rule": "smallest k with mean ROC AUC at least 98% of the expanded-plus-demographics reference",
    }
    regression_summary = summary.loc[summary["kind"].eq("regression")].sort_values("k")
    full_regression = regression_summary.iloc[-1]
    rmse_target = float(full_regression["rmse_mean"] * 1.05)
    reg_eligible = regression_summary.loc[regression_summary["rmse_mean"] <= rmse_target]
    selected_regression = reg_eligible.iloc[0] if not reg_eligible.empty else full_regression
    selected["regression"] = {
        "selected_k": int(selected_regression["k"]),
        "reference_rmse": float(full_regression["rmse_mean"]),
        "target_rmse": rmse_target,
        "selected_rmse": float(selected_regression["rmse_mean"]),
        "rule": "smallest k with mean RMSE no more than 5% above the expanded-plus-demographics reference",
    }
    return selected


def _fit_final_compact(
    table: pd.DataFrame,
    features: list[str],
    kind: str,
    selected_k: int,
    inner_folds: int,
    seed: int,
    output_dir: Path,
) -> pd.DataFrame:
    if kind == "classification":
        analysis = table
        y = analysis["target_pd"].to_numpy(int)
        labels = np.array([f"{d}__{label}" for d, label in zip(analysis["dataset"], y)])
        if min(pd.Series(labels).value_counts()) < inner_folds:
            labels = y
    else:
        analysis = table.loc[table["target_pd"].eq(1) & table["moca"].notna()]
        y = analysis["moca"].to_numpy(float)
        labels = analysis["dataset"].astype(str).to_numpy()
    search = xgb_workflow._fit_search(analysis[features], y, kind, seed, inner_folds, labels)
    importance = pd.Series(search.best_estimator_.feature_importances_, index=features).sort_values(ascending=False, kind="stable")
    selected = importance.index[:selected_k].tolist()
    estimator = clone(search.best_estimator_)
    estimator.fit(analysis[selected], y)
    model_path = output_dir / f"{kind}_minimal_{selected_k}_features.json"
    estimator.save_model(model_path)
    rows = pd.DataFrame({"kind": kind, "rank": np.arange(1, len(importance) + 1), "feature": importance.index, "importance": importance.to_numpy()})
    rows.to_csv(output_dir / f"{kind}_final_feature_ranking.csv", index=False)
    pd.DataFrame({"feature": selected, "rank": np.arange(1, len(selected) + 1)}).to_csv(output_dir / f"{kind}_minimal_features.csv", index=False)
    return pd.DataFrame([{"kind": kind, "selected_k": selected_k, "model_path": str(model_path), "features": json.dumps(selected)}])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/global"))
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--outer-repeats", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--k-values", nargs="+", type=int, default=DEFAULT_K)
    args = parser.parse_args()

    root = args.output_root
    output_dir = root / "statistics" / "xgboost_minimal_features"
    output_dir.mkdir(parents=True, exist_ok=True)
    table = xgb_workflow._load_table(root)
    candidates = xgb_workflow._feature_candidates(table)
    blocks = xgb_workflow._feature_blocks(candidates)
    expanded_features = blocks[MODEL_NAME]

    all_metrics = []
    all_predictions = []
    all_rankings = []
    for kind, seed in [("classification", args.seed), ("regression", args.seed + 100)]:
        metrics, predictions, summary, rankings = _run_minimal_curve(
            table,
            expanded_features,
            kind,
            args.k_values,
            args.outer_folds,
            args.outer_repeats,
            args.inner_folds,
            seed,
        )
        all_metrics.append(metrics)
        all_predictions.append(predictions)
        all_rankings.append(rankings)

    metrics = pd.concat(all_metrics, ignore_index=True)
    predictions = pd.concat(all_predictions, ignore_index=True)
    rankings = pd.concat(all_rankings, ignore_index=True)
    # The combined table has different metric columns; summarize each task separately.
    class_summary = xgb_workflow._summary(metrics.loc[metrics["kind"].eq("classification")], ["roc_auc", "average_precision", "balanced_accuracy", "sensitivity", "specificity", "brier_score", "log_loss"], ["kind", "k"])
    regression_summary = xgb_workflow._summary(metrics.loc[metrics["kind"].eq("regression")], ["r_squared", "rmse", "mae", "pearson_r", "spearman_rho"], ["kind", "k"])
    summary = pd.concat([class_summary, regression_summary], ignore_index=True)
    selection = _select_minimal(summary)

    final_models = pd.concat([
        _fit_final_compact(table, expanded_features, "classification", int(selection["classification"]["selected_k"]), args.inner_folds, args.seed, output_dir),
        _fit_final_compact(table, expanded_features, "regression", int(selection["regression"]["selected_k"]), args.inner_folds, args.seed + 100, output_dir),
    ], ignore_index=True)

    table.to_csv(output_dir / "analysis_table.csv.gz", index=False, compression="gzip")
    metrics.to_csv(output_dir / "minimal_outer_metrics.csv", index=False)
    predictions.to_csv(output_dir / "minimal_outer_predictions.csv.gz", index=False, compression="gzip")
    summary.to_csv(output_dir / "minimal_feature_summary.csv", index=False)
    rankings.to_csv(output_dir / "minimal_fold_feature_rankings.csv", index=False)
    final_models.to_csv(output_dir / "minimal_final_models.csv", index=False)
    (output_dir / "minimal_selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    _plot_feature_sets(blocks, root / "figures" / "summary" / "xgboost_model_feature_sets.png")
    _plot_minimal_curve(summary, selection, root / "figures" / "summary" / "xgboost_minimal_feature_curve.png")
    _plot_moca_predictions(predictions, int(selection["regression"]["selected_k"]), summary, root / "figures" / "summary" / "xgboost_moca_observed_predicted.png")

    print(f"Minimal classifier: {selection['classification']['selected_k']} features; ROC AUC={selection['classification']['selected_roc_auc']:.3f}")
    print(f"Minimal MoCA model: {selection['regression']['selected_k']} features; RMSE={selection['regression']['selected_rmse']:.3f}")
    print(f"Wrote minimal-feature analysis to {output_dir}")


if __name__ == "__main__":
    main()
