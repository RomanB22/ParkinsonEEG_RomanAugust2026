#!/usr/bin/env python3
"""Plot feature rankings and prediction contributions for the compact classifier."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from xgboost import XGBClassifier, DMatrix

import train_xgboost_multidataset as workflow


def _full_label(feature: str) -> str:
    if feature == "alpha_theta_ratio":
        return "Alpha/theta relative-power ratio"
    if feature == "theta_beta_ratio":
        return "Theta/beta relative-power ratio"
    if feature == "age_years":
        return "Age (years)"
    if feature == "sex_male":
        return "Sex (male)"
    if feature.startswith("psd__"):
        _, band, measure = feature.split("__")
        return f"PSD {band} {measure.replace('_', ' ')}"
    if feature.startswith("entropy__") or feature.startswith("within_bout__"):
        within = feature.startswith("within_bout__")
        _, band, metric, dimension = feature.split("__")
        prefix = "Within-bout" if within else "Whole-recording"
        metric_name = metric.replace("weighted_permutation_entropy", "weighted permutation entropy")
        metric_name = metric_name.replace("fisher_information", "Fisher information")
        return f"{prefix} {band} {metric_name} ({dimension.replace('D', 'D=')})"
    if feature.startswith("bout__"):
        _, band, metric = feature.split("__")
        metric_name = {
            "n_bouts": "number of bouts",
            "oscillatory_occupancy": "oscillatory occupancy",
            "bouts_per_minute": "bouts per minute",
            "duration_mean_s": "mean duration (seconds)",
            "amplitude_mean": "mean amplitude",
            "cycles_mean": "mean cycles",
        }.get(metric, metric.replace("_", " "))
        return f"{band.capitalize()} bout {metric_name}"
    return feature.replace("_", " ")


def _load_classifier_data(root: Path, output_dir: Path) -> tuple[pd.DataFrame, list[str], XGBClassifier]:
    table = workflow._load_table(root)
    feature_file = output_dir / "classification_minimal_features.csv"
    model_file = output_dir / "classification_minimal_24_features.json"
    if not feature_file.exists() or not model_file.exists():
        raise FileNotFoundError(
            "Compact classifier outputs are missing; run analyze_xgboost_minimal_features.py first"
        )
    features = pd.read_csv(feature_file).sort_values("rank")["feature"].tolist()
    model = XGBClassifier()
    model.load_model(model_file)
    return table, features, model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/global"))
    parser.add_argument("--top-n", type=int, default=24)
    args = parser.parse_args()

    root = args.output_root
    output_dir = root / "statistics" / "xgboost_minimal_features"
    table, features, model = _load_classifier_data(root, output_dir)
    top_n = min(args.top_n, len(features))
    features = features[:top_n]
    x = table[features]
    booster = model.get_booster()
    contributions = booster.predict(
        DMatrix(x, feature_names=features), pred_contribs=True
    )
    shap_values = contributions[:, :-1]
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    mean_shap = np.mean(shap_values, axis=0)
    gain = model.feature_importances_
    gain = gain / gain.sum() if gain.sum() else gain

    fold_rankings = pd.read_csv(output_dir / "minimal_fold_feature_rankings.csv")
    fold_rankings = fold_rankings.loc[
        (fold_rankings["kind"] == "classification")
        & fold_rankings["feature"].isin(features)
    ]
    rank_summary = (
        fold_rankings.groupby("feature")
        .agg(
            median_cv_rank=("rank", "median"),
            mean_cv_rank=("rank", "mean"),
            top12_fraction=("rank", lambda values: float(np.mean(values <= 12))),
            top24_fraction=("rank", lambda values: float(np.mean(values <= 24))),
        )
        .reset_index()
    )
    contributions_table = pd.DataFrame(
        {
            "feature": features,
            "mean_abs_shap_log_odds": mean_abs_shap,
            "mean_shap_log_odds": mean_shap,
            "normalized_gain": gain,
        }
    ).merge(rank_summary, on="feature", how="left")
    contributions_table = contributions_table.sort_values(
        "mean_abs_shap_log_odds", ascending=False
    ).reset_index(drop=True)
    contributions_table.insert(0, "contribution_rank", np.arange(1, len(contributions_table) + 1))
    contributions_table.to_csv(output_dir / "classifier_feature_contributions.csv", index=False)

    plot_table = contributions_table.sort_values("mean_abs_shap_log_odds", ascending=True)
    fig, ax = plt.subplots(figsize=(12, 10))
    labels = [_full_label(value) for value in plot_table["feature"]]
    colors = np.where(plot_table["mean_shap_log_odds"] >= 0, "#D55E00", "#0072B2")
    ax.barh(labels, plot_table["mean_abs_shap_log_odds"], color=colors)
    ax.set_xlabel("Mean absolute SHAP contribution (log-odds)")
    ax.set_title("Contribution to PD/Control predictions")
    ax.grid(axis="x", alpha=0.2)
    ax.text(
        0.98,
        0.02,
        "Orange = pushes toward PD\nBlue = pushes toward Control",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color="#555555",
    )

    fig.suptitle(
        f"Compact XGBoost classifier: ranked feature contributions\n"
        f"n={len(table)} participants; {top_n} features; SHAP values computed from the final refit model",
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    figure = root / "figures" / "summary" / "xgboost_classifier_feature_contributions.png"
    figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote contribution table to {output_dir / 'classifier_feature_contributions.csv'}")
    print(f"Wrote contribution figure to {figure}")


if __name__ == "__main__":
    main()
