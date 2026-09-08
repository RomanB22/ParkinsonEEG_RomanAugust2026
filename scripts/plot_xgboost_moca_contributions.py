#!/usr/bin/env python3
"""Plot feature rankings and prediction contributions for the compact MoCA model."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from xgboost import DMatrix, XGBRegressor

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/global"))
    parser.add_argument("--top-n", type=int, default=12)
    args = parser.parse_args()

    root = args.output_root
    output_dir = root / "statistics" / "xgboost_minimal_features"
    table = workflow._load_table(root)
    table = table.loc[table["target_pd"].eq(1) & table["moca"].notna()].copy()
    feature_file = output_dir / "regression_minimal_features.csv"
    model_file = output_dir / "regression_minimal_12_features.json"
    if not feature_file.exists() or not model_file.exists():
        raise FileNotFoundError(
            "Compact MoCA outputs are missing; run analyze_xgboost_minimal_features.py first"
        )
    features = pd.read_csv(feature_file).sort_values("rank")["feature"].tolist()
    features = features[: min(args.top_n, len(features))]
    model = XGBRegressor()
    model.load_model(model_file)

    booster = model.get_booster()
    contributions = booster.predict(
        DMatrix(table[features], feature_names=features), pred_contribs=True
    )
    shap_values = contributions[:, :-1]
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    mean_shap = np.mean(shap_values, axis=0)
    gain = model.feature_importances_
    gain = gain / gain.sum() if gain.sum() else gain

    fold_rankings = pd.read_csv(output_dir / "minimal_fold_feature_rankings.csv")
    fold_rankings = fold_rankings.loc[
        (fold_rankings["kind"] == "regression")
        & fold_rankings["feature"].isin(features)
    ]
    rank_summary = (
        fold_rankings.groupby("feature")
        .agg(
            median_cv_rank=("rank", "median"),
            mean_cv_rank=("rank", "mean"),
            top8_fraction=("rank", lambda values: float(np.mean(values <= 8))),
            top12_fraction=("rank", lambda values: float(np.mean(values <= 12))),
        )
        .reset_index()
    )
    contribution_table = pd.DataFrame(
        {
            "feature": features,
            "mean_abs_shap_moca_points": mean_abs_shap,
            "mean_shap_moca_points": mean_shap,
            "normalized_gain": gain,
        }
    ).merge(rank_summary, on="feature", how="left")
    contribution_table = contribution_table.sort_values(
        "mean_abs_shap_moca_points", ascending=False
    ).reset_index(drop=True)
    contribution_table.insert(0, "contribution_rank", np.arange(1, len(contribution_table) + 1))
    contribution_table.to_csv(output_dir / "moca_feature_contributions.csv", index=False)

    plot_table = contribution_table.sort_values("mean_abs_shap_moca_points", ascending=True)
    labels = [_full_label(feature) for feature in plot_table["feature"]]
    colors = np.where(plot_table["mean_shap_moca_points"] >= 0, "#D55E00", "#0072B2")
    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.barh(labels, plot_table["mean_abs_shap_moca_points"], color=colors)
    ax.set_xlabel("Mean absolute SHAP contribution (MoCA points)")
    ax.set_title("Contribution to MoCA predictions")
    ax.grid(axis="x", alpha=0.2)
    ax.text(
        0.98,
        0.02,
        "Orange = pushes toward higher MoCA\nBlue = pushes toward lower MoCA",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color="#555555",
    )

    fig.suptitle(
        f"Compact XGBoost MoCA model: ranked feature contributions\n"
        f"n={len(table)} PD participants; {len(features)} features; SHAP values from the final refit model",
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    figure = root / "figures" / "summary" / "xgboost_moca_feature_contributions.png"
    figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote contribution table to {output_dir / 'moca_feature_contributions.csv'}")
    print(f"Wrote contribution figure to {figure}")


if __name__ == "__main__":
    main()
