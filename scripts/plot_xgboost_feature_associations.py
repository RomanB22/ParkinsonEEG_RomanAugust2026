#!/usr/bin/env python3
"""Plot MoCA associations and Control/PD distributions for XGBoost features."""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, ttest_ind
from statsmodels.stats.multitest import multipletests

import train_xgboost_multidataset as workflow


CONTROL_COLOR = "#0072B2"
PD_COLOR = "#D55E00"
DATASET_COLORS = {
    "primary": "#0072B2",
    "ds007526-1.0.2": "#D55E00",
    "ds008768-1.0.0": "#009E73",
}


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


def _star(q_value: float) -> str:
    if not np.isfinite(q_value):
        return ""
    if q_value < 0.001:
        return "***"
    if q_value < 0.01:
        return "**"
    if q_value < 0.05:
        return "*"
    return ""


def _fdr(values: list[float]) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return multipletests(values, method="fdr_bh")[1]


def _moca_correlations(table: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows = []
    for feature in features:
        valid = table[[feature, "moca"]].apply(pd.to_numeric, errors="coerce").dropna()
        rho, p_value = spearmanr(valid[feature], valid["moca"])
        rows.append(
            {
                "feature": feature,
                "feature_label": _full_label(feature),
                "n": len(valid),
                "spearman_rho": float(rho),
                "p_value": float(p_value),
            }
        )
    result = pd.DataFrame(rows)
    result["q_value_bh"] = _fdr(result["p_value"].tolist())
    result["significance"] = result["q_value_bh"].map(_star)
    return result.sort_values("q_value_bh").reset_index(drop=True)


def _classifier_group_tests(table: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows = []
    for feature in features:
        control = pd.to_numeric(table.loc[table["group"].eq("Control"), feature], errors="coerce").dropna()
        pd_values = pd.to_numeric(table.loc[table["group"].eq("PD"), feature], errors="coerce").dropna()
        statistic, p_value = ttest_ind(pd_values, control, equal_var=False, nan_policy="omit")
        pooled_sd = np.sqrt(
            ((len(pd_values) - 1) * pd_values.var(ddof=1) + (len(control) - 1) * control.var(ddof=1))
            / max(len(pd_values) + len(control) - 2, 1)
        )
        smd = (pd_values.mean() - control.mean()) / pooled_sd if pooled_sd > 0 else np.nan
        rows.append(
            {
                "feature": feature,
                "feature_label": _full_label(feature),
                "n_control": len(control),
                "n_pd": len(pd_values),
                "control_mean": float(control.mean()),
                "pd_mean": float(pd_values.mean()),
                "pd_minus_control_smd": float(smd),
                "welch_t": float(statistic),
                "p_value": float(p_value),
            }
        )
    result = pd.DataFrame(rows)
    result["q_value_bh"] = _fdr(result["p_value"].tolist())
    result["significance"] = result["q_value_bh"].map(_star)
    return result.sort_values("q_value_bh").reset_index(drop=True)


def _plot_moca(table: pd.DataFrame, features: list[str], statistics: pd.DataFrame, output: Path) -> None:
    figure, axes = plt.subplots(3, 4, figsize=(16, 11), constrained_layout=True)
    axes = axes.ravel()
    stats = statistics.set_index("feature")
    for axis, feature in zip(axes, features):
        valid = table[[feature, "moca", "dataset"]].apply(
            lambda column: column if column.name == "dataset" else pd.to_numeric(column, errors="coerce")
        ).dropna(subset=[feature, "moca"])
        for dataset, group in valid.groupby("dataset", sort=False):
            axis.scatter(
                group[feature],
                group["moca"],
                s=22,
                alpha=0.68,
                color=DATASET_COLORS.get(dataset, "#555555"),
                label=dataset,
            )
        if len(valid) >= 3:
            coefficients = np.polyfit(valid[feature], valid["moca"], 1)
            x_values = np.linspace(valid[feature].min(), valid[feature].max(), 50)
            axis.plot(x_values, np.polyval(coefficients, x_values), color="#333333", linewidth=1.0)
        row = stats.loc[feature]
        axis.set_title(
            textwrap.fill(_full_label(feature), 29)
            + f"\nSpearman ρ={row['spearman_rho']:.2f}, q={row['q_value_bh']:.3g} {row['significance']}"
        )
        axis.set_xlabel("Feature value")
        axis.set_ylabel("MoCA")
        axis.grid(alpha=0.2)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.01))
    figure.suptitle("MoCA associations for the 12-feature XGBoost predictor\nPD participants only; q-values are Benjamini–Hochberg FDR-adjusted across 12 features", y=1.045)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _plot_violins(table: pd.DataFrame, features: list[str], statistics: pd.DataFrame, output: Path) -> None:
    figure, axes = plt.subplots(6, 4, figsize=(16, 22), constrained_layout=True)
    axes = axes.ravel()
    stats = statistics.set_index("feature")
    rng = np.random.default_rng(20260908)
    for axis, feature in zip(axes, features):
        control = pd.to_numeric(table.loc[table["group"].eq("Control"), feature], errors="coerce").dropna().to_numpy(float)
        pd_values = pd.to_numeric(table.loc[table["group"].eq("PD"), feature], errors="coerce").dropna().to_numpy(float)
        parts = axis.violinplot([control, pd_values], positions=[1, 2], showextrema=False, widths=0.8)
        for body, color in zip(parts["bodies"], [CONTROL_COLOR, PD_COLOR]):
            body.set_facecolor(color)
            body.set_edgecolor(color)
            body.set_alpha(0.65)
        for position, values, color in [(1, control, CONTROL_COLOR), (2, pd_values, PD_COLOR)]:
            median = np.median(values)
            axis.hlines(median, position - 0.22, position + 0.22, color="#222222", linewidth=2.0)
            jitter = rng.normal(position, 0.035, size=len(values))
            axis.scatter(jitter, values, s=3, alpha=0.12, color=color, rasterized=True)
        row = stats.loc[feature]
        axis.set_title(textwrap.fill(_full_label(feature), 28) + f"\nq={row['q_value_bh']:.3g} {row['significance']}")
        axis.set_xticks([1, 2], ["Control", "PD"], fontsize=8)
        axis.grid(axis="y", alpha=0.2)
    for axis in axes[len(features):]:
        axis.axis("off")
    figure.suptitle("Control versus PD distributions for classifier features\nWelch t-tests; stars indicate BH-FDR q<0.05 across 24 features", y=1.01)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/global"))
    args = parser.parse_args()
    root = args.output_root
    output_dir = root / "statistics" / "xgboost_minimal_features"
    table = workflow._load_table(root)
    moca_features = pd.read_csv(output_dir / "regression_minimal_features.csv").sort_values("rank")["feature"].tolist()
    classifier_features = pd.read_csv(output_dir / "classification_minimal_features.csv").sort_values("rank")["feature"].tolist()
    moca_table = table.loc[table["target_pd"].eq(1) & table["moca"].notna()].copy()
    moca_statistics = _moca_correlations(moca_table, moca_features)
    classifier_statistics = _classifier_group_tests(table, classifier_features)
    moca_statistics.to_csv(output_dir / "moca_feature_correlations.csv", index=False)
    classifier_statistics.to_csv(output_dir / "classifier_feature_group_tests.csv", index=False)
    _plot_moca(moca_table, moca_features, moca_statistics, root / "figures" / "summary" / "xgboost_moca_feature_associations.png")
    _plot_violins(table, classifier_features, classifier_statistics, root / "figures" / "summary" / "xgboost_classifier_feature_violins.png")
    print(f"Wrote MoCA correlations to {output_dir / 'moca_feature_correlations.csv'}")
    print(f"Wrote classifier group tests to {output_dir / 'classifier_feature_group_tests.csv'}")
    print(f"Wrote figures to {root / 'figures' / 'summary'}")


if __name__ == "__main__":
    main()
