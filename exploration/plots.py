"""Figures for feature audit, transparent models, and validation."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from src.runtime import configure_runtime

configure_runtime()

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)


def _save(fig: Any, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def feature_label(name: str) -> str:
    replacements = {
        "age_years": "Age (years)",
        "sex_male": "Sex (male=1)",
        "moca": "MOCA",
        "ordinal_global_entropy": "Ordinal entropy H",
        "ordinal_global_complexity": "Ordinal complexity C",
        "ordinal_global_fisher_information": "Fisher information F",
    }
    if name in replacements:
        return replacements[name]
    if name.startswith("psd_log2_"):
        return "PSD log₂ " + name.removeprefix("psd_log2_").replace("_vs_", " / ").replace("_", " ")
    if name == "aperiodic_exponent":
        return "Aperiodic exponent"
    if name.startswith("bout_ordinal_"):
        return "Within-bout ordinal " + name.removeprefix("bout_ordinal_").replace("_", " ")
    if name.startswith("bout_"):
        return "Bout " + name.removeprefix("bout_").replace("_", " ")
    if name.startswith("typical_"):
        return "Typical bout " + name.removeprefix("typical_").replace("_", " ")
    if name.startswith("ordinal_"):
        return "Ordinal " + name.removeprefix("ordinal_").replace("_", " ")
    return name.replace("_", " ").title()


def plot_feature_distributions(
    feature_table: pd.DataFrame,
    features: list[str],
    group_order: list[str],
    colors: dict[str, str],
    path: Path,
    dpi: int,
) -> None:
    """Plot subject-level distributions for all primary candidate features."""
    n_columns = 3
    n_rows = math.ceil(len(features) / n_columns)
    fig, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(4.3 * n_columns, 3.6 * n_rows),
        squeeze=False,
    )
    rng = np.random.default_rng(0)
    for axis, feature in zip(axes.flat, features):
        for group_index, group in enumerate(group_order, start=1):
            values = feature_table.loc[
                feature_table["group"].eq(group), feature
            ].dropna().to_numpy(dtype=float)
            if len(values) >= 2 and not np.allclose(values, values[0]):
                violin = axis.violinplot(
                    [values],
                    positions=[group_index],
                    widths=0.72,
                    showmedians=True,
                    showextrema=False,
                )
                violin["bodies"][0].set_facecolor(colors[group])
                violin["bodies"][0].set_edgecolor(colors[group])
                violin["bodies"][0].set_alpha(0.3)
                violin["cmedians"].set_color("black")
            jitter = rng.uniform(-0.06, 0.06, len(values))
            axis.scatter(
                group_index + jitter,
                values,
                color=colors[group],
                edgecolor="white",
                linewidth=0.3,
                s=14,
                alpha=0.75,
                zorder=3,
            )
        axis.set_xticks(
            range(1, len(group_order) + 1),
            [
                f"{group}\nn={feature_table['group'].eq(group).sum()}"
                for group in group_order
            ],
        )
        axis.set_title(feature_label(feature), fontsize=10)
        axis.grid(axis="y", alpha=0.2)
    for axis in axes.flat[len(features) :]:
        axis.set_visible(False)
    fig.suptitle("Prespecified subject-level candidate features", fontsize=14)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.985))
    _save(fig, path, dpi)


def plot_entropy_complexity_plane(
    feature_table: pd.DataFrame,
    group_order: list[str],
    colors: dict[str, str],
    path: Path,
    dpi: int,
) -> None:
    """Plot the global H-C plane with one point per subject."""
    fig, axis = plt.subplots(figsize=(7, 6))
    for group in group_order:
        selected = feature_table.loc[feature_table["group"].eq(group)]
        axis.scatter(
            selected["ordinal_global_entropy"],
            selected["ordinal_global_complexity"],
            color=colors[group],
            alpha=0.72,
            s=28,
            label=f"{group} (n={len(selected)})",
        )
    axis.set(
        xlabel="Permutation entropy H",
        ylabel="Statistical complexity C",
        title="Global ordinal entropy-complexity plane",
    )
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    fig.tight_layout()
    _save(fig, path, dpi)


def plot_feature_correlations(
    feature_table: pd.DataFrame,
    features: list[str],
    path: Path,
    dpi: int,
) -> None:
    """Plot an outcome-blind Spearman feature-correlation audit."""
    correlations = feature_table[features].corr(method="spearman")
    size = max(8.0, 0.65 * len(features))
    fig, axis = plt.subplots(figsize=(size, size))
    image = axis.imshow(correlations, vmin=-1.0, vmax=1.0, cmap="coolwarm")
    labels = [feature_label(feature) for feature in features]
    axis.set_xticks(np.arange(len(features)), labels, rotation=70, ha="right", fontsize=8)
    axis.set_yticks(np.arange(len(features)), labels, fontsize=8)
    axis.set_title("Spearman correlation of prespecified predictors")
    fig.colorbar(image, ax=axis, label="Spearman ρ", shrink=0.78)
    fig.tight_layout()
    _save(fig, path, dpi)


def plot_model_performance(
    performance: pd.DataFrame,
    model_order: list[str],
    path: Path,
    dpi: int,
) -> None:
    """Plot bootstrap performance intervals for major discrimination metrics."""
    metrics = ["roc_auc", "average_precision", "balanced_accuracy", "brier_score"]
    titles = ["ROC AUC", "Precision-recall AUC", "Balanced accuracy", "Brier score (lower is better)"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(17, 6), sharey=True)
    available = [model for model in model_order if model in set(performance["model"])]
    labels = [
        performance.loc[performance["model"].eq(model), "model_label"].iloc[0]
        for model in available
    ]
    y_positions = np.arange(len(available))
    for axis, metric, title in zip(axes, metrics, titles):
        selected = performance.loc[performance["metric"].eq(metric)].set_index("model").reindex(available)
        estimates = selected["estimate"].to_numpy(dtype=float)
        lower = selected["ci_lower"].to_numpy(dtype=float)
        upper = selected["ci_upper"].to_numpy(dtype=float)
        axis.errorbar(
            estimates,
            y_positions,
            xerr=np.vstack([estimates - lower, upper - estimates]),
            fmt="o",
            color="#0072B2",
            ecolor="0.45",
            capsize=3,
        )
        if metric in {"roc_auc", "balanced_accuracy"}:
            axis.axvline(0.5, color="0.6", linestyle="--", linewidth=1)
        axis.set_title(title, fontsize=10)
        axis.grid(axis="x", alpha=0.2)
        axis.set_yticks(y_positions, labels)
    axes[0].invert_yaxis()
    fig.suptitle("Repeated nested-CV performance with subject-bootstrap 95% intervals")
    fig.tight_layout()
    _save(fig, path, dpi)


def plot_roc_and_precision_recall(
    predictions: pd.DataFrame,
    model_order: list[str],
    path: Path,
    dpi: int,
) -> None:
    """Plot ROC and precision-recall curves from averaged out-of-fold predictions."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    cmap = plt.get_cmap("tab10")
    available = [model for model in model_order if model in set(predictions["model"])]
    for index, model in enumerate(available):
        selected = predictions.loc[predictions["model"].eq(model)]
        truth = selected["target_pd"].to_numpy(dtype=int)
        probability = selected["predicted_probability_pd"].to_numpy(dtype=float)
        false_positive, true_positive, _ = roc_curve(truth, probability)
        precision, recall, _ = precision_recall_curve(truth, probability)
        label = selected["model_label"].iloc[0]
        axes[0].plot(false_positive, true_positive, color=cmap(index), label=label)
        axes[1].plot(recall, precision, color=cmap(index), label=label)
    axes[0].plot([0, 1], [0, 1], color="0.6", linestyle="--", linewidth=1)
    prevalence = float(predictions.drop_duplicates("subject_id")["target_pd"].mean())
    axes[1].axhline(prevalence, color="0.6", linestyle="--", linewidth=1)
    axes[0].set(xlabel="False-positive rate", ylabel="True-positive rate", title="ROC curves")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision-recall curves")
    for axis in axes:
        axis.grid(alpha=0.2)
    axes[1].legend(frameon=False, fontsize=8, bbox_to_anchor=(1.04, 1), loc="upper left")
    fig.suptitle("Averaged repeated out-of-fold predictions")
    fig.tight_layout()
    _save(fig, path, dpi)


def plot_calibration(
    predictions: pd.DataFrame,
    model_order: list[str],
    path: Path,
    dpi: int,
) -> None:
    """Plot quantile-binned calibration curves and prediction histograms."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    cmap = plt.get_cmap("tab10")
    available = [model for model in model_order if model in set(predictions["model"])]
    for index, model in enumerate(available):
        selected = predictions.loc[predictions["model"].eq(model)]
        truth = selected["target_pd"].to_numpy(dtype=int)
        probability = selected["predicted_probability_pd"].to_numpy(dtype=float)
        observed, predicted = calibration_curve(
            truth, probability, n_bins=8, strategy="quantile"
        )
        label = selected["model_label"].iloc[0]
        axes[0].plot(predicted, observed, marker="o", color=cmap(index), label=label)
        axes[1].hist(probability, bins=np.linspace(0, 1, 16), histtype="step", linewidth=1.5, color=cmap(index), label=label)
    axes[0].plot([0, 1], [0, 1], color="0.6", linestyle="--", linewidth=1)
    axes[0].set(xlabel="Mean predicted probability", ylabel="Observed PD fraction", title="Calibration")
    axes[1].set(xlabel="Predicted PD probability", ylabel="Subject count", title="Prediction distributions")
    for axis in axes:
        axis.grid(alpha=0.2)
    axes[1].legend(frameon=False, fontsize=8, bbox_to_anchor=(1.04, 1), loc="upper left")
    fig.suptitle("Calibration of averaged repeated out-of-fold predictions")
    fig.tight_layout()
    _save(fig, path, dpi)


def plot_confusion_matrices(
    predictions: pd.DataFrame,
    model_order: list[str],
    path: Path,
    dpi: int,
) -> None:
    """Plot fixed-threshold confusion matrices for all models."""
    available = [model for model in model_order if model in set(predictions["model"])]
    n_columns = 3
    n_rows = math.ceil(len(available) / n_columns)
    fig, axes = plt.subplots(n_rows, n_columns, figsize=(4 * n_columns, 3.7 * n_rows), squeeze=False)
    for axis, model in zip(axes.flat, available):
        selected = predictions.loc[predictions["model"].eq(model)]
        matrix = confusion_matrix(
            selected["target_pd"], selected["predicted_class_pd"], labels=[0, 1]
        )
        image = axis.imshow(matrix, cmap="Blues", vmin=0)
        for row in range(2):
            for column in range(2):
                axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
        axis.set_xticks([0, 1], ["Control", "PD"])
        axis.set_yticks([0, 1], ["Control", "PD"])
        axis.set(xlabel="Predicted", ylabel="Observed", title=selected["model_label"].iloc[0])
        fig.colorbar(image, ax=axis, shrink=0.72)
    for axis in axes.flat[len(available) :]:
        axis.set_visible(False)
    fig.suptitle("Out-of-fold confusion matrices — thresholds selected in training only")
    fig.tight_layout()
    _save(fig, path, dpi)


def plot_coefficient_stability(
    coefficients: pd.DataFrame,
    model_name: str,
    path: Path,
    dpi: int,
) -> None:
    """Plot standardized coefficient median and central 95% across outer fits."""
    selected = coefficients.loc[coefficients["model"].eq(model_name)]
    rows = []
    for feature, values in selected.groupby("feature", sort=False):
        coefficients_array = values["coefficient_per_sd"].to_numpy(dtype=float)
        rows.append(
            {
                "feature": feature,
                "median": float(np.median(coefficients_array)),
                "lower": float(np.quantile(coefficients_array, 0.025)),
                "upper": float(np.quantile(coefficients_array, 0.975)),
            }
        )
    summary = pd.DataFrame.from_records(rows).sort_values("median")
    positions = np.arange(len(summary))
    fig, axis = plt.subplots(figsize=(9, max(4.5, 0.55 * len(summary))))
    axis.errorbar(
        summary["median"],
        positions,
        xerr=np.vstack([summary["median"] - summary["lower"], summary["upper"] - summary["median"]]),
        fmt="o",
        color="#0072B2",
        ecolor="0.45",
        capsize=3,
    )
    axis.axvline(0.0, color="black", linewidth=0.8)
    axis.set_yticks(positions, [feature_label(value) for value in summary["feature"]])
    axis.set(
        xlabel="Standardized logistic coefficient (PD positive)",
        title=f"Coefficient stability across outer fits — {selected['model_label'].iloc[0]}",
    )
    axis.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    _save(fig, path, dpi)


def plot_sweep_sensitivity(
    sweep_performance: pd.DataFrame,
    path: Path,
    dpi: int,
) -> None:
    """Plot ordinal-core ROC AUC across completed D/tau sweep runs."""
    fig, axis = plt.subplots(figsize=(8, 5.5))
    for dimension, selected in sweep_performance.groupby("embedding_dimension"):
        selected = selected.sort_values("delay_samples")
        axis.errorbar(
            selected["delay_samples"],
            selected["estimate"],
            yerr=np.vstack(
                [
                    selected["estimate"] - selected["ci_lower"],
                    selected["ci_upper"] - selected["estimate"],
                ]
            ),
            marker="o",
            capsize=3,
            label=f"D={dimension}",
        )
    axis.axhline(0.5, color="0.6", linestyle="--", linewidth=1)
    axis.set(
        xlabel="Delay τ (samples)",
        ylabel="Nested-CV ROC AUC",
        title="Ordinal H/C/F parameter sensitivity",
    )
    axis.set_xticks(sorted(sweep_performance["delay_samples"].unique()))
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    fig.tight_layout()
    _save(fig, path, dpi)
