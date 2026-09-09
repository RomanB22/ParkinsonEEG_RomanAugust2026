#!/usr/bin/env python3
"""Plot feature rankings and prediction contributions for the compact classifier."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
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


def _confidence_ellipse(ax: plt.Axes, values: np.ndarray, color: str) -> None:
    """Draw a robust 95% covariance ellipse for one diagnostic group."""
    if len(values) < 3:
        return
    covariance = np.cov(values, rowvar=False)
    if covariance.shape != (2, 2) or not np.isfinite(covariance).all():
        return
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.clip(eigenvalues, 0, None)
    if not np.any(eigenvalues > 0):
        return
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    # sqrt(chi2.ppf(.95, 2)) = 2.4477; use the exact value here to avoid
    # adding another dependency for a plotting-only diagnostic.
    scale_95 = 2.44774683068
    ellipse = Ellipse(
        xy=values.mean(axis=0),
        width=2 * scale_95 * np.sqrt(eigenvalues[0]),
        height=2 * scale_95 * np.sqrt(eigenvalues[1]),
        angle=angle,
        facecolor=color,
        edgecolor=color,
        linewidth=1.5,
        alpha=0.12,
        zorder=1,
    )
    ax.add_patch(ellipse)


def _compute_shap_projection(
    shap_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return participant coordinates, variance, and exact PCA coefficients."""
    centered = shap_values - shap_values.mean(axis=0, keepdims=True)
    _, singular_values, right_singular_vectors = np.linalg.svd(
        centered, full_matrices=False
    )
    coordinates = centered @ right_singular_vectors[:3].T
    variance = singular_values**2
    explained = variance[:3] / variance.sum() * 100
    return coordinates, explained, right_singular_vectors[:3]


def _plot_pairwise_projection(
    coordinates: np.ndarray,
    table: pd.DataFrame,
    output: Path,
    x_label: str,
    y_label: str,
    title: str,
    suptitle: str,
    note: str,
    show_ellipse: bool,
) -> None:
    """Plot a selected pair of participant-level embedding coordinates."""
    colors = {"Control": "#0072B2", "PD": "#D55E00"}
    fig, ax = plt.subplots(figsize=(12, 9))
    for group in ["Control", "PD"]:
        mask = table["group"].eq(group).to_numpy()
        points = coordinates[mask]
        ax.scatter(
            points[:, 0],
            points[:, 1],
            s=42,
            alpha=0.65,
            color=colors[group],
            edgecolor="white",
            linewidth=0.35,
            label=f"{group} (n={mask.sum()})",
            zorder=3,
        )
        if show_ellipse:
            _confidence_ellipse(ax, points, colors[group])
        centroid = points.mean(axis=0)
        ax.scatter(
            centroid[0],
            centroid[1],
            s=170,
            marker="X",
            color=colors[group],
            edgecolor="black",
            linewidth=0.8,
            zorder=5,
        )
        ax.annotate(
            f"{group} centroid",
            centroid,
            xytext=(7, 7),
            textcoords="offset points",
            color=colors[group],
            fontsize=9,
            weight="bold",
        )
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend(frameon=False, loc="best")
    ax.grid(alpha=0.16)
    ax.text(
        0.01,
        0.01,
        note,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        color="#555555",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 4},
    )
    fig.suptitle(suptitle, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_projection_2d(
    coordinates: np.ndarray,
    explained: np.ndarray,
    table: pd.DataFrame,
    output: Path,
) -> None:
    """Plot participants in the first two dimensions of XGBoost SHAP space."""
    colors = {"Control": "#0072B2", "PD": "#D55E00"}
    fig, ax = plt.subplots(figsize=(12, 9))
    for group in ["Control", "PD"]:
        mask = table["group"].eq(group).to_numpy()
        points = coordinates[mask, :2]
        ax.scatter(
            points[:, 0],
            points[:, 1],
            s=42,
            alpha=0.62,
            color=colors[group],
            edgecolor="white",
            linewidth=0.35,
            label=f"{group} (n={mask.sum()})",
            zorder=3,
        )
        _confidence_ellipse(ax, points, colors[group])
        centroid = points.mean(axis=0)
        ax.scatter(
            centroid[0],
            centroid[1],
            s=170,
            marker="X",
            color=colors[group],
            edgecolor="black",
            linewidth=0.8,
            zorder=5,
        )
        ax.annotate(
            f"{group} centroid",
            centroid,
            xytext=(7, 7),
            textcoords="offset points",
            color=colors[group],
            fontsize=9,
            weight="bold",
        )

    ax.axhline(0, color="#777777", linewidth=0.7, alpha=0.35)
    ax.axvline(0, color="#777777", linewidth=0.7, alpha=0.35)
    ax.set_xlabel(f"SHAP-space PC1 ({explained[0]:.1f}% variance)")
    ax.set_ylabel(f"SHAP-space PC2 ({explained[1]:.1f}% variance)")
    ax.set_title("PD/control differences in the XGBoost decision space")
    ax.legend(frameon=False, loc="best")
    ax.grid(alpha=0.16)
    ax.text(
        0.01,
        0.01,
        "Each point is one participant.\n"
        "Coordinates are PCA of 24-feature SHAP contributions (log-odds).\n"
        "Ellipses show the approximate 95% within-group covariance region.",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        color="#555555",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 4},
    )
    fig.suptitle(
        "Compact XGBoost classifier: 2D projection of participant-level model explanations\n"
        f"n={len(table)} participants; 24 features; final refit model",
        y=0.98,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _confidence_ellipsoid(ax, values: np.ndarray, color: str) -> None:
    """Draw an approximate 95% covariance ellipsoid in 3D."""
    if len(values) < 4:
        return
    covariance = np.cov(values, rowvar=False)
    if covariance.shape != (3, 3) or not np.isfinite(covariance).all():
        return
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.clip(eigenvalues, 0, None)
    if not np.any(eigenvalues > 0):
        return
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    u = np.linspace(0, 2 * np.pi, 28)
    v = np.linspace(0, np.pi, 16)
    sphere = np.stack(
        [
            np.cos(u)[None, :] * np.sin(v)[:, None],
            np.sin(u)[None, :] * np.sin(v)[:, None],
            np.broadcast_to(np.cos(v)[:, None], (len(v), len(u))),
        ],
        axis=0,
    ).reshape(3, -1)
    scale_95 = np.sqrt(7.814727903)  # chi-square 95th percentile, 3 dimensions
    ellipsoid = (
        eigenvectors
        @ (scale_95 * np.sqrt(eigenvalues)[:, None] * sphere)
        + values.mean(axis=0)[:, None]
    ).reshape(3, len(v), len(u))
    ax.plot_surface(
        ellipsoid[0],
        ellipsoid[1],
        ellipsoid[2],
        color=color,
        alpha=0.08,
        linewidth=0,
        shade=False,
        zorder=1,
    )


def _plot_projection_3d(
    coordinates: np.ndarray,
    explained: np.ndarray,
    table: pd.DataFrame,
    output: Path,
    coordinates_output: Path,
) -> None:
    """Plot participants in the first three dimensions of XGBoost SHAP space."""
    projection = pd.DataFrame(
        {
            "participant_id": table["participant_id"].to_numpy(),
            "dataset": table["dataset"].to_numpy(),
            "group": table["group"].to_numpy(),
            "pc1": coordinates[:, 0],
            "pc2": coordinates[:, 1],
            "pc3": coordinates[:, 2],
        }
    )
    projection.to_csv(coordinates_output, index=False)

    colors = {"Control": "#0072B2", "PD": "#D55E00"}
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection="3d")
    for group in ["Control", "PD"]:
        mask = table["group"].eq(group).to_numpy()
        points = coordinates[mask]
        ax.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            s=40,
            alpha=0.68,
            color=colors[group],
            edgecolor="white",
            linewidth=0.3,
            label=f"{group} (n={mask.sum()})",
            depthshade=False,
        )
        _confidence_ellipsoid(ax, points, colors[group])
        centroid = points.mean(axis=0)
        ax.scatter(
            centroid[0],
            centroid[1],
            centroid[2],
            s=180,
            marker="X",
            color=colors[group],
            edgecolor="black",
            linewidth=0.8,
            depthshade=False,
        )
        ax.text(
            centroid[0],
            centroid[1],
            centroid[2],
            f"  {group} centroid",
            color=colors[group],
            fontsize=9,
            weight="bold",
        )

    ax.set_xlabel(f"SHAP-space PC1 ({explained[0]:.1f}% variance)", labelpad=10)
    ax.set_ylabel(f"SHAP-space PC2 ({explained[1]:.1f}% variance)", labelpad=10)
    ax.set_zlabel(f"SHAP-space PC3 ({explained[2]:.1f}% variance)", labelpad=10)
    ax.set_title("PD/control differences in the 3D XGBoost decision space", pad=18)
    ax.legend(frameon=False, loc="upper right")
    # This higher, opposite-azimuth view exposes the PC3 spread while keeping
    # the main PC1 group separation visible.
    ax.view_init(elev=28, azim=35)
    ax.text2D(
        0.01,
        0.01,
        "Each point is one participant. Coordinates are PCA of 24-feature SHAP contributions (log-odds).\n"
        "Transparent surfaces show approximate 95% within-group covariance regions.",
        transform=ax.transAxes,
        fontsize=9,
        color="#555555",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 4},
    )
    fig.suptitle(
        "Compact XGBoost classifier: 3D projection of participant-level model explanations\n"
        f"n={len(table)} participants; 24 features; final refit model",
        y=0.96,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_pca_coefficients(
    features: list[str],
    pca_components: np.ndarray,
    explained: np.ndarray,
    output: Path,
) -> None:
    """Compare PC1-PC3 coefficients with grouped bars in one shared order."""
    coefficients = pca_components[:3].T
    order = np.argsort(np.abs(coefficients[:, 0]))[::-1]
    ordered = coefficients[order]
    labels = [_full_label(features[index]) for index in order]
    limit = float(np.max(np.abs(ordered)))
    x_limit = limit * 1.19
    positions = np.arange(len(labels))
    bar_height = 0.30
    colors = ["#0072B2", "#D55E00", "#009E73"]

    fig, ax = plt.subplots(figsize=(16, 14))
    fig.suptitle(
        "Feature weights defining the first three PCA directions\n"
        "Signed coefficients on participant-level XGBoost SHAP contributions",
        y=0.99,
        fontsize=16,
    )
    for component_index, color in enumerate(colors):
        values = ordered[:, component_index]
        offsets = positions + (component_index - 1) * bar_height
        ax.barh(
            offsets,
            values,
            height=bar_height * 0.94,
            color=color,
            alpha=0.9,
            label=(
                f"PC{component_index + 1} "
                f"({explained[component_index]:.2f}% variance)"
            ),
        )
        for position, value in zip(offsets, values):
            text_offset = 0.012 * limit
            ax.text(
                value + (text_offset if value >= 0 else -text_offset),
                position,
                f"{value:+.3f}",
                ha="left" if value >= 0 else "right",
                va="center",
                fontsize=7,
                color=color,
            )

    ax.axvline(0, color="#333333", linewidth=0.9)
    ax.set_xlim(-x_limit, x_limit)
    ax.set_yticks(positions)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Feature weight (PCA loading)")
    ax.set_ylabel("Classifier feature")
    ax.grid(axis="x", color="#D9D9D9", linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", frameon=False, fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _compute_umap_projection(shap_values: np.ndarray, n_components: int) -> np.ndarray:
    """Embed participant-level SHAP vectors with a reproducible UMAP fit."""
    # The managed environment cannot write numba caches inside the conda
    # environment, so give UMAP a writable, platform-independent cache path.
    os.environ.setdefault(
        "NUMBA_CACHE_DIR",
        str(Path(tempfile.gettempdir()) / "parkinson_eeg_numba_cache"),
    )
    import umap

    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=75,
        min_dist=0.10,
        metric="euclidean",
        random_state=20260908,
        transform_seed=20260908,
        n_jobs=1,
    )
    return reducer.fit_transform(shap_values)


def _plot_umap_feature_correlations(
    features: list[str],
    shap_values: np.ndarray,
    embedding: np.ndarray,
    output: Path,
    output_table: Path,
) -> None:
    """Plot descriptive feature correlations with UMAP dimensions 1-3."""
    centered_shap = shap_values - shap_values.mean(axis=0, keepdims=True)
    centered_embedding = embedding[:, :3] - embedding[:, :3].mean(axis=0, keepdims=True)
    numerator = centered_shap.T @ centered_embedding
    denominator = np.outer(
        np.linalg.norm(centered_shap, axis=0),
        np.linalg.norm(centered_embedding, axis=0),
    )
    correlations = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=float),
        where=denominator > 0,
    )
    pd.DataFrame(
        {
            "feature": features,
            "umap1_correlation": correlations[:, 0],
            "umap2_correlation": correlations[:, 1],
            "umap3_correlation": correlations[:, 2],
        }
    ).to_csv(output_table, index=False)

    order = np.argsort(np.abs(correlations[:, 0]))[::-1]
    ordered = correlations[order]
    labels = [_full_label(features[index]) for index in order]
    positions = np.arange(len(labels))
    bar_height = 0.30
    colors = ["#0072B2", "#D55E00", "#009E73"]
    limit = float(np.max(np.abs(ordered)))

    fig, ax = plt.subplots(figsize=(16, 14))
    fig.suptitle(
        "Feature associations with the first three UMAP dimensions\n"
        "Pearson correlations with participant-level XGBoost SHAP contributions",
        y=0.99,
        fontsize=16,
    )
    for dimension_index, color in enumerate(colors):
        values = ordered[:, dimension_index]
        offsets = positions + (dimension_index - 1) * bar_height
        ax.barh(
            offsets,
            values,
            height=bar_height * 0.94,
            color=color,
            alpha=0.9,
            label=f"UMAP-{dimension_index + 1}",
        )
        for position, value in zip(offsets, values):
            text_offset = 0.012 * limit
            ax.text(
                value + (text_offset if value >= 0 else -text_offset),
                position,
                f"{value:+.3f}",
                ha="left" if value >= 0 else "right",
                va="center",
                fontsize=7,
                color=color,
            )

    ax.axvline(0, color="#333333", linewidth=0.9)
    ax.set_xlim(-limit * 1.19, limit * 1.19)
    ax.set_yticks(positions)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Pearson correlation with UMAP coordinate")
    ax.set_ylabel("Classifier feature")
    ax.grid(axis="x", color="#D9D9D9", linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", frameon=False, fontsize=10)
    ax.text(
        0.01,
        0.01,
        "Descriptive correlations, not linear loadings; UMAP is nonlinear.",
        transform=ax.transAxes,
        fontsize=9,
        color="#555555",
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_umap_2d(embedding: np.ndarray, table: pd.DataFrame, output: Path) -> None:
    colors = {"Control": "#0072B2", "PD": "#D55E00"}
    fig, ax = plt.subplots(figsize=(12, 9))
    for group in ["Control", "PD"]:
        mask = table["group"].eq(group).to_numpy()
        points = embedding[mask]
        ax.scatter(
            points[:, 0],
            points[:, 1],
            s=42,
            alpha=0.65,
            color=colors[group],
            edgecolor="white",
            linewidth=0.35,
            label=f"{group} (n={mask.sum()})",
            zorder=3,
        )
        centroid = points.mean(axis=0)
        ax.scatter(
            centroid[0],
            centroid[1],
            s=170,
            marker="X",
            color=colors[group],
            edgecolor="black",
            linewidth=0.8,
            zorder=5,
        )
        ax.annotate(
            f"{group} centroid",
            centroid,
            xytext=(7, 7),
            textcoords="offset points",
            color=colors[group],
            fontsize=9,
            weight="bold",
        )
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_title("PD/control differences in 2D UMAP space")
    ax.legend(frameon=False, loc="best")
    ax.grid(alpha=0.16)
    ax.text(
        0.01,
        0.01,
        "Each point is one participant. UMAP inputs are the 24-feature XGBoost SHAP vectors.\n"
        "Parameters: n_neighbors=75, min_dist=0.10, Euclidean metric; no covariance shading shown.",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        color="#555555",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 4},
    )
    fig.suptitle(
        "Compact XGBoost classifier: 2D UMAP of participant-level model explanations\n"
        f"n={len(table)} participants; 24 features; final refit model",
        y=0.98,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_umap_3d(embedding: np.ndarray, table: pd.DataFrame, output: Path) -> None:
    colors = {"Control": "#0072B2", "PD": "#D55E00"}
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection="3d")
    for group in ["Control", "PD"]:
        mask = table["group"].eq(group).to_numpy()
        points = embedding[mask]
        ax.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            s=40,
            alpha=0.68,
            color=colors[group],
            edgecolor="white",
            linewidth=0.3,
            label=f"{group} (n={mask.sum()})",
            depthshade=False,
        )
        centroid = points.mean(axis=0)
        ax.scatter(
            centroid[0],
            centroid[1],
            centroid[2],
            s=180,
            marker="X",
            color=colors[group],
            edgecolor="black",
            linewidth=0.8,
            depthshade=False,
        )
        ax.text(
            centroid[0],
            centroid[1],
            centroid[2],
            f"  {group} centroid",
            color=colors[group],
            fontsize=9,
            weight="bold",
        )
    ax.set_xlabel("UMAP-1", labelpad=10)
    ax.set_ylabel("UMAP-2", labelpad=10)
    ax.set_zlabel("UMAP-3", labelpad=10)
    ax.set_title("PD/control differences in 3D UMAP space", pad=18)
    ax.legend(frameon=False, loc="upper right")
    ax.view_init(elev=28, azim=35)
    ax.text2D(
        0.01,
        0.01,
        "Each point is one participant. UMAP inputs are the 24-feature XGBoost SHAP vectors.\n"
        "Parameters: n_neighbors=75, min_dist=0.10, Euclidean metric; no covariance shading shown.",
        transform=ax.transAxes,
        fontsize=9,
        color="#555555",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 4},
    )
    fig.suptitle(
        "Compact XGBoost classifier: 3D UMAP of participant-level model explanations\n"
        f"n={len(table)} participants; 24 features; final refit model",
        y=0.96,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_contribution_bars(
    contributions_table: pd.DataFrame, table: pd.DataFrame, top_n: int, output: Path
) -> None:
    """Keep the original ranked contribution plot as a companion output."""
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
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


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

    figure_dir = root / "figures" / "summary"
    figure_dir.mkdir(parents=True, exist_ok=True)
    contribution_figure = figure_dir / "xgboost_classifier_feature_contributions_bar.png"
    projection_figure = figure_dir / "xgboost_classifier_feature_contributions.png"
    projection_figure_2d = figure_dir / "xgboost_classifier_feature_contributions_2d.png"
    projection_figure_pc1_pc3 = figure_dir / "xgboost_classifier_feature_contributions_pc1_pc3.png"
    projection_figure_pc2_pc3 = figure_dir / "xgboost_classifier_feature_contributions_pc2_pc3.png"
    coefficient_figure = figure_dir / "xgboost_classifier_shap_pca_coefficients.png"
    umap_figure_2d = figure_dir / "xgboost_classifier_shap_umap_2d.png"
    umap_figure_3d = figure_dir / "xgboost_classifier_shap_umap_3d.png"
    umap_correlation_figure = figure_dir / "xgboost_classifier_shap_umap_feature_correlations.png"
    umap_figure_1_3 = figure_dir / "xgboost_classifier_shap_umap_1_3.png"
    umap_figure_2_3 = figure_dir / "xgboost_classifier_shap_umap_2_3.png"
    projection_table = output_dir / "classifier_shap_projection.csv"
    umap_table_2d = output_dir / "classifier_shap_umap_2d.csv"
    umap_table_3d = output_dir / "classifier_shap_umap_3d.csv"
    umap_correlation_table = output_dir / "classifier_shap_umap_feature_correlations.csv"
    loading_table = output_dir / "classifier_shap_pca_loadings.csv"
    shap_coordinates, explained, pca_components = _compute_shap_projection(shap_values)
    pd.DataFrame(
        {
            "feature": features,
            "pc1_coefficient": pca_components[0],
            "pc2_coefficient": pca_components[1],
            "pc3_coefficient": pca_components[2],
            "pc1_abs_coefficient": np.abs(pca_components[0]),
            "pc2_abs_coefficient": np.abs(pca_components[1]),
            "pc3_abs_coefficient": np.abs(pca_components[2]),
            "pc1_variance_explained_percent": explained[0],
            "pc2_variance_explained_percent": explained[1],
            "pc3_variance_explained_percent": explained[2],
        }
    ).to_csv(loading_table, index=False)
    _plot_contribution_bars(contributions_table, table, top_n, contribution_figure)
    _plot_projection_2d(
        shap_coordinates,
        explained,
        table,
        projection_figure_2d,
    )
    _plot_projection_3d(
        shap_coordinates,
        explained,
        table,
        projection_figure,
        projection_table,
    )
    pca_note = (
        "Each point is one participant. Coordinates are PCA of 24-feature "
        "XGBoost SHAP contributions (log-odds)."
    )
    _plot_pairwise_projection(
        shap_coordinates[:, [0, 2]],
        table,
        projection_figure_pc1_pc3,
        f"SHAP-space PC1 ({explained[0]:.1f}% variance)",
        f"SHAP-space PC3 ({explained[2]:.1f}% variance)",
        "PD/control differences in the PC1–PC3 plane",
        "Compact XGBoost classifier: PC1–PC3 projection",
        pca_note,
        show_ellipse=True,
    )
    _plot_pairwise_projection(
        shap_coordinates[:, [1, 2]],
        table,
        projection_figure_pc2_pc3,
        f"SHAP-space PC2 ({explained[1]:.1f}% variance)",
        f"SHAP-space PC3 ({explained[2]:.1f}% variance)",
        "PD/control differences in the PC2–PC3 plane",
        "Compact XGBoost classifier: PC2–PC3 projection",
        pca_note,
        show_ellipse=True,
    )
    _plot_pca_coefficients(features, pca_components, explained, coefficient_figure)
    umap_coordinates_2d = _compute_umap_projection(shap_values, n_components=2)
    umap_coordinates_3d = _compute_umap_projection(shap_values, n_components=3)
    pd.DataFrame(
        {
            "participant_id": table["participant_id"].to_numpy(),
            "dataset": table["dataset"].to_numpy(),
            "group": table["group"].to_numpy(),
            "umap1": umap_coordinates_2d[:, 0],
            "umap2": umap_coordinates_2d[:, 1],
        }
    ).to_csv(umap_table_2d, index=False)
    pd.DataFrame(
        {
            "participant_id": table["participant_id"].to_numpy(),
            "dataset": table["dataset"].to_numpy(),
            "group": table["group"].to_numpy(),
            "umap1": umap_coordinates_3d[:, 0],
            "umap2": umap_coordinates_3d[:, 1],
            "umap3": umap_coordinates_3d[:, 2],
        }
    ).to_csv(umap_table_3d, index=False)
    _plot_umap_2d(umap_coordinates_2d, table, umap_figure_2d)
    _plot_umap_3d(umap_coordinates_3d, table, umap_figure_3d)
    _plot_umap_feature_correlations(
        features,
        shap_values,
        umap_coordinates_3d,
        umap_correlation_figure,
        umap_correlation_table,
    )
    umap_note = (
        "Each point is one participant. Coordinates are from the 3D UMAP fit "
        "of the 24-feature XGBoost SHAP vectors.\n"
        "Parameters: n_neighbors=75, min_dist=0.10, Euclidean metric; no covariance shading shown."
    )
    _plot_pairwise_projection(
        umap_coordinates_3d[:, [0, 2]],
        table,
        umap_figure_1_3,
        "UMAP-1",
        "UMAP-3",
        "PD/control differences in the UMAP-1–UMAP-3 plane",
        "Compact XGBoost classifier: UMAP-1–UMAP-3 projection",
        umap_note,
        show_ellipse=False,
    )
    _plot_pairwise_projection(
        umap_coordinates_3d[:, [1, 2]],
        table,
        umap_figure_2_3,
        "UMAP-2",
        "UMAP-3",
        "PD/control differences in the UMAP-2–UMAP-3 plane",
        "Compact XGBoost classifier: UMAP-2–UMAP-3 projection",
        umap_note,
        show_ellipse=False,
    )
    print(f"Wrote contribution table to {output_dir / 'classifier_feature_contributions.csv'}")
    print(f"Wrote contribution bar figure to {contribution_figure}")
    print(f"Wrote 2D SHAP projection to {projection_figure_2d}")
    print(f"Wrote 3D SHAP projection to {projection_figure}")
    print(f"Wrote PC1-PC3 projection to {projection_figure_pc1_pc3}")
    print(f"Wrote PC2-PC3 projection to {projection_figure_pc2_pc3}")
    print(f"Wrote PCA coefficient comparison to {coefficient_figure}")
    print(f"Wrote 2D UMAP projection to {umap_figure_2d}")
    print(f"Wrote 3D UMAP projection to {umap_figure_3d}")
    print(f"Wrote UMAP feature correlations to {umap_correlation_figure}")
    print(f"Wrote UMAP-1/UMAP-3 projection to {umap_figure_1_3}")
    print(f"Wrote UMAP-2/UMAP-3 projection to {umap_figure_2_3}")
    print(f"Wrote projection coordinates to {projection_table}")
    print(f"Wrote exact PCA feature coefficients to {loading_table}")
    for component_index in range(3):
        order = np.argsort(np.abs(pca_components[component_index]))[::-1][:3]
        summary = ", ".join(
            f"{features[index]} ({pca_components[component_index, index]:+.3f})"
            for index in order
        )
        print(
            f"PC{component_index + 1}: {explained[component_index]:.2f}% variance; "
            f"largest absolute coefficients: {summary}"
        )


if __name__ == "__main__":
    main()
