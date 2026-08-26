"""Transparent cohort, association, and spatial figures for MOCA analysis."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from src.runtime import configure_runtime

configure_runtime()

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd


DIMENSION_METRIC_TITLES = {
    "entropy": "Permutation entropy H",
    "complexity": "Statistical complexity C",
    "fisher_information": "Fisher information F",
    "renyi_entropy_alpha_0_1": "Rényi entropy Hα (α=0.1)",
    "renyi_complexity_alpha_0_1": "Rényi complexity Cα (α=0.1)",
    "renyi_entropy_alpha_0_5": "Rényi entropy Hα (α=0.5)",
    "renyi_complexity_alpha_0_5": "Rényi complexity Cα (α=0.5)",
    "renyi_entropy_alpha_0_9": "Rényi entropy Hα (α=0.9)",
    "renyi_complexity_alpha_0_9": "Rényi complexity Cα (α=0.9)",
    "renyi_entropy_alpha_1_1": "Rényi entropy Hα (α=1.1)",
    "renyi_complexity_alpha_1_1": "Rényi complexity Cα (α=1.1)",
    "renyi_entropy_alpha_2": "Rényi entropy Hα (α=2)",
    "renyi_complexity_alpha_2": "Rényi complexity Cα (α=2)",
    "renyi_entropy_alpha_5": "Rényi entropy Hα (α=5)",
    "renyi_complexity_alpha_5": "Rényi complexity Cα (α=5)",
    "renyi_entropy_alpha_10": "Rényi entropy Hα (α=10)",
    "renyi_complexity_alpha_10": "Rényi complexity Cα (α=10)",
}

MOCA_CATEGORY_BOUNDARY = 25.5


def plot_aperiodic_exponent_group_comparison(
    features: pd.DataFrame,
    comparison: pd.DataFrame,
    path: Path,
    dpi: int,
) -> None:
    """Show all-fit and QC-qualified subject means with adjusted estimates."""
    groups = ["Control", "PD"]
    fig, axes = plt.subplots(
        1, len(comparison), figsize=(7.0 * len(comparison), 6.2), squeeze=False
    )
    rng = np.random.default_rng(20260824)
    for axis, (_, result) in zip(axes.flat, comparison.iterrows()):
        table = features.loc[
            features["feature_id"].eq(result["feature_id"])
        ].dropna(subset=["value", "group"])
        values = [
            table.loc[table["group"].eq(group), "value"].to_numpy(dtype=float)
            for group in groups
        ]
        violin = axis.violinplot(
            values, positions=[0, 1], widths=0.8, showextrema=False
        )
        for body, color in zip(violin["bodies"], ["#0072B2", "#D55E00"]):
            body.set_facecolor(color)
            body.set_edgecolor(color)
            body.set_alpha(0.22)
        box = axis.boxplot(
            values,
            positions=[0, 1],
            widths=0.25,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "black", "linewidth": 1.5},
        )
        for patch, color in zip(box["boxes"], ["#0072B2", "#D55E00"]):
            patch.set_facecolor(color)
            patch.set_alpha(0.45)
        for position, (group, group_values, color) in enumerate(
            zip(groups, values, ["#0072B2", "#D55E00"])
        ):
            axis.scatter(
                position + rng.uniform(-0.13, 0.13, len(group_values)),
                group_values,
                color=color,
                alpha=0.62,
                s=24,
                edgecolors="white",
                linewidths=0.3,
                label=f"{group} (n={len(group_values)})",
            )
        axis.set_xticks([0, 1], groups)
        title = (
            "All 60 electrode fits"
            if result["feature_id"] == "aperiodic_exponent"
            else "QC-qualified sensitivity"
        )
        axis.set(ylabel="Subject-mean aperiodic exponent", title=title)
        axis.grid(axis="y", alpha=0.2)
        axis.legend(frameon=False, loc="upper left")
        axis.text(
            0.98,
            0.98,
            (
                "Age/sex-adjusted PD − Control\n"
                f"Δ={result['adjusted_pd_coefficient']:.3f} "
                f"[{result['adjusted_pd_ci_lower']:.3f}, "
                f"{result['adjusted_pd_ci_upper']:.3f}]\n"
                f"HC3 p={result['adjusted_pd_p_value']:.4g}; "
                f"BH q={result['adjusted_pd_p_fdr_bh']:.4g}\n"
                f"Hedges g={result['hedges_g_pd_minus_control']:.3f}; "
                f"Welch p={result['welch_p_value']:.4g}"
            ),
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "0.75", "alpha": 0.9},
        )
    fig.suptitle("Aperiodic exponent by diagnostic group (specparam, 1–50 Hz)")
    fig.tight_layout()
    _save(fig, path, dpi)


def _save(fig: Any, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)


def plot_cohort_audit(
    cohort: pd.DataFrame,
    features: pd.DataFrame,
    dictionary: pd.DataFrame,
    primary_group: str,
    path: Path,
    dpi: int,
) -> None:
    """Show MOCA range, covariates, and complete-case feature coverage."""
    selected = cohort.loc[cohort["group"].eq(primary_group)]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    bins = np.arange(cohort["moca"].min() - 0.5, cohort["moca"].max() + 1.5)
    for group, color in (("PD", "#D55E00"), ("Control", "#0072B2")):
        values = cohort.loc[cohort["group"].eq(group), "moca"].to_numpy(dtype=float)
        axes[0, 0].hist(values, bins=bins, alpha=0.45, color=color, label=f"{group} (n={len(values)})")
    axes[0, 0].set(xlabel="MOCA score", ylabel="Subjects", title="MOCA distribution")
    axes[0, 0].axvline(
        MOCA_CATEGORY_BOUNDARY, color="black", linestyle="--", linewidth=1.2,
        label="Impaired <26 | Normal 26–30",
    )
    axes[0, 0].legend(frameon=False)

    for gender, marker, color in (("F", "o", "#CC79A7"), ("M", "^", "#009E73")):
        values = selected.loc[selected["gender"].eq(gender)]
        axes[0, 1].scatter(
            values["age_years"], values["moca"], marker=marker, color=color, alpha=0.7, label=f"{gender} (n={len(values)})"
        )
    axes[0, 1].set(xlabel="Age (years)", ylabel="MOCA", title=f"{primary_group}: MOCA, age, and sex")
    axes[0, 1].axhline(
        MOCA_CATEGORY_BOUNDARY, color="black", linestyle="--", linewidth=1.2,
    )
    axes[0, 1].grid(alpha=0.2)
    axes[0, 1].legend(frameon=False)

    sex_counts = selected["gender"].value_counts().reindex(["F", "M"], fill_value=0)
    axes[1, 0].bar(sex_counts.index, sex_counts.values, color=["#CC79A7", "#009E73"])
    axes[1, 0].set(xlabel="Sex", ylabel="Subjects", title=f"{primary_group} covariate balance")
    for index, value in enumerate(sex_counts.values):
        axes[1, 0].text(index, value, str(value), ha="center", va="bottom")

    selected_features = features.loc[features["group"].eq(primary_group)]
    coverage = (
        selected_features.groupby("feature_id")["value"].count().reindex(dictionary["feature_id"])
    )
    colors = dictionary["family"].map(
        {
            "aperiodic": "#E69F00",
            "ordinal_broadband": "#0072B2",
            "ordinal_band": "#56B4E9",
            "bout_properties": "#D55E00",
            "bout_ordinal": "#CC79A7",
        }
    )
    axes[1, 1].bar(np.arange(len(coverage)), coverage.to_numpy(), color=colors)
    axes[1, 1].axhline(len(selected), color="black", linestyle="--", linewidth=1)
    axes[1, 1].set(
        xlabel="Prespecified EEG feature",
        ylabel="Complete PD subjects",
        title="Feature-wise complete-case coverage",
    )
    axes[1, 1].set_xticks([])
    axes[1, 1].set_ylim(0, len(selected) * 1.08)
    fig.suptitle("Quantitative-behavioral cohort and missing-data audit")
    fig.tight_layout()
    _save(fig, path, dpi)


def plot_family_forest(
    correlations: pd.DataFrame,
    family: str,
    path: Path,
    dpi: int,
) -> None:
    """Plot primary partial-Spearman estimates and bootstrap intervals."""
    selected = correlations.loc[
        correlations["family"].eq(family)
        & correlations["method"].eq("partial_spearman_age_sex")
    ].copy()
    selected = selected.sort_values("estimate")
    positions = np.arange(len(selected))
    significant = selected["fdr_reject"].to_numpy(dtype=bool)
    colors = np.where(significant, "#009E73", "#D55E00")
    fig, axis = plt.subplots(figsize=(10, max(4.5, 0.38 * len(selected))))
    for position, (_, row), color in zip(positions, selected.iterrows(), colors):
        lower_error = max(0.0, float(row["estimate"] - row["ci_lower"]))
        upper_error = max(0.0, float(row["ci_upper"] - row["estimate"]))
        axis.errorbar(
            row["estimate"],
            position,
            xerr=np.asarray([[lower_error], [upper_error]]),
            fmt="o",
            color=color,
            ecolor="0.5",
            capsize=2.5,
        )
    axis.axvline(0.0, color="black", linewidth=0.8)
    axis.set_yticks(positions, selected["feature_label"], fontsize=8)
    axis.set(
        xlabel="Partial Spearman ρ (age/sex adjusted)",
        title=f"MOCA associations — {family.replace('_', ' ')}",
    )
    axis.grid(axis="x", alpha=0.2)
    axis.text(
        0.01,
        0.01,
        "Green: BH-FDR significant within this family; intervals: subject bootstrap",
        transform=axis.transAxes,
        fontsize=8,
        va="bottom",
    )
    fig.tight_layout()
    _save(fig, path, dpi)


def plot_family_heatmap(
    correlations: pd.DataFrame,
    family: str,
    path: Path,
    dpi: int,
) -> None:
    """Compare adjusted and unadjusted estimates across a feature family."""
    selected = correlations.loc[correlations["family"].eq(family)].copy()
    feature_order = selected["feature_id"].drop_duplicates().tolist()
    methods = ["partial_spearman_age_sex", "spearman_unadjusted"]
    matrix = (
        selected.pivot(index="feature_id", columns="method", values="estimate")
        .reindex(index=feature_order, columns=methods)
    )
    labels = (
        selected.drop_duplicates("feature_id").set_index("feature_id")["feature_label"].reindex(feature_order)
    )
    fig, axis = plt.subplots(figsize=(7, max(4.5, 0.34 * len(matrix))))
    image = axis.imshow(matrix.to_numpy(), vmin=-1.0, vmax=1.0, cmap="coolwarm", aspect="auto")
    axis.set_xticks([0, 1], ["Partial Spearman\n(age + sex)", "Spearman\nunadjusted"])
    axis.set_yticks(np.arange(len(labels)), labels, fontsize=8)
    for row in range(len(matrix)):
        for column in range(2):
            value = matrix.iloc[row, column]
            if np.isfinite(value):
                axis.text(column, row, f"{value:.2f}", ha="center", va="center", fontsize=7)
    axis.set_title(f"Adjusted and unadjusted MOCA associations — {family.replace('_', ' ')}")
    fig.colorbar(image, ax=axis, label="Spearman ρ", shrink=0.75)
    fig.tight_layout()
    _save(fig, path, dpi)


def plot_family_scatter_grid(
    features: pd.DataFrame,
    correlations: pd.DataFrame,
    dictionary: pd.DataFrame,
    family: str,
    primary_group: str,
    path: Path,
    dpi: int,
) -> None:
    """Plot raw subject points while labeling the prespecified adjusted estimate."""
    specifications = dictionary.loc[dictionary["family"].eq(family)]
    primary = correlations.loc[
        correlations["family"].eq(family)
        & correlations["method"].eq("partial_spearman_age_sex")
    ].set_index("feature_id")
    columns = min(3, len(specifications)) if len(specifications) <= 18 else 4
    rows = math.ceil(len(specifications) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(4.2 * columns, 3.4 * rows), squeeze=False)
    rng = np.random.default_rng(0)
    for axis, (_, specification) in zip(axes.flat, specifications.iterrows()):
        feature_id = specification["feature_id"]
        selected = features.loc[
            features["feature_id"].eq(feature_id) & features["group"].eq(primary_group)
        ].dropna(subset=["value", "moca"])
        x = selected["value"].to_numpy(dtype=float)
        y = selected["moca"].to_numpy(dtype=float)
        axis.scatter(
            x,
            y + rng.uniform(-0.08, 0.08, len(y)),
            color="#D55E00",
            alpha=0.58,
            s=18,
            edgecolors="none",
        )
        if len(x) >= 2 and not np.allclose(x, x[0]):
            slope, intercept = np.polyfit(x, y, 1)
            line_x = np.linspace(float(x.min()), float(x.max()), 100)
            axis.plot(line_x, intercept + slope * line_x, color="0.25", linestyle="--", linewidth=1)
        row = primary.loc[feature_id]
        displayed_estimate = 0.0 if abs(float(row["estimate"])) < 0.005 else row["estimate"]
        star = " *FDR" if bool(row["fdr_reject"]) else ""
        axis.set_title(
            f"{specification['feature_label']}\npartial ρ={displayed_estimate:.2f}{star}",
            fontsize=9,
        )
        axis.set(xlabel=f"EEG feature ({specification['unit']})", ylabel="MOCA")
        axis.axhline(
            MOCA_CATEGORY_BOUNDARY, color="0.35", linestyle=":", linewidth=1.0
        )
        axis.grid(alpha=0.2)
    for axis in axes.flat[len(specifications) :]:
        axis.set_visible(False)
    title_separator = "\n" if columns == 1 else " — "
    fig.suptitle(
        f"{primary_group} subject-level observations{title_separator}"
        "Raw points; inference uses age/sex-adjusted ranks"
    )
    fig.tight_layout()
    _save(fig, path, dpi)


def _plot_topomap(
    axis: Any,
    values: np.ndarray,
    info: Any,
    vlim: tuple[float, float] = (-1.0, 1.0),
) -> Any | None:
    finite = np.flatnonzero(np.isfinite(values))
    if len(finite) < 4:
        axis.axis("off")
        axis.text(0.5, 0.5, "Insufficient finite electrodes", ha="center", va="center")
        return None
    selected_info = mne.pick_info(info, finite.tolist(), copy=True)
    image, _ = mne.viz.plot_topomap(
        values[finite],
        selected_info,
        axes=axis,
        show=False,
        sensors=True,
        contours=6,
        cmap="coolwarm",
        vlim=vlim,
    )
    return image


def plot_electrode_topomap_pages(
    electrode_correlations: pd.DataFrame,
    dictionary: pd.DataFrame,
    electrode_order: list[str],
    info: Any,
    output_dir: Path,
    dpi: int,
) -> None:
    """Write age/sex-adjusted MOCA-correlation maps grouped by domain and band."""
    groups = dictionary.groupby(["domain", "band"], sort=False)
    for (domain, band), specifications in groups:
        columns = min(5, len(specifications))
        rows = math.ceil(len(specifications) / columns)
        fig, axes = plt.subplots(rows, columns, figsize=(3.5 * columns, 3.5 * rows), squeeze=False)
        for axis, (_, specification) in zip(axes.flat, specifications.iterrows()):
            selected = electrode_correlations.loc[
                electrode_correlations["feature_id"].eq(specification["feature_id"])
            ].set_index("electrode")
            values = np.asarray(
                [selected["estimate"].get(electrode, np.nan) for electrode in electrode_order],
                dtype=float,
            )
            image = _plot_topomap(axis, values, info)
            significant = int(selected["fdr_reject_within_feature"].sum())
            axis.set_title(f"{specification['feature_label']}\nFDR electrodes: {significant}", fontsize=9)
            if image is not None:
                fig.colorbar(image, ax=axis, shrink=0.65)
        for axis in axes.flat[len(specifications) :]:
            axis.set_visible(False)
        fig.suptitle("PD partial Spearman ρ with MOCA, adjusted for age and sex")
        fig.tight_layout()
        _save(fig, output_dir / f"{domain}_{band}_moca_topomaps.png", dpi)


def plot_dimension_sensitivity_heatmaps(
    correlations: pd.DataFrame,
    path: Path,
    dpi: int,
) -> None:
    """Compare adjusted MOCA estimates across D for every ordinal scope."""
    selected = correlations.loc[
        correlations["method"].eq("partial_spearman_age_sex")
    ].copy()
    metrics = [
        metric for metric in DIMENSION_METRIC_TITLES if metric in set(selected["metric"])
    ]
    bands = selected["band"].drop_duplicates().tolist()
    dimensions = sorted(selected["embedding_dimension"].astype(int).unique())
    columns = 3
    rows = math.ceil(len(metrics) / columns)
    fig, axes = plt.subplots(
        rows, columns, figsize=(15.5, 4.7 * rows), sharey=True, squeeze=False
    )
    image = None
    for axis, metric in zip(axes.flat, metrics):
        values = (
            selected.loc[selected["metric"].eq(metric)]
            .pivot(index="band", columns="embedding_dimension", values="estimate")
            .reindex(index=bands, columns=dimensions)
        )
        rejects = (
            selected.loc[selected["metric"].eq(metric)]
            .pivot(index="band", columns="embedding_dimension", values="fdr_reject")
            .reindex(index=bands, columns=dimensions)
        )
        image = axis.imshow(
            values.to_numpy(dtype=float),
            vmin=-0.5,
            vmax=0.5,
            cmap="coolwarm",
            aspect="auto",
        )
        axis.set_xticks(np.arange(len(dimensions)), [f"D={value}" for value in dimensions])
        axis.set_yticks(
            np.arange(len(bands)),
            [band.replace("_", " ").title() for band in bands],
        )
        axis.set_title(DIMENSION_METRIC_TITLES[metric])
        for row in range(len(bands)):
            for column in range(len(dimensions)):
                value = values.iloc[row, column]
                if np.isfinite(value):
                    star = "*" if bool(rejects.iloc[row, column]) else ""
                    axis.text(
                        column,
                        row,
                        f"{value:.2f}{star}",
                        ha="center",
                        va="center",
                        fontsize=8,
                    )
    for axis in axes.flat[len(metrics) :]:
        axis.set_visible(False)
    if image is not None:
        colorbar_axis = fig.add_axes([0.91, 0.12, 0.015, 0.75])
        fig.colorbar(image, cax=colorbar_axis, label="Partial Spearman ρ")
    fig.suptitle(
        "MOCA ordinal analyses across embedding dimensions (age/sex adjusted)\n"
        "* BH-FDR significant within that D's separate 102-feature analysis block"
    )
    fig.subplots_adjust(left=0.10, right=0.88, bottom=0.07, top=0.90, hspace=0.28, wspace=0.18)
    _save(fig, path, dpi)


def plot_dimension_stability_lines(
    correlations: pd.DataFrame,
    path: Path,
    dpi: int,
) -> None:
    """Show effect direction and magnitude stability as embedding D changes."""
    selected = correlations.loc[
        correlations["method"].eq("partial_spearman_age_sex")
    ].copy()
    metrics = [
        metric for metric in DIMENSION_METRIC_TITLES if metric in set(selected["metric"])
    ]
    dimensions = sorted(selected["embedding_dimension"].astype(int).unique())
    bands = selected["band"].drop_duplicates().tolist()
    colors = plt.get_cmap("tab10")(np.linspace(0.0, 0.9, len(bands)))
    columns = 3
    rows = math.ceil(len(metrics) / columns)
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(15.5, 4.3 * rows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    for axis, metric in zip(axes.flat, metrics):
        metric_rows = selected.loc[selected["metric"].eq(metric)]
        for band, color in zip(bands, colors):
            values = (
                metric_rows.loc[metric_rows["band"].eq(band)]
                .set_index("embedding_dimension")["estimate"]
                .reindex(dimensions)
            )
            axis.plot(
                dimensions,
                values.to_numpy(dtype=float),
                marker="o",
                linewidth=1.4,
                color=color,
                label=band.replace("_", " ").title(),
            )
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_title(DIMENSION_METRIC_TITLES[metric])
        axis.set_xticks(dimensions)
        axis.set_xlabel("Embedding dimension D (τ=1)")
        axis.grid(alpha=0.2)
    for row in range(rows):
        axes[row, 0].set_ylabel("Partial Spearman ρ with MOCA")
    for axis in axes.flat[len(metrics) :]:
        axis.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=7, frameon=False)
    fig.suptitle("Stability of age/sex-adjusted ordinal–MOCA associations")
    fig.subplots_adjust(bottom=0.10, top=0.93, hspace=0.30, wspace=0.12)
    _save(fig, path, dpi)
