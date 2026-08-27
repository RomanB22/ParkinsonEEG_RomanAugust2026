"""Transparent figures for the whole-head severity-axis analysis."""

from __future__ import annotations

import math
import re
import textwrap
from pathlib import Path
from typing import Any, Sequence

from src.runtime import configure_runtime

configure_runtime()

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from src.plotting import save_figure


MOCA_CATEGORY_BOUNDARY = 25.5


def _token(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()


def plot_electrode_selection(info: mne.Info, output: Path, dpi: int) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    mne.viz.plot_sensors(info, kind="topomap", show_names=True, axes=ax, show=False)
    ax.set_title("All cohort-shared disease-progression electrodes")
    fig.tight_layout()
    save_figure(fig, output, dpi)


def plot_clinical_axes(
    cohort: pd.DataFrame,
    association: pd.DataFrame,
    output: Path,
    dpi: int,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    for gender, marker, label in (("F", "o", "Female"), ("M", "^", "Male")):
        selected = cohort.loc[cohort["gender"].eq(gender)]
        image = ax.scatter(
            selected["updrs"], selected["moca"], c=selected["age_years"],
            cmap="viridis", marker=marker, edgecolor="white", linewidth=0.5,
            s=52, alpha=0.85, label=label,
        )
    coefficients = np.polyfit(cohort["updrs"], cohort["moca"], 1)
    x = np.linspace(cohort["updrs"].min(), cohort["updrs"].max(), 100)
    ax.plot(x, np.polyval(coefficients, x), color="0.2", linewidth=1.5)
    ax.axhline(
        MOCA_CATEGORY_BOUNDARY, color="0.25", linestyle="--", linewidth=1.2,
        label="Impaired <26 | Normal 26–30",
    )
    adjusted = association.loc[
        association["method"].eq("partial_spearman_age_sex")
    ].iloc[0]
    ax.text(
        0.02, 0.98,
        f"Age/sex-adjusted partial ρ={adjusted['estimate']:.3f}\n"
        f"p={adjusted['p_value']:.3g}; n={int(adjusted['n_subjects'])}",
        transform=ax.transAxes, ha="left", va="top",
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "0.8"},
    )
    ax.set_xlabel("UPDRS (higher = worse motor severity)")
    ax.set_ylabel("MOCA (higher = better cognition)")
    ax.set_title("Clinical severity axes within Parkinson disease")
    ax.legend(frameon=False, loc="lower left")
    fig.colorbar(image, ax=ax, label="Age (years)")
    fig.tight_layout()
    save_figure(fig, output, dpi)


def plot_feature_scatter_pages(
    features: pd.DataFrame,
    dictionary: pd.DataFrame,
    correlations: pd.DataFrame,
    *,
    outcome: str,
    output_dir: Path,
    dpi: int,
    features_per_page: int,
    color: str,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    adjusted = correlations.loc[
        correlations["outcome"].eq(outcome)
        & correlations["method"].eq("partial_spearman_age_sex")
    ].set_index("feature_id")
    outputs: list[Path] = []
    specs = dictionary.reset_index(drop=True)
    for page_start in range(0, len(specs), features_per_page):
        page = specs.iloc[page_start : page_start + features_per_page]
        n_columns = 3
        n_rows = math.ceil(len(page) / n_columns)
        fig, axes = plt.subplots(
            n_rows, n_columns, figsize=(14.0, 4.0 * n_rows), squeeze=False
        )
        for axis, (_, specification) in zip(axes.flat, page.iterrows()):
            feature_id = str(specification["feature_id"])
            table = features.loc[features["feature_id"].eq(feature_id)].dropna(
                subset=["value", outcome]
            )
            for gender, marker in (("F", "o"), ("M", "^")):
                selected = table.loc[table["gender"].eq(gender)]
                axis.scatter(
                    selected[outcome], selected["value"], color=color, marker=marker,
                    edgecolor="white", linewidth=0.4, s=34, alpha=0.72,
                )
            if len(table) >= 3 and table[outcome].nunique() > 1:
                coefficients = np.polyfit(table[outcome], table["value"], 1)
                x = np.linspace(table[outcome].min(), table[outcome].max(), 100)
                axis.plot(x, np.polyval(coefficients, x), color="0.2", linewidth=1.2)
            if outcome == "moca":
                axis.axvline(
                    MOCA_CATEGORY_BOUNDARY, color="0.35", linestyle=":", linewidth=1.0
                )
            result = adjusted.loc[feature_id]
            q_text = "NA" if not np.isfinite(result["p_fdr_bh"]) else f"{result['p_fdr_bh']:.3g}"
            marker = " ★" if bool(result["fdr_reject"]) else ""
            axis.text(
                0.02, 0.98,
                f"partial ρ={result['estimate']:.3f}\nq={q_text}; n={int(result['n_subjects'])}{marker}",
                transform=axis.transAxes, ha="left", va="top", fontsize=8,
                bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none"},
            )
            axis.set_title("\n".join(textwrap.wrap(str(specification["feature_label"]), 34)), fontsize=9)
            axis.set_xlabel(outcome.upper())
            axis.set_ylabel(str(specification["unit"]))
            axis.grid(alpha=0.15)
        for axis in axes.flat[len(page) :]:
            axis.axis("off")
        fig.suptitle(
            f"Whole-head shared-electrode EEG quantities along {outcome.upper()} "
            "(circles=female, triangles=male; ★ family FDR)",
            fontsize=13,
        )
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.985))
        page_number = page_start // features_per_page + 1
        path = output_dir / f"{outcome}_scatter_page_{page_number:03d}.png"
        save_figure(fig, path, dpi)
        outputs.append(path)
    return outputs


def plot_forest_pages(
    correlations: pd.DataFrame,
    *,
    outcome: str,
    output_dir: Path,
    dpi: int,
    significant_color: str,
    rows_per_page: int = 35,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = correlations.loc[
        correlations["outcome"].eq(outcome)
        & correlations["method"].eq("partial_spearman_age_sex")
    ].copy()
    outputs: list[Path] = []
    for family, family_table in selected.groupby("family", sort=False):
        family_table = family_table.sort_values("estimate")
        for page_start in range(0, len(family_table), rows_per_page):
            page = family_table.iloc[page_start : page_start + rows_per_page]
            y = np.arange(len(page))
            colors = np.where(page["fdr_reject"], significant_color, "0.35")
            fig, ax = plt.subplots(figsize=(11.0, max(4.0, 0.31 * len(page) + 1.7)))
            finite = page[["estimate", "ci_lower", "ci_upper"]].notna().all(axis=1)
            plotted = page.loc[finite]
            plotted_y = y[finite.to_numpy()]
            plotted_colors = colors[finite.to_numpy()]
            for (_, row), row_y, row_color in zip(
                plotted.iterrows(), plotted_y, plotted_colors
            ):
                ax.errorbar(
                    row["estimate"],
                    row_y,
                    xerr=np.asarray(
                        [[
                            row["estimate"] - row["ci_lower"],
                            row["ci_upper"] - row["estimate"],
                        ]]
                    ).T,
                    fmt="none",
                    ecolor=row_color,
                    elinewidth=1.2,
                    capsize=2,
                )
            ax.scatter(page["estimate"], y, c=colors, s=30, zorder=3)
            labels = ["\n".join(textwrap.wrap(str(value), 52)) for value in page["feature_label"]]
            ax.set_yticks(y, labels, fontsize=8)
            ax.axvline(0.0, color="0.2", linewidth=1.0)
            ax.set_xlim(-1.02, 1.02)
            ax.set_xlabel(f"Age/sex-adjusted partial Spearman ρ with {outcome.upper()}")
            ax.set_title(f"{family}: whole-head severity associations")
            ax.grid(axis="x", alpha=0.2)
            fig.tight_layout()
            page_number = page_start // rows_per_page + 1
            path = output_dir / f"{_token(str(family))}_forest_page_{page_number:03d}.png"
            save_figure(fig, path, dpi)
            outputs.append(path)
    return outputs
