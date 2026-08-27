"""Figures for the prespecified eight-electrode sensitivity battery."""

from __future__ import annotations

import math
import re
import textwrap
from pathlib import Path

from src.runtime import configure_runtime

configure_runtime()

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from src.plotting import save_figure


def _token(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()


def plot_selection(info: mne.Info, output: Path, dpi: int) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    mne.viz.plot_sensors(info, kind="topomap", show_names=True, axes=ax, show=False)
    ax.set_title("Prespecified eight-electrode sensitivity subset")
    fig.tight_layout()
    save_figure(fig, output, dpi)


def plot_effect_pages(
    statistics: pd.DataFrame, output_dir: Path, dpi: int, *, rows_per_page: int
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for domain, table in statistics.groupby("domain", sort=False):
        table = table.sort_values("standardized_effect_pd_minus_control")
        for start in range(0, len(table), rows_per_page):
            page = table.iloc[start : start + rows_per_page]
            y = np.arange(len(page))
            colors = np.where(page["primary_fdr_reject_domain"], "#D55E00", "0.4")
            fig, ax = plt.subplots(figsize=(11, max(4.0, 0.29 * len(page) + 1.8)))
            ax.scatter(page["standardized_effect_pd_minus_control"], y, c=colors, s=34)
            ax.axvline(0, color="0.15", linewidth=1)
            ax.set_yticks(
                y,
                ["\n".join(textwrap.wrap(value, 50)) for value in page["feature_label"]],
                fontsize=8,
            )
            ax.set_xlabel("Standardized PD − Control effect")
            ax.set_title(f"Eight-electrode subject aggregate: {domain}")
            ax.grid(axis="x", alpha=0.2)
            fig.tight_layout()
            path = output_dir / f"{_token(domain)}_page_{start // rows_per_page + 1:03d}.png"
            save_figure(fig, path, dpi)
            outputs.append(path)
    return outputs


def plot_group_distribution_pages(
    values: pd.DataFrame,
    statistics: pd.DataFrame,
    output_dir: Path,
    dpi: int,
    *,
    features_per_page: int = 9,
) -> list[Path]:
    """Show every subject aggregate, with paired lines in the matched cohort."""
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    lookup = statistics.set_index("feature_id")
    rng = np.random.default_rng(20260826)
    for domain, domain_values in values.groupby("domain", sort=False):
        feature_ids = domain_values["feature_id"].drop_duplicates().tolist()
        for start in range(0, len(feature_ids), features_per_page):
            page = feature_ids[start : start + features_per_page]
            fig, axes = plt.subplots(
                math.ceil(len(page) / 3), 3,
                figsize=(13.5, 3.7 * math.ceil(len(page) / 3)), squeeze=False,
            )
            for ax, feature_id in zip(axes.flat, page):
                table = domain_values.loc[
                    domain_values["feature_id"].eq(feature_id)
                ].dropna(subset=["value"])
                arrays = [
                    table.loc[table["group"].eq(group), "value"].to_numpy(float)
                    for group in ("Control", "PD")
                ]
                if all(len(array) for array in arrays):
                    violin = ax.violinplot(arrays, positions=[0, 1], showextrema=False)
                    for body, color in zip(violin["bodies"], ("#0072B2", "#D55E00")):
                        body.set_facecolor(color)
                        body.set_alpha(0.22)
                if "match_pair_id" in table:
                    paired = table.pivot(
                        index="match_pair_id", columns="group", values="value"
                    ).dropna()
                    for _, row in paired.iterrows():
                        ax.plot([0, 1], [row["Control"], row["PD"]], color="0.75", linewidth=0.5)
                for position, (group, color) in enumerate(
                    (("Control", "#0072B2"), ("PD", "#D55E00"))
                ):
                    selected = table.loc[table["group"].eq(group), "value"].to_numpy(float)
                    jitter = rng.uniform(-0.09, 0.09, len(selected))
                    ax.scatter(position + jitter, selected, s=11, alpha=0.58, color=color)
                result = lookup.loc[feature_id]
                q = result["primary_p_fdr_bh_domain"]
                q_text = "NA" if not np.isfinite(q) else f"{q:.3g}"
                star = " ★" if bool(result["primary_fdr_reject_domain"]) else ""
                ax.text(
                    0.02, 0.98, f"q={q_text}{star}", transform=ax.transAxes,
                    ha="left", va="top", fontsize=8,
                    bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
                )
                ax.set_xticks([0, 1], ["Control", "PD"])
                ax.set_title("\n".join(textwrap.wrap(str(result["feature_label"]), 36)), fontsize=9)
                ax.grid(axis="y", alpha=0.15)
            for ax in axes.flat[len(page) :]:
                ax.axis("off")
            fig.suptitle(
                f"Eight-electrode subject distributions: {domain} (★ domain FDR)",
                fontsize=13,
            )
            fig.tight_layout(rect=(0, 0, 1, 0.98))
            path = output_dir / f"{_token(domain)}_page_{start // features_per_page + 1:03d}.png"
            save_figure(fig, path, dpi)
            outputs.append(path)
    return outputs


def plot_electrode_heatmaps(
    statistics: pd.DataFrame, output_dir: Path, dpi: int, *, rows_per_page: int = 35
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for domain, table in statistics.groupby("domain", sort=False):
        features = table["feature_id"].drop_duplicates().tolist()
        for start in range(0, len(features), rows_per_page):
            page_features = features[start : start + rows_per_page]
            page = table.loc[table["feature_id"].isin(page_features)]
            matrix = page.pivot(
                index="feature_label", columns="electrode",
                values="standardized_effect_pd_minus_control",
            )
            matrix = matrix.reindex(
                [_label for feature in page_features for _label in
                 page.loc[page["feature_id"].eq(feature), "feature_label"].head(1)]
            )
            limit = np.nanmax(np.abs(matrix.to_numpy(float)))
            limit = 1.0 if not np.isfinite(limit) or limit == 0 else limit
            fig, ax = plt.subplots(figsize=(10.5, max(4.2, 0.3 * len(matrix) + 2.0)))
            image = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=-limit, vmax=limit)
            reject = page.pivot(
                index="feature_label", columns="electrode",
                values="primary_fdr_reject_domain",
            ).reindex(index=matrix.index, columns=matrix.columns)
            for row_index, feature_label in enumerate(matrix.index):
                for column_index, electrode in enumerate(matrix.columns):
                    if bool(reject.loc[feature_label, electrode]):
                        ax.text(column_index, row_index, "★", ha="center", va="center", fontsize=7)
            ax.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=45, ha="right")
            ax.set_yticks(
                range(len(matrix)),
                ["\n".join(textwrap.wrap(value, 46)) for value in matrix.index],
                fontsize=7,
            )
            ax.set_title(f"Exploratory electrode effects: {domain} (★ strict domain FDR)")
            fig.colorbar(image, ax=ax, label="Standardized PD − Control effect")
            fig.tight_layout()
            path = output_dir / f"{_token(domain)}_page_{start // rows_per_page + 1:03d}.png"
            save_figure(fig, path, dpi)
            outputs.append(path)
    return outputs
