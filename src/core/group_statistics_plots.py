"""Topographic effect-size and significance maps for group statistics."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence

from core.runtime import configure_runtime

configure_runtime()

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from core.plotting import save_figure


def _safe_token(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()


def plot_electrode_group_statistics(
    statistics: pd.DataFrame,
    info: mne.Info,
    *,
    strata: Sequence[str],
    output_dir: str | Path,
    dpi: int,
    stratum_labels: dict[str, str] | None = None,
) -> list[Path]:
    """Plot standardized PD-Control effects and strict domain-wide FDR maps."""
    required = {
        "electrode",
        "metric",
        "standardized_effect_pd_minus_control",
        "primary_p_fdr_bh_domain",
        "primary_fdr_reject_domain",
        *strata,
    }
    missing = sorted(required - set(statistics))
    if missing:
        raise ValueError(f"Electrode statistics plot is missing columns: {missing}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    channel_order = list(info.ch_names)
    if len(channel_order) < 4:
        return []
    stratum_labels = stratum_labels or {}
    fdr_values = statistics.get("fdr_alpha", pd.Series([0.05])).dropna().unique()
    if len(fdr_values) != 1:
        raise ValueError("Electrode statistics must contain one FDR alpha")
    fdr_alpha = float(fdr_values[0])
    outputs: list[Path] = []
    for metric, metric_table in statistics.groupby("metric", sort=False):
        if strata:
            grouping: str | list[str] = strata[0] if len(strata) == 1 else list(strata)
            panels = list(metric_table.groupby(grouping, sort=False, dropna=False))
        else:
            panels = [("Broadband", metric_table)]
        n_rows = len(panels)
        fig, axes = plt.subplots(
            n_rows,
            2,
            figsize=(8.8, max(3.6, 3.15 * n_rows)),
            squeeze=False,
        )
        all_effects = metric_table["standardized_effect_pd_minus_control"].to_numpy(
            float
        )
        finite_effects = np.abs(all_effects[np.isfinite(all_effects)])
        effect_limit = max(0.25, float(np.quantile(finite_effects, 0.98))) if len(
            finite_effects
        ) else 1.0
        q_values = metric_table["primary_p_fdr_bh_domain"].to_numpy(float)
        finite_q = q_values[np.isfinite(q_values)]
        q_limit = max(
            -np.log10(fdr_alpha),
            float(np.nanmax(-np.log10(np.clip(finite_q, 1e-12, 1.0))))
            if len(finite_q)
            else -np.log10(fdr_alpha),
        )
        for row_index, (keys, selected) in enumerate(panels):
            indexed = selected.set_index("electrode")
            missing_channels = sorted(set(channel_order) - set(indexed.index))
            if missing_channels:
                for column_index in range(2):
                    axes[row_index, column_index].axis("off")
                    axes[row_index, column_index].text(
                        0.5,
                        0.5,
                        "Insufficient complete data\nfor every shared electrode",
                        ha="center",
                        va="center",
                    )
                continue
            effects = indexed.loc[
                channel_order, "standardized_effect_pd_minus_control"
            ].to_numpy(float)
            q = indexed.loc[channel_order, "primary_p_fdr_bh_domain"].to_numpy(float)
            significant = indexed.loc[
                channel_order, "primary_fdr_reject_domain"
            ].to_numpy(bool)
            if not np.all(np.isfinite(effects)) or not np.all(np.isfinite(q)):
                for column_index in range(2):
                    axes[row_index, column_index].axis("off")
                    axes[row_index, column_index].text(
                        0.5,
                        0.5,
                        "Insufficient complete data\nfor every shared electrode",
                        ha="center",
                        va="center",
                    )
                continue
            effect_image, _ = mne.viz.plot_topomap(
                effects,
                info,
                axes=axes[row_index, 0],
                show=False,
                cmap="viridis",
                vlim=(-effect_limit, effect_limit),
                contours=0,
                sensors=True,
                mask=significant,
                mask_params={
                    "marker": "o",
                    "markerfacecolor": "none",
                    "markeredgecolor": "black",
                    "linewidth": 1.1,
                    "markersize": 7,
                },
            )
            q_image, _ = mne.viz.plot_topomap(
                -np.log10(np.clip(q, 1e-12, 1.0)),
                info,
                axes=axes[row_index, 1],
                show=False,
                cmap="viridis",
                vlim=(0.0, q_limit),
                contours=0,
                sensors=True,
                mask=significant,
                mask_params={
                    "marker": "o",
                    "markerfacecolor": "none",
                    "markeredgecolor": "white",
                    "linewidth": 1.1,
                    "markersize": 7,
                },
            )
            if strata:
                key_values = keys if isinstance(keys, tuple) else (keys,)
                raw_label = " / ".join(str(value) for value in key_values)
                panel_label = stratum_labels.get(raw_label, raw_label)
            else:
                panel_label = "Broadband"
            axes[row_index, 0].set_title(f"{panel_label}: standardized effect")
            axes[row_index, 1].set_title(f"{panel_label}: −log10(domain q)")
            fig.colorbar(effect_image, ax=axes[row_index, 0], shrink=0.72)
            fig.colorbar(q_image, ax=axes[row_index, 1], shrink=0.72)
        fig.suptitle(
            f"{metric}: PD − Control "
            f"(rings mark strict domain-wide FDR q<{fdr_alpha:g})",
            fontsize=12,
        )
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.985))
        path = output_dir / f"{_safe_token(str(metric))}_group_statistics.png"
        save_figure(fig, path, dpi)
        outputs.append(path)
    return outputs
