#!/usr/bin/env python3
"""Generate population-level spider plots from global and rhythmicity outputs.

The radar radius is a within-dataset, within-band pooled percentile. This puts
burst properties with different units on a common scale without allowing a
large-valued property (for example, amplitude) to dominate the figure.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind


DATASETS = ["primary", "ds007526-1.0.2", "ds008768-1.0.0", "medication_state"]
DATASET_LABELS = {
    "primary": "Primary",
    "ds007526-1.0.2": "ds007526",
    "ds008768-1.0.0": "ds008768",
    "medication_state": "Medication state",
}
CANONICAL_BANDS = ["theta", "alpha", "beta", "gamma"]
LAVI_BANDS = ["theta_1", "alpha_1", "beta_1", "beta_2", "gamma_1"]
GROUP_ORDER = ["Control", "PD", "PD_OFF", "PD_ON"]
GROUP_COLORS = {
    "Control": "#666666",
    "PD": "#d95f02",
    "PD_OFF": "#7570b3",
    "PD_ON": "#1b9e77",
}

GLOBAL_PROPERTIES = {
    "Burst\ncount": "n_bouts",
    "Occupancy": "oscillatory_occupancy",
    "Bursts /\nminute": "bouts_per_minute",
    "Duration": "duration_mean_s",
    "Amplitude": "amplitude_mean",
    "Cycles": "cycles_mean",
}
ABBA_PROPERTIES = {
    "Bursts /\nminute": "bursts_per_minute",
    "Occupancy": "occupancy_percent",
    "Duration": "duration_s",
    "Cycles": "cycles",
    "Peak\namplitude": "peak_amplitude_uv",
    "Relative\namplitude": "relative_amplitude_db",
}
LAVI_PROPERTIES = {
    "Bout\ncount": "lavi_high_bout_count",
    "Occupancy": "lavi_high_occupancy",
    "Bout density": "lavi_high_bout_density_hz",
    "Width": "lavi_high_bout_width_hz_mean",
    "Peak excess": "lavi_high_bout_peak_excess_mean",
    "Peak\nfrequency": "lavi_high_bout_peak_frequency_hz_mean",
}


def _load_global(global_root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for dataset in DATASETS:
        path = global_root / "metrics" / dataset / "subject_features.csv.gz"
        table = pd.read_csv(path, low_memory=False)
        for band in CANONICAL_BANDS:
            renamed = {
                f"bout__{band}__{column}": column
                for column in GLOBAL_PROPERTIES.values()
            }
            selected = table[["participant_id", "group", *renamed]].rename(columns=renamed)
            selected.insert(0, "band", band)
            selected.insert(0, "dataset", dataset)
            frames.append(selected)
    return pd.concat(frames, ignore_index=True)


def _load_abba(rhythmicity_root: Path) -> pd.DataFrame:
    path = rhythmicity_root / "metrics" / "abba_burst_participant_metrics.csv.gz"
    table = pd.read_csv(path, low_memory=False)
    # band_name is the actual group-specific, LAVI-derived ABBA interval
    # (for example beta_1 versus beta_2), not merely its canonical parent.
    table = table.loc[table["band_name"].isin(LAVI_BANDS)].copy()
    table["band"] = table["band_name"].astype(str)
    columns = ["dataset", "participant_id", "group", "band", *ABBA_PROPERTIES.values()]
    return table[columns]


def _load_lavi(rhythmicity_root: Path) -> pd.DataFrame:
    path = rhythmicity_root / "metrics" / "lavi_burst_features_participant.csv.gz"
    table = pd.read_csv(path, low_memory=False)
    table = table.loc[table["band"].isin(CANONICAL_BANDS)]
    return table[["dataset", "participant_id", "group", "band", *LAVI_PROPERTIES.values()]]


def _summarize(
    table: pd.DataFrame,
    source: str,
    properties: dict[str, str],
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for (dataset, band), frame in table.groupby(["dataset", "band"], sort=False):
        ranked = frame.copy()
        for column in properties.values():
            values = pd.to_numeric(ranked[column], errors="coerce")
            ranked[column] = values.rank(method="average", pct=True) * 100.0
        for group, group_frame in frame.groupby("group", sort=False):
            ranked_group = ranked.loc[ranked["group"].eq(group)]
            for label, column in properties.items():
                raw = pd.to_numeric(group_frame[column], errors="coerce")
                percentile = pd.to_numeric(ranked_group[column], errors="coerce")
                records.append(
                    {
                        "source": source,
                        "dataset": dataset,
                        "band": band,
                        "group": group,
                        "property": label.replace("\n", " "),
                        "column": column,
                        "n": int(raw.notna().sum()),
                        "raw_median": float(raw.median()) if raw.notna().any() else np.nan,
                        "pooled_percentile_median": (
                            float(percentile.median()) if percentile.notna().any() else np.nan
                        ),
                        "control_comparison_p": np.nan,
                        "control_comparison_q_fdr_bh": np.nan,
                        "significant_vs_control_fdr": False,
                    }
                )
    summary = pd.DataFrame.from_records(records)
    # Test the same participant-level quantities displayed by the spider plot.
    # FDR is controlled across the properties within each dataset/band/group
    # profile. Medication OFF and ON are each compared independently to Control.
    for (dataset, band), frame in table.groupby(["dataset", "band"], sort=False):
        control = frame.loc[frame["group"].eq("Control")]
        if control.empty:
            continue
        for group in GROUP_ORDER[1:]:
            disease = frame.loc[frame["group"].eq(group)]
            if disease.empty:
                continue
            p_values: list[float] = []
            columns = list(properties.values())
            for column in columns:
                left = pd.to_numeric(control[column], errors="coerce").dropna().to_numpy(float)
                right = pd.to_numeric(disease[column], errors="coerce").dropna().to_numpy(float)
                if len(left) < 2 or len(right) < 2:
                    p_values.append(np.nan)
                else:
                    p_values.append(float(ttest_ind(left, right, equal_var=False).pvalue))
            q_values = _bh_fdr(np.asarray(p_values, dtype=float))
            for column, p_value, q_value in zip(columns, p_values, q_values):
                mask = (
                    summary["dataset"].eq(dataset)
                    & summary["band"].eq(band)
                    & summary["group"].eq(group)
                    & summary["column"].eq(column)
                )
                summary.loc[mask, "control_comparison_p"] = p_value
                summary.loc[mask, "control_comparison_q_fdr_bh"] = q_value
                summary.loc[mask, "significant_vs_control_fdr"] = bool(
                    np.isfinite(q_value) and q_value < 0.05
                )
    return summary


def _bh_fdr(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values, preserving missing entries."""
    adjusted = np.full(p_values.shape, np.nan, dtype=float)
    finite_indices = np.flatnonzero(np.isfinite(p_values))
    if not len(finite_indices):
        return adjusted
    finite = p_values[finite_indices]
    order = np.argsort(finite)
    ranked = finite[order]
    candidate = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    candidate = np.minimum.accumulate(candidate[::-1])[::-1]
    restored = np.empty_like(candidate)
    restored[order] = np.minimum(candidate, 1.0)
    adjusted[finite_indices] = restored
    return adjusted


def _plot_band(
    summary: pd.DataFrame,
    source: str,
    source_title: str,
    band: str,
    properties: dict[str, str],
    output_dir: Path,
) -> None:
    labels = list(properties)
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    closed_angles = np.r_[angles, angles[0]]
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12.5, 11.5),
        subplot_kw={"projection": "polar"},
    )
    legend_groups: set[str] = set()
    for axis, dataset in zip(axes.flat, DATASETS):
        panel = summary.loc[
            summary["dataset"].eq(dataset) & summary["band"].eq(band)
        ]
        axis.set_theta_offset(np.pi / 2)
        axis.set_theta_direction(-1)
        axis.set_xticks(angles, labels, fontsize=9)
        axis.set_ylim(0, 100)
        axis.set_yticks([25, 50, 75, 100], ["25", "50", "75", "100"], fontsize=7)
        axis.set_rlabel_position(8)
        axis.grid(color="#aaaaaa", alpha=0.35)
        axis.spines["polar"].set_color("#bbbbbb")
        for group in GROUP_ORDER:
            group_rows = panel.loc[panel["group"].eq(group)].set_index("column")
            if group_rows.empty:
                continue
            values = np.array(
                [group_rows["pooled_percentile_median"].get(column, np.nan) for column in properties.values()],
                dtype=float,
            )
            if not np.isfinite(values).all():
                continue
            closed_values = np.r_[values, values[0]]
            axis.plot(
                closed_angles,
                closed_values,
                color=GROUP_COLORS[group],
                linewidth=2.0,
                marker="o",
                markersize=3.5,
                label=group,
            )
            axis.fill(closed_angles, closed_values, color=GROUP_COLORS[group], alpha=0.07)
            significant = np.array(
                [
                    bool(group_rows["significant_vs_control_fdr"].get(column, False))
                    for column in properties.values()
                ]
            )
            for angle, value, is_significant in zip(angles, values, significant):
                if is_significant:
                    axis.text(
                        angle,
                        min(value + 7.0, 98.0),
                        "*",
                        color=GROUP_COLORS[group],
                        fontsize=15,
                        fontweight="bold",
                        ha="center",
                        va="center",
                        zorder=6,
                    )
            legend_groups.add(group)
        counts = (
            panel.groupby("group", sort=False)["n"].min().astype(int).to_dict()
            if not panel.empty
            else {}
        )
        count_text = ", ".join(
            f"{group} n={counts[group]}" for group in GROUP_ORDER if group in counts
        )
        axis.set_title(
            f"{DATASET_LABELS[dataset]}\n{count_text or 'No data'}",
            fontsize=11,
            fontweight="bold",
            pad=24,
        )
    handles = [
        plt.Line2D([0], [0], color=GROUP_COLORS[group], linewidth=2.5, marker="o", label=group)
        for group in GROUP_ORDER
        if group in legend_groups
    ]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False, bbox_to_anchor=(0.5, 0.945))
    fig.suptitle(
        f"{source_title}: {band.replace('_', ' ').title()} burst profile",
        fontsize=17,
        fontweight="bold",
        y=0.992,
    )
    fig.text(
        0.5,
        0.018,
        "Radius = group median of participant ranks pooled within each dataset and band. "
        "50 is the pooled participant median; farther out means a higher value. "
        "A colored * marks that group versus Control at BH-FDR q < .05 across spokes.",
        ha="center",
        fontsize=9,
        color="#444444",
    )
    fig.subplots_adjust(left=0.06, right=0.94, top=0.88, bottom=0.07, hspace=0.36, wspace=0.24)
    stem = output_dir / f"{source}_{band}_population_spider"
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--global-root", type=Path, default=Path("outputs/global"))
    parser.add_argument("--rhythmicity-root", type=Path, default=Path("outputs/rhythmicity"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/general_summmary"))
    parser.add_argument(
        "--canonical-bands",
        nargs="+",
        choices=CANONICAL_BANDS,
        default=CANONICAL_BANDS,
    )
    parser.add_argument(
        "--lavi-bands",
        nargs="+",
        choices=LAVI_BANDS,
        default=LAVI_BANDS,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    specifications = [
        (
            "global_bouts_canonical_bands",
            "Global temporal bouts · fixed canonical bands",
            _load_global(args.global_root),
            GLOBAL_PROPERTIES,
            args.canonical_bands,
        ),
        (
            "rhythmicity_abba_lavi_bands",
            "Rhythmicity ABBA bursts · group-specific LAVI bands",
            _load_abba(args.rhythmicity_root),
            ABBA_PROPERTIES,
            args.lavi_bands,
        ),
        (
            "rhythmicity_lavi_canonical_bands",
            "Rhythmicity LAVI-high spectral bouts · canonical parent bands",
            _load_lavi(args.rhythmicity_root),
            LAVI_PROPERTIES,
            args.canonical_bands,
        ),
    ]
    summaries = []
    for source, title, table, properties, bands in specifications:
        summary = _summarize(table, source, properties)
        summaries.append(summary)
        for band in bands:
            _plot_band(summary, source, title, band, properties, args.output_dir)
    pd.concat(summaries, ignore_index=True).to_csv(
        args.output_dir / "spider_plot_group_summaries.csv", index=False
    )
    print(f"Wrote spider plots and summaries to {args.output_dir}")


if __name__ == "__main__":
    main()
