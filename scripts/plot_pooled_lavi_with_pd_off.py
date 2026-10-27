#!/usr/bin/env python
"""Add descriptive PD-OFF violins to the pooled LAVI/eBOSC figure sets."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_pooled_lavi_determined_band_analysis import (
    BAND_NAMES,
    _band_definitions,
    _ebosc_worker,
    _lavi_tables,
)
from run_pooled_lavi_ebosc_analysis import (
    EBOSC_FEATURES,
    LAVI_BURST_FEATURES,
    LAVI_FEATURES,
    _stars,
)


GROUPS = ["Control", "PD + PD-ON", "PD-OFF"]
COLORS = {"Control": "#7f7f7f", "PD + PD-ON": "#d95f02", "PD-OFF": "#7570b3"}
CANONICAL_BANDS = ["theta", "alpha", "beta", "gamma"]
Annotation = tuple[str, str, str, str]


def _groups_with_off(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.loc[frame["group"].isin(["Control", "PD", "PD_ON", "PD_OFF"])].copy()
    result["group"] = result["group"].map({
        "Control": "Control", "PD": "PD + PD-ON", "PD_ON": "PD + PD-ON", "PD_OFF": "PD-OFF",
    })
    return result


def _plot(
    frame: pd.DataFrame,
    features: dict[str, str],
    bands: list[str],
    title: str,
    output: Path,
    annotations: dict[tuple[str, str], list[Annotation]] | None = None,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(17, 9), squeeze=False)
    rng = np.random.default_rng(20260913)
    offsets = {"Control": -0.25, "PD + PD-ON": 0.0, "PD-OFF": 0.25}
    for axis, (feature, label) in zip(axes.flat, features.items()):
        for band_index, band in enumerate(bands):
            for group in GROUPS:
                values = pd.to_numeric(
                    frame.loc[frame["band"].eq(band) & frame["group"].eq(group), feature], errors="coerce"
                ).dropna().to_numpy(float)
                position = band_index + offsets[group]
                if len(values) > 1 and np.ptp(values) > 0:
                    violin = axis.violinplot([values], positions=[position], widths=0.23, showmedians=True, showextrema=False)
                    body = violin["bodies"][0]
                    body.set_facecolor(COLORS[group]); body.set_edgecolor("white"); body.set_alpha(0.72)
                    violin["cmedians"].set_color("#202020"); violin["cmedians"].set_linewidth(1.0)
                if len(values):
                    keep = np.arange(len(values)) if len(values) <= 180 else rng.choice(len(values), 180, replace=False)
                    axis.scatter(
                        position + rng.uniform(-0.035, 0.035, len(keep)), values[keep], s=7,
                        color=COLORS[group], alpha=0.42, linewidth=0, rasterized=True,
                    )
        axis.set_title(label, fontsize=11, fontweight="bold")
        axis.set_xticks(
            range(len(bands)), [band.replace("_", " ").title() for band in bands], rotation=20, ha="right"
        )
        axis.grid(axis="y", alpha=0.20)
        axis.spines[["top", "right"]].set_visible(False)
        maximum_level = -1
        low, high = axis.get_ylim()
        span = high - low
        for band in bands:
            band_annotations = (annotations or {}).get((feature, band), [])
            if not band_annotations:
                continue
            band_index = bands.index(band)
            for level, (left_group, right_group, stars, detail) in enumerate(band_annotations):
                maximum_level = max(maximum_level, level)
                x1 = band_index + offsets[left_group]
                x2 = band_index + offsets[right_group]
                y = high + (0.025 + 0.085 * level) * span
                cap = 0.016 * span
                color = COLORS[right_group]
                axis.plot([x1, x1, x2, x2], [y, y + cap, y + cap, y], color=color, linewidth=1.15)
                axis.text(
                    (x1 + x2) / 2, y + 1.15 * cap, stars,
                    ha="center", va="bottom", fontsize=11, color=color,
                )
                if detail:
                    axis.text(
                        (x1 + x2) / 2, y - 0.45 * cap, detail,
                        ha="center", va="top", fontsize=7, color=color,
                    )
        if maximum_level >= 0:
            axis.set_ylim(low, high + (0.12 + 0.085 * maximum_level) * span)
    handles = [Patch(facecolor=COLORS[group], edgecolor="white", label=group) for group in GROUPS]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.952), ncol=3, frameon=False)
    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.995)
    fig.text(
        0.5, 0.902,
        "Orange brackets: Control vs PD + PD-ON; purple: Control vs PD-OFF; stars use BH-FDR within each analysis family (no OFF vs ON result survived)",
        ha="center", fontsize=9, color="#444444",
    )
    fig.tight_layout(rect=(0.02, 0.02, 0.99, 0.86))
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _significant_annotations(statistics: pd.DataFrame, family: str) -> dict[tuple[str, str], list[Annotation]]:
    annotations: dict[tuple[str, str], list[Annotation]] = {}
    selected = statistics.loc[statistics["family"].eq(family) & statistics["significant_fdr"]]
    for row in selected.itertuples(index=False):
        annotations.setdefault((str(row.feature), str(row.band)), []).append(
            ("Control", "PD + PD-ON", _stars(float(row.q_fdr_bh)), "")
        )
    return annotations


def _append_annotation(
    annotations: dict[tuple[str, str], list[Annotation]],
    feature: str,
    band: str,
    annotation: Annotation,
) -> None:
    annotations.setdefault((feature, band), []).append(annotation)


def _pd_off_ebosc(output_root: Path, definitions: pd.DataFrame, config_path: Path) -> pd.DataFrame:
    pd_definition = definitions.loc[definitions["group"].eq("PD + PD-ON")]
    bands = {
        row.band: (float(row.ebosc_low_hz), float(row.ebosc_high_hz))
        for row in pd_definition.itertuples(index=False)
    }
    canonical = pd.read_csv("outputs/global/canonical/medication_state/recordings.csv.gz", low_memory=False)
    canonical = canonical.loc[canonical["group"].eq("PD_OFF")]
    paths: list[str] = []
    for row in canonical.to_dict("records"):
        row["group"] = "PD-OFF"
        cache = output_root / "intermediate" / "ebosc_lavi_determined_bands_pd_off" / f"{row['recording_id']}_pd_case_limits.csv.gz"
        paths.append(_ebosc_worker((row, bands, str(config_path), str(cache))))
    electrode = pd.concat([pd.read_csv(path, low_memory=False) for path in paths], ignore_index=True)
    rows: list[pd.DataFrame] = []
    for band in BAND_NAMES:
        rename = {
            f"bout__{band}__n_bouts": "n_bouts",
            f"bout__{band}__oscillatory_occupancy": "oscillatory_occupancy",
            f"bout__{band}__bouts_per_minute": "bouts_per_minute",
            f"bout__{band}__duration_mean_s": "duration_mean_s",
            f"bout__{band}__amplitude_mean": "amplitude_mean",
            f"bout__{band}__cycles_mean": "cycles_mean",
        }
        subset = electrode[["dataset_id", "participant_id", "group", "electrode"] + list(rename)].rename(columns=rename)
        subset["band"] = band
        rows.append(subset)
    long = pd.concat(rows, ignore_index=True).rename(columns={"dataset_id": "dataset"})
    return long.groupby(["dataset", "participant_id", "group", "band"], as_index=False)[list(EBOSC_FEATURES)].mean(numeric_only=True)


def run(rhythmicity_root: Path, output_root: Path, config_path: Path) -> None:
    figure_root = output_root / "figures" / "with_pd_off"
    figure_root.mkdir(parents=True, exist_ok=True)
    canonical = _groups_with_off(pd.read_csv(rhythmicity_root / "metrics" / "participant_rhythmicity.csv.gz"))
    canonical_frequency_bouts = _groups_with_off(
        pd.read_csv(rhythmicity_root / "metrics" / "lavi_burst_features_participant.csv.gz")
    )
    canonical_statistics = pd.read_csv(output_root / "statistics" / "pooled_group_comparisons.csv")
    _plot(
        canonical, LAVI_FEATURES, CANONICAL_BANDS, "Pooled LAVI with PD-OFF",
        figure_root / "pooled_lavi_with_pd_off.png",
        _significant_annotations(canonical_statistics, "LAVI"),
    )
    _plot(
        canonical_frequency_bouts, LAVI_BURST_FEATURES, CANONICAL_BANDS,
        "Pooled LAVI frequency bouts with PD-OFF",
        figure_root / "pooled_lavi_frequency_bursts_with_pd_off.png",
        _significant_annotations(canonical_statistics, "LAVI frequency bouts"),
    )
    _plot(
        canonical, EBOSC_FEATURES, CANONICAL_BANDS,
        "Pooled canonical-band eBOSC bursts with PD-OFF",
        figure_root / "pooled_ebosc_bursts_with_pd_off.png",
        _significant_annotations(canonical_statistics, "eBOSC temporal bouts"),
    )

    definitions = _band_definitions(output_root / "statistics" / "pooled_lavi_abba_intervals.csv")
    off_definitions = definitions.loc[definitions["group"].eq("PD + PD-ON")].copy()
    off_definitions["group"] = "PD-OFF"
    definitions_with_off = pd.concat([definitions, off_definitions], ignore_index=True)
    _, custom_lavi = _lavi_tables(rhythmicity_root, definitions_with_off, include_pd_off=True)
    custom_statistics = pd.read_csv(output_root / "statistics" / "lavi_determined_band_group_comparisons.csv")
    _plot(
        custom_lavi, LAVI_FEATURES, BAND_NAMES,
        "Pooled LAVI-determined-band LAVI with PD-OFF",
        figure_root / "lavi_determined_band_lavi_with_pd_off.png",
        _significant_annotations(custom_statistics, "LAVI in LAVI-determined bands"),
    )
    frequency_annotations = _significant_annotations(
        custom_statistics, "LAVI frequency bouts in LAVI-determined bands"
    )
    _append_annotation(
        frequency_annotations,
        "lavi_high_bout_count",
        "theta_low",
        ("Control", "PD-OFF", "**", "q = 0.009"),
    )
    _plot(
        custom_lavi,
        LAVI_BURST_FEATURES,
        BAND_NAMES,
        "Pooled LAVI-determined-band frequency bouts with PD-OFF",
        figure_root / "lavi_determined_band_frequency_bursts_with_pd_off.png",
        annotations=frequency_annotations,
    )
    existing_ebosc = pd.read_csv(output_root / "metrics" / "lavi_determined_band_ebosc_participant_metrics.csv.gz")
    off_ebosc = _pd_off_ebosc(output_root, definitions, config_path)
    custom_ebosc = pd.concat([existing_ebosc, off_ebosc], ignore_index=True)
    _plot(
        custom_ebosc, EBOSC_FEATURES, BAND_NAMES,
        "Pooled LAVI-determined-band eBOSC bursts with PD-OFF",
        figure_root / "lavi_determined_band_ebosc_bursts_with_pd_off.png",
        _significant_annotations(custom_statistics, "eBOSC in LAVI-determined bands"),
    )
    pd.DataFrame({
        "group": GROUPS,
        "n": [152, 331, 15],
        "note": ["independent controls", "includes the 15 PD-ON observations", "same participants as PD-ON subset"],
    }).to_csv(figure_root / "cohort_counts.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rhythmicity-root", type=Path, default=Path("outputs/rhythmicity"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/rhythmicity/pooled_control_vs_pd_on"))
    parser.add_argument("--global-config", type=Path, default=Path("config/global_pipeline.json"))
    args = parser.parse_args()
    run(args.rhythmicity_root, args.output_root, args.global_config)
    print(args.output_root / "figures" / "with_pd_off")


if __name__ == "__main__":
    main()
