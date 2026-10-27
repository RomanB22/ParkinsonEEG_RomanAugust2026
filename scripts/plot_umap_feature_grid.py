#!/usr/bin/env python3
"""Plot UMAP-1/UMAP-3 colored by MoCA and selected EEG features."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D


FEATURES = [
    ("moca", "MoCA score"),
    ("psd__theta__relative_power", "Theta relative power"),
    ("psd__alpha__relative_power", "Alpha relative power"),
    ("psd__beta__relative_power", "Beta relative power"),
    ("bout__theta__oscillatory_occupancy", "Theta-burst occupancy"),
    ("bout__theta__bouts_per_minute", "Theta bursts per minute"),
    ("bout__beta__duration_mean_s", "Beta-burst duration (s)"),
]

GROUP_MARKERS = {"PD": "o", "Control": "D"}


def _load_data(root: Path) -> pd.DataFrame:
    stats = root / "statistics" / "xgboost_minimal_features"
    coordinates = pd.read_csv(stats / "classifier_shap_umap_3d.csv")
    table = pd.read_csv(stats / "analysis_table.csv.gz")
    required = [name for name, _ in FEATURES]
    missing = [name for name in required if name not in table.columns]
    if missing:
        raise ValueError(f"Missing requested feature columns: {missing}")
    merged = coordinates.merge(
        table[["dataset", "participant_id", "group", *required]],
        on=["dataset", "participant_id"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_table"),
    )
    if merged[required].isna().all(axis=None):
        raise ValueError("Requested features did not merge with UMAP coordinates")
    return merged


def _z_normalize_features(table: pd.DataFrame) -> pd.DataFrame:
    normalized = table.copy()
    for feature, _ in FEATURES:
        values = pd.to_numeric(normalized[feature], errors="coerce")
        normalized[feature] = (values - values.mean()) / values.std(ddof=0)
    return normalized


def _plot_panel(
    ax: plt.Axes,
    table: pd.DataFrame,
    feature: str,
    label: str,
    x_coordinate: str,
    y_coordinate: str,
    z_normalized: bool,
) -> None:
    values = pd.to_numeric(table[feature], errors="coerce").to_numpy(float)
    valid = np.isfinite(values)

    # Keep every participant visible; gray points indicate missing feature values.
    for group, marker in GROUP_MARKERS.items():
        group_mask = table["group"].eq(group).to_numpy()
        missing = group_mask & ~valid
        ax.scatter(
            table.loc[missing, x_coordinate],
            table.loc[missing, y_coordinate],
            s=24,
            marker=marker,
            color="#BDBDBD",
            alpha=0.45,
            edgecolors="#000000" if group == "Control" else "white",
            linewidths=0.35,
            zorder=1,
        )

    if valid.any():
        norm = Normalize(vmin=float(values[valid].min()), vmax=float(values[valid].max()))
        for group, marker in GROUP_MARKERS.items():
            mask = table["group"].eq(group).to_numpy() & valid
            ax.scatter(
                table.loc[mask, x_coordinate],
                table.loc[mask, y_coordinate],
                c=values[mask],
                cmap="viridis",
                norm=norm,
                s=32,
                marker=marker,
                alpha=0.86,
                edgecolors="#000000" if group == "Control" else "white",
                linewidths=0.5 if group == "Control" else 0.3,
                zorder=2,
            )
        # A separate ScalarMappable keeps one shared colorbar for both markers.
        points = plt.cm.ScalarMappable(norm=norm, cmap="viridis")
        colorbar = ax.figure.colorbar(points, ax=ax, fraction=0.046, pad=0.04)
        colorbar.set_label(
            f"z({label})" if z_normalized else label,
            fontsize=8,
        )
        colorbar.ax.tick_params(labelsize=7)

    ax.set_title(
        f"{label} (z-score)" if z_normalized else label,
        fontsize=11,
        pad=7,
    )
    ax.set_xlabel(x_coordinate.upper().replace("UMAP", "UMAP-"))
    ax.set_ylabel(y_coordinate.upper().replace("UMAP", "UMAP-"))
    ax.grid(alpha=0.14, linewidth=0.6)
    ax.set_axisbelow(True)


def make_figure(
    table: pd.DataFrame,
    output: Path,
    x_coordinate: str,
    y_coordinate: str,
    z_normalized: bool,
) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(18, 9), constrained_layout=True)
    axes = axes.ravel()

    for ax, (feature, label) in zip(axes, FEATURES):
        _plot_panel(
            ax,
            table,
            feature,
            label,
            x_coordinate,
            y_coordinate,
            z_normalized,
        )

    note_ax = axes[-1]
    note_ax.axis("off")
    note_ax.text(
        0.02,
        0.92,
        "Plot notes",
        fontsize=12,
        weight="bold",
        transform=note_ax.transAxes,
    )
    note_ax.text(
        0.02,
        0.78,
        "Same participant coordinates in every panel.\n"
        "Circles = PD; diamonds = Control.\n"
        "Gray points = missing feature values.\n"
        "Coordinates are from the 3D SHAP-space\n"
        "UMAP fit.",
        fontsize=10,
        va="top",
        transform=note_ax.transAxes,
    )
    note_ax.legend(
        handles=[
            Line2D(
                [0], [0], marker=GROUP_MARKERS["PD"], color="w", label="PD",
                markerfacecolor="#777777", markeredgecolor="white", markersize=8,
            ),
            Line2D(
                [0], [0], marker=GROUP_MARKERS["Control"], color="w", label="Control",
                markerfacecolor="#777777", markeredgecolor="white", markersize=8,
            ),
        ],
        loc="lower left",
        frameon=False,
        fontsize=10,
        title="Group marker shapes",
        title_fontsize=9,
    )

    n = len(table)
    moca_n = table["moca"].notna().sum()
    fig.suptitle(
        "Participant-level UMAP colored by cognition and selected EEG features\n"
        f"{x_coordinate.upper().replace('UMAP', 'UMAP-')} versus "
        f"{y_coordinate.upper().replace('UMAP', 'UMAP-')}; "
        f"{'z-normalized features; ' if z_normalized else ''}"
        f"n={n} participants; MoCA available for n={moca_n}",
        fontsize=15,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/global"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/global/figures/summary"),
    )
    args = parser.parse_args()
    table = _load_data(args.output_root)
    projections = [("umap1", "umap3"), ("umap1", "umap2"), ("umap2", "umap3")]
    for z_normalized, plot_table in [(False, table), (True, _z_normalize_features(table))]:
        normalization_suffix = "_znormalized" if z_normalized else ""
        for x_coordinate, y_coordinate in projections:
            projection_suffix = "" if (x_coordinate, y_coordinate) == ("umap1", "umap3") else f"_{x_coordinate}_{y_coordinate}"
            output = args.output_dir / (
                f"umap_feature_grid_moca_eeg{normalization_suffix}{projection_suffix}.png"
            )
            make_figure(
                plot_table,
                output,
                x_coordinate,
                y_coordinate,
                z_normalized,
            )
            print(f"Wrote {output}")

    # Preserve the earlier explicit UMAP-1/UMAP-3 filename with the current styling.
    legacy_output = args.output_dir / "umap_feature_grid_moca_eeg_umap1_umap3.png"
    make_figure(table, legacy_output, "umap1", "umap3", False)
    print(f"Wrote {legacy_output}")


if __name__ == "__main__":
    main()
