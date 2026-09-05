"""Append weighted permutation entropy to existing primary-analysis caches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from analyses.bouts.metrics import analyze_bout_segments
from analyses.ordinal.metrics import (
    filter_epoch_data,
    weighted_permutation_entropy_channels,
    weighted_permutation_entropy_epoch_data,
)
from analyses.ordinal.plots import (
    metric_color_limits,
    plot_group_band_topomaps,
    plot_group_topomaps,
)


METRIC = "weighted_permutation_entropy"


def _load_manifest(root: Path) -> dict:
    return json.loads((root / "manifest.json").read_text(encoding="utf-8"))


def _write(table: pd.DataFrame, path: Path) -> None:
    table.to_csv(path, index=False, float_format="%.17g", compression="infer")


def augment_ordinal(root: Path) -> None:
    manifest = _load_manifest(root)
    config = manifest["analysis_config"]
    metrics_dir = root / "metrics"
    electrode_path = metrics_dir / "electrode_metrics.csv"
    band_path = metrics_dir / "band_electrode_metrics.csv"
    inputs = pd.read_csv(metrics_dir / "analyzed_inputs.csv")
    electrodes = json.loads((metrics_dir / "electrode_sets.json").read_text())[
        "common_electrodes"
    ]
    ordinal = config["ordinal"]
    dx, tau = int(ordinal["embedding_dimension"]), int(ordinal["delay_samples"])
    electrode_table = pd.read_csv(electrode_path)
    band_table = pd.read_csv(band_path)

    broadband_values: dict[tuple[str, str], float] = {}
    band_values: dict[tuple[str, str, str], float] = {}
    common_info = None
    for input_row in inputs.itertuples(index=False):
        subject = str(input_row.subject_id)
        epochs = mne.read_epochs(str(input_row.epoch_file), preload=True, verbose="ERROR")
        picks = [epochs.ch_names.index(name) for name in electrodes]
        data = epochs.get_data(picks=picks, copy=True)
        if common_info is None:
            common_info = mne.pick_info(epochs.info, picks, copy=True)
            common_info["bads"] = []
        sfreq = float(epochs.info["sfreq"])
        filtered_by_band = {
            band: filter_epoch_data(
                data,
                sfreq=sfreq,
                low_hz=float(limits[0]),
                high_hz=float(limits[1]),
                order=int(config["band_filter"]["order"]),
            )
            for band, limits in config["bands"].items()
        }
        broadband = weighted_permutation_entropy_channels(
            data, dx=dx, tau=tau, tie_precision=None
        )
        for index, electrode in enumerate(electrodes):
            broadband_values[(subject, electrode)] = float(broadband[index])
            for band in config["bands"]:
                band_values[(subject, band, electrode)] = float(
                    weighted_permutation_entropy_channels(
                        filtered_by_band[band], dx=dx, tau=tau, tie_precision=None
                    )[index]
                )

    electrode_table[METRIC] = [
        broadband_values[(str(row.subject_id), str(row.electrode))]
        for row in electrode_table.itertuples(index=False)
    ]
    band_table[METRIC] = [
        band_values[(str(row.subject_id), str(row.band), str(row.electrode))]
        for row in band_table.itertuples(index=False)
    ]
    _write(electrode_table, electrode_path)
    _write(band_table, band_path)

    # Only the newly added metric is plotted. Existing figure files are left
    # untouched, and the new figures are written to a dedicated directory.
    figure_dir = root / "figures" / "topomaps" / METRIC
    figure_dir.mkdir(parents=True, exist_ok=True)
    metric_limits = metric_color_limits(electrode_table, (METRIC,))
    groups = [group for group in config["plots"].get("group_order", []) if group in set(electrode_table["group"])]
    if len(groups) < 2:
        groups = sorted(electrode_table["group"].dropna().astype(str).unique())
    plot_group_topomaps(
        electrode_table,
        common_info,
        groups,
        metric_limits,
        figure_dir / "broadband_group_mean_topomap.png",
        int(config["plots"].get("dpi", 150)),
        metrics=(METRIC,),
        metric_set_label="weighted permutation entropy",
    )
    band_order = list(config["bands"])
    band_limits = {
        band: metric_color_limits(
            band_table.loc[band_table["band"].eq(band)], (METRIC,)
        )
        for band in band_order
    }
    plot_group_band_topomaps(
        band_table,
        common_info,
        groups,
        band_order,
        config["plots"].get("band_display_names", {}),
        band_limits,
        figure_dir / "bands",
        int(config["plots"].get("dpi", 150)),
        metrics=(METRIC,),
        metric_set_label="weighted permutation entropy",
        filename_suffix="weighted_permutation_entropy_group_topomap",
    )


def augment_bouts(root: Path) -> None:
    manifest = _load_manifest(root)
    config = manifest["analysis_config"]
    metrics_dir = root / "metrics"
    electrode_path = metrics_dir / "subject_electrode_band_metrics.csv"
    electrode_table = pd.read_csv(electrode_path)
    inputs = pd.read_csv(metrics_dir / "analyzed_inputs.csv")
    electrodes = json.loads((metrics_dir / "electrode_sets.json").read_text())[
        "common_electrodes"
    ]
    ordinal = config["ordinal"]
    dx, tau = int(ordinal["embedding_dimension"]), int(ordinal["delay_samples"])
    scale_free_root = Path(config["input"]["scale_free_output_dir"])
    values: dict[tuple[str, str, str], float] = {}
    bout_rows: dict[tuple[str, str, str, int], float] = {}
    for input_row in inputs.itertuples(index=False):
        subject = str(input_row.subject_id)
        epochs = mne.read_epochs(str(input_row.epoch_file), preload=True, verbose="ERROR")
        picks = [epochs.ch_names.index(name) for name in electrodes]
        data = epochs.get_data(picks=picks, copy=True)
        sfreq = float(epochs.info["sfreq"])
        episodes_path = scale_free_root / "intermediate" / "episodes" / f"{subject}_bout_episodes.csv.gz"
        episodes = pd.read_csv(episodes_path)
        for index, electrode in enumerate(electrodes):
            for band, limits in config["bands"].items():
                filtered = filter_epoch_data(
                    data[:, index : index + 1, :], sfreq=sfreq,
                    low_hz=float(limits[0]), high_hz=float(limits[1]),
                    order=int(config["band_filter"]["order"]),
                )[:, 0, :]
                selected = episodes.loc[
                    episodes["electrode"].eq(electrode) & episodes["band"].eq(band)
                ].drop(columns=["subject_id", "group", "electrode"], errors="ignore")
                _, summary, bouts, _ = analyze_bout_segments(
                    filtered, selected, dx=dx, tau=tau, tie_precision=None
                )
                values[(subject, band, electrode)] = float(summary[METRIC])
                if len(bouts):
                    for row in bouts.itertuples(index=False):
                        bout_rows[(subject, band, electrode, int(row.bout_sequence_number))] = float(
                            row.weighted_permutation_entropy
                        )

    electrode_table[METRIC] = [
        values[(str(row.subject_id), str(row.band), str(row.electrode))]
        for row in electrode_table.itertuples(index=False)
    ]
    _write(electrode_table, electrode_path)
    for path in sorted((root / "intermediate" / "bout_metrics").glob("*_bout_ordinal_metrics.csv.gz")):
        subject = path.name.split("_bout_ordinal_metrics", 1)[0]
        table = pd.read_csv(path)
        table[METRIC] = [
            bout_rows[(subject, str(row.band), str(row.electrode), int(row.bout_sequence_number))]
            for row in table.itertuples(index=False)
        ]
        _write(table, path)

    # Reuse the same plotting implementation as the signal-level ordinal
    # analysis, with one weighted-entropy panel per group and band.
    first_input = inputs.iloc[0]
    epochs = mne.read_epochs(str(first_input.epoch_file), preload=False, verbose="ERROR")
    picks = [epochs.ch_names.index(name) for name in electrodes]
    common_info = mne.pick_info(epochs.info, picks, copy=True)
    common_info["bads"] = []
    figure_dir = root / "figures" / "topomaps" / METRIC
    band_order = list(config["bands"])
    groups = [group for group in config["plots"].get("group_order", []) if group in set(electrode_table["group"])]
    if len(groups) < 2:
        groups = sorted(electrode_table["group"].dropna().astype(str).unique())
    band_limits = {
        band: metric_color_limits(
            electrode_table.loc[electrode_table["band"].eq(band)], (METRIC,)
        )
        for band in band_order
    }
    plot_group_band_topomaps(
        electrode_table,
        common_info,
        groups,
        band_order,
        config["plots"].get("band_display_names", {}),
        band_limits,
        figure_dir / "bands",
        int(config["plots"].get("dpi", 150)),
        metrics=(METRIC,),
        metric_set_label="weighted permutation entropy within bouts",
        filename_suffix="weighted_permutation_entropy_group_topomap",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", choices=("full", "matched", "both"), default="both")
    args = parser.parse_args()
    roots = [Path("outputs") / cohort for cohort in ("full", "matched") if args.cohort in (cohort, "both")]
    for root in roots:
        augment_ordinal(root / "ordinal")
        augment_bouts(root / "bouts")
        print(f"Augmented weighted permutation entropy: {root}")


if __name__ == "__main__":
    main()
