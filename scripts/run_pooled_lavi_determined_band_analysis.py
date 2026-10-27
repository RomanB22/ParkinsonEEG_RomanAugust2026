#!/usr/bin/env python
"""Repeat pooled LAVI and eBOSC analyses in pooled LAVI-determined bands."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any

import mne
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from analyses.rhythmicity.lavi_burst_features import _summarize_band
from global_pipeline.analysis import _analyze_recording
from global_pipeline.schema import load_global_config
from run_pooled_lavi_ebosc_analysis import (
    EBOSC_FEATURES,
    LAVI_BURST_FEATURES,
    LAVI_FEATURES,
    _family_statistics,
    _plot_family,
    _pool_groups,
)


BAND_NAMES = ["theta_low", "alpha_high", "beta_low", "beta_high", "gamma_low"]
BAND_NAME_BY_SEGMENT = {3: "theta_low", 4: "alpha_high", 5: "beta_low", 6: "beta_high", 7: "gamma_low"}
COMMON_EBOSC_LIMITS = (4.0, 40.0)


def _band_definitions(interval_path: Path) -> pd.DataFrame:
    intervals = pd.read_csv(interval_path)
    selected = intervals.loc[intervals["segment_index"].isin(BAND_NAME_BY_SEGMENT)].copy()
    selected["band"] = selected["segment_index"].map(BAND_NAME_BY_SEGMENT)
    selected["lavi_low_hz"] = selected["start_frequency_hz"]
    selected["lavi_high_hz"] = selected["end_frequency_hz"]
    selected["ebosc_low_hz"] = selected["lavi_low_hz"].clip(lower=COMMON_EBOSC_LIMITS[0])
    selected["ebosc_high_hz"] = selected["lavi_high_hz"].clip(upper=COMMON_EBOSC_LIMITS[1])
    selected["ebosc_clipped_to_common_support"] = (
        ~np.isclose(selected["lavi_low_hz"], selected["ebosc_low_hz"])
        | ~np.isclose(selected["lavi_high_hz"], selected["ebosc_high_hz"])
    )
    columns = [
        "group", "band", "direction", "lavi_low_hz", "lavi_high_hz",
        "ebosc_low_hz", "ebosc_high_hz", "ebosc_clipped_to_common_support",
        "start_bin_center_hz", "end_bin_center_hz", "n_frequency_bins",
    ]
    result = selected[columns].sort_values(
        ["group", "band"], key=lambda values: values.map({name: index for index, name in enumerate(BAND_NAMES)}) if values.name == "band" else values
    ).reset_index(drop=True)
    if result.groupby("group")["band"].nunique().ne(len(BAND_NAMES)).any():
        raise ValueError("The pooled ABBA interval table does not contain all resolved bands")
    return result


def _lavi_tables(
    rhythmicity_root: Path,
    definitions: pd.DataFrame,
    *,
    include_pd_off: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    from lavi import abba

    electrode_metadata = pd.read_csv(
        rhythmicity_root / "metrics" / "electrode_rhythmicity.csv.gz", low_memory=False
    )
    if include_pd_off:
        electrode_metadata = electrode_metadata.loc[
            electrode_metadata["group"].isin(["Control", "PD", "PD_ON", "PD_OFF"])
        ].copy()
        electrode_metadata["group"] = electrode_metadata["group"].map({
            "Control": "Control", "PD": "PD + PD-ON", "PD_ON": "PD + PD-ON", "PD_OFF": "PD-OFF",
        })
    else:
        electrode_metadata = _pool_groups(electrode_metadata)
    metadata = electrode_metadata[
        ["dataset", "recording_id", "participant_id", "group", "electrode"]
    ].drop_duplicates()
    definitions_by_group = {
        group: frame.set_index("band") for group, frame in definitions.groupby("group")
    }
    rows: list[dict[str, Any]] = []
    for (dataset, recording_id), recording_metadata in tqdm(
        metadata.groupby(["dataset", "recording_id"], sort=False), desc="LAVI-determined band summaries"
    ):
        profile_path = rhythmicity_root / "profiles" / str(dataset) / f"{recording_id}.npz"
        if not profile_path.is_file():
            continue
        with np.load(profile_path, allow_pickle=False) as saved:
            lavi = np.asarray(saved["lavi"], dtype=float)
            frequencies = np.asarray(saved["foi"], dtype=float)
            channels = [str(value) for value in saved["channels"].tolist()]
            if "sigvect" in saved:
                sigvect = np.asarray(saved["sigvect"], dtype=float)
            else:
                # Profiles created before sigvect persistence can be restored
                # exactly because the default median-baseline ABBA is deterministic.
                _, _, restored = abba(lavi, frequencies)
                sigvect = np.asarray(restored, dtype=float)
        channel_metadata = recording_metadata.set_index("electrode")
        for channel_index, channel in enumerate(channels):
            if channel not in channel_metadata.index:
                continue
            meta = channel_metadata.loc[channel]
            group = str(meta["group"])
            for band in BAND_NAMES:
                definition = definitions_by_group[group].loc[band]
                low, high = float(definition["lavi_low_hz"]), float(definition["lavi_high_hz"])
                mask = np.isfinite(lavi[channel_index]) & (frequencies >= low) & (frequencies <= high)
                values = lavi[channel_index, mask]
                selected_frequencies = frequencies[mask]
                if len(values):
                    peak = int(np.nanargmax(values))
                    basic = {
                        "lavi_mean": float(np.nanmean(values)),
                        "lavi_median": float(np.nanmedian(values)),
                        "lavi_peak": float(values[peak]),
                        "lavi_peak_frequency_hz": float(selected_frequencies[peak]),
                        "high_rhythmicity_fraction": float(np.mean(sigvect[channel_index, mask] > 0)),
                        "low_rhythmicity_fraction": float(np.mean(sigvect[channel_index, mask] < 0)),
                    }
                else:
                    basic = {feature: np.nan for feature in LAVI_FEATURES}
                frequency_bursts = _summarize_band(
                    lavi[channel_index], sigvect[channel_index], frequencies, (low, high)
                )
                rows.append({
                    "dataset": dataset, "recording_id": recording_id,
                    "participant_id": str(meta["participant_id"]), "group": group,
                    "electrode": channel, "band": band, "direction": definition["direction"],
                    "band_low_hz": low, "band_high_hz": high, **basic, **frequency_bursts,
                })
    electrode = pd.DataFrame(rows)
    keys = ["dataset", "participant_id", "group", "band", "direction", "band_low_hz", "band_high_hz"]
    participant_electrode = electrode.groupby(keys + ["electrode"], as_index=False)[list(LAVI_FEATURES) + list(LAVI_BURST_FEATURES)].mean(numeric_only=True)
    participant = participant_electrode.groupby(keys, as_index=False)[list(LAVI_FEATURES) + list(LAVI_BURST_FEATURES)].mean(numeric_only=True)
    return electrode, participant


def _ebosc_worker(task: tuple[dict[str, Any], dict[str, tuple[float, float]], str, str]) -> str:
    record, bands, config_path, cache_path_string = task
    cache_path = Path(cache_path_string)
    if cache_path.is_file():
        return cache_path_string
    base = load_global_config(config_path).for_dataset(str(record["dataset_id"]))
    ebosc = dict(base.ebosc_settings)
    ebosc.update(frequency_min_hz=COMMON_EBOSC_LIMITS[0], frequency_max_hz=COMMON_EBOSC_LIMITS[1])
    aperiodic = dict(base.aperiodic_settings)
    aperiodic["frequency_range_hz"] = list(COMMON_EBOSC_LIMITS)
    analysis_config = replace(
        base, bands=bands, permutation_dimensions=(), ebosc_settings=ebosc,
        aperiodic_settings=aperiodic,
    )
    features, _ = _analyze_recording(record, analysis_config)
    wanted = [column for column in features if column.startswith("bout__")]
    result = features[["dataset_id", "recording_id", "participant_id", "group", "electrode"] + wanted]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".csv.gz", dir=cache_path.parent, delete=False) as handle:
            temporary = Path(handle.name)
        result.to_csv(temporary, index=False, compression="gzip")
        temporary.replace(cache_path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return cache_path_string


def _ebosc_table(
    rhythmicity_root: Path,
    output_root: Path,
    definitions: pd.DataFrame,
    config_path: Path,
    workers: int,
) -> pd.DataFrame:
    metadata = _pool_groups(pd.read_csv(
        rhythmicity_root / "metrics" / "electrode_rhythmicity.csv.gz", low_memory=False
    ))
    included_recordings = metadata[["dataset", "recording_id", "participant_id", "group"]].drop_duplicates()
    canonical = pd.read_csv("outputs/global/canonical/recordings.csv.gz", low_memory=False)
    canonical = canonical.merge(
        included_recordings,
        left_on=["dataset_id", "recording_id"], right_on=["dataset", "recording_id"],
        how="inner", suffixes=("", "_pooled"),
    )
    definitions_by_group = {
        group: {row.band: (float(row.ebosc_low_hz), float(row.ebosc_high_hz)) for row in frame.itertuples(index=False)}
        for group, frame in definitions.groupby("group")
    }
    signature_payload = {
        "bands": definitions_by_group, "common_support": COMMON_EBOSC_LIMITS,
        "aperiodic_fit": "best_bic_4_40", "power_percentile": 0.95, "minimum_cycles": 3.0,
    }
    signature = hashlib.sha256(json.dumps(signature_payload, sort_keys=True).encode()).hexdigest()[:12]
    cache_root = output_root / "intermediate" / "ebosc_lavi_determined_bands"
    tasks: list[tuple[dict[str, Any], dict[str, tuple[float, float]], str, str]] = []
    for row in canonical.to_dict("records"):
        pooled_group = str(row["group_pooled"])
        row["group"] = pooled_group
        cache_path = cache_root / str(row["dataset_id"]) / f"{row['recording_id']}_{signature}.csv.gz"
        tasks.append((row, definitions_by_group[pooled_group], str(config_path), str(cache_path)))
    paths: list[str] = []
    if workers == 1:
        for task in tqdm(tasks, desc="eBOSC in LAVI-determined bands"):
            paths.append(_ebosc_worker(task))
    else:
        try:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(_ebosc_worker, task) for task in tasks]
                for future in tqdm(as_completed(futures), total=len(futures), desc="eBOSC in LAVI-determined bands"):
                    paths.append(future.result())
        except PermissionError:
            # Managed macOS environments can deny POSIX semaphore creation.
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(_ebosc_worker, task) for task in tasks]
                for future in tqdm(as_completed(futures), total=len(futures), desc="eBOSC in LAVI-determined bands (threads)"):
                    paths.append(future.result())
    electrode = pd.concat([pd.read_csv(path, low_memory=False) for path in paths], ignore_index=True)
    rows: list[pd.DataFrame] = []
    for band in BAND_NAMES:
        columns = {
            f"bout__{band}__n_bouts": "n_bouts",
            f"bout__{band}__oscillatory_occupancy": "oscillatory_occupancy",
            f"bout__{band}__bouts_per_minute": "bouts_per_minute",
            f"bout__{band}__duration_mean_s": "duration_mean_s",
            f"bout__{band}__amplitude_mean": "amplitude_mean",
            f"bout__{band}__cycles_mean": "cycles_mean",
        }
        subset = electrode[["dataset_id", "recording_id", "participant_id", "group", "electrode"] + list(columns)].rename(columns=columns)
        subset["band"] = band
        rows.append(subset)
    long_electrode = pd.concat(rows, ignore_index=True).rename(columns={"dataset_id": "dataset"})
    participant_electrode = long_electrode.groupby(
        ["dataset", "participant_id", "group", "electrode", "band"], as_index=False
    )[list(EBOSC_FEATURES)].mean(numeric_only=True)
    return participant_electrode.groupby(
        ["dataset", "participant_id", "group", "band"], as_index=False
    )[list(EBOSC_FEATURES)].mean(numeric_only=True)


def run(rhythmicity_root: Path, output_root: Path, config_path: Path, workers: int) -> dict[str, Any]:
    metrics_root = output_root / "metrics"; metrics_root.mkdir(parents=True, exist_ok=True)
    statistics_root = output_root / "statistics"; statistics_root.mkdir(parents=True, exist_ok=True)
    figures_root = output_root / "figures"; figures_root.mkdir(parents=True, exist_ok=True)
    definitions = _band_definitions(statistics_root / "pooled_lavi_abba_intervals.csv")
    definitions.to_csv(statistics_root / "lavi_determined_band_definitions.csv", index=False)

    electrode, participant = _lavi_tables(rhythmicity_root, definitions)
    electrode.to_csv(metrics_root / "lavi_determined_band_electrode_metrics.csv.gz", index=False)
    participant.to_csv(metrics_root / "lavi_determined_band_participant_metrics.csv.gz", index=False)
    lavi_stats = _family_statistics(participant, LAVI_FEATURES, "LAVI in LAVI-determined bands", BAND_NAMES)
    frequency_bout_stats = _family_statistics(participant, LAVI_BURST_FEATURES, "LAVI frequency bouts in LAVI-determined bands", BAND_NAMES)
    _plot_family(participant, lavi_stats, LAVI_FEATURES, "Pooled LAVI in pooled LAVI-determined bands", figures_root / "lavi_determined_band_lavi_violins.png", BAND_NAMES)
    _plot_family(participant, frequency_bout_stats, LAVI_BURST_FEATURES, "Pooled LAVI frequency bouts in pooled LAVI-determined bands", figures_root / "lavi_determined_band_frequency_burst_violins.png", BAND_NAMES)

    ebosc = _ebosc_table(rhythmicity_root, output_root, definitions, config_path, workers)
    ebosc.to_csv(metrics_root / "lavi_determined_band_ebosc_participant_metrics.csv.gz", index=False)
    ebosc_stats = _family_statistics(ebosc, EBOSC_FEATURES, "eBOSC in LAVI-determined bands", BAND_NAMES)
    _plot_family(ebosc, ebosc_stats, EBOSC_FEATURES, "Pooled eBOSC bursts in pooled LAVI-determined bands", figures_root / "lavi_determined_band_ebosc_burst_violins.png", BAND_NAMES)
    statistics = pd.concat([lavi_stats, frequency_bout_stats, ebosc_stats], ignore_index=True)
    statistics["band_definition"] = "pooled_group_specific_lavi_abba"
    statistics["band_selection_scope"] = "same pooled cohort used for estimation and testing"
    statistics["confirmatory_inference"] = False
    statistics.to_csv(statistics_root / "lavi_determined_band_group_comparisons.csv", index=False)
    summary = {
        "bands": BAND_NAMES, "n_participants": int(participant[["dataset", "participant_id"]].drop_duplicates().shape[0]),
        "eBOSC_common_frequency_support_hz": list(COMMON_EBOSC_LIMITS),
        "eBOSC_participant_rows": len(ebosc), "statistics_rows": len(statistics),
        "group_specific_band_limits": True,
        "interpretation": "exploratory; bands were selected and tested in the same pooled cohort",
    }
    (output_root / "lavi_determined_band_manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rhythmicity-root", type=Path, default=Path("outputs/rhythmicity"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/rhythmicity/pooled_control_vs_pd_on"))
    parser.add_argument("--global-config", type=Path, default=Path("config/global_pipeline.json"))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    print(json.dumps(run(args.rhythmicity_root, args.output_root, args.global_config, args.workers), indent=2))


if __name__ == "__main__":
    main()
