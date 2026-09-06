"""Streaming PSD, entropy, bout, and clinical analyses for canonical datasets.

The analysis unit is one recording.  Epochs are never concatenated in memory:
MNE reads a small block, features are accumulated, and the block is released
before the next one is loaded.  The saved tables are feature-sized, not
sample-sized, and are gzip-compressed.
"""

from __future__ import annotations

import json
import math
from itertools import combinations
from pathlib import Path
from typing import Any

import mne
import numpy as np
import pandas as pd
import xarray as xr
from scipy.signal import hilbert, welch
from scipy.stats import mannwhitneyu, pearsonr, rankdata, spearmanr, ttest_ind

from analyses.bouts.metrics import ordinal_counts, shannon_metrics_from_counts
from analyses.ordinal.metrics import (
    filter_epoch_data,
    metrics_from_probabilities,
    weighted_permutation_entropy_epoch_data,
)

from .converter import convert_config
from .schema import CANONICAL_COLUMNS, GlobalConfig, load_global_config, read_canonical_table


ENTROPY_METRICS = (
    "entropy",
    "complexity",
    "fisher_information",
    "weighted_permutation_entropy",
)


def _write_table(table: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False, compression="gzip", float_format="%.10g")


def _epoch_data(
    epochs: mne.BaseEpochs,
    start: int,
    stop: int,
    picks: list[int],
    channel_names: list[str],
) -> xr.DataArray:
    """Read one block and downcast it immediately to reduce peak memory."""
    try:
        data = epochs.get_data(picks=picks, item=slice(start, stop), copy=True)
    except TypeError:  # Compatibility with older MNE releases.
        data = epochs[start:stop].get_data(picks=picks, copy=True)
    values = np.asarray(data, dtype=np.float32)
    return xr.DataArray(
        values,
        dims=("epoch", "channel", "time"),
        coords={
            "epoch": np.arange(start, stop, dtype=np.int64),
            "channel": channel_names,
            "time": np.arange(values.shape[-1], dtype=np.int64) / float(epochs.info["sfreq"]),
        },
        name="eeg",
    )


def _runs(mask: np.ndarray, minimum_samples: int) -> list[tuple[int, int]]:
    padded = np.concatenate(([False], np.asarray(mask, dtype=bool), [False]))
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    return [
        (int(start), int(stop))
        for start, stop in zip(changes[::2], changes[1::2])
        if int(stop - start) >= minimum_samples
    ]


def _new_entropy_state(dx: int) -> dict[str, Any]:
    return {
        "counts": np.zeros(math.factorial(dx), dtype=np.int64),
        "weighted_sum": 0.0,
        "n_blocks": 0,
        "n_patterns": 0,
        "n_ties": 0,
    }


def _analyze_recording(record: dict[str, Any], config: GlobalConfig) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = Path(record["epoch_path"])
    epochs = mne.read_epochs(path, preload=False, verbose="ERROR")
    eeg_picks = list(mne.pick_types(epochs.info, eeg=True, exclude=[]))
    if not eeg_picks:
        raise ValueError(f"{path}: no EEG channels found")
    channels = [epochs.ch_names[index] for index in eeg_picks]
    sfreq = float(epochs.info["sfreq"])
    n_epochs = len(epochs)
    # ``epochs.times`` is metadata and does not trigger a preload of the full
    # recording, which is important for long studies.
    n_samples = int(len(epochs.times))
    if n_epochs < 1:
        raise ValueError(f"{path}: no accepted epochs")

    psd_sum: np.ndarray | None = None
    psd_count = 0
    frequencies: np.ndarray | None = None
    entropy: dict[str, list[dict[str, Any]]] = {
        band: [_new_entropy_state(config.embedding_dimension) for _ in channels]
        for band in config.bands
    }
    bouts: dict[str, list[dict[str, Any]]] = {
        band: [
            {
                "n_bouts": 0,
                "n_bout_samples": 0,
                "duration_sum": 0.0,
                "duration_sq_sum": 0.0,
                "amplitude_sum": 0.0,
                "cycle_sum": 0.0,
                "within": _new_entropy_state(config.embedding_dimension),
            }
            for _ in channels
        ]
        for band in config.bands
    }

    for start in range(0, n_epochs, config.block_epochs):
        stop = min(start + config.block_epochs, n_epochs)
        block = _epoch_data(epochs, start, stop, eeg_picks, channels)
        block_n = block.sizes["epoch"]
        current_frequencies, block_psd = welch(
            block.data,
            fs=sfreq,
            window="hann",
            nperseg=min(block.sizes["time"], max(8, int(round(2.0 * sfreq)))),
            axis=-1,
            detrend="constant",
        )
        if frequencies is None:
            frequencies = np.asarray(current_frequencies, dtype=float)
            psd_sum = np.zeros((len(channels), len(frequencies)), dtype=np.float64)
        elif not np.allclose(frequencies, current_frequencies):
            raise RuntimeError(f"{path}: PSD frequency grid changed between blocks")
        psd_da = xr.DataArray(
            np.asarray(block_psd, dtype=np.float32),
            dims=("epoch", "channel", "frequency"),
            coords={
                "epoch": block.coords["epoch"],
                "channel": channels,
                "frequency": frequencies,
            },
            name="welch_psd",
        )
        psd_sum += psd_da.sum(dim="epoch").astype(np.float64).data
        psd_count += block_n

        for band, (low, high) in config.bands.items():
            filtered = filter_epoch_data(
                block.data.astype(np.float64, copy=False),
                sfreq=sfreq,
                low_hz=low,
                high_hz=high,
                order=4,
            )
            center = max((low + high) / 2.0, 0.1)
            minimum_samples = max(
                config.embedding_dimension + 1,
                int(math.ceil(config.bout_minimum_cycles * sfreq / center)),
            )
            for channel_index in range(len(channels)):
                state = entropy[band][channel_index]
                probabilities, n_patterns, n_ties = _pooled_block_probabilities(
                    filtered[:, channel_index, :], config.embedding_dimension, config.delay_samples
                )
                state["counts"] += np.rint(probabilities * n_patterns).astype(np.int64)
                state["n_patterns"] += n_patterns
                state["n_ties"] += n_ties
                state["weighted_sum"] += weighted_permutation_entropy_epoch_data(
                    filtered[:, channel_index, :],
                    dx=config.embedding_dimension,
                    tau=config.delay_samples,
                ) * block_n
                state["n_blocks"] += block_n

                bout_state = bouts[band][channel_index]
                for epoch_index in range(block_n):
                    signal = filtered[epoch_index, channel_index]
                    amplitude = np.abs(hilbert(signal))
                    threshold = np.percentile(amplitude, config.bout_threshold_percentile)
                    for bout_start, bout_stop in _runs(amplitude >= threshold, minimum_samples):
                        segment = signal[bout_start:bout_stop]
                        duration = (bout_stop - bout_start) / sfreq
                        bout_state["n_bouts"] += 1
                        bout_state["n_bout_samples"] += bout_stop - bout_start
                        bout_state["duration_sum"] += duration
                        bout_state["duration_sq_sum"] += duration * duration
                        bout_state["amplitude_sum"] += float(np.mean(amplitude[bout_start:bout_stop]))
                        bout_state["cycle_sum"] += duration * center
                        counts, ties = ordinal_counts(
                            segment,
                            dx=config.embedding_dimension,
                            tau=config.delay_samples,
                        )
                        bout_state["within"]["counts"] += counts
                        bout_state["within"]["n_patterns"] += int(counts.sum())
                        bout_state["within"]["n_ties"] += ties
                        if int(counts.sum()):
                            bout_state["within"]["weighted_sum"] += float(
                                weighted_permutation_entropy_epoch_data(
                                    segment[None, :],
                                    dx=config.embedding_dimension,
                                    tau=config.delay_samples,
                                )
                            ) * int(counts.sum())

    if psd_sum is None or frequencies is None or psd_count == 0:
        raise RuntimeError(f"{path}: no PSD features were calculated")
    mean_psd = psd_sum / psd_count
    rows: list[dict[str, Any]] = []
    duration_minutes = n_epochs * n_samples / sfreq / 60.0
    metadata = {key: record.get(key, "") for key in CANONICAL_COLUMNS if key not in {"epoch_path", "raw_path"}}
    for channel_index, electrode in enumerate(channels):
        total_mask = (frequencies >= min(low for low, _ in config.bands.values())) & (
            frequencies <= max(high for _, high in config.bands.values())
        )
        total_power = float(np.trapezoid(mean_psd[channel_index, total_mask], frequencies[total_mask]))
        row = {**metadata, "electrode": electrode, "sampling_frequency_hz": sfreq, "n_epochs": n_epochs}
        for band, (low, high) in config.bands.items():
            band_mask = (frequencies >= low) & (frequencies <= high)
            band_power = float(np.trapezoid(mean_psd[channel_index, band_mask], frequencies[band_mask]))
            row[f"psd__{band}__absolute_power"] = band_power
            row[f"psd__{band}__relative_power"] = band_power / total_power if total_power > 0 else np.nan
            state = entropy[band][channel_index]
            if state["n_patterns"]:
                values = metrics_from_probabilities(
                    state["counts"] / state["n_patterns"], dx=config.embedding_dimension
                )
                for metric in ENTROPY_METRICS:
                    value = (
                        state["weighted_sum"] / state["n_blocks"]
                        if metric == "weighted_permutation_entropy"
                        else values[metric]
                    )
                    row[f"entropy__{band}__{metric}"] = value
            else:
                for metric in ENTROPY_METRICS:
                    row[f"entropy__{band}__{metric}"] = np.nan
            bout_state = bouts[band][channel_index]
            count = bout_state["n_bouts"]
            row[f"bout__{band}__n_bouts"] = count
            row[f"bout__{band}__oscillatory_occupancy"] = (
                bout_state["n_bout_samples"] / (n_epochs * n_samples) if n_epochs * n_samples else np.nan
            )
            row[f"bout__{band}__bouts_per_minute"] = count / duration_minutes if duration_minutes else np.nan
            row[f"bout__{band}__duration_mean_s"] = bout_state["duration_sum"] / count if count else np.nan
            row[f"bout__{band}__amplitude_mean"] = bout_state["amplitude_sum"] / count if count else np.nan
            row[f"bout__{band}__cycles_mean"] = bout_state["cycle_sum"] / count if count else np.nan
            within = bout_state["within"]
            if within["n_patterns"]:
                values = shannon_metrics_from_counts(
                    within["counts"], dx=config.embedding_dimension
                )
                for metric in ENTROPY_METRICS:
                    row[f"within_bout__{band}__{metric}"] = (
                        within["weighted_sum"] / within["n_patterns"]
                        if metric == "weighted_permutation_entropy"
                        else values[metric]
                    )
            else:
                for metric in ENTROPY_METRICS:
                    row[f"within_bout__{band}__{metric}"] = np.nan
        rows.append(row)
    return pd.DataFrame.from_records(rows), {"channels": channels, "sfreq": sfreq, "frequencies": frequencies}


def _pooled_block_probabilities(data: np.ndarray, dx: int, tau: int) -> tuple[np.ndarray, int, int]:
    counts = None
    patterns = 0
    ties = 0
    from analyses.ordinal.metrics import ordinal_probabilities

    for epoch in np.asarray(data):
        probabilities, n_patterns, n_ties = ordinal_probabilities(
            epoch[None, :], dx=dx, tau=tau
        )
        if counts is None:
            counts = np.zeros_like(probabilities)
        counts += probabilities * n_patterns
        patterns += n_patterns
        ties += n_ties
    if counts is None or patterns == 0:
        raise ValueError("No ordinal patterns available in epoch block")
    return counts / patterns, patterns, ties


def _bh(p_values: np.ndarray) -> np.ndarray:
    result = np.full(len(p_values), np.nan, dtype=float)
    valid = np.isfinite(p_values)
    if not valid.any():
        return result
    values = p_values[valid]
    order = np.argsort(values)
    adjusted = values[order] * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result_indices = np.flatnonzero(valid)[order]
    result[result_indices] = np.minimum(adjusted, 1.0)
    return result


def group_statistics(table: pd.DataFrame, config: GlobalConfig) -> pd.DataFrame:
    feature_columns = [column for column in table.columns if column.startswith(("psd__", "entropy__", "bout__", "within_bout__"))]
    rows: list[dict[str, Any]] = []
    for dataset_id, dataset_table in table.groupby("dataset_id", sort=False):
        groups = sorted(str(value) for value in dataset_table["group"].dropna().unique())
        for group_a, group_b in combinations(groups, 2):
            for electrode, electrode_table in dataset_table.groupby("electrode", sort=False):
                for feature in feature_columns:
                    selected_a = electrode_table.loc[electrode_table["group"].eq(group_a), feature].dropna().to_numpy(float)
                    selected_b = electrode_table.loc[electrode_table["group"].eq(group_b), feature].dropna().to_numpy(float)
                    if len(selected_a) < 2 or len(selected_b) < 2:
                        t_stat = t_p = u_stat = u_p = np.nan
                    else:
                        welch = ttest_ind(selected_a, selected_b, equal_var=False)
                        mann = mannwhitneyu(selected_a, selected_b, alternative="two-sided")
                        t_stat, t_p = float(welch.statistic), float(welch.pvalue)
                        u_stat, u_p = float(mann.statistic), float(mann.pvalue)
                    rows.append({
                        "dataset_id": dataset_id,
                        "electrode": electrode,
                        "group_a": group_a,
                        "group_b": group_b,
                        "feature": feature,
                        "n_a": len(selected_a),
                        "n_b": len(selected_b),
                        "mean_a": float(np.mean(selected_a)) if len(selected_a) else np.nan,
                        "mean_b": float(np.mean(selected_b)) if len(selected_b) else np.nan,
                        "welch_t": t_stat,
                        "welch_p": t_p,
                        "mann_whitney_u": u_stat,
                        "mann_whitney_p": u_p,
                    })
    result = pd.DataFrame.from_records(rows)
    if not result.empty:
        result["welch_p_fdr_bh"] = _bh(result["welch_p"].to_numpy(float))
        result["mann_whitney_p_fdr_bh"] = _bh(result["mann_whitney_p"].to_numpy(float))
        result["fdr_alpha"] = config.fdr_alpha
    return result


def _partial_spearman(frame: pd.DataFrame, feature: str, outcome: str) -> tuple[float, float, int]:
    columns = [feature, outcome]
    covariates: list[str] = []
    if "age_years" in frame and frame["age_years"].notna().any():
        columns.append("age_years")
        covariates.append("age_years")
    if "sex" in frame:
        encoded = frame["sex"].astype(str).str.lower().isin({"m", "male", "1"}).astype(float)
        frame = frame.assign(sex_male=encoded)
        if encoded.nunique() > 1:
            columns.append("sex_male")
            covariates.append("sex_male")
    selected = frame[columns].apply(pd.to_numeric, errors="coerce").dropna()
    if len(selected) < max(6, len(covariates) + 4):
        return np.nan, np.nan, len(selected)
    y = rankdata(selected[outcome].to_numpy(float))
    x = rankdata(selected[feature].to_numpy(float))
    if covariates:
        z = np.column_stack([np.ones(len(selected)), *[rankdata(selected[name].to_numpy(float)) for name in covariates]])
        y = y - z @ np.linalg.lstsq(z, y, rcond=None)[0]
        x = x - z @ np.linalg.lstsq(z, x, rcond=None)[0]
        if np.std(x) == 0.0 or np.std(y) == 0.0:
            return np.nan, np.nan, len(selected)
        correlation = pearsonr(x, y)
    else:
        correlation = spearmanr(selected[feature], selected[outcome])
    return float(correlation.statistic), float(correlation.pvalue), len(selected)


def clinical_correlations(table: pd.DataFrame, config: GlobalConfig) -> pd.DataFrame:
    feature_columns = [column for column in table.columns if column.startswith(("bout__", "within_bout__", "entropy__"))]
    # One row per biological participant/condition prevents electrodes and repeated
    # epochs from being mistaken for independent clinical observations.
    subject = table.groupby(["dataset_id", "participant_id", "group"], dropna=False)[feature_columns + ["age_years", "sex", "updrs", "moca", "mmse"]].mean(numeric_only=True).reset_index()
    # Restore nonnumeric metadata needed by the partial model.
    metadata = table[["dataset_id", "participant_id", "group", "age_years", "sex", "updrs", "moca", "mmse"]].drop_duplicates(["dataset_id", "participant_id", "group"])
    subject = subject.drop(columns=["age_years", "updrs", "moca", "mmse", "sex"], errors="ignore").merge(metadata, on=["dataset_id", "participant_id", "group"], how="left")
    rows: list[dict[str, Any]] = []
    for (dataset_id, group), selected in subject.groupby(["dataset_id", "group"], sort=False):
        if not str(group).lower().startswith("pd"):
            continue
        for outcome in ("updrs", "moca", "mmse"):
            if outcome not in selected or selected[outcome].notna().sum() < 6:
                continue
            for feature in feature_columns:
                rho, p_value, n = _partial_spearman(selected.copy(), feature, outcome)
                rows.append({
                    "dataset_id": dataset_id,
                    "group": group,
                    "outcome": outcome,
                    "feature": feature,
                    "n_subjects": n,
                    "partial_spearman_rho": rho,
                    "p_value": p_value,
                    "covariates": "age_years,sex",
                })
    result = pd.DataFrame.from_records(rows)
    if not result.empty:
        result["p_fdr_bh"] = result.groupby(["dataset_id", "outcome"], sort=False)["p_value"].transform(
            lambda values: _bh(values.to_numpy(float))
        )
        result["fdr_alpha"] = config.fdr_alpha
    return result


def _topomap(table: pd.DataFrame, config: GlobalConfig, dataset_id: str, domain: str, feature_template: str, output: Path) -> None:
    import matplotlib.pyplot as plt

    selected = table.loc[table["dataset_id"].eq(dataset_id)].copy()
    if selected.empty:
        return
    feature_columns = [column for column in selected if column.startswith(feature_template)]
    if not feature_columns:
        return
    groups = sorted(selected["group"].dropna().unique())
    montage = mne.channels.make_standard_montage("standard_1005")
    positions = montage.get_positions()["ch_pos"]
    channels = [channel for channel in selected["electrode"].drop_duplicates() if channel in positions]
    if not channels:
        return
    fig, axes = plt.subplots(len(groups), len(feature_columns), figsize=(3.2 * len(feature_columns), 3.0 * len(groups)), squeeze=False)
    for row_index, group in enumerate(groups):
        for column_index, feature in enumerate(feature_columns):
            values = selected.loc[selected["group"].eq(group)].groupby("electrode")[feature].mean()
            values = np.asarray([values.get(channel, np.nan) for channel in channels], dtype=float)
            valid = np.isfinite(values)
            if valid.sum() < 3:
                axes[row_index, column_index].axis("off")
                continue
            info = mne.create_info(channels, sfreq=100.0, ch_types="eeg")
            info.set_montage(montage, on_missing="ignore", verbose="ERROR")
            mne.viz.plot_topomap(values, info, axes=axes[row_index, column_index], show=False, contours=0, names=None)
            axes[row_index, column_index].set_title(f"{group}: {feature.split('__')[-1]}")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def run_global_pipeline(
    config_path: str | Path,
    *,
    skip_figures: bool = False,
    overwrite: bool = False,
    dataset_ids: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    config = load_global_config(config_path)
    output = config.output_root
    output.mkdir(parents=True, exist_ok=True)
    canonical_path = output / "canonical" / "recordings.csv.gz"
    requested_ids = tuple(dataset.dataset_id for dataset in config.enabled_datasets) if dataset_ids is None else tuple(dataset_ids)
    available_ids = {dataset.dataset_id for dataset in config.enabled_datasets}
    unknown = sorted(set(requested_ids) - available_ids)
    if unknown:
        raise ValueError(f"Unknown or disabled dataset(s): {unknown}; enabled choices: {sorted(available_ids)}")
    if overwrite or not canonical_path.exists():
        canonical = convert_config(config, dataset_ids=requested_ids)
    else:
        canonical = read_canonical_table(canonical_path)
        cached_ids = tuple(sorted(canonical["dataset_id"].dropna().unique()))
        if set(cached_ids) != set(requested_ids):
            canonical = convert_config(config, dataset_ids=requested_ids)
        else:
            canonical = canonical.loc[canonical["dataset_id"].isin(requested_ids)].copy()
    recording_tables: list[pd.DataFrame] = []
    diagnostics: list[dict[str, Any]] = []
    for record in canonical.to_dict(orient="records"):
        features, info = _analyze_recording(record, config)
        recording_tables.append(features)
        diagnostics.append({"recording_id": record["recording_id"], "n_channels": len(info["channels"]), "sampling_frequency_hz": info["sfreq"]})
    feature_table = pd.concat(recording_tables, ignore_index=True)
    _write_table(feature_table, output / "metrics" / "recording_features.csv.gz")
    subject_features = feature_table.groupby(["dataset_id", "participant_id", "group"], dropna=False).mean(numeric_only=True).reset_index()
    _write_table(subject_features, output / "metrics" / "subject_features.csv.gz")
    stats = group_statistics(feature_table, config)
    correlations = clinical_correlations(feature_table, config)
    _write_table(stats, output / "statistics" / "group_statistics.csv.gz")
    _write_table(correlations, output / "statistics" / "clinical_correlations.csv.gz")
    if not skip_figures:
        for dataset in config.enabled_datasets:
            if dataset.dataset_id not in requested_ids:
                continue
            _topomap(feature_table, config, dataset.dataset_id, "psd", "psd__", output / "figures" / dataset.dataset_id / "psd_topomaps.png")
            _topomap(feature_table, config, dataset.dataset_id, "entropy", "entropy__", output / "figures" / dataset.dataset_id / "entropy_topomaps.png")
            _topomap(feature_table, config, dataset.dataset_id, "within_bout", "within_bout__", output / "figures" / dataset.dataset_id / "within_bout_entropy_topomaps.png")
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "output_root": str(output),
        "n_datasets": int(canonical["dataset_id"].nunique()),
        "n_recordings": int(len(canonical)),
        "n_subjects": int(canonical["participant_id"].nunique()),
        "groups": canonical["group"].value_counts().to_dict(),
        "entropy_metrics": list(ENTROPY_METRICS),
        "memory_policy": "MNE epochs are read in block_epochs-sized blocks; raw samples are not saved in feature tables",
        "diagnostics": diagnostics,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
