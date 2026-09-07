"""Subject-level PSD, entropy, bout, and clinical analyses.

The analysis unit is one cleaned recording (a subject/session/condition when a
dataset has sessions).  All accepted epochs for that unit are loaded together,
concatenated in recording order, analyzed, and immediately reduced to compact
feature tables and permutation-pattern sufficient statistics.  Raw samples
and ordinal symbol sequences are never written to the intermediate state.
"""

from __future__ import annotations

import json
import itertools
import math
import re
import tempfile
from itertools import combinations
from pathlib import Path
from typing import Any

import mne
import numpy as np
import pandas as pd
import xarray as xr
from scipy.signal import hilbert
from scipy.stats import mannwhitneyu, pearsonr, rankdata, spearmanr, ttest_ind
from tqdm.auto import tqdm

from analyses.bouts.metrics import ordinal_counts, shannon_metrics_from_counts
from analyses.ordinal.metrics import (
    filter_epoch_data,
    metrics_from_probabilities,
    ordinal_probabilities,
    weighted_permutation_entropy_epoch_data,
)
from analyses.psd.metrics import compute_subject_electrode_psd
from analyses.scale_free.metrics import fit_specparam_spectrum

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


def _write_table_atomic(table: pd.DataFrame, path: Path) -> None:
    """Write one subject result so an interrupted run leaves no partial CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", suffix=path.suffix, prefix=f".{path.stem}.", dir=path.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
        table.to_csv(temporary, index=False, compression="gzip", float_format="%.10g")
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


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
        "n_epochs": 0,
        "n_patterns": 0,
        "n_ties": 0,
    }


def _analyze_recording(record: dict[str, Any], config: GlobalConfig) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = Path(record["epoch_path"])
    # The requested analysis unit is the complete cleaned recording.  This
    # deliberately preloads all accepted epochs for one subject/condition;
    # no epoch block is analyzed independently or discarded before the
    # subject-level accumulators are complete.
    epochs = mne.read_epochs(path, preload=True, verbose="ERROR")
    eeg_picks = list(mne.pick_types(epochs.info, eeg=True, exclude=[]))
    if not eeg_picks:
        raise ValueError(f"{path}: no EEG channels found")
    channels = [epochs.ch_names[index] for index in eeg_picks]
    sfreq = float(epochs.info["sfreq"])
    n_epochs = len(epochs)
    n_samples = int(len(epochs.times))
    if n_epochs < 1:
        raise ValueError(f"{path}: no accepted epochs")

    data = np.asarray(epochs.get_data(picks=eeg_picks, copy=True), dtype=np.float32)
    signal = xr.DataArray(
        data,
        dims=("epoch", "channel", "time"),
        coords={
            "epoch": np.arange(n_epochs, dtype=np.int64),
            "channel": channels,
            "time": np.arange(n_samples, dtype=np.int64) / sfreq,
        },
        name="eeg_subject",
    )
    # The accepted epochs are concatenated in temporal/file order.  Every
    # analysis below uses this complete recording-level signal.  In
    # particular, ordinal windows and Hilbert bouts are allowed to span the
    # former epoch boundaries, as requested by the global analysis contract.
    concatenated = xr.DataArray(
        signal.data.transpose(1, 0, 2).reshape(len(channels), -1),
        dims=("channel", "sample"),
        coords={
            "channel": channels,
            "sample": np.arange(n_epochs * n_samples, dtype=np.int64) / sfreq,
        },
        name="eeg_subject_concatenated",
    )

    frequencies, mean_psd = compute_subject_electrode_psd(
        signal.data,
        sfreq,
        fmin=min(low for low, _ in config.bands.values()),
        fmax=max(high for _, high in config.bands.values()),
    )
    mean_psd = np.asarray(mean_psd, dtype=np.float64)
    entropy: dict[str, list[dict[str, Any]]] = {
        band: [
            {str(dx): _new_entropy_state(dx) for dx in config.permutation_dimensions}
            for _ in channels
        ]
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
                "within": {
                    str(dx): _new_entropy_state(dx)
                    for dx in config.permutation_dimensions
                },
            }
            for _ in channels
        ]
        for band in config.bands
    }

    for band, (low, high) in config.bands.items():
        filtered = filter_epoch_data(
            concatenated.data[None, ...].astype(np.float64, copy=False),
            sfreq=sfreq,
            low_hz=low,
            high_hz=high,
            order=4,
        )[0]
        center = max((low + high) / 2.0, 0.1)
        minimum_samples = max(
            max(config.permutation_dimensions) + 1,
            int(math.ceil(config.bout_minimum_cycles * sfreq / center)),
        )
        for channel_index in range(len(channels)):
            for dx in config.permutation_dimensions:
                state = entropy[band][channel_index][str(dx)]
                probabilities, n_patterns, n_ties = ordinal_probabilities(
                    filtered[channel_index][None, :],
                    dx=dx,
                    tau=config.delay_samples,
                )
                state["counts"] += np.rint(probabilities * n_patterns).astype(np.int64)
                state["n_patterns"] += n_patterns
                state["n_ties"] += n_ties
                state["weighted_sum"] += weighted_permutation_entropy_epoch_data(
                    filtered[channel_index][None, :], dx=dx, tau=config.delay_samples
                )
                state["n_epochs"] += 1

            bout_state = bouts[band][channel_index]
            epoch_signal = filtered[channel_index]
            amplitude = np.abs(hilbert(epoch_signal))
            threshold = np.percentile(amplitude, config.bout_threshold_percentile)
            for bout_start, bout_stop in _runs(amplitude >= threshold, minimum_samples):
                segment = epoch_signal[bout_start:bout_stop]
                duration = (bout_stop - bout_start) / sfreq
                bout_state["n_bouts"] += 1
                bout_state["n_bout_samples"] += bout_stop - bout_start
                bout_state["duration_sum"] += duration
                bout_state["duration_sq_sum"] += duration * duration
                bout_state["amplitude_sum"] += float(np.mean(amplitude[bout_start:bout_stop]))
                bout_state["cycle_sum"] += duration * center
                for dx in config.permutation_dimensions:
                    within = bout_state["within"][str(dx)]
                    counts, ties = ordinal_counts(
                        segment, dx=dx, tau=config.delay_samples
                    )
                    within["counts"] += counts
                    within["n_patterns"] += int(counts.sum())
                    within["n_ties"] += ties
                    if int(counts.sum()):
                        within["weighted_sum"] += float(
                            weighted_permutation_entropy_epoch_data(
                                segment[None, :], dx=dx, tau=config.delay_samples
                            )
                        ) * int(counts.sum())

    fit_low, fit_high = (
        float(value) for value in config.aperiodic_settings["frequency_range_hz"]
    )
    aperiodic_bands = {
        band: (max(float(low), fit_low), min(float(high), fit_high))
        for band, (low, high) in config.bands.items()
        if max(float(low), fit_low) < min(float(high), fit_high)
    }
    aperiodic_by_channel: list[dict[str, float]] = []
    for channel_index in range(len(channels)):
        try:
            aperiodic, _, _ = fit_specparam_spectrum(
                frequencies,
                mean_psd[channel_index],
                aperiodic_bands,
                config.aperiodic_settings,
            )
        except Exception:
            aperiodic = {}
        numeric_values: dict[str, float] = {}
        for metric, value in aperiodic.items():
            if isinstance(value, str):
                continue
            try:
                numeric_values[metric] = float(value)
            except (TypeError, ValueError):
                continue
        aperiodic_by_channel.append(numeric_values)

    rows: list[dict[str, Any]] = []
    duration_minutes = n_epochs * n_samples / sfreq / 60.0
    metadata = {key: record.get(key, "") for key in CANONICAL_COLUMNS if key not in {"epoch_path", "raw_path"}}
    for channel_index, electrode in enumerate(channels):
        total_mask = (frequencies >= min(low for low, _ in config.bands.values())) & (
            frequencies <= max(high for _, high in config.bands.values())
        )
        total_power = float(np.trapezoid(mean_psd[channel_index, total_mask], frequencies[total_mask]))
        row = {**metadata, "electrode": electrode, "sampling_frequency_hz": sfreq, "n_epochs": n_epochs}
        for metric, value in aperiodic_by_channel[channel_index].items():
            row[f"aperiodic__broadband__{metric}"] = value
        for band, (low, high) in config.bands.items():
            band_mask = (frequencies >= low) & (frequencies <= high)
            band_power = float(np.trapezoid(mean_psd[channel_index, band_mask], frequencies[band_mask]))
            row[f"psd__{band}__absolute_power"] = band_power
            row[f"psd__{band}__relative_power"] = band_power / total_power if total_power > 0 else np.nan
            for dx in config.permutation_dimensions:
                state = entropy[band][channel_index][str(dx)]
                suffix = f"__D{dx}"
                if state["n_patterns"]:
                    values = metrics_from_probabilities(
                        state["counts"] / state["n_patterns"], dx=dx
                    )
                    for metric in ENTROPY_METRICS:
                        value = (
                            state["weighted_sum"] / state["n_epochs"]
                            if metric == "weighted_permutation_entropy"
                            else values[metric]
                        )
                        row[f"entropy__{band}__{metric}{suffix}"] = value
                else:
                    for metric in ENTROPY_METRICS:
                        row[f"entropy__{band}__{metric}{suffix}"] = np.nan
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
            for dx in config.permutation_dimensions:
                within = bout_state["within"][str(dx)]
                suffix = f"__D{dx}"
                if within["n_patterns"]:
                    values = shannon_metrics_from_counts(within["counts"], dx=dx)
                    for metric in ENTROPY_METRICS:
                        row[f"within_bout__{band}__{metric}{suffix}"] = (
                            within["weighted_sum"] / within["n_patterns"]
                            if metric == "weighted_permutation_entropy"
                            else values[metric]
                        )
                else:
                    for metric in ENTROPY_METRICS:
                        row[f"within_bout__{band}__{metric}{suffix}"] = np.nan
        rows.append(row)
    pattern_arrays: dict[str, np.ndarray] = {}
    for dx in config.permutation_dimensions:
        pattern_arrays[f"permutation_order__D{dx}"] = np.asarray(
            list(itertools.permutations(range(dx))), dtype=np.int8
        )
        for band in config.bands:
            for channel_index, channel in enumerate(channels):
                safe_channel = f"ch{channel_index:03d}_{_safe_filename(channel)}"
                full_state = entropy[band][channel_index][str(dx)]
                within_state = bouts[band][channel_index]["within"][str(dx)]
                pattern_arrays[
                    f"full__{band}__{safe_channel}__D{dx}__counts"
                ] = full_state["counts"].astype(np.uint64, copy=False)
                pattern_arrays[
                    f"full__{band}__{safe_channel}__D{dx}__weighted_sum"
                ] = np.asarray([full_state["weighted_sum"]], dtype=np.float64)
                pattern_arrays[
                    f"full__{band}__{safe_channel}__D{dx}__n_patterns"
                ] = np.asarray([full_state["n_patterns"]], dtype=np.uint64)
                pattern_arrays[
                    f"within__{band}__{safe_channel}__D{dx}__counts"
                ] = within_state["counts"].astype(np.uint64, copy=False)
                pattern_arrays[
                    f"within__{band}__{safe_channel}__D{dx}__weighted_sum"
                ] = np.asarray([within_state["weighted_sum"]], dtype=np.float64)
                pattern_arrays[
                    f"within__{band}__{safe_channel}__D{dx}__n_patterns"
                ] = np.asarray([within_state["n_patterns"]], dtype=np.uint64)
    return pd.DataFrame.from_records(rows), {
        "channels": channels,
        "sfreq": sfreq,
        "frequencies": np.asarray(frequencies, dtype=float),
        # Keep only the channel-averaged spectrum for plotting.  The full
        # channel x frequency PSD remains local to this recording; it is not
        # retained in the subject feature table or global aggregation state.
        "mean_psd": np.mean(mean_psd, axis=0),
        "pattern_arrays": pattern_arrays,
    }


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
    feature_columns = [column for column in table.columns if column.startswith(("psd__", "aperiodic__", "entropy__", "bout__", "within_bout__"))]
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


def _safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def _analysis_signature(config: GlobalConfig) -> str:
    """Return the analysis contract used to validate resumable subject files."""
    contract = {
        "schema": 3,
        "bands": config.bands,
        "permutation_dimensions": config.permutation_dimensions,
        "embedding_dimension": config.embedding_dimension,
        "delay_samples": config.delay_samples,
        "bout_threshold_percentile": config.bout_threshold_percentile,
        "bout_minimum_cycles": config.bout_minimum_cycles,
        "aperiodic_settings": config.aperiodic_settings,
    }
    return json.dumps(contract, sort_keys=True, separators=(",", ":"))


def _subject_cache_paths(output: Path, record: dict[str, Any]) -> dict[str, Path]:
    dataset = _safe_filename(record["dataset_id"])
    recording = _safe_filename(record["recording_id"])
    directory = output / "intermediate" / "subjects" / dataset
    return {
        "features": directory / f"{recording}_features.csv.gz",
        "patterns": directory / f"{recording}_permutation_patterns.npz",
        "metadata": directory / f"{recording}_metadata.json",
    }


def _save_subject_cache(
    output: Path,
    record: dict[str, Any],
    features: pd.DataFrame,
    info: dict[str, Any],
    config: GlobalConfig,
) -> None:
    """Persist one complete subject result and its D=3..7 pattern state."""
    paths = _subject_cache_paths(output, record)
    paths["features"].parent.mkdir(parents=True, exist_ok=True)
    epoch_stat = Path(record["epoch_path"]).stat()
    arrays = dict(info["pattern_arrays"])
    arrays["frequencies"] = np.asarray(info["frequencies"], dtype=np.float64)
    arrays["mean_psd"] = np.asarray(info["mean_psd"], dtype=np.float64)

    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", suffix=".npz", prefix=f".{paths['patterns'].stem}.",
            dir=paths["patterns"].parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
        np.savez_compressed(temporary, **arrays)
        temporary.replace(paths["patterns"])
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()

    _write_table_atomic(features, paths["features"])
    metadata = {
        "schema_version": 3,
        "analysis_signature": _analysis_signature(config),
        "dataset_id": record["dataset_id"],
        "recording_id": record["recording_id"],
        "participant_id": record["participant_id"],
        "epoch_path": str(Path(record["epoch_path"]).resolve()),
        "epoch_size": int(epoch_stat.st_size),
        "epoch_mtime_ns": int(epoch_stat.st_mtime_ns),
        "n_rows": int(len(features)),
        "channels": list(info["channels"]),
        "sfreq": float(info["sfreq"]),
        "permutation_dimensions": list(config.permutation_dimensions),
        "aperiodic_analysis": config.aperiodic_settings,
        "pattern_file": paths["patterns"].name,
        "feature_file": paths["features"].name,
    }
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", prefix=f".{paths['metadata'].stem}.",
            dir=paths["metadata"].parent, encoding="utf-8", delete=False
        ) as handle:
            temporary = Path(handle.name)
            json.dump(metadata, handle, indent=2)
        temporary.replace(paths["metadata"])
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _load_subject_cache(
    output: Path,
    record: dict[str, Any],
    config: GlobalConfig,
) -> tuple[pd.DataFrame, dict[str, Any]] | None:
    """Load a complete subject cache, or return ``None`` when it is stale."""
    paths = _subject_cache_paths(output, record)
    if not all(path.exists() for path in paths.values()):
        return None
    try:
        metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
        epoch_path = Path(record["epoch_path"])
        epoch_stat = epoch_path.stat()
        if (
            metadata.get("schema_version") != 3
            or metadata.get("analysis_signature") != _analysis_signature(config)
            or metadata.get("epoch_path") != str(epoch_path.resolve())
            or metadata.get("epoch_size") != int(epoch_stat.st_size)
            or metadata.get("epoch_mtime_ns") != int(epoch_stat.st_mtime_ns)
            or metadata.get("permutation_dimensions") != list(config.permutation_dimensions)
        ):
            return None
        features = pd.read_csv(paths["features"], compression="infer")
        if len(features) != int(metadata.get("n_rows", -1)):
            return None
        with np.load(paths["patterns"], allow_pickle=False) as bundle:
            required = [
                "frequencies",
                "mean_psd",
                *[f"permutation_order__D{dx}" for dx in config.permutation_dimensions],
            ]
            if any(key not in bundle for key in required):
                return None
            frequencies = np.asarray(bundle["frequencies"], dtype=float)
            mean_psd = np.asarray(bundle["mean_psd"], dtype=float)
        return features, {
            "channels": list(metadata.get("channels", [])),
            "sfreq": float(metadata["sfreq"]),
            "frequencies": frequencies,
            "mean_psd": mean_psd,
            "cache_reused": True,
        }
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None


def _feature_map(
    selected: pd.DataFrame,
    group: str,
    feature: str,
    channels: list[str],
) -> np.ndarray:
    values = selected.loc[selected["group"].eq(group)].groupby("electrode")[feature].mean()
    return np.asarray([values.get(channel, np.nan) for channel in channels], dtype=float)


def _stable_limits(values: np.ndarray, *, symmetric: bool = False) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return (-1.0, 1.0)
    if symmetric:
        bound = float(np.max(np.abs(finite)))
        bound = max(bound, 1e-12)
        return (-bound, bound)
    low, high = float(np.min(finite)), float(np.max(finite))
    if np.isclose(low, high):
        padding = max(abs(low) * 0.05, 1e-12)
        low -= padding
        high += padding
    return low, high


def _topomap(
    table: pd.DataFrame,
    config: GlobalConfig,
    dataset_id: str,
    domain: str,
    feature_template: str,
    output: Path,
) -> None:
    import matplotlib.pyplot as plt

    selected = table.loc[table["dataset_id"].eq(dataset_id)].copy()
    if selected.empty:
        return
    feature_columns = [column for column in selected if column.startswith(feature_template)]
    if feature_template in {"entropy__", "within_bout__"}:
        feature_columns = [
            column for column in feature_columns
            if column.endswith(f"__D{config.embedding_dimension}")
        ]
    if not feature_columns:
        return
    groups = sorted(selected["group"].dropna().unique())
    montage = mne.channels.make_standard_montage("standard_1005")
    positions = montage.get_positions()["ch_pos"]
    channels = [channel for channel in selected["electrode"].drop_duplicates() if channel in positions]
    if not channels:
        return
    maps: dict[str, list[np.ndarray]] = {
        feature: [_feature_map(selected, group, feature, channels) for group in groups]
        for feature in feature_columns
    }
    limits = {
        feature: _stable_limits(np.concatenate(group_values))
        for feature, group_values in maps.items()
    }
    fig, axes = plt.subplots(
        len(groups),
        len(feature_columns),
        figsize=(3.4 * len(feature_columns), 3.2 * len(groups)),
        squeeze=False,
        constrained_layout=True,
    )
    images = {}
    info = mne.create_info(channels, sfreq=100.0, ch_types="eeg")
    info.set_montage(montage, on_missing="ignore", verbose="ERROR")
    for row_index, group in enumerate(groups):
        for column_index, feature in enumerate(feature_columns):
            values = maps[feature][row_index]
            valid = np.isfinite(values)
            if valid.sum() < 3:
                axes[row_index, column_index].axis("off")
                continue
            image, _ = mne.viz.plot_topomap(
                values,
                info,
                axes=axes[row_index, column_index],
                show=False,
                contours=0,
                names=None,
                vlim=limits[feature],
            )
            images.setdefault(feature, image)
            axes[row_index, column_index].set_title(f"{group}: {feature.split('__')[-1]}")
    for column_index, feature in enumerate(feature_columns):
        if feature in images:
            fig.colorbar(images[feature], ax=axes[:, column_index].tolist(), shrink=0.75)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def _contrast_topomap(
    table: pd.DataFrame,
    statistics: pd.DataFrame,
    config: GlobalConfig,
    dataset_id: str,
    feature_template: str,
    group_a: str,
    group_b: str,
    output: Path,
) -> None:
    """Plot group_b - group_a using one symmetric scale per feature.

    White electrode markers indicate electrodes passing the configured
    Benjamini-Hochberg FDR threshold for the Welch test.
    """
    import matplotlib.pyplot as plt

    selected = table.loc[table["dataset_id"].eq(dataset_id)].copy()
    feature_columns = [column for column in selected if column.startswith(feature_template)]
    if feature_template in {"entropy__", "within_bout__"}:
        feature_columns = [
            column for column in feature_columns
            if column.endswith(f"__D{config.embedding_dimension}")
        ]
    if selected.empty or not feature_columns:
        return
    montage = mne.channels.make_standard_montage("standard_1005")
    positions = montage.get_positions()["ch_pos"]
    channels = [channel for channel in selected["electrode"].drop_duplicates() if channel in positions]
    if not channels:
        return
    info = mne.create_info(channels, sfreq=100.0, ch_types="eeg")
    info.set_montage(montage, on_missing="ignore", verbose="ERROR")
    pair_stats = statistics.loc[
        statistics["dataset_id"].eq(dataset_id)
        & statistics["group_a"].eq(group_a)
        & statistics["group_b"].eq(group_b)
    ]
    fig, axes = plt.subplots(
        1,
        len(feature_columns),
        figsize=(3.4 * len(feature_columns), 3.5),
        squeeze=False,
        constrained_layout=True,
    )
    images = {}
    mask_params = {
        "marker": "o",
        "markerfacecolor": "white",
        "markeredgecolor": "black",
        "linewidth": 0,
        "markersize": 4,
    }
    for column_index, feature in enumerate(feature_columns):
        values_a = _feature_map(selected, group_a, feature, channels)
        values_b = _feature_map(selected, group_b, feature, channels)
        differences = values_b - values_a
        feature_stats = pair_stats.loc[pair_stats["feature"].eq(feature)].set_index("electrode")
        p_values = np.asarray(
            [feature_stats.get("welch_p_fdr_bh", pd.Series(dtype=float)).get(channel, np.nan) for channel in channels],
            dtype=float,
        )
        mask = np.isfinite(p_values) & (p_values <= config.fdr_alpha) & np.isfinite(differences)
        valid = np.isfinite(differences)
        if valid.sum() < 3:
            axes[0, column_index].axis("off")
            continue
        image, _ = mne.viz.plot_topomap(
            differences,
            info,
            axes=axes[0, column_index],
            show=False,
            contours=0,
            names=None,
            cmap="RdBu_r",
            vlim=_stable_limits(differences, symmetric=True),
            mask=mask,
            mask_params=mask_params,
        )
        images[feature] = image
        axes[0, column_index].set_title(feature.split("__")[-1])
    for column_index, feature in enumerate(feature_columns):
        if feature in images:
            fig.colorbar(images[feature], ax=axes[0, column_index], shrink=0.75)
    fig.suptitle(f"{dataset_id}: {group_b} - {group_a} ({feature_template.rstrip('_')})")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def _plot_psd_spectra(
    spectra: list[dict[str, Any]],
    dataset_id: str,
    output: Path,
) -> None:
    """Plot recording-level mean PSD with a 95% CI for each population."""
    import matplotlib.pyplot as plt

    selected = [item for item in spectra if item["dataset_id"] == dataset_id]
    if not selected:
        return
    reference = np.asarray(selected[0]["frequencies"], dtype=float)
    groups = sorted({str(item["group"]) for item in selected})
    fig, axis = plt.subplots(figsize=(9, 5.5))
    for group in groups:
        group_items = [item for item in selected if str(item["group"]) == group]
        curves = []
        for item in group_items:
            frequency = np.asarray(item["frequencies"], dtype=float)
            curve = np.asarray(item["mean_psd"], dtype=float)
            if not np.array_equal(frequency, reference):
                curve = np.interp(reference, frequency, curve, left=np.nan, right=np.nan)
            curves.append(curve)
        values = np.asarray(curves, dtype=float)
        mean = np.nanmean(values, axis=0)
        n = np.sum(np.isfinite(values), axis=0)
        standard_deviation = np.nanstd(values, axis=0, ddof=1)
        standard_error = standard_deviation / np.sqrt(np.maximum(n, 1))
        interval = np.where(n > 1, 1.96 * standard_error, 0.0)
        line = axis.plot(reference, mean, linewidth=1.8, label=f"{group} (n={len(group_items)})")[0]
        axis.fill_between(
            reference,
            np.maximum(mean - interval, np.finfo(float).tiny),
            mean + interval,
            color=line.get_color(),
            alpha=0.2,
        )
    axis.set_xlabel("Frequency (Hz)")
    axis.set_ylabel("Mean PSD across EEG electrodes (µV²/Hz)")
    axis.set_title(f"{dataset_id}: mean PSD ± 95% CI")
    axis.set_xlim(left=0.0)
    axis.set_yscale("log")
    axis.grid(True, alpha=0.25)
    axis.legend()
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def _plot_clinical_scatter(
    table: pd.DataFrame,
    dataset_id: str,
    outcome: str,
    family: str,
    config: GlobalConfig,
    output: Path,
) -> None:
    """Plot subject-level PD features against one available clinical score."""
    import matplotlib.pyplot as plt

    prefix = f"{family}__"
    feature_columns = [column for column in table if column.startswith(prefix)]
    if family in {"within_bout", "entropy"}:
        feature_columns = [
            column for column in feature_columns
            if column.endswith(f"__D{config.embedding_dimension}")
        ]
    if not feature_columns or outcome not in table:
        return
    selected = table.loc[table["dataset_id"].eq(dataset_id)].copy()
    selected = selected.loc[selected["group"].astype(str).str.lower().str.startswith("pd")]
    subject = (
        selected.groupby(["participant_id", "group"], dropna=False)[feature_columns + [outcome]]
        .mean(numeric_only=True)
        .reset_index()
    )
    usable = [feature for feature in feature_columns if subject[feature].notna().sum() >= 2]
    if subject[outcome].notna().sum() < 2 or not usable:
        return
    ncols = min(4, len(usable))
    nrows = int(math.ceil(len(usable) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.3 * nrows), squeeze=False)
    colors = {group: f"C{index}" for index, group in enumerate(sorted(subject["group"].dropna().unique()))}
    for index, feature in enumerate(usable):
        axis = axes.flat[index]
        points = subject[["group", outcome, feature]].dropna()
        for group, group_points in points.groupby("group", sort=False):
            axis.scatter(
                group_points[outcome],
                group_points[feature],
                s=28,
                alpha=0.8,
                color=colors.get(group, "C0"),
                label=str(group),
            )
        if len(points) >= 2 and points[outcome].nunique() > 1:
            slope, intercept = np.polyfit(points[outcome].to_numpy(float), points[feature].to_numpy(float), 1)
            x_values = np.linspace(points[outcome].min(), points[outcome].max(), 50)
            axis.plot(x_values, intercept + slope * x_values, color="black", linewidth=1.0)
        if len(points) >= 3 and points[outcome].nunique() > 1 and points[feature].nunique() > 1:
            rho, p_value = spearmanr(points[outcome], points[feature])
            annotation = f"rho={rho:.2f}, p={p_value:.3g}"
        else:
            annotation = f"n={len(points)}"
        axis.text(0.03, 0.97, annotation, transform=axis.transAxes, va="top", fontsize=8)
        short_name = feature.removeprefix(prefix).replace("__", " / ")
        axis.set_title(short_name)
        axis.set_xlabel(outcome.upper())
        axis.set_ylabel(family.replace("_", " "))
        axis.grid(True, alpha=0.2)
    for axis in axes.flat[len(usable):]:
        axis.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right")
    fig.suptitle(f"{dataset_id}: PD {family.replace('_', ' ')} vs {outcome.upper()}")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def _plot_entropy_plane(
    table: pd.DataFrame,
    dataset_id: str,
    family: str,
    vertical_metric: str,
    config: GlobalConfig,
    output: Path,
) -> None:
    """Plot subject-level HxC or HxF planes for each frequency band.

    Electrode values are averaged within participant and condition before
    plotting, so the points represent biological observations rather than
    treating electrodes as independent participants.
    """
    import matplotlib.pyplot as plt

    prefix = f"{family}__"
    metric_prefix = f"{prefix}"
    entropy_columns = [column for column in table if column.startswith(metric_prefix)]
    bands = sorted(
        {
            column.removeprefix(metric_prefix).split("__", 1)[0]
            for column in entropy_columns
            if "__" in column.removeprefix(metric_prefix)
        }
    )
    if not bands:
        return
    selected = table.loc[table["dataset_id"].eq(dataset_id)].copy()
    if selected.empty:
        return
    subject_columns = ["participant_id", "group"]
    metric_columns = []
    for band in bands:
        metric_columns.extend(
            [
                f"{family}__{band}__entropy__D{config.embedding_dimension}",
                f"{family}__{band}__{vertical_metric}__D{config.embedding_dimension}",
            ]
        )
    metric_columns = [column for column in metric_columns if column in selected]
    subject = (
        selected.groupby(subject_columns, dropna=False)[metric_columns]
        .mean(numeric_only=True)
        .reset_index()
    )
    groups = sorted(str(value) for value in subject["group"].dropna().unique())
    if not groups:
        return
    colors = {group: f"C{index}" for index, group in enumerate(groups)}
    fig, axes = plt.subplots(
        1,
        len(bands),
        figsize=(4.4 * len(bands), 4.0),
        squeeze=False,
        constrained_layout=True,
    )
    plotted = False
    for column_index, band in enumerate(bands):
        axis = axes[0, column_index]
        horizontal = f"{family}__{band}__entropy__D{config.embedding_dimension}"
        vertical = f"{family}__{band}__{vertical_metric}__D{config.embedding_dimension}"
        if horizontal not in subject or vertical not in subject:
            axis.axis("off")
            continue
        for group in groups:
            points = subject.loc[
                subject["group"].astype(str).eq(group),
                [horizontal, vertical],
            ].dropna()
            if points.empty:
                continue
            plotted = True
            axis.scatter(
                points[horizontal],
                points[vertical],
                s=34,
                alpha=0.8,
                color=colors[group],
                label=f"{group} (n={len(points)})",
            )
        axis.set_title(band)
        axis.set_xlabel("Entropy H")
        axis.set_ylabel("Complexity C" if vertical_metric == "complexity" else "Fisher information F")
        axis.grid(True, alpha=0.2)
    if not plotted:
        plt.close(fig)
        return
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right")
    plane_name = "H×C" if vertical_metric == "complexity" else "H×F"
    fig.suptitle(f"{dataset_id}: {family.replace('_', ' ')} {plane_name} planes")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def run_global_pipeline(
    config_path: str | Path,
    *,
    skip_figures: bool = False,
    overwrite: bool = False,
    dataset_ids: list[str] | tuple[str, ...] | None = None,
    show_progress: bool = True,
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
    spectra: list[dict[str, Any]] = []
    analysis_exclusions: list[dict[str, Any]] = []
    records = canonical.to_dict(orient="records")
    progress = tqdm(
        records,
        desc="Global analysis",
        unit="recording",
        disable=not show_progress,
        dynamic_ncols=True,
    )
    for record in progress:
        try:
            cached = None if overwrite else _load_subject_cache(output, record, config)
            if cached is None:
                features, info = _analyze_recording(record, config)
                _save_subject_cache(output, record, features, info, config)
                cache_reused = False
            else:
                features, info = cached
                cache_reused = True
            if show_progress:
                progress.set_postfix(
                    dataset=record["dataset_id"],
                    recording=record["recording_id"],
                    cached="yes" if cache_reused else "no",
                )
        except ValueError as error:
            if "no accepted epochs" not in str(error):
                raise
            analysis_exclusions.append(
                {
                    "dataset_id": record["dataset_id"],
                    "recording_id": record["recording_id"],
                    "participant_id": record["participant_id"],
                    "epoch_path": record["epoch_path"],
                    "reason": str(error),
                }
            )
            continue
        recording_tables.append(features)
        spectra.append({
            "dataset_id": record["dataset_id"],
            "participant_id": record["participant_id"],
            "recording_id": record["recording_id"],
            "group": record["group"],
            "frequencies": info["frequencies"],
            "mean_psd": info["mean_psd"],
        })
        diagnostics.append({
            "dataset_id": record["dataset_id"],
            "recording_id": record["recording_id"],
            "n_channels": len(info["channels"]),
            "sampling_frequency_hz": info["sfreq"],
            "subject_cache_reused": cache_reused,
        })
    if not recording_tables:
        raise RuntimeError("No recordings with accepted epochs were available for analysis")
    feature_table = pd.concat(recording_tables, ignore_index=True)
    _write_table(feature_table, output / "metrics" / "recording_features.csv.gz")
    if analysis_exclusions:
        _write_table(
            pd.DataFrame.from_records(analysis_exclusions),
            output / "metrics" / "analysis_exclusions.csv.gz",
        )
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
            dataset_figures = output / "figures" / dataset.dataset_id
            _plot_psd_spectra(spectra, dataset.dataset_id, dataset_figures / "psd_mean_ci.png")
            _topomap(feature_table, config, dataset.dataset_id, "psd", "psd__", output / "figures" / dataset.dataset_id / "psd_topomaps.png")
            _topomap(feature_table, config, dataset.dataset_id, "aperiodic", "aperiodic__", output / "figures" / dataset.dataset_id / "aperiodic_topomaps.png")
            _topomap(feature_table, config, dataset.dataset_id, "entropy", "entropy__", output / "figures" / dataset.dataset_id / "entropy_topomaps.png")
            _topomap(feature_table, config, dataset.dataset_id, "within_bout", "within_bout__", output / "figures" / dataset.dataset_id / "within_bout_entropy_topomaps.png")
            selected_groups = sorted(feature_table.loc[feature_table["dataset_id"].eq(dataset.dataset_id), "group"].dropna().unique())
            for group_a, group_b in combinations(selected_groups, 2):
                pair_name = f"{_safe_filename(group_b)}_minus_{_safe_filename(group_a)}"
                _contrast_topomap(
                    feature_table,
                    stats,
                    config,
                    dataset.dataset_id,
                    "psd__",
                    group_a,
                    group_b,
                    dataset_figures / f"psd_contrast_{pair_name}_topomaps.png",
                )
                _contrast_topomap(
                    feature_table,
                    stats,
                    config,
                    dataset.dataset_id,
                    "aperiodic__",
                    group_a,
                    group_b,
                    dataset_figures / f"aperiodic_contrast_{pair_name}_topomaps.png",
                )
                _contrast_topomap(
                    feature_table,
                    stats,
                    config,
                    dataset.dataset_id,
                    "entropy__",
                    group_a,
                    group_b,
                    dataset_figures / f"entropy_contrast_{pair_name}_topomaps.png",
                )
                _contrast_topomap(
                    feature_table,
                    stats,
                    config,
                    dataset.dataset_id,
                    "within_bout__",
                    group_a,
                    group_b,
                    dataset_figures / f"within_bout_entropy_contrast_{pair_name}_topomaps.png",
                )
            for outcome in ("moca", "mmse", "updrs"):
                _plot_clinical_scatter(
                    feature_table,
                    dataset.dataset_id,
                    outcome,
                    "bout",
                    config,
                    dataset_figures / f"scatter_{outcome}_bout.png",
                )
                _plot_clinical_scatter(
                    feature_table,
                    dataset.dataset_id,
                    outcome,
                    "within_bout",
                    config,
                    dataset_figures / f"scatter_{outcome}_within_bout.png",
                )
            for family, filename_prefix in (
                ("entropy", "entropy"),
                ("within_bout", "within_bout_entropy"),
            ):
                _plot_entropy_plane(
                    feature_table,
                    dataset.dataset_id,
                    family,
                    "complexity",
                    config,
                    dataset_figures / f"{filename_prefix}_hxc_planes.png",
                )
                _plot_entropy_plane(
                    feature_table,
                    dataset.dataset_id,
                    family,
                    "fisher_information",
                    config,
                    dataset_figures / f"{filename_prefix}_hxf_planes.png",
                )
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "output_root": str(output),
        "n_datasets": int(canonical["dataset_id"].nunique()),
        "n_recordings": int(len(canonical)),
        "n_analyzed_recordings": int(len(recording_tables)),
        "n_excluded_recordings": int(len(analysis_exclusions)),
        "n_subjects": int(canonical["participant_id"].nunique()),
        "groups": canonical["group"].value_counts().to_dict(),
        "analysis_exclusions": analysis_exclusions,
        "entropy_metrics": list(ENTROPY_METRICS),
        "memory_policy": "One cleaned recording is loaded and analyzed at a time; raw samples are released after the subject cache is saved",
        "subject_cache": str(output / "intermediate" / "subjects"),
        "permutation_dimensions": list(config.permutation_dimensions),
        "analysis_signature": _analysis_signature(config),
        "diagnostics": diagnostics,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
