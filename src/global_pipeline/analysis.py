"""Subject-level PSD, entropy, bout, and clinical analyses.

The analysis unit is one cleaned recording (a subject/session/condition when a
dataset has sessions). All accepted epochs for that unit are loaded together,
then reduced to compact feature tables and permutation-pattern sufficient
statistics. PSD uses the subject-level Welch spectrum; filtering, ordinal
windows, and eBOSC bouts preserve epoch boundaries. Raw samples and ordinal
symbol sequences are never written to the intermediate state.
"""

from __future__ import annotations

import json
import itertools
import math
import re
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations
from pathlib import Path
from typing import Any

import mne
import numpy as np
import pandas as pd
import xarray as xr
from scipy.signal import hilbert
from scipy.stats import mannwhitneyu, rankdata, spearmanr, t as student_t, ttest_ind
from tqdm.auto import tqdm

from analyses.bouts.metrics import analyze_bout_segments, shannon_metrics_from_counts
from analyses.ordinal.metrics import (
    filter_epoch_data,
    metrics_from_probabilities,
    ordinal_probabilities,
    weighted_permutation_entropy_epoch_data,
)
from analyses.psd.metrics import bootstrap_median_ci, compute_subject_electrode_psd, to_db
from analyses.scale_free.metrics import (
    aperiodic_wavelet_background,
    detect_frequency_episodes,
    ebosc_wavelet_power,
    extract_band_bouts,
    fit_specparam_spectrum,
    power_thresholds,
    summarize_bouts,
)
from analyses.scale_free.typical_bouts import mean_centered_analytic

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


def _new_entropy_state(dx: int) -> dict[str, Any]:
    return {
        "counts": np.zeros(math.factorial(dx), dtype=np.int64),
        "weighted_sum": 0.0,
        "n_epochs": 0,
        "n_patterns": 0,
        "n_ties": 0,
    }


def _assess_aperiodic_fit(
    metrics: dict[str, Any],
    curves: dict[str, np.ndarray],
    qc: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate one selected specparam fit using the legacy QC thresholds."""
    reasons: list[str] = []
    try:
        observed = np.asarray(curves["observed_psd_uv2_hz"], dtype=float)
        modeled = np.asarray(curves["modeled_psd_uv2_hz"], dtype=float)
        residual = np.log10(observed) - np.log10(modeled)
        residual_bias = float(np.mean(residual))
        residual_sd = float(np.std(residual))
        residual_max = float(np.max(np.abs(residual)))
    except (KeyError, TypeError, ValueError, FloatingPointError):
        residual_bias = residual_sd = residual_max = np.nan
        reasons.append("invalid_residual_spectrum")
    exponent = float(metrics.get("aperiodic_exponent", np.nan))
    r_squared = float(metrics.get("specparam_r_squared", np.nan))
    error_mae = float(metrics.get("specparam_error_mae", np.nan))
    exponent_low, exponent_high = (float(value) for value in qc["exponent_range"])
    if not np.isfinite(r_squared) or r_squared < float(qc["minimum_r_squared"]):
        reasons.append("r_squared_below_minimum")
    if not np.isfinite(error_mae) or error_mae > float(qc["maximum_error_mae_log10"]):
        reasons.append("mae_above_maximum")
    if not np.isfinite(exponent) or not exponent_low <= exponent <= exponent_high:
        reasons.append("exponent_outside_range")
    if not np.isfinite(residual_max) or residual_max > float(qc["maximum_absolute_residual_log10"]):
        reasons.append("residual_above_maximum")
    return {
        "pass": not reasons,
        "reasons": "pass" if not reasons else ";".join(reasons),
        "residual_bias_log10": residual_bias,
        "residual_sd_log10": residual_sd,
        "residual_max_abs_log10": residual_max,
        "raw_exponent": exponent,
        "raw_r_squared": r_squared,
        "raw_error_mae_log10": error_mae,
    }


def _analyze_recording(record: dict[str, Any], config: GlobalConfig) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = Path(record["epoch_path"])
    # One recording is loaded at a time. Epochs remain a separate dimension
    # for filtering, ordinal encoding, and bout detection; this prevents
    # rejected-data gaps and epoch boundaries from becoming artificial EEG.
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

    frequencies, electrode_psd = compute_subject_electrode_psd(
        signal.data,
        sfreq,
        fmin=min(low for low, _ in config.bands.values()),
        fmax=max(high for _, high in config.bands.values()),
    )
    electrode_psd = np.asarray(electrode_psd, dtype=np.float64)

    fit_low, fit_high = (
        float(value) for value in config.aperiodic_settings["frequency_range_hz"]
    )
    aperiodic_bands = {
        band: (max(float(low), fit_low), min(float(high), fit_high))
        for band, (low, high) in config.bands.items()
        if max(float(low), fit_low) < min(float(high), fit_high)
    }
    aperiodic_by_channel: list[dict[str, float]] = []
    aperiodic_curves: list[dict[str, np.ndarray] | None] = []
    aperiodic_qc: list[dict[str, Any]] = []
    for channel_index in range(len(channels)):
        try:
            aperiodic, _, curves = fit_specparam_spectrum(
                frequencies,
                electrode_psd[channel_index],
                aperiodic_bands,
                config.aperiodic_settings,
            )
            qc_result = _assess_aperiodic_fit(
                aperiodic, curves, config.aperiodic_qc_settings
            )
        except Exception as error:
            aperiodic = {}
            curves = None
            qc_result = {
                "pass": False,
                "reasons": f"fit_failed:{type(error).__name__}",
                "residual_bias_log10": np.nan,
                "residual_sd_log10": np.nan,
                "residual_max_abs_log10": np.nan,
                "raw_exponent": np.nan,
                "raw_r_squared": np.nan,
                "raw_error_mae_log10": np.nan,
            }
        numeric_values: dict[str, float] = {}
        for metric, value in aperiodic.items():
            if isinstance(value, str):
                continue
            try:
                numeric_values[metric] = float(value)
            except (TypeError, ValueError):
                continue
        # A failed QC fit remains available through the diagnostic columns,
        # but its exponent/aperiodic metrics and curves are not inferential
        # inputs and cannot trigger eBOSC detection.
        aperiodic_by_channel.append(numeric_values if qc_result["pass"] else {})
        aperiodic_curves.append(curves if qc_result["pass"] else None)
        aperiodic_qc.append(qc_result)

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
                "summary": {},
                "within": {
                    str(dx): _new_entropy_state(dx)
                    for dx in config.permutation_dimensions
                },
            }
            for _ in channels
        ]
        for band in config.bands
    }

    ebosc = config.ebosc_settings
    wavelet_frequencies = np.arange(
        float(ebosc["frequency_min_hz"]),
        float(ebosc["frequency_max_hz"]) + 0.5 * float(ebosc["frequency_step_hz"]),
        float(ebosc["frequency_step_hz"]),
    )
    edge_samples = int(round(float(ebosc["edge_padding_seconds"]) * sfreq))
    data_uv = signal.data.astype(np.float64, copy=False) * 1e6
    band_names = list(config.bands)
    half_window_samples = int(
        round(float(ebosc["figure_window_seconds"]) * sfreq)
    )
    n_bout_figure_samples = 2 * half_window_samples + 1
    bout_representations = {
        "times_seconds": np.arange(
            -half_window_samples, half_window_samples + 1, dtype=float
        ) / sfreq,
        "waveforms": np.full(
            (len(channels), len(band_names), n_bout_figure_samples), np.nan
        ),
        "phase_phasors": np.full(
            (len(channels), len(band_names), n_bout_figure_samples), np.nan + 0j
        ),
        "phase_aligned_shapes": np.full(
            (len(channels), len(band_names), n_bout_figure_samples), np.nan
        ),
        "bout_counts": np.zeros((len(channels), len(band_names)), dtype=np.int64),
        "baseline_amplitude_uv": np.full(
            (len(channels), len(band_names)), np.nan
        ),
    }

    for channel_index in range(len(channels)):
        # Keep filtering boundary-safe while retaining only one channel's
        # band-pass results in memory. The wavelet transform is shared by all
        # bands for this electrode and is released before the next electrode.
        filtered_by_band = {
            band: filter_epoch_data(
                signal.data[:, channel_index : channel_index + 1, :].astype(
                    np.float64, copy=False
                ),
                sfreq=sfreq,
                low_hz=low,
                high_hz=high,
                order=4,
            )[:, 0, :]
            for band, (low, high) in config.bands.items()
        }
        curves = aperiodic_curves[channel_index]
        wavelet_power = None
        detected = None
        thresholds = None
        if curves is not None:
            wavelet_power = ebosc_wavelet_power(
                data_uv[:, channel_index, :],
                sfreq=sfreq,
                frequencies=wavelet_frequencies,
                wavenumber=float(ebosc["wavenumber"]),
            )
            interior = (
                wavelet_power
                if edge_samples == 0
                else wavelet_power[..., edge_samples:-edge_samples]
            )
            mean_wavelet_power = np.mean(interior, axis=(0, 2))
            background = aperiodic_wavelet_background(
                curves["frequencies_hz"],
                curves["modeled_psd_uv2_hz"],
                curves["aperiodic_psd_uv2_hz"],
                wavelet_frequencies,
                mean_wavelet_power,
            )
            thresholds = power_thresholds(
                background, float(ebosc["power_percentile"])
            )
            detected = detect_frequency_episodes(
                wavelet_power,
                sfreq=sfreq,
                frequencies=wavelet_frequencies,
                thresholds=thresholds,
                minimum_cycles=float(ebosc["minimum_cycles"]),
                edge_padding_samples=edge_samples,
            )

        for band, (low, high) in config.bands.items():
            channel_filtered = filtered_by_band[band]
            for dx in config.permutation_dimensions:
                state = entropy[band][channel_index][str(dx)]
                probabilities, n_patterns, n_ties = ordinal_probabilities(
                    channel_filtered,
                    dx=dx,
                    tau=config.delay_samples,
                )
                state["counts"] += np.rint(probabilities * n_patterns).astype(np.int64)
                state["n_patterns"] += n_patterns
                state["n_ties"] += n_ties
                state["weighted_sum"] += weighted_permutation_entropy_epoch_data(
                    channel_filtered, dx=dx, tau=config.delay_samples
                )
                state["n_epochs"] = 1

            bout_state = bouts[band][channel_index]
            band_mask = np.zeros((n_epochs, n_samples), dtype=bool)
            episodes = pd.DataFrame()
            if (
                detected is not None
                and wavelet_power is not None
                and thresholds is not None
                and np.any(
                (wavelet_frequencies >= low) & (wavelet_frequencies <= high)
                )
            ):
                episodes, band_mask = extract_band_bouts(
                    detected,
                    wavelet_power,
                    thresholds,
                    wavelet_frequencies,
                    band=band,
                    band_limits=(low, high),
                    sfreq=sfreq,
                )
                bout_state["summary"] = summarize_bouts(
                    episodes,
                    band_mask,
                    sfreq=sfreq,
                    edge_padding_samples=edge_samples,
                )
            else:
                bout_state["summary"] = {
                    "n_bouts": 0,
                    "oscillatory_occupancy": np.nan,
                    "bouts_per_minute": 0.0,
                    "bout_duration_mean_s": np.nan,
                    "bout_amplitude_mean": np.nan,
                    "bout_cycles_mean": np.nan,
                }
            bout_state["n_bouts"] = int(bout_state["summary"].get("n_bouts", 0))
            if len(episodes):
                band_index = band_names.index(band)
                analytic = hilbert(channel_filtered, axis=-1)
                amplitude = np.abs(analytic)
                amplitude_interior = (
                    amplitude
                    if edge_samples == 0
                    else amplitude[:, edge_samples:-edge_samples]
                )
                baseline = float(np.median(amplitude_interior))
                if np.isfinite(baseline) and baseline > 0.0:
                    waveform, phasor, shape, retained = mean_centered_analytic(
                        analytic / baseline,
                        episodes,
                        half_window_samples=half_window_samples,
                    )
                    bout_representations["waveforms"][channel_index, band_index] = waveform
                    bout_representations["phase_phasors"][channel_index, band_index] = phasor
                    bout_representations["phase_aligned_shapes"][channel_index, band_index] = shape
                    bout_representations["bout_counts"][channel_index, band_index] = retained
                    bout_representations["baseline_amplitude_uv"][channel_index, band_index] = baseline
                for dx in config.permutation_dimensions:
                    within = bout_state["within"][str(dx)]
                    pooled, pooled_summary, _, _ = analyze_bout_segments(
                        channel_filtered,
                        episodes,
                        dx=dx,
                        tau=config.delay_samples,
                    )
                    within["counts"] = pooled
                    within["n_patterns"] = int(pooled.sum())
                    within["n_ties"] = int(pooled_summary.get("n_exact_tied_patterns", 0))
                    within["weighted_sum"] = float(
                        pooled_summary.get("weighted_permutation_entropy", 0.0)
                    ) * within["n_patterns"]
                    within["n_epochs"] = 1

    rows: list[dict[str, Any]] = []
    metadata = {key: record.get(key, "") for key in CANONICAL_COLUMNS if key not in {"epoch_path", "raw_path"}}
    for channel_index, electrode in enumerate(channels):
        total_mask = (frequencies >= min(low for low, _ in config.bands.values())) & (
            frequencies <= max(high for _, high in config.bands.values())
        )
        total_power = float(np.trapezoid(electrode_psd[channel_index, total_mask], frequencies[total_mask]))
        row = {**metadata, "electrode": electrode, "sampling_frequency_hz": sfreq, "n_epochs": n_epochs}
        for metric, value in aperiodic_by_channel[channel_index].items():
            row[f"aperiodic__broadband__{metric}"] = value
        qc_result = aperiodic_qc[channel_index]
        row.update(
            {
                "aperiodic_qc_pass": int(qc_result["pass"]),
                "aperiodic_qc_reasons": str(qc_result["reasons"]),
                "aperiodic_qc_residual_bias_log10": qc_result["residual_bias_log10"],
                "aperiodic_qc_residual_sd_log10": qc_result["residual_sd_log10"],
                "aperiodic_qc_residual_max_abs_log10": qc_result["residual_max_abs_log10"],
                "aperiodic_qc_raw_exponent": qc_result["raw_exponent"],
                "aperiodic_qc_raw_r_squared": qc_result["raw_r_squared"],
                "aperiodic_qc_raw_error_mae_log10": qc_result["raw_error_mae_log10"],
            }
        )
        for band, (low, high) in config.bands.items():
            band_mask = (frequencies >= low) & (frequencies <= high)
            band_power = float(np.trapezoid(electrode_psd[channel_index, band_mask], frequencies[band_mask]))
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
            summary = bout_state["summary"]
            row[f"bout__{band}__n_bouts"] = count
            row[f"bout__{band}__oscillatory_occupancy"] = summary.get("oscillatory_occupancy", np.nan)
            row[f"bout__{band}__bouts_per_minute"] = summary.get("bouts_per_minute", np.nan)
            row[f"bout__{band}__duration_mean_s"] = summary.get("bout_duration_mean_s", np.nan)
            row[f"bout__{band}__amplitude_mean"] = summary.get("bout_amplitude_mean", np.nan)
            row[f"bout__{band}__cycles_mean"] = summary.get("bout_cycles_mean", np.nan)
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
        # Match the old PSD pipeline: one subject-level curve is the median
        # across that recording's available EEG electrodes.
        "subject_psd": np.median(electrode_psd, axis=0),
        "bout_representations": bout_representations,
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
    if (
        np.allclose(selected[feature].to_numpy(float), selected[feature].iloc[0])
        or np.allclose(selected[outcome].to_numpy(float), selected[outcome].iloc[0])
    ):
        return np.nan, np.nan, len(selected)
    y = rankdata(selected[outcome].to_numpy(float))
    x = rankdata(selected[feature].to_numpy(float))
    if covariates:
        z = np.column_stack([np.ones(len(selected)), *[rankdata(selected[name].to_numpy(float)) for name in covariates]])
        y = y - z @ np.linalg.lstsq(z, y, rcond=None)[0]
        x = x - z @ np.linalg.lstsq(z, x, rcond=None)[0]
        if np.std(x) == 0.0 or np.std(y) == 0.0:
            return np.nan, np.nan, len(selected)
        rho = float(np.corrcoef(x, y)[0, 1])
        covariate_rank = int(np.linalg.matrix_rank(z) - 1)
        degrees_freedom = len(selected) - covariate_rank - 2
        if degrees_freedom <= 0:
            p_value = np.nan
        elif abs(rho) >= 1.0:
            p_value = 0.0
        else:
            statistic = rho * np.sqrt(
                degrees_freedom / max(1.0 - rho**2, np.finfo(float).tiny)
            )
            p_value = float(2.0 * student_t.sf(abs(statistic), degrees_freedom))
    else:
        correlation = spearmanr(selected[feature], selected[outcome])
        rho = float(correlation.statistic)
        p_value = float(correlation.pvalue)
    return rho, p_value, len(selected)


def clinical_correlations(table: pd.DataFrame, config: GlobalConfig) -> pd.DataFrame:
    feature_columns = [column for column in table.columns if column.startswith(("bout__", "within_bout__", "entropy__"))]
    # One row per biological participant/condition prevents electrodes and repeated
    # epochs from being mistaken for independent clinical observations.
    subject_source = table[
        ["dataset_id", "participant_id", "group"]
        + feature_columns
        + ["age_years", "sex", "updrs", "moca", "mmse"]
    ].copy()
    subject = subject_source.groupby(
        ["dataset_id", "participant_id", "group"], dropna=False
    )[feature_columns + ["age_years", "sex", "updrs", "moca", "mmse"]].mean(
        numeric_only=True
    ).reset_index()
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
                    "feature_family": feature.split("__", 1)[0],
                    "method": "partial_spearman_age_sex",
                    "n_subjects": n,
                    "rho": rho,
                    "partial_spearman_rho": rho,
                    "p_value": p_value,
                    "covariates": "age_years,sex",
                })
                complete = selected[[feature, outcome]].apply(
                    pd.to_numeric, errors="coerce"
                ).dropna()
                if (
                    len(complete) >= 3
                    and not np.allclose(complete[feature].to_numpy(float), complete[feature].iloc[0])
                    and not np.allclose(complete[outcome].to_numpy(float), complete[outcome].iloc[0])
                ):
                    unadjusted = spearmanr(complete[feature], complete[outcome])
                    unadjusted_rho = float(unadjusted.statistic)
                    unadjusted_p = float(unadjusted.pvalue)
                else:
                    unadjusted_rho = unadjusted_p = np.nan
                rows.append({
                    "dataset_id": dataset_id,
                    "group": group,
                    "outcome": outcome,
                    "feature": feature,
                    "feature_family": feature.split("__", 1)[0],
                    "method": "spearman_unadjusted",
                    "n_subjects": int(len(complete)),
                    "rho": unadjusted_rho,
                    "partial_spearman_rho": np.nan,
                    "spearman_rho": unadjusted_rho,
                    "p_value": unadjusted_p,
                    "covariates": "none",
                })
    result = pd.DataFrame.from_records(rows)
    if not result.empty:
        result["p_fdr_bh"] = np.nan
        for _, indices in result.groupby(
            ["dataset_id", "outcome", "feature_family", "method"], sort=False
        ).groups.items():
            result.loc[indices, "p_fdr_bh"] = _bh(
                result.loc[indices, "p_value"].to_numpy(float)
            )
        result["fdr_scope"] = "dataset,outcome,feature_family,method"
        result["fdr_alpha"] = config.fdr_alpha
    return result


def _safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def _feature_label(feature: str) -> str:
    """Convert a canonical feature column into a readable plot label."""
    parts = str(feature).split("__")
    if len(parts) < 3:
        return str(feature).replace("_", " ")
    family, band = parts[0], parts[1]
    metric = " / ".join(parts[2:]).replace("_", " ")
    return f"{family.replace('_', ' ').title()} — {band.title()} — {metric}"


def _analysis_signature(config: GlobalConfig) -> str:
    """Return the analysis contract used to validate resumable subject files."""
    contract = {
        "schema": 5,
        "bands": config.bands,
        "permutation_dimensions": config.permutation_dimensions,
        "embedding_dimension": config.embedding_dimension,
        "delay_samples": config.delay_samples,
        "bout_threshold_percentile": config.bout_threshold_percentile,
        "bout_minimum_cycles": config.bout_minimum_cycles,
        "aperiodic_settings": config.aperiodic_settings,
        "aperiodic_qc_settings": config.aperiodic_qc_settings,
        "ebosc_settings": config.ebosc_settings,
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
    arrays["subject_psd"] = np.asarray(info["subject_psd"], dtype=np.float64)
    bout_representations = info["bout_representations"]
    arrays["bout_times_seconds"] = np.asarray(
        bout_representations["times_seconds"], dtype=np.float64
    )
    arrays["bout_waveforms"] = np.asarray(
        bout_representations["waveforms"], dtype=np.float64
    )
    arrays["bout_phase_phasors"] = np.asarray(
        bout_representations["phase_phasors"], dtype=np.complex128
    )
    arrays["bout_phase_aligned_shapes"] = np.asarray(
        bout_representations["phase_aligned_shapes"], dtype=np.float64
    )
    arrays["bout_counts"] = np.asarray(
        bout_representations["bout_counts"], dtype=np.int64
    )
    arrays["bout_baseline_amplitude_uv"] = np.asarray(
        bout_representations["baseline_amplitude_uv"], dtype=np.float64
    )

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
        "schema_version": 5,
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
        "aperiodic_qc": config.aperiodic_qc_settings,
        "ebosc_analysis": config.ebosc_settings,
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
            metadata.get("schema_version") != 5
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
                "subject_psd",
                "bout_times_seconds",
                "bout_waveforms",
                "bout_phase_phasors",
                "bout_phase_aligned_shapes",
                "bout_counts",
                "bout_baseline_amplitude_uv",
                *[f"permutation_order__D{dx}" for dx in config.permutation_dimensions],
            ]
            if any(key not in bundle for key in required):
                return None
            frequencies = np.asarray(bundle["frequencies"], dtype=float)
            subject_psd = np.asarray(bundle["subject_psd"], dtype=float)
            bout_representations = {
                "times_seconds": np.asarray(bundle["bout_times_seconds"], dtype=float),
                "waveforms": np.asarray(bundle["bout_waveforms"], dtype=float),
                "phase_phasors": np.asarray(bundle["bout_phase_phasors"], dtype=complex),
                "phase_aligned_shapes": np.asarray(
                    bundle["bout_phase_aligned_shapes"], dtype=float
                ),
                "bout_counts": np.asarray(bundle["bout_counts"], dtype=np.int64),
                "baseline_amplitude_uv": np.asarray(
                    bundle["bout_baseline_amplitude_uv"], dtype=float
                ),
            }
        return features, {
            "channels": list(metadata.get("channels", [])),
            "sfreq": float(metadata["sfreq"]),
            "frequencies": frequencies,
            "subject_psd": subject_psd,
            "bout_representations": bout_representations,
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

    feature_columns = [column for column in table if column.startswith(feature_template)]
    if feature_template in {"entropy__", "within_bout__"}:
        feature_columns = [
            column for column in feature_columns
            if column.endswith(f"__D{config.embedding_dimension}")
        ]
    if not feature_columns:
        return
    selected_columns = list(dict.fromkeys(
        ["dataset_id", "group", "electrode"] + feature_columns
    ))
    selected = table.loc[
        table["dataset_id"].eq(dataset_id), selected_columns
    ].copy()
    if selected.empty:
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
    # Put one feature per row so that the metric being compared is easy to
    # follow, especially when a family contains many bands or dimensions.
    fig, axes = plt.subplots(
        len(feature_columns),
        len(groups),
        figsize=(3.4 * len(groups), 3.2 * len(feature_columns)),
        squeeze=False,
        constrained_layout=True,
    )
    images = {}
    info = mne.create_info(channels, sfreq=100.0, ch_types="eeg")
    info.set_montage(montage, on_missing="ignore", verbose="ERROR")
    for row_index, feature in enumerate(feature_columns):
        for column_index, group in enumerate(groups):
            values = maps[feature][column_index]
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
            axes[row_index, column_index].set_title(
                f"{group}\n{_feature_label(feature)}",
                fontsize=8,
            )
    for row_index, feature in enumerate(feature_columns):
        if feature in images:
            fig.colorbar(images[feature], ax=axes[row_index, :].tolist(), shrink=0.75)
    fig.suptitle(
        f"{dataset_id}: {domain.replace('_', ' ').title()} topomaps",
        fontsize=13,
    )
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
        len(feature_columns),
        1,
        figsize=(4.8, 3.5 * len(feature_columns)),
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
    significant_count = 0
    for row_index, feature in enumerate(feature_columns):
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
            axes[row_index, 0].axis("off")
            continue
        image, _ = mne.viz.plot_topomap(
            differences,
            info,
            axes=axes[row_index, 0],
            show=False,
            contours=0,
            names=None,
            cmap="RdBu_r",
            vlim=_stable_limits(differences, symmetric=True),
            mask=mask,
            mask_params=mask_params,
        )
        images[feature] = image
        significant_count += int(mask.sum())
        axes[row_index, 0].set_title(_feature_label(feature), fontsize=9)
    for row_index, feature in enumerate(feature_columns):
        if feature in images:
            fig.colorbar(images[feature], ax=axes[row_index, 0], shrink=0.75)
    fig.suptitle(
        f"{dataset_id}: {group_b} − {group_a} — "
        f"{feature_template.rstrip('_').replace('_', ' ').title()} topomaps\n"
        f"white dots = Welch BH-FDR p < {config.fdr_alpha:g} "
        f"({significant_count} significant electrode-feature maps)",
        fontsize=13,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def _plot_psd_spectra(
    spectra: list[dict[str, Any]],
    dataset_id: str,
    output: Path,
) -> None:
    """Plot robust subject-level median PSD with a bootstrap 95% CI."""
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
            curve = np.asarray(item["subject_psd"], dtype=float)
            if not np.array_equal(frequency, reference):
                curve = np.interp(reference, frequency, curve, left=np.nan, right=np.nan)
            curves.append(curve)
        values = np.asarray(curves, dtype=float)
        values = values[np.all(np.isfinite(values), axis=1)]
        if len(values) >= 2:
            center, lower, upper = bootstrap_median_ci(
                values,
                n_resamples=2000,
                confidence_level=0.95,
                seed=20260826 + len(group),
            )
        elif len(values) == 1:
            center = lower = upper = values[0]
        else:
            continue
        line = axis.plot(
            reference,
            to_db(center),
            linewidth=1.8,
            label=f"{group} (n={len(values)})",
        )[0]
        axis.fill_between(
            reference,
            to_db(lower),
            to_db(upper),
            color=line.get_color(),
            alpha=0.2,
        )
    axis.set_xlabel("Frequency (Hz)")
    axis.set_ylabel("PSD (dB µV²/Hz)")
    axis.set_title(f"{dataset_id}: median PSD ± 95% bootstrap CI")
    axis.set_xlim(left=0.0)
    axis.grid(True, alpha=0.25)
    axis.legend()
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def _plot_average_detected_bouts(
    spectra: list[dict[str, Any]],
    dataset_id: str,
    config: GlobalConfig,
    output: Path,
) -> None:
    """Plot subject-balanced average signals centered on detected eBOSC bouts."""
    import matplotlib.pyplot as plt

    selected = [item for item in spectra if item["dataset_id"] == dataset_id]
    if not selected:
        return
    all_bands = list(config.bands)
    first = selected[0].get("bout_representations")
    if not first:
        return
    times = np.asarray(first["times_seconds"], dtype=float)
    available_band_indices = [
        band_index
        for band_index, _ in enumerate(all_bands)
        if any(
            np.any(np.asarray(item["bout_representations"]["bout_counts"])[:, band_index] > 0)
            for item in selected
            if item.get("bout_representations") is not None
        )
    ]
    if not available_band_indices:
        return
    bands = [all_bands[index] for index in available_band_indices]
    n_bands = len(bands)
    curves: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for item in selected:
        representation = item.get("bout_representations")
        if representation is None:
            continue
        waveforms = np.asarray(representation["waveforms"], dtype=float)
        phasors = np.asarray(representation["phase_phasors"], dtype=complex)
        shapes = np.asarray(representation["phase_aligned_shapes"], dtype=float)
        counts = np.asarray(representation["bout_counts"], dtype=int)
        baselines = np.asarray(representation["baseline_amplitude_uv"], dtype=float)
        group = str(item["group"])
        for band_index in available_band_indices:
            band = all_bands[band_index]
            valid = (
                (counts[:, band_index] > 0)
                & np.isfinite(baselines[:, band_index])
                & np.all(np.isfinite(waveforms[:, band_index]), axis=1)
                & np.all(np.isfinite(phasors[:, band_index]), axis=1)
                & np.all(np.isfinite(shapes[:, band_index]), axis=1)
            )
            if not valid.any():
                continue
            envelope = (waveforms[valid, band_index] - 1.0) * baselines[valid, band_index, None]
            shape = shapes[valid, band_index] * baselines[valid, band_index, None]
            phase = phasors[valid, band_index]
            subject_curve = {
                "envelope": np.mean(envelope, axis=0),
                "shape": np.mean(shape, axis=0),
                "phase": np.mean(phase, axis=0),
                "n_bouts": int(counts[valid, band_index].sum()),
            }
            curves.setdefault(group, {}).setdefault(band, []).append(subject_curve)

    groups = sorted(curves)
    if not groups:
        return
    colors = {group: f"C{index}" for index, group in enumerate(groups)}
    fig, axes = plt.subplots(
        n_bands,
        3,
        figsize=(17, 4.0 * n_bands),
        squeeze=False,
        constrained_layout=True,
    )

    def mean_ci(values: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        array = np.asarray(values, dtype=float)
        array = array[np.all(np.isfinite(array), axis=1)]
        if len(array) == 0:
            missing = np.full(len(times), np.nan)
            return missing, missing, missing
        center = np.mean(array, axis=0)
        if len(array) < 2:
            return center, center.copy(), center.copy()
        sem = np.std(array, axis=0, ddof=1) / np.sqrt(len(array))
        critical = float(student_t.ppf(0.975, len(array) - 1))
        return center, center - critical * sem, center + critical * sem

    for band_index, band in enumerate(bands):
        envelope_axis, phase_axis, shape_axis = axes[band_index]
        phase_r_axis = phase_axis.twinx()
        for group in groups:
            entries = curves.get(group, {}).get(band, [])
            if not entries:
                continue
            envelope, envelope_low, envelope_high = mean_ci(
                [entry["envelope"] for entry in entries]
            )
            shape, shape_low, shape_high = mean_ci(
                [entry["shape"] for entry in entries]
            )
            label = f"{group} (n={len(entries)}, bouts={sum(entry['n_bouts'] for entry in entries):,})"
            color = colors[group]
            envelope_axis.plot(times, envelope, color=color, linewidth=1.6, label=label)
            envelope_axis.fill_between(times, envelope_low, envelope_high, color=color, alpha=0.2)
            shape_axis.plot(times, shape, color=color, linewidth=1.6, label=label)
            shape_axis.fill_between(times, shape_low, shape_high, color=color, alpha=0.2)
            phase_vectors = np.asarray([entry["phase"] for entry in entries], dtype=complex)
            mean_vector = np.mean(phase_vectors, axis=0)
            phase = np.unwrap(np.angle(mean_vector))
            phase -= phase[len(phase) // 2]
            phase_axis.plot(times, phase / np.pi, color=color, linewidth=1.4, label=f"{group} phase")
            phase_r_axis.plot(
                times,
                np.abs(mean_vector),
                color=color,
                linewidth=1.0,
                linestyle=":",
                alpha=0.65,
            )
        display = band.replace("_", " ").title()
        for axis in (envelope_axis, phase_axis, shape_axis):
            axis.axvline(0.0, color="0.35", linestyle="--", linewidth=0.8)
            axis.grid(alpha=0.18)
            axis.set_xlabel("Time from bout center (s)")
        envelope_axis.set_title(f"{display} — average bout envelope")
        envelope_axis.set_ylabel("Hilbert amplitude above baseline (µV)")
        phase_axis.set_title(f"{display} — relative Hilbert phase")
        phase_axis.set_ylabel("Circular phase (π radians)")
        phase_r_axis.set_ylabel("Phase consistency R")
        phase_r_axis.set_ylim(0.0, 1.05)
        shape_axis.set_title(f"{display} — phase-aligned average signal")
        shape_axis.set_ylabel("Band-passed signal (µV)")
        if envelope_axis.get_legend_handles_labels()[0]:
            envelope_axis.legend(frameon=False, fontsize=7)
        if shape_axis.get_legend_handles_labels()[0]:
            shape_axis.legend(frameon=False, fontsize=7)
    fig.suptitle(
        f"{dataset_id}: average signals around detected eBOSC bouts\n"
        "electrodes averaged within recording; confidence bands are across recordings"
    )
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
    selected_columns = list(dict.fromkeys(
        ["dataset_id", "participant_id", "group", outcome] + feature_columns
    ))
    selected = table.loc[
        table["dataset_id"].eq(dataset_id), selected_columns
    ].copy()
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


def _plot_subject_violins(
    table: pd.DataFrame,
    dataset_id: str,
    family: str,
    config: GlobalConfig,
    output: Path,
) -> None:
    """Compare electrode-averaged feature values at the subject level.

    Each plotted observation is one participant and group/condition.  The
    electrode rows are averaged before plotting, so electrodes do not inflate
    the apparent sample size or the width of the distributions.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    prefix = f"{family}__"
    feature_columns = [column for column in table if column.startswith(prefix)]
    if family in {"entropy", "within_bout"}:
        feature_columns = [
            column for column in feature_columns
            if column.endswith(f"__D{config.embedding_dimension}")
        ]
    if not feature_columns:
        return

    selected_columns = [
        "dataset_id", "participant_id", "group", *feature_columns
    ]
    selected = table.loc[
        table["dataset_id"].eq(dataset_id), selected_columns
    ].copy()
    selected = selected.loc[selected["group"].notna()]
    if selected.empty:
        return
    subject = (
        selected.groupby(["participant_id", "group"], dropna=False)[feature_columns]
        .mean(numeric_only=True)
        .reset_index()
    )
    groups = sorted(str(value) for value in subject["group"].dropna().unique())
    if not groups:
        return
    colors = {group: f"C{index}" for index, group in enumerate(groups)}
    ncols = min(4, len(feature_columns))
    nrows = int(math.ceil(len(feature_columns) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.0 * ncols, 3.3 * nrows),
        squeeze=False,
        constrained_layout=True,
    )
    positions = np.arange(1, len(groups) + 1, dtype=float)
    pairwise_p: dict[tuple[str, str, str], float] = {}
    for feature in feature_columns:
        for left_index, left_group in enumerate(groups[:-1]):
            left_values = subject.loc[
                subject["group"].astype(str).eq(left_group), feature
            ].to_numpy(dtype=float)
            left_values = left_values[np.isfinite(left_values)]
            for right_group in groups[left_index + 1:]:
                right_values = subject.loc[
                    subject["group"].astype(str).eq(right_group), feature
                ].to_numpy(dtype=float)
                right_values = right_values[np.isfinite(right_values)]
                if left_values.size < 2 or right_values.size < 2:
                    continue
                test = ttest_ind(
                    left_values,
                    right_values,
                    equal_var=False,
                    nan_policy="omit",
                )
                if np.isfinite(test.pvalue):
                    pairwise_p[(feature, left_group, right_group)] = float(test.pvalue)
    pairwise_q = _bh(np.asarray(list(pairwise_p.values()), dtype=float))
    pairwise_q = dict(zip(pairwise_p, pairwise_q))
    for feature_index, feature in enumerate(feature_columns):
        axis = axes.flat[feature_index]
        plotted = False
        significant_pairs: list[str] = []
        for position, group in zip(positions, groups):
            values = subject.loc[
                subject["group"].astype(str).eq(group), feature
            ].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            if values.size == 0:
                continue
            plotted = True
            if values.size >= 2 and not np.isclose(values.min(), values.max()):
                violin = axis.violinplot(
                    values,
                    positions=[position],
                    widths=0.72,
                    showmeans=False,
                    showmedians=True,
                    showextrema=True,
                )
                for body in violin["bodies"]:
                    body.set_facecolor(colors[group])
                    body.set_edgecolor(colors[group])
                    body.set_alpha(0.65)
                violin["cmedians"].set_color("black")
                violin["cbars"].set_color(colors[group])
                violin["cmins"].set_color(colors[group])
                violin["cmaxes"].set_color(colors[group])
            jitter = np.linspace(-0.08, 0.08, values.size)
            axis.scatter(
                np.full(values.size, position) + jitter,
                values,
                s=11,
                color=colors[group],
                edgecolor="white",
                linewidth=0.35,
                alpha=0.75,
                zorder=3,
            )
        if plotted:
            axis.set_xticks(positions, groups, rotation=25, ha="right")
            # Correct all pairwise feature comparisons in this dataset/family
            # together. The raw p-value and adjusted q-value are both shown,
            # but only q < alpha is called significant.
            for (tested_feature, left_group, right_group), raw_p in pairwise_p.items():
                if tested_feature != feature:
                    continue
                q_value = pairwise_q[(tested_feature, left_group, right_group)]
                if q_value < config.fdr_alpha:
                    significant_pairs.append(
                        f"{left_group} vs {right_group}: "
                        f"p={raw_p:.3g}, q={q_value:.3g}"
                    )
            title = _feature_label(feature)
            if significant_pairs:
                title += "\n* Welch BH-FDR q<" + f"{config.fdr_alpha:g}: " + "; ".join(significant_pairs)
            else:
                title += f"\nno Welch BH-FDR q<{config.fdr_alpha:g}"
            axis.set_title(title, fontsize=8)
            axis.grid(axis="y", alpha=0.2)
            axis.set_ylabel("Subject-average value")
        else:
            axis.axis("off")
    for axis in axes.flat[len(feature_columns):]:
        axis.axis("off")
    handles = [
        Line2D(
            [], [], color=colors[group], marker="o", linestyle="none", label=group
        )
        for group in groups
    ]
    axes.flat[0].legend(handles=handles, frameon=False, fontsize=8)
    fig.suptitle(
        f"{dataset_id}: subject-level {family.replace('_', ' ')} distributions\n"
        "Each point is one participant/condition; annotations show raw Welch p and BH-FDR q"
    )
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


def _analyze_recording_worker(
    record: dict[str, Any], config: GlobalConfig
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    """Process one complete recording in one analysis worker."""
    features, info = _analyze_recording(record, config)
    return record, features, info


def _plot_dataset_results(
    feature_table: pd.DataFrame,
    statistics: pd.DataFrame,
    spectra: list[dict[str, Any]],
    dataset_id: str,
    config: GlobalConfig,
    output: Path,
) -> None:
    """Render every figure for one finished dataset."""
    dataset_figures = output / "figures" / dataset_id
    _plot_psd_spectra(spectra, dataset_id, dataset_figures / "psd_mean_ci.png")
    _plot_average_detected_bouts(
        spectra,
        dataset_id,
        config,
        dataset_figures / "average_detected_bouts.png",
    )
    _topomap(feature_table, config, dataset_id, "psd", "psd__", dataset_figures / "psd_topomaps.png")
    _topomap(feature_table, config, dataset_id, "aperiodic", "aperiodic__", dataset_figures / "aperiodic_topomaps.png")
    _topomap(feature_table, config, dataset_id, "entropy", "entropy__", dataset_figures / "entropy_topomaps.png")
    _topomap(feature_table, config, dataset_id, "within_bout", "within_bout__", dataset_figures / "within_bout_entropy_topomaps.png")
    selected_groups = sorted(feature_table["group"].dropna().unique())
    for group_a, group_b in combinations(selected_groups, 2):
        pair_name = f"{_safe_filename(group_b)}_minus_{_safe_filename(group_a)}"
        for template, filename in (
            ("psd__", f"psd_contrast_{pair_name}_topomaps.png"),
            ("aperiodic__", f"aperiodic_contrast_{pair_name}_topomaps.png"),
            ("entropy__", f"entropy_contrast_{pair_name}_topomaps.png"),
            ("within_bout__", f"within_bout_entropy_contrast_{pair_name}_topomaps.png"),
        ):
            _contrast_topomap(
                feature_table,
                statistics,
                config,
                dataset_id,
                template,
                group_a,
                group_b,
                dataset_figures / filename,
            )
    for outcome in ("moca", "mmse", "updrs"):
        _plot_clinical_scatter(
            feature_table,
            dataset_id,
            outcome,
            "bout",
            config,
            dataset_figures / f"scatter_{outcome}_bout.png",
        )
        _plot_clinical_scatter(
            feature_table,
            dataset_id,
            outcome,
            "within_bout",
            config,
            dataset_figures / f"scatter_{outcome}_within_bout.png",
        )
    for family in ("psd", "aperiodic", "entropy", "bout", "within_bout"):
        _plot_subject_violins(
            feature_table,
            dataset_id,
            family,
            config,
            dataset_figures / f"subject_violins_{family}.png",
        )
    for family, filename_prefix in (
        ("entropy", "entropy"),
        ("within_bout", "within_bout_entropy"),
    ):
        _plot_entropy_plane(
            feature_table,
            dataset_id,
            family,
            "complexity",
            config,
            dataset_figures / f"{filename_prefix}_hxc_planes.png",
        )
        _plot_entropy_plane(
            feature_table,
            dataset_id,
            family,
            "fisher_information",
            config,
            dataset_figures / f"{filename_prefix}_hxf_planes.png",
        )


def _write_qc_tables(
    feature_table: pd.DataFrame,
    metrics_dir: Path,
    config: GlobalConfig,
) -> None:
    """Write electrode- and participant-level aperiodic QC summaries."""
    qc_columns = [
        "dataset_id",
        "recording_id",
        "participant_id",
        "group",
        "electrode",
        "aperiodic_qc_pass",
        "aperiodic_qc_reasons",
        "aperiodic_qc_residual_bias_log10",
        "aperiodic_qc_residual_sd_log10",
        "aperiodic_qc_residual_max_abs_log10",
        "aperiodic_qc_raw_exponent",
        "aperiodic_qc_raw_r_squared",
        "aperiodic_qc_raw_error_mae_log10",
    ]
    _write_table(
        feature_table[qc_columns],
        metrics_dir / "aperiodic_fit_qc.csv.gz",
    )
    subject_qc = (
        feature_table.groupby(
            ["dataset_id", "participant_id", "group"], dropna=False
        )["aperiodic_qc_pass"]
        .agg(n_electrodes="size", n_qc_pass_electrodes="sum")
        .reset_index()
    )
    subject_qc["qc_pass_fraction"] = (
        subject_qc["n_qc_pass_electrodes"] / subject_qc["n_electrodes"]
    )
    subject_qc["subject_qc_pass"] = subject_qc["qc_pass_fraction"].ge(
        config.aperiodic_qc_settings["minimum_subject_qc_fraction"]
    )
    _write_table(
        subject_qc,
        metrics_dir / "aperiodic_subject_qc.csv.gz",
    )


def _run_global_pipeline_parallel(
    config_path: str | Path,
    *,
    skip_figures: bool,
    overwrite: bool,
    dataset_ids: list[str] | tuple[str, ...] | None,
    show_progress: bool,
    analysis_workers: int,
) -> dict[str, Any]:
    config = load_global_config(config_path)
    if analysis_workers < 1:
        raise ValueError("analysis_workers must be positive")
    output = config.output_root
    output.mkdir(parents=True, exist_ok=True)
    canonical_path = output / "canonical" / "recordings.csv.gz"
    requested_ids = (
        tuple(dataset.dataset_id for dataset in config.enabled_datasets)
        if dataset_ids is None
        else tuple(dataset_ids)
    )
    available_ids = {dataset.dataset_id for dataset in config.enabled_datasets}
    unknown = sorted(set(requested_ids) - available_ids)
    if unknown:
        raise ValueError(
            f"Unknown or disabled dataset(s): {unknown}; enabled choices: "
            f"{sorted(available_ids)}"
        )
    if overwrite or not canonical_path.exists():
        canonical = convert_config(config, dataset_ids=requested_ids)
    else:
        canonical = read_canonical_table(canonical_path)
        cached_ids = tuple(sorted(canonical["dataset_id"].dropna().unique()))
        if set(cached_ids) != set(requested_ids):
            canonical = convert_config(config, dataset_ids=requested_ids)
        else:
            canonical = canonical.loc[
                canonical["dataset_id"].isin(requested_ids)
            ].copy()

    requested_order = {dataset_id: index for index, dataset_id in enumerate(requested_ids)}
    dataset_order = sorted(
        requested_ids,
        key=lambda dataset_id: (
            int(canonical["dataset_id"].eq(dataset_id).sum()),
            requested_order[dataset_id],
        ),
    )
    all_feature_tables: list[pd.DataFrame] = []
    all_spectra: list[dict[str, Any]] = []
    all_diagnostics: list[dict[str, Any]] = []
    all_exclusions: list[dict[str, Any]] = []

    for dataset_id in dataset_order:
        dataset_records = canonical.loc[
            canonical["dataset_id"].eq(dataset_id)
        ].to_dict(orient="records")
        feature_parts: list[pd.DataFrame] = []
        dataset_spectra: list[dict[str, Any]] = []
        dataset_diagnostics: list[dict[str, Any]] = []
        dataset_exclusions: list[dict[str, Any]] = []
        progress = tqdm(
            total=len(dataset_records),
            desc=f"Analyze {dataset_id}",
            unit="recording",
            disable=not show_progress,
            dynamic_ncols=True,
        )

        def accept_result(
            record: dict[str, Any],
            features: pd.DataFrame,
            info: dict[str, Any],
            cache_reused: bool,
        ) -> None:
            feature_parts.append(features)
            dataset_spectra.append(
                {
                    "dataset_id": record["dataset_id"],
                    "participant_id": record["participant_id"],
                    "recording_id": record["recording_id"],
                    "group": record["group"],
                    "frequencies": info["frequencies"],
                    "subject_psd": info["subject_psd"],
                    "bout_representations": info["bout_representations"],
                }
            )
            dataset_diagnostics.append(
                {
                    "dataset_id": record["dataset_id"],
                    "recording_id": record["recording_id"],
                    "n_channels": len(info["channels"]),
                    "sampling_frequency_hz": info["sfreq"],
                    "subject_cache_reused": cache_reused,
                }
            )

        pending: dict[Any, dict[str, Any]] = {}
        for record in dataset_records:
            cached = None if overwrite else _load_subject_cache(output, record, config)
            if cached is not None:
                features, info = cached
                accept_result(record, features, info, True)
                progress.update(1)
                if show_progress:
                    progress.set_postfix(recording=record["recording_id"], cached="yes")
            else:
                pending[record["recording_id"]] = record

        def handle_failure(record: dict[str, Any], error: Exception) -> None:
            if not isinstance(error, ValueError) or "no accepted epochs" not in str(error):
                raise error
            dataset_exclusions.append(
                {
                    "dataset_id": record["dataset_id"],
                    "recording_id": record["recording_id"],
                    "participant_id": record["participant_id"],
                    "epoch_path": record["epoch_path"],
                    "reason": str(error),
                }
            )

        if analysis_workers == 1:
            for record in pending.values():
                try:
                    features, info = _analyze_recording(record, config)
                    _save_subject_cache(output, record, features, info, config)
                    accept_result(record, features, info, False)
                except Exception as error:
                    handle_failure(record, error)
                progress.update(1)
                if show_progress:
                    progress.set_postfix(recording=record["recording_id"], cached="no")
        else:
            with ProcessPoolExecutor(max_workers=analysis_workers) as executor:
                futures = {
                    executor.submit(_analyze_recording_worker, record, config): record
                    for record in pending.values()
                }
                for future in as_completed(futures):
                    record = futures.pop(future)
                    try:
                        _, features, info = future.result()
                        _save_subject_cache(output, record, features, info, config)
                        accept_result(record, features, info, False)
                    except Exception as error:
                        handle_failure(record, error)
                    progress.update(1)
                    if show_progress:
                        progress.set_postfix(recording=record["recording_id"], cached="no")
        progress.close()

        all_exclusions.extend(dataset_exclusions)
        all_diagnostics.extend(dataset_diagnostics)
        all_spectra.extend(dataset_spectra)
        if not feature_parts:
            continue
        dataset_table = pd.concat(feature_parts, ignore_index=True)
        all_feature_tables.append(dataset_table)
        dataset_stats = group_statistics(dataset_table, config)
        dataset_correlations = clinical_correlations(dataset_table, config)
        dataset_metrics = output / "metrics" / dataset_id
        dataset_statistics = output / "statistics" / dataset_id
        _write_table(dataset_table, dataset_metrics / "recording_features.csv.gz")
        _write_qc_tables(dataset_table, dataset_metrics, config)
        _write_table(
            dataset_table.groupby(
                ["dataset_id", "participant_id", "group"], dropna=False
            ).mean(numeric_only=True).reset_index(),
            dataset_metrics / "subject_features.csv.gz",
        )
        if dataset_exclusions:
            _write_table(
                pd.DataFrame.from_records(dataset_exclusions),
                dataset_metrics / "analysis_exclusions.csv.gz",
            )
        _write_table(dataset_stats, dataset_statistics / "group_statistics.csv.gz")
        _write_table(
            dataset_correlations,
            dataset_statistics / "clinical_correlations.csv.gz",
        )
        if not skip_figures:
            _plot_dataset_results(
                dataset_table,
                dataset_stats,
                dataset_spectra,
                dataset_id,
                config,
                output,
            )

    if not all_feature_tables:
        raise RuntimeError("No recordings with accepted epochs were available for analysis")
    feature_table = pd.concat(all_feature_tables, ignore_index=True)
    _write_table(feature_table, output / "metrics" / "recording_features.csv.gz")
    _write_qc_tables(feature_table, output / "metrics", config)
    _write_table(
        feature_table.groupby(
            ["dataset_id", "participant_id", "group"], dropna=False
        ).mean(numeric_only=True).reset_index(),
        output / "metrics" / "subject_features.csv.gz",
    )
    if all_exclusions:
        _write_table(
            pd.DataFrame.from_records(all_exclusions),
            output / "metrics" / "analysis_exclusions.csv.gz",
        )
    stats = group_statistics(feature_table, config)
    correlations = clinical_correlations(feature_table, config)
    _write_table(stats, output / "statistics" / "group_statistics.csv.gz")
    _write_table(correlations, output / "statistics" / "clinical_correlations.csv.gz")
    manifest = {
        "schema_version": 2,
        "status": "complete",
        "output_root": str(output),
        "dataset_order": dataset_order,
        "dataset_sizes": {
            dataset_id: int(canonical["dataset_id"].eq(dataset_id).sum())
            for dataset_id in dataset_order
        },
        "analysis_workers": int(analysis_workers),
        "n_datasets": int(canonical["dataset_id"].nunique()),
        "n_recordings": int(len(canonical)),
        "n_analyzed_recordings": int(
            sum(table["recording_id"].nunique() for table in all_feature_tables)
        ),
        "n_excluded_recordings": int(len(all_exclusions)),
        "n_subjects": int(canonical["participant_id"].nunique()),
        "groups": canonical["group"].value_counts().to_dict(),
        "analysis_exclusions": all_exclusions,
        "entropy_metrics": list(ENTROPY_METRICS),
        "memory_policy": "One cleaned recording per worker; parent writes caches and releases completed worker results after aggregation",
        "subject_cache": str(output / "intermediate" / "subjects"),
        "permutation_dimensions": list(config.permutation_dimensions),
        "analysis_signature": _analysis_signature(config),
        "diagnostics": all_diagnostics,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def run_global_pipeline(
    config_path: str | Path,
    *,
    skip_figures: bool = False,
    overwrite: bool = False,
    dataset_ids: list[str] | tuple[str, ...] | None = None,
    show_progress: bool = True,
    analysis_workers: int = 1,
) -> dict[str, Any]:
    return _run_global_pipeline_parallel(
        config_path,
        skip_figures=skip_figures,
        overwrite=overwrite,
        dataset_ids=dataset_ids,
        show_progress=show_progress,
        analysis_workers=analysis_workers,
    )
