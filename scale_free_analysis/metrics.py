"""Numerical building blocks for spectral, bout, and cycle analyses."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from bycycle.features import compute_shape_features
from scipy.signal import fftconvolve
from scipy.stats import chi2
from specparam import SpectralModel
from specparam.data.periodic import get_band_peak


APERIODIC_FEATURES = (
    "aperiodic_offset",
    "aperiodic_exponent",
    "specparam_r_squared",
    "specparam_error_mae",
)

BAND_FEATURES = (
    "peak_present",
    "peak_frequency_hz",
    "peak_power_log10",
    "peak_power_linear",
    "peak_bandwidth_hz",
    "oscillatory_occupancy",
    "bouts_per_minute",
    "bout_duration_mean_s",
    "bout_duration_median_s",
    "bout_cycles_mean",
    "inter_bout_interval_mean_s",
    "bout_power_mean",
    "bout_amplitude_mean",
    "bout_snr_mean",
    "cycle_amplitude_mean_uv",
    "cycle_band_amplitude_mean_uv",
    "cycle_frequency_mean_hz",
    "cycle_period_mean_s",
    "cycle_amplitude_std_uv",
    "cycle_amplitude_cv",
    "cycle_period_std_s",
    "cycle_period_cv",
    "rise_decay_symmetry_mean",
    "peak_trough_symmetry_mean",
)


def fit_specparam_spectrum(
    frequencies: np.ndarray,
    power_spectrum: np.ndarray,
    bands: Mapping[str, tuple[float, float] | list[float]],
    settings: Mapping[str, Any],
) -> tuple[dict[str, float], list[dict[str, Any]], dict[str, np.ndarray]]:
    """Fit one linear PSD and return aperiodic, band-peak, and curve results."""
    freqs = np.asarray(frequencies, dtype=float)
    power = np.asarray(power_spectrum, dtype=float)
    if freqs.ndim != 1 or power.shape != freqs.shape:
        raise ValueError("frequencies and power_spectrum must be matching vectors")
    if not np.all(np.isfinite(power)) or np.any(power <= 0.0):
        raise ValueError("specparam requires finite positive linear power")

    model = SpectralModel(
        aperiodic_mode=str(settings["aperiodic_mode"]),
        peak_width_limits=tuple(float(value) for value in settings["peak_width_limits_hz"]),
        max_n_peaks=int(settings["max_n_peaks"]),
        min_peak_height=float(settings["min_peak_height"]),
        peak_threshold=float(settings["peak_threshold"]),
        verbose=False,
    )
    fit_range = tuple(float(value) for value in settings["frequency_range_hz"])
    model.fit(freqs, power, fit_range)
    if not model.results.has_model:
        raise RuntimeError("specparam failed to fit the power spectrum")

    aperiodic = np.asarray(model.get_params("aperiodic"), dtype=float)
    if aperiodic.shape != (2,):
        raise RuntimeError("Fixed specparam mode did not return [offset, exponent]")
    metrics = model.results.metrics.results
    aperiodic_row = {
        "aperiodic_offset": float(aperiodic[0]),
        "aperiodic_exponent": float(aperiodic[1]),
        "specparam_r_squared": float(metrics.get("gof_rsquared", np.nan)),
        "specparam_error_mae": float(metrics.get("error_mae", np.nan)),
        "n_detected_peaks": int(model.results.n_peaks),
    }

    band_rows: list[dict[str, Any]] = []
    for band, limits in bands.items():
        fitted_peak = np.asarray(
            get_band_peak(model, limits, attribute="fit"), dtype=float
        )
        converted_peak = np.asarray(
            get_band_peak(model, limits, attribute="converted"), dtype=float
        )
        present = bool(np.all(np.isfinite(fitted_peak)))
        band_rows.append(
            {
                "band": str(band),
                "band_low_hz": float(limits[0]),
                "band_high_hz": float(limits[1]),
                "peak_present": int(present),
                "peak_frequency_hz": float(fitted_peak[0]) if present else np.nan,
                "peak_power_log10": float(fitted_peak[1]) if present else np.nan,
                "peak_power_linear": (
                    float(converted_peak[1])
                    if present and np.all(np.isfinite(converted_peak))
                    else np.nan
                ),
                "peak_bandwidth_hz": float(fitted_peak[2]) if present else np.nan,
            }
        )

    curves = {
        "frequencies_hz": model.data.freqs.copy(),
        "observed_psd_uv2_hz": model.data.get_data("full", "linear").copy(),
        "modeled_psd_uv2_hz": model.results.model.get_component("full", "linear").copy(),
        "aperiodic_psd_uv2_hz": model.results.model.get_component(
            "aperiodic", "linear"
        ).copy(),
        "periodic_psd_uv2_hz": model.results.model.get_component(
            "peak", "linear"
        ).copy(),
    }
    return aperiodic_row, band_rows, curves


def ebosc_wavelet_power(
    epoch_signals: np.ndarray,
    *,
    sfreq: float,
    frequencies: np.ndarray,
    wavenumber: float,
) -> np.ndarray:
    """Vectorized equivalent of ``ebosc.BOSC.BOSC_tf`` for separate epochs.

    The eBOSC Morlet definition and exact full-convolution crop are retained,
    while all epochs at one electrode are convolved together. Epoch boundaries
    remain independent.
    """
    signals = np.asarray(epoch_signals, dtype=np.float64)
    freqs = np.asarray(frequencies, dtype=float)
    if signals.ndim != 2:
        raise ValueError("epoch_signals must have shape (epochs, samples)")
    if not np.all(np.isfinite(signals)):
        raise ValueError("Wavelet analysis requires finite signals")
    if np.any(freqs <= 0.0) or np.any(freqs >= float(sfreq) / 2.0):
        raise ValueError("Wavelet frequencies must lie between zero and Nyquist")
    if float(wavenumber) <= 0.0:
        raise ValueError("wavenumber must be positive")

    power = np.empty((signals.shape[0], len(freqs), signals.shape[1]), dtype=np.float64)
    for frequency_index, frequency in enumerate(freqs):
        temporal_sd = 1.0 / (2.0 * np.pi * (frequency / float(wavenumber)))
        amplitude = 1.0 / np.sqrt(temporal_sd * np.sqrt(np.pi))
        times = np.arange(-3.6 * temporal_sd, 3.6 * temporal_sd, 1.0 / sfreq)
        wavelet = (
            amplitude
            * np.exp(-(times**2) / (2.0 * temporal_sd**2))
            * np.exp(1j * 2.0 * np.pi * frequency * times)
        )
        convolution = fftconvolve(
            signals,
            wavelet[np.newaxis, :],
            mode="full",
            axes=-1,
        )
        start = int(np.ceil(len(wavelet) / 2.0)) - 1
        power[:, frequency_index, :] = np.abs(
            convolution[:, start : start + signals.shape[1]]
        ) ** 2
    return power


def aperiodic_wavelet_background(
    model_frequencies: np.ndarray,
    modeled_psd: np.ndarray,
    aperiodic_psd: np.ndarray,
    wavelet_frequencies: np.ndarray,
    mean_wavelet_power: np.ndarray,
) -> np.ndarray:
    """Map the specparam aperiodic curve into eBOSC wavelet-power units."""
    model_freqs = np.asarray(model_frequencies, dtype=float)
    modeled = np.asarray(modeled_psd, dtype=float)
    aperiodic = np.asarray(aperiodic_psd, dtype=float)
    wavelet_freqs = np.asarray(wavelet_frequencies, dtype=float)
    wavelet_mean = np.asarray(mean_wavelet_power, dtype=float)
    modeled_at_wavelets = np.interp(wavelet_freqs, model_freqs, modeled)
    aperiodic_at_wavelets = np.interp(wavelet_freqs, model_freqs, aperiodic)
    scale = np.divide(
        wavelet_mean,
        modeled_at_wavelets,
        out=np.full_like(wavelet_mean, np.nan),
        where=modeled_at_wavelets > 0.0,
    )
    background = aperiodic_at_wavelets * scale
    if not np.all(np.isfinite(background)) or np.any(background <= 0.0):
        raise RuntimeError("Could not map specparam background to wavelet-power units")
    return background


def power_thresholds(background: np.ndarray, percentile: float) -> np.ndarray:
    """Return BOSC/eBOSC chi-square power thresholds for a background curve."""
    if not 0.0 < float(percentile) < 1.0:
        raise ValueError("percentile must be between zero and one")
    values = np.asarray(background, dtype=float)
    return chi2.ppf(float(percentile), 2) * values / 2.0


def _retain_minimum_runs(mask: np.ndarray, minimum_samples: int) -> np.ndarray:
    result = np.zeros_like(mask, dtype=bool)
    padded = np.pad(np.asarray(mask, dtype=np.int8), (1, 1))
    edges = np.diff(padded)
    starts = np.flatnonzero(edges == 1)
    stops = np.flatnonzero(edges == -1)
    for start, stop in zip(starts, stops):
        if stop - start >= int(minimum_samples):
            result[start:stop] = True
    return result


def detect_frequency_episodes(
    wavelet_power: np.ndarray,
    *,
    sfreq: float,
    frequencies: np.ndarray,
    thresholds: np.ndarray,
    minimum_cycles: float,
    edge_padding_samples: int,
) -> np.ndarray:
    """Apply eBOSC power and frequency-specific duration criteria per epoch."""
    power = np.asarray(wavelet_power, dtype=float)
    freqs = np.asarray(frequencies, dtype=float)
    thresholds = np.asarray(thresholds, dtype=float)
    if power.ndim != 3 or power.shape[1] != len(freqs):
        raise ValueError("wavelet_power must have shape (epochs, frequencies, samples)")
    if thresholds.shape != freqs.shape:
        raise ValueError("thresholds must match frequencies")
    if edge_padding_samples < 0:
        raise ValueError("edge padding cannot be negative")
    if 2 * edge_padding_samples >= power.shape[2]:
        raise ValueError("edge padding removes every sample in an epoch")
    detected = np.zeros_like(power, dtype=bool)
    for epoch_index in range(power.shape[0]):
        for frequency_index, frequency in enumerate(freqs):
            minimum_samples = int(
                math.ceil(float(minimum_cycles) * float(sfreq) / frequency)
            )
            above_threshold = (
                power[epoch_index, frequency_index] > thresholds[frequency_index]
            )
            if edge_padding_samples:
                above_threshold[:edge_padding_samples] = False
                above_threshold[-edge_padding_samples:] = False
            detected[epoch_index, frequency_index] = _retain_minimum_runs(
                above_threshold,
                minimum_samples,
            )
    return detected


def extract_band_bouts(
    detected: np.ndarray,
    wavelet_power: np.ndarray,
    thresholds: np.ndarray,
    frequencies: np.ndarray,
    *,
    band: str,
    band_limits: tuple[float, float] | list[float],
    sfreq: float,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Collapse a detected time-frequency matrix into band-resolved bouts."""
    low_hz, high_hz = (float(value) for value in band_limits)
    freqs = np.asarray(frequencies, dtype=float)
    frequency_mask = (freqs >= low_hz) & (freqs <= high_hz)
    if not frequency_mask.any():
        raise ValueError(f"Band {band} contains no wavelet frequencies")
    selected_detected = detected[:, frequency_mask, :]
    selected_power = wavelet_power[:, frequency_mask, :]
    selected_thresholds = np.asarray(thresholds, dtype=float)[frequency_mask]
    selected_freqs = freqs[frequency_mask]
    band_mask = np.any(selected_detected, axis=1)

    rows: list[dict[str, Any]] = []
    for epoch_index, epoch_mask in enumerate(band_mask):
        padded = np.pad(epoch_mask.astype(np.int8), (1, 1))
        edges = np.diff(padded)
        starts = np.flatnonzero(edges == 1)
        stops = np.flatnonzero(edges == -1)
        previous_stop: int | None = None
        for bout_index, (start, stop) in enumerate(zip(starts, stops), start=1):
            segment_detected = selected_detected[epoch_index, :, start:stop]
            segment_power = selected_power[epoch_index, :, start:stop]
            snr = segment_power / selected_thresholds[:, np.newaxis]
            masked_snr = np.where(segment_detected, snr, -np.inf)
            dominant_indices = np.argmax(masked_snr, axis=0)
            dominant_frequencies = selected_freqs[dominant_indices]
            duration = (stop - start) / float(sfreq)
            rows.append(
                {
                    "band": band,
                    "band_low_hz": low_hz,
                    "band_high_hz": high_hz,
                    "epoch_index": int(epoch_index),
                    "bout_index_within_epoch": int(bout_index),
                    "start_sample": int(start),
                    "stop_sample_exclusive": int(stop),
                    "onset_s": float(start / sfreq),
                    "offset_s": float(stop / sfreq),
                    "duration_s": float(duration),
                    "mean_frequency_hz": float(np.mean(dominant_frequencies)),
                    "n_cycles": float(duration * np.mean(dominant_frequencies)),
                    "mean_wavelet_power": float(
                        np.mean(segment_power[segment_detected])
                    ),
                    "mean_wavelet_amplitude": float(
                        np.mean(np.sqrt(segment_power[segment_detected]))
                    ),
                    "mean_threshold_ratio": float(np.mean(snr[segment_detected])),
                    "inter_bout_interval_s": (
                        float((start - previous_stop) / sfreq)
                        if previous_stop is not None
                        else np.nan
                    ),
                }
            )
            previous_stop = int(stop)
    return pd.DataFrame.from_records(rows), band_mask


def summarize_bouts(
    episodes: pd.DataFrame,
    band_mask: np.ndarray,
    *,
    sfreq: float,
    edge_padding_samples: int = 0,
) -> dict[str, float | int]:
    """Summarize one subject/electrode/band bout table."""
    valid_samples_per_epoch = band_mask.shape[1] - 2 * int(edge_padding_samples)
    if valid_samples_per_epoch <= 0:
        raise ValueError("edge padding leaves no samples for bout summaries")
    analyzed_samples = band_mask.shape[0] * valid_samples_per_epoch
    occupancy = float(np.sum(band_mask) / analyzed_samples)
    analyzed_minutes = analyzed_samples / float(sfreq) / 60.0
    n_bouts = len(episodes)
    if not n_bouts:
        return {
            "n_bouts": 0,
            "oscillatory_occupancy": occupancy,
            "bouts_per_minute": 0.0,
            "bout_duration_mean_s": np.nan,
            "bout_duration_median_s": np.nan,
            "bout_cycles_mean": np.nan,
            "inter_bout_interval_mean_s": np.nan,
            "bout_power_mean": np.nan,
            "bout_amplitude_mean": np.nan,
            "bout_snr_mean": np.nan,
        }
    return {
        "n_bouts": int(n_bouts),
        "oscillatory_occupancy": occupancy,
        "bouts_per_minute": float(n_bouts / analyzed_minutes),
        "bout_duration_mean_s": float(episodes["duration_s"].mean()),
        "bout_duration_median_s": float(episodes["duration_s"].median()),
        "bout_cycles_mean": float(episodes["n_cycles"].mean()),
        "inter_bout_interval_mean_s": float(episodes["inter_bout_interval_s"].mean()),
        "bout_power_mean": float(episodes["mean_wavelet_power"].mean()),
        "bout_amplitude_mean": float(episodes["mean_wavelet_amplitude"].mean()),
        "bout_snr_mean": float(episodes["mean_threshold_ratio"].mean()),
    }


def cycles_within_bouts(
    epoch_signal_uv: np.ndarray,
    band_mask: np.ndarray,
    *,
    sfreq: float,
    band_limits: tuple[float, float] | list[float],
    minimum_overlap: float,
) -> pd.DataFrame:
    """Use bycycle shape features and retain cycles overlapping detected bouts."""
    features = compute_shape_features(
        np.asarray(epoch_signal_uv, dtype=float),
        float(sfreq),
        tuple(float(value) for value in band_limits),
        center_extrema="peak",
    )
    if features.empty:
        return features
    keep = []
    fractions = []
    for _, row in features.iterrows():
        start = max(0, int(row["sample_last_trough"]))
        stop = min(len(band_mask), int(row["sample_next_trough"]) + 1)
        fraction = float(np.mean(band_mask[start:stop])) if stop > start else 0.0
        keep.append(fraction >= float(minimum_overlap))
        fractions.append(fraction)
    selected = features.loc[keep].copy()
    selected["bout_overlap_fraction"] = np.asarray(fractions)[keep]
    selected["cycle_period_s"] = selected["period"] / float(sfreq)
    selected["cycle_frequency_hz"] = np.divide(
        float(sfreq),
        selected["period"],
        out=np.full(len(selected), np.nan, dtype=float),
        where=selected["period"].to_numpy(dtype=float) > 0.0,
    )
    return selected


def summarize_cycles(cycles: pd.DataFrame) -> dict[str, float | int]:
    """Summarize bycycle rows retained inside eBOSC bouts."""
    if cycles.empty:
        return {"n_cycles": 0, **{name: np.nan for name in BAND_FEATURES[14:]}}
    amplitude = cycles["volt_amp"].to_numpy(dtype=float)
    period = cycles["cycle_period_s"].to_numpy(dtype=float)
    amplitude_mean = float(np.mean(amplitude))
    period_mean = float(np.mean(period))
    return {
        "n_cycles": int(len(cycles)),
        "cycle_amplitude_mean_uv": amplitude_mean,
        "cycle_band_amplitude_mean_uv": float(cycles["band_amp"].mean()),
        "cycle_frequency_mean_hz": float(cycles["cycle_frequency_hz"].mean()),
        "cycle_period_mean_s": period_mean,
        "cycle_amplitude_std_uv": float(np.std(amplitude, ddof=1)) if len(cycles) > 1 else 0.0,
        "cycle_amplitude_cv": (
            float(np.std(amplitude, ddof=1) / amplitude_mean)
            if len(cycles) > 1 and amplitude_mean != 0.0
            else np.nan
        ),
        "cycle_period_std_s": float(np.std(period, ddof=1)) if len(cycles) > 1 else 0.0,
        "cycle_period_cv": (
            float(np.std(period, ddof=1) / period_mean)
            if len(cycles) > 1 and period_mean != 0.0
            else np.nan
        ),
        "rise_decay_symmetry_mean": float(cycles["time_rdsym"].mean()),
        "peak_trough_symmetry_mean": float(cycles["time_ptsym"].mean()),
    }
