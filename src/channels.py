"""Conservative bad-channel detection with visible, saved reasons."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import mne
from scipy.signal import welch


def robust_z(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    center = np.nanmedian(values)
    scale = 1.4826 * np.nanmedian(np.abs(values - center))
    if not np.isfinite(scale) or scale < np.finfo(float).eps:
        scale = np.nanstd(values)
    if not np.isfinite(scale) or scale < np.finfo(float).eps:
        return np.zeros_like(values)
    return (values - center) / scale


@dataclass
class BadChannelResult:
    bad_channels: list[str]
    reasons: dict[str, list[str]]
    candidates: dict[str, list[str]]
    metrics: pd.DataFrame


def detect_bad_channels(raw, config: dict[str, Any]) -> BadChannelResult:
    """Flag only flat channels or channels failing at least two independent tests."""
    picks = np.asarray(mne.pick_types(raw.info, eeg=True, exclude=[]), dtype=int)
    names = [raw.ch_names[index] for index in picks]
    data_uv = raw.get_data(picks=picks) * 1e6
    sfreq = float(raw.info["sfreq"])

    std_uv = np.std(data_uv, axis=1)
    peak_to_peak_uv = np.ptp(data_uv, axis=1)

    stride = max(1, int(round(sfreq / 100.0)))
    correlation_data = data_uv[:, ::stride]
    correlation_data -= np.median(correlation_data, axis=1, keepdims=True)
    correlation_scale = np.std(correlation_data, axis=1, keepdims=True)
    standardized = np.divide(
        correlation_data,
        correlation_scale,
        out=np.zeros_like(correlation_data),
        where=correlation_scale > np.finfo(float).eps,
    )
    correlation = standardized @ standardized.T / standardized.shape[1]
    np.fill_diagonal(correlation, np.nan)
    median_correlation = np.asarray(
        [np.nanmedian(row[np.isfinite(row)]) if np.isfinite(row).any() else 0.0 for row in correlation]
    )

    frequencies, power = welch(
        data_uv,
        fs=sfreq,
        nperseg=min(data_uv.shape[1], int(round(2.0 * sfreq))),
        axis=1,
    )
    low_mask = (frequencies >= 1.0) & (frequencies < 30.0)
    high_mask = (frequencies >= 30.0) & (frequencies <= 50.0)
    low_power = np.trapezoid(power[:, low_mask], frequencies[low_mask], axis=1)
    high_power = np.trapezoid(power[:, high_mask], frequencies[high_mask], axis=1)
    high_frequency_ratio = high_power / np.maximum(low_power, np.finfo(float).tiny)

    std_z = robust_z(np.log10(np.maximum(std_uv, np.finfo(float).tiny)))
    ptp_z = robust_z(np.log10(np.maximum(peak_to_peak_uv, np.finfo(float).tiny)))
    corr_z = robust_z(median_correlation)
    hf_z = robust_z(np.log10(np.maximum(high_frequency_ratio, np.finfo(float).tiny)))

    threshold = float(config["metric_robust_z"])
    correlation_threshold = float(config["correlation_robust_z"])
    flat_threshold = float(config["flat_std_uv"])
    min_correlation = float(config["minimum_median_correlation"])
    minimum_flags = int(config["minimum_independent_flags"])

    reasons: dict[str, list[str]] = {}
    candidates: dict[str, list[str]] = {}
    for index, name in enumerate(names):
        flags: list[str] = []
        if std_uv[index] < flat_threshold:
            flags.append("flat_signal")
        if std_z[index] > threshold:
            flags.append("high_variance")
        if ptp_z[index] > threshold:
            flags.append("extreme_peak_to_peak")
        if corr_z[index] < -correlation_threshold or median_correlation[index] < min_correlation:
            flags.append("poor_channel_correlation")
        if hf_z[index] > threshold:
            flags.append("excess_high_frequency_power")
        if flags:
            candidates[name] = flags
        if "flat_signal" in flags or len(flags) >= minimum_flags:
            reasons[name] = flags

    metrics = pd.DataFrame(
        {
            "channel": names,
            "std_uv": std_uv,
            "std_robust_z": std_z,
            "peak_to_peak_uv": peak_to_peak_uv,
            "peak_to_peak_robust_z": ptp_z,
            "median_correlation": median_correlation,
            "correlation_robust_z": corr_z,
            "high_frequency_ratio": high_frequency_ratio,
            "high_frequency_robust_z": hf_z,
            "candidate_reasons": [";".join(candidates.get(name, [])) for name in names],
            "confirmed_bad": [name in reasons for name in names],
        }
    )
    return BadChannelResult(sorted(reasons), reasons, candidates, metrics)
