"""Numerical PSD estimation, band integration, and bootstrap summaries."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from scipy.signal import welch


def compute_subject_electrode_psd(
    epoch_data: np.ndarray,
    sfreq: float,
    *,
    fmin: float = 1.0,
    fmax: float = 50.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return one Welch PSD per electrode after concatenating accepted epochs.

    Input samples are expected in volts, as returned by MNE. Output density is
    converted to µV²/Hz. Epochs are concatenated in their array order for each
    electrode. Welch then uses one non-overlapping four-second segment per
    accepted epoch, retaining the native 0.25 Hz grid at 120 Hz. Thus the
    subject/electrode result is the pooled Welch mean, not a median of separate
    epoch PSDs.
    """
    data = np.asarray(epoch_data, dtype=np.float64)
    if data.ndim != 3:
        raise ValueError("epoch_data must have shape (epochs, channels, samples)")
    if data.shape[0] == 0 or data.shape[1] == 0:
        raise ValueError("At least one epoch and one EEG channel are required")
    if data.shape[2] < 4:
        raise ValueError("Epochs are too short for PSD estimation")
    if not np.all(np.isfinite(data)):
        raise ValueError("PSD analysis requires finite epoch samples")
    if not 0 <= fmin < fmax <= sfreq / 2:
        raise ValueError("Require 0 <= fmin < fmax <= Nyquist")

    # (epochs, channels, samples) -> (channels, concatenated samples). This
    # transpose is essential: a direct reshape would interleave channels.
    concatenated = data.transpose(1, 0, 2).reshape(data.shape[1], -1)
    frequencies, density_v2_hz = welch(
        concatenated,
        fs=float(sfreq),
        window="hann",
        nperseg=data.shape[2],
        noverlap=0,
        nfft=data.shape[2],
        detrend="constant",
        return_onesided=True,
        scaling="density",
        axis=-1,
        average="mean",
    )
    frequency_mask = (frequencies >= fmin) & (frequencies <= fmax)
    if frequency_mask.sum() < 2:
        raise ValueError("The requested PSD interval contains fewer than two bins")
    density_uv2_hz = density_v2_hz[..., frequency_mask] * 1e12
    return frequencies[frequency_mask], density_uv2_hz


def integrate_bands(
    frequencies: np.ndarray,
    electrode_psd: np.ndarray,
    bands: Mapping[str, tuple[float, float] | list[float]],
) -> dict[str, np.ndarray]:
    """Integrate linear PSD into absolute band power in µV²."""
    frequencies = np.asarray(frequencies, dtype=float)
    psd = np.asarray(electrode_psd, dtype=float)
    if frequencies.ndim != 1 or psd.shape[-1] != len(frequencies):
        raise ValueError("PSD's final dimension must match the frequency vector")
    if not np.all(np.diff(frequencies) > 0):
        raise ValueError("Frequencies must be strictly increasing")
    results: dict[str, np.ndarray] = {}
    for name, limits in bands.items():
        low, high = (float(value) for value in limits)
        if not low < high:
            raise ValueError(f"Band {name} must have increasing limits")
        mask = (frequencies >= low) & (frequencies <= high)
        if mask.sum() < 2:
            raise ValueError(f"Band {name} contains fewer than two frequency bins")
        results[name] = np.trapezoid(psd[..., mask], frequencies[mask], axis=-1)
    return results


def bootstrap_median_ci(
    values: np.ndarray,
    *,
    n_resamples: int = 2000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pointwise nonparametric bootstrap CI for a subject-level median."""
    array = np.asarray(values, dtype=float)
    if array.ndim != 2:
        raise ValueError("values must have shape (subjects, features)")
    if array.shape[0] < 2:
        raise ValueError("At least two subjects are required for a confidence interval")
    if not np.all(np.isfinite(array)):
        raise ValueError("Bootstrap inputs must be finite")
    if n_resamples < 100:
        raise ValueError("n_resamples must be at least 100")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1")

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, array.shape[0], size=(n_resamples, array.shape[0]))
    resampled_medians = np.median(array[indices], axis=1)
    alpha = (1.0 - confidence_level) / 2.0
    lower, upper = np.quantile(resampled_medians, [alpha, 1.0 - alpha], axis=0)
    return np.median(array, axis=0), lower, upper


def to_db(values: np.ndarray) -> np.ndarray:
    """Convert a positive power or density referenced to one µV unit to dB."""
    array = np.asarray(values, dtype=float)
    return 10.0 * np.log10(np.maximum(array, np.finfo(float).tiny))
