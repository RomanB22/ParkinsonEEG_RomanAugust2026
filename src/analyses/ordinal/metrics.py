"""Ordinal probability pooling and information-theory metrics."""

from __future__ import annotations

import itertools
import math
import warnings
from functools import lru_cache
from typing import Any

import numpy as np
import ordpy
import pandas as pd
from scipy.signal import butter, sosfiltfilt


CORE_METRICS = ("entropy", "complexity", "fisher_information")
WEIGHTED_METRIC = "weighted_permutation_entropy"
RENYI_ALPHA_METRICS = (
    (0.1, "renyi_entropy_alpha_0_1", "renyi_complexity_alpha_0_1"),
    (0.5, "renyi_entropy_alpha_0_5", "renyi_complexity_alpha_0_5"),
    (0.9, "renyi_entropy_alpha_0_9", "renyi_complexity_alpha_0_9"),
    (1.1, "renyi_entropy_alpha_1_1", "renyi_complexity_alpha_1_1"),
    (2.0, "renyi_entropy_alpha_2", "renyi_complexity_alpha_2"),
    (5.0, "renyi_entropy_alpha_5", "renyi_complexity_alpha_5"),
    (10.0, "renyi_entropy_alpha_10", "renyi_complexity_alpha_10"),
)
RENYI_ALPHAS = tuple(alpha for alpha, _, _ in RENYI_ALPHA_METRICS)
RENYI_METRICS = tuple(
    metric
    for _, entropy_metric, complexity_metric in RENYI_ALPHA_METRICS
    for metric in (entropy_metric, complexity_metric)
)
METRICS = (*CORE_METRICS, WEIGHTED_METRIC, *RENYI_METRICS)


@lru_cache(maxsize=None)
def _permutation_code_lookup(dx: int) -> tuple[np.ndarray, np.ndarray]:
    """Map compact base-D permutation codes to lexicographic indices."""
    permutations = np.asarray(list(itertools.permutations(range(dx))), dtype=np.int64)
    powers = np.power(dx, np.arange(dx - 1, -1, -1), dtype=np.int64)
    permutation_codes = permutations @ powers
    lookup = np.full(dx**dx, -1, dtype=np.int64)
    lookup[permutation_codes] = np.arange(len(permutations), dtype=np.int64)
    return lookup, powers


def filter_epoch_data(
    data: np.ndarray,
    *,
    sfreq: float,
    low_hz: float,
    high_hz: float,
    order: int = 4,
) -> np.ndarray:
    """Zero-phase band-pass each accepted epoch without joining epoch gaps."""
    array = np.asarray(data, dtype=np.float64)
    if array.ndim != 3:
        raise ValueError("data must have shape (epochs, channels, samples)")
    if not np.all(np.isfinite(array)):
        raise ValueError("Band filtering requires finite epoch samples")
    nyquist = float(sfreq) / 2.0
    if not 0.0 < float(low_hz) < float(high_hz) < nyquist:
        raise ValueError(
            f"Band must satisfy 0 < low_hz < high_hz < Nyquist ({nyquist:g} Hz)"
        )
    if int(order) != order or int(order) < 1:
        raise ValueError("Filter order must be a positive integer")
    sos = butter(
        int(order),
        [float(low_hz), float(high_hz)],
        btype="bandpass",
        fs=float(sfreq),
        output="sos",
    )
    # Filtering along the final axis treats every epoch/channel independently.
    # Thus rejected-data gaps and boundaries cannot produce filter transitions.
    return sosfiltfilt(sos, array, axis=-1)


def _validate_parameters(dx: int, tau: int) -> None:
    if not 2 <= dx <= 7:
        raise ValueError("embedding_dimension must be between 2 and 7")
    if tau < 1:
        raise ValueError("delay must be at least 1 sample")


def ordinal_probabilities(
    epoch_data: np.ndarray,
    *,
    dx: int = 3,
    tau: int = 1,
    tie_precision: int | None = None,
) -> tuple[np.ndarray, int, int]:
    """Pool ordinal-pattern probabilities without crossing epoch boundaries.

    Parameters
    ----------
    epoch_data
        Two-dimensional array shaped ``(epochs, samples)`` for one electrode.
    dx, tau
        Bandt-Pompe embedding dimension and delay in samples.
    tie_precision
        ``None`` means no rounding and therefore retains all floating-point
        decimals, matching the default tie policy used by ``ordpy``.

    Returns
    -------
    probabilities, n_patterns, n_exact_ties
        Probabilities are in lexicographic permutation order, as required by
        ``ordpy.fisher_shannon``. Patterns that would span two accepted epochs
        are excluded.
    """
    _validate_parameters(dx, tau)
    data = np.asarray(epoch_data, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError("epoch_data must have shape (epochs, samples)")
    if data.shape[0] == 0:
        raise ValueError("At least one accepted epoch is required")
    if not np.all(np.isfinite(data)):
        raise ValueError("Ordinal analysis requires finite epoch samples")

    span = (dx - 1) * tau
    n_times = data.shape[1]
    if n_times <= span:
        raise ValueError(
            f"Epochs contain {n_times} samples but dx={dx}, tau={tau} "
            f"requires at least {span + 1}"
        )

    # This is the vectorized equivalent of ordpy.ordinal_sequence for a time
    # series. Keeping epochs as the first dimension prevents pattern windows
    # from ever spanning a rejected-data gap.
    ranked_data = data if tie_precision is None else np.round(data, tie_precision)
    windows = np.lib.stride_tricks.sliding_window_view(
        ranked_data, span + 1, axis=1
    )[..., ::tau]
    symbols = np.argsort(windows, axis=-1).reshape(-1, dx)

    # Encode each permutation as a base-D integer and look up its lexicographic
    # permutation index. This is mathematically identical to row-wise unique
    # counting but avoids repeatedly sorting a large structured array at D>=5.
    lookup, powers = _permutation_code_lookup(dx)
    permutation_indices = lookup[symbols @ powers]
    if np.any(permutation_indices < 0):
        raise RuntimeError("Ordinal symbols contain an invalid permutation")
    counts = np.bincount(
        permutation_indices, minlength=math.factorial(dx)
    ).astype(np.int64, copy=False)

    n_patterns = int(counts.sum())
    if n_patterns != data.shape[0] * (n_times - span):
        raise RuntimeError("Ordinal-pattern count does not match the epoch-safe expectation")

    # Count exact ties at full input precision for provenance. No rounding or
    # jitter is introduced when tie_precision is None.
    sorted_windows = np.sort(windows, axis=-1)
    n_exact_ties = int(np.any(np.diff(sorted_windows, axis=-1) == 0, axis=-1).sum())
    return counts / n_patterns, n_patterns, n_exact_ties


def metrics_from_probabilities(
    probabilities: np.ndarray,
    *,
    dx: int,
) -> dict[str, float]:
    """Calculate Shannon, Fisher, and configured Rényi ordinal quantities."""
    entropy, complexity = ordpy.complexity_entropy(probabilities, dx=dx, probs=True)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Be mindful the correct calculation of Fisher information.*",
            category=UserWarning,
        )
        fisher_entropy, fisher = ordpy.fisher_shannon(
            probabilities, dx=dx, probs=True
        )
    if not np.isclose(entropy, fisher_entropy, rtol=1e-12, atol=1e-12):
        raise RuntimeError("ordpy returned inconsistent entropy values for HxC and HxF")

    renyi_pairs = np.asarray(
        ordpy.renyi_complexity_entropy(
            probabilities,
            alpha=RENYI_ALPHAS,
            dx=dx,
            probs=True,
        ),
        dtype=float,
    )
    if renyi_pairs.shape != (len(RENYI_ALPHAS), 2):
        raise RuntimeError("ordpy.renyi_complexity_entropy returned an unexpected shape")

    result = {
        "entropy": float(entropy),
        "complexity": float(complexity),
        "fisher_information": float(fisher),
    }
    for index, (_, entropy_metric, complexity_metric) in enumerate(
        RENYI_ALPHA_METRICS
    ):
        result[entropy_metric] = float(renyi_pairs[index, 0])
        result[complexity_metric] = float(renyi_pairs[index, 1])

    values = np.asarray(list(result.values()), dtype=float)
    if not np.all(np.isfinite(values)):
        raise RuntimeError("ordpy returned a non-finite metric")
    return result


def weighted_permutation_entropy_epoch_data(
    epoch_data: np.ndarray,
    *,
    dx: int,
    tau: int,
    tie_precision: int | None = None,
) -> float:
    """Calculate epoch-safe weighted permutation entropy with ordpy.

    ``ordpy.weighted_permutation_entropy`` accepts one uninterrupted series.
    Calling it separately for every accepted epoch prevents an embedding from
    crossing an epoch/rejection boundary; the returned value is the
    pattern-count-weighted mean across epochs.
    """
    _validate_parameters(dx, tau)
    data = np.asarray(epoch_data, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError("epoch_data must have shape (epochs, samples)")
    span = (int(dx) - 1) * int(tau)
    if data.shape[0] == 0 or data.shape[1] <= span:
        raise ValueError("Each epoch must contain at least one embedding window")
    if not np.all(np.isfinite(data)):
        raise ValueError("Weighted permutation entropy requires finite epoch samples")
    values = [
        float(
            ordpy.weighted_permutation_entropy(
                epoch,
                dx=int(dx),
                taux=int(tau),
                tie_precision=tie_precision,
            )
        )
        for epoch in data
    ]
    weights = np.full(len(values), data.shape[1] - span, dtype=float)
    result = float(np.average(values, weights=weights))
    if not np.isfinite(result):
        raise RuntimeError("ordpy returned a non-finite weighted permutation entropy")
    return result


def analyze_epoch_data(
    data: np.ndarray,
    channel_names: list[str],
    *,
    subject_id: str,
    group: str,
    sfreq: float,
    dx: int = 3,
    tau: int = 1,
    tie_precision: int | None = None,
) -> pd.DataFrame:
    """Calculate Shannon, Fisher, and Rényi quantities for each electrode."""
    array = np.asarray(data, dtype=np.float64)
    if array.ndim != 3:
        raise ValueError("data must have shape (epochs, channels, samples)")
    if array.shape[1] != len(channel_names):
        raise ValueError("channel_names does not match the channel dimension")

    rows: list[dict[str, Any]] = []
    for channel_index, electrode in enumerate(channel_names):
        probabilities, n_patterns, n_exact_ties = ordinal_probabilities(
            array[:, channel_index, :],
            dx=dx,
            tau=tau,
            tie_precision=tie_precision,
        )
        metric_values = metrics_from_probabilities(probabilities, dx=dx)
        metric_values[WEIGHTED_METRIC] = weighted_permutation_entropy_epoch_data(
            array[:, channel_index, :],
            dx=dx,
            tau=tau,
            tie_precision=tie_precision,
        )
        rows.append(
            {
                "subject_id": subject_id,
                "group": group,
                "electrode": electrode,
                **metric_values,
                "n_epochs": int(array.shape[0]),
                "samples_per_epoch": int(array.shape[2]),
                "n_samples": int(array.shape[0] * array.shape[2]),
                "n_ordinal_patterns": n_patterns,
                "n_exact_tied_patterns": n_exact_ties,
                "exact_tie_fraction": n_exact_ties / n_patterns,
                "sampling_frequency_hz": float(sfreq),
                "embedding_dimension": int(dx),
                "delay_samples": int(tau),
                "delay_seconds": float(tau / sfreq),
                "tie_precision": "full_float64" if tie_precision is None else str(tie_precision),
            }
        )
    return pd.DataFrame.from_records(rows)


def subject_electrode_means(electrode_metrics: pd.DataFrame) -> pd.DataFrame:
    """Average the electrode-level metrics within each participant."""
    required = {"subject_id", "group", "electrode", *METRICS}
    missing = sorted(required - set(electrode_metrics.columns))
    if missing:
        raise ValueError(f"Missing electrode metric columns: {missing}")
    means = (
        electrode_metrics.groupby(["subject_id", "group"], sort=True)[list(METRICS)]
        .mean()
        .reset_index()
    )
    counts = (
        electrode_metrics.groupby(["subject_id", "group"], sort=True)["electrode"]
        .nunique()
        .rename("n_electrodes")
        .reset_index()
    )
    return means.merge(counts, on=["subject_id", "group"], validate="one_to_one")


def band_subject_electrode_means(electrode_metrics: pd.DataFrame) -> pd.DataFrame:
    """Average electrode-level metrics within every participant and band."""
    required = {
        "subject_id",
        "group",
        "band",
        "band_low_hz",
        "band_high_hz",
        "electrode",
        *METRICS,
    }
    missing = sorted(required - set(electrode_metrics.columns))
    if missing:
        raise ValueError(f"Missing band metric columns: {missing}")
    keys = ["subject_id", "group", "band", "band_low_hz", "band_high_hz"]
    means = (
        electrode_metrics.groupby(keys, sort=False)[list(METRICS)].mean().reset_index()
    )
    counts = (
        electrode_metrics.groupby(keys, sort=False)["electrode"]
        .nunique()
        .rename("n_electrodes")
        .reset_index()
    )
    return means.merge(counts, on=keys, validate="one_to_one")
