"""Ordinal probability pooling and information-theory metrics."""

from __future__ import annotations

import itertools
import math
import warnings
from typing import Any

import numpy as np
import ordpy
import pandas as pd


METRICS = ("entropy", "complexity", "fisher_information")


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

    permutations = list(itertools.permutations(range(dx)))
    permutation_index = {pattern: index for index, pattern in enumerate(permutations)}
    counts = np.zeros(math.factorial(dx), dtype=np.int64)
    unique, unique_counts = np.unique(symbols, axis=0, return_counts=True)
    for pattern, count in zip(unique, unique_counts):
        counts[permutation_index[tuple(int(value) for value in pattern)]] = int(count)

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
) -> tuple[float, float, float]:
    """Calculate normalized H, statistical complexity, and Fisher information."""
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
    values = np.asarray([entropy, complexity, fisher], dtype=float)
    if not np.all(np.isfinite(values)):
        raise RuntimeError("ordpy returned a non-finite metric")
    return tuple(float(value) for value in values)


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
    """Calculate one H/C/F triplet per electrode for one participant."""
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
        entropy, complexity, fisher = metrics_from_probabilities(
            probabilities, dx=dx
        )
        rows.append(
            {
                "subject_id": subject_id,
                "group": group,
                "electrode": electrode,
                "entropy": entropy,
                "complexity": complexity,
                "fisher_information": fisher,
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
