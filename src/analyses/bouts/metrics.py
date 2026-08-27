"""Boundary-safe ordinal representations and Shannon metrics for EEG bouts."""

from __future__ import annotations

import itertools
import math
import warnings
from functools import lru_cache
from typing import Any, Iterable

import numpy as np
import ordpy
import pandas as pd


METRICS = ("entropy", "complexity", "fisher_information")


def validate_ordinal_parameters(dx: int, tau: int) -> None:
    if not 2 <= int(dx) <= 7:
        raise ValueError("embedding_dimension must be between 2 and 7")
    if int(tau) < 1:
        raise ValueError("delay_samples must be at least one")


@lru_cache(maxsize=None)
def ordinal_patterns(dx: int) -> tuple[tuple[int, ...], ...]:
    validate_ordinal_parameters(dx, 1)
    return tuple(itertools.permutations(range(int(dx))))


@lru_cache(maxsize=None)
def _pattern_lookup(dx: int) -> dict[tuple[int, ...], int]:
    return {pattern: index for index, pattern in enumerate(ordinal_patterns(dx))}


def ordinal_counts(
    signal: np.ndarray,
    *,
    dx: int,
    tau: int,
    tie_precision: int | None = None,
) -> tuple[np.ndarray, int]:
    """Return lexicographically ordered pattern counts for one uninterrupted bout.

    A segment that is too short to contain one embedding returns an all-zero
    vector. This makes short-bout exclusions explicit without joining segments.
    """
    validate_ordinal_parameters(dx, tau)
    values = np.asarray(signal, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("signal must be one-dimensional")
    if not np.all(np.isfinite(values)):
        raise ValueError("Ordinal analysis requires finite samples")
    span = (int(dx) - 1) * int(tau)
    counts = np.zeros(math.factorial(int(dx)), dtype=np.int64)
    if len(values) <= span:
        return counts, 0

    ranked = values if tie_precision is None else np.round(values, int(tie_precision))
    windows = np.lib.stride_tricks.sliding_window_view(ranked, span + 1)[..., ::tau]
    symbols = np.argsort(windows, axis=-1)
    unique, unique_counts = np.unique(symbols, axis=0, return_counts=True)
    lookup = _pattern_lookup(int(dx))
    for pattern, count in zip(unique, unique_counts):
        counts[lookup[tuple(int(value) for value in pattern)]] = int(count)

    sorted_windows = np.sort(windows, axis=-1)
    ties = int(np.any(np.diff(sorted_windows, axis=-1) == 0, axis=-1).sum())
    if int(counts.sum()) != len(values) - span:
        raise RuntimeError("Ordinal-pattern count does not match segment length")
    return counts, ties


def shannon_metrics_from_counts(counts: np.ndarray, *, dx: int) -> dict[str, float]:
    """Calculate regular permutation entropy, complexity, and Fisher information."""
    values = np.asarray(counts, dtype=np.int64)
    expected = math.factorial(int(dx))
    if values.shape != (expected,):
        raise ValueError(f"counts must have shape ({expected},)")
    if np.any(values < 0):
        raise ValueError("counts cannot be negative")
    total = int(values.sum())
    if total == 0:
        return {metric: np.nan for metric in METRICS}
    probabilities = values / total
    entropy, complexity = ordpy.complexity_entropy(
        probabilities, dx=int(dx), probs=True
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Be mindful the correct calculation of Fisher information.*",
            category=UserWarning,
        )
        fisher_entropy, fisher = ordpy.fisher_shannon(
            probabilities, dx=int(dx), probs=True
        )
    if not np.isclose(entropy, fisher_entropy, rtol=1e-12, atol=1e-12):
        raise RuntimeError("ordpy returned inconsistent entropy values")
    result = {
        "entropy": float(entropy),
        "complexity": float(complexity),
        "fisher_information": float(fisher),
    }
    if not np.all(np.isfinite(list(result.values()))):
        raise RuntimeError("ordpy returned a non-finite metric")
    return result


def pattern_diagnostics(counts: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(counts, dtype=np.int64)
    total = int(values.sum())
    observed = int(np.count_nonzero(values))
    return {
        "n_ordinal_patterns": total,
        "n_observed_pattern_states": observed,
        "ordinal_state_space_size": int(len(values)),
        "ordinal_state_space_coverage": float(observed / len(values)),
    }


def analyze_bout_segments(
    epoch_signals: np.ndarray,
    episodes: pd.DataFrame,
    *,
    dx: int,
    tau: int,
    tie_precision: int | None = None,
) -> tuple[np.ndarray, dict[str, Any], pd.DataFrame, dict[str, Any] | None]:
    """Analyze variable-length bouts and pool counts without joining boundaries.

    Parameters
    ----------
    epoch_signals
        Band-pass filtered data shaped ``(epochs, samples)`` for one electrode.
    episodes
        Detected bouts with epoch, start, and exclusive-stop sample indices.

    Returns
    -------
    pooled_counts, pooled_summary, bout_metrics, example
        ``pooled_counts`` is the exact subject/electrode/band ordinal
        representation. ``example`` contains the first analyzable bout.
    """
    signals = np.asarray(epoch_signals, dtype=np.float64)
    if signals.ndim != 2:
        raise ValueError("epoch_signals must have shape (epochs, samples)")
    required = {"epoch_index", "start_sample", "stop_sample_exclusive"}
    missing = sorted(required - set(episodes.columns))
    if missing and len(episodes):
        raise ValueError(f"episodes are missing columns: {missing}")

    pooled = np.zeros(math.factorial(int(dx)), dtype=np.int64)
    bout_rows: list[dict[str, Any]] = []
    example: dict[str, Any] | None = None
    total_ties = 0
    analyzable = 0
    for episode_number, (_, episode) in enumerate(episodes.iterrows(), start=1):
        epoch_index = int(episode["epoch_index"])
        start = int(episode["start_sample"])
        stop = int(episode["stop_sample_exclusive"])
        if not 0 <= epoch_index < signals.shape[0]:
            raise ValueError(f"Bout {episode_number} has invalid epoch index {epoch_index}")
        if not 0 <= start < stop <= signals.shape[1]:
            raise ValueError(
                f"Bout {episode_number} has invalid sample interval [{start}, {stop})"
            )
        segment = signals[epoch_index, start:stop]
        counts, ties = ordinal_counts(
            segment, dx=dx, tau=tau, tie_precision=tie_precision
        )
        diagnostics = pattern_diagnostics(counts)
        analyzable_bout = int(diagnostics["n_ordinal_patterns"] > 0)
        metrics = shannon_metrics_from_counts(counts, dx=dx)
        pooled += counts
        total_ties += ties
        analyzable += analyzable_bout
        row = episode.to_dict()
        row.update(
            {
                "bout_sequence_number": int(episode_number),
                **metrics,
                **diagnostics,
                "n_exact_tied_patterns": int(ties),
                "exact_tie_fraction": (
                    float(ties / diagnostics["n_ordinal_patterns"])
                    if diagnostics["n_ordinal_patterns"]
                    else np.nan
                ),
                "analyzable_ordinal_bout": analyzable_bout,
            }
        )
        bout_rows.append(row)
        if example is None and analyzable_bout:
            example = {
                "epoch_index": epoch_index,
                "start_sample": start,
                "stop_sample_exclusive": stop,
                "signal": segment.copy(),
                "counts": counts.copy(),
            }

    pooled_diagnostics = pattern_diagnostics(pooled)
    pooled_summary = {
        **shannon_metrics_from_counts(pooled, dx=dx),
        **pooled_diagnostics,
        "n_detected_bouts": int(len(episodes)),
        "n_analyzable_ordinal_bouts": int(analyzable),
        "n_short_bouts_excluded": int(len(episodes) - analyzable),
        "n_exact_tied_patterns": int(total_ties),
        "exact_tie_fraction": (
            float(total_ties / pooled_diagnostics["n_ordinal_patterns"])
            if pooled_diagnostics["n_ordinal_patterns"]
            else np.nan
        ),
    }
    return pooled, pooled_summary, pd.DataFrame.from_records(bout_rows), example


def pool_count_vectors(vectors: Iterable[np.ndarray], *, dx: int) -> np.ndarray:
    """Pool precomputed boundary-safe count vectors."""
    result = np.zeros(math.factorial(int(dx)), dtype=np.int64)
    for vector in vectors:
        values = np.asarray(vector, dtype=np.int64)
        if values.shape != result.shape:
            raise ValueError("All count vectors must match the configured state space")
        result += values
    return result

