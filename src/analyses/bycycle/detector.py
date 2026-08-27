"""Numerical routines for independent bycycle burst detection."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from bycycle.features import compute_burst_features, compute_shape_features


METRICS = (
    "oscillatory_occupancy",
    "bouts_per_minute",
    "bout_duration_mean_s",
    "bout_duration_median_s",
    "bout_cycles_mean",
    "inter_bout_interval_mean_s",
    "burst_cycle_fraction",
    "cycle_amplitude_mean_uv",
    "cycle_frequency_mean_hz",
    "amplitude_consistency_mean",
    "period_consistency_mean",
    "monotonicity_mean",
)


def _minimum_runs(values: np.ndarray, minimum: int) -> np.ndarray:
    """Return a copy retaining only True runs at least ``minimum`` long."""
    result = np.asarray(values, dtype=bool).copy()
    if not len(result):
        return result
    edges = np.diff(np.pad(result.astype(np.int8), (1, 1)))
    starts = np.flatnonzero(edges == 1)
    stops = np.flatnonzero(edges == -1)
    for start, stop in zip(starts, stops):
        if stop - start < int(minimum):
            result[start:stop] = False
    return result


def detect_epoch_bursts(
    signal_uv: np.ndarray,
    *,
    sfreq: float,
    band_limits: tuple[float, float] | list[float],
    settings: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Detect bursts in one epoch from cycle consistency, independently of eBOSC.

    The implementation uses bycycle for extrema, shape, and burst-feature
    extraction. Thresholding is applied locally so bycycle 1.2 remains
    compatible with pandas copy-on-write arrays.
    """
    signal = np.asarray(signal_uv, dtype=float)
    if signal.ndim != 1 or not np.all(np.isfinite(signal)):
        raise ValueError("signal_uv must be one finite vector")
    fs = float(sfreq)
    if fs <= 0.0:
        raise ValueError("sfreq must be positive")
    limits = tuple(float(value) for value in band_limits)
    shapes = compute_shape_features(
        signal,
        fs,
        limits,
        center_extrema=str(settings["center_extrema"]),
    ).reset_index(drop=True)
    mask = np.zeros(len(signal), dtype=bool)
    if shapes.empty:
        return shapes.assign(is_burst=pd.Series(dtype=bool)), pd.DataFrame(), mask

    burst_features = compute_burst_features(shapes, signal, burst_method="cycles")
    cycles = pd.concat(
        [shapes.reset_index(drop=True), burst_features.reset_index(drop=True)], axis=1
    )
    candidate = (
        cycles["amp_fraction"].to_numpy(float)
        > float(settings["amplitude_fraction_threshold"])
    )
    candidate &= (
        cycles["amp_consistency"].to_numpy(float)
        > float(settings["amplitude_consistency_threshold"])
    )
    candidate &= (
        cycles["period_consistency"].to_numpy(float)
        > float(settings["period_consistency_threshold"])
    )
    candidate &= (
        cycles["monotonicity"].to_numpy(float)
        > float(settings["monotonicity_threshold"])
    )
    edge = int(round(float(settings["edge_padding_seconds"]) * fs))
    interior_stop = len(signal) - edge
    eligible = (
        cycles["sample_last_trough"].to_numpy(int) >= edge
    ) & (cycles["sample_next_trough"].to_numpy(int) < interior_stop)
    cycles["edge_eligible"] = eligible
    candidate &= eligible
    if len(candidate):
        candidate[0] = False
        candidate[-1] = False
    cycles["is_burst"] = _minimum_runs(
        candidate, int(settings["minimum_consecutive_cycles"])
    )
    cycles["cycle_period_s"] = cycles["period"].to_numpy(float) / fs
    cycles["cycle_frequency_hz"] = np.divide(
        fs,
        cycles["period"].to_numpy(float),
        out=np.full(len(cycles), np.nan),
        where=cycles["period"].to_numpy(float) > 0.0,
    )

    selected_indices = np.flatnonzero(cycles["is_burst"].to_numpy(bool))
    event_rows: list[dict[str, float | int]] = []
    if len(selected_indices):
        split_points = np.flatnonzero(np.diff(selected_indices) > 1) + 1
        for bout_index, indices in enumerate(
            np.split(selected_indices, split_points), start=1
        ):
            selected = cycles.iloc[indices]
            start = max(edge, int(selected.iloc[0]["sample_last_trough"]))
            stop = min(interior_stop, int(selected.iloc[-1]["sample_next_trough"]) + 1)
            if stop <= start:
                continue
            mask[start:stop] = True
            event_rows.append(
                {
                    "bout_index_within_epoch": bout_index,
                    "start_sample": start,
                    "stop_sample_exclusive": stop,
                    "onset_s": start / fs,
                    "offset_s": stop / fs,
                    "duration_s": (stop - start) / fs,
                    "n_cycles": int(len(selected)),
                    "cycle_frequency_mean_hz": float(
                        selected["cycle_frequency_hz"].mean()
                    ),
                    "cycle_amplitude_mean_uv": float(selected["volt_amp"].mean()),
                }
            )
    events = pd.DataFrame.from_records(event_rows)
    return cycles, events, mask


def summarize_detection(
    cycles: pd.DataFrame,
    events: pd.DataFrame,
    *,
    analyzed_duration_s: float,
) -> dict[str, float | int]:
    """Summarize pooled epoch-level cycles and events for one electrode/band."""
    duration = float(analyzed_duration_s)
    if duration <= 0.0:
        raise ValueError("analyzed_duration_s must be positive")
    if cycles.empty or "edge_eligible" not in cycles:
        eligible = pd.DataFrame(columns=cycles.columns)
    else:
        eligible = cycles.loc[cycles["edge_eligible"].astype(bool)].copy()
    if eligible.empty or "is_burst" not in eligible:
        bursting = pd.DataFrame(columns=eligible.columns)
    else:
        bursting = eligible.loc[eligible["is_burst"].astype(bool)].copy()
    n_bouts = int(len(events))
    total_bout_duration = float(events["duration_s"].sum()) if n_bouts else 0.0
    finite_intervals = (
        events["inter_bout_interval_s"].dropna().to_numpy(float)
        if n_bouts and "inter_bout_interval_s" in events
        else np.asarray([], dtype=float)
    )

    def mean(column: str) -> float:
        return float(bursting[column].mean()) if len(bursting) else np.nan

    return {
        "n_candidate_cycles": int(len(eligible)),
        "n_burst_cycles": int(len(bursting)),
        "n_bouts": n_bouts,
        "analyzed_duration_s": duration,
        "oscillatory_occupancy": total_bout_duration / duration,
        "bouts_per_minute": n_bouts / (duration / 60.0),
        "bout_duration_mean_s": (
            float(events["duration_s"].mean()) if n_bouts else np.nan
        ),
        "bout_duration_median_s": (
            float(events["duration_s"].median()) if n_bouts else np.nan
        ),
        "bout_cycles_mean": float(events["n_cycles"].mean()) if n_bouts else np.nan,
        "inter_bout_interval_mean_s": (
            float(np.mean(finite_intervals)) if len(finite_intervals) else np.nan
        ),
        "burst_cycle_fraction": (
            float(len(bursting) / len(eligible)) if len(eligible) else np.nan
        ),
        "cycle_amplitude_mean_uv": mean("volt_amp"),
        "cycle_frequency_mean_hz": mean("cycle_frequency_hz"),
        "amplitude_consistency_mean": mean("amp_consistency"),
        "period_consistency_mean": mean("period_consistency"),
        "monotonicity_mean": mean("monotonicity"),
    }
