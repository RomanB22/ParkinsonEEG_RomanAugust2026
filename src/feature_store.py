"""Validated reuse of expensive subject/electrode feature arrays."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_subject_electrode_psd(
    path: str | Path,
    *,
    subjects: list[str],
    electrodes: list[str],
    frequency_range_hz: tuple[float, float],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Load and subset the canonical PSD cube with strict identity checks."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    with np.load(source, allow_pickle=False) as payload:
        required = {"subject_ids", "electrodes", "frequencies_hz", "psd_uv2_hz"}
        missing = sorted(required - set(payload.files))
        if missing:
            raise ValueError(f"PSD cache {source} is missing arrays: {missing}")
        cached_subjects = [str(value) for value in payload["subject_ids"]]
        cached_electrodes = [str(value) for value in payload["electrodes"]]
        frequencies = np.asarray(payload["frequencies_hz"], dtype=float)
        cube = np.asarray(payload["psd_uv2_hz"], dtype=float)
    expected_shape = (len(cached_subjects), len(cached_electrodes), len(frequencies))
    if cube.shape != expected_shape:
        raise ValueError(f"PSD cache cube has shape {cube.shape}, expected {expected_shape}")
    if len(set(cached_subjects)) != len(cached_subjects):
        raise ValueError("PSD cache contains duplicate subjects")
    if len(set(cached_electrodes)) != len(cached_electrodes):
        raise ValueError("PSD cache contains duplicate electrodes")
    missing_subjects = sorted(set(subjects) - set(cached_subjects))
    missing_electrodes = sorted(set(electrodes) - set(cached_electrodes))
    if missing_subjects or missing_electrodes:
        raise ValueError(
            f"PSD cache lacks subjects={missing_subjects} electrodes={missing_electrodes}"
        )
    low, high = frequency_range_hz
    mask = (frequencies >= float(low)) & (frequencies <= float(high))
    selected_frequencies = frequencies[mask]
    if len(selected_frequencies) < 2 or selected_frequencies[0] != float(low) or selected_frequencies[-1] != float(high):
        raise ValueError(
            f"PSD cache does not cover the exact requested range {frequency_range_hz}"
        )
    subject_index = {name: index for index, name in enumerate(cached_subjects)}
    electrode_indices = [cached_electrodes.index(name) for name in electrodes]
    selected = {
        subject: cube[subject_index[subject], electrode_indices][:, mask]
        for subject in subjects
    }
    if not all(np.all(np.isfinite(values)) for values in selected.values()):
        raise ValueError("PSD cache contains non-finite values")
    return selected_frequencies, selected

