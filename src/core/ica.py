"""ICA fitting, conservative suggestions, and explicitly reviewed removal."""

from __future__ import annotations

from typing import Any

import mne
import numpy as np
import pandas as pd
from scipy.signal import welch


FRONTAL_CHANNELS = ("Fp1", "Fp2", "AF7", "AF8", "AF3", "AF4")
ICLABEL_CLASSES = (
    "brain",
    "muscle_artifact",
    "eye_blink",
    "heart_beat",
    "line_noise",
    "channel_noise",
    "other",
)
ICLABEL_DISPLAY_NAMES = {
    "brain": "brain",
    "muscle_artifact": "muscle artifact",
    "eye_blink": "eye blink",
    "heart_beat": "heart beat",
    "line_noise": "line noise",
    "channel_noise": "channel noise",
    "other": "other",
}
ICLABEL_ARTIFACT_CLASSES = ICLABEL_CLASSES[1:6]


def make_ica_copy(raw, config: dict[str, Any], no_downsampling: bool = False):
    copy = raw.copy()
    copy.filter(
        l_freq=float(config["fit_l_freq"]),
        h_freq=float(config["fit_h_freq"]),
        method="fir",
        phase="zero",
        picks="eeg",
        skip_by_annotation=("edge", "bad_acq_skip"),
        verbose="ERROR",
    )
    target = config.get("temporary_resample_sfreq")
    if not no_downsampling and target and float(target) < copy.info["sfreq"]:
        copy.resample(float(target), npad="auto", verbose="ERROR")
    return copy


def fit_ica(raw, config: dict[str, Any], no_downsampling: bool = False):
    raw_for_ica = make_ica_copy(raw, config, no_downsampling=no_downsampling)
    ica = mne.preprocessing.ICA(
        n_components=config["n_components"],
        method=str(config["method"]),
        fit_params={"extended": bool(config["extended"])},
        random_state=int(config["random_state"]),
        max_iter=int(config["max_iter"]),
    )
    ica.fit(
        raw_for_ica,
        picks="eeg",
        reject_by_annotation=True,
        verbose="ERROR",
    )
    return ica, raw_for_ica


def _add_iclabel_scores(
    scores: pd.DataFrame,
    probabilities: np.ndarray,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Add ICLabel probabilities and return artifact-to-brain ranked rows."""
    probabilities = np.asarray(probabilities, dtype=float)
    expected_shape = (len(scores), len(ICLABEL_CLASSES))
    if probabilities.shape != expected_shape:
        raise ValueError(
            f"ICLabel returned shape {probabilities.shape}; expected {expected_shape}"
        )
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("ICLabel returned non-finite probabilities")

    result = scores.copy()
    for index, label in enumerate(ICLABEL_CLASSES):
        result[f"iclabel_{label}_probability"] = probabilities[:, index]

    artifact_indices = [ICLABEL_CLASSES.index(label) for label in ICLABEL_ARTIFACT_CLASSES]
    artifact_probabilities = probabilities[:, artifact_indices]
    artifact_probability = artifact_probabilities.sum(axis=1)
    strongest_artifact_probability = artifact_probabilities.max(axis=1)
    predicted_indices = probabilities.argmax(axis=1)
    predicted_labels = [ICLABEL_CLASSES[index] for index in predicted_indices]

    result["iclabel_predicted_label"] = [
        ICLABEL_DISPLAY_NAMES[label] for label in predicted_labels
    ]
    result["iclabel_predicted_probability"] = probabilities[
        np.arange(len(result)), predicted_indices
    ]
    result["iclabel_artifact_probability"] = artifact_probability
    # Positive values favor a known artifact class, negative values favor
    # brain, and ICLabel's uncertain "other" class naturally remains near the
    # middle instead of being mislabeled as either clean or artifactual.
    result["iclabel_artifact_brain_contrast"] = (
        artifact_probability - result["iclabel_brain_probability"].to_numpy()
    )
    result["iclabel_strongest_artifact_probability"] = strongest_artifact_probability

    artifact_threshold = float(config["iclabel_artifact_probability_threshold"])
    class_threshold = float(config["iclabel_minimum_class_probability"])
    predicted_is_artifact = np.asarray(
        [label in ICLABEL_ARTIFACT_CLASSES for label in predicted_labels], dtype=bool
    )
    proposed = (
        predicted_is_artifact
        & (artifact_probability >= artifact_threshold)
        & (strongest_artifact_probability >= class_threshold)
    )
    result["proposed_exclusion"] = proposed
    result["automatic_removal"] = False

    # Stable sorting keeps the original component number as the final tie-break.
    result = result.sort_values(
        ["iclabel_artifact_brain_contrast", "iclabel_artifact_probability", "component"],
        ascending=[False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    result.insert(0, "artifact_rank", np.arange(1, len(result) + 1, dtype=int))
    return result


def _iclabel_probabilities(ica, raw, config: dict[str, Any]) -> np.ndarray:
    try:
        from mne_icalabel.iclabel import iclabel_label_components
    except ImportError as error:
        raise RuntimeError(
            "ICLabel is enabled but mne-icalabel is not installed. Install it in "
            "MNE_August2026 with: python -m pip install mne-icalabel onnxruntime"
        ) from error

    return iclabel_label_components(
        raw,
        ica,
        inplace=True,
        backend=str(config.get("iclabel_backend", "onnx")),
    )


def score_ica_components(ica, raw, config: dict[str, Any]) -> pd.DataFrame:
    sources = ica.get_sources(raw).get_data()
    sfreq = float(raw.info["sfreq"])
    frontal = [
        name for name in FRONTAL_CHANNELS if name in raw.ch_names and name not in raw.info["bads"]
    ]
    if frontal:
        frontal_signal = np.mean(raw.get_data(picks=frontal), axis=0)
        frontal_correlation = np.asarray(
            [abs(np.corrcoef(source, frontal_signal)[0, 1]) for source in sources]
        )
    else:
        frontal_correlation = np.full(sources.shape[0], np.nan)

    components = ica.get_components()
    component_names = list(ica.ch_names)
    frontal_indices = [component_names.index(name) for name in frontal if name in component_names]
    if frontal_indices:
        frontal_weight = np.median(np.abs(components[frontal_indices, :]), axis=0)
        overall_weight = np.median(np.abs(components), axis=0)
        frontal_weight_ratio = frontal_weight / np.maximum(overall_weight, np.finfo(float).tiny)
    else:
        frontal_weight_ratio = np.full(sources.shape[0], np.nan)

    frequencies, power = welch(
        sources,
        fs=sfreq,
        nperseg=min(sources.shape[1], int(round(4.0 * sfreq))),
        axis=1,
    )
    low = np.trapezoid(power[:, (frequencies >= 1) & (frequencies < 4)], frequencies[(frequencies >= 1) & (frequencies < 4)], axis=1)
    total = np.trapezoid(power[:, (frequencies >= 1) & (frequencies <= 40)], frequencies[(frequencies >= 1) & (frequencies <= 40)], axis=1)
    low_frequency_ratio = low / np.maximum(total, np.finfo(float).tiny)

    suggested = (
        (frontal_correlation >= float(config["suggestion_frontal_correlation"]))
        & (frontal_weight_ratio >= float(config["suggestion_frontal_weight_ratio"]))
    )
    scores = pd.DataFrame(
        {
            "component": np.arange(sources.shape[0], dtype=int),
            "label": [f"IC{index:03d}" for index in range(sources.shape[0])],
            "frontal_correlation": frontal_correlation,
            "frontal_weight_ratio": frontal_weight_ratio,
            "low_frequency_power_ratio": low_frequency_ratio,
            "suggested_ocular_review": suggested,
        }
    )
    if not bool(config.get("iclabel_enabled", True)):
        scores["proposed_exclusion"] = False
        scores["automatic_removal"] = False
        return scores
    probabilities = _iclabel_probabilities(ica, raw, config)
    return _add_iclabel_scores(scores, probabilities, config)


def proposed_ica_exclusions(scores: pd.DataFrame) -> tuple[list[int], dict[int, str]]:
    """Return machine-proposed exclusions and auditable probability reasons."""
    proposed = scores.loc[scores["proposed_exclusion"]].copy()
    components = proposed["component"].astype(int).tolist()
    reasons = {}
    for row in proposed.itertuples(index=False):
        component = int(row.component)
        reasons[component] = (
            f"ICLabel proposal: {row.iclabel_predicted_label} "
            f"p={row.iclabel_predicted_probability:.3f}; known-artifact total "
            f"p={row.iclabel_artifact_probability:.3f}. Visual confirmation required."
        )
    return components, reasons


def apply_ica_exclusions(raw, ica, components: list[int]):
    """Apply an explicit, already-selected ICA exclusion list to a copy."""
    invalid = [component for component in components if component < 0 or component >= ica.n_components_]
    if invalid:
        raise ValueError(f"Reviewed ICA indices outside 0..{ica.n_components_ - 1}: {invalid}")
    cleaned = raw.copy()
    ica.exclude = list(components)
    ica.apply(cleaned, exclude=components, verbose="ERROR")
    return cleaned
