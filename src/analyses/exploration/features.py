"""Build a transparent one-row-per-subject modeling table."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


FORBIDDEN_MODEL_COLUMNS = {
    "participant_id",
    "subject_id",
    "group",
    "target_pd",
    "ID",
    "EEG",
    "TYPE",
    "UPDRS",
    "updrs",
}

LEAKAGE_EXCLUSIONS = {
    "participant_id": "Join key only; never a predictor.",
    "ID": "Administrative identifier; may encode enrollment structure.",
    "EEG": "Recording identifier; may encode diagnosis or acquisition order.",
    "TYPE": "Perfect copy of the PD/Control outcome in this cohort.",
    "UPDRS": "Unavailable for every Control and therefore reveals diagnosis by missingness.",
    "GROUP": "Classification outcome, not a predictor.",
}

SWEEP_PATTERN = re.compile(r"D(?P<dimension>\d+)_tau(?P<delay>\d+)$")


def summarize_typical_bout_shapes(
    path: str | Path,
    requested_bands: list[str],
) -> pd.DataFrame:
    """Reduce each subject's bout curves to four transparent scalars per band."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Typical-bout input does not exist: {path}")
    with np.load(path) as payload:
        required = {
            "subject_ids",
            "bands",
            "times_seconds",
            "normalized_amplitude_envelopes",
            "relative_phase_phasors",
            "bout_counts",
        }
        missing = sorted(required - set(payload.files))
        if missing:
            raise ValueError(f"{path} is missing typical-bout arrays: {missing}")
        subject_ids = payload["subject_ids"].astype(str)
        available_bands = payload["bands"].astype(str).tolist()
        times = payload["times_seconds"].astype(float)
        envelopes = payload["normalized_amplitude_envelopes"].astype(float)
        phasors = payload["relative_phase_phasors"].astype(np.complex128)
        counts = payload["bout_counts"].astype(int)
    missing_bands = sorted(set(requested_bands) - set(available_bands))
    if missing_bands:
        raise ValueError(f"Typical-bout input is missing bands: {missing_bands}")
    if envelopes.shape != phasors.shape or envelopes.shape[:3] != counts.shape:
        raise ValueError("Typical-bout arrays have inconsistent dimensions")
    if envelopes.shape[0] != len(subject_ids) or envelopes.shape[-1] != len(times):
        raise ValueError("Typical-bout subject or time axes are inconsistent")
    if len(times) < 3 or not np.all(np.diff(times) > 0.0):
        raise ValueError("Typical-bout time axis must be strictly increasing")

    central_phase_window = np.abs(times) <= 0.25
    peak_search_window = np.abs(times) <= 0.25
    sample_period = float(np.median(np.diff(times)))
    rows: list[dict[str, Any]] = []
    for subject_index, subject_id in enumerate(subject_ids):
        row: dict[str, Any] = {"subject_id": subject_id}
        for band in requested_bands:
            band_index = available_bands.index(band)
            valid_electrodes = counts[subject_index, :, band_index] > 0
            valid_electrodes &= np.all(
                np.isfinite(envelopes[subject_index, :, band_index]), axis=1
            )
            valid_electrodes &= np.all(
                np.isfinite(phasors[subject_index, :, band_index]), axis=1
            )
            if not valid_electrodes.any():
                raise ValueError(f"{subject_id}/{band} has no usable typical-bout curve")
            envelope = np.mean(
                envelopes[subject_index, valid_electrodes, band_index], axis=0
            )
            candidate_indices = np.flatnonzero(peak_search_window)
            peak_index = int(
                candidate_indices[np.argmax(envelope[peak_search_window])]
            )
            peak = float(envelope[peak_index])
            half_height = 1.0 + 0.5 * (peak - 1.0)
            left = peak_index
            right = peak_index
            while left > 0 and envelope[left - 1] >= half_height:
                left -= 1
            while right + 1 < len(envelope) and envelope[right + 1] >= half_height:
                right += 1
            width = float((right - left + 1) * sample_period)
            excess = np.clip(envelope - 1.0, 0.0, None)
            pre = float(np.trapezoid(excess[times < 0.0], times[times < 0.0]))
            post = float(np.trapezoid(excess[times > 0.0], times[times > 0.0]))
            asymmetry = (post - pre) / (post + pre) if post + pre > 0.0 else 0.0
            phase_consistency = float(
                np.mean(
                    np.abs(
                        phasors[
                            subject_index,
                            valid_electrodes,
                            band_index,
                        ][:, central_phase_window]
                    )
                )
            )
            prefix = f"typical_{band}"
            row[f"{prefix}_envelope_peak_ratio"] = peak
            row[f"{prefix}_envelope_half_height_width_s"] = width
            row[f"{prefix}_envelope_asymmetry"] = float(asymmetry)
            row[f"{prefix}_relative_phase_consistency"] = phase_consistency
        rows.append(row)
    table = pd.DataFrame.from_records(rows)
    if table["subject_id"].duplicated().any():
        raise ValueError("Typical-bout input contains duplicate subjects")
    return table


def _read_csv(path: str | Path, required: set[str]) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Required modeling input does not exist: {path}")
    table = pd.read_csv(path)
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    return table


def _validate_subjects(
    name: str,
    table: pd.DataFrame,
    expected_subjects: set[str],
) -> None:
    observed = set(table["subject_id"].astype(str))
    missing = sorted(expected_subjects - observed)
    extra = sorted(observed - expected_subjects)
    if missing or extra:
        raise ValueError(
            f"{name} subject mismatch; missing={missing[:10]}, extra={extra[:10]}"
        )


def validate_model_features(
    feature_table: pd.DataFrame,
    models: dict[str, dict[str, Any]],
) -> None:
    """Reject unavailable, duplicated, or diagnosis-revealing predictors."""
    for model_name, specification in models.items():
        features = [str(value) for value in specification["features"]]
        if not features:
            raise ValueError(f"Model {model_name} has no features")
        if len(features) != len(set(features)):
            raise ValueError(f"Model {model_name} contains duplicate features")
        forbidden = sorted(set(features) & FORBIDDEN_MODEL_COLUMNS)
        if forbidden:
            raise ValueError(f"Model {model_name} contains forbidden features: {forbidden}")
        missing = sorted(set(features) - set(feature_table.columns))
        if missing:
            raise ValueError(f"Model {model_name} is missing features: {missing}")
        values = feature_table[features].to_numpy(dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Model {model_name} contains non-finite predictor values")


def build_feature_table(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Merge metadata, primary ordinal metrics, and PSD into one subject table."""
    inputs = config["input"]
    participants = _read_csv(
        inputs["participants_file"],
        {"participant_id", "GROUP", "AGE", "GENDER", "MOCA", "UPDRS", "TYPE", "ID", "EEG"},
    )
    if participants["participant_id"].duplicated().any():
        raise ValueError("Participant IDs must be unique")
    groups = set(participants["GROUP"].astype(str))
    if groups != {"PD", "Control"}:
        raise ValueError(f"Expected exactly PD and Control groups, found {sorted(groups)}")
    subject_ids = set(participants["participant_id"].astype(str))

    candidate_settings = config["candidate_features"]
    ordinal_core_metrics = [
        "entropy", "weighted_permutation_entropy", "complexity", "fisher_information"
    ]
    renyi_metrics = [str(value) for value in candidate_settings["renyi_metrics"]]
    ordinal_metrics = [*ordinal_core_metrics, *renyi_metrics]
    ordinal_global = _read_csv(
        inputs["ordinal_global_file"],
        {"subject_id", "group", "n_electrodes", *ordinal_metrics},
    )
    _validate_subjects("ordinal global", ordinal_global, subject_ids)
    if ordinal_global["subject_id"].duplicated().any():
        raise ValueError("Ordinal global table must contain one row per subject")
    ordinal_global = ordinal_global.rename(
        columns={
            **{metric: f"ordinal_global_{metric}" for metric in ordinal_metrics},
            "n_electrodes": "ordinal_n_electrodes",
            "group": "ordinal_group",
        }
    )

    ordinal_band = _read_csv(
        inputs["ordinal_band_file"],
        {"subject_id", "group", "band", *ordinal_metrics},
    )
    _validate_subjects("ordinal band", ordinal_band, subject_ids)
    requested_bands = [str(value) for value in config["ordinal_model_bands"]]
    selected_ordinal = ordinal_band.loc[ordinal_band["band"].isin(requested_bands)].copy()
    found_bands = set(selected_ordinal["band"])
    if found_bands != set(requested_bands):
        raise ValueError(
            f"Ordinal bands missing from input: {sorted(set(requested_bands) - found_bands)}"
        )
    ordinal_wide = selected_ordinal.pivot(
        index="subject_id",
        columns="band",
        values=ordinal_metrics,
    )
    ordinal_wide.columns = [
        f"ordinal_{band}_{metric}" for metric, band in ordinal_wide.columns
    ]
    ordinal_wide = ordinal_wide.reset_index()

    psd = _read_csv(
        inputs["psd_subject_band_file"],
        {"subject_id", "group", "band", "n_electrodes", "median_relative_band_power"},
    )
    _validate_subjects("PSD subject band", psd, subject_ids)
    psd_settings = config["psd_log_ratio"]
    numerator_bands = [str(value) for value in psd_settings["numerator_bands"]]
    reference_band = str(psd_settings["reference_band"])
    required_bands = set(numerator_bands) | {reference_band}
    if not required_bands.issubset(set(psd["band"])):
        raise ValueError(f"PSD bands missing from input: {sorted(required_bands - set(psd['band']))}")
    psd_wide = psd.pivot(
        index="subject_id", columns="band", values="median_relative_band_power"
    )
    if (psd_wide[list(required_bands)] <= 0.0).any().any():
        raise ValueError("PSD log ratios require strictly positive relative powers")
    log_base = float(psd_settings["log_base"])
    if not np.isclose(log_base, 2.0):
        raise ValueError("This pipeline requires base-2 PSD log ratios")
    log_denominator = np.log(log_base)
    psd_features = pd.DataFrame({"subject_id": psd_wide.index})
    for band in numerator_bands:
        psd_features[f"psd_log2_{band}_vs_{reference_band}"] = (
            np.log(psd_wide[band].to_numpy() / psd_wide[reference_band].to_numpy())
            / log_denominator
        )
    electrode_counts = psd.groupby("subject_id")["n_electrodes"].min()
    psd_features["psd_n_electrodes"] = electrode_counts.reindex(psd_wide.index).to_numpy()

    aperiodic = _read_csv(
        inputs["aperiodic_subject_file"],
        {"subject_id", "group", "aperiodic_exponent"},
    )
    _validate_subjects("aperiodic", aperiodic, subject_ids)
    if aperiodic["subject_id"].duplicated().any():
        raise ValueError("Aperiodic table must contain one row per subject")
    aperiodic = aperiodic.rename(columns={"group": "aperiodic_group"})[
        ["subject_id", "aperiodic_group", "aperiodic_exponent"]
    ]

    bout_bands = [str(value) for value in candidate_settings["bout_bands"]]
    bout_metrics = [str(value) for value in candidate_settings["bout_metrics"]]
    bouts = _read_csv(
        inputs["bout_subject_file"],
        {"subject_id", "group", "band", *bout_metrics},
    )
    _validate_subjects("bout properties", bouts, subject_ids)
    selected_bouts = bouts.loc[bouts["band"].isin(bout_bands)]
    if set(selected_bouts["band"]) != set(bout_bands):
        raise ValueError("Bout-property input is missing a requested band")
    bout_wide = selected_bouts.pivot(
        index="subject_id", columns="band", values=bout_metrics
    )
    bout_wide.columns = [f"bout_{band}_{metric}" for metric, band in bout_wide.columns]
    bout_wide = bout_wide.reset_index()

    bout_ordinal_metrics = [
        str(value) for value in candidate_settings["bout_ordinal_metrics"]
    ]
    bout_ordinal = _read_csv(
        inputs["bout_ordinal_subject_file"],
        {"subject_id", "group", "band", *bout_ordinal_metrics},
    )
    _validate_subjects("within-bout ordinal", bout_ordinal, subject_ids)
    selected_bout_ordinal = bout_ordinal.loc[
        bout_ordinal["band"].isin(bout_bands)
    ]
    if set(selected_bout_ordinal["band"]) != set(bout_bands):
        raise ValueError("Within-bout ordinal input is missing a requested band")
    bout_ordinal_wide = selected_bout_ordinal.pivot(
        index="subject_id", columns="band", values=bout_ordinal_metrics
    )
    bout_ordinal_wide.columns = [
        f"bout_ordinal_{band}_{metric}"
        for metric, band in bout_ordinal_wide.columns
    ]
    bout_ordinal_wide = bout_ordinal_wide.reset_index()

    typical_bouts = summarize_typical_bout_shapes(
        inputs["typical_bout_file"], bout_bands
    )
    _validate_subjects("typical-bout shapes", typical_bouts, subject_ids)

    table = participants.rename(
        columns={
            "participant_id": "subject_id",
            "GROUP": "group",
            "AGE": "age_years",
            "GENDER": "gender",
            "MOCA": "moca",
        }
    )
    table["subject_id"] = table["subject_id"].astype(str)
    if not set(table["gender"].astype(str)).issubset({"M", "F"}):
        raise ValueError("GENDER must contain only M and F")
    table["sex_male"] = table["gender"].astype(str).eq("M").astype(int)
    table["target_pd"] = table["group"].eq("PD").astype(int)
    table = table.merge(
        ordinal_global,
        on="subject_id",
        how="left",
        validate="one_to_one",
    )
    table = table.merge(ordinal_wide, on="subject_id", how="left", validate="one_to_one")
    table = table.merge(psd_features, on="subject_id", how="left", validate="one_to_one")
    table = table.merge(aperiodic, on="subject_id", how="left", validate="one_to_one")
    table = table.merge(bout_wide, on="subject_id", how="left", validate="one_to_one")
    table = table.merge(
        bout_ordinal_wide, on="subject_id", how="left", validate="one_to_one"
    )
    table = table.merge(
        typical_bouts, on="subject_id", how="left", validate="one_to_one"
    )
    if not table["group"].eq(table["ordinal_group"]).all():
        raise ValueError("Participant and ordinal group labels disagree")
    if not table["group"].eq(table["aperiodic_group"]).all():
        raise ValueError("Participant and aperiodic group labels disagree")
    table = table.drop(columns=["ordinal_group", "aperiodic_group"])
    table = table.drop(columns=["ID", "EEG", "TYPE", "UPDRS"])
    table = table.sort_values("subject_id").reset_index(drop=True)
    validate_model_features(table, config["models"])

    provenance_rows = []
    for model_name, specification in config["models"].items():
        for feature in specification["features"]:
            if feature.startswith("ordinal_"):
                source = "ordinal analysis"
            elif feature.startswith("psd_"):
                source = "PSD analysis"
            elif feature.startswith("aperiodic_"):
                source = "scale-free specparam analysis"
            elif feature.startswith("bout_ordinal_"):
                source = "within-bout ordinal analysis"
            elif feature.startswith("bout_"):
                source = "scale-free bout analysis"
            elif feature.startswith("typical_"):
                source = "subject-balanced typical-bout analysis"
            else:
                source = "participant metadata"
            provenance_rows.append(
                {
                    "model": model_name,
                    "model_label": specification["label"],
                    "model_role": specification["role"],
                    "feature": feature,
                    "source": source,
                    "included": True,
                    "exclusion_reason": "",
                }
            )
    for feature, reason in LEAKAGE_EXCLUSIONS.items():
        provenance_rows.append(
            {
                "model": "all",
                "model_label": "All models",
                "model_role": "excluded",
                "feature": feature,
                "source": "participant metadata",
                "included": False,
                "exclusion_reason": reason,
            }
        )
    return table, pd.DataFrame.from_records(provenance_rows)


def discover_completed_sweeps(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Return configured ordinal sweep tables, ignoring legacy settings."""
    root = Path(config["input"]["ordinal_sweep_root"])
    expected_dimensions = {
        int(value) for value in config["ordinal_sweep"]["expected_dimensions"]
    }
    expected_delays = {
        int(value) for value in config["ordinal_sweep"]["expected_delays"]
    }
    completed = []
    if not root.exists():
        return completed
    for directory in sorted(path for path in root.iterdir() if path.is_dir()):
        match = SWEEP_PATTERN.fullmatch(directory.name)
        metrics_path = directory / "metrics" / "subject_electrode_mean_metrics.csv"
        if match is None or not metrics_path.exists():
            continue
        dimension = int(match.group("dimension"))
        delay = int(match.group("delay"))
        if dimension not in expected_dimensions or delay not in expected_delays:
            continue
        completed.append(
            {
                "embedding_dimension": dimension,
                "delay_samples": delay,
                "path": str(metrics_path),
            }
        )
    return completed
