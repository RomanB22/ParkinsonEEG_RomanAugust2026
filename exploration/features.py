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

    ordinal_global = _read_csv(
        inputs["ordinal_global_file"],
        {"subject_id", "group", "entropy", "complexity", "fisher_information", "n_electrodes"},
    )
    _validate_subjects("ordinal global", ordinal_global, subject_ids)
    if ordinal_global["subject_id"].duplicated().any():
        raise ValueError("Ordinal global table must contain one row per subject")
    ordinal_global = ordinal_global.rename(
        columns={
            "entropy": "ordinal_global_entropy",
            "complexity": "ordinal_global_complexity",
            "fisher_information": "ordinal_global_fisher_information",
            "n_electrodes": "ordinal_n_electrodes",
            "group": "ordinal_group",
        }
    )

    ordinal_band = _read_csv(
        inputs["ordinal_band_file"],
        {"subject_id", "group", "band", "entropy", "complexity", "fisher_information"},
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
        values=["entropy", "complexity", "fisher_information"],
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
    if not table["group"].eq(table["ordinal_group"]).all():
        raise ValueError("Participant and ordinal group labels disagree")
    table = table.drop(columns=["ordinal_group"])
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
    """Return completed ordinal sweep global tables with parsed D and tau."""
    root = Path(config["input"]["ordinal_sweep_root"])
    completed = []
    if not root.exists():
        return completed
    for directory in sorted(path for path in root.iterdir() if path.is_dir()):
        match = SWEEP_PATTERN.fullmatch(directory.name)
        metrics_path = directory / "metrics" / "subject_electrode_mean_metrics.csv"
        if match is None or not metrics_path.exists():
            continue
        completed.append(
            {
                "embedding_dimension": int(match.group("dimension")),
                "delay_samples": int(match.group("delay")),
                "path": str(metrics_path),
            }
        )
    return completed
