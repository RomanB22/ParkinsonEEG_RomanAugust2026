"""Build subject-balanced MOCA and EEG feature tables with strict provenance."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


METRIC_LABELS = {
    "entropy": "Permutation entropy H",
    "complexity": "Statistical complexity C",
    "fisher_information": "Fisher information F",
    "oscillatory_occupancy": "Oscillatory occupancy",
    "bouts_per_minute": "Bouts per minute",
    "bout_duration_mean_s": "Mean bout duration",
    "bout_cycles_mean": "Mean cycles per bout",
    "bout_snr_mean": "Mean bout threshold ratio",
}

METRIC_UNITS = {
    "entropy": "normalized",
    "complexity": "normalized",
    "fisher_information": "normalized",
    "oscillatory_occupancy": "proportion",
    "bouts_per_minute": "bouts/minute",
    "bout_duration_mean_s": "seconds",
    "bout_cycles_mean": "cycles",
    "bout_snr_mean": "ratio",
}

BAND_LABELS = {
    "delta": "Delta",
    "theta": "Theta",
    "alpha": "Alpha",
    "beta": "Beta",
    "low_gamma": "Low gamma",
    "broad_5_15": "Broad 5–15 Hz",
    "low_beta": "Low beta",
    "high_beta": "High beta",
}


def _read_csv(path: str | Path, required: set[str]) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Required quantitative-behavioral input does not exist: {path}"
        )
    table = pd.read_csv(path)
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    return table


def _validate_subject_coverage(
    table: pd.DataFrame, expected_subjects: set[str], source_name: str
) -> None:
    observed = set(table["subject_id"].astype(str))
    missing = sorted(expected_subjects - observed)
    extra = sorted(observed - expected_subjects)
    if missing or extra:
        raise ValueError(
            f"{source_name} subject mismatch; missing={missing[:10]}, extra={extra[:10]}"
        )


def _feature_specification(
    *,
    feature_id: str,
    family: str,
    domain: str,
    band: str,
    metric: str,
    source_file: str,
    analysis_level: str,
) -> dict[str, Any]:
    band_label = BAND_LABELS.get(band, band.replace("_", " ").title())
    metric_label = METRIC_LABELS.get(metric, metric.replace("_", " ").title())
    if band == "broadband":
        label = f"Broadband {metric_label}"
    elif domain == "bout_ordinal":
        label = f"{band_label} within-bout {metric_label}"
    elif domain == "bout":
        label = f"{band_label} {metric_label}"
    else:
        label = f"{band_label} {metric_label}"
    return {
        "feature_id": feature_id,
        "family": family,
        "domain": domain,
        "band": band,
        "metric": metric,
        "feature_label": label,
        "unit": METRIC_UNITS[metric],
        "source_file": str(Path(source_file).resolve()),
        "analysis_level": analysis_level,
    }


def _append_subject_features(
    rows: list[pd.DataFrame],
    dictionary: list[dict[str, Any]],
    table: pd.DataFrame,
    *,
    source_file: str,
    family: str,
    domain: str,
    metrics: list[str],
    bands: list[str] | None,
) -> None:
    selected_bands = ["broadband"] if bands is None else bands
    for band in selected_bands:
        selected = table if bands is None else table.loc[table["band"].eq(band)]
        if selected["subject_id"].duplicated().any():
            raise ValueError(f"{source_file}: duplicate subject rows for {band}")
        if selected.empty:
            raise ValueError(f"{source_file}: requested band is unavailable: {band}")
        for metric in metrics:
            prefix = {
                "ordinal_broadband": "ordinal_broadband",
                "ordinal_band": "ordinal_band",
                "bout_properties": "bout",
                "bout_ordinal": "bout_ordinal",
            }[family]
            feature_id = (
                f"{prefix}_{metric}"
                if band == "broadband"
                else f"{prefix}_{band}_{metric}"
            )
            rows.append(
                selected[["subject_id", metric]]
                .rename(columns={metric: "value"})
                .assign(feature_id=feature_id)
            )
            dictionary.append(
                _feature_specification(
                    feature_id=feature_id,
                    family=family,
                    domain=domain,
                    band=band,
                    metric=metric,
                    source_file=source_file,
                    analysis_level="subject_mean_across_shared_electrodes",
                )
            )


def load_cohort(config: dict[str, Any]) -> pd.DataFrame:
    path = config["input"]["participants_file"]
    participants = _read_csv(
        path,
        {"participant_id", "GROUP", "AGE", "GENDER", "MOCA"},
    )
    if participants["participant_id"].duplicated().any():
        raise ValueError("Participant IDs must be unique")
    if not set(participants["GENDER"].astype(str)).issubset({"M", "F"}):
        raise ValueError("GENDER must contain only M and F")
    cohort = participants.rename(
        columns={
            "participant_id": "subject_id",
            "GROUP": "group",
            "AGE": "age_years",
            "GENDER": "gender",
            "MOCA": "moca",
            "UPDRS": "updrs",
        }
    ).copy()
    cohort["subject_id"] = cohort["subject_id"].astype(str)
    cohort["sex_male"] = cohort["gender"].astype(str).eq("M").astype(int)
    required_numeric = ["age_years", "moca", "sex_male"]
    if not np.all(np.isfinite(cohort[required_numeric].to_numpy(dtype=float))):
        raise ValueError("MOCA, age, and sex must be complete for the prespecified cohort")
    return cohort.sort_values("subject_id").reset_index(drop=True)


def build_subject_features(
    config: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return cohort, long subject features, and a feature dictionary."""
    cohort = load_cohort(config)
    expected_subjects = set(cohort["subject_id"])
    inputs = config["input"]
    requested = config["features"]
    ordinal_metrics = [str(value) for value in requested["ordinal_metrics"]]
    bout_properties = [str(value) for value in requested["bout_properties"]]
    bout_ordinal_metrics = [str(value) for value in requested["bout_ordinal_metrics"]]
    required_ordinal = {"subject_id", "group", "n_electrodes", *ordinal_metrics}

    ordinal_subject = _read_csv(inputs["ordinal_subject_file"], required_ordinal)
    ordinal_band = _read_csv(
        inputs["ordinal_band_subject_file"], required_ordinal | {"band"}
    )
    bout_subject = _read_csv(
        inputs["bout_subject_file"],
        {"subject_id", "group", "band", "n_electrodes", *bout_properties},
    )
    bout_ordinal_subject = _read_csv(
        inputs["bout_ordinal_subject_file"],
        {"subject_id", "group", "band", "n_electrodes", *bout_ordinal_metrics},
    )
    for name, table in (
        ("ordinal broadband", ordinal_subject),
        ("ordinal bands", ordinal_band),
        ("bout properties", bout_subject),
        ("within-bout ordinal", bout_ordinal_subject),
    ):
        _validate_subject_coverage(table, expected_subjects, name)
        if not set(table["group"].astype(str)).issubset(set(cohort["group"])):
            raise ValueError(f"{name}: invalid group labels")
        source_groups = table[["subject_id", "group"]].drop_duplicates()
        if source_groups["subject_id"].duplicated().any():
            raise ValueError(f"{name}: inconsistent group labels within subjects")
        expected_groups = cohort[["subject_id", "group"]]
        compared = expected_groups.merge(
            source_groups,
            on="subject_id",
            suffixes=("_metadata", "_source"),
            validate="one_to_one",
        )
        if not compared["group_metadata"].eq(compared["group_source"]).all():
            raise ValueError(f"{name}: group labels disagree with participant metadata")
        expected_electrodes = int(config["expected"]["shared_electrodes"])
        if not table["n_electrodes"].eq(expected_electrodes).all():
            raise ValueError(
                f"{name} must use exactly {expected_electrodes} shared electrodes"
            )

    feature_rows: list[pd.DataFrame] = []
    dictionary_rows: list[dict[str, Any]] = []
    _append_subject_features(
        feature_rows,
        dictionary_rows,
        ordinal_subject,
        source_file=inputs["ordinal_subject_file"],
        family="ordinal_broadband",
        domain="ordinal",
        metrics=ordinal_metrics,
        bands=None,
    )
    _append_subject_features(
        feature_rows,
        dictionary_rows,
        ordinal_band,
        source_file=inputs["ordinal_band_subject_file"],
        family="ordinal_band",
        domain="ordinal",
        metrics=ordinal_metrics,
        bands=[str(value) for value in requested["ordinal_bands"]],
    )
    _append_subject_features(
        feature_rows,
        dictionary_rows,
        bout_subject,
        source_file=inputs["bout_subject_file"],
        family="bout_properties",
        domain="bout",
        metrics=bout_properties,
        bands=[str(value) for value in requested["bout_bands"]],
    )
    _append_subject_features(
        feature_rows,
        dictionary_rows,
        bout_ordinal_subject,
        source_file=inputs["bout_ordinal_subject_file"],
        family="bout_ordinal",
        domain="bout_ordinal",
        metrics=bout_ordinal_metrics,
        bands=[str(value) for value in requested["bout_bands"]],
    )
    features = pd.concat(feature_rows, ignore_index=True)
    dictionary = pd.DataFrame.from_records(dictionary_rows)
    if dictionary["feature_id"].duplicated().any():
        raise RuntimeError("Feature identifiers must be unique")
    if features.duplicated(["subject_id", "feature_id"]).any():
        raise ValueError("Subject features contain duplicated subject/feature rows")
    features = features.merge(
        cohort[["subject_id", "group", "moca", "age_years", "gender", "sex_male"]],
        on="subject_id",
        how="left",
        validate="many_to_one",
    ).merge(dictionary, on="feature_id", how="left", validate="many_to_one")
    expected_rows = len(cohort) * len(dictionary)
    if len(features) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} subject-feature rows, found {len(features)}"
        )
    return cohort, features, dictionary


def _append_electrode_features(
    rows: list[pd.DataFrame],
    table: pd.DataFrame,
    *,
    family: str,
    metrics: list[str],
    bands: list[str] | None,
) -> None:
    selected_bands = ["broadband"] if bands is None else bands
    for band in selected_bands:
        selected = table if bands is None else table.loc[table["band"].eq(band)]
        for metric in metrics:
            prefix = {
                "ordinal_broadband": "ordinal_broadband",
                "ordinal_band": "ordinal_band",
                "bout_properties": "bout",
                "bout_ordinal": "bout_ordinal",
            }[family]
            feature_id = (
                f"{prefix}_{metric}"
                if band == "broadband"
                else f"{prefix}_{band}_{metric}"
            )
            rows.append(
                selected[["subject_id", "group", "electrode", metric]]
                .rename(columns={metric: "value"})
                .assign(feature_id=feature_id)
            )


def build_electrode_features(
    config: dict[str, Any], cohort: pd.DataFrame, dictionary: pd.DataFrame
) -> tuple[pd.DataFrame, list[str]]:
    """Return secondary electrode-level features for spatial correlations."""
    inputs = config["input"]
    requested = config["features"]
    ordinal_metrics = [str(value) for value in requested["ordinal_metrics"]]
    bout_properties = [str(value) for value in requested["bout_properties"]]
    bout_ordinal_metrics = [str(value) for value in requested["bout_ordinal_metrics"]]
    ordinal_electrode = _read_csv(
        inputs["ordinal_electrode_file"],
        {"subject_id", "group", "electrode", *ordinal_metrics},
    )
    ordinal_band_electrode = _read_csv(
        inputs["ordinal_band_electrode_file"],
        {"subject_id", "group", "electrode", "band", *ordinal_metrics},
    )
    bout_electrode = _read_csv(
        inputs["bout_electrode_file"],
        {"subject_id", "group", "electrode", "band", *bout_properties},
    )
    bout_ordinal_electrode = _read_csv(
        inputs["bout_ordinal_electrode_file"],
        {"subject_id", "group", "electrode", "band", *bout_ordinal_metrics},
    )
    expected_subjects = set(cohort["subject_id"])
    for name, table in (
        ("ordinal broadband electrodes", ordinal_electrode),
        ("ordinal band electrodes", ordinal_band_electrode),
        ("bout-property electrodes", bout_electrode),
        ("within-bout ordinal electrodes", bout_ordinal_electrode),
    ):
        _validate_subject_coverage(table, expected_subjects, name)
        if table.duplicated(
            ["subject_id", "electrode"] + (["band"] if "band" in table else [])
        ).any():
            raise ValueError(f"{name} contains duplicate rows")

    electrode_set_path = Path(inputs["ordinal_electrode_sets_file"])
    if not electrode_set_path.exists():
        raise FileNotFoundError(f"Missing electrode set: {electrode_set_path}")
    electrode_payload = json.loads(electrode_set_path.read_text(encoding="utf-8"))
    electrode_order = [str(value) for value in electrode_payload["common_electrodes"]]
    expected_count = int(config["expected"]["shared_electrodes"])
    if len(electrode_order) != expected_count:
        raise ValueError(f"Expected {expected_count} shared electrodes")
    expected_electrodes = set(electrode_order)
    for name, table in (
        ("ordinal broadband electrodes", ordinal_electrode),
        ("ordinal band electrodes", ordinal_band_electrode),
        ("bout-property electrodes", bout_electrode),
        ("within-bout ordinal electrodes", bout_ordinal_electrode),
    ):
        if set(table["electrode"].astype(str)) != expected_electrodes:
            raise ValueError(f"{name} does not use the prespecified shared-electrode set")

    rows: list[pd.DataFrame] = []
    _append_electrode_features(
        rows,
        ordinal_electrode,
        family="ordinal_broadband",
        metrics=ordinal_metrics,
        bands=None,
    )
    _append_electrode_features(
        rows,
        ordinal_band_electrode,
        family="ordinal_band",
        metrics=ordinal_metrics,
        bands=[str(value) for value in requested["ordinal_bands"]],
    )
    _append_electrode_features(
        rows,
        bout_electrode,
        family="bout_properties",
        metrics=bout_properties,
        bands=[str(value) for value in requested["bout_bands"]],
    )
    _append_electrode_features(
        rows,
        bout_ordinal_electrode,
        family="bout_ordinal",
        metrics=bout_ordinal_metrics,
        bands=[str(value) for value in requested["bout_bands"]],
    )
    features = pd.concat(rows, ignore_index=True)
    if features.duplicated(["subject_id", "electrode", "feature_id"]).any():
        raise ValueError("Electrode features contain duplicated rows")
    features = features.merge(
        cohort[["subject_id", "group", "moca", "age_years", "sex_male"]],
        on=["subject_id", "group"],
        how="left",
        validate="many_to_one",
    ).merge(dictionary, on="feature_id", how="left", validate="many_to_one")
    if features["family"].isna().any():
        raise RuntimeError("Electrode features are missing dictionary records")
    return features, electrode_order


def subject_feature_matrix(
    cohort: pd.DataFrame, subject_features: pd.DataFrame
) -> pd.DataFrame:
    """Create a documented one-row-per-subject wide audit table."""
    wide = subject_features.pivot(index="subject_id", columns="feature_id", values="value")
    wide.columns.name = None
    return cohort.merge(
        wide.reset_index(), on="subject_id", how="left", validate="one_to_one"
    )
