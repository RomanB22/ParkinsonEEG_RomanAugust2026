"""Build one value per PD subject from all cohort-shared electrodes."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from quantitative_behavioral.features import BAND_LABELS, METRIC_LABELS, METRIC_UNITS


def resolve_shared_electrodes(config: dict[str, Any]) -> list[str]:
    """Read the canonical cohort-shared electrode set from ordinal provenance."""
    scope = config["electrode_scope"]
    if scope.get("policy") != "all_cohort_shared_electrodes":
        raise ValueError("Disease progression must use all cohort-shared electrodes")
    source = Path(config["input"]["ordinal_electrode_sets_file"])
    if not source.exists():
        raise FileNotFoundError(f"Missing disease-progression electrode set: {source}")
    payload = json.loads(source.read_text(encoding="utf-8"))
    electrodes = [str(value) for value in payload.get("common_electrodes", [])]
    if not electrodes or len(electrodes) != len(set(electrodes)):
        raise ValueError("common_electrodes must be a non-empty unique list")
    expected = int(scope["expected_count"])
    if len(electrodes) != expected:
        raise ValueError(
            f"Expected {expected} cohort-shared electrodes, found {len(electrodes)}"
        )
    return electrodes


def _read_csv(path: str | Path, required: set[str]) -> pd.DataFrame:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Missing disease-progression input: {source}")
    table = pd.read_csv(source)
    missing = sorted(required - set(table))
    if missing:
        raise ValueError(f"{source} is missing columns: {missing}")
    return table


def load_pd_cohort(config: dict[str, Any]) -> pd.DataFrame:
    participants = _read_csv(
        config["input"]["participants_file"],
        {"participant_id", "GROUP", "AGE", "GENDER", "MOCA", "UPDRS"},
    )
    if participants["participant_id"].duplicated().any():
        raise ValueError("Participant IDs must be unique")
    cohort = participants.loc[
        participants["GROUP"].astype(str).eq(str(config["analysis"]["group"]))
    ].rename(
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
    if not set(cohort["gender"].astype(str)).issubset({"M", "F"}):
        raise ValueError("GENDER must contain only M and F")
    cohort["sex_male"] = cohort["gender"].astype(str).eq("M").astype(int)
    numeric = ["age_years", "moca", "updrs", "sex_male"]
    for column in numeric:
        cohort[column] = pd.to_numeric(cohort[column], errors="raise")
    if not np.all(np.isfinite(cohort[numeric].to_numpy(float))):
        raise ValueError("PD age, sex, MOCA, and UPDRS must be complete")
    if len(cohort) < int(config["analysis"]["minimum_subjects"]):
        raise ValueError("PD cohort is smaller than analysis.minimum_subjects")
    return cohort.sort_values("subject_id").reset_index(drop=True)


def _select_electrodes(
    table: pd.DataFrame,
    cohort: pd.DataFrame,
    electrodes: Sequence[str],
    *,
    source_name: str,
    bands: Sequence[str] | None = None,
) -> pd.DataFrame:
    expected_subjects = set(cohort["subject_id"])
    selected = table.loc[table["subject_id"].astype(str).isin(expected_subjects)].copy()
    selected["subject_id"] = selected["subject_id"].astype(str)
    if bands is not None:
        selected = selected.loc[selected["band"].astype(str).isin(bands)].copy()
    selected = selected.loc[selected["electrode"].astype(str).isin(electrodes)].copy()
    keys = ["subject_id", *(["band"] if bands is not None else [])]
    expected_rows = len(electrodes)
    counts = selected.groupby(keys, sort=False)["electrode"].nunique()
    expected_keys = len(expected_subjects) * (len(bands) if bands is not None else 1)
    if len(counts) != expected_keys or not counts.eq(expected_rows).all():
        raise ValueError(
            f"{source_name} must contain all {expected_rows} selected electrodes "
            "for every PD subject and requested band"
        )
    duplicate_keys = [
        "subject_id",
        "electrode",
        *(["band"] if bands is not None else []),
    ]
    if selected.duplicated(duplicate_keys).any():
        raise ValueError(f"{source_name} contains duplicate subject/electrode rows")
    if not selected["group"].astype(str).eq(str(cohort["group"].iloc[0])).all():
        raise ValueError(f"{source_name} group labels disagree with participant metadata")
    return selected


def _metric_label(metric: str) -> str:
    if metric == "relative_band_power":
        return "Relative band power"
    if metric == "aperiodic_exponent_qc":
        return "QC-qualified aperiodic exponent"
    return METRIC_LABELS.get(metric, metric.replace("_", " ").title())


def _metric_unit(metric: str) -> str:
    if metric == "relative_band_power":
        return "proportion"
    return METRIC_UNITS.get(metric, "dimensionless")


def _append_features(
    feature_tables: list[pd.DataFrame],
    dictionary_rows: list[dict[str, Any]],
    selected: pd.DataFrame,
    *,
    metrics: Sequence[str],
    family: str,
    domain: str,
    source_file: str,
    aggregation: str,
    bands: Sequence[str] | None,
) -> None:
    requested_bands = [None] if bands is None else list(bands)
    for band in requested_bands:
        band_table = selected if band is None else selected.loc[selected["band"].eq(band)]
        for metric in metrics:
            keys = ["subject_id", "group"]
            grouped = band_table.groupby(keys, sort=False)[metric]
            values = (grouped.mean() if aggregation == "mean" else grouped.median()).rename(
                "value"
            )
            counts = grouped.count().rename("n_electrodes_contributing")
            summary = pd.concat([values, counts], axis=1).reset_index()
            band_token = "broadband" if band is None else str(band)
            feature_id = f"{family}_{band_token}_{metric}"
            summary["feature_id"] = feature_id
            feature_tables.append(summary)
            band_label = "Broadband" if band is None else BAND_LABELS.get(
                str(band), str(band).replace("_", " ").title()
            )
            context = "within-bout " if family == "bout_ordinal" else ""
            dictionary_rows.append(
                {
                    "feature_id": feature_id,
                    "feature_label": f"{band_label} {context}{_metric_label(metric)}",
                    "family": family,
                    "domain": domain,
                    "band": band_token,
                    "metric": metric,
                    "unit": _metric_unit(metric),
                    "aggregation": f"{aggregation} across cohort-shared electrodes",
                    "source_file": str(Path(source_file).resolve()),
                }
            )


def build_shared_electrode_features(
    config: dict[str, Any], cohort: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Return long subject features and their complete provenance dictionary."""
    inputs = config["input"]
    requested = config["features"]
    electrodes = resolve_shared_electrodes(config)
    ordinal_metrics = [str(value) for value in requested["ordinal_metrics"]]
    ordinal_bands = [str(value) for value in requested["ordinal_bands"]]
    psd_bands = [str(value) for value in requested["psd_bands"]]
    bout_bands = [str(value) for value in requested["bout_bands"]]
    bout_properties = [str(value) for value in requested["bout_properties"]]
    bout_ordinal_metrics = [str(value) for value in requested["bout_ordinal_metrics"]]

    ordinal = _select_electrodes(
        _read_csv(
            inputs["ordinal_electrode_file"],
            {
                "subject_id", "group", "electrode", "embedding_dimension",
                "delay_samples", *ordinal_metrics,
            },
        ), cohort, electrodes, source_name="ordinal broadband"
    )
    ordinal_band = _select_electrodes(
        _read_csv(
            inputs["ordinal_band_electrode_file"],
            {
                "subject_id", "group", "electrode", "band",
                "embedding_dimension", "delay_samples", *ordinal_metrics,
            },
        ), cohort, electrodes, source_name="ordinal bands", bands=ordinal_bands
    )
    expected_dimension = int(requested["embedding_dimension"])
    expected_delay = int(requested["delay_samples"])
    for name, table in (("ordinal broadband", ordinal), ("ordinal bands", ordinal_band)):
        if not table["embedding_dimension"].eq(expected_dimension).all():
            raise ValueError(f"{name} does not use D={expected_dimension}")
        if not table["delay_samples"].eq(expected_delay).all():
            raise ValueError(f"{name} does not use tau={expected_delay}")
    psd_metric = str(requested["psd_metric"])
    psd = _select_electrodes(
        _read_csv(
            inputs["psd_electrode_file"],
            {"subject_id", "group", "electrode", "band", psd_metric},
        ), cohort, electrodes, source_name="PSD bands", bands=psd_bands
    )
    aperiodic_source = _select_electrodes(
        _read_csv(
            inputs["aperiodic_electrode_file"],
            {
                "subject_id", "group", "electrode", "aperiodic_exponent",
                "specparam_fit_qc_pass",
            },
        ), cohort, electrodes, source_name="aperiodic exponent"
    )
    bout = _select_electrodes(
        _read_csv(
            inputs["bout_electrode_file"],
            {"subject_id", "group", "electrode", "band", *bout_properties},
        ), cohort, electrodes, source_name="bout properties", bands=bout_bands
    )
    bout_ordinal = _select_electrodes(
        _read_csv(
            inputs["bout_ordinal_electrode_file"],
            {"subject_id", "group", "electrode", "band", *bout_ordinal_metrics},
        ), cohort, electrodes, source_name="within-bout ordinal", bands=bout_bands
    )

    feature_tables: list[pd.DataFrame] = []
    dictionary_rows: list[dict[str, Any]] = []
    _append_features(
        feature_tables, dictionary_rows, ordinal, metrics=ordinal_metrics,
        family="ordinal", domain="ordinal_broadband",
        source_file=inputs["ordinal_electrode_file"], aggregation="mean", bands=None,
    )
    _append_features(
        feature_tables, dictionary_rows, ordinal_band, metrics=ordinal_metrics,
        family="ordinal", domain="ordinal_band",
        source_file=inputs["ordinal_band_electrode_file"], aggregation="mean",
        bands=ordinal_bands,
    )
    _append_features(
        feature_tables, dictionary_rows, psd, metrics=[psd_metric],
        family="psd", domain="psd_relative_power",
        source_file=inputs["psd_electrode_file"], aggregation="median", bands=psd_bands,
    )
    _append_features(
        feature_tables, dictionary_rows, aperiodic_source,
        metrics=["aperiodic_exponent"], family="aperiodic", domain="aperiodic",
        source_file=inputs["aperiodic_electrode_file"], aggregation="mean", bands=None,
    )
    qc_values = aperiodic_source["specparam_fit_qc_pass"]
    qc_mask = (
        qc_values
        if pd.api.types.is_bool_dtype(qc_values)
        else qc_values.astype(str).str.lower().eq("true")
    )
    qc = aperiodic_source.loc[qc_mask].copy()
    qc_summary = (
        qc.groupby(["subject_id", "group"], sort=False)["aperiodic_exponent"]
        .agg(value="mean", n_electrodes_contributing="count")
        .reset_index()
    )
    minimum_qc = math.ceil(
        float(requested["minimum_aperiodic_qc_fraction"]) * len(electrodes)
    )
    qc_summary.loc[qc_summary["n_electrodes_contributing"].lt(minimum_qc), "value"] = np.nan
    missing_subjects = set(cohort["subject_id"]) - set(qc_summary["subject_id"])
    if missing_subjects:
        additions = cohort.loc[cohort["subject_id"].isin(missing_subjects), ["subject_id", "group"]].copy()
        additions["value"] = np.nan
        additions["n_electrodes_contributing"] = 0
        qc_summary = pd.concat([qc_summary, additions], ignore_index=True)
    qc_feature_id = "aperiodic_broadband_aperiodic_exponent_qc"
    qc_summary["feature_id"] = qc_feature_id
    feature_tables.append(qc_summary)
    dictionary_rows.append(
        {
            "feature_id": qc_feature_id,
            "feature_label": "Broadband QC-qualified aperiodic exponent",
            "family": "aperiodic",
            "domain": "aperiodic",
            "band": "broadband",
            "metric": "aperiodic_exponent_qc",
            "unit": "dimensionless",
            "aggregation": (
                f"mean across at least {minimum_qc} of {len(electrodes)} "
                "QC-passing cohort-shared electrodes"
            ),
            "source_file": str(Path(inputs["aperiodic_electrode_file"]).resolve()),
        }
    )
    _append_features(
        feature_tables, dictionary_rows, bout, metrics=bout_properties,
        family="bout", domain="bout_properties",
        source_file=inputs["bout_electrode_file"], aggregation="mean", bands=bout_bands,
    )
    _append_features(
        feature_tables, dictionary_rows, bout_ordinal, metrics=bout_ordinal_metrics,
        family="bout_ordinal", domain="within_bout_ordinal",
        source_file=inputs["bout_ordinal_electrode_file"], aggregation="mean",
        bands=bout_bands,
    )

    features = pd.concat(feature_tables, ignore_index=True)
    dictionary = pd.DataFrame.from_records(dictionary_rows)
    if dictionary["feature_id"].duplicated().any():
        raise RuntimeError("Disease-progression feature IDs must be unique")
    if features.duplicated(["subject_id", "feature_id"]).any():
        raise ValueError("Disease-progression features duplicate subject/feature rows")
    features = features.merge(
        cohort[
            ["subject_id", "group", "age_years", "gender", "sex_male", "moca", "updrs"]
        ],
        on=["subject_id", "group"],
        how="left",
        validate="many_to_one",
    ).merge(dictionary, on="feature_id", how="left", validate="many_to_one")
    expected = len(cohort) * len(dictionary)
    if len(features) != expected:
        raise RuntimeError(f"Expected {expected} subject-feature rows, found {len(features)}")
    return (
        features.sort_values(["feature_id", "subject_id"]).reset_index(drop=True),
        dictionary,
        electrodes,
    )


def subject_feature_matrix(cohort: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    wide = features.pivot(index="subject_id", columns="feature_id", values="value")
    wide.columns.name = None
    return cohort.merge(wide.reset_index(), on="subject_id", how="left", validate="one_to_one")
