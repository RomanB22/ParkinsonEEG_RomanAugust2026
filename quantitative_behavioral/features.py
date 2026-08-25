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
    "renyi_entropy_alpha_0_1": "Rényi entropy Hα (α=0.1)",
    "renyi_complexity_alpha_0_1": "Rényi complexity Cα (α=0.1)",
    "renyi_entropy_alpha_0_5": "Rényi entropy Hα (α=0.5)",
    "renyi_complexity_alpha_0_5": "Rényi complexity Cα (α=0.5)",
    "renyi_entropy_alpha_0_9": "Rényi entropy Hα (α=0.9)",
    "renyi_complexity_alpha_0_9": "Rényi complexity Cα (α=0.9)",
    "renyi_entropy_alpha_1_1": "Rényi entropy Hα (α=1.1)",
    "renyi_complexity_alpha_1_1": "Rényi complexity Cα (α=1.1)",
    "renyi_entropy_alpha_2": "Rényi entropy Hα (α=2)",
    "renyi_complexity_alpha_2": "Rényi complexity Cα (α=2)",
    "renyi_entropy_alpha_5": "Rényi entropy Hα (α=5)",
    "renyi_complexity_alpha_5": "Rényi complexity Cα (α=5)",
    "renyi_entropy_alpha_10": "Rényi entropy Hα (α=10)",
    "renyi_complexity_alpha_10": "Rényi complexity Cα (α=10)",
    "oscillatory_occupancy": "Oscillatory occupancy",
    "bouts_per_minute": "Bouts per minute",
    "bout_duration_mean_s": "Mean bout duration",
    "bout_cycles_mean": "Mean cycles per bout",
    "bout_snr_mean": "Mean bout threshold ratio",
    "aperiodic_exponent": "Aperiodic exponent",
    "aperiodic_exponent_qc": "QC-qualified aperiodic exponent",
}

METRIC_UNITS = {
    "entropy": "normalized",
    "complexity": "normalized",
    "fisher_information": "normalized",
    "renyi_entropy_alpha_0_1": "normalized",
    "renyi_complexity_alpha_0_1": "normalized",
    "renyi_entropy_alpha_0_5": "normalized",
    "renyi_complexity_alpha_0_5": "normalized",
    "renyi_entropy_alpha_0_9": "normalized",
    "renyi_complexity_alpha_0_9": "normalized",
    "renyi_entropy_alpha_1_1": "normalized",
    "renyi_complexity_alpha_1_1": "normalized",
    "renyi_entropy_alpha_2": "normalized",
    "renyi_complexity_alpha_2": "normalized",
    "renyi_entropy_alpha_5": "normalized",
    "renyi_complexity_alpha_5": "normalized",
    "renyi_entropy_alpha_10": "normalized",
    "renyi_complexity_alpha_10": "normalized",
    "oscillatory_occupancy": "proportion",
    "bouts_per_minute": "bouts/minute",
    "bout_duration_mean_s": "seconds",
    "bout_cycles_mean": "cycles",
    "bout_snr_mean": "ratio",
    "aperiodic_exponent": "dimensionless",
    "aperiodic_exponent_qc": "dimensionless",
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
                "aperiodic": "aperiodic",
                "ordinal_broadband": "ordinal_broadband",
                "ordinal_band": "ordinal_band",
                "bout_properties": "bout",
                "bout_ordinal": "bout_ordinal",
            }[family]
            feature_id = metric if family == "aperiodic" else (
                f"{prefix}_{metric}" if band == "broadband" else f"{prefix}_{band}_{metric}"
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
    aperiodic_metrics = [str(value) for value in requested["aperiodic_metrics"]]
    ordinal_metrics = [str(value) for value in requested["ordinal_metrics"]]
    bout_properties = [str(value) for value in requested["bout_properties"]]
    bout_ordinal_metrics = [str(value) for value in requested["bout_ordinal_metrics"]]
    required_ordinal = {"subject_id", "group", "n_electrodes", *ordinal_metrics}

    aperiodic_subject = _read_csv(
        inputs["aperiodic_subject_file"],
        {"subject_id", "group", "n_electrodes", "aperiodic_exponent"},
    )
    aperiodic_qc_subject = _read_csv(
        inputs["aperiodic_qc_subject_file"],
        {
            "subject_id",
            "group",
            "n_electrodes",
            "aperiodic_exponent_qc_qualified",
        },
    ).rename(
        columns={"aperiodic_exponent_qc_qualified": "aperiodic_exponent_qc"}
    )
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
        ("aperiodic exponent", aperiodic_subject),
        ("QC-qualified aperiodic exponent", aperiodic_qc_subject),
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
        aperiodic_subject,
        source_file=inputs["aperiodic_subject_file"],
        family="aperiodic",
        domain="aperiodic",
        metrics=["aperiodic_exponent"],
        bands=None,
    )
    _append_subject_features(
        feature_rows,
        dictionary_rows,
        aperiodic_qc_subject,
        source_file=inputs["aperiodic_qc_subject_file"],
        family="aperiodic",
        domain="aperiodic",
        metrics=["aperiodic_exponent_qc"],
        bands=None,
    )
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
                "aperiodic": "aperiodic",
                "ordinal_broadband": "ordinal_broadband",
                "ordinal_band": "ordinal_band",
                "bout_properties": "bout",
                "bout_ordinal": "bout_ordinal",
            }[family]
            feature_id = metric if family == "aperiodic" else (
                f"{prefix}_{metric}" if band == "broadband" else f"{prefix}_{band}_{metric}"
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
    aperiodic_metrics = [str(value) for value in requested["aperiodic_metrics"]]
    ordinal_metrics = [str(value) for value in requested["ordinal_metrics"]]
    bout_properties = [str(value) for value in requested["bout_properties"]]
    bout_ordinal_metrics = [str(value) for value in requested["bout_ordinal_metrics"]]
    aperiodic_electrode = _read_csv(
        inputs["aperiodic_electrode_file"],
        {"subject_id", "group", "electrode", *aperiodic_metrics},
    )
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
        ("aperiodic electrodes", aperiodic_electrode),
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
        ("aperiodic electrodes", aperiodic_electrode),
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
        aperiodic_electrode,
        family="aperiodic",
        metrics=aperiodic_metrics,
        bands=None,
    )
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


def _dimension_source_paths(
    config: dict[str, Any], dimension: int
) -> dict[str, Path]:
    settings = config["dimension_sensitivity"]
    root = Path(settings["ordinal_output_root"])
    delay = int(settings["delay_samples"])
    metrics_dir = root / f"D{dimension}_tau{delay}" / "metrics"
    return {
        "subject": metrics_dir / "subject_electrode_mean_metrics.csv",
        "band_subject": metrics_dir / "band_subject_electrode_mean_metrics.csv",
        "electrode": metrics_dir / "electrode_metrics.csv",
        "band_electrode": metrics_dir / "band_electrode_metrics.csv",
        "electrode_sets": metrics_dir / "electrode_sets.json",
    }


def build_dimension_sensitivity_features(
    config: dict[str, Any], cohort: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """Load regular and Rényi ordinal quantities in separate D analysis blocks."""
    settings = config["dimension_sensitivity"]
    dimensions = [int(value) for value in settings["embedding_dimensions"]]
    delay = int(settings["delay_samples"])
    metrics = [str(value) for value in settings["metrics"]]
    bands = [str(value) for value in settings["bands"]]
    expected_subjects = set(cohort["subject_id"])
    expected_count = int(config["expected"]["shared_electrodes"])
    subject_rows: list[pd.DataFrame] = []
    electrode_rows: list[pd.DataFrame] = []
    dictionary_rows: list[dict[str, Any]] = []
    electrode_order: list[str] | None = None

    for dimension in dimensions:
        paths = _dimension_source_paths(config, dimension)
        subject = _read_csv(
            paths["subject"], {"subject_id", "group", "n_electrodes", *metrics}
        )
        band_subject = _read_csv(
            paths["band_subject"],
            {"subject_id", "group", "band", "n_electrodes", *metrics},
        )
        electrode = _read_csv(
            paths["electrode"], {"subject_id", "group", "electrode", *metrics}
        )
        band_electrode = _read_csv(
            paths["band_electrode"],
            {"subject_id", "group", "electrode", "band", *metrics},
        )
        for name, table in (
            (f"D={dimension} broadband subject", subject),
            (f"D={dimension} band subject", band_subject),
            (f"D={dimension} broadband electrode", electrode),
            (f"D={dimension} band electrode", band_electrode),
        ):
            _validate_subject_coverage(table, expected_subjects, name)
            if "n_electrodes" in table and not table["n_electrodes"].eq(expected_count).all():
                raise ValueError(f"{name} must use exactly {expected_count} electrodes")
            source_groups = table[["subject_id", "group"]].drop_duplicates()
            if source_groups["subject_id"].duplicated().any():
                raise ValueError(f"{name}: inconsistent group labels within subjects")
            compared = cohort[["subject_id", "group"]].merge(
                source_groups,
                on="subject_id",
                suffixes=("_metadata", "_source"),
                validate="one_to_one",
            )
            if not compared["group_metadata"].eq(compared["group_source"]).all():
                raise ValueError(f"{name}: group labels disagree with participant metadata")

        payload_path = paths["electrode_sets"]
        if not payload_path.exists():
            raise FileNotFoundError(f"Missing electrode set: {payload_path}")
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        current_order = [str(value) for value in payload["common_electrodes"]]
        if len(current_order) != expected_count:
            raise ValueError(f"D={dimension} does not use {expected_count} electrodes")
        if electrode_order is None:
            electrode_order = current_order
        elif current_order != electrode_order:
            raise ValueError("Embedding dimensions do not use the same electrode order")

        for band in ["broadband", *bands]:
            selected_subject = subject if band == "broadband" else band_subject.loc[
                band_subject["band"].eq(band)
            ]
            selected_electrode = electrode if band == "broadband" else band_electrode.loc[
                band_electrode["band"].eq(band)
            ]
            if selected_subject.empty or selected_electrode.empty:
                raise ValueError(f"D={dimension}: requested band is unavailable: {band}")
            if selected_subject["subject_id"].duplicated().any():
                raise ValueError(f"D={dimension}/{band}: duplicate subject rows")
            if selected_electrode.duplicated(["subject_id", "electrode"]).any():
                raise ValueError(f"D={dimension}/{band}: duplicate electrode rows")
            if set(selected_electrode["electrode"].astype(str)) != set(current_order):
                raise ValueError(f"D={dimension}/{band}: inconsistent electrode set")

            for metric in metrics:
                feature_id = f"ordinal_D{dimension}_tau{delay}_{band}_{metric}"
                band_label = BAND_LABELS.get(band, band.replace("_", " ").title())
                metric_label = METRIC_LABELS[metric]
                if metric.startswith("renyi_"):
                    alpha_token = metric.rsplit("_alpha_", 1)[1]
                    renyi_alpha = float(alpha_token.replace("_", "."))
                    quantity_set = f"renyi_alpha_{alpha_token}"
                else:
                    renyi_alpha = np.nan
                    quantity_set = "regular"
                feature_label = (
                    f"D={dimension} broadband {metric_label}"
                    if band == "broadband"
                    else f"D={dimension} {band_label} {metric_label}"
                )
                subject_rows.append(
                    selected_subject[["subject_id", metric]]
                    .rename(columns={metric: "value"})
                    .assign(feature_id=feature_id)
                )
                electrode_rows.append(
                    selected_electrode[["subject_id", "group", "electrode", metric]]
                    .rename(columns={metric: "value"})
                    .assign(feature_id=feature_id)
                )
                dictionary_rows.append(
                    {
                        "feature_id": feature_id,
                        "family": f"ordinal_D{dimension}",
                        "domain": f"ordinal_D{dimension}",
                        "band": band,
                        "metric": metric,
                        "feature_label": feature_label,
                        "unit": METRIC_UNITS[metric],
                        "source_file": str(
                            (paths["subject"] if band == "broadband" else paths["band_subject"]).resolve()
                        ),
                        "analysis_level": "subject_mean_across_shared_electrodes",
                        "embedding_dimension": dimension,
                        "delay_samples": delay,
                        "quantity_set": quantity_set,
                        "renyi_alpha": renyi_alpha,
                    }
                )

    dictionary = pd.DataFrame.from_records(dictionary_rows)
    subject_features = pd.concat(subject_rows, ignore_index=True).merge(
        cohort[["subject_id", "group", "moca", "age_years", "gender", "sex_male"]],
        on="subject_id",
        how="left",
        validate="many_to_one",
    ).merge(dictionary, on="feature_id", how="left", validate="many_to_one")
    electrode_features = pd.concat(electrode_rows, ignore_index=True).merge(
        cohort[["subject_id", "group", "moca", "age_years", "sex_male"]],
        on=["subject_id", "group"],
        how="left",
        validate="many_to_one",
    ).merge(dictionary, on="feature_id", how="left", validate="many_to_one")
    if dictionary["feature_id"].duplicated().any():
        raise RuntimeError("Dimension-sensitivity feature identifiers are not unique")
    if subject_features.duplicated(["subject_id", "feature_id"]).any():
        raise ValueError("Dimension-sensitivity subject features contain duplicates")
    if electrode_features.duplicated(["subject_id", "electrode", "feature_id"]).any():
        raise ValueError("Dimension-sensitivity electrode features contain duplicates")
    expected_features = len(dimensions) * (1 + len(bands)) * len(metrics)
    if len(dictionary) != expected_features:
        raise RuntimeError(
            f"Expected {expected_features} dimension-sensitivity features, found {len(dictionary)}"
        )
    return subject_features, dictionary, electrode_features, electrode_order or []
