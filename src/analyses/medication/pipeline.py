"""End-to-end feature and inference pipeline for ds002778."""

from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any

from core.runtime import configure_runtime

configure_runtime()

import mne
import numpy as np
import pandas as pd

from .cohort import build_cohort
from .features import extract_features
from .plots import plot_condition_features, plot_mmse_features
from .statistics import compute_condition_statistics, compute_mmse_statistics


def load_analysis_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "dataset",
        "input",
        "output_dir",
        "bands",
        "psd",
        "ordinal",
        "specparam",
        "aperiodic_fit_qc",
        "ebosc",
        "duration_sensitivity",
        "statistics",
        "plots",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Missing ds002778 analysis config sections: {missing}")
    if str(config["dataset"]["openneuro_accession"]) != "ds002778-1.0.5":
        raise ValueError("This pipeline is prespecified for ds002778-1.0.5")
    if config["dataset"]["expected_condition_counts"] != {
        "HC": 16,
        "PD_OFF": 15,
        "PD_ON": 15,
    }:
        raise ValueError("Expected condition counts must preserve the complete dataset")
    if int(config["ordinal"]["embedding_dimension"]) != 6:
        raise ValueError("The primary ordinal embedding dimension must be D=6")
    if int(config["ordinal"]["delay_samples"]) != 1:
        raise ValueError("The primary ordinal delay must be one sample")
    if [float(value) for value in config["specparam"]["frequency_range_hz"]] != [
        4.0,
        50.0,
    ]:
        raise ValueError("The aperiodic fit range must remain 4-50 Hz")
    fit_qc = config["aperiodic_fit_qc"]
    if not 0.0 <= float(fit_qc["minimum_r_squared"]) <= 1.0:
        raise ValueError("aperiodic_fit_qc.minimum_r_squared must be in [0, 1]")
    if not 0.0 < float(fit_qc["minimum_subject_qc_fraction"]) <= 1.0:
        raise ValueError(
            "aperiodic_fit_qc.minimum_subject_qc_fraction must be in (0, 1]"
        )
    statistics = config["statistics"]
    if int(statistics["minimum_pairs"]) < 5:
        raise ValueError("statistics.minimum_pairs must be at least five")
    if int(statistics["bootstrap_resamples"]) < 100:
        raise ValueError("statistics.bootstrap_resamples must be at least 100")
    if not 0.0 < float(statistics["fdr_alpha"]) < 1.0:
        raise ValueError("statistics.fdr_alpha must be between zero and one")
    return config


def _write_csv(table: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False, float_format="%.17g")


def _write_cohort_report(
    participants: pd.DataFrame,
    recordings: pd.DataFrame,
    path: Path,
) -> None:
    counts = recordings["condition"].value_counts().to_dict()
    mmse = participants.groupby("diagnosis")["mmse"].agg(
        ["count", "mean", "std", "min", "median", "max"]
    )
    flagged = participants.loc[
        participants["provenance_sensitivity_exclusion"], "participant_id"
    ].tolist()
    report = f"""# ds002778 medication-state cohort audit

- Participants: {len(participants)}
- Recordings: {len(recordings)}
- Conditions: {counts}
- Complete PD ON/OFF pairs: {int((recordings.loc[recordings['diagnosis'].eq('PD')].groupby('participant_id')['condition'].nunique() == 2).sum())}
- MMSE range: {participants['mmse'].min():g}–{participants['mmse'].max():g}
- MMSE values below the dataset's normal threshold (>24): {int((participants['mmse'] <= 24).sum())}
- Preprocessing-provenance sensitivity exclusions: {flagged}

## MMSE by diagnosis

```
{mmse.to_string(float_format=lambda value: f'{value:.3f}')}
```

MMSE is participant-level and does not vary between PD medication sessions.
Accordingly, ON/OFF inference targets EEG change; MMSE analyses target
cross-sectional EEG associations and whether the within-participant EEG
medication response varies with MMSE.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")


def run_analysis(
    config_path: str | Path,
    *,
    subjects: list[str] | None = None,
    output_dir_override: str | Path | None = None,
    epochs_dir_override: str | Path | None = None,
    metadata_only: bool = False,
    statistics_only: bool = False,
    include_ordinal: bool = True,
    include_bouts: bool = True,
    skip_figures: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    config_path = Path(config_path)
    config = load_analysis_config(config_path)
    if output_dir_override is not None:
        config["output_dir"] = str(output_dir_override)
    if epochs_dir_override is not None:
        config["input"]["epochs_dir"] = str(epochs_dir_override)
    output_dir = Path(config["output_dir"])
    metadata_dir = output_dir / "metadata"
    features_dir = output_dir / "features"
    statistics_dir = output_dir / "statistics"
    figures_dir = output_dir / "figures"
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists() and not overwrite and not metadata_only and not statistics_only:
        raise FileExistsError(
            f"ds002778 outputs already exist at {manifest_path}; rerun with --overwrite"
        )

    participants, recordings = build_cohort(
        config["dataset"]["dataset_dir"],
        task=config["dataset"]["task"],
        expected_counts=config["dataset"]["expected_condition_counts"],
    )
    _write_csv(participants, metadata_dir / "participants.csv")
    _write_csv(recordings, metadata_dir / "recordings.csv")
    _write_cohort_report(participants, recordings, metadata_dir / "cohort_audit.md")
    if metadata_only:
        return {
            "status": "metadata_complete",
            "n_participants": len(participants),
            "n_recordings": len(recordings),
            "output_dir": str(output_dir),
        }

    subject_feature_path = features_dir / "subject_features_long.csv"
    if statistics_only:
        if not subject_feature_path.is_file():
            raise FileNotFoundError(
                f"Statistics-only run requires {subject_feature_path}"
            )
        subject_features = pd.read_csv(subject_feature_path)
        electrode_feature_path = features_dir / "electrode_features_long.csv"
        electrode_features = (
            pd.read_csv(electrode_feature_path)
            if electrode_feature_path.is_file()
            else pd.DataFrame()
        )
        feature_products: dict[str, pd.DataFrame] = {}
    else:
        feature_products = extract_features(
            config,
            recordings,
            subjects=subjects,
            include_ordinal=include_ordinal,
            include_bouts=include_bouts,
        )
        subject_features = feature_products["subject_features"]
        electrode_features = feature_products["electrode_features"]
        _write_csv(subject_features, subject_feature_path)
        _write_csv(
            feature_products["electrode_features"],
            features_dir / "electrode_features_long.csv",
        )
        _write_csv(feature_products["subject_psd"], features_dir / "subject_psd.csv")
        _write_csv(
            feature_products["feature_dictionary"],
            features_dir / "feature_dictionary.csv",
        )
        _write_csv(feature_products["input_epochs"], features_dir / "input_epochs.csv")
        inventory = json.loads(feature_products["inventory"].iloc[0]["payload"])
        (features_dir / "electrode_inventory.json").write_text(
            json.dumps(inventory, indent=2) + "\n", encoding="utf-8"
        )

    print("Computing subject-level condition and MMSE inference...")
    condition_statistics = compute_condition_statistics(
        subject_features, recordings, config
    )
    mmse_statistics = compute_mmse_statistics(subject_features, recordings, config)
    _write_csv(condition_statistics, statistics_dir / "condition_contrasts.csv")
    _write_csv(mmse_statistics, statistics_dir / "mmse_associations.csv")
    if not electrode_features.empty:
        print("Computing secondary electrode-level inference...")
        electrode_condition_statistics = compute_condition_statistics(
            electrode_features, recordings, config
        )
        electrode_mmse_statistics = compute_mmse_statistics(
            electrode_features, recordings, config
        )
        _write_csv(
            electrode_condition_statistics,
            statistics_dir / "electrode_condition_contrasts.csv",
        )
        _write_csv(
            electrode_mmse_statistics,
            statistics_dir / "electrode_mmse_associations.csv",
        )

    figure_paths: list[Path] = []
    if not skip_figures:
        figure_paths.extend(
            plot_condition_features(
                subject_features,
                recordings,
                figures_dir / "conditions",
                config,
            )
        )
        figure_paths.extend(
            plot_mmse_features(
                subject_features,
                recordings,
                figures_dir / "mmse",
                config,
            )
        )

    analyzed_recordings = int(subject_features["recording_id"].nunique())
    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis": "ds002778 medication-state and MMSE",
        "analysis_config": str(config_path.resolve()),
        "dataset_dir": str(Path(config["dataset"]["dataset_dir"]).resolve()),
        "output_dir": str(output_dir.resolve()),
        "n_participants_in_dataset": int(len(participants)),
        "n_recordings_in_dataset": int(len(recordings)),
        "n_recordings_analyzed": analyzed_recordings,
        "condition_counts": recordings.loc[
            recordings["recording_id"].isin(subject_features["recording_id"].unique()),
            "condition",
        ].value_counts().to_dict(),
        "duration_variants": sorted(subject_features["duration_variant"].unique()),
        "n_features": int(subject_features["feature_id"].nunique()),
        "ordinal_included": bool(subject_features["family"].eq("ordinal").any()),
        "ebosc_bouts_included": bool(subject_features["family"].eq("bouts").any()),
        "figures": [str(path.resolve()) for path in figure_paths],
        "scientific_notes": [
            "PD ON/OFF recordings are paired by participant.",
            "MMSE is modeled continuously and is not treated as a session-varying outcome.",
            "All MMSE values are in the dataset-defined normal range (>24).",
            "No diagnostic classification or machine-learning analysis is performed.",
        ],
        "software": {
            "python": platform.python_version(),
            "mne": mne.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "statsmodels": version("statsmodels"),
            "specparam": version("specparam"),
            "ordpy": version("ordpy"),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
