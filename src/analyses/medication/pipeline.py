"""End-to-end feature and inference pipeline for ds002778."""

from __future__ import annotations

import json
import platform
import shutil
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
from .comparison_plots import plot_comparable_pipeline_figures
from .domain_outputs import publish_domain_outputs
from .statistical_plots import plot_complete_statistical_battery
from .typical_bouts import generate_typical_bout_gallery
from .features import extract_features
from .plots import (
    FOCUSED_BOUT_MMSE_FEATURES,
    FOCUSED_DELTA_UPDRS_FEATURES,
    plot_condition_features,
    plot_focused_bout_mmse,
    plot_focused_updrs,
    plot_mmse_features,
    select_focused_bout_mmse_rows,
    select_focused_delta_updrs_rows,
)
from .statistics import (
    compute_condition_statistics,
    compute_mmse_statistics,
    compute_updrs_statistics,
)


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
        "within_bout_ordinal",
        "typical_bouts",
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
    within_bout = config["within_bout_ordinal"]
    if within_bout.get("metrics") != [
        "entropy",
        "weighted_permutation_entropy",
        "complexity",
        "fisher_information",
    ]:
        raise ValueError(
            "within_bout_ordinal.metrics must preserve entropy, weighted entropy, complexity, "
            "and fisher_information"
        )
    if within_bout.get("pooling") != (
        "pool_pattern_counts_without_crossing_bout_or_epoch_boundaries"
    ):
        raise ValueError("Within-bout ordinal pooling must preserve every boundary")
    typical_bouts = config["typical_bouts"]
    if float(typical_bouts["center_window_seconds"]) <= 0.0:
        raise ValueError("typical_bouts.center_window_seconds must be positive")
    if int(typical_bouts["workers"]) < 1:
        raise ValueError("typical_bouts.workers must be positive")
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
    updrs = recordings.groupby("condition")["total_updrs"].agg(
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

## Total UPDRS by medication condition

```
{updrs.to_string(float_format=lambda value: f'{value:.3f}')}
```

Total UPDRS is session-specific and available for every PD OFF and PD ON
recording. UPDRS analyses use same-session OFF and ON associations plus paired
ON-minus-OFF EEG change versus ON-minus-OFF UPDRS change. Healthy controls are
not assigned a comparable Total UPDRS outcome.
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
    subject_psd_path = features_dir / "subject_psd.csv"
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
        if not subject_psd_path.is_file():
            raise FileNotFoundError(
                f"Statistics-only figure generation requires {subject_psd_path}"
            )
        subject_psd = pd.read_csv(subject_psd_path)
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
        subject_psd = feature_products["subject_psd"]
        _write_csv(subject_features, subject_feature_path)
        _write_csv(
            feature_products["electrode_features"],
            features_dir / "electrode_features_long.csv",
        )
        _write_csv(subject_psd, subject_psd_path)
        _write_csv(
            feature_products["feature_dictionary"],
            features_dir / "feature_dictionary.csv",
        )
        _write_csv(feature_products["input_epochs"], features_dir / "input_epochs.csv")
        bout_episodes = feature_products["bout_episodes"]
        episode_root = output_dir / "intermediate" / "episodes"
        if not bout_episodes.empty:
            episode_root.mkdir(parents=True, exist_ok=True)
            enriched_episodes = bout_episodes.merge(
                recordings[["recording_id", "condition"]],
                on="recording_id",
                validate="many_to_one",
            ).rename(columns={"recording_id": "subject_id", "condition": "group"})
            for recording_id, table in enriched_episodes.groupby("subject_id", sort=False):
                _write_csv(
                    table,
                    episode_root / f"{recording_id}_bout_episodes.csv.gz",
                )
        inventory = json.loads(feature_products["inventory"].iloc[0]["payload"])
        (features_dir / "electrode_inventory.json").write_text(
            json.dumps(inventory, indent=2) + "\n", encoding="utf-8"
        )

    print("Computing subject-level condition, MMSE, and Total UPDRS inference...")
    condition_statistics = compute_condition_statistics(
        subject_features, recordings, config
    )
    mmse_statistics = compute_mmse_statistics(subject_features, recordings, config)
    updrs_statistics = compute_updrs_statistics(
        subject_features, recordings, config
    )
    _write_csv(condition_statistics, statistics_dir / "condition_contrasts.csv")
    _write_csv(mmse_statistics, statistics_dir / "mmse_associations.csv")
    _write_csv(updrs_statistics, statistics_dir / "updrs_associations.csv")
    focused_bout_mmse = select_focused_bout_mmse_rows(mmse_statistics)
    _write_csv(
        focused_bout_mmse,
        statistics_dir / "focused_bout_mmse_associations.csv",
    )
    # Retain the original focused-output path for downstream consumers while
    # expanding it to contain the newly requested bout properties.
    _write_csv(
        focused_bout_mmse,
        statistics_dir / "within_bout_theta_mmse_associations.csv",
    )
    focused_bout_updrs = select_focused_bout_mmse_rows(updrs_statistics)
    focused_delta_updrs = select_focused_delta_updrs_rows(updrs_statistics)
    _write_csv(
        focused_bout_updrs,
        statistics_dir / "focused_bout_updrs_associations.csv",
    )
    _write_csv(
        focused_delta_updrs,
        statistics_dir / "focused_delta_ordinal_updrs_associations.csv",
    )
    electrode_condition_statistics = pd.DataFrame()
    electrode_mmse_statistics = pd.DataFrame()
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
        focused_bout_path = plot_focused_bout_mmse(
            subject_features,
            recordings,
            mmse_statistics,
            figures_dir
            / "mmse"
            / "bouts_and_within_bout_ordinal_mmse_correlations.png",
            config,
        )
        if focused_bout_path is not None:
            legacy_focused_bout_path = (
                figures_dir / "mmse" / "within_bout_theta_mmse_correlations.png"
            )
            shutil.copy2(focused_bout_path, legacy_focused_bout_path)
            figure_paths.append(focused_bout_path)
        focused_bout_updrs_path = plot_focused_updrs(
            subject_features,
            recordings,
            updrs_statistics,
            figures_dir
            / "updrs"
            / "bouts_and_within_bout_ordinal_updrs_correlations.png",
            config,
            feature_specifications=FOCUSED_BOUT_MMSE_FEATURES,
            title="Focused bout EEG metrics versus Total UPDRS",
        )
        if focused_bout_updrs_path is not None:
            figure_paths.append(focused_bout_updrs_path)
        focused_delta_updrs_path = plot_focused_updrs(
            subject_features,
            recordings,
            updrs_statistics,
            figures_dir / "updrs" / "delta_ordinal_updrs_correlations.png",
            config,
            feature_specifications=FOCUSED_DELTA_UPDRS_FEATURES,
            title="Delta ordinal EEG metrics versus Total UPDRS",
        )
        if focused_delta_updrs_path is not None:
            figure_paths.append(focused_delta_updrs_path)
        figure_paths.extend(
            plot_comparable_pipeline_figures(
                subject_features=subject_features,
                electrode_features=electrode_features,
                subject_psd=subject_psd,
                electrode_condition_statistics=electrode_condition_statistics,
                electrode_mmse_statistics=electrode_mmse_statistics,
                recordings=recordings,
                output_dir=figures_dir / "comparable_pipeline",
                config=config,
            )
        )
        figure_paths.extend(
            plot_complete_statistical_battery(
                subject_features=subject_features,
                recordings=recordings,
                condition_statistics=condition_statistics,
                mmse_statistics=mmse_statistics,
                output_dir=figures_dir / "statistical_battery",
                config=config,
            )
        )
        if include_bouts:
            figure_paths.extend(
                generate_typical_bout_gallery(
                    config=config,
                    recordings=recordings,
                    subject_features=subject_features,
                    electrode_features=electrode_features,
                    input_epochs=pd.read_csv(features_dir / "input_epochs.csv"),
                    output_dir=output_dir,
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
        "within_bout_ordinal_included": bool(
            subject_features["family"].eq("within_bout_ordinal").any()
        ),
        "n_figures": len(figure_paths),
        "comparable_figure_directory": str(
            (figures_dir / "comparable_pipeline").resolve()
        ),
        "figures": [str(path.resolve()) for path in figure_paths],
        "scientific_notes": [
            "PD ON/OFF recordings are paired by participant.",
            "MMSE is modeled continuously and is not treated as a session-varying outcome.",
            "Total UPDRS is modeled as a session-specific PD outcome in OFF, ON, and paired ON-minus-OFF change analyses.",
            "All MMSE values are in the dataset-defined normal range (>24).",
            "No diagnostic classification or machine-learning analysis is performed.",
            "Within-bout ordinal patterns are pooled without crossing bout or epoch boundaries.",
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
    domain_outputs = publish_domain_outputs(output_dir)
    manifest["domain_outputs"] = {
        name: {
            "path": str((output_dir / name).resolve()),
            "n_figures": int(summary["n_figures"]),
        }
        for name, summary in domain_outputs.items()
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
