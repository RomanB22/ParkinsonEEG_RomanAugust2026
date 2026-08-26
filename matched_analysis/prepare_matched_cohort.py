#!/usr/bin/env python
"""Create one canonical age/sex-matched cohort and downstream configurations."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from exploration.matching import match_control_pd_pairs


CONFIG_OUTPUTS = {
    "psd": ("psd_analysis/config.json", "psd_analysis/processed_matched"),
    "ordinal": (
        "ordinal_analysis/config.json",
        "ordinal_analysis/processed_matched",
    ),
    "scale_free": (
        "scale_free_analysis/config.json",
        "scale_free_analysis/processed_matched",
    ),
    "bycycle_burst": (
        "bycycle_burst_analysis/config.json",
        "bycycle_burst_analysis/processed_matched",
    ),
    "bout": ("bout_analyses/config.json", "bout_analyses/processed_matched"),
    "exploration": (
        "exploration/config.json",
        "exploration/processed_matched",
    ),
    "quantitative_behavioral": (
        "quantitative_behavioral/config.json",
        "quantitative_behavioral/processed_matched",
    ),
    "disease_progression": (
        "disease_progression/config.json",
        "disease_progression/processed_matched",
    ),
    "eight_electrode_analysis": (
        "eight_electrode_analysis/config.json",
        "eight_electrode_analysis/processed_matched",
    ),
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def prepare_matched_cohort(
    participants_file: str | Path = "processed/metadata/subjects.csv",
    output_root: str | Path = "matched_analysis/processed",
    *,
    maximum_age_difference_years: float = 5.0,
) -> dict[str, Any]:
    """Match once, save the subject manifest, and derive every pipeline config."""
    participants_path = Path(participants_file)
    output_root = Path(output_root)
    participants = pd.read_csv(participants_path)
    required = {"participant_id", "GROUP", "AGE", "GENDER"}
    missing = sorted(required - set(participants.columns))
    if missing:
        raise ValueError(f"Participant table is missing columns: {missing}")
    if participants["participant_id"].duplicated().any():
        raise ValueError("Participant IDs must be unique")

    matching_input = (
        participants[["participant_id", "GROUP", "AGE", "GENDER"]]
        .rename(
            columns={
                "participant_id": "subject_id",
                "GROUP": "group",
                "AGE": "age_years",
                "GENDER": "gender",
            }
        )
        .copy()
    )
    matching_input["age_years"] = pd.to_numeric(
        matching_input["age_years"], errors="raise"
    )
    matching_input["sex_male"] = matching_input["gender"].eq("M").astype(int)
    matching_input["target_pd"] = matching_input["group"].eq("PD").astype(int)
    matched, pairs, balance = match_control_pd_pairs(
        matching_input,
        maximum_age_difference_years=maximum_age_difference_years,
    )

    matched_metadata = participants.merge(
        matched[["subject_id", "match_pair_id"]].rename(
            columns={"subject_id": "participant_id"}
        ),
        on="participant_id",
        how="inner",
        validate="one_to_one",
    ).sort_values(["match_pair_id", "GROUP"])
    output_root.mkdir(parents=True, exist_ok=True)
    metadata_path = output_root / "matched_subjects.csv"
    pairs_path = output_root / "demographic_match_pairs.csv"
    balance_path = output_root / "demographic_balance.csv"
    subject_ids_path = output_root / "subject_ids.txt"
    matched_metadata.to_csv(metadata_path, index=False)
    pairs.to_csv(pairs_path, index=False)
    balance.to_csv(balance_path, index=False)
    subject_ids_path.write_text(
        "\n".join(matched_metadata["participant_id"].astype(str)) + "\n",
        encoding="utf-8",
    )

    configs_dir = output_root / "configs"
    generated_configs: dict[str, str] = {}
    for name, (source_name, analysis_output) in CONFIG_OUTPUTS.items():
        config = _load_json(Path(source_name))
        config["input"]["participants_file"] = str(metadata_path)
        config["output_dir"] = analysis_output
        if name == "bycycle_burst":
            config["input"]["reference_ebosc_output_dir"] = (
                "scale_free_analysis/processed_matched"
            )
        elif name == "exploration":
            config["input"].update(
                {
                    "ordinal_global_file": "ordinal_analysis/processed_matched/metrics/subject_electrode_mean_metrics.csv",
                    "ordinal_band_file": "ordinal_analysis/processed_matched/metrics/band_subject_electrode_mean_metrics.csv",
                    "psd_subject_band_file": "psd_analysis/processed_matched/metrics/subject_band_power.csv",
                    "aperiodic_subject_file": "scale_free_analysis/processed_matched/metrics/subject_aperiodic_metrics.csv",
                    "bout_subject_file": "scale_free_analysis/processed_matched/metrics/subject_band_metrics.csv",
                    "bout_ordinal_subject_file": "bout_analyses/processed_matched/metrics/subject_band_metrics.csv",
                    "typical_bout_file": "scale_free_analysis/processed_matched/intermediate/typical_bouts/subject_electrode_band_envelopes.npz",
                    "ordinal_sweep_root": "ordinal_analysis/parameter_sweep_matched",
                }
            )
            config["demographic_matching"]["output_dir"] = analysis_output
            config["demographic_matching"]["precomputed_pairs_file"] = str(
                pairs_path
            )
            config["demographic_matching"]["precomputed_balance_file"] = str(
                balance_path
            )
        elif name == "quantitative_behavioral":
            config["input"].update(
                {
                    "ordinal_subject_file": "ordinal_analysis/parameter_sweep_matched/D6_tau1/metrics/subject_electrode_mean_metrics.csv",
                    "ordinal_band_subject_file": "ordinal_analysis/parameter_sweep_matched/D6_tau1/metrics/band_subject_electrode_mean_metrics.csv",
                    "ordinal_electrode_file": "ordinal_analysis/parameter_sweep_matched/D6_tau1/metrics/electrode_metrics.csv",
                    "ordinal_band_electrode_file": "ordinal_analysis/parameter_sweep_matched/D6_tau1/metrics/band_electrode_metrics.csv",
                    "ordinal_electrode_sets_file": "ordinal_analysis/parameter_sweep_matched/D6_tau1/metrics/electrode_sets.json",
                    "bout_subject_file": "scale_free_analysis/processed_matched/metrics/subject_band_metrics.csv",
                    "bout_electrode_file": "scale_free_analysis/processed_matched/metrics/electrode_band_metrics.csv",
                    "aperiodic_subject_file": "scale_free_analysis/processed_matched/metrics/subject_aperiodic_metrics.csv",
                    "aperiodic_qc_subject_file": "scale_free_analysis/processed_matched/metrics/subject_aperiodic_qc_metrics.csv",
                    "aperiodic_electrode_file": "scale_free_analysis/processed_matched/metrics/electrode_aperiodic_metrics.csv",
                    "bout_ordinal_subject_file": "bout_analyses/processed_matched/metrics/subject_band_metrics.csv",
                    "bout_ordinal_electrode_file": "bout_analyses/processed_matched/metrics/subject_electrode_band_metrics.csv",
                }
            )
            config["dimension_sensitivity"]["ordinal_output_root"] = (
                "ordinal_analysis/parameter_sweep_matched"
            )
        elif name == "disease_progression":
            config["input"].update(
                {
                    "ordinal_electrode_file": "ordinal_analysis/parameter_sweep_matched/D6_tau1/metrics/electrode_metrics.csv",
                    "ordinal_band_electrode_file": "ordinal_analysis/parameter_sweep_matched/D6_tau1/metrics/band_electrode_metrics.csv",
                    "ordinal_electrode_sets_file": "ordinal_analysis/parameter_sweep_matched/D6_tau1/metrics/electrode_sets.json",
                    "psd_electrode_file": "psd_analysis/processed_matched/metrics/subject_electrode_band_power.csv",
                    "aperiodic_electrode_file": "scale_free_analysis/processed_matched/metrics/electrode_aperiodic_metrics.csv",
                    "bout_electrode_file": "scale_free_analysis/processed_matched/metrics/electrode_band_metrics.csv",
                    "bout_ordinal_electrode_file": "bout_analyses/processed_matched/metrics/subject_electrode_band_metrics.csv",
                }
            )
        elif name == "eight_electrode_analysis":
            config["input"].update(
                {
                    "psd_electrode_file": "psd_analysis/processed_matched/metrics/subject_electrode_band_power.csv",
                    "ordinal_electrode_file": "ordinal_analysis/parameter_sweep_matched/D6_tau1/metrics/electrode_metrics.csv",
                    "ordinal_band_electrode_file": "ordinal_analysis/parameter_sweep_matched/D6_tau1/metrics/band_electrode_metrics.csv",
                    "aperiodic_electrode_file": "scale_free_analysis/processed_matched/metrics/electrode_aperiodic_metrics.csv",
                    "bout_electrode_file": "scale_free_analysis/processed_matched/metrics/electrode_band_metrics.csv",
                    "bout_ordinal_electrode_file": "bout_analyses/processed_matched/metrics/subject_electrode_band_metrics.csv",
                }
            )
        config_path = configs_dir / f"{name}.json"
        _write_json(config, config_path)
        generated_configs[name] = str(config_path)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_participants_file": str(participants_path.resolve()),
        "matched_participants_file": str(metadata_path.resolve()),
        "matching": {
            "exact_variable": "GENDER",
            "distance_variable": "AGE",
            "algorithm": "optimal_linear_sum_assignment_without_replacement_within_sex",
            "maximum_age_difference_years": maximum_age_difference_years,
        },
        "n_pairs": int(len(pairs)),
        "n_subjects": int(len(matched_metadata)),
        "group_counts": matched_metadata["GROUP"].value_counts().to_dict(),
        "subject_ids": matched_metadata["participant_id"].astype(str).tolist(),
        "generated_configs": generated_configs,
        "output_policy": (
            "Every matched analysis reads this same participant table and writes to a "
            "separate processed_matched directory; full-cohort outputs are never overwritten."
        ),
    }
    _write_json(manifest, output_root / "manifest.json")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--participants", default="processed/metadata/subjects.csv")
    parser.add_argument("--output-root", default="matched_analysis/processed")
    parser.add_argument("--maximum-age-difference-years", type=float, default=5.0)
    args = parser.parse_args()
    manifest = prepare_matched_cohort(
        args.participants,
        args.output_root,
        maximum_age_difference_years=args.maximum_age_difference_years,
    )
    print(
        f"Prepared {manifest['n_pairs']} matched pairs "
        f"({manifest['n_subjects']} subjects)"
    )


if __name__ == "__main__":
    main()
