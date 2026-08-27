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

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from analyses.exploration.matching import match_control_pd_pairs


CONFIG_OUTPUTS = {
    "psd": ("config/analyses/psd.json", "outputs/matched/psd"),
    "ordinal": (
        "config/analyses/ordinal.json",
        "outputs/matched/ordinal",
    ),
    "scale_free": (
        "config/analyses/scale_free.json",
        "outputs/matched/scale_free",
    ),
    "bycycle": (
        "config/analyses/bycycle.json",
        "outputs/matched/bycycle",
    ),
    "bouts": ("config/analyses/bouts.json", "outputs/matched/bouts"),
    "exploration": (
        "config/analyses/exploration.json",
        "outputs/matched/exploration",
    ),
    "behavioral": (
        "config/analyses/behavioral.json",
        "outputs/matched/behavioral",
    ),
    "progression": (
        "config/analyses/progression.json",
        "outputs/matched/progression",
    ),
    "eight_electrode": (
        "config/analyses/eight_electrode.json",
        "outputs/matched/eight_electrode",
    ),
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def prepare_matched_cohort(
    participants_file: str | Path = "processed/metadata/subjects.csv",
    output_root: str | Path = "outputs/matched/cohort",
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
        if name == "ordinal":
            config["input"]["feature_source_output_dir"] = (
                "outputs/full/ordinal"
            )
            config["input"]["feature_source_sweep_root"] = (
                "outputs/full/ordinal_sweep"
            )
        elif name == "bycycle":
            config["input"]["reference_ebosc_output_dir"] = (
                "outputs/matched/scale_free"
            )
        elif name == "scale_free":
            config["input"]["feature_source_output_dir"] = (
                "outputs/full/scale_free"
            )
        elif name == "bouts":
            config["input"]["scale_free_output_dir"] = (
                "outputs/matched/scale_free"
            )
        elif name == "exploration":
            config["input"].update(
                {
                    "ordinal_global_file": "outputs/matched/ordinal/metrics/subject_electrode_mean_metrics.csv",
                    "ordinal_band_file": "outputs/matched/ordinal/metrics/band_subject_electrode_mean_metrics.csv",
                    "psd_subject_band_file": "outputs/matched/psd/metrics/subject_band_power.csv",
                    "aperiodic_subject_file": "outputs/matched/scale_free/metrics/subject_aperiodic_metrics.csv",
                    "bout_subject_file": "outputs/matched/scale_free/metrics/subject_band_metrics.csv",
                    "bout_ordinal_subject_file": "outputs/matched/bouts/metrics/subject_band_metrics.csv",
                    "typical_bout_file": "outputs/matched/scale_free/intermediate/typical_bouts/subject_electrode_band_envelopes.npz",
                    "ordinal_sweep_root": "outputs/matched/ordinal_sweep",
                }
            )
            config["demographic_matching"]["output_dir"] = analysis_output
            config["demographic_matching"]["precomputed_pairs_file"] = str(
                pairs_path
            )
            config["demographic_matching"]["precomputed_balance_file"] = str(
                balance_path
            )
        elif name == "behavioral":
            config["input"].update(
                {
                    "ordinal_subject_file": "outputs/matched/ordinal/metrics/subject_electrode_mean_metrics.csv",
                    "ordinal_band_subject_file": "outputs/matched/ordinal/metrics/band_subject_electrode_mean_metrics.csv",
                    "ordinal_electrode_file": "outputs/matched/ordinal/metrics/electrode_metrics.csv",
                    "ordinal_band_electrode_file": "outputs/matched/ordinal/metrics/band_electrode_metrics.csv",
                    "ordinal_electrode_sets_file": "outputs/matched/ordinal/metrics/electrode_sets.json",
                    "bout_subject_file": "outputs/matched/scale_free/metrics/subject_band_metrics.csv",
                    "bout_electrode_file": "outputs/matched/scale_free/metrics/electrode_band_metrics.csv",
                    "aperiodic_subject_file": "outputs/matched/scale_free/metrics/subject_aperiodic_metrics.csv",
                    "aperiodic_qc_subject_file": "outputs/matched/scale_free/metrics/subject_aperiodic_qc_metrics.csv",
                    "aperiodic_electrode_file": "outputs/matched/scale_free/metrics/electrode_aperiodic_metrics.csv",
                    "bout_ordinal_subject_file": "outputs/matched/bouts/metrics/subject_band_metrics.csv",
                    "bout_ordinal_electrode_file": "outputs/matched/bouts/metrics/subject_electrode_band_metrics.csv",
                }
            )
            config["dimension_sensitivity"]["ordinal_output_root"] = (
                "outputs/matched/ordinal_sweep"
            )
            config["dimension_sensitivity"]["primary_output_dir"] = (
                "outputs/matched/ordinal"
            )
        elif name == "progression":
            config["input"].update(
                {
                    "ordinal_electrode_file": "outputs/matched/ordinal/metrics/electrode_metrics.csv",
                    "ordinal_band_electrode_file": "outputs/matched/ordinal/metrics/band_electrode_metrics.csv",
                    "ordinal_electrode_sets_file": "outputs/matched/ordinal/metrics/electrode_sets.json",
                    "psd_electrode_file": "outputs/matched/psd/metrics/subject_electrode_band_power.csv",
                    "aperiodic_electrode_file": "outputs/matched/scale_free/metrics/electrode_aperiodic_metrics.csv",
                    "bout_electrode_file": "outputs/matched/scale_free/metrics/electrode_band_metrics.csv",
                    "bout_ordinal_electrode_file": "outputs/matched/bouts/metrics/subject_electrode_band_metrics.csv",
                }
            )
        elif name == "eight_electrode":
            config["input"].update(
                {
                    "psd_electrode_file": "outputs/matched/psd/metrics/subject_electrode_band_power.csv",
                    "ordinal_electrode_file": "outputs/matched/ordinal/metrics/electrode_metrics.csv",
                    "ordinal_band_electrode_file": "outputs/matched/ordinal/metrics/band_electrode_metrics.csv",
                    "aperiodic_electrode_file": "outputs/matched/scale_free/metrics/electrode_aperiodic_metrics.csv",
                    "bout_electrode_file": "outputs/matched/scale_free/metrics/electrode_band_metrics.csv",
                    "bout_ordinal_electrode_file": "outputs/matched/bouts/metrics/subject_electrode_band_metrics.csv",
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
            "separate outputs/matched directory; full-cohort outputs are never overwritten."
        ),
        "feature_cache_policy": (
            "Compatible cohort-independent ordinal and scale-free subject features are "
            "filtered from full-cohort caches after strict parameter, electrode, subject, "
            "and row-grid validation. Matched summaries, paired inference, FDR, and figures "
            "are recomputed."
        ),
    }
    _write_json(manifest, output_root / "manifest.json")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--participants", default="processed/metadata/subjects.csv")
    parser.add_argument("--output-root", default="outputs/matched/cohort")
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
