#!/usr/bin/env python
"""Re-aggregate bout and within-bout ordinal results using fit-QC electrodes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scale_free_analysis.fit_qc_sensitivity import run_fit_qc_sensitivity
from quantitative_behavioral.fit_qc_sensitivity import (
    run_behavioral_fit_qc_sensitivity,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Propagate audited specparam fit QC into bout-property and within-bout "
            "ordinal sensitivity summaries."
        )
    )
    parser.add_argument("--scale-free-output", default="scale_free_analysis/processed")
    parser.add_argument("--bout-ordinal-output", default="bout_analyses/processed")
    parser.add_argument(
        "--participants", default="processed/metadata/subjects.csv"
    )
    parser.add_argument(
        "--behavioral-config",
        default="quantitative_behavioral/config.json",
        help="Quantitative-behavioral config receiving the fit-QC MOCA sensitivity",
    )
    parser.add_argument(
        "--behavioral-scale-free-qc-subject-file",
        help="Override the fit-QC bout-property subject table for MOCA analysis",
    )
    parser.add_argument(
        "--behavioral-bout-ordinal-qc-subject-file",
        help="Override the fit-QC within-bout ordinal subject table for MOCA analysis",
    )
    parser.add_argument("--minimum-subject-qc-fraction", type=float, default=0.8)
    parser.add_argument("--fdr-alpha", type=float, default=0.05)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Accepted for compatibility; deterministic QC outputs are regenerated",
    )
    args = parser.parse_args()
    result = run_fit_qc_sensitivity(
        scale_free_output=args.scale_free_output,
        bout_ordinal_output=args.bout_ordinal_output,
        participants_file=args.participants,
        minimum_subject_qc_fraction=args.minimum_subject_qc_fraction,
        fdr_alpha=args.fdr_alpha,
        dpi=args.dpi,
    )
    behavioral_arguments = {"config_path": args.behavioral_config}
    if args.behavioral_scale_free_qc_subject_file:
        behavioral_arguments["scale_free_qc_subject_file"] = (
            args.behavioral_scale_free_qc_subject_file
        )
    if args.behavioral_bout_ordinal_qc_subject_file:
        behavioral_arguments["bout_ordinal_qc_subject_file"] = (
            args.behavioral_bout_ordinal_qc_subject_file
        )
    behavioral = run_behavioral_fit_qc_sensitivity(**behavioral_arguments)
    print(
        "Fit-QC sensitivity complete: "
        f"{result['n_qualified_subjects']}/{result['n_subjects']} subjects qualified; "
        f"MOCA sensitivity uses {behavioral['n_qualified_pd_subjects']} PD subjects"
    )


if __name__ == "__main__":
    main()
