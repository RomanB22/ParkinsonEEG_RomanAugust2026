#!/usr/bin/env python
"""CLI for the accepted-duration QC sensitivity layer."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from duration_qc_analysis.pipeline import run_duration_sensitivity


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Recompute group, MOCA, and prediction results after requiring at least "
            "60 seconds of accepted EEG."
        )
    )
    parser.add_argument(
        "--config",
        default="duration_qc_analysis/config.json",
        help="Duration-QC configuration file",
    )
    parser.add_argument("--matched", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--skip-models", action="store_true")
    args = parser.parse_args()
    manifest = run_duration_sensitivity(
        args.config,
        matched=args.matched,
        overwrite=args.overwrite,
        quick=args.quick,
        skip_models=args.skip_models,
    )
    print(
        f"Duration-QC sensitivity complete: {manifest['n_included_subjects']}/"
        f"{manifest['n_input_subjects']} subjects included"
    )


if __name__ == "__main__":
    main()

