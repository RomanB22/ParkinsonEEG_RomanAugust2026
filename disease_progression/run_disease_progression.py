#!/usr/bin/env python
"""Run the selected-electrode Parkinson severity-axis analysis."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from disease_progression.pipeline import run_analysis


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="disease_progression/config.json")
    parser.add_argument("--output-dir")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--bootstrap-resamples", type=int)
    args = parser.parse_args()
    manifest = run_analysis(
        args.config,
        output_dir_override=args.output_dir,
        overwrite=args.overwrite,
        bootstrap_resamples_override=args.bootstrap_resamples,
    )
    print(
        f"Disease-progression analysis complete: {manifest['n_pd_subjects']} PD "
        f"subjects, {manifest['n_features']} features"
    )


if __name__ == "__main__":
    main()
