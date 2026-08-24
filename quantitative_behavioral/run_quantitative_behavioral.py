#!/usr/bin/env python
"""Command-line entry point for MOCA associations with EEG quantities."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from quantitative_behavioral.pipeline import run_analysis


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze cross-sectional MOCA associations with regular ordinal "
            "quantities, bout properties, and within-bout ordinal quantities."
        )
    )
    parser.add_argument(
        "--config",
        default="quantitative_behavioral/config.json",
        help="Analysis configuration (default: quantitative_behavioral/config.json)",
    )
    parser.add_argument("--output-dir", help="Optional output-directory override")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--bootstrap-resamples",
        type=int,
        help="Override bootstrap count (minimum 100; useful for development pilots)",
    )
    args = parser.parse_args()
    manifest = run_analysis(
        args.config,
        output_dir_override=args.output_dir,
        overwrite=args.overwrite,
        bootstrap_resamples_override=args.bootstrap_resamples,
    )
    print(
        f"Completed quantitative-behavioral analysis for "
        f"{manifest['n_primary_pd_subjects']} PD subjects; "
        f"outputs: {manifest['analysis_config']['output_dir']}"
    )


if __name__ == "__main__":
    main()

