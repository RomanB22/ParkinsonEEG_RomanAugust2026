#!/usr/bin/env python
"""Command-line entry point for the ordinal EEG analysis."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ordinal_analysis.pipeline import run_analysis


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calculate ordpy entropy, complexity, and Fisher information from cleaned EEG epochs."
    )
    parser.add_argument(
        "--config",
        default="ordinal_analysis/config.json",
        help="Analysis configuration (default: ordinal_analysis/config.json)",
    )
    parser.add_argument(
        "--subjects",
        nargs="*",
        help="Optional BIDS participant IDs; default is every participant",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing ordinal-analysis result files",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the subject/analysis-stage progress bar",
    )
    args = parser.parse_args()
    manifest = run_analysis(
        args.config,
        subjects=args.subjects,
        overwrite=args.overwrite,
        show_progress=not args.no_progress,
    )
    print(
        f"Completed ordinal analysis for {manifest['n_subjects']} subjects; "
        f"outputs: {manifest['analysis_config']['output_dir']}"
    )


if __name__ == "__main__":
    main()
