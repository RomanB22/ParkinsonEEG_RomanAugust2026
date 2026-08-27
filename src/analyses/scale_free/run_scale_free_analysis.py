#!/usr/bin/env python
"""Command-line entry point for oscillatory-bout and cycle analysis."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from analyses.scale_free.pipeline import run_analysis


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run specparam decomposition, aperiodic-relative eBOSC bout "
            "detection, bycycle characterization, and PD/Control comparisons."
        )
    )
    parser.add_argument(
        "--config",
        default="config/analyses/scale_free.json",
        help="Analysis configuration (default: config/analyses/scale_free.json)",
    )
    parser.add_argument(
        "--subjects",
        nargs="*",
        help="Optional participant IDs; default is every participant",
    )
    parser.add_argument(
        "--channels",
        nargs="*",
        help="Optional shared electrodes for a development run; default is every shared electrode",
    )
    parser.add_argument(
        "--output-dir",
        help="Optional output-directory override, useful for development runs",
    )
    parser.add_argument(
        "--feature-source-output-dir",
        help="Optional compatible scale-free output to reuse for a cohort subset",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument(
        "--skip-specparam-gallery",
        action="store_true",
        help="Skip the flat one-all-electrode-figure-per-subject gallery",
    )
    args = parser.parse_args()
    manifest = run_analysis(
        args.config,
        subjects=args.subjects,
        channels=args.channels,
        output_dir_override=args.output_dir,
        feature_source_output_dir_override=args.feature_source_output_dir,
        overwrite=args.overwrite,
        show_progress=not args.no_progress,
        skip_specparam_gallery=args.skip_specparam_gallery,
    )
    print(
        f"Completed scale-free analysis for {manifest['n_subjects']} subjects; "
        f"outputs: {manifest['analysis_config']['output_dir']}"
    )


if __name__ == "__main__":
    main()
