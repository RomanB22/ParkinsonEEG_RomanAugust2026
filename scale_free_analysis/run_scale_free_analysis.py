#!/usr/bin/env python
"""Command-line entry point for oscillatory-bout and cycle analysis."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scale_free_analysis.pipeline import run_analysis


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run specparam decomposition, aperiodic-relative eBOSC bout "
            "detection, bycycle characterization, and PD/Control comparisons."
        )
    )
    parser.add_argument(
        "--config",
        default="scale_free_analysis/config.json",
        help="Analysis configuration (default: scale_free_analysis/config.json)",
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
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()
    manifest = run_analysis(
        args.config,
        subjects=args.subjects,
        channels=args.channels,
        output_dir_override=args.output_dir,
        overwrite=args.overwrite,
        show_progress=not args.no_progress,
    )
    print(
        f"Completed scale-free analysis for {manifest['n_subjects']} subjects; "
        f"outputs: {manifest['analysis_config']['output_dir']}"
    )


if __name__ == "__main__":
    main()
