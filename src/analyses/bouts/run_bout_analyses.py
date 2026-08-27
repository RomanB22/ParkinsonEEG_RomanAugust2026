#!/usr/bin/env python
"""Command-line entry point for ordinal analysis within detected EEG bouts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from analyses.bouts.pipeline import run_analysis


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Detect aperiodic-relative eBOSC bouts and calculate regular "
            "permutation entropy, complexity, and Fisher information within them."
        )
    )
    parser.add_argument(
        "--config",
        default="config/analyses/bouts.json",
        help="Analysis configuration (default: config/analyses/bouts.json)",
    )
    parser.add_argument(
        "--subjects",
        nargs="*",
        help="Optional participant IDs; default is every participant",
    )
    parser.add_argument(
        "--channels",
        nargs="*",
        help="Optional shared electrodes for a development run",
    )
    parser.add_argument(
        "--output-dir",
        help="Optional output-directory override, useful for pilot runs",
    )
    parser.add_argument(
        "--scale-free-output-dir",
        help="Optional compatible scale-free cache override for pilot runs",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument(
        "--skip-figures",
        action="store_true",
        help="Calculate all metric and statistical tables without rendering figures",
    )
    args = parser.parse_args()
    manifest = run_analysis(
        args.config,
        subjects=args.subjects,
        channels=args.channels,
        output_dir_override=args.output_dir,
        scale_free_output_dir_override=args.scale_free_output_dir,
        overwrite=args.overwrite,
        show_progress=not args.no_progress,
        generate_figures=not args.skip_figures,
    )
    print(
        f"Completed bout ordinal analysis for {manifest['n_subjects']} subjects; "
        f"outputs: {manifest['analysis_config']['output_dir']}"
    )


if __name__ == "__main__":
    main()
