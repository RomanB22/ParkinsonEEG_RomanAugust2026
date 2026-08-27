#!/usr/bin/env python
"""Command-line entry point for transparent PD versus Control modeling."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from analyses.exploration.pipeline import run_analysis


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build conservative subject-level ordinal/PSD models with repeated "
            "nested validation and fully documented figures."
        )
    )
    parser.add_argument(
        "--config",
        default="config/analyses/exploration.json",
        help="Analysis configuration (default: config/analyses/exploration.json)",
    )
    parser.add_argument(
        "--output-dir",
        help="Optional output override, useful for development validation",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use two outer repeats, 100 bootstraps, and 20 permutations",
    )
    parser.add_argument("--skip-sweep", action="store_true")
    parser.add_argument("--skip-permutations", action="store_true")
    parser.add_argument(
        "--matched-demographics",
        action="store_true",
        help=(
            "Run the exact-sex, optimal-age matched 49-pair sensitivity cohort; "
            "age and sex are removed from every model"
        ),
    )
    args = parser.parse_args()
    manifest = run_analysis(
        args.config,
        output_dir_override=args.output_dir,
        overwrite=args.overwrite,
        quick=args.quick,
        skip_sweep=args.skip_sweep,
        skip_permutations=args.skip_permutations,
        matched_demographics=args.matched_demographics,
    )
    print(
        f"Completed exploration for {manifest['n_subjects']} subjects; "
        f"outputs: {manifest['analysis_config']['output_dir']}"
    )


if __name__ == "__main__":
    main()
