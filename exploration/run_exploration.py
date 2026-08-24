#!/usr/bin/env python
"""Command-line entry point for transparent PD versus Control modeling."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from exploration.pipeline import run_analysis


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build conservative subject-level ordinal/PSD models with repeated "
            "nested validation and fully documented figures."
        )
    )
    parser.add_argument(
        "--config",
        default="exploration/config.json",
        help="Analysis configuration (default: exploration/config.json)",
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
    args = parser.parse_args()
    manifest = run_analysis(
        args.config,
        output_dir_override=args.output_dir,
        overwrite=args.overwrite,
        quick=args.quick,
        skip_sweep=args.skip_sweep,
        skip_permutations=args.skip_permutations,
    )
    print(
        f"Completed exploration for {manifest['n_subjects']} subjects; "
        f"outputs: {manifest['analysis_config']['output_dir']}"
    )


if __name__ == "__main__":
    main()
