#!/usr/bin/env python
"""Command-line entry point for the PSD analysis."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from psd_analysis.pipeline import run_analysis


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calculate group median EEG PSD confidence bands and band-power topomaps."
    )
    parser.add_argument("--config", default="psd_analysis/config.json")
    parser.add_argument(
        "--subjects",
        nargs="*",
        help="Optional BIDS participant IDs; default is every participant",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    manifest = run_analysis(args.config, subjects=args.subjects, overwrite=args.overwrite)
    print(
        f"Completed PSD analysis for {manifest['n_subjects']} subjects; "
        f"outputs: {manifest['analysis_config']['output_dir']}"
    )


if __name__ == "__main__":
    main()
