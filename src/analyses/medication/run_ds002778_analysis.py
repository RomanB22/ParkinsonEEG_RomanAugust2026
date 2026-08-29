#!/usr/bin/env python
"""Command-line entry point for ds002778 medication/MMSE analysis."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from analyses.medication.pipeline import run_analysis


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze healthy controls and paired PD ON/OFF recordings in ds002778."
    )
    parser.add_argument(
        "--config", default="config/analyses/ds002778.json"
    )
    parser.add_argument("--subjects", nargs="*", help="Optional participant or recording IDs")
    parser.add_argument("--output-dir", help="Optional output override for pilots")
    parser.add_argument("--epochs-dir", help="Optional cleaned-epoch input override for pilots")
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--statistics-only", action="store_true")
    parser.add_argument("--skip-ordinal", action="store_true")
    parser.add_argument("--skip-bouts", action="store_true")
    parser.add_argument("--skip-figures", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.metadata_only and args.statistics_only:
        parser.error("--metadata-only and --statistics-only are mutually exclusive")
    manifest = run_analysis(
        args.config,
        subjects=args.subjects,
        output_dir_override=args.output_dir,
        epochs_dir_override=args.epochs_dir,
        metadata_only=args.metadata_only,
        statistics_only=args.statistics_only,
        include_ordinal=not args.skip_ordinal,
        include_bouts=not args.skip_bouts,
        skip_figures=args.skip_figures,
        overwrite=args.overwrite,
    )
    print(
        f"ds002778 analysis status: {manifest.get('status', 'complete')} | "
        f"recordings: {manifest.get('n_recordings_analyzed', manifest.get('n_recordings'))} | "
        f"output: {manifest.get('output_dir')}"
    )


if __name__ == "__main__":
    main()
