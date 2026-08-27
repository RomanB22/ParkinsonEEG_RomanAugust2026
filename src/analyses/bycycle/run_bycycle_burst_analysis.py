#!/usr/bin/env python
"""CLI for independent cycle-consistency burst detection."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from analyses.bycycle.pipeline import run_analysis


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Detect EEG bursts independently with bycycle cycle consistency, "
            "run group statistics, and compare detections with eBOSC."
        )
    )
    parser.add_argument("--config", default="config/analyses/bycycle.json")
    parser.add_argument("--subjects", nargs="*")
    parser.add_argument("--channels", nargs="*")
    parser.add_argument("--output-dir")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()
    manifest = run_analysis(
        args.config,
        subjects=args.subjects,
        channels=args.channels,
        output_dir_override=args.output_dir,
        workers=args.workers,
        overwrite=args.overwrite,
        show_progress=not args.no_progress,
    )
    print(
        f"Completed independent bycycle burst analysis for {manifest['n_subjects']} "
        f"subjects; outputs: {manifest['analysis_config']['output_dir']}"
    )


if __name__ == "__main__":
    main()
