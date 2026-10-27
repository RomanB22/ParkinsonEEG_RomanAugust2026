#!/usr/bin/env python
"""Run the independent all-electrode LAVI rhythmicity analysis."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from analyses.rhythmicity.pipeline import run_analysis


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/analyses/rhythmicity.json")
    parser.add_argument("--datasets", nargs="*")
    parser.add_argument("--recordings", nargs="*")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-figures", action="store_true")
    args = parser.parse_args()
    run_analysis(args.config, datasets=args.datasets, recordings=args.recordings, workers=args.workers, overwrite=args.overwrite, generate_figures=not args.skip_figures)


if __name__ == "__main__":
    main()
