#!/usr/bin/env python
"""Run the prespecified eight-electrode sensitivity analysis."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from eight_electrode_analysis.pipeline import run_analysis


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="eight_electrode_analysis/config.json")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    result = run_analysis(args.config, overwrite=args.overwrite)
    print(
        f"Eight-electrode analysis complete: {result['n_subjects']} subjects, "
        f"{result['n_features']} features"
    )


if __name__ == "__main__":
    main()
