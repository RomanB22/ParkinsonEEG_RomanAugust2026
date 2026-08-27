#!/usr/bin/env python
"""Generate subject-balanced bout envelope, phase, and shape figures."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from analyses.scale_free.typical_bouts import generate_typical_bout_gallery


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/analyses/scale_free.json")
    parser.add_argument("--workers", type=int)
    parser.add_argument(
        "--overwrite", action="store_true", help="Accepted for pipeline compatibility"
    )
    args = parser.parse_args()
    result = generate_typical_bout_gallery(args.config, workers=args.workers)
    print(
        f"Typical-bout gallery ready: {result['n_subjects']} subjects, "
        f"{result['n_electrodes']} electrodes, {len(result['bands'])} bands"
    )


if __name__ == "__main__":
    main()
