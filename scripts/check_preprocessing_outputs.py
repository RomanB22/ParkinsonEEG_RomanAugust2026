#!/usr/bin/env python
"""Check whether every subject has outputs from the current preprocessing contract."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_preprocessing import _subject_output_is_complete
from src.config import load_config
from src.dataset import discover_recordings, subject_id_from_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/preprocessing.yaml")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    recordings = discover_recordings(
        config["project"]["dataset_dir"], config["project"]["task"]
    )
    stale = [
        subject_id_from_path(path)
        for path in recordings
        if not _subject_output_is_complete(
            config["project"]["output_dir"],
            subject_id_from_path(path),
            review_only=False,
            config=config,
        )
    ]
    if stale:
        if not args.quiet:
            preview = ", ".join(stale[:10])
            print(
                f"Preprocessing is missing or stale for {len(stale)}/{len(recordings)} "
                f"subjects ({preview}{'...' if len(stale) > 10 else ''})"
            )
        raise SystemExit(1)
    if not args.quiet:
        print(f"Current preprocessing outputs found for {len(recordings)} subjects")


if __name__ == "__main__":
    main()
