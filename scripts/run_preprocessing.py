#!/usr/bin/env python
"""Sequential, fail-visible batch runner for all or selected participants."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import is_ica_review_confirmed, load_config
from src.dataset import discover_recordings, subject_id_from_path
from src.metadata import expected_channels_from_dataset, update_preprocessing_qc
from src.preprocessing import process_subject


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/preprocessing.yaml")
    parser.add_argument("--subjects", nargs="*", help="Optional participant IDs; default is all recordings")
    parser.add_argument("--review-only", action="store_true")
    review_group = parser.add_mutually_exclusive_group()
    review_group.add_argument("--allow-unreviewed", action="store_true")
    review_group.add_argument(
        "--skip-manual-ica-review",
        action="store_true",
        help="Automatically apply high-confidence ICLabel proposals without visual confirmation",
    )
    parser.add_argument(
        "--no-ica-downsampling",
        "--no-downsampling",
        dest="no_downsampling",
        action="store_true",
        help="Keep ICA at the final 120 Hz rate instead of reducing its temporary copy to 100 Hz",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.review_only and args.skip_manual_ica_review:
        parser.error("--skip-manual-ica-review is only meaningful during cleaning")
    config = load_config(args.config)
    dataset_dir = config["project"]["dataset_dir"]
    task = config["project"]["task"]
    recordings = discover_recordings(dataset_dir, task)
    if args.subjects:
        requested = set(args.subjects)
        recordings = [path for path in recordings if subject_id_from_path(path) in requested]
        found = {subject_id_from_path(path) for path in recordings}
        missing = sorted(requested - found)
        if missing:
            raise FileNotFoundError(f"No recording found for: {missing}")

    if not args.review_only and not args.allow_unreviewed and not args.skip_manual_ica_review:
        unreviewed = [
            subject_id_from_path(path)
            for path in recordings
            if not is_ica_review_confirmed(config, subject_id_from_path(path))
        ]
        if unreviewed:
            preview = ", ".join(unreviewed[:10])
            raise SystemExit(
                f"Refusing to clean {len(unreviewed)} unreviewed ICA decompositions ({preview}...). "
                "Run --review-only first, then add an explicit list for every subject."
            )

    expected = expected_channels_from_dataset(dataset_dir, task, config["channels"]["auxiliary_names"])
    rows = []
    for index, set_path in enumerate(recordings, start=1):
        subject_id = subject_id_from_path(set_path)
        print(f"[{index}/{len(recordings)}] {subject_id}")
        result = process_subject(
            set_path,
            config,
            expected,
            review_only=args.review_only,
            require_review=not args.allow_unreviewed,
            no_downsampling=args.no_downsampling,
            overwrite=args.overwrite,
            config_path=args.config,
            skip_manual_ica_review=args.skip_manual_ica_review,
        )
        if result.qc_row:
            rows.append(result.qc_row)
            update_preprocessing_qc(config["project"]["output_dir"], rows)
    if rows:
        print(f"Completed {len(rows)} participant(s)")


if __name__ == "__main__":
    main()
