#!/usr/bin/env python
"""Run or review one participant."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.dataset import recording_for_subject
from src.metadata import expected_channels_from_dataset, update_preprocessing_qc
from src.preprocessing import process_subject


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("subject_id", help="BIDS participant ID, for example sub-001")
    parser.add_argument("--config", default="config/preprocessing.yaml")
    parser.add_argument("--review-only", action="store_true", help="Stop after saving ICA review material")
    review_group = parser.add_mutually_exclusive_group()
    review_group.add_argument("--allow-unreviewed", action="store_true", help="Proceed conservatively with no ICA removal when no explicit review entry exists")
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
        help="Keep the temporary ICA copy at the final 120 Hz rate instead of reducing it to 100 Hz",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.review_only and args.skip_manual_ica_review:
        parser.error("--skip-manual-ica-review is only meaningful during cleaning")
    config = load_config(args.config)
    dataset_dir = config["project"]["dataset_dir"]
    expected = expected_channels_from_dataset(dataset_dir, config["project"]["task"], config["channels"]["auxiliary_names"])
    set_path = recording_for_subject(dataset_dir, args.subject_id, config["project"]["task"])
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
        path = update_preprocessing_qc(config["project"]["output_dir"], [result.qc_row])
        print(f"QC table updated: {path}")


if __name__ == "__main__":
    main()
