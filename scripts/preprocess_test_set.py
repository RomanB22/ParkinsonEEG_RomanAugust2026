#!/usr/bin/env python
"""Run the required one-Parkinson/one-Control pilot."""

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
    parser.add_argument("--config", default="config/preprocessing.yaml")
    parser.add_argument("--review-only", action="store_true")
    parser.add_argument(
        "--skip-manual-ica-review",
        action="store_true",
        help="Automatically apply high-confidence ICLabel proposals without visual confirmation",
    )
    parser.add_argument(
        "--no-ica-downsampling",
        "--no-downsampling",
        dest="no_downsampling",
        action="store_true",
        help="Keep ICA at the final 120 Hz rate (old --no-downsampling name remains an alias)",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.review_only and args.skip_manual_ica_review:
        parser.error("--skip-manual-ica-review is only meaningful during cleaning")
    config = load_config(args.config)
    dataset_dir = config["project"]["dataset_dir"]
    task = config["project"]["task"]
    expected = expected_channels_from_dataset(dataset_dir, task, config["channels"]["auxiliary_names"])
    rows = []
    for subject_id in config["pilot_subjects"]:
        result = process_subject(
            recording_for_subject(dataset_dir, subject_id, task),
            config,
            expected,
            review_only=args.review_only,
            require_review=True,
            no_downsampling=args.no_downsampling,
            overwrite=args.overwrite,
            config_path=args.config,
            skip_manual_ica_review=args.skip_manual_ica_review,
        )
        if result.qc_row:
            rows.append(result.qc_row)
    if rows:
        print(f"QC table updated: {update_preprocessing_qc(config['project']['output_dir'], rows)}")


if __name__ == "__main__":
    main()
