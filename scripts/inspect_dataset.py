#!/usr/bin/env python
"""Inspect all participant metadata and channel layouts without cleaning data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.metadata import inspect_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/preprocessing.yaml")
    args = parser.parse_args()
    summary = inspect_dataset(load_config(args.config))
    print(f"Inspected {summary['n_recordings']} recordings from {summary['n_subjects']} participants")
    print(f"Groups: {summary['group_counts']}")
    print(f"Common channels: {len(summary['common_channels'])}; layouts: {summary['n_layouts']}")
    print("Outputs: processed/metadata and processed/qc/dataset_inspection")


if __name__ == "__main__":
    main()

