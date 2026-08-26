#!/usr/bin/env python
"""Regenerate subject-average bycycle group figures from saved metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from bycycle_burst_analysis.detector import METRICS
from bycycle_burst_analysis.pipeline import load_analysis_config
from bycycle_burst_analysis.plots import plot_subject_average_violins


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate one PD/Control violin plot per independent-bycycle metric "
            "using saved subject means across shared electrodes."
        )
    )
    parser.add_argument("--config", default="bycycle_burst_analysis/config.json")
    parser.add_argument("--output-dir", help="Optional bycycle output-directory override")
    parser.add_argument("--dpi", type=int, help="Optional figure DPI override")
    args = parser.parse_args()

    config = load_analysis_config(args.config)
    output = Path(args.output_dir or config["output_dir"])
    metrics_path = output / "metrics" / "subject_band_metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"Missing {metrics_path}; complete independent burst detection first"
        )
    subject = pd.read_csv(metrics_path)
    bands = {
        str(name): tuple(float(value) for value in limits)
        for name, limits in config["bands"].items()
    }
    configured_groups = [str(group) for group in config["plots"]["group_order"]]
    present_groups = set(subject["group"].astype(str))
    group_order = [group for group in configured_groups if group in present_groups]
    group_order.extend(sorted(present_groups - set(group_order)))
    colors = {
        group: str(config["plots"]["group_colors"].get(group, "0.4"))
        for group in group_order
    }
    labels = {
        band: (
            f"{config['plots']['band_display_names'].get(band, band)}\n"
            f"{limits[0]:g}–{limits[1]:g} Hz"
        )
        for band, limits in bands.items()
    }
    outputs = plot_subject_average_violins(
        subject,
        metrics=list(METRICS),
        bands=list(bands),
        group_order=group_order,
        colors=colors,
        band_labels=labels,
        output_dir=output / "figures" / "group_comparisons",
        dpi=int(args.dpi if args.dpi is not None else config["plots"]["dpi"]),
    )
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["n_subject_average_violin_figures"] = len(outputs)
        manifest["subject_average_violin_policy"] = (
            "Each plotted point is one subject after arithmetic averaging across "
            "all cohort-shared electrodes; broad_5_15 is descriptive only."
        )
        manifest_path.write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
    print(
        f"Created {len(outputs)} subject-average bycycle group figures in "
        f"{output / 'figures' / 'group_comparisons'}"
    )


if __name__ == "__main__":
    main()
