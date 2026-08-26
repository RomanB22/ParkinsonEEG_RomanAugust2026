#!/usr/bin/env python
"""Rerun specparam fit QC and range sensitivity from saved spectral curves."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from scale_free_analysis.aperiodic_diagnostics import run_aperiodic_diagnostics
from scale_free_analysis.pipeline import load_analysis_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit every saved selected specparam fit and refit the configured "
            "fixed-versus-knee frequency-range sensitivity models without "
            "rerunning eBOSC/bycycle."
        )
    )
    parser.add_argument("--config", default="scale_free_analysis/config.json")
    parser.add_argument("--output-dir", help="Override the scale-free output directory")
    args = parser.parse_args()
    config = load_analysis_config(args.config)
    output_dir = Path(args.output_dir or config["output_dir"])
    metrics_path = output_dir / "metrics" / "electrode_aperiodic_metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"Missing {metrics_path}; run the scale-free analysis first"
        )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    results = run_aperiodic_diagnostics(
        output_dir,
        pd.read_csv(metrics_path),
        config,
        logger=logging.getLogger("aperiodic_diagnostics"),
    )
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        metrics = results["electrode_metrics"]
        manifest["analysis_config"] = config
        manifest["specparam_fit_qc"] = {
            "thresholds": config["aperiodic_fit_qc"],
            "n_fits": int(len(metrics)),
            "n_qc_pass": int(metrics["specparam_fit_qc_pass"].sum()),
            "qc_pass_fraction": float(metrics["specparam_fit_qc_pass"].mean()),
            "frequency_ranges_hz": config["aperiodic_sensitivity"][
                "frequency_ranges_hz"
            ],
            "n_range_sensitivity_fits": int(
                len(results["electrode_sensitivity"])
            ),
            "policy": config["aperiodic_fit_qc"]["policy"],
        }
        manifest_path.write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
    print(
        "Aperiodic diagnostics complete: "
        f"{len(results['electrode_metrics'])} primary fits and "
        f"{len(results['electrode_sensitivity'])} range-sensitivity fits"
    )


if __name__ == "__main__":
    main()
