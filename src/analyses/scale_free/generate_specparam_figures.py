#!/usr/bin/env python
"""Generate all decomposition figures from existing scale-free intermediates."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd

from analyses.scale_free.specparam_gallery import generate_specparam_gallery


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Render exactly one all-electrode specparam overview per subject "
            "in one flat folder from saved intermediates."
        )
    )
    parser.add_argument("--config", default="config/analyses/scale_free.json")
    parser.add_argument("--output-dir", help="Override the scale-free output directory")
    parser.add_argument("--dpi", type=int, help="Override gallery DPI")
    parser.add_argument("--workers", type=int, help="Override parallel worker count")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate existing PNGs; default resumes and skips them",
    )
    parser.add_argument(
        "--overwrite-subject-overviews",
        action="store_true",
        help="Backward-compatible alias to regenerate the all-electrode figures",
    )
    args = parser.parse_args()
    with Path(args.config).open(encoding="utf-8") as stream:
        config = json.load(stream)
    output_dir = Path(args.output_dir or config["output_dir"])
    metrics_path = output_dir / "metrics" / "electrode_aperiodic_metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"Missing {metrics_path}; run the scale-free analysis first"
        )
    metrics = pd.read_csv(metrics_path)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    index = generate_specparam_gallery(
        output_dir / "intermediate" / "spectra",
        metrics,
        output_dir / "figures" / "specparam_decomposition",
        dpi=int(
            args.dpi
            if args.dpi is not None
            else config["plots"].get("specparam_gallery_dpi", 100)
        ),
        workers=int(
            args.workers
            if args.workers is not None
            else config["plots"].get("specparam_gallery_workers", 1)
        ),
        overwrite=args.overwrite,
        overwrite_subject_overviews=args.overwrite_subject_overviews,
        logger=logging.getLogger("specparam_gallery"),
    )
    metrics_index = output_dir / "metrics" / "specparam_figure_index.csv"
    index.to_csv(metrics_index, index=False, float_format="%.17g")
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["n_specparam_decomposition_figures"] = int(len(index))
        manifest["n_specparam_subject_overview_figures"] = int(
            index["subject_id"].nunique()
        )
        manifest["specparam_gallery_enabled"] = True
        manifest["specparam_gallery_policy"] = (
            "Flat single-folder layout with exactly one all-electrode PNG per subject "
            "and one root HTML index; no electrode PNGs or subject folders. Figures "
            "reuse saved fitted spectral curves without refitting specparam."
        )
        manifest["specparam_gallery_layout"] = (
            "flat_single_folder_one_all_electrode_png_per_subject"
        )
        manifest_path.write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
    rendered = int(index["rendered_this_run"].sum())
    print(
        f"Specparam gallery ready: {len(index)} all-electrode subject figures "
        f"in one folder ({rendered} rendered now); "
        f"open {output_dir / 'figures' / 'specparam_decomposition' / 'index.html'}"
    )


if __name__ == "__main__":
    main()
