"""Publish ds002778 results under the same domain ownership used by outputs/full."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


DOMAIN_FAMILIES = {
    "psd": ("psd",),
    "ordinal": ("ordinal",),
    "scale_free": ("aperiodic", "periodic_peak"),
    "bouts": ("bouts", "within_bout_ordinal"),
}

FIGURE_TOKENS = {
    "psd": ("/psd/", "_psd_", "psd_", "relative_power"),
    "ordinal": ("ordinal_",),
    "scale_free": ("aperiodic_", "periodic_peak_"),
    "bouts": ("bouts_", "within_bout_ordinal_"),
}


def _write_csv(table: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False, float_format="%.17g")


def _copy_figures(
    source_root: Path,
    output_root: Path,
    tokens: tuple[str, ...],
) -> list[Path]:
    copied: list[Path] = []
    if not source_root.is_dir():
        return copied
    for source in sorted(source_root.rglob("*.png")):
        relative = source.relative_to(source_root)
        searchable = "/" + relative.as_posix()
        if not any(token in searchable for token in tokens):
            continue
        destination = output_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied.append(destination)
    return copied


def publish_domain_outputs(output_dir: str | Path) -> dict[str, dict[str, Any]]:
    """Split the recording-aware analysis cache into domain-owned products.

    The canonical long tables remain available for cross-domain paired models.
    Domain folders contain filtered copies so their organization matches the
    primary pipeline without pretending ON/OFF recordings are independent.
    """
    root = Path(output_dir)
    feature_root = root / "features"
    statistics_root = root / "statistics"
    figure_root = root / "figures"
    subject_features = pd.read_csv(feature_root / "subject_features_long.csv")
    electrode_features = pd.read_csv(feature_root / "electrode_features_long.csv")
    condition = pd.read_csv(statistics_root / "condition_contrasts.csv")
    mmse = pd.read_csv(statistics_root / "mmse_associations.csv")
    electrode_condition = pd.read_csv(
        statistics_root / "electrode_condition_contrasts.csv"
    )
    electrode_mmse = pd.read_csv(
        statistics_root / "electrode_mmse_associations.csv"
    )
    source_manifest_path = root / "manifest.json"
    source_manifest = (
        json.loads(source_manifest_path.read_text(encoding="utf-8"))
        if source_manifest_path.is_file()
        else {}
    )

    summaries: dict[str, dict[str, Any]] = {}
    for domain, families in DOMAIN_FAMILIES.items():
        domain_root = root / domain
        metrics_root = domain_root / "metrics"
        figures_root = domain_root / "figures"
        selected_subject = subject_features.loc[subject_features["family"].isin(families)]
        selected_electrode = electrode_features.loc[electrode_features["family"].isin(families)]
        selected_condition = condition.loc[condition["family"].isin(families)]
        selected_mmse = mmse.loc[mmse["family"].isin(families)]
        selected_electrode_condition = electrode_condition.loc[
            electrode_condition["family"].isin(families)
        ]
        selected_electrode_mmse = electrode_mmse.loc[
            electrode_mmse["family"].isin(families)
        ]
        _write_csv(selected_subject, metrics_root / "subject_features.csv")
        _write_csv(selected_electrode, metrics_root / "electrode_features.csv")
        _write_csv(selected_condition, metrics_root / "condition_contrasts.csv")
        _write_csv(selected_mmse, metrics_root / "mmse_associations.csv")
        _write_csv(
            selected_electrode_condition,
            metrics_root / "electrode_condition_contrasts.csv",
        )
        _write_csv(
            selected_electrode_mmse,
            metrics_root / "electrode_mmse_associations.csv",
        )
        if domain == "psd":
            shutil.copy2(feature_root / "subject_psd.csv", metrics_root / "subject_psd.csv")
        figures = _copy_figures(
            figure_root,
            figures_root,
            FIGURE_TOKENS[domain],
        )
        manifest = {
            "schema_version": 1,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "dataset": "ds002778",
            "domain": domain,
            "families": list(families),
            "sampling_unit": "recording with participant-aware paired inference",
            "source_analysis_manifest": str(source_manifest_path.resolve()),
            "source_analysis_created_utc": source_manifest.get("created_utc"),
            "n_subject_feature_rows": int(len(selected_subject)),
            "n_electrode_feature_rows": int(len(selected_electrode)),
            "n_condition_statistics": int(len(selected_condition)),
            "n_mmse_statistics": int(len(selected_mmse)),
            "n_figures": len(figures),
            "figures": [str(path.resolve()) for path in figures],
        }
        domain_root.mkdir(parents=True, exist_ok=True)
        (domain_root / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        summaries[domain] = manifest

    behavioral_root = root / "behavioral"
    _write_csv(mmse, behavioral_root / "metrics" / "mmse_associations.csv")
    _write_csv(
        electrode_mmse,
        behavioral_root / "metrics" / "electrode_mmse_associations.csv",
    )
    behavioral_figures = _copy_figures(
        figure_root,
        behavioral_root / "figures",
        ("/mmse/", "/correlations/"),
    )
    behavioral_manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "ds002778",
        "domain": "behavioral",
        "outcome": "MMSE",
        "sampling_unit": "participant; PD medication sessions remain paired",
        "n_statistics": int(len(mmse)),
        "n_electrode_statistics": int(len(electrode_mmse)),
        "n_figures": len(behavioral_figures),
        "figures": [str(path.resolve()) for path in behavioral_figures],
    }
    behavioral_root.mkdir(parents=True, exist_ok=True)
    (behavioral_root / "manifest.json").write_text(
        json.dumps(behavioral_manifest, indent=2) + "\n", encoding="utf-8"
    )
    summaries["behavioral"] = behavioral_manifest
    if source_manifest_path.is_file():
        source_manifest["domain_outputs"] = {
            name: {
                "path": str((root / name).resolve()),
                "n_figures": int(summary["n_figures"]),
            }
            for name, summary in summaries.items()
        }
        source_manifest_path.write_text(
            json.dumps(source_manifest, indent=2) + "\n", encoding="utf-8"
        )
    return summaries
