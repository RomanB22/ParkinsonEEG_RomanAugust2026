"""Recompute transparent group inference on a prespecified eight-channel subset."""

from __future__ import annotations

import json
import logging
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.runtime import configure_runtime

configure_runtime()

import matplotlib
import mne
import numpy as np
import pandas as pd
import scipy

from analyses.bouts.pipeline import GROUP_METRICS
from analyses.ordinal.metrics import METRICS as ORDINAL_METRICS
from analyses.scale_free.metrics import APERIODIC_FEATURES, BAND_FEATURES
from core.group_statistics import compute_group_statistics

from .plots import (
    plot_effect_pages,
    plot_electrode_heatmaps,
    plot_group_distribution_pages,
    plot_selection,
)


ELECTRODES = ["F4", "P4", "O2", "P6", "CP2", "CP1", "PO7", "P8"]


def load_analysis_config(path: str | Path) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {"input", "output_dir", "electrodes", "bands", "statistics", "plots"}
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Missing eight-electrode config sections: {missing}")
    if config["electrodes"] != ELECTRODES:
        raise ValueError(f"electrodes must be exactly {ELECTRODES}")
    bands = config["bands"]
    canonical_ordinal = ["delta", "theta", "alpha", "beta", "low_gamma"]
    canonical_bouts = ["theta", "alpha", "low_beta", "high_beta"]
    if bands["psd"] != canonical_ordinal or bands["ordinal"] != canonical_ordinal:
        raise ValueError("Eight-electrode PSD and ordinal bands must be canonical")
    if bands["bout"] != canonical_bouts:
        raise ValueError("Eight-electrode bout bands must be canonical")
    if not 0 < float(config["statistics"]["confidence_level"]) < 1:
        raise ValueError("Invalid confidence level")
    if not 0 < float(config["statistics"]["fdr_alpha"]) < 1:
        raise ValueError("Invalid FDR alpha")
    return config


def _read(path: str | Path, required: set[str]) -> pd.DataFrame:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Missing eight-electrode source: {source}")
    table = pd.read_csv(source)
    missing = sorted(required - set(table))
    if missing:
        raise ValueError(f"{source} is missing columns: {missing}")
    return table


def _feature_id(domain: str, row: pd.Series | dict[str, Any]) -> str:
    band = row.get("band", "broadband")
    return f"{domain}_{band}_{row['metric']}"


def _label(domain: str, band: str, metric: str) -> str:
    domain_label = domain.replace("_", " ").title()
    band_label = "Broadband" if band == "broadband" else band.replace("_", " ").title()
    return f"{domain_label}: {band_label} {metric.replace('_', ' ')}"


def _select_complete(
    table: pd.DataFrame,
    participants: pd.DataFrame,
    *,
    electrodes: list[str],
    strata: list[str],
    bands: list[str] | None,
    source_name: str,
) -> pd.DataFrame:
    subject_ids = set(participants["participant_id"].astype(str))
    selected = table.loc[table["subject_id"].astype(str).isin(subject_ids)].copy()
    selected["subject_id"] = selected["subject_id"].astype(str)
    selected["electrode"] = selected["electrode"].astype(str)
    selected = selected.loc[selected["electrode"].isin(electrodes)].copy()
    if bands is not None:
        selected = selected.loc[selected["band"].astype(str).isin(bands)].copy()
    duplicate_keys = ["subject_id", "electrode", *strata]
    if selected.duplicated(duplicate_keys).any():
        raise ValueError(f"{source_name} contains duplicate subject/electrode rows")
    count_keys = ["subject_id", *strata]
    counts = selected.groupby(count_keys, dropna=False)["electrode"].nunique()
    n_strata = len(bands) if bands is not None else 1
    if len(counts) != len(subject_ids) * n_strata or not counts.eq(len(electrodes)).all():
        raise ValueError(
            f"{source_name} must contain all {len(electrodes)} electrodes for every "
            "participant and requested stratum"
        )
    if set(selected["subject_id"]) != subject_ids:
        raise ValueError(f"{source_name} does not cover the complete analysis cohort")
    return selected


def _subject_values(
    table: pd.DataFrame,
    participants: pd.DataFrame,
    *,
    metrics: tuple[str, ...],
    strata: list[str],
    domain: str,
    aggregation: str,
) -> pd.DataFrame:
    keys = ["subject_id", *strata]
    grouped = table.groupby(keys, dropna=False)[list(metrics)]
    values = grouped.mean() if aggregation == "mean" else grouped.median()
    counts = grouped.count()
    long = values.reset_index().melt(keys, var_name="metric", value_name="value")
    count_long = counts.reset_index().melt(
        keys, var_name="metric", value_name="n_electrodes_contributing"
    )
    long = long.merge(count_long, on=[*keys, "metric"], validate="one_to_one")
    if "band" not in long:
        long["band"] = "broadband"
    metadata_columns = ["participant_id", "GROUP"]
    if "match_pair_id" in participants:
        metadata_columns.append("match_pair_id")
    metadata = participants[metadata_columns].rename(
        columns={"participant_id": "subject_id", "GROUP": "group"}
    )
    metadata["subject_id"] = metadata["subject_id"].astype(str)
    long = long.merge(metadata, on="subject_id", validate="many_to_one")
    long["domain"] = domain
    long["feature_id"] = long.apply(lambda row: _feature_id(domain, row), axis=1)
    long["feature_label"] = long.apply(
        lambda row: _label(domain, str(row["band"]), str(row["metric"])), axis=1
    )
    long["subject_aggregation"] = aggregation
    return long


def build_analysis_tables(
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    inputs = config["input"]
    participants = _read(
        inputs["participants_file"], {"participant_id", "GROUP", "AGE", "GENDER"}
    )
    if participants["participant_id"].duplicated().any():
        raise ValueError("Participant IDs must be unique")
    electrodes = list(config["electrodes"])
    specs = [
        (
            "psd_relative_power", inputs["psd_electrode_file"],
            ("relative_band_power",), ["band"], config["bands"]["psd"], "median",
        ),
        (
            "ordinal_broadband", inputs["ordinal_electrode_file"],
            tuple(ORDINAL_METRICS), [], None, "mean",
        ),
        (
            "ordinal_band", inputs["ordinal_band_electrode_file"],
            tuple(ORDINAL_METRICS), ["band"], config["bands"]["ordinal"], "mean",
        ),
        (
            "aperiodic", inputs["aperiodic_electrode_file"],
            tuple(APERIODIC_FEATURES), [], None, "mean",
        ),
        (
            "periodic_bout", inputs["bout_electrode_file"],
            tuple(BAND_FEATURES), ["band"], config["bands"]["bout"], "mean",
        ),
        (
            "within_bout_ordinal", inputs["bout_ordinal_electrode_file"],
            tuple(GROUP_METRICS), ["band"], config["bands"]["bout"], "mean",
        ),
    ]
    subject_stats: list[pd.DataFrame] = []
    electrode_stats: list[pd.DataFrame] = []
    subject_values: list[pd.DataFrame] = []
    dictionary_rows: list[dict[str, Any]] = []
    for domain, source, metrics, strata, bands, aggregation in specs:
        required = {"subject_id", "group", "electrode", *metrics, *strata}
        selected = _select_complete(
            _read(source, required), participants, electrodes=electrodes,
            strata=strata, bands=bands, source_name=domain,
        )
        subject, electrode = compute_group_statistics(
            selected, participants, metrics=metrics, strata=strata, domain=domain,
            subject_aggregation=aggregation,
            confidence_level=float(config["statistics"]["confidence_level"]),
            fdr_alpha=float(config["statistics"]["fdr_alpha"]),
        )
        for result in (subject, electrode):
            if "band" not in result:
                result["band"] = "broadband"
            result["feature_id"] = result.apply(
                lambda row: _feature_id(domain, row), axis=1
            )
            result["feature_label"] = result.apply(
                lambda row: _label(domain, str(row["band"]), str(row["metric"])), axis=1
            )
        subject_stats.append(subject)
        electrode_stats.append(electrode)
        subject_values.append(
            _subject_values(
                selected, participants, metrics=metrics, strata=strata,
                domain=domain, aggregation=aggregation,
            )
        )
        requested_bands = ["broadband"] if bands is None else bands
        for band in requested_bands:
            for metric in metrics:
                dictionary_rows.append(
                    {
                        "feature_id": f"{domain}_{band}_{metric}",
                        "feature_label": _label(domain, band, metric),
                        "domain": domain,
                        "band": band,
                        "metric": metric,
                        "subject_aggregation": aggregation,
                        "source_file": str(Path(source).resolve()),
                        "electrodes": ",".join(electrodes),
                    }
                )
    dictionary = pd.DataFrame.from_records(dictionary_rows)
    if dictionary["feature_id"].duplicated().any():
        raise RuntimeError("Eight-electrode feature IDs must be unique")
    return (
        pd.concat(subject_values, ignore_index=True),
        dictionary,
        pd.concat(subject_stats, ignore_index=True),
        pd.concat(electrode_stats, ignore_index=True),
    )


def _info(config: dict[str, Any]) -> mne.Info:
    files = sorted(Path().glob(config["input"]["epoch_example_glob"]))
    if not files:
        raise FileNotFoundError("No cleaned epoch file is available for sensor positions")
    epochs = mne.read_epochs(files[0], preload=False, verbose="ERROR")
    missing = sorted(set(ELECTRODES) - set(epochs.ch_names))
    if missing:
        raise ValueError(f"Sensor-position source is missing: {missing}")
    return mne.pick_info(
        epochs.info, [epochs.ch_names.index(name) for name in ELECTRODES], copy=True
    )


def _write_report(
    output: Path, participants: pd.DataFrame, dictionary: pd.DataFrame,
    subject: pd.DataFrame, electrode: pd.DataFrame,
) -> None:
    lines = [
        "# Eight-electrode sensitivity analysis", "",
        "This is a deliberately separate sensitivity battery using exactly F4, P4, O2, "
        "P6, CP2, CP1, PO7, and P8. Whole-head primary outputs are unchanged.", "",
        f"Participants: {len(participants)}; features: {len(dictionary)}.",
        "Full-cohort inference adjusts for age and sex; matched inference is paired.",
        "Only canonical, non-overlapping bands are included.", "",
        "## Corrected results", "",
        "| Domain | Subject tests | Subject FDR | Electrode tests | Strict electrode FDR |",
        "|---|---:|---:|---:|---:|",
    ]
    for domain in dictionary["domain"].drop_duplicates():
        s = subject.loc[subject["domain"].eq(domain)]
        e = electrode.loc[electrode["domain"].eq(domain)]
        lines.append(
            f"| {domain} | {len(s)} | {int(s['primary_fdr_reject_domain'].sum())} | "
            f"{len(e)} | {int(e['primary_fdr_reject_domain'].sum())} |"
        )
    lines.extend(
        ["", "Electrode-wise tests are exploratory. Use the strict domain-wide electrode "
         "FDR columns for formal spatial claims; within-feature FDR is localization only.", ""]
    )
    output.write_text("\n".join(lines), encoding="utf-8")


def run_analysis(
    config_path: str | Path = "config/analyses/eight_electrode.json", *, overwrite: bool = False
) -> dict[str, Any]:
    config_path = Path(config_path)
    config = load_analysis_config(config_path)
    output = Path(config["output_dir"])
    sentinel = output / "manifest.json"
    if sentinel.exists() and not overwrite:
        raise FileExistsError(f"Eight-electrode outputs exist at {sentinel}")
    output.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("eight_electrode_analysis")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(output / "eight_electrode_analysis.log", mode="w")
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)
    logger.info("Recomputing group inference from the eight selected electrodes")
    participants = pd.read_csv(config["input"]["participants_file"])
    values, dictionary, subject, electrode = build_analysis_tables(config)
    metrics = output / "metrics"
    metrics.mkdir(parents=True, exist_ok=True)
    values.to_csv(metrics / "subject_feature_values.csv", index=False, float_format="%.17g")
    dictionary.to_csv(metrics / "feature_dictionary.csv", index=False)
    subject.to_csv(metrics / "group_subject_statistics.csv", index=False, float_format="%.17g")
    electrode.to_csv(metrics / "group_electrode_statistics.csv", index=False, float_format="%.17g")
    pd.DataFrame(
        {"electrode": ELECTRODES, "selection_order": range(1, 9),
         "role": "prespecified eight-electrode sensitivity subset"}
    ).to_csv(metrics / "electrode_selection.csv", index=False)
    figures = output / "figures"
    dpi = int(config["plots"]["dpi"])
    plot_selection(_info(config), figures / "electrode_selection.png", dpi)
    effect_plots = plot_effect_pages(
        subject, figures / "subject_effects", dpi,
        rows_per_page=int(config["plots"]["rows_per_page"]),
    )
    distribution_plots = plot_group_distribution_pages(
        values, subject, figures / "group_distributions", dpi
    )
    heatmaps = plot_electrode_heatmaps(electrode, figures / "electrode_effects", dpi)
    _write_report(output / "REPORT.md", participants, dictionary, subject, electrode)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config_file": str(config_path.resolve()),
        "analysis_config": config,
        "software": {
            "python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "scipy": scipy.__version__,
            "mne": mne.__version__, "matplotlib": matplotlib.__version__,
        },
        "analysis_role": "prespecified eight-electrode sensitivity analysis",
        "electrodes": ELECTRODES,
        "n_electrodes": len(ELECTRODES),
        "n_subjects": int(len(participants)),
        "n_features": int(len(dictionary)),
        "n_subject_tests": int(len(subject)),
        "n_electrode_tests": int(len(electrode)),
        "n_effect_pages": len(effect_plots),
        "n_group_distribution_pages": len(distribution_plots),
        "n_electrode_heatmaps": len(heatmaps),
        "excluded_from_inference": [],
    }
    sentinel.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    logger.info("Completed eight-electrode sensitivity analysis")
    return manifest
