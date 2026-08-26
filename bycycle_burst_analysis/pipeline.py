"""End-to-end independent bycycle burst-detection sensitivity pipeline."""

from __future__ import annotations

import json
import logging
import platform
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any

from src.runtime import configure_runtime

configure_runtime()

import matplotlib
import mne
import numpy as np
import pandas as pd
import scipy
from scipy.stats import spearmanr
from tqdm.auto import tqdm

from src.dataset import ordered_channel_inventory
from src.group_statistics import compute_group_statistics
from src.group_statistics_plots import plot_electrode_group_statistics
from src.output_cleanup import remove_retired_band_outputs

from .detector import METRICS, detect_epoch_bursts, summarize_detection
from .plots import (
    plot_detection_coverage,
    plot_detection_example,
    plot_event_agreement,
    plot_group_metric_violins,
    plot_metric_agreement,
    plot_subject_average_violins,
)


SUBJECT_PATTERN = re.compile(r"(sub-\d+)")


def load_analysis_config(path: str | Path) -> dict[str, Any]:
    """Load and validate the independent detector configuration."""
    with Path(path).open(encoding="utf-8") as stream:
        config = json.load(stream)
    required = {"input", "output_dir", "bands", "detector", "statistics", "plots"}
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Missing bycycle burst config sections: {missing}")
    expected = ["theta", "alpha", "low_beta", "high_beta"]
    if list(config["bands"]) != expected:
        raise ValueError(f"bands must be ordered as {expected}")
    for band, limits in config["bands"].items():
        if len(limits) != 2 or not 0 < float(limits[0]) < float(limits[1]):
            raise ValueError(f"Invalid band limits for {band}")
    detector = config["detector"]
    threshold_names = (
        "amplitude_fraction_threshold",
        "amplitude_consistency_threshold",
        "period_consistency_threshold",
        "monotonicity_threshold",
    )
    for name in threshold_names:
        if not 0.0 <= float(detector[name]) <= 1.0:
            raise ValueError(f"detector.{name} must be in [0, 1]")
    if int(detector["minimum_consecutive_cycles"]) < 1:
        raise ValueError("minimum_consecutive_cycles must be positive")
    if float(detector["edge_padding_seconds"]) < 0.0:
        raise ValueError("edge_padding_seconds cannot be negative")
    if int(detector["workers"]) < 1:
        raise ValueError("detector.workers must be positive")
    statistics = config["statistics"]
    unknown_metrics = sorted(set(statistics["inferential_metrics"]) - set(METRICS))
    if unknown_metrics:
        raise ValueError(f"Unknown inferential metrics: {unknown_metrics}")
    unknown_bands = sorted(set(statistics.get("exclude_bands", [])) - set(config["bands"]))
    if unknown_bands:
        raise ValueError(f"Unknown excluded bands: {unknown_bands}")
    if not 0.0 < float(statistics["fdr_alpha"]) < 1.0:
        raise ValueError("statistics.fdr_alpha must be in (0, 1)")
    if not 0.0 < float(statistics["confidence_level"]) < 1.0:
        raise ValueError("statistics.confidence_level must be in (0, 1)")
    return config


def _participants(path: Path) -> pd.DataFrame:
    separator = "\t" if path.suffix.lower() == ".tsv" else ","
    table = pd.read_csv(path, sep=separator)
    required = {"participant_id", "GROUP", "AGE", "GENDER"}
    missing = sorted(required - set(table))
    if missing:
        raise ValueError(f"Participant table is missing columns: {missing}")
    if table["participant_id"].duplicated().any():
        raise ValueError("Participant IDs must be unique")
    return table


def _epoch_files(directory: Path, pattern: str) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for path in sorted(directory.glob(pattern)):
        match = SUBJECT_PATTERN.search(path.name)
        if match:
            if match.group(1) in files:
                raise ValueError(f"Multiple epoch files found for {match.group(1)}")
            files[match.group(1)] = path
    return files


def _write_csv(table: pd.DataFrame, path: Path, *, compressed: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(
        path, index=False, float_format="%.17g", compression="gzip" if compressed else None
    )


def _logger(output: Path, overwrite: bool) -> logging.Logger:
    output.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("bycycle_burst_analysis")
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(console)
    file_handler = logging.FileHandler(
        output / "bycycle_burst_analysis.log", mode="w" if overwrite else "a"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def _process_subject(task: dict[str, Any]) -> dict[str, Any]:
    """Worker: process every shared electrode and band for one subject."""
    subject_id = str(task["subject_id"])
    group = str(task["group"])
    bands = task["bands"]
    settings = task["settings"]
    output = Path(task["output_dir"])
    channels = list(task["channels"])
    epochs = mne.read_epochs(task["epoch_file"], preload=True, verbose="ERROR")
    picks = [epochs.ch_names.index(channel) for channel in channels]
    data_uv = epochs.get_data(picks=picks, copy=True) * 1e6
    sfreq = float(epochs.info["sfreq"])
    n_epochs, _, n_samples = data_uv.shape
    edge = int(round(float(settings["edge_padding_seconds"]) * sfreq))
    if 2 * edge >= n_samples:
        raise ValueError(f"{subject_id}: edge padding removes the entire epoch")
    analyzed_duration_s = n_epochs * (n_samples - 2 * edge) / sfreq
    metric_rows: list[dict[str, Any]] = []
    event_tables: list[pd.DataFrame] = []
    burst_cycle_tables: list[pd.DataFrame] = []
    first_electrode_masks: dict[str, list[np.ndarray]] = {}

    for electrode_index, electrode in enumerate(channels):
        for band, limits in bands.items():
            cycle_tables: list[pd.DataFrame] = []
            band_events: list[pd.DataFrame] = []
            epoch_masks: list[np.ndarray] = []
            for epoch_index in range(n_epochs):
                cycles, events, mask = detect_epoch_bursts(
                    data_uv[epoch_index, electrode_index],
                    sfreq=sfreq,
                    band_limits=limits,
                    settings=settings,
                )
                if electrode_index == 0:
                    epoch_masks.append(mask)
                if not cycles.empty:
                    cycles = cycles.copy()
                    cycles.insert(0, "epoch_index", epoch_index)
                    cycle_tables.append(cycles)
                if not events.empty:
                    events = events.copy()
                    events.insert(0, "epoch_index", epoch_index)
                    events["inter_bout_interval_s"] = np.nan
                    if len(events) > 1:
                        events.loc[events.index[1:], "inter_bout_interval_s"] = (
                            events["onset_s"].to_numpy(float)[1:]
                            - events["offset_s"].to_numpy(float)[:-1]
                        )
                    band_events.append(events)
            cycles_all = pd.concat(cycle_tables, ignore_index=True) if cycle_tables else pd.DataFrame()
            events_all = pd.concat(band_events, ignore_index=True) if band_events else pd.DataFrame()
            summary = summarize_detection(
                cycles_all, events_all, analyzed_duration_s=analyzed_duration_s
            )
            base = {
                "subject_id": subject_id,
                "group": group,
                "electrode": electrode,
                "band": band,
                "band_low_hz": float(limits[0]),
                "band_high_hz": float(limits[1]),
                "sfreq_hz": sfreq,
                "n_epochs": n_epochs,
                "epoch_duration_s": n_samples / sfreq,
            }
            metric_rows.append({**base, **summary})
            if not events_all.empty:
                event_tables.append(pd.concat([
                    pd.DataFrame({key: [value] * len(events_all) for key, value in base.items()}),
                    events_all.reset_index(drop=True),
                ], axis=1))
            if not cycles_all.empty:
                burst_cycles = cycles_all.loc[cycles_all["is_burst"].astype(bool)].copy()
                if not burst_cycles.empty:
                    burst_cycle_tables.append(pd.concat([
                        pd.DataFrame({key: [value] * len(burst_cycles) for key, value in base.items()}),
                        burst_cycles.reset_index(drop=True),
                    ], axis=1))
            if electrode_index == 0:
                first_electrode_masks[band] = epoch_masks

    example: dict[str, Any] | None = None
    if first_electrode_masks:
        detection_counts = np.zeros(n_epochs, dtype=int)
        for epoch_masks in first_electrode_masks.values():
            detection_counts += np.asarray([int(mask.sum()) for mask in epoch_masks])
        example_epoch = int(np.argmax(detection_counts))
        example = {
            "subject_id": subject_id,
            "electrode": channels[0],
            "epoch_index": example_epoch,
            "sfreq": sfreq,
            "signal_uv": data_uv[example_epoch, 0].copy(),
            "masks": {
                band: epoch_masks[example_epoch]
                for band, epoch_masks in first_electrode_masks.items()
            },
        }

    events_subject = pd.concat(event_tables, ignore_index=True) if event_tables else pd.DataFrame()
    cycles_subject = (
        pd.concat(burst_cycle_tables, ignore_index=True) if burst_cycle_tables else pd.DataFrame()
    )
    _write_csv(
        events_subject,
        output / "intermediate" / "episodes" / f"{subject_id}_bycycle_bouts.csv.gz",
        compressed=True,
    )
    _write_csv(
        cycles_subject,
        output / "intermediate" / "cycles" / f"{subject_id}_burst_cycles.csv.gz",
        compressed=True,
    )
    return {
        "metrics": metric_rows,
        "example": example,
        "input": {
            "subject_id": subject_id,
            "group": group,
            "epoch_file": str(Path(task["epoch_file"]).resolve()),
            "n_epochs": n_epochs,
            "sfreq_hz": sfreq,
            "epoch_duration_s": n_samples / sfreq,
            "analyzed_duration_s": analyzed_duration_s,
            "n_shared_electrodes": len(channels),
        },
    }


def _subject_means(electrode: pd.DataFrame) -> pd.DataFrame:
    keys = ["subject_id", "group", "band", "band_low_hz", "band_high_hz"]
    values = ["n_candidate_cycles", "n_burst_cycles", "n_bouts", "analyzed_duration_s", *METRICS]
    result = electrode.groupby(keys, sort=False)[values].mean().reset_index()
    result["n_electrodes"] = electrode.groupby(keys, sort=False)["electrode"].nunique().to_numpy()
    return result


def _interval_intersection(first: np.ndarray, second: np.ndarray) -> float:
    i = j = 0
    total = 0.0
    while i < len(first) and j < len(second):
        total += max(0.0, min(first[i, 1], second[j, 1]) - max(first[i, 0], second[j, 0]))
        if first[i, 1] <= second[j, 1]:
            i += 1
        else:
            j += 1
    return float(total)


def _event_agreement(
    electrode_metrics: pd.DataFrame,
    new_events: pd.DataFrame,
    reference_root: Path,
) -> pd.DataFrame:
    """Compare bycycle and eBOSC event-time masks without joining across epochs."""
    rows: list[dict[str, Any]] = []
    for subject_id, subject_metrics in electrode_metrics.groupby("subject_id", sort=False):
        reference_path = reference_root / "intermediate" / "episodes" / f"{subject_id}_bout_episodes.csv.gz"
        if not reference_path.exists():
            continue
        reference = pd.read_csv(reference_path)
        new_subject = new_events.loc[new_events["subject_id"].eq(subject_id)] if not new_events.empty else new_events
        for metric_row in subject_metrics.itertuples(index=False):
            keys = {
                "subject_id": subject_id,
                "group": metric_row.group,
                "electrode": metric_row.electrode,
                "band": metric_row.band,
            }
            first = reference.loc[
                reference["electrode"].eq(metric_row.electrode)
                & reference["band"].eq(metric_row.band)
            ]
            second = new_subject.loc[
                new_subject["electrode"].eq(metric_row.electrode)
                & new_subject["band"].eq(metric_row.band)
            ] if not new_subject.empty else new_subject
            intersection = first_duration = second_duration = 0.0
            epochs = sorted(set(first.get("epoch_index", pd.Series(dtype=int)).tolist()) | set(second.get("epoch_index", pd.Series(dtype=int)).tolist()))
            for epoch_index in epochs:
                first_intervals = first.loc[first["epoch_index"].eq(epoch_index), ["onset_s", "offset_s"]].sort_values("onset_s").to_numpy(float)
                second_intervals = second.loc[second["epoch_index"].eq(epoch_index), ["onset_s", "offset_s"]].sort_values("onset_s").to_numpy(float)
                first_duration += float(np.sum(first_intervals[:, 1] - first_intervals[:, 0])) if len(first_intervals) else 0.0
                second_duration += float(np.sum(second_intervals[:, 1] - second_intervals[:, 0])) if len(second_intervals) else 0.0
                intersection += _interval_intersection(first_intervals, second_intervals)
            denominator = first_duration + second_duration
            union = first_duration + second_duration - intersection
            rows.append({
                **keys,
                "ebosc_detected_duration_s": first_duration,
                "bycycle_detected_duration_s": second_duration,
                "intersection_duration_s": intersection,
                "dice": 2.0 * intersection / denominator if denominator > 0 else np.nan,
                "jaccard": intersection / union if union > 0 else np.nan,
                "both_empty": bool(denominator == 0.0),
            })
    return pd.DataFrame.from_records(rows)


def _metric_agreement(
    subject: pd.DataFrame,
    reference_root: Path,
    metrics: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    reference_path = reference_root / "metrics" / "subject_band_metrics.csv"
    if not reference_path.exists():
        return pd.DataFrame(), pd.DataFrame()
    reference = pd.read_csv(reference_path)
    keys = ["subject_id", "group", "band"]
    paired = reference[keys + metrics].merge(
        subject[keys + metrics], on=keys, suffixes=("_ebosc", "_bycycle"), validate="one_to_one"
    )
    rows = []
    for band in paired["band"].drop_duplicates():
        selected = paired.loc[paired["band"].eq(band)]
        for metric in metrics:
            values = selected[[f"{metric}_ebosc", f"{metric}_bycycle"]].dropna()
            if len(values) >= 3:
                result = spearmanr(values.iloc[:, 0], values.iloc[:, 1])
                rho, p_value = float(result.statistic), float(result.pvalue)
            else:
                rho = p_value = np.nan
            rows.append({"band": band, "metric": metric, "n_subjects": len(values), "spearman_rho": rho, "spearman_p_value": p_value})
    return paired, pd.DataFrame.from_records(rows)


def _collect_subject_results(
    iterator: Any,
    *,
    total: int,
    show_progress: bool,
    logger: logging.Logger,
) -> list[dict[str, Any]]:
    """Collect worker results with a live bar and log-file heartbeats."""
    results: list[dict[str, Any]] = []
    started = time.monotonic()
    heartbeat_step = max(1, int(np.ceil(total / 20.0)))
    progress = tqdm(
        iterator,
        total=total,
        disable=not show_progress,
        desc="independent bycycle bursts",
        unit="subject",
        dynamic_ncols=True,
        mininterval=0.5,
    )
    for completed, result in enumerate(progress, start=1):
        results.append(result)
        subject_id = str(result["input"]["subject_id"])
        if show_progress:
            progress.set_postfix_str(subject_id, refresh=False)
        elapsed = time.monotonic() - started
        remaining = elapsed / completed * (total - completed)
        logger.debug(
            "bycycle progress %d/%d | subject=%s | elapsed=%.1fs | eta=%.1fs",
            completed,
            total,
            subject_id,
            elapsed,
            remaining,
        )
        if not show_progress and (
            completed == 1 or completed % heartbeat_step == 0 or completed == total
        ):
            logger.info(
                "bycycle progress %d/%d (%.1f%%) | last=%s | ETA %.1f min",
                completed,
                total,
                100.0 * completed / total,
                subject_id,
                remaining / 60.0,
            )
    progress.close()
    return results


def run_analysis(
    config_path: str | Path = "bycycle_burst_analysis/config.json",
    *,
    subjects: list[str] | None = None,
    channels: list[str] | None = None,
    output_dir_override: str | Path | None = None,
    workers: int | None = None,
    overwrite: bool = False,
    show_progress: bool = True,
) -> dict[str, Any]:
    """Run independent bycycle burst detection and eBOSC agreement QC."""
    config_path = Path(config_path)
    config = load_analysis_config(config_path)
    if output_dir_override is not None:
        config["output_dir"] = str(output_dir_override)
    if workers is not None:
        config["detector"]["workers"] = int(workers)
    output = Path(config["output_dir"])
    result_path = output / "metrics" / "subject_electrode_band_metrics.csv"
    if result_path.exists() and not overwrite:
        raise FileExistsError(f"bycycle burst outputs already exist at {result_path}; rerun with --overwrite")
    if overwrite:
        remove_retired_band_outputs(output)
    logger = _logger(output, overwrite)
    participants = _participants(Path(config["input"]["participants_file"]))
    files = _epoch_files(Path(config["input"]["epochs_dir"]), str(config["input"]["epoch_glob"]))
    expected_subjects = participants["participant_id"].astype(str).tolist()
    if subjects:
        requested = list(dict.fromkeys(subjects))
        unknown = sorted(set(requested) - set(expected_subjects))
        if unknown:
            raise ValueError(f"Unknown participant IDs: {unknown}")
        expected_subjects = requested
        participants = participants.loc[participants["participant_id"].astype(str).isin(requested)].copy()
    missing = sorted(set(expected_subjects) - set(files))
    if missing:
        raise FileNotFoundError(f"Missing cleaned epoch files for: {missing}")
    inventory: dict[str, list[str]] = {}
    first_info = None
    for subject_id in expected_subjects:
        epochs = mne.read_epochs(files[subject_id], preload=False, verbose="ERROR")
        picks = mne.pick_types(epochs.info, eeg=True, exclude=[])
        inventory[subject_id] = [epochs.ch_names[pick] for pick in picks]
        if first_info is None:
            first_info = epochs.info.copy()
    common_channels, electrode_union = ordered_channel_inventory(inventory)
    if channels:
        requested_channels = list(dict.fromkeys(channels))
        unavailable = sorted(set(requested_channels) - set(common_channels))
        if unavailable:
            raise ValueError(f"Requested channels are not shared by every subject: {unavailable}")
        common_channels = [channel for channel in common_channels if channel in requested_channels]
    if not common_channels:
        raise ValueError("No electrodes shared by every analyzed subject")
    assert first_info is not None
    common_info = mne.pick_info(
        first_info, [first_info["ch_names"].index(channel) for channel in common_channels], copy=True
    )
    common_info["bads"] = []
    group_lookup = participants.set_index("participant_id")["GROUP"].astype(str).to_dict()
    bands = {name: tuple(float(value) for value in limits) for name, limits in config["bands"].items()}
    tasks = [{
        "subject_id": subject_id,
        "group": group_lookup[subject_id],
        "epoch_file": str(files[subject_id]),
        "channels": common_channels,
        "bands": bands,
        "settings": config["detector"],
        "output_dir": str(output),
    } for subject_id in expected_subjects]
    logger.info(
        "Starting independent bycycle detector | subjects=%d | shared_electrodes=%d | workers=%d",
        len(tasks), len(common_channels), int(config["detector"]["workers"]),
    )
    worker_count = min(int(config["detector"]["workers"]), len(tasks))
    if worker_count == 1:
        iterator = map(_process_subject, tasks)
        results = _collect_subject_results(
            iterator,
            total=len(tasks),
            show_progress=show_progress,
            logger=logger,
        )
    else:
        try:
            with ProcessPoolExecutor(max_workers=worker_count) as executor:
                futures = [executor.submit(_process_subject, task) for task in tasks]
                iterator = (future.result() for future in as_completed(futures))
                results = _collect_subject_results(
                    iterator,
                    total=len(tasks),
                    show_progress=show_progress,
                    logger=logger,
                )
        except PermissionError as error:
            logger.warning(
                "Process workers are unavailable on this system (%s); falling back to one worker",
                error,
            )
            results = _collect_subject_results(
                map(_process_subject, tasks),
                total=len(tasks),
                show_progress=show_progress,
                logger=logger,
            )
    subject_order = {subject_id: index for index, subject_id in enumerate(expected_subjects)}
    results.sort(key=lambda result: subject_order[str(result["input"]["subject_id"])])
    electrode = pd.DataFrame.from_records([row for result in results for row in result["metrics"]])
    subject = _subject_means(electrode)
    inputs = pd.DataFrame.from_records([result["input"] for result in results])
    all_event_paths = [output / "intermediate" / "episodes" / f"{subject_id}_bycycle_bouts.csv.gz" for subject_id in expected_subjects]
    event_tables = []
    for path in all_event_paths:
        if not path.exists() or path.stat().st_size <= 30:
            continue
        try:
            event_tables.append(pd.read_csv(path))
        except pd.errors.EmptyDataError:
            continue
    events = pd.concat([table for table in event_tables if len(table)], ignore_index=True) if any(len(table) for table in event_tables) else pd.DataFrame()

    statistics = config["statistics"]
    inferential_bands = [band for band in bands if band not in set(statistics.get("exclude_bands", []))]
    inferential = electrode.loc[electrode["band"].isin(inferential_bands)].copy()
    inferential_metrics = list(statistics["inferential_metrics"])
    subject_statistics, electrode_statistics = compute_group_statistics(
        inferential,
        participants,
        metrics=inferential_metrics,
        strata=("band",),
        domain="independent_bycycle_bursts",
        subject_aggregation=str(statistics["subject_aggregation"]),
        confidence_level=float(statistics["confidence_level"]),
        fdr_alpha=float(statistics["fdr_alpha"]),
    )
    reference_root = Path(config["input"]["reference_ebosc_output_dir"])
    event_agreement = _event_agreement(electrode, events, reference_root)
    paired_metrics, metric_agreement = _metric_agreement(subject, reference_root, inferential_metrics)
    group_summary = subject.groupby(["band", "group"], sort=False)[
        ["n_bouts", *METRICS]
    ].agg(["count", "mean", "std", "median"]).reset_index()
    group_summary.columns = ["_".join(part for part in column if part) if isinstance(column, tuple) else column for column in group_summary.columns]

    metrics_dir = output / "metrics"
    _write_csv(inputs, metrics_dir / "analyzed_inputs.csv")
    _write_csv(electrode, metrics_dir / "subject_electrode_band_metrics.csv")
    _write_csv(subject, metrics_dir / "subject_band_metrics.csv")
    _write_csv(group_summary, metrics_dir / "group_summary.csv")
    _write_csv(subject_statistics, metrics_dir / "group_subject_statistics.csv")
    _write_csv(electrode_statistics, metrics_dir / "group_electrode_statistics.csv")
    _write_csv(event_agreement, metrics_dir / "detector_event_agreement.csv")
    _write_csv(paired_metrics, metrics_dir / "detector_paired_subject_metrics.csv")
    _write_csv(metric_agreement, metrics_dir / "detector_metric_agreement.csv")
    electrode_payload = {
        "common_electrodes": common_channels,
        "n_common_electrodes": len(common_channels),
        "electrode_union": electrode_union,
        "n_electrode_union": len(electrode_union),
        "policy": "Only electrodes present in every analyzed subject are used.",
    }
    (metrics_dir / "electrode_sets.json").write_text(json.dumps(electrode_payload, indent=2) + "\n", encoding="utf-8")

    group_order = [group for group in config["plots"]["group_order"] if group in set(subject["group"])]
    colors = {group: str(config["plots"]["group_colors"].get(group, "0.4")) for group in group_order}
    labels = {band: f"{config['plots']['band_display_names'].get(band, band)}\n{limits[0]:g}–{limits[1]:g} Hz" for band, limits in bands.items()}
    dpi = int(config["plots"]["dpi"])
    figures = output / "figures"
    example = results[0]["example"]
    if example is not None:
        plot_detection_example(
            example["signal_uv"], example["masks"], sfreq=example["sfreq"], bands=bands,
            band_labels=labels, subject_id=example["subject_id"], electrode=example["electrode"],
            epoch_index=example["epoch_index"],
            path=figures / "qc" / "independent_detection_example.png", dpi=dpi,
        )
    plot_group_metric_violins(
        subject, metrics=inferential_metrics, bands=inferential_bands, group_order=group_order,
        colors=colors, band_labels=labels, path=figures / "group_primary_metrics.png", dpi=dpi,
    )
    subject_violin_figures = plot_subject_average_violins(
        subject,
        metrics=list(METRICS),
        bands=list(bands),
        group_order=group_order,
        colors=colors,
        band_labels=labels,
        output_dir=figures / "group_comparisons",
        dpi=dpi,
    )
    plot_detection_coverage(
        subject, bands=list(bands), group_order=group_order, colors=colors, band_labels=labels,
        path=figures / "qc" / "detection_coverage.png", dpi=dpi,
    )
    if not event_agreement.empty:
        plot_event_agreement(
            event_agreement, bands=inferential_bands, group_order=group_order, colors=colors,
            band_labels=labels, path=figures / "agreement" / "event_mask_dice.png", dpi=dpi,
        )
    if not paired_metrics.empty:
        plot_metric_agreement(
            paired_metrics, metrics=inferential_metrics, bands=inferential_bands, band_labels=labels,
            path=figures / "agreement" / "subject_metric_scatter.png", dpi=dpi,
        )
    statistical_figures = plot_electrode_group_statistics(
        electrode_statistics, common_info, strata=("band",),
        output_dir=figures / "group_statistics", dpi=dpi,
        stratum_labels={band: labels[band] for band in inferential_bands},
    )
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config_file": str(config_path.resolve()),
        "analysis_config": config,
        "software": {
            "python": platform.python_version(), "mne": mne.__version__, "numpy": np.__version__,
            "pandas": pd.__version__, "scipy": scipy.__version__, "matplotlib": matplotlib.__version__,
            "bycycle": version("bycycle"),
        },
        "n_subjects": len(expected_subjects),
        "n_shared_electrodes": len(common_channels),
        "bands": list(bands),
        "independence_policy": "No eBOSC power mask or specparam threshold enters bycycle detection.",
        "epoch_boundary_policy": "Accepted epochs are analyzed separately; no cycle or burst crosses a boundary.",
        "edge_policy": f"Cycles must lie within a {config['detector']['edge_padding_seconds']:g}-second interior margin.",
        "statistics_policy": {
            "primary_unit": "subject",
            "inferential_metrics": inferential_metrics,
            "excluded_bands": list(statistics.get("exclude_bands", [])),
            "full_cohort_model": "OLS adjusted for age and sex with HC3 robust SE",
            "matched_cohort_model": "paired t test by match_pair_id",
            "fdr_scope": "BH across every metric-by-canonical-band test in this detector domain",
        },
        "reference_detector": str(reference_root),
        "n_event_agreement_rows": len(event_agreement),
        "n_statistical_figures": len(statistical_figures),
        "n_subject_average_violin_figures": len(subject_violin_figures),
        "subject_average_violin_policy": (
            "Each plotted point is one subject after arithmetic averaging across "
            "all cohort-shared electrodes and the four canonical bands."
        ),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    logger.info("Independent bycycle burst analysis completed | output=%s", output)
    return manifest
