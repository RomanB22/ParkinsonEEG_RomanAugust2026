"""End-to-end ordinal analysis of cleaned EEG epochs."""

from __future__ import annotations

import json
import logging
import platform
import re
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any

from src.runtime import configure_runtime
from src.dataset import ordered_channel_inventory

configure_runtime()

import matplotlib
import mne
import numpy as np
import pandas as pd
import scipy
from tqdm.auto import tqdm

from .metrics import (
    CORE_METRICS,
    METRICS,
    RENYI_ALPHAS,
    RENYI_ALPHA_METRICS,
    analyze_epoch_data,
    band_subject_electrode_means,
    filter_epoch_data,
    subject_electrode_means,
)
from .plots import (
    band_metric_color_limits,
    electrode_metric_zscores,
    group_mean_symmetric_color_limits,
    metric_color_limits,
    plot_electrode_plane_pages,
    plot_electrode_violins,
    plot_group_band_topomaps,
    plot_group_standardized_topomaps,
    plot_group_topomaps,
    plot_subject_average_planes,
    plot_subject_average_violins,
    plot_subject_band_topomaps,
    plot_subject_topomaps,
)


SUBJECT_PATTERN = re.compile(r"(sub-\d+)")


def load_analysis_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as stream:
        config = json.load(stream)
    required = {"input", "output_dir", "ordinal", "bands", "band_filter", "plots"}
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Missing ordinal-analysis config sections: {missing}")
    ordinal = config["ordinal"]
    dx = int(ordinal["embedding_dimension"])
    tau = int(ordinal["delay_samples"])
    if not 2 <= dx <= 7:
        raise ValueError("ordinal.embedding_dimension must be between 2 and 7")
    if tau < 1:
        raise ValueError("ordinal.delay_samples must be at least 1")
    if ordinal.get("tie_precision") is not None:
        raise ValueError(
            "This analysis is configured to retain every signal decimal; "
            "ordinal.tie_precision must be null"
        )
    bands = config["bands"]
    if not isinstance(bands, dict) or not bands:
        raise ValueError("bands must be a non-empty mapping")
    for name, limits in bands.items():
        if not isinstance(name, str) or not name:
            raise ValueError("Every band must have a non-empty string name")
        if not isinstance(limits, list) or len(limits) != 2:
            raise ValueError(f"bands.{name} must contain [low_hz, high_hz]")
        low_hz, high_hz = (float(value) for value in limits)
        if not 0.0 < low_hz < high_hz:
            raise ValueError(f"bands.{name} must satisfy 0 < low_hz < high_hz")
    filter_config = config["band_filter"]
    if filter_config.get("method") != "butterworth_sos_sosfiltfilt":
        raise ValueError(
            "band_filter.method must be 'butterworth_sos_sosfiltfilt'"
        )
    if filter_config.get("phase") != "zero_phase":
        raise ValueError("band_filter.phase must be 'zero_phase'")
    if (
        filter_config.get("epoch_boundary_policy")
        != "filter_each_accepted_epoch_independently"
    ):
        raise ValueError(
            "band_filter.epoch_boundary_policy must be "
            "'filter_each_accepted_epoch_independently'"
        )
    filter_order = filter_config.get("order")
    if not isinstance(filter_order, int) or filter_order < 1:
        raise ValueError("band_filter.order must be a positive integer")
    return config


def _participants(path: Path) -> pd.DataFrame:
    separator = "\t" if path.suffix.lower() == ".tsv" else ","
    table = pd.read_csv(path, sep=separator)
    required = {"participant_id", "GROUP"}
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"Participant table is missing columns: {missing}")
    if table["participant_id"].duplicated().any():
        raise ValueError("Participant IDs must be unique")
    return table


def _epoch_files(directory: Path, pattern: str) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for path in sorted(directory.glob(pattern)):
        match = SUBJECT_PATTERN.search(path.name)
        if match is None:
            continue
        subject_id = match.group(1)
        if subject_id in files:
            raise ValueError(f"Multiple epoch files found for {subject_id}")
        files[subject_id] = path
    return files


def _configure_logger(output_dir: Path, overwrite: bool) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "ordinal_analysis.log"
    mode = "w" if overwrite else "a"
    logger = logging.getLogger("ordinal_analysis")
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    file_handler = logging.FileHandler(log_path, mode=mode)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def _group_summary(table: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    grouped = table.groupby(by, sort=True)
    rows = []
    for keys, selected in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(by, keys))
        row["n_subjects"] = int(selected["subject_id"].nunique())
        for metric in METRICS:
            values = selected[metric].to_numpy(dtype=float)
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else np.nan
            row[f"{metric}_median"] = float(np.median(values))
        rows.append(row)
    return pd.DataFrame.from_records(rows)


def _write_csv(table: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False, float_format="%.17g")


def run_analysis(
    config_path: str | Path,
    *,
    subjects: list[str] | None = None,
    overwrite: bool = False,
    show_progress: bool = True,
    generate_figures: bool = True,
) -> dict[str, Any]:
    config_path = Path(config_path)
    config = load_analysis_config(config_path)
    input_config = config["input"]
    output_dir = Path(config["output_dir"])
    result_path = output_dir / "metrics" / "electrode_metrics.csv"
    if result_path.exists() and not overwrite:
        raise FileExistsError(
            f"Ordinal outputs already exist at {result_path}; rerun with --overwrite"
        )
    logger = _configure_logger(output_dir, overwrite)

    participant_table = _participants(Path(input_config["participants_file"]))
    files = _epoch_files(
        Path(input_config["epochs_dir"]), str(input_config["epoch_glob"])
    )
    expected_subjects = participant_table["participant_id"].astype(str).tolist()
    if subjects:
        requested = list(dict.fromkeys(subjects))
        unknown = sorted(set(requested) - set(expected_subjects))
        if unknown:
            raise ValueError(f"Unknown participant IDs: {unknown}")
        expected_subjects = requested
    missing_files = sorted(set(expected_subjects) - set(files))
    if missing_files:
        raise FileNotFoundError(f"Missing cleaned epoch files for: {missing_files}")

    available_channels: dict[str, list[str]] = {}
    for subject_id in expected_subjects:
        epochs = mne.read_epochs(files[subject_id], preload=False, verbose="ERROR")
        eeg_picks = mne.pick_types(epochs.info, eeg=True, exclude=[])
        available_channels[subject_id] = [
            epochs.ch_names[pick] for pick in eeg_picks
        ]
    common_channels, electrode_union = ordered_channel_inventory(available_channels)

    groups = participant_table.set_index("participant_id")["GROUP"].astype(str).to_dict()
    dx = int(config["ordinal"]["embedding_dimension"])
    tau = int(config["ordinal"]["delay_samples"])
    tie_precision = config["ordinal"].get("tie_precision")
    bands = {
        str(name): (float(limits[0]), float(limits[1]))
        for name, limits in config["bands"].items()
    }
    filter_order = int(config["band_filter"]["order"])
    logger.info(
        "Starting ordinal analysis | subjects=%d | dx=%d | tau=%d | "
        "tie_precision=%s | bands=%s | filter_order=%d | shared_electrodes=%d",
        len(expected_subjects),
        dx,
        tau,
        tie_precision,
        ",".join(bands),
        filter_order,
        len(common_channels),
    )

    metric_tables = []
    band_metric_tables = []
    subject_infos: dict[str, mne.Info] = {}
    input_rows = []
    analysis_progress = tqdm(
        total=len(expected_subjects) * (1 + len(bands)),
        desc=f"Ordinal metrics (d={dx})",
        unit="stage",
        dynamic_ncols=True,
        disable=not show_progress,
    )
    for index, subject_id in enumerate(expected_subjects, start=1):
        path = files[subject_id]
        analysis_progress.set_postfix_str(f"{subject_id} | loading", refresh=True)
        logger.debug("[%d/%d] %s | %s", index, len(expected_subjects), subject_id, path)
        epochs = mne.read_epochs(path, preload=True, verbose="ERROR")
        picks = [epochs.ch_names.index(channel) for channel in common_channels]
        data = epochs.get_data(picks=picks, copy=True)
        channel_names = list(common_channels)
        info = mne.pick_info(epochs.info, picks, copy=True)
        info["bads"] = []
        subject_infos[subject_id] = info
        analysis_progress.set_postfix_str(f"{subject_id} | broadband", refresh=True)
        subject_metrics = analyze_epoch_data(
            data,
            channel_names,
            subject_id=subject_id,
            group=groups[subject_id],
            sfreq=float(epochs.info["sfreq"]),
            dx=dx,
            tau=tau,
            tie_precision=tie_precision,
        )
        metric_tables.append(subject_metrics)
        analysis_progress.update()
        sfreq = float(epochs.info["sfreq"])
        for band, (low_hz, high_hz) in bands.items():
            analysis_progress.set_postfix_str(f"{subject_id} | {band}", refresh=True)
            logger.debug(
                "[%d/%d] %s | band=%s | %.3g-%.3g Hz",
                index,
                len(expected_subjects),
                subject_id,
                band,
                low_hz,
                high_hz,
            )
            filtered = filter_epoch_data(
                data,
                sfreq=sfreq,
                low_hz=low_hz,
                high_hz=high_hz,
                order=filter_order,
            )
            band_metrics = analyze_epoch_data(
                filtered,
                channel_names,
                subject_id=subject_id,
                group=groups[subject_id],
                sfreq=sfreq,
                dx=dx,
                tau=tau,
                tie_precision=tie_precision,
            )
            band_metrics.insert(2, "band", band)
            band_metrics.insert(3, "band_low_hz", low_hz)
            band_metrics.insert(4, "band_high_hz", high_hz)
            band_metrics["band_filter_method"] = "butterworth_sos_sosfiltfilt"
            band_metrics["band_filter_order"] = filter_order
            band_metrics["band_filter_phase"] = "zero_phase"
            band_metric_tables.append(band_metrics)
            analysis_progress.update()
        input_rows.append(
            {
                "subject_id": subject_id,
                "group": groups[subject_id],
                "epoch_file": str(path.resolve()),
                "n_epochs": len(epochs),
                "n_electrodes": len(channel_names),
                "n_available_electrodes": len(available_channels[subject_id]),
                "sampling_frequency_hz": float(epochs.info["sfreq"]),
            }
        )
    analysis_progress.close()

    electrode_metrics = pd.concat(metric_tables, ignore_index=True)
    band_electrode_metrics = pd.concat(band_metric_tables, ignore_index=True)
    subject_means = subject_electrode_means(electrode_metrics)
    band_subject_means = band_subject_electrode_means(band_electrode_metrics)
    electrode_summary = _group_summary(electrode_metrics, ["group", "electrode"])
    subject_summary = _group_summary(subject_means, ["group"])
    band_electrode_summary = _group_summary(
        band_electrode_metrics,
        ["band", "band_low_hz", "band_high_hz", "group", "electrode"],
    )
    band_subject_summary = _group_summary(
        band_subject_means,
        ["band", "band_low_hz", "band_high_hz", "group"],
    )
    input_table = pd.DataFrame.from_records(input_rows)

    metrics_dir = output_dir / "metrics"
    _write_csv(electrode_metrics, metrics_dir / "electrode_metrics.csv")
    _write_csv(subject_means, metrics_dir / "subject_electrode_mean_metrics.csv")
    _write_csv(electrode_summary, metrics_dir / "group_electrode_summary.csv")
    _write_csv(subject_summary, metrics_dir / "group_subject_mean_summary.csv")
    _write_csv(band_electrode_metrics, metrics_dir / "band_electrode_metrics.csv")
    _write_csv(
        band_subject_means,
        metrics_dir / "band_subject_electrode_mean_metrics.csv",
    )
    _write_csv(
        band_electrode_summary,
        metrics_dir / "group_band_electrode_summary.csv",
    )
    _write_csv(
        band_subject_summary,
        metrics_dir / "group_band_subject_mean_summary.csv",
    )
    _write_csv(input_table, metrics_dir / "analyzed_inputs.csv")

    common_payload = {
        "common_electrodes": common_channels,
        "n_common_electrodes": len(common_channels),
        "electrode_union": electrode_union,
        "n_electrode_union": len(electrode_union),
        "analysis_electrode_policy": (
            "Every metric, aggregation, table, and figure uses only electrodes "
            "present in every analyzed subject."
        ),
    }
    (metrics_dir / "electrode_sets.json").write_text(
        json.dumps(common_payload, indent=2) + "\n", encoding="utf-8"
    )

    if not generate_figures:
        manifest = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "config_file": str(config_path.resolve()),
            "analysis_config": config,
            "software": {
                "python": platform.python_version(),
                "ordpy": version("ordpy"),
                "mne": mne.__version__,
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "matplotlib": matplotlib.__version__,
                "scipy": scipy.__version__,
            },
            "n_subjects": len(expected_subjects),
            "group_counts": input_table["group"].value_counts().to_dict(),
            "n_electrode_rows": len(electrode_metrics),
            "n_band_electrode_rows": len(band_electrode_metrics),
            "n_band_subject_rows": len(band_subject_means),
            "n_common_electrodes": len(common_channels),
            "n_electrode_union": len(electrode_union),
            "analysis_electrode_policy": (
                "Only electrodes present in every analyzed subject are loaded for ordinal "
                "metrics and included in every metric table."
            ),
            "figures_generated": False,
            "figure_policy": (
                "Figures were intentionally skipped for this parameter-sensitivity input; "
                "the quantitative-behavioral pipeline generates the inferential figures."
            ),
            "tie_handling": (
                "tie_precision=None: ordinal ranking uses original float64 samples with no "
                "decimal rounding and no artificial jitter."
            ),
            "epoch_pooling": (
                "Ordinal pattern counts are pooled across accepted epochs; patterns crossing "
                "epoch boundaries are excluded."
            ),
            "band_filtering": (
                "Each accepted epoch and electrode is independently band-pass filtered with "
                f"a {filter_order}th-order Butterworth SOS and scipy.signal.sosfiltfilt."
            ),
        }
        (output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        logger.info("Ordinal metrics completed without figures | output=%s", output_dir)
        return manifest

    electrode_order = list(common_channels)
    common_info = next(iter(subject_infos.values())).copy()

    configured_groups = [str(group) for group in config["plots"]["group_order"]]
    present_groups = set(electrode_metrics["group"].unique())
    group_order = [group for group in configured_groups if group in present_groups]
    group_order.extend(sorted(present_groups - set(group_order)))
    configured_colors = config["plots"]["group_colors"]
    fallback_colors = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")
    colors = {
        group: str(configured_colors.get(group, fallback_colors[index % len(fallback_colors)]))
        for index, group in enumerate(group_order)
    }
    dpi = int(config["plots"]["dpi"])
    channels_per_page = int(config["plots"]["channels_per_plane_page"])
    figures_dir = output_dir / "figures"

    logger.info("Creating group violin plots")
    plot_electrode_violins(
        electrode_metrics,
        electrode_order,
        group_order,
        colors,
        figures_dir / "violins",
        dpi,
    )
    plot_subject_average_violins(
        subject_means,
        group_order,
        colors,
        figures_dir / "violins" / "subject_electrode_mean_violins.png",
        dpi,
    )
    logger.info("Creating Shannon, Fisher, and Rényi entropy-complexity planes")
    plot_electrode_plane_pages(
        electrode_metrics,
        electrode_order,
        group_order,
        colors,
        figures_dir / "planes",
        dpi,
        channels_per_page,
    )
    plot_subject_average_planes(
        subject_means,
        group_order,
        colors,
        figures_dir / "planes" / "subject_electrode_mean_hxc_hxf.png",
        dpi,
    )

    limits = metric_color_limits(electrode_metrics)
    logger.info("Creating %d subject topomap figures", len(subject_infos))
    plot_subject_topomaps(
        electrode_metrics,
        subject_infos,
        limits,
        figures_dir / "topomaps" / "subjects",
        dpi,
    )
    plot_group_topomaps(
        electrode_metrics,
        common_info,
        group_order,
        limits,
        figures_dir / "topomaps" / "group_mean_topomaps.png",
        dpi,
    )
    standardized_metrics = electrode_metric_zscores(electrode_metrics)
    standardized_limits = group_mean_symmetric_color_limits(
        standardized_metrics, common_channels
    )
    plot_group_standardized_topomaps(
        standardized_metrics,
        common_info,
        group_order,
        standardized_limits,
        figures_dir / "topomaps" / "group_mean_zscored_topomaps.png",
        dpi,
        "Broadband",
    )

    band_order = list(bands)
    configured_band_labels = config["plots"].get("band_display_names", {})
    band_labels = {
        band: str(configured_band_labels.get(band, band.replace("_", " ").title()))
        for band in band_order
    }
    logger.info(
        "Creating band-resolved Shannon, Fisher, and Rényi violins and planes"
    )
    for band in band_order:
        selected = band_electrode_metrics.loc[band_electrode_metrics["band"].eq(band)]
        selected_means = band_subject_means.loc[band_subject_means["band"].eq(band)]
        band_dir = figures_dir / "bands" / band
        label = f"{band_labels[band]} ({bands[band][0]:g}–{bands[band][1]:g} Hz)"
        plot_electrode_violins(
            selected,
            electrode_order,
            group_order,
            colors,
            band_dir / "violins",
            dpi,
            analysis_label=label,
        )
        plot_subject_average_violins(
            selected_means,
            group_order,
            colors,
            band_dir / "violins" / "subject_electrode_mean_violins.png",
            dpi,
            analysis_label=label,
        )
        plot_electrode_plane_pages(
            selected,
            electrode_order,
            group_order,
            colors,
            band_dir / "planes",
            dpi,
            channels_per_page,
            analysis_label=label,
        )
        plot_subject_average_planes(
            selected_means,
            group_order,
            colors,
            band_dir / "planes" / "subject_electrode_mean_hxc_hxf.png",
            dpi,
            analysis_label=label,
        )

    band_limits = band_metric_color_limits(band_electrode_metrics, band_order)
    logger.info("Creating %d band-resolved subject topomap figures", len(subject_infos))
    plot_subject_band_topomaps(
        band_electrode_metrics,
        subject_infos,
        band_order,
        band_labels,
        band_limits,
        figures_dir / "bands" / "topomaps" / "subjects",
        dpi,
    )
    plot_group_band_topomaps(
        band_electrode_metrics,
        common_info,
        group_order,
        band_order,
        band_labels,
        band_limits,
        figures_dir / "bands" / "topomaps" / "group_means",
        dpi,
    )
    standardized_band_metrics = electrode_metric_zscores(
        band_electrode_metrics, strata=("band",)
    )
    standardized_band_limits = {
        band: group_mean_symmetric_color_limits(
            standardized_band_metrics.loc[
                standardized_band_metrics["band"].eq(band)
            ],
            common_channels,
        )
        for band in band_order
    }
    logger.info("Creating electrode-wise z-scored group band topomaps")
    for band in band_order:
        standardized_label = band_labels[band]
        if "hz" not in standardized_label.lower():
            standardized_label = (
                f"{standardized_label} ({bands[band][0]:g}–{bands[band][1]:g} Hz)"
            )
        plot_group_standardized_topomaps(
            standardized_band_metrics.loc[
                standardized_band_metrics["band"].eq(band)
            ],
            common_info,
            group_order,
            standardized_band_limits[band],
            figures_dir
            / "bands"
            / "topomaps"
            / "group_means_zscored"
            / f"{band}_group_mean_zscored_topomaps.png",
            dpi,
            standardized_label,
        )

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config_file": str(config_path.resolve()),
        "analysis_config": config,
        "software": {
            "python": platform.python_version(),
            "ordpy": version("ordpy"),
            "mne": mne.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "matplotlib": matplotlib.__version__,
            "scipy": scipy.__version__,
        },
        "n_subjects": len(expected_subjects),
        "group_counts": input_table["group"].value_counts().to_dict(),
        "n_electrode_rows": len(electrode_metrics),
        "n_band_electrode_rows": len(band_electrode_metrics),
        "n_band_subject_rows": len(band_subject_means),
        "n_common_electrodes": len(common_channels),
        "n_electrode_union": len(electrode_union),
        "figures_generated": True,
        "analysis_electrode_policy": (
            "Only electrodes present in every analyzed subject are loaded for ordinal "
            "metrics and included in any table, aggregation, or figure."
        ),
        "tie_handling": (
            "tie_precision=None: ordinal ranking uses original float64 samples with no decimal "
            "rounding and no artificial jitter; exact tied embedding windows are counted in "
            "electrode_metrics.csv."
        ),
        "epoch_pooling": (
            "Ordinal pattern counts are pooled across accepted epochs. Patterns crossing epoch "
            "boundaries are excluded before Shannon, Fisher, and Rényi quantities are "
            "calculated."
        ),
        "renyi": {
            "function": "ordpy.renyi_complexity_entropy",
            "alphas": [float(alpha) for alpha in RENYI_ALPHAS],
            "probability_input": True,
            "outputs": {
                f"alpha_{alpha:g}": {
                    "entropy_column": entropy_metric,
                    "complexity_column": complexity_metric,
                }
                for alpha, entropy_metric, complexity_metric in RENYI_ALPHA_METRICS
            },
        },
        "band_filtering": (
            "Each accepted epoch and electrode is independently band-pass filtered with a "
            f"{filter_order}th-order Butterworth SOS and scipy.signal.sosfiltfilt. Filtering "
            "is zero-phase and never crosses epoch boundaries or rejected-data gaps. Ordinal "
            "patterns are then pooled across epochs with boundary-crossing embeddings excluded."
        ),
        "subject_average_definition": (
            "Arithmetic mean of every subject's electrode-level Shannon, Fisher, and Rényi "
            "quantities across the electrodes shared by every analyzed subject; the average-"
            "referenced EEG waveform is not averaged across channels."
        ),
        "topomap_scale_limits": {
            metric: [float(value) for value in limits[metric]]
            for metric in CORE_METRICS
        },
        "electrode_zscore_topomap_policy": (
            "For each metric, values are z-scored across all subjects pooled across groups "
            "within each electrode (and within each band for band-resolved maps), using "
            "population standard deviation (ddof=0). Constant combinations map to zero. "
            "Group means are plotted on shared symmetric zero-centered limits."
        ),
        "topomap_zscore_scale_limits": {
            metric: [float(value) for value in standardized_limits[metric]]
            for metric in CORE_METRICS
        },
        "band_topomap_scale_limits": {
            band: {
                metric: [float(value) for value in band_limits[band][metric]]
                for metric in CORE_METRICS
            }
            for band in band_order
        },
        "band_topomap_zscore_scale_limits": {
            band: {
                metric: [float(value) for value in standardized_band_limits[band][metric]]
                for metric in CORE_METRICS
            }
            for band in band_order
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    logger.info("Ordinal analysis completed | output=%s", output_dir)
    return manifest
