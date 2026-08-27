"""End-to-end eBOSC bout detection and within-bout ordinal analysis."""

from __future__ import annotations

import json
import logging
import math
import platform
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any

from core.runtime import configure_runtime

configure_runtime()

import matplotlib
import mne
import numpy as np
import pandas as pd
import scipy
from tqdm.auto import tqdm

from analyses.ordinal.metrics import filter_epoch_data
from analyses.scale_free.metrics import (
    detect_frequency_episodes,
    ebosc_wavelet_power,
)
from core.cache import replace_with_relative_symlink, same_json_settings
from core.analysis_io import discover_epoch_files as _epoch_files
from core.analysis_io import load_participants as _participants
from core.analysis_logging import configure_analysis_logger
from core.dataset import ordered_channel_inventory
from core.group_statistics import compute_group_statistics
from core.group_statistics_plots import plot_electrode_group_statistics
from core.output_cleanup import remove_retired_band_outputs

from .metrics import (
    METRICS,
    analyze_bout_segments,
    ordinal_patterns,
    validate_ordinal_parameters,
)
from .plots import (
    plot_bout_diagnostics,
    plot_detection_example,
    plot_electrode_violins,
    plot_group_topomaps,
    plot_ordinal_example,
    plot_ordinal_planes,
    plot_subject_metric_violins,
    plot_subject_topomaps,
)


GROUP_METRICS = (
    *METRICS,
    "oscillatory_occupancy",
    "bouts_per_minute",
    "bout_duration_mean_s",
    "bout_duration_median_s",
)


def load_analysis_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as stream:
        config = json.load(stream)
    required = {
        "input",
        "output_dir",
        "bands",
        "psd",
        "specparam",
        "ebosc",
        "ordinal",
        "band_filter",
        "cache",
        "statistics",
        "plots",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Missing bout-analysis config sections: {missing}")
    if not config["input"].get("scale_free_output_dir"):
        raise ValueError("input.scale_free_output_dir is required")

    bands = config["bands"]
    if not isinstance(bands, dict) or not bands:
        raise ValueError("bands must be a non-empty mapping")
    for band, limits in bands.items():
        if not isinstance(limits, list) or len(limits) != 2:
            raise ValueError(f"bands.{band} must contain [low_hz, high_hz]")
        low_hz, high_hz = (float(value) for value in limits)
        if not 0.0 < low_hz < high_hz:
            raise ValueError(f"Invalid frequency limits for {band}")

    psd_min = float(config["psd"]["fmin_hz"])
    psd_max = float(config["psd"]["fmax_hz"])
    fit_range = [float(value) for value in config["specparam"]["frequency_range_hz"]]
    if not 0.0 <= psd_min < psd_max:
        raise ValueError("psd requires 0 <= fmin_hz < fmax_hz")
    if (
        len(fit_range) != 2
        or not psd_min <= fit_range[0] < fit_range[1] <= psd_max
    ):
        raise ValueError(
            "specparam.frequency_range_hz must be contained within the PSD range"
        )
    specparam = config["specparam"]
    if str(specparam.get("aperiodic_mode")) != "best_bic":
        raise ValueError("specparam.aperiodic_mode must be 'best_bic'")
    if specparam.get("aperiodic_modes") != ["fixed", "knee"]:
        raise ValueError("specparam.aperiodic_modes must be ['fixed', 'knee']")
    if str(specparam.get("model_selection_criterion")) != "bic":
        raise ValueError("specparam.model_selection_criterion must be 'bic'")
    if fit_range != [4.0, 50.0]:
        raise ValueError("Fixed and knee bout-threshold models must use 4–50 Hz")
    if float(specparam.get("knee_frequency_outlier_z_threshold", 0.0)) != 2.0:
        raise ValueError("The within-subject knee-frequency outlier threshold must be 2 SD")

    ebosc = config["ebosc"]
    frequency_min = float(ebosc["frequency_min_hz"])
    frequency_max = float(ebosc["frequency_max_hz"])
    frequency_step = float(ebosc["frequency_step_hz"])
    if not 0.0 < frequency_min < frequency_max or frequency_step <= 0.0:
        raise ValueError("Invalid eBOSC frequency grid")
    if any(
        float(limits[0]) < frequency_min or float(limits[1]) > frequency_max
        for limits in bands.values()
    ):
        raise ValueError("Every band must lie within the eBOSC frequency grid")
    if not 0.0 < float(ebosc["power_percentile"]) < 1.0:
        raise ValueError("ebosc.power_percentile must be between zero and one")
    if float(ebosc["minimum_cycles"]) <= 0.0:
        raise ValueError("ebosc.minimum_cycles must be positive")
    if float(ebosc["edge_padding_seconds"]) < 0.0:
        raise ValueError("ebosc.edge_padding_seconds cannot be negative")

    ordinal = config["ordinal"]
    validate_ordinal_parameters(
        int(ordinal["embedding_dimension"]), int(ordinal["delay_samples"])
    )
    if not 3 <= int(ordinal["embedding_dimension"]) <= 6:
        raise ValueError("Within-bout embedding_dimension must be between 3 and 6")
    if int(ordinal["delay_samples"]) != 1:
        raise ValueError("Within-bout ordinal analysis requires tau=1")
    if ordinal.get("tie_precision") is not None:
        raise ValueError("ordinal.tie_precision must be null to preserve full precision")
    if ordinal.get("pooling") != "pool_pattern_counts_without_crossing_bout_or_epoch_boundaries":
        raise ValueError("ordinal.pooling must preserve bout and epoch boundaries")

    band_filter = config["band_filter"]
    if band_filter.get("method") != "butterworth_sos_sosfiltfilt":
        raise ValueError("Unsupported band_filter.method")
    if band_filter.get("phase") != "zero_phase":
        raise ValueError("band_filter.phase must be zero_phase")
    if band_filter.get("boundary_policy") != "filter_each_accepted_epoch_before_slicing_bouts":
        raise ValueError("band_filter.boundary_policy must filter epochs before slicing bouts")
    if not isinstance(band_filter.get("order"), int) or int(band_filter["order"]) < 1:
        raise ValueError("band_filter.order must be a positive integer")
    cache = config["cache"]
    if cache.get("reuse_scale_free_detection") is not True:
        raise ValueError(
            "cache.reuse_scale_free_detection must be true; bout detection belongs "
            "to the upstream scale-free stage"
        )
    if not isinstance(cache.get("link_reused_episode_and_threshold_files"), bool):
        raise ValueError(
            "cache.link_reused_episode_and_threshold_files must be boolean"
        )
    if int(config["plots"].get("dpi", 150)) < 50:
        raise ValueError("plots.dpi must be at least 50")
    statistics = config["statistics"]
    if not 0.0 < float(statistics["fdr_alpha"]) < 1.0:
        raise ValueError("statistics.fdr_alpha must be between zero and one")
    if not 0.0 < float(statistics["confidence_level"]) < 1.0:
        raise ValueError("statistics.confidence_level must be between zero and one")
    if statistics.get("subject_aggregation") != "mean":
        raise ValueError("Bout statistics.subject_aggregation must be mean")
    unknown_exclusions = sorted(set(statistics.get("exclude_bands", [])) - set(bands))
    if unknown_exclusions:
        raise ValueError(f"Unknown statistics.exclude_bands: {unknown_exclusions}")
    return config


def _configure_logger(output_dir: Path, overwrite: bool) -> logging.Logger:
    return configure_analysis_logger(
        "bout_analyses",
        output_dir,
        filename="bout_analyses.log",
        overwrite=overwrite,
    )


def _write_csv(table: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False, float_format="%.17g", compression="infer")


def _load_scale_free_cache(
    config: dict[str, Any],
    expected_subjects: list[str],
    common_channels: list[str],
) -> tuple[Path, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Validate and load the upstream per-subject detection cache.

    PSD fitting, spectral parameterization, wavelets, and eBOSC detection are
    cohort-independent feature calculations. They are owned by the scale-free
    stage and reused here so within-bout ordinal encoding does not repeat them.
    """
    root = Path(config["input"]["scale_free_output_dir"])
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing scale-free manifest: {manifest_path}. Run the scale-free "
            "analysis before within-bout ordinal analysis."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source = manifest.get("analysis_config", {})
    for section in ("bands", "specparam", "ebosc"):
        if not same_json_settings(source.get(section), config.get(section)):
            raise ValueError(
                f"Scale-free cache has incompatible {section} settings: "
                f"{manifest_path}"
            )
    source_psd = source.get("psd", {})
    for key in ("fmin_hz", "fmax_hz"):
        if float(source_psd.get(key, np.nan)) != float(config["psd"][key]):
            raise ValueError(
                f"Scale-free cache has incompatible psd.{key}: {manifest_path}"
            )

    inputs_path = root / "metrics" / "analyzed_inputs.csv"
    aperiodic_path = root / "metrics" / "electrode_aperiodic_metrics.csv"
    band_path = root / "metrics" / "electrode_band_metrics.csv"
    for path in (inputs_path, aperiodic_path, band_path):
        if not path.exists():
            raise FileNotFoundError(f"Scale-free cache is incomplete: {path}")
    source_inputs = pd.read_csv(inputs_path)
    aperiodic = pd.read_csv(aperiodic_path)
    band_metrics = pd.read_csv(band_path)

    requested_subjects = set(expected_subjects)
    missing_subjects = sorted(requested_subjects - set(source_inputs["subject_id"]))
    if missing_subjects:
        raise ValueError(
            f"Scale-free cache is missing requested subjects: {missing_subjects}"
        )
    aperiodic = aperiodic.loc[
        aperiodic["subject_id"].isin(expected_subjects)
        & aperiodic["electrode"].isin(common_channels)
    ].copy()
    band_metrics = band_metrics.loc[
        band_metrics["subject_id"].isin(expected_subjects)
        & band_metrics["electrode"].isin(common_channels)
        & band_metrics["band"].isin(config["bands"])
    ].copy()
    expected_electrodes = len(expected_subjects) * len(common_channels)
    expected_bands = expected_electrodes * len(config["bands"])
    if aperiodic.duplicated(["subject_id", "electrode"]).any() or len(aperiodic) != expected_electrodes:
        raise ValueError("Scale-free aperiodic cache is not a complete subject/electrode grid")
    if (
        band_metrics.duplicated(["subject_id", "electrode", "band"]).any()
        or len(band_metrics) != expected_bands
    ):
        raise ValueError(
            "Scale-free bout cache is not a complete subject/electrode/band grid"
        )
    for subject_id in expected_subjects:
        for subdirectory, suffix in (
            ("episodes", "bout_episodes.csv.gz"),
            ("thresholds", "ebosc_thresholds.csv.gz"),
            ("spectra", "specparam_spectra.npz"),
        ):
            path = root / "intermediate" / subdirectory / f"{subject_id}_{suffix}"
            if not path.exists():
                raise FileNotFoundError(f"Scale-free cache is incomplete: {path}")
    return root, source_inputs, aperiodic, band_metrics


def _cached_detection_example(
    source_root: Path,
    subject_id: str,
    electrode: str,
    band: str,
    signal_uv: np.ndarray,
    episodes: pd.DataFrame,
    config: dict[str, Any],
    sfreq: float,
) -> dict[str, Any]:
    """Recreate one transparent TFR panel while reusing fitted cache values."""
    threshold_table = pd.read_csv(
        source_root
        / "intermediate"
        / "thresholds"
        / f"{subject_id}_ebosc_thresholds.csv.gz"
    )
    selected = threshold_table.loc[
        threshold_table["electrode"].eq(electrode)
    ].sort_values("frequency_hz")
    selected_modes = selected["specparam_aperiodic_mode"].dropna().unique()
    if len(selected_modes) != 1:
        raise ValueError(
            f"Threshold cache must identify one selected aperiodic mode for {electrode}"
        )
    wavelet_frequencies = selected["frequency_hz"].to_numpy(dtype=float)
    thresholds = selected["power_threshold"].to_numpy(dtype=float)
    background = selected[
        "specparam_aperiodic_wavelet_background"
    ].to_numpy(dtype=float)
    mean_wavelet_power = selected["mean_wavelet_power"].to_numpy(dtype=float)
    wavelet_power = ebosc_wavelet_power(
        signal_uv,
        sfreq=sfreq,
        frequencies=wavelet_frequencies,
        wavenumber=float(config["ebosc"]["wavenumber"]),
    )
    edge_samples = int(
        round(float(config["ebosc"]["edge_padding_seconds"]) * sfreq)
    )
    detected = detect_frequency_episodes(
        wavelet_power,
        sfreq=sfreq,
        frequencies=wavelet_frequencies,
        thresholds=thresholds,
        minimum_cycles=float(config["ebosc"]["minimum_cycles"]),
        edge_padding_samples=edge_samples,
    )
    example_epoch = int(episodes.iloc[0]["epoch_index"])
    band_mask = np.zeros(signal_uv.shape, dtype=bool)
    for row in episodes.itertuples(index=False):
        band_mask[
            int(row.epoch_index),
            int(row.start_sample) : int(row.stop_sample_exclusive),
        ] = True

    with np.load(
        source_root
        / "intermediate"
        / "spectra"
        / f"{subject_id}_specparam_spectra.npz",
        allow_pickle=False,
    ) as spectra:
        electrodes = spectra["electrodes"].astype(str).tolist()
        index = electrodes.index(electrode)
        curve_names = [
            "observed_psd_uv2_hz",
            "modeled_psd_uv2_hz",
            "aperiodic_psd_uv2_hz",
            "periodic_psd_uv2_hz",
        ]
        curve_names.extend(
            name
            for name in (
                "fixed_aperiodic_psd_uv2_hz",
                "knee_aperiodic_psd_uv2_hz",
            )
            if name in spectra.files
        )
        curves = {
            "frequencies_hz": spectra["frequencies_hz"].copy(),
            **{
                name: spectra[name][index].copy()
                for name in curve_names
            },
        }
    return {
        "subject_id": subject_id,
        "electrode": electrode,
        "band": band,
        "specparam_aperiodic_mode": str(selected_modes[0]),
        "sfreq": sfreq,
        "signal_uv": signal_uv[example_epoch].copy(),
        "wavelet_frequencies_hz": wavelet_frequencies,
        "wavelet_power": wavelet_power[example_epoch].copy(),
        "thresholds": thresholds,
        "background": background,
        "mean_wavelet_power": mean_wavelet_power,
        "detected": detected[example_epoch].copy(),
        "band_mask": band_mask[example_epoch].copy(),
        **curves,
    }


def _subject_band_means(electrode_metrics: pd.DataFrame) -> pd.DataFrame:
    keys = ["subject_id", "group", "band", "band_low_hz", "band_high_hz"]
    means = electrode_metrics.groupby(keys, sort=False)[list(METRICS)].mean().reset_index()
    diagnostics = (
        electrode_metrics.groupby(keys, sort=False)
        .agg(
            n_electrodes=("electrode", "nunique"),
            n_electrodes_with_ordinal_metrics=("entropy", "count"),
            n_detected_bouts=("n_detected_bouts", "sum"),
            n_analyzable_ordinal_bouts=("n_analyzable_ordinal_bouts", "sum"),
            n_ordinal_patterns=("n_ordinal_patterns", "sum"),
            ordinal_state_space_coverage_mean=("ordinal_state_space_coverage", "mean"),
        )
        .reset_index()
    )
    return means.merge(diagnostics, on=keys, validate="one_to_one")


def _group_summary(subject_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (group, band, low_hz, high_hz), selected in subject_metrics.groupby(
        ["group", "band", "band_low_hz", "band_high_hz"], sort=False
    ):
        row: dict[str, Any] = {
            "group": group,
            "band": band,
            "band_low_hz": low_hz,
            "band_high_hz": high_hz,
            "n_subjects": int(selected["subject_id"].nunique()),
        }
        for metric in METRICS:
            values = selected[metric].dropna().to_numpy(dtype=float)
            row[f"{metric}_n_subjects"] = int(len(values))
            row[f"{metric}_mean"] = float(np.mean(values)) if len(values) else np.nan
            row[f"{metric}_std"] = (
                float(np.std(values, ddof=1)) if len(values) > 1 else np.nan
            )
            row[f"{metric}_median"] = float(np.median(values)) if len(values) else np.nan
        rows.append(row)
    return pd.DataFrame.from_records(rows)


def run_analysis(
    config_path: str | Path,
    *,
    subjects: list[str] | None = None,
    channels: list[str] | None = None,
    output_dir_override: str | Path | None = None,
    scale_free_output_dir_override: str | Path | None = None,
    overwrite: bool = False,
    show_progress: bool = True,
    generate_figures: bool = True,
) -> dict[str, Any]:
    config_path = Path(config_path)
    config = load_analysis_config(config_path)
    if output_dir_override is not None:
        config["output_dir"] = str(output_dir_override)
    if scale_free_output_dir_override is not None:
        config["input"]["scale_free_output_dir"] = str(
            scale_free_output_dir_override
        )
    output_dir = Path(config["output_dir"])
    result_path = output_dir / "metrics" / "subject_electrode_band_metrics.csv"
    if result_path.exists() and not overwrite:
        raise FileExistsError(
            f"Bout-analysis outputs already exist at {result_path}; rerun with --overwrite"
        )
    if overwrite:
        remove_retired_band_outputs(output_dir)
    logger = _configure_logger(output_dir, overwrite)

    participant_table = _participants(Path(config["input"]["participants_file"]))
    files = _epoch_files(
        Path(config["input"]["epochs_dir"]), str(config["input"]["epoch_glob"])
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
        available_channels[subject_id] = [epochs.ch_names[pick] for pick in eeg_picks]
    common_channels, electrode_union = ordered_channel_inventory(available_channels)
    if channels:
        requested_channels = list(dict.fromkeys(channels))
        unavailable = sorted(set(requested_channels) - set(common_channels))
        if unavailable:
            raise ValueError(f"Requested channels are not shared by every subject: {unavailable}")
        common_channels = [name for name in common_channels if name in requested_channels]
        if not common_channels:
            raise ValueError("No shared channels remain after channel selection")

    groups = participant_table.set_index("participant_id")["GROUP"].astype(str).to_dict()
    bands = {
        str(name): (float(limits[0]), float(limits[1]))
        for name, limits in config["bands"].items()
    }
    band_order = list(bands)
    dx = int(config["ordinal"]["embedding_dimension"])
    tau = int(config["ordinal"]["delay_samples"])
    tie_precision = config["ordinal"].get("tie_precision")
    state_space = math.factorial(dx)
    scale_free_root, scale_free_inputs, cached_aperiodic, cached_bands = (
        _load_scale_free_cache(config, expected_subjects, common_channels)
    )
    aperiodic_lookup = cached_aperiodic.set_index(["subject_id", "electrode"])
    band_lookup = cached_bands.set_index(["subject_id", "electrode", "band"])
    scale_free_input_lookup = scale_free_inputs.set_index("subject_id")
    logger.info(
        "Starting cached bout ordinal analysis | subjects=%d | shared_electrodes=%d | "
        "bands=%s | D=%d | tau=%d | detection_source=%s",
        len(expected_subjects),
        len(common_channels),
        ",".join(band_order),
        dx,
        tau,
        scale_free_root,
    )

    electrode_rows: list[dict[str, Any]] = []
    input_rows: list[dict[str, Any]] = []
    diagnostic_episode_tables: list[pd.DataFrame] = []
    subject_infos: dict[str, Any] = {}
    detection_example: dict[str, Any] | None = None
    ordinal_example: dict[str, Any] | None = None
    progress = tqdm(
        total=len(expected_subjects) * len(common_channels),
        desc="cached bouts + ordinal metrics",
        unit="electrode",
        dynamic_ncols=True,
        disable=not show_progress,
    )

    for subject_number, subject_id in enumerate(expected_subjects, start=1):
        path = files[subject_id]
        logger.info("[%d/%d] %s | %s", subject_number, len(expected_subjects), subject_id, path)
        epochs = mne.read_epochs(path, preload=True, verbose="ERROR")
        picks = [epochs.ch_names.index(channel) for channel in common_channels]
        data_v = epochs.get_data(picks=picks, copy=True)
        data_uv = data_v * 1e6
        sfreq = float(epochs.info["sfreq"])
        info = mne.pick_info(epochs.info, picks, copy=True)
        info["bads"] = []
        subject_infos[subject_id] = info
        cached_input = scale_free_input_lookup.loc[subject_id]
        if (
            int(cached_input["n_epochs"]) != len(epochs)
            or int(cached_input["samples_per_epoch"]) != data_v.shape[2]
            or not np.isclose(float(cached_input["sampling_frequency_hz"]), sfreq)
        ):
            raise ValueError(
                f"{subject_id}: cleaned epochs changed after the scale-free cache "
                "was created; rerun scale-free analysis"
            )

        episode_source = (
            scale_free_root
            / "intermediate"
            / "episodes"
            / f"{subject_id}_bout_episodes.csv.gz"
        )
        subject_episodes = pd.read_csv(episode_source)
        subject_episodes = subject_episodes.loc[
            subject_episodes["electrode"].isin(common_channels)
            & subject_episodes["band"].isin(band_order)
        ].copy()
        if len(subject_episodes):
            diagnostic_episode_tables.append(
                subject_episodes[
                    ["subject_id", "group", "electrode", "band", "duration_s"]
                ].copy()
            )

        filtered_by_band = {
            band: filter_epoch_data(
                data_uv,
                sfreq=sfreq,
                low_hz=limits[0],
                high_hz=limits[1],
                order=int(config["band_filter"]["order"]),
            )
            for band, limits in bands.items()
        }

        subject_counts = np.zeros(
            (len(common_channels), len(band_order), state_space), dtype=np.int64
        )
        subject_bout_metric_tables: list[pd.DataFrame] = []

        for channel_index, electrode in enumerate(common_channels):
            progress.set_postfix_str(f"{subject_id} | {electrode}", refresh=False)
            aperiodic = aperiodic_lookup.loc[(subject_id, electrode)]
            for band_index, (band, limits) in enumerate(bands.items()):
                episodes = subject_episodes.loc[
                    subject_episodes["electrode"].eq(electrode)
                    & subject_episodes["band"].eq(band)
                ].copy()
                analysis_episodes = episodes.drop(
                    columns=["subject_id", "group", "electrode"],
                    errors="ignore",
                )
                cached_band = band_lookup.loc[(subject_id, electrode, band)]
                pooled_counts, ordinal_summary, bout_metrics, segment_example = (
                    analyze_bout_segments(
                        filtered_by_band[band][:, channel_index, :],
                        analysis_episodes,
                        dx=dx,
                        tau=tau,
                        tie_precision=tie_precision,
                    )
                )
                subject_counts[channel_index, band_index] = pooled_counts
                electrode_rows.append(
                    {
                        "subject_id": subject_id,
                        "group": groups[subject_id],
                        "electrode": electrode,
                        "band": band,
                        "band_low_hz": limits[0],
                        "band_high_hz": limits[1],
                        **ordinal_summary,
                        "oscillatory_occupancy": cached_band[
                            "oscillatory_occupancy"
                        ],
                        "bouts_per_minute": cached_band["bouts_per_minute"],
                        "bout_duration_mean_s": cached_band[
                            "bout_duration_mean_s"
                        ],
                        "bout_duration_median_s": cached_band[
                            "bout_duration_median_s"
                        ],
                        "sampling_frequency_hz": sfreq,
                        "embedding_dimension": dx,
                        "delay_samples": tau,
                        "delay_seconds": tau / sfreq,
                        "tie_precision": "full_float64",
                        "aperiodic_exponent": float(
                            aperiodic["aperiodic_exponent"]
                        ),
                        "specparam_r_squared": float(
                            aperiodic["specparam_r_squared"]
                        ),
                    }
                )
                if len(bout_metrics):
                    enriched_bout_metrics = bout_metrics.copy()
                    enriched_bout_metrics.insert(0, "electrode", electrode)
                    enriched_bout_metrics.insert(0, "group", groups[subject_id])
                    enriched_bout_metrics.insert(0, "subject_id", subject_id)
                    subject_bout_metric_tables.append(enriched_bout_metrics)
                if segment_example is not None and ordinal_example is None:
                    ordinal_example = {
                        "subject_id": subject_id,
                        "group": groups[subject_id],
                        "electrode": electrode,
                        "band": band,
                        "sfreq": sfreq,
                        "embedding_dimension": dx,
                        "delay_samples": tau,
                        **segment_example,
                    }
                    example_epoch = int(segment_example["epoch_index"])
                    detection_example = {
                        "group": groups[subject_id],
                        **_cached_detection_example(
                            scale_free_root,
                            subject_id,
                            electrode,
                            band,
                            data_uv[:, channel_index, :],
                            episodes,
                            config,
                            sfreq,
                        ),
                    }
            progress.update()

        intermediate = output_dir / "intermediate"
        _write_csv(
            pd.concat(subject_bout_metric_tables, ignore_index=True)
            if subject_bout_metric_tables
            else pd.DataFrame(),
            intermediate / "bout_metrics" / f"{subject_id}_bout_ordinal_metrics.csv.gz",
        )
        for subdirectory, suffix in (
            ("episodes", "bout_episodes.csv.gz"),
            ("thresholds", "ebosc_thresholds.csv.gz"),
        ):
            destination = intermediate / subdirectory / f"{subject_id}_{suffix}"
            source_path = scale_free_root / "intermediate" / subdirectory / destination.name
            if bool(config["cache"]["link_reused_episode_and_threshold_files"]):
                replace_with_relative_symlink(source_path, destination)
            elif destination.is_symlink() or destination.exists():
                destination.unlink()
        counts_path = intermediate / "ordinal_counts" / f"{subject_id}_ordinal_counts.npz"
        counts_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            counts_path,
            subject_id=np.asarray(subject_id),
            electrodes=np.asarray(common_channels),
            bands=np.asarray(band_order),
            pattern_labels=np.asarray(
                ["".join(str(value) for value in pattern) for pattern in ordinal_patterns(dx)]
            ),
            counts=subject_counts,
            embedding_dimension=np.asarray(dx),
            delay_samples=np.asarray(tau),
        )
        input_rows.append(
            {
                "subject_id": subject_id,
                "group": groups[subject_id],
                "epoch_file": str(path.resolve()),
                "n_epochs": int(len(epochs)),
                "n_electrodes": int(len(common_channels)),
                "n_available_electrodes": int(len(available_channels[subject_id])),
                "samples_per_epoch": int(data_v.shape[2]),
                "sampling_frequency_hz": sfreq,
                "scale_free_cache_manifest": str(
                    (scale_free_root / "manifest.json").resolve()
                ),
            }
        )
    progress.close()

    electrode_metrics = pd.DataFrame.from_records(electrode_rows)
    subject_metrics = _subject_band_means(electrode_metrics)
    group_summary = _group_summary(subject_metrics)
    diagnostic_episodes = (
        pd.concat(diagnostic_episode_tables, ignore_index=True)
        if diagnostic_episode_tables
        else pd.DataFrame(columns=["subject_id", "group", "electrode", "band", "duration_s"])
    )
    statistics_config = config["statistics"]
    inferential_bands = [
        band for band in band_order
        if band not in set(statistics_config.get("exclude_bands", []))
    ]
    inferential_metrics = electrode_metrics.loc[
        electrode_metrics["band"].isin(inferential_bands)
    ].copy()
    subject_statistics, electrode_statistics = compute_group_statistics(
        inferential_metrics,
        participant_table,
        metrics=GROUP_METRICS,
        strata=("band",),
        domain="bout_detection_and_within_bout_ordinal",
        subject_aggregation=str(statistics_config["subject_aggregation"]),
        confidence_level=float(statistics_config["confidence_level"]),
        fdr_alpha=float(statistics_config["fdr_alpha"]),
    )
    metrics_dir = output_dir / "metrics"
    _write_csv(electrode_metrics, metrics_dir / "subject_electrode_band_metrics.csv")
    _write_csv(subject_metrics, metrics_dir / "subject_band_metrics.csv")
    _write_csv(group_summary, metrics_dir / "group_band_summary.csv")
    _write_csv(pd.DataFrame.from_records(input_rows), metrics_dir / "analyzed_inputs.csv")
    _write_csv(diagnostic_episodes, metrics_dir / "bout_duration_records.csv.gz")
    _write_csv(subject_statistics, metrics_dir / "group_subject_statistics.csv")
    _write_csv(electrode_statistics, metrics_dir / "group_electrode_statistics.csv")
    electrode_payload = {
        "common_electrodes": common_channels,
        "n_common_electrodes": len(common_channels),
        "electrode_union": electrode_union,
        "n_electrode_union": len(electrode_union),
        "analysis_electrode_policy": "Every analysis uses only electrodes present in every analyzed subject.",
    }
    (metrics_dir / "electrode_sets.json").write_text(
        json.dumps(electrode_payload, indent=2) + "\n", encoding="utf-8"
    )

    figures_dir = output_dir / "figures"
    dpi = int(config["plots"]["dpi"])
    configured_groups = [str(value) for value in config["plots"]["group_order"]]
    present_groups = set(electrode_metrics["group"])
    group_order = [group for group in configured_groups if group in present_groups]
    group_order.extend(sorted(present_groups - set(group_order)))
    colors = {
        group: str(config["plots"]["group_colors"].get(group, "0.4"))
        for group in group_order
    }
    band_labels = {
        band: str(config["plots"]["band_display_names"].get(band, band))
        for band in band_order
    }
    if generate_figures and detection_example is not None:
        plot_detection_example(
            detection_example, figures_dir / "steps" / "01_bout_detection.png", dpi
        )
    if ordinal_example is not None:
        if generate_figures:
            plot_ordinal_example(
                ordinal_example,
                figures_dir / "steps" / "02_ordinal_encoding.png",
                dpi,
            )
        counts = np.asarray(ordinal_example["counts"], dtype=np.int64)
        probability_table = pd.DataFrame(
            {
                "pattern_index": np.arange(len(counts)),
                "pattern": [
                    "".join(str(value) for value in pattern) for pattern in ordinal_patterns(dx)
                ],
                "count": counts,
                "probability": counts / counts.sum(),
            }
        )
        _write_csv(probability_table, metrics_dir / "example_bout_ordinal_distribution.csv")
    statistical_figures: list[Path] = []
    if generate_figures:
        plot_bout_diagnostics(
            diagnostic_episodes,
            electrode_metrics,
            group_order,
            colors,
            band_order,
            band_labels,
            figures_dir / "quality" / "bout_and_ordinal_diagnostics.png",
            dpi,
        )
        plot_subject_metric_violins(
            subject_metrics,
            group_order,
            colors,
            band_order,
            band_labels,
            figures_dir / "group" / "subject_metric_violins.png",
            dpi,
        )
        plot_ordinal_planes(
            subject_metrics,
            group_order,
            colors,
            band_order,
            band_labels,
            figures_dir / "group" / "subject_ordinal_planes.png",
            dpi,
        )
        plot_electrode_violins(
            electrode_metrics,
            common_channels,
            group_order,
            colors,
            band_order,
            band_labels,
            figures_dir / "electrodes",
            dpi,
        )
        common_info = next(iter(subject_infos.values())).copy()
        statistical_figures = plot_electrode_group_statistics(
            electrode_statistics,
            common_info,
            strata=("band",),
            output_dir=figures_dir / "group_statistics",
            dpi=dpi,
            stratum_labels={band: band_labels[band] for band in inferential_bands},
        )
        plot_group_topomaps(
            electrode_metrics,
            common_info,
            group_order,
            band_order,
            band_labels,
            common_channels,
            figures_dir / "topomaps" / "group_mean_topomaps.png",
            dpi,
        )
        if bool(config["plots"].get("subject_topomaps", True)):
            plot_subject_topomaps(
                electrode_metrics,
                subject_infos,
                band_order,
                band_labels,
                common_channels,
                figures_dir / "topomaps" / "subjects",
                dpi,
            )

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config_file": str(config_path.resolve()),
        "analysis_config": config,
        "software": {
            "python": platform.python_version(),
            "mne": mne.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "matplotlib": matplotlib.__version__,
            "ordpy": version("ordpy"),
            "specparam": version("specparam"),
            "ebosc": version("ebosc"),
        },
        "n_subjects": len(expected_subjects),
        "group_counts": pd.Series([groups[subject] for subject in expected_subjects]).value_counts().to_dict(),
        "n_common_electrodes": len(common_channels),
        "n_electrode_union": len(electrode_union),
        "specparam_primary_fit_range_hz": config["specparam"][
            "frequency_range_hz"
        ],
        "specparam_primary_fit_range_id": (
            f"{float(config['specparam']['frequency_range_hz'][0]):g}_"
            f"{float(config['specparam']['frequency_range_hz'][1]):g}Hz"
        ),
        "specparam_model_selection": {
            "candidate_modes": config["specparam"]["aperiodic_modes"],
            "criterion": config["specparam"]["model_selection_criterion"],
            "threshold_background": "selected_subject_electrode_model",
            "knee_frequency_outlier_z_threshold": config["specparam"][
                "knee_frequency_outlier_z_threshold"
            ],
        },
        "n_subject_electrode_band_rows": len(electrode_metrics),
        "feature_cache": {
            "scale_free_manifest": str(
                (scale_free_root / "manifest.json").resolve()
            ),
            "reused_calculations": [
                "welch_psd",
                "specparam",
                "ebosc_wavelet_power",
                "ebosc_detection",
                "bout_properties",
            ],
            "calculated_here": [
                "bandpass_filtering",
                "within_bout_ordinal_encoding",
                "within_bout_ordinal_metrics",
            ],
            "legacy_episode_threshold_paths_are_symlinks": bool(
                config["cache"]["link_reused_episode_and_threshold_files"]
            ),
        },
        "n_detected_bouts": int(electrode_metrics["n_detected_bouts"].sum()),
        "n_analyzable_ordinal_bouts": int(electrode_metrics["n_analyzable_ordinal_bouts"].sum()),
        "ordinal_state_space_size": state_space,
        "ordinal_metrics": list(METRICS),
        "renyi_metrics_included": False,
        "figures_generated": bool(generate_figures),
        "bout_boundary_policy": (
            "Each ordinal representation is created inside one detected bout. Pattern counts "
            "are pooled only after encoding, so embeddings never cross bout or epoch boundaries."
        ),
        "filtering_policy": (
            "Each accepted epoch is zero-phase band-pass filtered independently before detected "
            "bout intervals are sliced from it. Short bouts are never filtered in isolation."
        ),
        "electrode_policy": (
            "Only the electrode intersection shared by every analyzed subject contributes."
        ),
        "statistical_inference": {
            "tested_metrics": list(GROUP_METRICS),
            "primary_unit": "subject",
            "full_cohort_model": "OLS adjusted for age and sex with HC3 robust SE",
            "matched_cohort_model": "paired t test by match_pair_id; paired Wilcoxon saved as sensitivity",
            "subject_fdr_scope": "all canonical band-by-metric tests in the bout domain",
            "electrode_status": "exploratory localization; electrodes are not independent observations",
            "formal_electrode_fdr": "BH across every electrode-by-band-by-metric test in the bout domain",
            "excluded_bands": list(statistics_config.get("exclude_bands", [])),
            "exclusion_reason": "Overlapping visualization-only bands are excluded from formal inference",
            "n_subject_tests": int(len(subject_statistics)),
            "n_electrode_tests": int(len(electrode_statistics)),
            "n_statistical_figures": int(len(statistical_figures)),
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    logger.info(
        "Completed bout ordinal analysis | rows=%d | bouts=%d | analyzable=%d",
        len(electrode_metrics),
        manifest["n_detected_bouts"],
        manifest["n_analyzable_ordinal_bouts"],
    )
    return manifest
