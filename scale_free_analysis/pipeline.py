"""End-to-end spectral parameterization, bout, and cycle analysis."""

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

configure_runtime()

import matplotlib
import mne
import numpy as np
import pandas as pd
import scipy
from scipy.stats import mannwhitneyu, ttest_ind
from tqdm.auto import tqdm

from psd_analysis.metrics import compute_subject_electrode_psd
from src.dataset import ordered_channel_inventory
from src.group_statistics import compute_group_statistics
from src.group_statistics_plots import plot_electrode_group_statistics

from .aperiodic_diagnostics import run_aperiodic_diagnostics
from .metrics import (
    APERIODIC_FEATURES,
    BAND_FEATURES,
    aperiodic_wavelet_background,
    cycles_within_bouts,
    detect_frequency_episodes,
    ebosc_wavelet_power,
    extract_band_bouts,
    fit_specparam_spectrum,
    power_thresholds,
    summarize_bouts,
    summarize_cycles,
)
from .plots import (
    plot_aperiodic_topomaps,
    plot_band_topomaps,
    plot_bout_example,
    plot_cycle_example,
    plot_effect_sizes,
    plot_group_distributions,
    plot_spectral_example,
)
from .specparam_gallery import generate_specparam_gallery


SUBJECT_PATTERN = re.compile(r"(sub-\d+)")


def load_analysis_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as stream:
        config = json.load(stream)
    required = {
        "input",
        "output_dir",
        "bands",
        "psd",
        "specparam",
        "aperiodic_fit_qc",
        "aperiodic_sensitivity",
        "ebosc",
        "bycycle",
        "typical_bouts",
        "statistics",
        "plots",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Missing scale-free analysis config sections: {missing}")
    if str(config["specparam"]["aperiodic_mode"]) != "fixed":
        raise ValueError("This pipeline requires specparam.aperiodic_mode='fixed'")
    qc = config["aperiodic_fit_qc"]
    if not 0.0 <= float(qc["minimum_r_squared"]) <= 1.0:
        raise ValueError("aperiodic_fit_qc.minimum_r_squared must be in [0, 1]")
    if float(qc["maximum_error_mae_log10"]) <= 0.0:
        raise ValueError("aperiodic_fit_qc.maximum_error_mae_log10 must be positive")
    if float(qc["maximum_absolute_residual_log10"]) <= 0.0:
        raise ValueError(
            "aperiodic_fit_qc.maximum_absolute_residual_log10 must be positive"
        )
    if not 0.0 < float(qc["minimum_subject_qc_fraction"]) <= 1.0:
        raise ValueError(
            "aperiodic_fit_qc.minimum_subject_qc_fraction must be in (0, 1]"
        )
    exponent_range = [float(value) for value in qc["exponent_range"]]
    if len(exponent_range) != 2 or exponent_range[0] >= exponent_range[1]:
        raise ValueError("aperiodic_fit_qc.exponent_range must increase")
    sensitivity = config["aperiodic_sensitivity"]
    if int(sensitivity["workers"]) < 1:
        raise ValueError("aperiodic_sensitivity.workers must be positive")
    ranges = [
        [float(value) for value in limits]
        for limits in sensitivity["frequency_ranges_hz"]
    ]
    if [float(value) for value in config["specparam"]["frequency_range_hz"]] not in ranges:
        raise ValueError("Aperiodic sensitivity must contain the primary frequency range")
    bands = config["bands"]
    expected_bands = ["theta", "alpha", "low_beta", "high_beta", "broad_5_15"]
    if list(bands) != expected_bands:
        raise ValueError(
            "bands must be theta, alpha, low_beta, high_beta, and broad_5_15 in order"
        )
    for name, limits in bands.items():
        if len(limits) != 2 or not 0.0 < float(limits[0]) < float(limits[1]):
            raise ValueError(f"Invalid frequency limits for {name}")
    ebosc = config["ebosc"]
    if not 0.0 < float(ebosc["power_percentile"]) < 1.0:
        raise ValueError("ebosc.power_percentile must be between zero and one")
    if float(ebosc["minimum_cycles"]) <= 0.0:
        raise ValueError("ebosc.minimum_cycles must be positive")
    if int(config["plots"].get("specparam_gallery_workers", 1)) < 1:
        raise ValueError("plots.specparam_gallery_workers must be at least one")
    if int(config["plots"].get("specparam_gallery_dpi", 100)) < 50:
        raise ValueError("plots.specparam_gallery_dpi must be at least 50")
    overlap = float(config["bycycle"]["minimum_bout_overlap_fraction"])
    if not 0.0 <= overlap <= 1.0:
        raise ValueError("bycycle.minimum_bout_overlap_fraction must be between zero and one")
    typical = config["typical_bouts"]
    if float(typical["center_window_seconds"]) <= 0.0:
        raise ValueError("typical_bouts.center_window_seconds must be positive")
    if int(typical["bandpass_filter_order"]) < 1:
        raise ValueError("typical_bouts.bandpass_filter_order must be positive")
    if not 0.0 < float(typical["confidence_level"]) < 1.0:
        raise ValueError("typical_bouts.confidence_level must be between zero and one")
    if int(typical["workers"]) < 1:
        raise ValueError("typical_bouts.workers must be positive")
    statistics = config["statistics"]
    if not 0.0 < float(statistics["fdr_alpha"]) < 1.0:
        raise ValueError("statistics.fdr_alpha must be between zero and one")
    if not 0.0 < float(statistics["confidence_level"]) < 1.0:
        raise ValueError("statistics.confidence_level must be between zero and one")
    if statistics.get("subject_aggregation") != "mean":
        raise ValueError("Scale-free statistics.subject_aggregation must be mean")
    unknown_exclusions = sorted(set(statistics.get("exclude_bands", [])) - set(bands))
    if unknown_exclusions:
        raise ValueError(f"Unknown statistics.exclude_bands: {unknown_exclusions}")
    return config


def _participants(path: Path) -> pd.DataFrame:
    separator = "\t" if path.suffix.lower() == ".tsv" else ","
    table = pd.read_csv(path, sep=separator)
    required = {"participant_id", "GROUP"}
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
        if match is None:
            continue
        subject_id = match.group(1)
        if subject_id in files:
            raise ValueError(f"Multiple epoch files found for {subject_id}")
        files[subject_id] = path
    return files


def _configure_logger(output_dir: Path, overwrite: bool) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("scale_free_analysis")
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(console)
    file_handler = logging.FileHandler(
        output_dir / "scale_free_analysis.log",
        mode="w" if overwrite else "a",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def _write_csv(table: pd.DataFrame, path: Path, *, compressed: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    compression = "gzip" if compressed else None
    table.to_csv(
        path,
        index=False,
        float_format="%.17g",
        compression=compression,
    )


def _describe(table: pd.DataFrame, by: list[str], features: tuple[str, ...]) -> pd.DataFrame:
    rows = []
    for keys, selected in table.groupby(by, sort=False, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(by, keys))
        row["n_subjects"] = int(selected["subject_id"].nunique())
        for feature in features:
            values = selected[feature].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            row[f"{feature}_n"] = int(len(values))
            row[f"{feature}_mean"] = float(np.mean(values)) if len(values) else np.nan
            row[f"{feature}_std"] = (
                float(np.std(values, ddof=1)) if len(values) > 1 else np.nan
            )
            row[f"{feature}_median"] = float(np.median(values)) if len(values) else np.nan
        rows.append(row)
    return pd.DataFrame.from_records(rows)


def _hedges_g(pd_values: np.ndarray, control_values: np.ndarray) -> float:
    n_pd, n_control = len(pd_values), len(control_values)
    if n_pd < 2 or n_control < 2:
        return np.nan
    pooled_variance = (
        (n_pd - 1) * np.var(pd_values, ddof=1)
        + (n_control - 1) * np.var(control_values, ddof=1)
    ) / (n_pd + n_control - 2)
    if not np.isfinite(pooled_variance) or pooled_variance <= 0.0:
        return 0.0 if np.isclose(np.mean(pd_values), np.mean(control_values)) else np.nan
    correction = 1.0 - 3.0 / (4.0 * (n_pd + n_control) - 9.0)
    return float(correction * (np.mean(pd_values) - np.mean(control_values)) / np.sqrt(pooled_variance))


def _fdr_bh(p_values: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    p_values = np.asarray(p_values, dtype=float)
    adjusted = np.full_like(p_values, np.nan)
    reject = np.zeros(len(p_values), dtype=bool)
    finite_indices = np.flatnonzero(np.isfinite(p_values))
    if not len(finite_indices):
        return adjusted, reject
    finite = p_values[finite_indices]
    order = np.argsort(finite)
    ranked = finite[order]
    m = len(ranked)
    corrected = ranked * m / np.arange(1, m + 1)
    corrected = np.minimum.accumulate(corrected[::-1])[::-1]
    corrected = np.clip(corrected, 0.0, 1.0)
    adjusted_finite = np.empty_like(corrected)
    adjusted_finite[order] = corrected
    adjusted[finite_indices] = adjusted_finite
    reject[finite_indices] = adjusted_finite <= float(alpha)
    return adjusted, reject


def _comparisons(
    aperiodic_subjects: pd.DataFrame,
    band_subjects: pd.DataFrame,
    *,
    fdr_alpha: float,
) -> pd.DataFrame:
    rows = []
    domains = [
        ("aperiodic", "broadband", aperiodic_subjects, APERIODIC_FEATURES),
    ]
    for band in band_subjects["band"].drop_duplicates():
        domains.append(
            (
                "band",
                str(band),
                band_subjects.loc[band_subjects["band"].eq(band)],
                BAND_FEATURES,
            )
        )
    for domain, band, table, features in domains:
        for feature in features:
            pd_values = table.loc[table["group"].eq("PD"), feature].dropna().to_numpy(dtype=float)
            control_values = table.loc[table["group"].eq("Control"), feature].dropna().to_numpy(dtype=float)
            if len(pd_values) >= 2 and len(control_values) >= 2:
                welch = ttest_ind(pd_values, control_values, equal_var=False)
                mann = mannwhitneyu(pd_values, control_values, alternative="two-sided")
                welch_statistic, welch_p = float(welch.statistic), float(welch.pvalue)
                mann_statistic, mann_p = float(mann.statistic), float(mann.pvalue)
            else:
                welch_statistic = welch_p = mann_statistic = mann_p = np.nan
            rows.append(
                {
                    "domain": domain,
                    "band": band,
                    "metric": feature,
                    "n_pd": int(len(pd_values)),
                    "n_control": int(len(control_values)),
                    "pd_mean": float(np.mean(pd_values)) if len(pd_values) else np.nan,
                    "control_mean": float(np.mean(control_values)) if len(control_values) else np.nan,
                    "welch_t": welch_statistic,
                    "welch_p": welch_p,
                    "mann_whitney_u": mann_statistic,
                    "mann_whitney_p": mann_p,
                    "hedges_g_pd_minus_control": _hedges_g(pd_values, control_values),
                }
            )
    result = pd.DataFrame.from_records(rows)
    adjusted, reject = _fdr_bh(result["welch_p"].to_numpy(dtype=float), fdr_alpha)
    result["welch_p_fdr_bh"] = adjusted
    result["fdr_alpha"] = float(fdr_alpha)
    result["fdr_reject"] = reject
    return result


def _subject_means(table: pd.DataFrame, keys: list[str], features: tuple[str, ...]) -> pd.DataFrame:
    means = table.groupby(keys, sort=False)[list(features)].mean().reset_index()
    counts = (
        table.groupby(keys, sort=False)["electrode"]
        .nunique()
        .rename("n_electrodes")
        .reset_index()
    )
    return means.merge(counts, on=keys, validate="one_to_one")


def run_analysis(
    config_path: str | Path,
    *,
    subjects: list[str] | None = None,
    channels: list[str] | None = None,
    output_dir_override: str | Path | None = None,
    overwrite: bool = False,
    show_progress: bool = True,
    skip_specparam_gallery: bool = False,
) -> dict[str, Any]:
    config_path = Path(config_path)
    config = load_analysis_config(config_path)
    if output_dir_override is not None:
        config["output_dir"] = str(output_dir_override)
    output_dir = Path(config["output_dir"])
    result_path = output_dir / "metrics" / "subject_band_metrics.csv"
    if result_path.exists() and not overwrite:
        raise FileExistsError(
            f"Scale-free outputs already exist at {result_path}; rerun with --overwrite"
        )
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
    missing = sorted(set(expected_subjects) - set(files))
    if missing:
        raise FileNotFoundError(f"Missing cleaned epoch files for: {missing}")

    available_channels: dict[str, list[str]] = {}
    for subject_id in expected_subjects:
        epochs = mne.read_epochs(files[subject_id], preload=False, verbose="ERROR")
        picks = mne.pick_types(epochs.info, eeg=True, exclude=[])
        available_channels[subject_id] = [epochs.ch_names[pick] for pick in picks]
    common_channels, electrode_union = ordered_channel_inventory(available_channels)
    if channels:
        requested_channels = list(dict.fromkeys(channels))
        unavailable = sorted(set(requested_channels) - set(common_channels))
        if unavailable:
            raise ValueError(f"Requested channels are not shared by every subject: {unavailable}")
        common_channels = [channel for channel in common_channels if channel in requested_channels]
        if not common_channels:
            raise ValueError("No channels remain after channel selection")

    group_lookup = participant_table.set_index("participant_id")["GROUP"].astype(str).to_dict()
    bands = {
        str(name): (float(limits[0]), float(limits[1]))
        for name, limits in config["bands"].items()
    }
    band_order = list(bands)
    ebosc_config = config["ebosc"]
    wavelet_frequencies = np.arange(
        float(ebosc_config["frequency_min_hz"]),
        float(ebosc_config["frequency_max_hz"]) + 0.5 * float(ebosc_config["frequency_step_hz"]),
        float(ebosc_config["frequency_step_hz"]),
    )
    logger.info(
        "Starting scale-free analysis | subjects=%d | shared_electrodes=%d | bands=%s",
        len(expected_subjects),
        len(common_channels),
        ",".join(band_order),
    )

    aperiodic_rows: list[dict[str, Any]] = []
    band_rows: list[dict[str, Any]] = []
    input_rows: list[dict[str, Any]] = []
    subject_infos: dict[str, mne.Info] = {}
    spectral_example: dict[str, Any] | None = None
    bout_example: dict[str, Any] | None = None
    cycle_example: dict[str, Any] | None = None
    progress = tqdm(
        total=len(expected_subjects) * len(common_channels),
        desc="specparam + eBOSC + bycycle",
        unit="electrode",
        dynamic_ncols=True,
        disable=not show_progress,
    )

    for subject_index, subject_id in enumerate(expected_subjects, start=1):
        path = files[subject_id]
        logger.info("[%d/%d] %s | %s", subject_index, len(expected_subjects), subject_id, path)
        epochs = mne.read_epochs(path, preload=True, verbose="ERROR")
        picks = [epochs.ch_names.index(channel) for channel in common_channels]
        data_v = epochs.get_data(picks=picks, copy=True)
        data_uv = data_v * 1e6
        sfreq = float(epochs.info["sfreq"])
        edge_samples = int(round(float(ebosc_config["edge_padding_seconds"]) * sfreq))
        info = mne.pick_info(epochs.info, picks, copy=True)
        info["bads"] = []
        subject_infos[subject_id] = info
        psd_frequencies, electrode_psd = compute_subject_electrode_psd(
            data_v,
            sfreq,
            fmin=float(config["psd"]["fmin_hz"]),
            fmax=float(config["psd"]["fmax_hz"]),
        )

        subject_spectra: dict[str, list[np.ndarray]] = {
            "observed_psd_uv2_hz": [],
            "modeled_psd_uv2_hz": [],
            "aperiodic_psd_uv2_hz": [],
            "periodic_psd_uv2_hz": [],
        }
        subject_episode_tables = []
        subject_cycle_tables = []
        subject_threshold_rows = []

        for channel_index, electrode in enumerate(common_channels):
            progress.set_postfix_str(f"{subject_id} | {electrode}", refresh=False)
            aperiodic, periodic_bands, curves = fit_specparam_spectrum(
                psd_frequencies,
                electrode_psd[channel_index],
                bands,
                config["specparam"],
            )
            aperiodic_rows.append(
                {
                    "subject_id": subject_id,
                    "group": group_lookup[subject_id],
                    "electrode": electrode,
                    **aperiodic,
                }
            )
            periodic_lookup = {row["band"]: row for row in periodic_bands}
            for name in subject_spectra:
                subject_spectra[name].append(curves[name])
            if spectral_example is None:
                spectral_example = {
                    "subject_id": subject_id,
                    "electrode": electrode,
                    **curves,
                }

            wavelet_power = ebosc_wavelet_power(
                data_uv[:, channel_index, :],
                sfreq=sfreq,
                frequencies=wavelet_frequencies,
                wavenumber=float(ebosc_config["wavenumber"]),
            )
            interior = (
                wavelet_power
                if edge_samples == 0
                else wavelet_power[..., edge_samples:-edge_samples]
            )
            mean_wavelet_power = np.mean(interior, axis=(0, 2))
            background = aperiodic_wavelet_background(
                curves["frequencies_hz"],
                curves["modeled_psd_uv2_hz"],
                curves["aperiodic_psd_uv2_hz"],
                wavelet_frequencies,
                mean_wavelet_power,
            )
            thresholds = power_thresholds(
                background, float(ebosc_config["power_percentile"])
            )
            detected = detect_frequency_episodes(
                wavelet_power,
                sfreq=sfreq,
                frequencies=wavelet_frequencies,
                thresholds=thresholds,
                minimum_cycles=float(ebosc_config["minimum_cycles"]),
                edge_padding_samples=edge_samples,
            )
            for frequency, observed, background_value, threshold in zip(
                wavelet_frequencies, mean_wavelet_power, background, thresholds
            ):
                subject_threshold_rows.append(
                    {
                        "subject_id": subject_id,
                        "electrode": electrode,
                        "frequency_hz": float(frequency),
                        "mean_wavelet_power": float(observed),
                        "specparam_aperiodic_wavelet_background": float(background_value),
                        "power_threshold": float(threshold),
                    }
                )

            for band, limits in bands.items():
                episodes_table, band_mask = extract_band_bouts(
                    detected,
                    wavelet_power,
                    thresholds,
                    wavelet_frequencies,
                    band=band,
                    band_limits=limits,
                    sfreq=sfreq,
                )
                if not episodes_table.empty:
                    episodes_table.insert(0, "electrode", electrode)
                    episodes_table.insert(0, "group", group_lookup[subject_id])
                    episodes_table.insert(0, "subject_id", subject_id)
                    subject_episode_tables.append(episodes_table)
                bout_summary = summarize_bouts(
                    episodes_table,
                    band_mask,
                    sfreq=sfreq,
                    edge_padding_samples=edge_samples,
                )

                cycle_tables = []
                for epoch_index in range(data_uv.shape[0]):
                    if not band_mask[epoch_index].any():
                        continue
                    try:
                        cycles = cycles_within_bouts(
                            data_uv[epoch_index, channel_index],
                            band_mask[epoch_index],
                            sfreq=sfreq,
                            band_limits=limits,
                            minimum_overlap=float(
                                config["bycycle"]["minimum_bout_overlap_fraction"]
                            ),
                        )
                    except (IndexError, ValueError) as error:
                        logger.debug(
                            "%s/%s/%s/epoch-%d: bycycle skipped: %s",
                            subject_id,
                            electrode,
                            band,
                            epoch_index,
                            error,
                        )
                        continue
                    if not cycles.empty:
                        cycles.insert(0, "epoch_index", int(epoch_index))
                        cycle_tables.append(cycles)
                all_cycles = (
                    pd.concat(cycle_tables, ignore_index=True)
                    if cycle_tables
                    else pd.DataFrame()
                )
                cycle_summary = summarize_cycles(all_cycles)
                if not all_cycles.empty:
                    all_cycles.insert(0, "band_high_hz", limits[1])
                    all_cycles.insert(0, "band_low_hz", limits[0])
                    all_cycles.insert(0, "band", band)
                    all_cycles.insert(0, "electrode", electrode)
                    all_cycles.insert(0, "group", group_lookup[subject_id])
                    all_cycles.insert(0, "subject_id", subject_id)
                    subject_cycle_tables.append(all_cycles)

                band_rows.append(
                    {
                        "subject_id": subject_id,
                        "group": group_lookup[subject_id],
                        "electrode": electrode,
                        **periodic_lookup[band],
                        **bout_summary,
                        **cycle_summary,
                    }
                )

                if bout_example is None and band_mask.any():
                    example_epoch = int(np.flatnonzero(band_mask.any(axis=1))[0])
                    band_frequency_mask = (
                        (wavelet_frequencies >= limits[0])
                        & (wavelet_frequencies <= limits[1])
                    )
                    bout_example = {
                        "subject_id": subject_id,
                        "electrode": electrode,
                        "band": band,
                        "sfreq": sfreq,
                        "signal_uv": data_uv[example_epoch, channel_index].copy(),
                        "band_mask": band_mask[example_epoch].copy(),
                        "wavelet_frequencies_hz": wavelet_frequencies[band_frequency_mask].copy(),
                        "wavelet_power": wavelet_power[example_epoch, band_frequency_mask].copy(),
                    }
                    example_cycles = all_cycles.loc[
                        all_cycles["epoch_index"].eq(example_epoch)
                    ].copy() if not all_cycles.empty else pd.DataFrame()
                    cycle_example = {
                        "subject_id": subject_id,
                        "electrode": electrode,
                        "band": band,
                        "sfreq": sfreq,
                        "signal_uv": data_uv[example_epoch, channel_index].copy(),
                        "cycles": example_cycles,
                    }
            progress.update()

        intermediate_dir = output_dir / "intermediate"
        (intermediate_dir / "spectra").mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            intermediate_dir / "spectra" / f"{subject_id}_specparam_spectra.npz",
            electrodes=np.asarray(common_channels),
            frequencies_hz=curves["frequencies_hz"],
            **{
                name: np.asarray(values, dtype=float)
                for name, values in subject_spectra.items()
            },
        )
        episode_output = (
            pd.concat(subject_episode_tables, ignore_index=True)
            if subject_episode_tables
            else pd.DataFrame()
        )
        cycle_output = (
            pd.concat(subject_cycle_tables, ignore_index=True)
            if subject_cycle_tables
            else pd.DataFrame()
        )
        _write_csv(
            episode_output,
            intermediate_dir / "episodes" / f"{subject_id}_bout_episodes.csv.gz",
            compressed=True,
        )
        _write_csv(
            cycle_output,
            intermediate_dir / "cycles" / f"{subject_id}_bycycle_cycles.csv.gz",
            compressed=True,
        )
        _write_csv(
            pd.DataFrame.from_records(subject_threshold_rows),
            intermediate_dir / "thresholds" / f"{subject_id}_ebosc_thresholds.csv.gz",
            compressed=True,
        )
        input_rows.append(
            {
                "subject_id": subject_id,
                "group": group_lookup[subject_id],
                "epoch_file": str(path.resolve()),
                "n_epochs": len(epochs),
                "n_electrodes": len(common_channels),
                "n_available_electrodes": len(available_channels[subject_id]),
                "samples_per_epoch": data_v.shape[2],
                "sampling_frequency_hz": sfreq,
                "accepted_duration_seconds": float(len(epochs) * data_v.shape[2] / sfreq),
            }
        )
    progress.close()

    aperiodic_electrodes = pd.DataFrame.from_records(aperiodic_rows)
    band_electrodes = pd.DataFrame.from_records(band_rows)
    logger.info("Running specparam fit QC and fixed-mode frequency-range sensitivity")
    aperiodic_diagnostics = run_aperiodic_diagnostics(
        output_dir,
        aperiodic_electrodes,
        config,
        logger=logger,
    )
    aperiodic_electrodes = aperiodic_diagnostics["electrode_metrics"]
    aperiodic_subjects = _subject_means(
        aperiodic_electrodes,
        ["subject_id", "group"],
        APERIODIC_FEATURES,
    )
    band_subjects = _subject_means(
        band_electrodes,
        ["subject_id", "group", "band", "band_low_hz", "band_high_hz"],
        BAND_FEATURES,
    )
    group_aperiodic = _describe(
        aperiodic_subjects,
        ["group"],
        APERIODIC_FEATURES,
    )
    group_bands = _describe(
        band_subjects,
        ["band", "band_low_hz", "band_high_hz", "group"],
        BAND_FEATURES,
    )
    comparisons = _comparisons(
        aperiodic_subjects,
        band_subjects,
        fdr_alpha=float(config["statistics"]["fdr_alpha"]),
    )
    statistics_config = config["statistics"]
    aperiodic_subject_statistics, aperiodic_electrode_statistics = (
        compute_group_statistics(
            aperiodic_electrodes,
            participant_table,
            metrics=APERIODIC_FEATURES,
            domain="scale_free_aperiodic",
            subject_aggregation=str(statistics_config["subject_aggregation"]),
            confidence_level=float(statistics_config["confidence_level"]),
            fdr_alpha=float(statistics_config["fdr_alpha"]),
        )
    )
    inferential_bands = [
        band for band in band_order
        if band not in set(statistics_config.get("exclude_bands", []))
    ]
    inferential_band_electrodes = band_electrodes.loc[
        band_electrodes["band"].isin(inferential_bands)
    ].copy()
    band_subject_statistics, band_electrode_statistics = compute_group_statistics(
        inferential_band_electrodes,
        participant_table,
        metrics=BAND_FEATURES,
        strata=("band",),
        domain="scale_free_periodic_and_bouts",
        subject_aggregation=str(statistics_config["subject_aggregation"]),
        confidence_level=float(statistics_config["confidence_level"]),
        fdr_alpha=float(statistics_config["fdr_alpha"]),
    )

    metrics_dir = output_dir / "metrics"
    _write_csv(pd.DataFrame.from_records(input_rows), metrics_dir / "analyzed_inputs.csv")
    _write_csv(aperiodic_electrodes, metrics_dir / "electrode_aperiodic_metrics.csv")
    _write_csv(band_electrodes, metrics_dir / "electrode_band_metrics.csv")
    _write_csv(aperiodic_subjects, metrics_dir / "subject_aperiodic_metrics.csv")
    _write_csv(band_subjects, metrics_dir / "subject_band_metrics.csv")
    _write_csv(group_aperiodic, metrics_dir / "group_aperiodic_summary.csv")
    _write_csv(group_bands, metrics_dir / "group_band_summary.csv")
    _write_csv(comparisons, metrics_dir / "pd_control_comparisons.csv")
    _write_csv(
        aperiodic_subject_statistics,
        metrics_dir / "group_subject_statistics_aperiodic.csv",
    )
    _write_csv(
        aperiodic_electrode_statistics,
        metrics_dir / "group_electrode_statistics_aperiodic.csv",
    )
    _write_csv(
        band_subject_statistics,
        metrics_dir / "group_subject_statistics_periodic_bout.csv",
    )
    _write_csv(
        band_electrode_statistics,
        metrics_dir / "group_electrode_statistics_periodic_bout.csv",
    )

    common_info = next(iter(subject_infos.values())).copy()
    configured_groups = [str(group) for group in config["plots"]["group_order"]]
    present_groups = set(aperiodic_subjects["group"])
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
    dpi = int(config["plots"]["dpi"])
    figures_dir = output_dir / "figures"
    specparam_gallery_index = pd.DataFrame()
    gallery_enabled = bool(
        config["plots"].get("specparam_gallery_enabled", True)
    ) and not skip_specparam_gallery
    if gallery_enabled:
        logger.info(
            "Creating subject/electrode specparam gallery | figures=%d | workers=%d",
            len(aperiodic_electrodes),
            int(config["plots"].get("specparam_gallery_workers", 1)),
        )
        specparam_gallery_index = generate_specparam_gallery(
            output_dir / "intermediate" / "spectra",
            aperiodic_electrodes,
            figures_dir / "specparam_decomposition",
            dpi=int(config["plots"].get("specparam_gallery_dpi", 100)),
            workers=int(config["plots"].get("specparam_gallery_workers", 1)),
            overwrite=overwrite,
            logger=logger,
        )
        _write_csv(
            specparam_gallery_index,
            metrics_dir / "specparam_figure_index.csv",
        )
    else:
        logger.info("Skipping subject/electrode specparam gallery")
    if spectral_example is not None:
        example_metrics = aperiodic_electrodes.loc[
            aperiodic_electrodes["subject_id"].eq(spectral_example["subject_id"])
            & aperiodic_electrodes["electrode"].eq(spectral_example["electrode"])
        ].iloc[0]
        spectral_example.update(
            {
                "group": example_metrics["group"],
                "aperiodic_exponent": example_metrics["aperiodic_exponent"],
                "specparam_r_squared": example_metrics["specparam_r_squared"],
                "specparam_error_mae": example_metrics["specparam_error_mae"],
                "specparam_fit_qc_pass": example_metrics["specparam_fit_qc_pass"],
                "specparam_fit_qc_reasons": example_metrics[
                    "specparam_fit_qc_reasons"
                ],
            }
        )
        plot_spectral_example(
            spectral_example,
            figures_dir / "examples" / "specparam_decomposition.png",
            dpi,
        )
    if bout_example is not None:
        plot_bout_example(
            bout_example,
            figures_dir / "examples" / "detected_bout_and_time_frequency.png",
            dpi,
        )
    if cycle_example is not None:
        plot_cycle_example(
            cycle_example,
            figures_dir / "examples" / "bycycle_waveform_landmarks.png",
            dpi,
        )
    plot_group_distributions(
        band_subjects,
        group_order,
        colors,
        band_order,
        band_labels,
        figures_dir / "group_comparisons",
        dpi,
    )
    if len(common_channels) >= 4:
        plot_aperiodic_topomaps(
            aperiodic_electrodes,
            common_info,
            group_order,
            figures_dir / "topomaps" / "aperiodic_offset_exponent.png",
            dpi,
        )
        plot_band_topomaps(
            band_electrodes,
            common_info,
            group_order,
            band_order,
            band_labels,
            figures_dir / "topomaps",
            dpi,
        )
        logger.info("Creating electrode-wise PD-Control statistical maps")
        aperiodic_statistical_figures = plot_electrode_group_statistics(
            aperiodic_electrode_statistics,
            common_info,
            strata=(),
            output_dir=figures_dir / "group_statistics" / "aperiodic",
            dpi=dpi,
        )
        band_statistical_figures = plot_electrode_group_statistics(
            band_electrode_statistics,
            common_info,
            strata=("band",),
            output_dir=figures_dir / "group_statistics" / "periodic_bout",
            dpi=dpi,
            stratum_labels={band: band_labels[band] for band in inferential_bands},
        )
    else:
        logger.info("Skipping topomaps because fewer than four electrodes were selected")
        aperiodic_statistical_figures = []
        band_statistical_figures = []
    if {"PD", "Control"}.issubset(present_groups):
        plot_effect_sizes(
            comparisons,
            band_order,
            band_labels,
            figures_dir / "group_comparisons" / "pd_control_effect_sizes.png",
            dpi,
        )

    electrode_payload = {
        "common_electrodes": common_channels,
        "n_common_electrodes": len(common_channels),
        "electrode_union": electrode_union,
        "n_electrode_union": len(electrode_union),
        "analysis_electrode_policy": (
            "Every analysis uses only electrodes present in every analyzed subject."
        ),
    }
    (metrics_dir / "electrode_sets.json").write_text(
        json.dumps(electrode_payload, indent=2) + "\n", encoding="utf-8"
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
            "specparam": version("specparam"),
            "ebosc": version("ebosc"),
            "bycycle": version("bycycle"),
            "neurodsp": version("neurodsp"),
        },
        "n_subjects": len(expected_subjects),
        "group_counts": pd.Series([group_lookup[s] for s in expected_subjects]).value_counts().to_dict(),
        "n_common_electrodes": len(common_channels),
        "n_electrode_union": len(electrode_union),
        "n_specparam_decomposition_figures": int(len(specparam_gallery_index)),
        "n_specparam_subject_overview_figures": int(
            specparam_gallery_index["subject_id"].nunique()
            if not specparam_gallery_index.empty
            else 0
        ),
        "specparam_fit_qc": {
            "thresholds": config["aperiodic_fit_qc"],
            "n_fits": int(len(aperiodic_electrodes)),
            "n_qc_pass": int(aperiodic_electrodes["specparam_fit_qc_pass"].sum()),
            "qc_pass_fraction": float(
                aperiodic_electrodes["specparam_fit_qc_pass"].mean()
            ),
            "frequency_ranges_hz": config["aperiodic_sensitivity"][
                "frequency_ranges_hz"
            ],
            "n_range_sensitivity_fits": int(
                len(aperiodic_diagnostics["electrode_sensitivity"])
            ),
            "policy": config["aperiodic_fit_qc"]["policy"],
        },
        "specparam_gallery_enabled": bool(gallery_enabled),
        "specparam_gallery_policy": (
            "One primary all-electrode overview per subject, plus detailed "
            "subject/electrode decomposition PNGs and HTML indexes. Figures reuse "
            "saved fitted spectral curves and do not refit specparam."
        ),
        "epoch_boundary_policy": (
            "Only accepted cleaned epochs are analyzed. PSD periodograms are pooled, while "
            "wavelets, bout detection, and bycycle are applied independently within each "
            "epoch so no bout or cycle crosses an epoch boundary."
        ),
        "aperiodic_threshold_policy": (
            "specparam fixed-mode aperiodic PSD is mapped to the exact eBOSC Morlet-power "
            "scale using the ratio between mean wavelet power and the full specparam model. "
            "The BOSC chi-square percentile and frequency-specific duration thresholds are "
            "then applied to this aperiodic background."
        ),
        "ebosc_implementation": (
            "The vectorized wavelet transform exactly reproduces ebosc.BOSC.BOSC_tf's Morlet "
            "definition and crop while keeping epochs separate. Detection follows BOSC/eBOSC "
            "power-plus-duration criteria; band bouts are contiguous supra-threshold runs."
        ),
        "bycycle_selection_policy": (
            "bycycle shape features are computed within accepted epochs and retained only "
            "when the configured fraction of the trough-to-trough cycle overlaps an eBOSC "
            "band-bout mask."
        ),
        "statistics_policy": {
            "primary_unit": "subject",
            "full_cohort_model": "OLS adjusted for age and sex with HC3 robust SE",
            "matched_cohort_model": "paired t test by match_pair_id; paired Wilcoxon saved as sensitivity",
            "subject_fdr_scope": "separate aperiodic and canonical periodic/bout domains",
            "electrode_status": "exploratory localization; electrodes are not independent observations",
            "formal_electrode_fdr": "BH across every electrode-by-metric test in each domain",
            "legacy_table": "pd_control_comparisons.csv retains unadjusted Welch/Mann-Whitney results for compatibility",
            "excluded_bands": list(statistics_config.get("exclude_bands", [])),
            "exclusion_reason": "Overlapping visualization-only bands are excluded from formal inference",
            "n_subject_tests": int(
                len(aperiodic_subject_statistics) + len(band_subject_statistics)
            ),
            "n_electrode_tests": int(
                len(aperiodic_electrode_statistics) + len(band_electrode_statistics)
            ),
            "n_statistical_figures": int(
                len(aperiodic_statistical_figures) + len(band_statistical_figures)
            ),
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    logger.info("Scale-free analysis completed | output=%s", output_dir)
    return manifest
