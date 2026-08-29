"""Extract session-balanced EEG features for the ds002778 medication study."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from core.runtime import configure_runtime

configure_runtime()

import mne
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from analyses.ordinal.metrics import analyze_epoch_data, filter_epoch_data
from analyses.psd.metrics import (
    compute_subject_electrode_psd,
    integrate_bands,
    relative_band_powers,
)
from analyses.scale_free.metrics import (
    aperiodic_wavelet_background,
    detect_frequency_episodes,
    ebosc_wavelet_power,
    extract_band_bouts,
    fit_specparam_spectrum,
    power_thresholds,
    summarize_bouts,
)
from core.analysis_io import discover_epoch_files
from core.dataset import ordered_channel_inventory


CORE_ORDINAL_METRICS = ("entropy", "complexity", "fisher_information")
APERIODIC_METRICS = (
    "aperiodic_offset",
    "aperiodic_exponent",
    "specparam_r_squared",
    "specparam_error_mae",
)
BOUT_METRICS = (
    "oscillatory_occupancy",
    "bouts_per_minute",
    "bout_duration_mean_s",
    "bout_cycles_mean",
    "bout_snr_mean",
)


def _finite_aggregate(values: Iterable[float], method: str) -> float:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return np.nan
    return float(np.median(array) if method == "median" else np.mean(array))


def _subject_row(
    recording_id: str,
    duration_variant: str,
    family: str,
    domain: str,
    band: str,
    metric: str,
    value: float,
    *,
    n_epochs: int,
    n_electrodes: int,
    aggregation: str,
) -> dict[str, Any]:
    feature_id = f"{domain}_{metric}" if band == "broadband" else f"{domain}_{band}_{metric}"
    return {
        "recording_id": recording_id,
        "duration_variant": duration_variant,
        "feature_id": feature_id,
        "family": family,
        "domain": domain,
        "band": band,
        "metric": metric,
        "value": float(value),
        "n_epochs": int(n_epochs),
        "n_electrodes": int(n_electrodes),
        "electrode_aggregation": aggregation,
    }


def _electrode_rows(
    recording_id: str,
    duration_variant: str,
    electrode: str,
    family: str,
    domain: str,
    band: str,
    metrics: dict[str, float],
    *,
    n_epochs: int,
) -> list[dict[str, Any]]:
    rows = []
    for metric, value in metrics.items():
        feature_id = f"{domain}_{metric}" if band == "broadband" else f"{domain}_{band}_{metric}"
        rows.append(
            {
                "recording_id": recording_id,
                "duration_variant": duration_variant,
                "electrode": electrode,
                "feature_id": feature_id,
                "family": family,
                "domain": domain,
                "band": band,
                "metric": metric,
                "value": float(value),
                "n_epochs": int(n_epochs),
            }
        )
    return rows


def _evenly_spaced_epochs(data: np.ndarray, count: int) -> np.ndarray:
    if count > len(data):
        raise ValueError("Cannot retain more equalized epochs than are available")
    if count == len(data):
        return data
    indices = np.linspace(0, len(data) - 1, num=count, dtype=int)
    if len(np.unique(indices)) != count:
        raise RuntimeError("Equal-duration epoch selection produced duplicate indices")
    return data[indices]


def _analyze_variant(
    data: np.ndarray,
    channel_names: list[str],
    *,
    recording_id: str,
    duration_variant: str,
    sfreq: float,
    config: dict[str, Any],
    include_ordinal: bool,
    include_bouts: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    subject_rows: list[dict[str, Any]] = []
    electrode_rows: list[dict[str, Any]] = []
    psd_rows: list[dict[str, Any]] = []
    n_epochs, n_electrodes = int(data.shape[0]), int(data.shape[1])
    psd_config = config["psd"]
    frequencies, electrode_psd = compute_subject_electrode_psd(
        data,
        sfreq,
        fmin=float(psd_config["fmin_hz"]),
        fmax=float(psd_config["fmax_hz"]),
    )
    bands = {
        str(name): tuple(float(value) for value in limits)
        for name, limits in config["bands"].items()
    }
    absolute = integrate_bands(frequencies, electrode_psd, bands)
    relative, total = relative_band_powers(
        frequencies,
        electrode_psd,
        bands,
        total_range=(float(psd_config["fmin_hz"]), float(psd_config["fmax_hz"])),
    )
    for electrode_index, electrode in enumerate(channel_names):
        electrode_rows.extend(
            _electrode_rows(
                recording_id,
                duration_variant,
                electrode,
                "psd",
                "psd",
                "broadband",
                {"total_power_uv2": total[electrode_index]},
                n_epochs=n_epochs,
            )
        )
        for band in bands:
            electrode_rows.extend(
                _electrode_rows(
                    recording_id,
                    duration_variant,
                    electrode,
                    "psd",
                    "psd",
                    band,
                    {
                        "absolute_power_uv2": absolute[band][electrode_index],
                        "relative_power": relative[band][electrode_index],
                    },
                    n_epochs=n_epochs,
                )
            )
    subject_rows.append(
        _subject_row(
            recording_id,
            duration_variant,
            "psd",
            "psd",
            "broadband",
            "total_power_uv2",
            _finite_aggregate(total, "median"),
            n_epochs=n_epochs,
            n_electrodes=n_electrodes,
            aggregation="median",
        )
    )
    for band in bands:
        for metric, values in (
            ("absolute_power_uv2", absolute[band]),
            ("relative_power", relative[band]),
        ):
            subject_rows.append(
                _subject_row(
                    recording_id,
                    duration_variant,
                    "psd",
                    "psd",
                    band,
                    metric,
                    _finite_aggregate(values, "median"),
                    n_epochs=n_epochs,
                    n_electrodes=n_electrodes,
                    aggregation="median",
                )
            )
    subject_psd = np.median(electrode_psd, axis=0)
    for frequency, value in zip(frequencies, subject_psd):
        psd_rows.append(
            {
                "recording_id": recording_id,
                "duration_variant": duration_variant,
                "frequency_hz": float(frequency),
                "median_psd_uv2_hz": float(value),
                "n_epochs": n_epochs,
                "n_electrodes": n_electrodes,
            }
        )

    if include_ordinal:
        ordinal = config["ordinal"]
        ordinal_inputs: list[tuple[str, np.ndarray]] = [("broadband", data)]
        for band in ordinal["bands"]:
            low, high = bands[str(band)]
            ordinal_inputs.append(
                (
                    str(band),
                    filter_epoch_data(
                        data,
                        sfreq=sfreq,
                        low_hz=low,
                        high_hz=high,
                        order=int(ordinal["bandpass_filter_order"]),
                    ),
                )
            )
        for band, ordinal_data in ordinal_inputs:
            table = analyze_epoch_data(
                ordinal_data,
                channel_names,
                subject_id=recording_id,
                group="session_recording",
                sfreq=sfreq,
                dx=int(ordinal["embedding_dimension"]),
                tau=int(ordinal["delay_samples"]),
                tie_precision=None,
            )
            for row in table.itertuples(index=False):
                values = {metric: float(getattr(row, metric)) for metric in CORE_ORDINAL_METRICS}
                electrode_rows.extend(
                    _electrode_rows(
                        recording_id,
                        duration_variant,
                        str(row.electrode),
                        "ordinal",
                        "ordinal",
                        band,
                        values,
                        n_epochs=n_epochs,
                    )
                )
            for metric in CORE_ORDINAL_METRICS:
                subject_rows.append(
                    _subject_row(
                        recording_id,
                        duration_variant,
                        "ordinal",
                        "ordinal",
                        band,
                        metric,
                        _finite_aggregate(table[metric], "mean"),
                        n_epochs=n_epochs,
                        n_electrodes=n_electrodes,
                        aggregation="mean",
                    )
                )

    specparam_settings = config["specparam"]
    bout_settings = config["ebosc"]
    wavelet_frequencies = np.arange(
        float(bout_settings["frequency_min_hz"]),
        float(bout_settings["frequency_max_hz"])
        + 0.5 * float(bout_settings["frequency_step_hz"]),
        float(bout_settings["frequency_step_hz"]),
    )
    aperiodic_by_metric: dict[str, list[float]] = {metric: [] for metric in APERIODIC_METRICS}
    fit_qc_passes: list[float] = []
    fit_qc_exponents: list[float] = []
    peak_by_band_metric: dict[tuple[str, str], list[float]] = {}
    bout_by_band_metric: dict[tuple[str, str], list[float]] = {}
    for electrode_index, electrode in enumerate(channel_names):
        try:
            aperiodic, peak_rows, curves = fit_specparam_spectrum(
                frequencies,
                electrode_psd[electrode_index],
                bands,
                specparam_settings,
            )
        except Exception:
            aperiodic = {metric: np.nan for metric in APERIODIC_METRICS}
            peak_rows = []
            curves = None
        aperiodic_values = {
            metric: float(aperiodic.get(metric, np.nan)) for metric in APERIODIC_METRICS
        }
        fit_qc = config["aperiodic_fit_qc"]
        if curves is None:
            maximum_residual = np.nan
        else:
            observed = np.log10(
                np.maximum(curves["observed_psd_uv2_hz"], np.finfo(float).tiny)
            )
            modeled = np.log10(
                np.maximum(curves["modeled_psd_uv2_hz"], np.finfo(float).tiny)
            )
            maximum_residual = float(np.max(np.abs(observed - modeled)))
        exponent_low, exponent_high = (
            float(value) for value in fit_qc["exponent_range"]
        )
        fit_pass = bool(
            np.isfinite(aperiodic_values["aperiodic_exponent"])
            and exponent_low
            <= aperiodic_values["aperiodic_exponent"]
            <= exponent_high
            and aperiodic_values["specparam_r_squared"]
            >= float(fit_qc["minimum_r_squared"])
            and aperiodic_values["specparam_error_mae"]
            <= float(fit_qc["maximum_error_mae_log10"])
            and np.isfinite(maximum_residual)
            and maximum_residual
            <= float(fit_qc["maximum_absolute_residual_log10"])
        )
        electrode_rows.extend(
            _electrode_rows(
                recording_id,
                duration_variant,
                electrode,
                "aperiodic",
                "aperiodic",
                "broadband",
                aperiodic_values,
                n_epochs=n_epochs,
            )
        )
        for metric, value in aperiodic_values.items():
            aperiodic_by_metric[metric].append(value)
        fit_qc_passes.append(float(fit_pass))
        if fit_pass:
            fit_qc_exponents.append(aperiodic_values["aperiodic_exponent"])
        electrode_rows.extend(
            _electrode_rows(
                recording_id,
                duration_variant,
                electrode,
                "aperiodic_qc",
                "aperiodic",
                "broadband",
                {
                    "fit_qc_pass": float(fit_pass),
                    "maximum_absolute_residual_log10": maximum_residual,
                    "aperiodic_exponent_qc": (
                        aperiodic_values["aperiodic_exponent"] if fit_pass else np.nan
                    ),
                },
                n_epochs=n_epochs,
            )
        )
        for peak in peak_rows:
            band = str(peak["band"])
            for metric in ("peak_present", "peak_frequency_hz", "peak_power_linear"):
                value = float(peak.get(metric, np.nan))
                peak_by_band_metric.setdefault((band, metric), []).append(value)
                electrode_rows.extend(
                    _electrode_rows(
                        recording_id,
                        duration_variant,
                        electrode,
                        "periodic_peak",
                        "periodic_peak",
                        band,
                        {metric: value},
                        n_epochs=n_epochs,
                    )
                )
        if include_bouts and curves is not None:
            wavelet_power = ebosc_wavelet_power(
                data[:, electrode_index, :],
                sfreq=sfreq,
                frequencies=wavelet_frequencies,
                wavenumber=float(bout_settings["wavenumber"]),
            )
            background = aperiodic_wavelet_background(
                curves["frequencies_hz"],
                curves["modeled_psd_uv2_hz"],
                curves["aperiodic_psd_uv2_hz"],
                wavelet_frequencies,
                wavelet_power.mean(axis=(0, 2)) * 1e12,
            )
            thresholds = power_thresholds(
                background, float(bout_settings["power_percentile"])
            ) / 1e12
            edge_padding_samples = int(
                round(float(bout_settings["edge_padding_sec"]) * sfreq)
            )
            detected = detect_frequency_episodes(
                wavelet_power,
                sfreq=sfreq,
                frequencies=wavelet_frequencies,
                thresholds=thresholds,
                minimum_cycles=float(bout_settings["minimum_cycles"]),
                edge_padding_samples=edge_padding_samples,
            )
            for band in bout_settings["bands"]:
                episodes, band_mask = extract_band_bouts(
                    detected,
                    wavelet_power,
                    thresholds,
                    wavelet_frequencies,
                    band=str(band),
                    band_limits=bands[str(band)],
                    sfreq=sfreq,
                )
                summary = summarize_bouts(
                    episodes,
                    band_mask,
                    sfreq=sfreq,
                    edge_padding_samples=edge_padding_samples,
                )
                values = {metric: float(summary[metric]) for metric in BOUT_METRICS}
                electrode_rows.extend(
                    _electrode_rows(
                        recording_id,
                        duration_variant,
                        electrode,
                        "bouts",
                        "bouts",
                        str(band),
                        values,
                        n_epochs=n_epochs,
                    )
                )
                for metric, value in values.items():
                    bout_by_band_metric.setdefault((str(band), metric), []).append(value)

    for metric, values in aperiodic_by_metric.items():
        subject_rows.append(
            _subject_row(
                recording_id,
                duration_variant,
                "aperiodic",
                "aperiodic",
                "broadband",
                metric,
                _finite_aggregate(values, "mean"),
                n_epochs=n_epochs,
                n_electrodes=n_electrodes,
                aggregation="mean",
            )
        )
    fit_qc_fraction = _finite_aggregate(fit_qc_passes, "mean")
    subject_rows.append(
        _subject_row(
            recording_id,
            duration_variant,
            "aperiodic_qc",
            "aperiodic",
            "broadband",
            "fit_qc_fraction",
            fit_qc_fraction,
            n_epochs=n_epochs,
            n_electrodes=n_electrodes,
            aggregation="mean",
        )
    )
    subject_rows.append(
        _subject_row(
            recording_id,
            duration_variant,
            "aperiodic_qc",
            "aperiodic",
            "broadband",
            "aperiodic_exponent_qc",
            (
                _finite_aggregate(fit_qc_exponents, "mean")
                if fit_qc_fraction
                >= float(config["aperiodic_fit_qc"]["minimum_subject_qc_fraction"])
                else np.nan
            ),
            n_epochs=n_epochs,
            n_electrodes=n_electrodes,
            aggregation="mean_qc_qualified_electrodes",
        )
    )
    for (band, metric), values in peak_by_band_metric.items():
        subject_rows.append(
            _subject_row(
                recording_id,
                duration_variant,
                "periodic_peak",
                "periodic_peak",
                band,
                metric,
                _finite_aggregate(values, "mean"),
                n_epochs=n_epochs,
                n_electrodes=n_electrodes,
                aggregation="mean",
            )
        )
    for (band, metric), values in bout_by_band_metric.items():
        subject_rows.append(
            _subject_row(
                recording_id,
                duration_variant,
                "bouts",
                "bouts",
                band,
                metric,
                _finite_aggregate(values, "mean"),
                n_epochs=n_epochs,
                n_electrodes=n_electrodes,
                aggregation="mean",
            )
        )
    return subject_rows, electrode_rows, psd_rows


def extract_features(
    config: dict[str, Any],
    recordings: pd.DataFrame,
    *,
    subjects: list[str] | None = None,
    include_ordinal: bool = True,
    include_bouts: bool = True,
) -> dict[str, pd.DataFrame]:
    """Extract all configured features from cleaned recording-level epochs."""
    inputs = config["input"]
    files = discover_epoch_files(inputs["epochs_dir"], inputs["epoch_glob"])
    selected = recordings.copy()
    if subjects:
        requested = set(subjects)
        selected = selected.loc[
            selected["recording_id"].isin(requested)
            | selected["participant_id"].isin(requested)
        ].copy()
        found = set(selected["recording_id"]) | set(selected["participant_id"])
        missing = sorted(requested - found)
        if missing:
            raise ValueError(f"Unknown requested participants/recordings: {missing}")
    missing_files = sorted(set(selected["recording_id"]) - set(files))
    if missing_files:
        raise FileNotFoundError(
            f"Missing cleaned ds002778 epochs for: {missing_files[:10]}"
        )

    channel_inventory: dict[str, list[str]] = {}
    epoch_counts: dict[str, int] = {}
    for recording_id in selected["recording_id"]:
        epochs = mne.read_epochs(files[recording_id], preload=False, verbose="ERROR")
        picks = mne.pick_types(epochs.info, eeg=True, exclude=[])
        channel_inventory[recording_id] = [epochs.ch_names[pick] for pick in picks]
        epoch_counts[recording_id] = len(epochs)
    common_channels, union_channels = ordered_channel_inventory(channel_inventory)
    minimum_required = int(config["duration_sensitivity"]["minimum_retained_epochs"])
    if min(epoch_counts.values()) < minimum_required:
        offenders = {
            key: value for key, value in epoch_counts.items() if value < minimum_required
        }
        raise ValueError(
            f"Recordings below minimum retained-epoch threshold {minimum_required}: {offenders}"
        )
    equalized_count = min(epoch_counts.values())
    variants = [("all_retained", None)]
    if bool(config["duration_sensitivity"]["enabled"]):
        variants.append(("equalized_duration", equalized_count))

    subject_rows: list[dict[str, Any]] = []
    electrode_rows: list[dict[str, Any]] = []
    psd_rows: list[dict[str, Any]] = []
    input_rows: list[dict[str, Any]] = []
    for recording_id in tqdm(
        selected["recording_id"],
        desc="ds002778 EEG features",
        unit="recording",
        dynamic_ncols=True,
    ):
        epochs = mne.read_epochs(files[recording_id], preload=True, verbose="ERROR")
        picks = [epochs.ch_names.index(channel) for channel in common_channels]
        data = epochs.get_data(picks=picks, copy=True)
        sfreq = float(epochs.info["sfreq"])
        input_rows.append(
            {
                "recording_id": recording_id,
                "epoch_file": str(files[recording_id].resolve()),
                "n_epochs_retained": len(data),
                "equalized_n_epochs": equalized_count,
                "sampling_frequency_hz": sfreq,
                "n_common_electrodes": len(common_channels),
                "n_available_electrodes": len(channel_inventory[recording_id]),
            }
        )
        for variant, count in variants:
            selected_data = data if count is None else _evenly_spaced_epochs(data, count)
            subject, electrode, psd = _analyze_variant(
                selected_data,
                common_channels,
                recording_id=recording_id,
                duration_variant=variant,
                sfreq=sfreq,
                config=config,
                include_ordinal=include_ordinal and bool(config["ordinal"]["enabled"]),
                include_bouts=include_bouts and bool(config["ebosc"]["enabled"]),
            )
            subject_rows.extend(subject)
            electrode_rows.extend(electrode)
            psd_rows.extend(psd)

    subject_table = pd.DataFrame.from_records(subject_rows)
    electrode_table = pd.DataFrame.from_records(electrode_rows)
    psd_table = pd.DataFrame.from_records(psd_rows)
    dictionary = (
        subject_table[
            ["feature_id", "family", "domain", "band", "metric", "electrode_aggregation"]
        ]
        .drop_duplicates()
        .sort_values(["family", "feature_id"])
        .reset_index(drop=True)
    )
    inventory = {
        "common_channels": common_channels,
        "union_channels": union_channels,
        "n_common_channels": len(common_channels),
        "n_union_channels": len(union_channels),
        "equalized_n_epochs": equalized_count,
    }
    return {
        "subject_features": subject_table,
        "electrode_features": electrode_table,
        "subject_psd": psd_table,
        "feature_dictionary": dictionary,
        "input_epochs": pd.DataFrame.from_records(input_rows),
        "inventory": pd.DataFrame([{"payload": json.dumps(inventory)}]),
    }
