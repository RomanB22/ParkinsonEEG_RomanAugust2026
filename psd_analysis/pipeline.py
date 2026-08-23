"""End-to-end subject-balanced PSD analysis of cleaned EEG epochs."""

from __future__ import annotations

import json
import logging
import platform
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.runtime import configure_runtime

configure_runtime()

import matplotlib
import mne
import numpy as np
import pandas as pd
import scipy

from .metrics import bootstrap_median_ci, compute_subject_electrode_psd, integrate_bands, to_db
from .plots import plot_group_band_topomaps, plot_group_median_psd


SUBJECT_PATTERN = re.compile(r"(sub-\d+)")


def load_psd_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as stream:
        config = json.load(stream)
    required = {"input", "output_dir", "psd", "bands", "bootstrap", "plots"}
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Missing PSD-analysis config sections: {missing}")
    psd = config["psd"]
    fixed_choices = {
        "method": "welch",
        "window": "hann",
        "epoch_combination": "concatenate_in_temporal_order",
        "welch_aggregation": "mean",
    }
    for name, expected in fixed_choices.items():
        if str(psd.get(name, "")).lower() != expected:
            raise ValueError(f"psd.{name} must be {expected!r}")
    fmin, fmax = float(psd["fmin_hz"]), float(psd["fmax_hz"])
    if not 0 <= fmin < fmax:
        raise ValueError("psd requires 0 <= fmin_hz < fmax_hz")
    if int(config["bootstrap"]["n_resamples"]) < 100:
        raise ValueError("bootstrap.n_resamples must be at least 100")
    for name, limits in config["bands"].items():
        low, high = (float(value) for value in limits)
        if not fmin <= low < high <= fmax:
            raise ValueError(f"Band {name} must be contained in the PSD interval")
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
    files = {}
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
    logger = logging.getLogger("psd_analysis")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (
        logging.StreamHandler(),
        logging.FileHandler(output_dir / "psd_analysis.log", mode="w" if overwrite else "a"),
    ):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def _write_csv(table: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False, float_format="%.17g")


def run_analysis(
    config_path: str | Path,
    *,
    subjects: list[str] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    config_path = Path(config_path)
    config = load_psd_config(config_path)
    output_dir = Path(config["output_dir"])
    result_path = output_dir / "metrics" / "group_median_psd.csv"
    if result_path.exists() and not overwrite:
        raise FileExistsError(f"PSD outputs exist at {result_path}; rerun with --overwrite")
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

    group_lookup = participant_table.set_index("participant_id")["GROUP"].astype(str).to_dict()
    fmin = float(config["psd"]["fmin_hz"])
    fmax = float(config["psd"]["fmax_hz"])
    logger.info("Starting PSD analysis | subjects=%d | %.2f-%.2f Hz", len(expected_subjects), fmin, fmax)

    subject_psds: dict[str, dict[str, np.ndarray]] = {}
    subject_infos: dict[str, mne.Info] = {}
    input_rows = []
    frequencies = None
    for index, subject_id in enumerate(expected_subjects, start=1):
        path = files[subject_id]
        logger.info("[%d/%d] %s | %s", index, len(expected_subjects), subject_id, path)
        epochs = mne.read_epochs(path, preload=True, verbose="ERROR")
        picks = mne.pick_types(epochs.info, eeg=True, exclude=[])
        data = epochs.get_data(picks=picks, copy=True)
        names = [epochs.ch_names[pick] for pick in picks]
        current_frequencies, electrode_psd = compute_subject_electrode_psd(
            data,
            float(epochs.info["sfreq"]),
            fmin=fmin,
            fmax=fmax,
        )
        if frequencies is None:
            frequencies = current_frequencies
        elif not np.array_equal(frequencies, current_frequencies):
            raise ValueError(f"{subject_id}: PSD frequency grid differs from prior subjects")
        subject_psds[subject_id] = {
            name: electrode_psd[channel_index] for channel_index, name in enumerate(names)
        }
        info = mne.pick_info(epochs.info, picks, copy=True)
        info["bads"] = []
        subject_infos[subject_id] = info
        input_rows.append(
            {
                "subject_id": subject_id,
                "group": group_lookup[subject_id],
                "epoch_file": str(path.resolve()),
                "n_epochs": len(epochs),
                "n_electrodes": len(names),
                "samples_per_epoch": data.shape[2],
                "concatenated_samples_per_electrode": int(len(epochs) * data.shape[2]),
                "concatenated_duration_sec": float(len(epochs) * data.shape[2] / epochs.info["sfreq"]),
                "sampling_frequency_hz": float(epochs.info["sfreq"]),
                "frequency_resolution_hz": float(current_frequencies[1] - current_frequencies[0]),
            }
        )
    assert frequencies is not None

    electrode_union: list[str] = []
    common = set(next(iter(subject_infos.values())).ch_names)
    for info in subject_infos.values():
        common.intersection_update(info.ch_names)
        for channel in info.ch_names:
            if channel not in electrode_union:
                electrode_union.append(channel)
    common_channels = [
        channel for channel in next(iter(subject_infos.values())).ch_names if channel in common
    ]
    union_index = {channel: index for index, channel in enumerate(electrode_union)}
    cube = np.full(
        (len(expected_subjects), len(electrode_union), len(frequencies)), np.nan, dtype=np.float64
    )
    for subject_index, subject_id in enumerate(expected_subjects):
        for electrode, values in subject_psds[subject_id].items():
            cube[subject_index, union_index[electrode], :] = values

    common_indices = [union_index[channel] for channel in common_channels]
    subject_global_psd = np.median(cube[:, common_indices, :], axis=1)
    subject_global_rows = []
    for subject_index, subject_id in enumerate(expected_subjects):
        for frequency_index, frequency in enumerate(frequencies):
            value = float(subject_global_psd[subject_index, frequency_index])
            subject_global_rows.append(
                {
                    "subject_id": subject_id,
                    "group": group_lookup[subject_id],
                    "frequency_hz": float(frequency),
                    "median_across_common_electrodes_psd_uv2_hz": value,
                    "median_across_common_electrodes_psd_db_uv2_hz": float(to_db(value)),
                }
            )
    subject_global_table = pd.DataFrame.from_records(subject_global_rows)

    configured_groups = [str(group) for group in config["plots"]["group_order"]]
    present_groups = {group_lookup[subject_id] for subject_id in expected_subjects}
    group_order = [group for group in configured_groups if group in present_groups]
    group_order.extend(sorted(present_groups - set(group_order)))
    bootstrap = config["bootstrap"]
    group_psd_rows = []
    subject_groups = np.asarray([group_lookup[subject_id] for subject_id in expected_subjects])
    for group_index, group in enumerate(group_order):
        values = subject_global_psd[subject_groups == group]
        median, lower, upper = bootstrap_median_ci(
            values,
            n_resamples=int(bootstrap["n_resamples"]),
            confidence_level=float(bootstrap["confidence_level"]),
            seed=int(bootstrap["random_seed"]) + group_index,
        )
        for frequency, center, low, high in zip(frequencies, median, lower, upper):
            group_psd_rows.append(
                {
                    "group": group,
                    "n_subjects": len(values),
                    "frequency_hz": float(frequency),
                    "median_psd_uv2_hz": float(center),
                    "ci_lower_psd_uv2_hz": float(low),
                    "ci_upper_psd_uv2_hz": float(high),
                    "median_psd_db_uv2_hz": float(to_db(center)),
                    "ci_lower_psd_db_uv2_hz": float(to_db(low)),
                    "ci_upper_psd_db_uv2_hz": float(to_db(high)),
                }
            )
    group_psd_table = pd.DataFrame.from_records(group_psd_rows)

    bands = {name: tuple(limits) for name, limits in config["bands"].items()}
    band_arrays = integrate_bands(frequencies, cube, bands)
    subject_band_rows = []
    for subject_index, subject_id in enumerate(expected_subjects):
        for electrode_index, electrode in enumerate(electrode_union):
            if np.isnan(cube[subject_index, electrode_index]).all():
                continue
            for band, values in band_arrays.items():
                power = float(values[subject_index, electrode_index])
                subject_band_rows.append(
                    {
                        "subject_id": subject_id,
                        "group": group_lookup[subject_id],
                        "electrode": electrode,
                        "band": band,
                        "band_low_hz": float(bands[band][0]),
                        "band_high_hz": float(bands[band][1]),
                        "band_power_uv2": power,
                        "band_power_db_uv2": float(to_db(power)),
                    }
                )
    subject_band_table = pd.DataFrame.from_records(subject_band_rows)
    group_band_rows = []
    for (group, electrode, band), selected in subject_band_table.groupby(
        ["group", "electrode", "band"], sort=True
    ):
        values = selected["band_power_uv2"].to_numpy(dtype=float)
        group_band_rows.append(
            {
                "group": group,
                "electrode": electrode,
                "band": band,
                "band_low_hz": float(selected["band_low_hz"].iloc[0]),
                "band_high_hz": float(selected["band_high_hz"].iloc[0]),
                "n_subjects": len(values),
                "median_band_power_uv2": float(np.median(values)),
                "median_band_power_db_uv2": float(to_db(np.median(values))),
                "iqr_lower_band_power_uv2": float(np.quantile(values, 0.25)),
                "iqr_upper_band_power_uv2": float(np.quantile(values, 0.75)),
            }
        )
    group_band_table = pd.DataFrame.from_records(group_band_rows)

    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(pd.DataFrame.from_records(input_rows), metrics_dir / "analyzed_inputs.csv")
    _write_csv(subject_global_table, metrics_dir / "subject_global_psd.csv")
    _write_csv(group_psd_table, metrics_dir / "group_median_psd.csv")
    _write_csv(subject_band_table, metrics_dir / "subject_electrode_band_power.csv")
    _write_csv(group_band_table, metrics_dir / "group_electrode_band_power.csv")
    np.savez_compressed(
        metrics_dir / "subject_electrode_psd.npz",
        subject_ids=np.asarray(expected_subjects),
        groups=subject_groups,
        electrodes=np.asarray(electrode_union),
        frequencies_hz=frequencies,
        psd_uv2_hz=cube,
    )

    colors_config = config["plots"]["group_colors"]
    fallback = ("#D55E00", "#0072B2", "#009E73", "#CC79A7")
    colors = {
        group: str(colors_config.get(group, fallback[index % len(fallback)]))
        for index, group in enumerate(group_order)
    }
    dpi = int(config["plots"]["dpi"])
    logger.info("Creating group median PSD confidence-band figure")
    plot_group_median_psd(
        group_psd_table,
        group_order,
        colors,
        output_dir / "figures" / "group_median_psd_with_ci.png",
        dpi,
    )
    first_info = next(iter(subject_infos.values()))
    common_picks = [first_info.ch_names.index(channel) for channel in common_channels]
    common_info = mne.pick_info(first_info, common_picks, copy=True)
    common_info["bads"] = []
    display_names = config["plots"].get("band_display_names", {})
    band_labels = {
        band: f"{display_names.get(band, band.replace('_', ' ').title())}\n"
        f"{limits[0]:g}–{limits[1]:g} Hz"
        for band, limits in bands.items()
    }
    logger.info("Creating group band-power topomaps")
    topomap_limits = plot_group_band_topomaps(
        group_band_table.loc[group_band_table["electrode"].isin(common_channels)],
        common_info,
        list(bands),
        band_labels,
        group_order,
        output_dir / "figures" / "group_median_band_power_topomaps.png",
        dpi,
    )

    electrode_payload = {
        "common_electrodes": common_channels,
        "n_common_electrodes": len(common_channels),
        "electrode_union": electrode_union,
        "n_electrode_union": len(electrode_union),
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
        },
        "n_subjects": len(expected_subjects),
        "group_counts": pd.Series(subject_groups).value_counts().to_dict(),
        "n_common_electrodes": len(common_channels),
        "n_electrode_union": len(electrode_union),
        "frequency_bins": len(frequencies),
        "frequency_resolution_hz": float(frequencies[1] - frequencies[0]),
        "aggregation": (
            "Accepted epochs are concatenated in stored temporal order for each electrode. "
            "One Welch call uses non-overlapping four-second Hann windows and mean Welch "
            "aggregation to produce each subject/electrode PSD; there is no median-across-"
            "epochs step. The global subject PSD is the median across 60 common electrodes, "
            "and the group curve is the median across subjects. Conversion to dB occurs only "
            "after aggregation."
        ),
        "confidence_interval": (
            "Pointwise nonparametric percentile bootstrap of subjects around the group median."
        ),
        "topomap": (
            "Absolute band power is integrated from each concatenated subject/electrode linear PSD; "
            "group maps show the electrode-wise subject median on 60 common electrodes."
        ),
        "topomap_db_limits": {
            band: [float(value) for value in limits] for band, limits in topomap_limits.items()
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    logger.info("PSD analysis completed | output=%s", output_dir)
    return manifest
