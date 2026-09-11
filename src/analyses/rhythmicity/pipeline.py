"""All-electrode rhythmicity analysis using the Python LAVI toolbox.

The module deliberately lives outside the global pipeline.  It consumes the
existing cleaned epochs and global recording-level burst metrics, and writes a
separate output tree so rhythmicity can be enabled or omitted independently.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator, NullFormatter
import mne
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, ttest_ind, t as student_t
from tqdm.auto import tqdm

from core.runtime import configure_runtime

configure_runtime()


DEFAULT_BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 50.0),
}

BAND_COLORS = {
    "delta": "#4C78A8",
    "theta": "#72B7B2",
    "alpha": "#F2CF5B",
    "beta": "#F58518",
    "gamma": "#E45756",
}

GROUP_COLORS = {"Control": "#7f7f7f", "PD": "#d95f02", "PD_OFF": "#7570b3", "PD_ON": "#1b9e77"}

# “Significant” means FDR-adjusted q < .05.  “Relevant” additionally requires
# a minimum practical effect size, avoiding emphasis on tiny but precise effects.
FDR_ALPHA = 0.05
RELEVANT_ABS_RHO = 0.30
RELEVANT_ABS_HEDGES_G = 0.50


def _significance_stars(q_value: float) -> str:
    """Return conventional significance stars for an FDR-adjusted q value."""
    if not np.isfinite(q_value) or q_value >= FDR_ALPHA:
        return ""
    if q_value < 0.001:
        return "***"
    if q_value < 0.01:
        return "**"
    return "*"


def _ordered_bands(values: Any) -> list[str]:
    names = list(values)
    return [name for name in DEFAULT_BANDS if name in names] + [name for name in names if name not in DEFAULT_BANDS]


@dataclass(frozen=True)
class RecordingTask:
    dataset: str
    recording_id: str
    participant_id: str
    session_id: str
    group: str
    epoch_path: str
    profile_path: str


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_config(config: dict[str, Any]) -> None:
    required = {"global_output_root", "output_dir", "lavi", "bands"}
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Rhythmicity config is missing: {missing}")
    lavi = config["lavi"]
    if float(lavi["lag_cycles"]) <= 0 or float(lavi["wavelet_width"]) <= 0:
        raise ValueError("LAVI lag_cycles and wavelet_width must be positive")
    if int(lavi["n_frequencies"]) < 10:
        raise ValueError("LAVI frequency grid is too small")
    if not float(lavi["min_frequency_hz"]) < float(lavi["max_frequency_hz"]):
        raise ValueError("LAVI frequency limits are invalid")
    if int(config.get("workers", 1)) < 1:
        raise ValueError("workers must be positive")


def _frequency_grid(config: dict[str, Any]) -> np.ndarray:
    lavi = config["lavi"]
    return np.geomspace(
        float(lavi["min_frequency_hz"]),
        float(lavi["max_frequency_hz"]),
        int(lavi["n_frequencies"]),
    )


def _load_tasks(
    config: dict[str, Any],
    datasets: list[str] | None,
    output: Path,
    recordings: list[str] | None = None,
) -> list[RecordingTask]:
    global_root = Path(config["global_output_root"])
    canonical_root = global_root / "canonical"
    selected = set(datasets) if datasets else None
    selected_recordings = set(recordings) if recordings else None
    tasks: list[RecordingTask] = []
    for manifest_path in sorted(canonical_root.glob("*/recordings.csv.gz")):
        dataset = manifest_path.parent.name
        if selected is not None and dataset not in selected:
            continue
        table = pd.read_csv(manifest_path, low_memory=False)
        profile_root = output / "profiles" / dataset
        profile_root.mkdir(parents=True, exist_ok=True)
        for _, row in table.iterrows():
            epoch_path = str(row.get("epoch_path", ""))
            if not epoch_path or not Path(epoch_path).is_file():
                continue
            recording_id = str(row["recording_id"])
            if selected_recordings is not None and recording_id not in selected_recordings:
                continue
            tasks.append(
                RecordingTask(
                    dataset=dataset,
                    recording_id=recording_id,
                    participant_id=str(row["participant_id"]),
                    session_id="" if pd.isna(row.get("session_id")) else str(row["session_id"]),
                    group=str(row.get("group", "")),
                    epoch_path=epoch_path,
                    profile_path=str(profile_root / f"{recording_id}.npz"),
                )
            )
    if not tasks:
        raise FileNotFoundError("No readable canonical recording epoch files were found")
    return tasks


def _band_summaries(
    lavi: np.ndarray,
    foi: np.ndarray,
    channels: list[str],
    bands: dict[str, tuple[float, float]],
    sigvect: list[np.ndarray],
    task: RecordingTask,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    electrode_rows: list[dict[str, Any]] = []
    abba_rows: list[dict[str, Any]] = []
    for ci, channel in enumerate(channels):
        profile = lavi[ci]
        valid = np.isfinite(profile)
        if not valid.any():
            continue
        for band, (low, high) in bands.items():
            mask = valid & (foi >= low) & (foi <= high)
            if not mask.any():
                continue
            band_values = profile[mask]
            band_freqs = foi[mask]
            peak_local = int(np.nanargmax(band_values))
            status = np.asarray(sigvect[ci], dtype=float)[mask]
            electrode_rows.append(
                {
                    "dataset": task.dataset,
                    "recording_id": task.recording_id,
                    "participant_id": task.participant_id,
                    "session_id": task.session_id,
                    "group": task.group,
                    "electrode": channel,
                    "band": band,
                    "lavi_mean": float(np.nanmean(band_values)),
                    "lavi_median": float(np.nanmedian(band_values)),
                    "lavi_peak": float(band_values[peak_local]),
                    "lavi_peak_frequency_hz": float(band_freqs[peak_local]),
                    "high_rhythmicity_fraction": float(np.mean(status > 0)),
                    "low_rhythmicity_fraction": float(np.mean(status < 0)),
                    "n_frequencies": int(mask.sum()),
                }
            )
            for direction, label in ((1, "high"), (-1, "low")):
                indices = np.where(mask & (np.asarray(sigvect[ci]) == direction / 2))[0]
                if len(indices):
                    abba_rows.append(
                        {
                            "dataset": task.dataset,
                            "recording_id": task.recording_id,
                            "participant_id": task.participant_id,
                            "session_id": task.session_id,
                            "group": task.group,
                            "electrode": channel,
                            "band": band,
                            "direction": label,
                            "frequency_start_hz": float(foi[indices.min()]),
                            "frequency_end_hz": float(foi[indices.max()]),
                            "n_frequency_bins": int(len(indices)),
                        }
                    )
    return electrode_rows, abba_rows


def _process_recording(task: RecordingTask, config: dict[str, Any], foi: np.ndarray) -> dict[str, Any]:
    """Worker: calculate all-channel LAVI for one recording."""
    from lavi import abba, prepare_lavi

    epochs = mne.read_epochs(task.epoch_path, preload=True, verbose=False)
    if len(epochs) == 0:
        raise ValueError("no retained epochs after preprocessing")
    eeg_picks = mne.pick_types(epochs.info, eeg=True, exclude=[])
    channels = [epochs.ch_names[index] for index in eeg_picks]
    data = epochs.get_data(picks=eeg_picks, units="uV")
    n_epochs, n_channels, n_samples = data.shape
    flattened = data.transpose(1, 0, 2).reshape(n_channels, n_epochs * n_samples)
    valid_channels: list[int] = []
    for index, signal in enumerate(flattened):
        if np.isfinite(signal).all() and np.nanstd(signal) > 0:
            flattened[index] = signal - np.mean(signal)
            valid_channels.append(index)
    if not valid_channels:
        raise ValueError(f"No finite non-constant EEG channels in {task.epoch_path}")
    flattened = flattened[valid_channels]
    channels = [channels[index] for index in valid_channels]
    lavi, out_cfg = prepare_lavi(
        {
            "foi": foi,
            "fs": float(epochs.info["sfreq"]),
            "lag": float(config["lavi"]["lag_cycles"]),
            "width": float(config["lavi"]["wavelet_width"]),
            "verbose": False,
        },
        flattened,
    )
    surrogate_reps = int(config["lavi"].get("surrogate_reps", 0))
    if surrogate_reps > 0:
        from lavi import compute_pink_lavi

        pink = compute_pink_lavi(
            {
                "Pink_reps": surrogate_reps,
                "foi": out_cfg["foi"],
                "fs": float(epochs.info["sfreq"]),
                "lag": float(config["lavi"]["lag_cycles"]),
                "width": float(config["lavi"]["wavelet_width"]),
                "verbose": False,
            },
            flattened,
        )
        lower = np.nanpercentile(pink, 2.5, axis=0).T
        upper = np.nanpercentile(pink, 97.5, axis=0).T
        siglim = np.stack([lower, upper], axis=-1)
        borders, var_names, sigvect = abba(lavi, out_cfg["foi"], siglim=siglim)
    else:
        # The package's default ABBA mode uses the subject/channel median as
        # the baseline. This is fast enough for a full all-electrode pass;
        # IAAFT significance is available by setting surrogate_reps > 0.
        borders, var_names, sigvect = abba(lavi, out_cfg["foi"])
    profile_path = Path(task.profile_path)
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        profile_path,
        lavi=lavi.astype(np.float32),
        foi=np.asarray(out_cfg["foi"], dtype=np.float32),
        channels=np.asarray(channels),
        sigvect=np.asarray(sigvect, dtype=np.float32),
        sampling_frequency_hz=np.asarray([epochs.info["sfreq"]]),
        n_epochs=np.asarray([n_epochs]),
    )
    bands = {
        str(name): (float(limits[0]), float(limits[1]))
        for name, limits in config["bands"].items()
    }
    electrode_rows, abba_rows = _band_summaries(lavi, out_cfg["foi"], channels, bands, sigvect, task)
    return {
        "task": asdict(task),
        "electrode_rows": electrode_rows,
        "abba_rows": abba_rows,
        "profile_mean": np.nanmean(lavi, axis=0).astype(float).tolist(),
        "foi": np.asarray(out_cfg["foi"], dtype=float).tolist(),
        "n_channels": len(channels),
        "n_epochs": int(n_epochs),
        "sampling_frequency_hz": float(epochs.info["sfreq"]),
    }


def _load_cached_result(task: RecordingTask, config: dict[str, Any]) -> dict[str, Any]:
    """Reconstruct a worker result from a completed per-recording profile."""
    from lavi import abba

    with np.load(task.profile_path, allow_pickle=False) as cached:
        lavi = np.asarray(cached["lavi"], dtype=float)
        foi = np.asarray(cached["foi"], dtype=float)
        channels = [str(value) for value in cached["channels"].tolist()]
        sigvect = np.asarray(cached["sigvect"], dtype=float) if "sigvect" in cached else None
        n_epochs = int(cached["n_epochs"][0]) if "n_epochs" in cached else -1
        sfreq = float(cached["sampling_frequency_hz"][0]) if "sampling_frequency_hz" in cached else np.nan
    if sigvect is None:
        _, _, sigvect_list = abba(lavi, foi)
    else:
        sigvect_list = [row for row in sigvect]
    bands = {str(name): (float(limits[0]), float(limits[1])) for name, limits in config["bands"].items()}
    electrode_rows, abba_rows = _band_summaries(lavi, foi, channels, bands, sigvect_list, task)
    return {
        "task": asdict(task),
        "electrode_rows": electrode_rows,
        "abba_rows": abba_rows,
        "profile_mean": np.nanmean(lavi, axis=0).astype(float).tolist(),
        "foi": foi.tolist(),
        "n_channels": len(channels),
        "n_epochs": n_epochs,
        "sampling_frequency_hz": sfreq,
    }


def _load_burst_features(root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted((root / "metrics").glob("*/recording_features.csv.gz")):
        frame = pd.read_csv(path, low_memory=False)
        wanted = [c for c in frame.columns if c.startswith("bout__")]
        keys = [
            c for c in (
                "dataset_id", "recording_id", "participant_id", "session_id", "group", "electrode",
                "moca", "mmse", "age_years", "updrs",
            ) if c in frame
        ]
        rows.append(frame[keys + wanted].rename(columns={"dataset_id": "dataset"}))
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _aggregate_tables(electrode: pd.DataFrame, burst: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    numeric = ["lavi_mean", "lavi_median", "lavi_peak", "lavi_peak_frequency_hz", "high_rhythmicity_fraction", "low_rhythmicity_fraction"]
    participant_electrode = (
        electrode.groupby(["dataset", "participant_id", "group", "electrode", "band"], as_index=False)[numeric]
        .mean()
    )
    participant = (
        participant_electrode.groupby(["dataset", "participant_id", "group", "band"], as_index=False)[numeric]
        .mean()
    )
    if burst.empty:
        return participant_electrode, participant
    burst = burst.rename(columns={"dataset_id": "dataset"}) if "dataset_id" in burst else burst
    burst_band_rows = []
    burst_metrics = ["n_bouts", "oscillatory_occupancy", "bouts_per_minute", "duration_mean_s", "amplitude_mean", "cycles_mean"]
    for band in sorted(electrode["band"].unique()):
        subset = burst.copy()
        subset["band"] = band
        for metric in burst_metrics:
            source = f"bout__{band}__{metric}"
            if source in subset:
                subset[metric] = pd.to_numeric(subset[source], errors="coerce")
        keys = ["dataset", "recording_id", "participant_id", "group", "electrode", "band"]
        keep = keys + [metric for metric in burst_metrics if metric in subset]
        burst_band_rows.append(subset[keep])
    burst_long = pd.concat(burst_band_rows, ignore_index=True)
    merge_keys = ["dataset", "participant_id", "group", "electrode", "band"]
    burst_participant = burst_long.groupby(merge_keys, as_index=False)[burst_metrics].mean(numeric_only=True)
    participant_electrode = participant_electrode.merge(burst_participant, on=merge_keys, how="left")
    participant = participant_electrode.groupby(["dataset", "participant_id", "group", "band"], as_index=False)[numeric + burst_metrics].mean(numeric_only=True)
    # Clinical/demographic values are recording-level metadata.  Keep one
    # participant-level value (rather than repeating it once per electrode).
    clinical_columns = [column for column in ("moca", "mmse", "age_years", "updrs") if column in burst]
    if clinical_columns:
        clinical_keys = ["dataset", "participant_id", "group"]
        clinical = burst[clinical_keys + clinical_columns].copy()
        for column in clinical_columns:
            clinical[column] = pd.to_numeric(clinical[column], errors="coerce")
        clinical = clinical.groupby(clinical_keys, as_index=False)[clinical_columns].mean(numeric_only=True)
        participant = participant.merge(clinical, on=clinical_keys, how="left")
        participant_electrode = participant_electrode.merge(clinical, on=clinical_keys, how="left")
    return participant_electrode, participant


def _correlations(participant: pd.DataFrame) -> pd.DataFrame:
    burst_metrics = ["n_bouts", "oscillatory_occupancy", "bouts_per_minute", "duration_mean_s", "amplitude_mean", "cycles_mean"]
    rows: list[dict[str, Any]] = []
    for (dataset, band), frame in participant.groupby(["dataset", "band"]):
        for metric in burst_metrics:
            if metric not in frame:
                continue
            paired = frame[["lavi_mean", metric]].apply(pd.to_numeric, errors="coerce").dropna()
            if len(paired) < 5 or paired.nunique().min() < 2:
                rho, p_value = np.nan, np.nan
            else:
                rho, p_value = spearmanr(paired["lavi_mean"], paired[metric])
            rows.append({"dataset": dataset, "band": band, "burst_metric": metric, "n": len(paired), "rho": rho, "p_value": p_value})
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["q_fdr_bh"] = np.nan
    finite = result["p_value"].notna()
    for dataset in result.loc[finite, "dataset"].unique():
        index = result.index[finite & result["dataset"].eq(dataset)]
        p = result.loc[index, "p_value"].to_numpy(float)
        order = np.argsort(p)
        adjusted = np.minimum.accumulate((p[order] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
        q = np.empty_like(adjusted)
        q[order] = np.minimum(adjusted, 1.0)
        result.loc[index, "q_fdr_bh"] = q
    result["significant_fdr"] = result["q_fdr_bh"] < FDR_ALPHA
    result["relevant_effect"] = result["rho"].abs() >= RELEVANT_ABS_RHO
    result["significant_and_relevant"] = result["significant_fdr"] & result["relevant_effect"]
    return result


def _fdr_bh(values: pd.Series | np.ndarray) -> np.ndarray:
    """Benjamini–Hochberg adjusted p values, preserving input order."""
    p = np.asarray(values, dtype=float)
    output = np.full(p.shape, np.nan, dtype=float)
    finite = np.isfinite(p)
    if not finite.any():
        return output
    indices = np.where(finite)[0]
    order = np.argsort(p[finite])
    sorted_p = p[finite][order]
    adjusted = sorted_p * len(sorted_p) / np.arange(1, len(sorted_p) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    output[indices[order]] = np.minimum(adjusted, 1.0)
    return output


def _group_statistics(participant: pd.DataFrame) -> pd.DataFrame:
    """Welch group comparisons on participant-level mean LAVI per band."""
    rows: list[dict[str, Any]] = []
    bands = _ordered_bands(dict.fromkeys(participant["band"]))
    for (dataset, band), frame in participant.groupby(["dataset", "band"]):
        groups = set(frame["group"].dropna())
        candidate_pairs = [("PD", "Control"), ("PD_OFF", "Control"), ("PD_ON", "Control"), ("PD_ON", "PD_OFF")]
        for group_a, group_b in candidate_pairs:
            if group_a not in groups or group_b not in groups:
                continue
            x = pd.to_numeric(frame.loc[frame["group"].eq(group_a), "lavi_mean"], errors="coerce").dropna().to_numpy(float)
            y = pd.to_numeric(frame.loc[frame["group"].eq(group_b), "lavi_mean"], errors="coerce").dropna().to_numpy(float)
            mean_a = float(np.mean(x)) if len(x) else np.nan
            mean_b = float(np.mean(y)) if len(y) else np.nan
            sd_a = float(np.std(x, ddof=1)) if len(x) > 1 else np.nan
            sd_b = float(np.std(y, ddof=1)) if len(y) > 1 else np.nan
            pooled_sd = float(np.sqrt(((len(x) - 1) * sd_a**2 + (len(y) - 1) * sd_b**2) / (len(x) + len(y) - 2))) if len(x) > 1 and len(y) > 1 else np.nan
            cohens_d = (mean_a - mean_b) / pooled_sd if np.isfinite(pooled_sd) and pooled_sd > 0 else np.nan
            correction = (1.0 - 3.0 / (4.0 * (len(x) + len(y)) - 9.0)) if len(x) + len(y) > 2 else np.nan
            hedges_g = cohens_d * correction if np.isfinite(cohens_d) and np.isfinite(correction) else np.nan
            if len(x) >= 2 and len(y) >= 2 and np.unique(np.r_[x, y]).size > 1:
                test = ttest_ind(x, y, equal_var=False, nan_policy="omit")
                statistic, p_value = float(test.statistic), float(test.pvalue)
                variance = (sd_a**2 / len(x)) + (sd_b**2 / len(y))
                dof = float(variance**2 / ((sd_a**2 / len(x))**2 / (len(x) - 1) + (sd_b**2 / len(y))**2 / (len(y) - 1))) if variance > 0 else np.nan
                critical = float(student_t.ppf(0.975, dof)) if np.isfinite(dof) else np.nan
                half_width = critical * np.sqrt(variance) if np.isfinite(critical) else np.nan
            else:
                statistic, p_value, dof, half_width = np.nan, np.nan, np.nan, np.nan
            rows.append({
                "dataset": dataset, "band": band,
                "group_a": group_a, "group_b": group_b,
                "comparison": f"{group_a} - {group_b}",
                "n_a": len(x), "n_b": len(y), "mean_a": mean_a, "mean_b": mean_b,
                "sd_a": sd_a, "sd_b": sd_b,
                "mean_difference": mean_a - mean_b if np.isfinite(mean_a) and np.isfinite(mean_b) else np.nan,
                "cohens_d": cohens_d, "hedges_g": hedges_g,
                "ci95_low": (mean_a - mean_b - half_width) if np.isfinite(half_width) else np.nan,
                "ci95_high": (mean_a - mean_b + half_width) if np.isfinite(half_width) else np.nan,
                "welch_t": statistic, "degrees_of_freedom": dof, "p_value": p_value,
            })
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["q_fdr_bh"] = np.nan
    for (dataset, comparison), index in result.groupby(["dataset", "comparison"]).groups.items():
        result.loc[index, "q_fdr_bh"] = _fdr_bh(result.loc[index, "p_value"])
    result["significant_fdr"] = result["q_fdr_bh"] < FDR_ALPHA
    result["relevant_effect"] = result["hedges_g"].abs() >= RELEVANT_ABS_HEDGES_G
    result["significant_and_relevant"] = result["significant_fdr"] & result["relevant_effect"]
    return result.sort_values(["dataset", "band", "comparison"], key=lambda col: col.map({band: i for i, band in enumerate(bands)}) if col.name == "band" else col).reset_index(drop=True)


def _electrode_statistics(participant_electrode: pd.DataFrame) -> pd.DataFrame:
    """Welch tests per electrode, with BH-FDR correction within band/comparison."""
    rows: list[dict[str, Any]] = []
    pairs = [("PD", "Control"), ("PD_OFF", "Control"), ("PD_ON", "Control"), ("PD_ON", "PD_OFF")]
    for (dataset, band, electrode), frame in participant_electrode.groupby(["dataset", "band", "electrode"]):
        groups = set(frame["group"].dropna())
        for group_a, group_b in pairs:
            if group_a not in groups or group_b not in groups:
                continue
            x = pd.to_numeric(frame.loc[frame["group"].eq(group_a), "lavi_mean"], errors="coerce").dropna().to_numpy(float)
            y = pd.to_numeric(frame.loc[frame["group"].eq(group_b), "lavi_mean"], errors="coerce").dropna().to_numpy(float)
            if len(x) >= 2 and len(y) >= 2 and np.unique(np.r_[x, y]).size > 1:
                p_value = float(ttest_ind(x, y, equal_var=False, nan_policy="omit").pvalue)
            else:
                p_value = np.nan
            rows.append({"dataset": dataset, "band": band, "electrode": electrode, "group_a": group_a, "group_b": group_b, "comparison": f"{group_a} - {group_b}", "n_a": len(x), "n_b": len(y), "mean_difference": (float(np.mean(x)) - float(np.mean(y))) if len(x) and len(y) else np.nan, "p_value": p_value})
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["q_fdr_bh"] = np.nan
    for (dataset, band, comparison), index in result.groupby(["dataset", "band", "comparison"]).groups.items():
        result.loc[index, "q_fdr_bh"] = _fdr_bh(result.loc[index, "p_value"])
    result["significant_fdr"] = result["q_fdr_bh"] < FDR_ALPHA
    return result


def _clinical_correlations(participant: pd.DataFrame) -> pd.DataFrame:
    """Spearman associations between each LAVI quantity and available clinical scores."""
    lavi_metrics = [
        "lavi_mean", "lavi_median", "lavi_peak", "lavi_peak_frequency_hz",
        "high_rhythmicity_fraction", "low_rhythmicity_fraction",
    ]
    outcomes = [("moca", "MoCA"), ("mmse", "MMSE")]
    rows: list[dict[str, Any]] = []
    for (dataset, band), frame in participant.groupby(["dataset", "band"]):
        for outcome, outcome_label in outcomes:
            if outcome not in frame or pd.to_numeric(frame[outcome], errors="coerce").notna().sum() < 5:
                continue
            for metric in lavi_metrics:
                paired = frame[[metric, outcome]].apply(pd.to_numeric, errors="coerce").dropna()
                if len(paired) < 5 or paired.nunique().min() < 2:
                    rho, p_value = np.nan, np.nan
                else:
                    rho, p_value = spearmanr(paired[metric], paired[outcome])
                rows.append({"dataset": dataset, "band": band, "clinical_measure": outcome_label, "clinical_column": outcome, "lavi_metric": metric, "n": len(paired), "rho": rho, "p_value": p_value})
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["q_fdr_bh"] = np.nan
    finite = result["p_value"].notna()
    for (dataset, measure, metric), index in result.loc[finite].groupby(["dataset", "clinical_measure", "lavi_metric"]).groups.items():
        result.loc[index, "q_fdr_bh"] = _fdr_bh(result.loc[index, "p_value"])
    result["significant_fdr"] = result["q_fdr_bh"] < FDR_ALPHA
    result["relevant_effect"] = result["rho"].abs() >= RELEVANT_ABS_RHO
    result["significant_and_relevant"] = result["significant_fdr"] & result["relevant_effect"]
    return result


def _relationship_summary(
    group_statistics: pd.DataFrame,
    burst_correlations: pd.DataFrame,
    clinical_correlations: pd.DataFrame,
) -> pd.DataFrame:
    """Combine inferential results into one searchable significance table."""
    rows: list[dict[str, Any]] = []
    if not group_statistics.empty:
        for _, row in group_statistics.iterrows():
            rows.append({
                "relationship_type": "group_difference", "dataset": row["dataset"], "band": row["band"],
                "relationship": row["comparison"], "lavi_quantity": "lavi_mean",
                "effect_measure": "Hedges g", "effect_value": row["hedges_g"],
                "p_value": row["p_value"], "q_fdr_bh": row["q_fdr_bh"],
                "significant_fdr": row["significant_fdr"], "relevant_effect": row["relevant_effect"],
                "significant_and_relevant": row["significant_and_relevant"],
                "n": min(row["n_a"], row["n_b"]),
            })
    if not burst_correlations.empty:
        for _, row in burst_correlations.iterrows():
            rows.append({
                "relationship_type": "burst_correlation", "dataset": row["dataset"], "band": row["band"],
                "relationship": f"LAVI mean vs {row['burst_metric']}", "lavi_quantity": "lavi_mean",
                "effect_measure": "Spearman rho", "effect_value": row["rho"],
                "p_value": row["p_value"], "q_fdr_bh": row["q_fdr_bh"],
                "significant_fdr": row["significant_fdr"], "relevant_effect": row["relevant_effect"],
                "significant_and_relevant": row["significant_and_relevant"], "n": row["n"],
            })
    if not clinical_correlations.empty:
        for _, row in clinical_correlations.iterrows():
            rows.append({
                "relationship_type": "clinical_correlation", "dataset": row["dataset"], "band": row["band"],
                "relationship": f"{row['lavi_metric']} vs {row['clinical_measure']}", "lavi_quantity": row["lavi_metric"],
                "effect_measure": "Spearman rho", "effect_value": row["rho"],
                "p_value": row["p_value"], "q_fdr_bh": row["q_fdr_bh"],
                "significant_fdr": row["significant_fdr"], "relevant_effect": row["relevant_effect"],
                "significant_and_relevant": row["significant_and_relevant"], "n": row["n"],
            })
    return pd.DataFrame(rows)


def _save_profile_figure(results: list[dict[str, Any]], output: Path) -> None:
    rows = []
    for item in results:
        rows.append({
            "dataset": item["task"]["dataset"],
            "participant_id": item["task"]["participant_id"],
            "group": item["task"]["group"],
            "profile": item["profile_mean"],
            "foi": item["foi"],
        })
    datasets = list(dict.fromkeys(item["dataset"] for item in rows))
    fig, axes = plt.subplots(1, len(datasets), figsize=(5.0 * len(datasets), 4.4), sharey=True, squeeze=False, constrained_layout=True)
    tick_values = np.array([4, 5, 8, 10, 13, 20, 30, 45], dtype=float)
    for axis, dataset in zip(axes.flat, datasets):
        subset = [item for item in rows if item["dataset"] == dataset]
        foi = np.asarray(subset[0]["foi"])
        for band, (low, high) in DEFAULT_BANDS.items():
            if high >= foi.min() and low <= foi.max():
                clipped_low, clipped_high = max(low, foi.min()), min(high, foi.max())
                axis.axvspan(clipped_low, clipped_high, color=BAND_COLORS[band], alpha=0.07, lw=0, zorder=0)
                # The x-axis is logarithmic, so place each label at the
                # geometric midpoint of its visible frequency span.
                band_midpoint = float(np.sqrt(clipped_low * clipped_high))
                axis.text(
                    band_midpoint,
                    0.97,
                    band.title(),
                    transform=axis.get_xaxis_transform(),
                    ha="center",
                    va="top",
                    fontsize=8,
                    fontweight="bold",
                    color="#555555",
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.65, "pad": 1.2},
                    zorder=3,
                )
        for group in dict.fromkeys(item["group"] for item in subset):
            group_rows = [item for item in subset if item["group"] == group]
            # The inferential unit is the participant.  If a participant has
            # multiple recordings/sessions, average those profiles first so
            # that sessions cannot create extra visual weight.
            participant_profiles = []
            for participant_id in dict.fromkeys(item["participant_id"] for item in group_rows):
                recording_profiles = np.asarray([item["profile"] for item in group_rows if item["participant_id"] == participant_id], dtype=float)
                participant_profiles.append(np.nanmean(recording_profiles, axis=0))
            profiles = np.asarray(participant_profiles, dtype=float)
            color = GROUP_COLORS.get(group, "#777777")
            mean_profile = np.nanmean(profiles, axis=0)
            n_participants = np.sum(np.isfinite(profiles), axis=0)
            if len(profiles) > 1:
                sd_profile = np.nanstd(profiles, axis=0, ddof=1)
                sem_profile = sd_profile / np.sqrt(np.maximum(n_participants, 1))
                critical = np.array([student_t.ppf(0.975, max(int(n) - 1, 1)) if n > 1 else np.nan for n in n_participants])
                ci = critical * sem_profile
                axis.fill_between(foi, mean_profile - ci, mean_profile + ci, color=color, alpha=0.18, linewidth=0, zorder=1, label=f"{group} 95% CI")
            axis.plot(foi, mean_profile, color=color, linewidth=2.6, label=f"{group} mean (n={len(profiles)})", zorder=2)
        axis.set_xscale("log"); axis.set_xlim(foi.min(), foi.max()); axis.set_ylim(0, 1); axis.grid(alpha=0.22, which="both"); axis.set_title(dataset, fontsize=11, fontweight="bold")
        axis.xaxis.set_major_locator(FixedLocator(tick_values[(tick_values >= foi.min()) & (tick_values <= foi.max())]))
        axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        axis.xaxis.set_minor_formatter(NullFormatter())
        axis.set_xlabel("Frequency (Hz)")
    axes[0, 0].set_ylabel("LAVI")
    present_groups = list(dict.fromkeys(item["group"] for item in rows))
    group_handles = []
    for group in present_groups:
        label = group.replace("PD_", "PD-") if group.startswith("PD_") else group
        group_handles.append(Line2D([], [], color=GROUP_COLORS.get(group, "#777777"), linewidth=2.6, label=label))
    if group_handles:
        fig.legend(group_handles, [handle.get_label() for handle in group_handles], loc="upper center", bbox_to_anchor=(0.5, 1.01), ncol=len(group_handles), frameon=False, fontsize=9, title="Group (line; ribbon = 95% participant CI)", title_fontsize=8)
    fig.suptitle("All-electrode LAVI profiles", y=1.03, fontsize=14, fontweight="bold")
    fig.savefig(output, dpi=300, bbox_inches="tight"); plt.close(fig)


def _make_info(channels: list[str], sfreq: float = 250.0):
    info = mne.create_info(channels, sfreq=sfreq, ch_types="eeg")
    montage = mne.channels.make_standard_montage("standard_1020")
    info.set_montage(montage, on_missing="ignore")
    return info


def _plot_topomap(
    values: pd.Series,
    output_axis,
    title: str,
    vmin: float = 0.0,
    vmax: float = 1.0,
    cmap: str = "viridis",
    significant_channels: set[str] | None = None,
) -> bool:
    channels = [str(ch) for ch in values.index]
    info = _make_info(channels)
    data = values.to_numpy(float)
    usable = np.array([
        np.isfinite(info["chs"][i]["loc"][:3]).all()
        and np.linalg.norm(info["chs"][i]["loc"][:3]) > 0
        and np.isfinite(data[i])
        for i in range(len(channels))
    ])
    # Standard montages contain a few aliases at identical coordinates (for
    # example I1/I2). MNE refuses those for interpolation, so keep one channel
    # per unique 2-D location while retaining the numeric table unchanged.
    unique_indices: list[int] = []
    seen_positions: set[tuple[float, float]] = set()
    for index in np.where(usable)[0]:
        position = tuple(np.round(info["chs"][int(index)]["loc"][:2], 8))
        if position in seen_positions:
            continue
        seen_positions.add(position)
        unique_indices.append(int(index))
    usable = np.zeros(len(channels), dtype=bool)
    usable[unique_indices] = True
    if usable.sum() < 3:
        output_axis.axis("off"); output_axis.set_title(f"{title}\n(no montage)", fontsize=8); return False
    info = mne.pick_info(info, unique_indices)
    data = data[unique_indices]
    mask = None
    if significant_channels:
        mask = np.asarray([channel in significant_channels for channel in info.ch_names], dtype=bool)
    im, _ = mne.viz.plot_topomap(
        data,
        info,
        axes=output_axis,
        show=False,
        sensors=True,
        contours=0,
        vlim=(vmin, vmax),
        cmap=cmap,
        extrapolate="head",
        image_interp="cubic",
        mask=mask,
        mask_params={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "black", "linestyle": "None", "linewidth": 0.5, "markersize": 5} if mask is not None else None,
    )
    output_axis.set_title(title, fontsize=8)
    return True


def _save_topomap_outputs(electrode: pd.DataFrame, output: Path) -> None:
    topo_root = output / "figures" / "topomaps"; topo_root.mkdir(parents=True, exist_ok=True)
    bands = _ordered_bands(dict.fromkeys(electrode["band"]))
    electrode_stats = _electrode_statistics(electrode)
    # Group means and contrasts, one publication-style figure per dataset.
    # The group maps use a robust LAVI scale (with a shared colourbar), while
    # the contrast column is centred on zero and uses its observed range.  This
    # makes the spatial structure visible without changing the underlying data.
    for dataset, frame in electrode.groupby("dataset"):
        groups = list(dict.fromkeys(frame["group"]))
        if {"PD", "Control"}.issubset(groups):
            contrast_group, contrast_label = "PD_minus_Control", "PD − Control"
            contrast_groups = ("PD", "Control")
        elif {"PD_ON", "PD_OFF"}.issubset(groups):
            contrast_group, contrast_label = "PD_ON_minus_PD_OFF", "PD-ON − PD-OFF"
            contrast_groups = ("PD_ON", "PD_OFF")
        else:
            contrast_group, contrast_label, contrast_groups = None, "contrast unavailable", ()
        panels = groups + ([contrast_group] if contrast_group is not None else [])

        # Gather values first so limits are stable across rows in this figure.
        group_panel_values: dict[tuple[str, str], pd.Series] = {}
        contrast_panel_values: dict[str, pd.Series] = {}
        all_group_values: list[np.ndarray] = []
        all_contrast_values: list[np.ndarray] = []
        for band in bands:
            subset = frame.loc[frame["band"].eq(band)]
            for group in groups:
                values = subset.loc[subset["group"].eq(group)].groupby("electrode")["lavi_mean"].mean()
                group_panel_values[(band, group)] = values
                finite = values.to_numpy(float); finite = finite[np.isfinite(finite)]
                if len(finite): all_group_values.append(finite)
            if contrast_group is not None:
                left = subset.loc[subset["group"].eq(contrast_groups[0])].groupby("electrode")["lavi_mean"].mean()
                right = subset.loc[subset["group"].eq(contrast_groups[1])].groupby("electrode")["lavi_mean"].mean()
                values = left.subtract(right, fill_value=np.nan)
                contrast_panel_values[band] = values
                finite = values.to_numpy(float); finite = finite[np.isfinite(finite)]
                if len(finite): all_contrast_values.append(finite)
        finite_groups = np.concatenate(all_group_values) if all_group_values else np.array([], dtype=float)
        if len(finite_groups):
            group_lo, group_hi = np.nanpercentile(finite_groups, [2, 98])
            pad = max(0.02, 0.08 * (group_hi - group_lo))
            group_vmin, group_vmax = max(0.0, float(group_lo - pad)), min(1.0, float(group_hi + pad))
            if group_vmax - group_vmin < 0.1: group_vmin, group_vmax = 0.0, 1.0
        else:
            group_vmin, group_vmax = 0.0, 1.0
        finite_contrasts = np.concatenate(all_contrast_values) if all_contrast_values else np.array([], dtype=float)
        contrast_limit = max(0.05, float(np.nanpercentile(np.abs(finite_contrasts), 98))) if len(finite_contrasts) else 0.05

        fig, axes = plt.subplots(
            len(bands), len(panels), figsize=(3.35 * len(panels), 3.0 * len(bands)), squeeze=False
        )
        group_image = contrast_image = None
        for ri, band in enumerate(bands):
            for ci, group in enumerate(panels):
                if group in groups:
                    values = group_panel_values[(band, group)]
                    image_ok = _plot_topomap(values, axes[ri, ci], f"{band.title()}\n{group}", group_vmin, group_vmax, "viridis")
                    if image_ok and group_image is None and axes[ri, ci].images:
                        group_image = axes[ri, ci].images[-1]
                else:
                    values = contrast_panel_values[band]
                    significant = set()
                    if not electrode_stats.empty:
                        comparison_key = contrast_label.replace(" − ", " - ")
                        significant = set(electrode_stats.loc[
                            electrode_stats["dataset"].eq(dataset)
                            & electrode_stats["band"].eq(band)
                            & electrode_stats["comparison"].eq(comparison_key)
                            & electrode_stats["significant_fdr"], "electrode"
                        ].astype(str))
                    image_ok = _plot_topomap(values, axes[ri, ci], f"{band.title()}\n{contrast_label}", -contrast_limit, contrast_limit, "viridis", significant_channels=significant)
                    if image_ok and contrast_image is None and axes[ri, ci].images:
                        contrast_image = axes[ri, ci].images[-1]
                    if image_ok:
                        axes[ri, ci].text(0.5, -0.08, f"{len(significant)} FDR-significant electrodes", transform=axes[ri, ci].transAxes, ha="center", va="top", fontsize=7)
            axes[ri, 0].set_ylabel(band.title(), fontsize=10, fontweight="bold")
        fig.subplots_adjust(left=0.07, right=0.84, top=0.88, bottom=0.06, hspace=0.55, wspace=0.10)
        if group_image is not None:
            cax = fig.add_axes((0.855, 0.56, 0.018, 0.29))
            cb = fig.colorbar(group_image, cax=cax); cb.set_label("LAVI", fontsize=9); cb.ax.tick_params(labelsize=8)
        if contrast_image is not None:
            cax = fig.add_axes((0.855, 0.16, 0.018, 0.29))
            cb = fig.colorbar(contrast_image, cax=cax); cb.set_label("LAVI contrast", fontsize=9); cb.ax.tick_params(labelsize=8)
        fig.suptitle(f"Rhythmicity topomaps — {dataset}", y=0.97, fontsize=14, fontweight="bold")
        subtitle = f"Smooth viridis interpolation; group maps ({group_vmin:.2f}–{group_vmax:.2f})"
        if contrast_group is not None: subtitle += f"; {contrast_label} (±{contrast_limit:.3f})"
        fig.text(0.5, 0.935, subtitle, ha="center", fontsize=8.5, color="#444444")
        fig.savefig(topo_root / f"{dataset}_group_topomaps.png", dpi=300, bbox_inches="tight"); plt.close(fig)
    # Participant galleries are multi-page PDFs to avoid producing hundreds of large PNGs.
    for dataset, frame in electrode.groupby("dataset"):
        pdf_path = topo_root / f"{dataset}_participant_topomaps.pdf"
        with PdfPages(pdf_path) as pdf:
            for participant, participant_frame in frame.groupby("participant_id"):
                fig, axes = plt.subplots(1, len(bands), figsize=(3.25 * len(bands) + 0.45, 3.2), squeeze=False)
                image = None
                for ci, band in enumerate(bands):
                    values = participant_frame.loc[participant_frame["band"].eq(band)].groupby("electrode")["lavi_mean"].mean()
                    image_ok = _plot_topomap(values, axes[0, ci], band.title(), 0.0, 1.0, "viridis")
                    if image_ok and image is None and axes[0, ci].images:
                        image = axes[0, ci].images[-1]
                fig.subplots_adjust(left=0.03, right=0.90, top=0.82, bottom=0.05, wspace=0.10)
                if image is not None:
                    cax = fig.add_axes((0.925, 0.20, 0.018, 0.52))
                    cb = fig.colorbar(image, cax=cax); cb.set_label("LAVI", fontsize=9); cb.ax.tick_params(labelsize=8)
                fig.suptitle(f"{dataset} — {participant}", y=0.96, fontsize=12, fontweight="bold")
                fig.text(0.5, 0.90, "All-electrode LAVI (viridis; 0–1)", ha="center", fontsize=8, color="#444444")
                pdf.savefig(fig); plt.close(fig)

    # One publication-style cross-dataset contrast figure.  A single viridis
    # scale is estimated from the observed contrasts so subtle LAVI differences
    # remain visible instead of being washed out by a fixed ±0.3 range.
    datasets = list(dict.fromkeys(electrode["dataset"]))
    contrast_values: dict[tuple[str, str], tuple[pd.Series, str, set[str]]] = {}
    all_contrasts: list[np.ndarray] = []
    for dataset in datasets:
        frame = electrode.loc[electrode["dataset"].eq(dataset)]
        groups = set(frame["group"])
        for band in bands:
            subset = frame.loc[frame["band"].eq(band)]
            if {"PD", "Control"}.issubset(groups):
                left = subset.loc[subset["group"].eq("PD")].groupby("electrode")["lavi_mean"].mean()
                right = subset.loc[subset["group"].eq("Control")].groupby("electrode")["lavi_mean"].mean()
                values, label = left.subtract(right, fill_value=np.nan), "PD − Control"
            elif {"PD_ON", "PD_OFF"}.issubset(groups):
                left = subset.loc[subset["group"].eq("PD_ON")].groupby("electrode")["lavi_mean"].mean()
                right = subset.loc[subset["group"].eq("PD_OFF")].groupby("electrode")["lavi_mean"].mean()
                values, label = left.subtract(right, fill_value=np.nan), "PD-ON − PD-OFF"
            else:
                values, label = pd.Series(dtype=float), "contrast unavailable"
            significant = set()
            if not electrode_stats.empty:
                comparison_key = label.replace(" − ", " - ")
                significant = set(electrode_stats.loc[
                    electrode_stats["dataset"].eq(dataset)
                    & electrode_stats["band"].eq(band)
                    & electrode_stats["comparison"].eq(comparison_key)
                    & electrode_stats["significant_fdr"],
                    "electrode",
                ].astype(str))
            contrast_values[(dataset, band)] = (values, label, significant)
            finite = values.to_numpy(float); finite = finite[np.isfinite(finite)]
            if len(finite):
                all_contrasts.append(finite)
    finite_contrasts = np.concatenate(all_contrasts) if all_contrasts else np.array([], dtype=float)
    limit = max(0.05, float(np.nanpercentile(np.abs(finite_contrasts), 98))) if len(finite_contrasts) else 0.05
    fig, axes = plt.subplots(len(bands), len(datasets), figsize=(12.5, 3.0 * len(bands)), squeeze=False)
    image = None
    for ri, band in enumerate(bands):
        for ci, dataset in enumerate(datasets):
            values, label, significant = contrast_values[(dataset, band)]
            image_ok = _plot_topomap(values, axes[ri, ci], f"{dataset}\n{label}", -limit, limit, "viridis", significant_channels=significant)
            if image_ok:
                # Recover the image artist for the shared colourbar.
                image = axes[ri, ci].images[-1] if axes[ri, ci].images else image
            if ci == 0:
                axes[ri, ci].set_ylabel(band.title(), fontsize=10, fontweight="bold")
            if image_ok:
                axes[ri, ci].text(0.5, -0.08, f"{len(significant)} FDR-significant electrodes", transform=axes[ri, ci].transAxes, ha="center", va="top", fontsize=7)
    fig.subplots_adjust(left=0.08, right=0.90, top=0.90, bottom=0.04, hspace=0.32, wspace=0.08)
    if image is not None:
        colorbar_axis = fig.add_axes((0.92, 0.20, 0.022, 0.60))
        colorbar = fig.colorbar(image, cax=colorbar_axis)
        colorbar.set_label("LAVI contrast", fontsize=10)
    fig.suptitle("Rhythmicity topomap contrasts across datasets", y=0.98, fontsize=15, fontweight="bold")
    fig.text(0.5, 0.945, "Viridis scale uses the observed 98th-percentile contrast range; values are group differences", ha="center", fontsize=9, color="#444444")
    fig.savefig(output / "figures" / "rhythmicity_topomap_summary.png", dpi=300, bbox_inches="tight"); plt.close(fig)


def _save_group_effects(participant: pd.DataFrame, output: Path, statistics: pd.DataFrame | None = None) -> None:
    datasets = list(dict.fromkeys(participant["dataset"]))
    bands = _ordered_bands(dict.fromkeys(participant["band"]))
    fig, axes = plt.subplots(len(bands), len(datasets), figsize=(4.6 * len(datasets), 2.9 * len(bands)), squeeze=False, sharey=False, constrained_layout=True)
    group_order = ["Control", "PD", "PD_OFF", "PD_ON"]
    rng = np.random.default_rng(20260911)
    for ri, band in enumerate(bands):
        # Use one readable y-range per band row, shared across datasets.  This
        # preserves cross-dataset comparability while avoiding the compression
        # caused by a fixed 0–1 range when values cluster around ~0.3–0.8.
        row_values = pd.to_numeric(participant.loc[participant["band"].eq(band), "lavi_mean"], errors="coerce").dropna().to_numpy(float)
        if len(row_values):
            row_min, row_max = float(np.min(row_values)), float(np.max(row_values))
            padding = max(0.04, 0.12 * (row_max - row_min))
            row_ylim = (max(0.0, row_min - padding), min(1.0, row_max + padding))
            if row_ylim[1] - row_ylim[0] < 0.12:
                midpoint = float(np.mean(row_ylim)); row_ylim = (max(0.0, midpoint - 0.06), min(1.0, midpoint + 0.06))
        else:
            row_ylim = (0.0, 1.0)
        for ci, dataset in enumerate(datasets):
            axis = axes[ri, ci]; subset = participant.loc[participant["band"].eq(band) & participant["dataset"].eq(dataset)]
            groups = [group for group in group_order if group in set(subset["group"])] + [group for group in subset["group"].unique() if group not in group_order]
            data = [subset.loc[subset["group"].eq(group), "lavi_mean"].dropna().to_numpy(float) for group in groups]
            if data:
                violin = axis.violinplot(data, positions=np.arange(len(groups)), showmeans=False, showmedians=True, showextrema=True, widths=0.82)
                for body, group in zip(violin["bodies"], groups):
                    body.set_facecolor(GROUP_COLORS.get(group, "#777777")); body.set_edgecolor("white"); body.set_linewidth(0.8); body.set_alpha(0.75)
                for gi, (group, values) in enumerate(zip(groups, data)):
                    if len(values):
                        jitter = rng.uniform(-0.095, 0.095, size=len(values))
                        axis.scatter(np.full(len(values), gi) + jitter, values, s=18, color=GROUP_COLORS.get(group, "#777777"), edgecolor="white", linewidth=0.45, alpha=0.9, zorder=3)
                        # y is in axes coordinates so labels remain inside the
                        # panel when the band-specific data limits change.
                        axis.text(gi, 0.98, f"n={len(values)}", transform=axis.get_xaxis_transform(), ha="center", va="top", fontsize=7, color="#444444")
            labels = ["PD-OFF" if group == "PD_OFF" else "PD-ON" if group == "PD_ON" else group for group in groups]
            axis.set_xticks(np.arange(len(groups)), labels, rotation=25, ha="right", fontsize=8); axis.set_ylim(*row_ylim); axis.yaxis.set_major_locator(MaxNLocator(nbins=5)); axis.grid(axis="y", alpha=0.22)
            if ci == 0: axis.set_ylabel(f"{band.title()} mean LAVI")
            if ri == 0: axis.set_title(dataset, fontsize=11, fontweight="bold")
            if statistics is not None and not statistics.empty:
                sig = statistics[(statistics["dataset"] == dataset) & (statistics["band"] == band)]
                q_values = sig["q_fdr_bh"].dropna().to_numpy(float)
                if len(q_values):
                    axis.text(0.02, 0.98, "q=" + ", ".join(f"{q:.3f}" for q in q_values), transform=axis.transAxes, ha="left", va="top", fontsize=7, color="#555555")
                # Draw brackets only for FDR-significant comparisons.  The
                # bracket positions are in data coordinates and stacked so
                # multiple medication-state comparisons remain readable.
                group_positions = {group: index for index, group in enumerate(groups)}
                significant = sig[sig["q_fdr_bh"] < FDR_ALPHA].sort_values("q_fdr_bh")
                span = max(row_ylim[1] - row_ylim[0], 0.1)
                for bracket_index, (_, comparison) in enumerate(significant.iterrows()):
                    if comparison["group_a"] not in group_positions or comparison["group_b"] not in group_positions:
                        continue
                    left = min(group_positions[comparison["group_a"]], group_positions[comparison["group_b"]])
                    right = max(group_positions[comparison["group_a"]], group_positions[comparison["group_b"]])
                    height = row_ylim[1] - (0.08 + 0.08 * bracket_index) * span
                    cap = 0.018 * span
                    axis.plot([left, left, right, right], [height - cap, height, height, height - cap], color="#222222", linewidth=0.8, zorder=4, clip_on=False)
                    axis.text((left + right) / 2, height + 0.008 * span, _significance_stars(float(comparison["q_fdr_bh"])), ha="center", va="bottom", fontsize=9, fontweight="bold", color="#111111", zorder=5, clip_on=False)
    fig.suptitle("Band-level rhythmicity by group (participant points overlaid)", fontsize=14, fontweight="bold")
    fig.savefig(output / "lavi_band_group_effects.png", dpi=300, bbox_inches="tight"); plt.close(fig)


def _save_burst_heatmap(correlations: pd.DataFrame, output: Path) -> None:
    if correlations.empty: return
    datasets = list(dict.fromkeys(correlations["dataset"])); bands = _ordered_bands(dict.fromkeys(correlations["band"])); metrics = list(dict.fromkeys(correlations["burst_metric"]))
    fig, axes = plt.subplots(len(datasets), 1, figsize=(11, 2.9 * len(datasets)), squeeze=False, constrained_layout=True)
    finite_rho = correlations["rho"].dropna().to_numpy(float)
    vmax = max(0.2, float(np.nanmax(np.abs(finite_rho)))) if len(finite_rho) else 0.2
    for axis, dataset in zip(axes.flat, datasets):
        subset = correlations.loc[correlations["dataset"].eq(dataset)].pivot(index="band", columns="burst_metric", values="rho").reindex(index=bands, columns=metrics)
        image = axis.imshow(np.ma.masked_invalid(subset.to_numpy(float)), cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        axis.set_xticks(range(len(metrics)), [m.replace("_", " ") for m in metrics], rotation=35, ha="right", fontsize=8); axis.set_yticks(range(len(bands)), [b.title() for b in bands]); axis.set_title(dataset)
        for ri in range(len(bands)):
            for ci in range(len(metrics)):
                value = subset.iloc[ri, ci]
                if np.isfinite(value):
                    match = correlations[(correlations["dataset"] == dataset) & (correlations["band"] == bands[ri]) & (correlations["burst_metric"] == metrics[ci])]
                    suffix = ""
                    if not match.empty:
                        row = match.iloc[0]
                        suffix = ("*" if bool(row.get("significant_fdr", False)) else "") + ("†" if bool(row.get("relevant_effect", False)) else "")
                    axis.text(ci, ri, f"{value:.2f}{suffix}", ha="center", va="center", fontsize=8, color="black")
    fig.colorbar(image, ax=axes.ravel().tolist(), label="Spearman rho", shrink=0.82); fig.suptitle("LAVI–burst associations", fontsize=14, fontweight="bold"); fig.savefig(output / "lavi_burst_associations.png", dpi=300, bbox_inches="tight"); plt.close(fig)


def _save_group_comparison_figure(statistics: pd.DataFrame, output: Path) -> None:
    if statistics.empty:
        return
    datasets = list(dict.fromkeys(statistics["dataset"]))
    bands = _ordered_bands(dict.fromkeys(statistics["band"]))
    comparisons = list(dict.fromkeys(statistics["comparison"]))
    fig, axes = plt.subplots(len(bands), len(datasets), figsize=(4.8 * len(datasets), 2.8 * len(bands)), squeeze=False, sharex=False, constrained_layout=True)
    for ri, band in enumerate(bands):
        for ci, dataset in enumerate(datasets):
            axis = axes[ri, ci]
            subset = statistics[(statistics["dataset"] == dataset) & (statistics["band"] == band)]
            subset = subset.set_index("comparison").reindex(comparisons).dropna(subset=["mean_difference"], how="all").reset_index()
            if subset.empty:
                axis.axis("off"); continue
            y = np.arange(len(subset))
            color = ["#d95f02" if "PD" in str(label) else "#555555" for label in subset["comparison"]]
            lower = subset["mean_difference"] - subset["ci95_low"]
            upper = subset["ci95_high"] - subset["mean_difference"]
            axis.errorbar(subset["mean_difference"], y, xerr=np.vstack([lower, upper]), fmt="o", ms=5, capsize=3, color="#333333", ecolor=color[0] if len(set(color)) == 1 else "#555555")
            for yi, (_, row) in enumerate(subset.iterrows()):
                q = row["q_fdr_bh"]
                if np.isfinite(q):
                    axis.text(row["ci95_high"] if np.isfinite(row["ci95_high"]) else row["mean_difference"], yi, f"  q={q:.3f}", va="center", fontsize=7)
            axis.axvline(0, color="#222222", lw=0.8, ls="--")
            axis.set_yticks(y, subset["comparison"].str.replace("PD_OFF", "PD-OFF").str.replace("PD_ON", "PD-ON"), fontsize=8)
            axis.grid(axis="x", alpha=0.22)
            axis.set_title(f"{dataset} — {band.title()}", fontsize=10, fontweight="bold")
            axis.set_xlabel("Mean LAVI difference (95% CI)")
    fig.suptitle("Welch group comparisons of participant-level LAVI", fontsize=14, fontweight="bold")
    fig.savefig(output / "lavi_group_comparisons.png", dpi=300, bbox_inches="tight"); plt.close(fig)


def _save_clinical_correlation_figure(participant: pd.DataFrame, correlations: pd.DataFrame, output: Path) -> None:
    """Plot participant-level LAVI/clinical scatterplots in one band × dataset grid."""
    if participant.empty:
        return
    datasets = list(dict.fromkeys(participant["dataset"]))
    bands = _ordered_bands(dict.fromkeys(participant["band"]))
    if not bands or not datasets:
        return
    # Use MoCA for the standard datasets and MMSE for the medication-state
    # dataset, matching the clinical variable available for each cohort.
    dataset_outcomes: dict[str, tuple[str, str]] = {}
    for dataset in datasets:
        preferred = ("mmse", "MMSE") if dataset == "medication_state" else ("moca", "MoCA")
        fallback = ("moca", "MoCA") if preferred[0] == "mmse" else ("mmse", "MMSE")
        if preferred[0] in participant and pd.to_numeric(participant.loc[participant["dataset"] == dataset, preferred[0]], errors="coerce").notna().any():
            dataset_outcomes[dataset] = preferred
        elif fallback[0] in participant and pd.to_numeric(participant.loc[participant["dataset"] == dataset, fallback[0]], errors="coerce").notna().any():
            dataset_outcomes[dataset] = fallback
        else:
            dataset_outcomes[dataset] = preferred
    fig, axes = plt.subplots(len(bands), len(datasets), figsize=(4.4 * len(datasets), 2.9 * len(bands)), squeeze=False, sharey=False, constrained_layout=True)
    for bi, band in enumerate(bands):
        for ci, dataset in enumerate(datasets):
            axis = axes[bi, ci]
            column, label = dataset_outcomes[dataset]
            frame = participant[(participant["dataset"] == dataset) & (participant["band"] == band)].copy()
            frame["lavi_mean"] = pd.to_numeric(frame["lavi_mean"], errors="coerce")
            if column in frame:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
                frame = frame.dropna(subset=["lavi_mean", column])
            else:
                frame = frame.iloc[0:0]
            if frame.empty:
                axis.axis("off")
                axis.text(0.5, 0.5, f"No {label}\ndata", ha="center", va="center", fontsize=9, color="#777777")
                continue
            groups = [group for group in ["Control", "PD", "PD_OFF", "PD_ON"] if group in set(frame["group"])]
            for group in groups:
                subset = frame[frame["group"] == group]
                axis.scatter(subset["lavi_mean"], subset[column], s=24, alpha=0.78, color=GROUP_COLORS.get(group, "#777777"), edgecolor="white", linewidth=0.35, label=group)
            if len(frame) >= 5 and frame["lavi_mean"].nunique() > 1 and frame[column].nunique() > 1:
                x = frame["lavi_mean"].to_numpy(float); y = frame[column].to_numpy(float)
                slope, intercept = np.polyfit(x, y, 1)
                grid = np.linspace(float(x.min()), float(x.max()), 80)
                axis.plot(grid, intercept + slope * grid, color="#222222", linewidth=1.1)
                rho, p_value = spearmanr(x, y)
            else:
                rho, p_value = np.nan, np.nan
            match = correlations[(correlations["dataset"] == dataset) & (correlations["band"] == band) & (correlations["clinical_measure"] == label) & (correlations["lavi_metric"] == "lavi_mean")] if not correlations.empty else pd.DataFrame()
            q_value = float(match.iloc[0]["q_fdr_bh"]) if not match.empty and np.isfinite(match.iloc[0]["q_fdr_bh"]) else np.nan
            suffix = (" *" if np.isfinite(q_value) and q_value < FDR_ALPHA else "") + ("†" if not match.empty and bool(match.iloc[0].get("relevant_effect", False)) else "")
            axis.text(0.04, 0.96, f"n={len(frame)}\nρ={rho:.2f}{suffix}\nq={q_value:.2g}" if np.isfinite(q_value) else f"n={len(frame)}\nρ={rho:.2f}", transform=axis.transAxes, va="top", fontsize=8, bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "none"})
            if bi == 0:
                axis.set_title(f"{dataset}\n{label}", fontsize=10, fontweight="bold")
            if ci == 0:
                axis.set_ylabel(f"{band.title()} — {label}")
            else:
                axis.set_ylabel("")
            axis.set_xlabel("Band mean LAVI")
            axis.grid(alpha=0.2)
    legend_handles: dict[str, Any] = {}
    for axis in axes.flat:
        handles, labels = axis.get_legend_handles_labels()
        legend_handles.update(dict(zip(labels, handles)))
    if legend_handles:
        fig.legend(list(legend_handles.values()), list(legend_handles.keys()), loc="upper center", bbox_to_anchor=(0.5, 1.01), ncol=min(4, len(legend_handles)), frameon=False)
    fig.suptitle("Participant-level LAVI associations with MoCA/MMSE", fontsize=14, fontweight="bold", y=1.02)
    fig.savefig(output / "lavi_clinical_correlations.png", dpi=300, bbox_inches="tight"); plt.close(fig)


def run_analysis(config_path: str | Path, *, datasets: list[str] | None = None, recordings: list[str] | None = None, workers: int | None = None, overwrite: bool = False, generate_figures: bool = True) -> dict[str, Any]:
    config = _read_json(Path(config_path)); _validate_config(config)
    output = Path(config["output_dir"]); output.mkdir(parents=True, exist_ok=True)
    if overwrite and (output / "metrics").exists():
        logging.warning("overwrite requested; existing rhythmicity files will be replaced")
    tasks = _load_tasks(config, datasets, output, recordings)
    foi = _frequency_grid(config)
    n_workers = int(workers or config.get("workers", max(1, min(4, os.cpu_count() or 1))))
    cached_tasks = [] if overwrite else [task for task in tasks if Path(task.profile_path).is_file()]
    pending_tasks = [task for task in tasks if task not in cached_tasks]
    results: list[dict[str, Any]] = [_load_cached_result(task, config) for task in cached_tasks]
    failures: list[dict[str, str]] = []
    # Some managed/macOS environments deny the semaphore-limit query made by
    # ProcessPoolExecutor. Fall back to threads there; the expensive wavelet
    # and array operations release the GIL, and the same worker count/progress
    # semantics are preserved.
    try:
        executor = ProcessPoolExecutor(max_workers=n_workers)
    except (PermissionError, OSError) as error:
        logging.warning("Process workers unavailable (%s); falling back to threaded workers", error)
        executor = ThreadPoolExecutor(max_workers=n_workers)
    with executor:
        future_tasks = {executor.submit(_process_recording, task, config, foi): task for task in pending_tasks}
        for future in tqdm(as_completed(future_tasks), total=len(future_tasks), desc="Rhythmicity recordings"):
            task = future_tasks[future]
            try:
                results.append(future.result())
            except Exception as error:
                logging.error("Skipping rhythmicity recording %s/%s: %s", task.dataset, task.recording_id, error)
                failures.append({"dataset": task.dataset, "recording_id": task.recording_id, "reason": str(error)})
    electrode = pd.DataFrame([row for result in results for row in result["electrode_rows"]])
    abba_table = pd.DataFrame([row for result in results for row in result["abba_rows"]])
    if electrode.empty:
        raise RuntimeError("LAVI produced no electrode summaries")
    burst = _load_burst_features(Path(config["global_output_root"]))
    participant_electrode, participant = _aggregate_tables(electrode, burst)
    metrics_root = output / "metrics"; statistics_root = output / "statistics"; figures_root = output / "figures"
    metrics_root.mkdir(exist_ok=True); statistics_root.mkdir(exist_ok=True); figures_root.mkdir(exist_ok=True)
    electrode.to_csv(metrics_root / "electrode_rhythmicity.csv.gz", index=False)
    participant_electrode.to_csv(metrics_root / "participant_electrode_rhythmicity.csv.gz", index=False)
    participant.to_csv(metrics_root / "participant_rhythmicity.csv.gz", index=False)
    abba_table.to_csv(metrics_root / "abba_bands.csv.gz", index=False)
    correlations = _correlations(participant)
    correlations.to_csv(statistics_root / "lavi_burst_correlations.csv", index=False)
    group_statistics = _group_statistics(participant)
    group_statistics.to_csv(statistics_root / "lavi_group_comparisons.csv", index=False)
    electrode_statistics = _electrode_statistics(participant_electrode)
    electrode_statistics.to_csv(statistics_root / "lavi_electrode_comparisons.csv", index=False)
    clinical_correlations = _clinical_correlations(participant)
    clinical_correlations.to_csv(statistics_root / "lavi_clinical_correlations.csv", index=False)
    relationship_summary = _relationship_summary(group_statistics, correlations, clinical_correlations)
    relationship_summary.to_csv(statistics_root / "lavi_relationship_summary.csv", index=False)
    manifest = {"analysis": "all_electrode_lavi_rhythmicity", "config": config, "n_requested_recordings": len(tasks), "n_completed_recordings": len(results), "n_failed_recordings": len(failures), "failed_recordings": failures, "n_electrode_band_rows": len(electrode), "n_participants": int(participant["participant_id"].nunique()), "lavi_frequency_hz": foi.tolist(), "workers": n_workers, "resumed_recordings": len(cached_tasks), "surrogate_reps": int(config["lavi"].get("surrogate_reps", 0)), "abba_mode": "iaaft_95_percentile" if int(config["lavi"].get("surrogate_reps", 0)) > 0 else "channel_median_baseline"}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if generate_figures:
        _save_profile_figure(results, figures_root / "lavi_profiles_by_dataset.png")
        _save_group_effects(participant, figures_root, group_statistics)
        _save_group_comparison_figure(group_statistics, figures_root)
        _save_clinical_correlation_figure(participant, clinical_correlations, figures_root)
        _save_burst_heatmap(correlations, figures_root)
        _save_topomap_outputs(participant_electrode, output)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/analyses/rhythmicity.json")
    parser.add_argument("--datasets", nargs="*")
    parser.add_argument("--recordings", nargs="*")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-figures", action="store_true")
    args = parser.parse_args()
    manifest = run_analysis(args.config, datasets=args.datasets, recordings=args.recordings, workers=args.workers, overwrite=args.overwrite, generate_figures=not args.skip_figures)
    print(f"Completed rhythmicity analysis for {manifest['n_completed_recordings']} recordings")


if __name__ == "__main__":
    main()
