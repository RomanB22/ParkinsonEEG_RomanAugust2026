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
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
import mne
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, ttest_ind
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
    data = epochs.get_data(picks="eeg", units="uV")
    channels = list(epochs.copy().pick("eeg").ch_names)
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


def _load_burst_features(root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted((root / "metrics").glob("*/recording_features.csv.gz")):
        frame = pd.read_csv(path, low_memory=False)
        wanted = [c for c in frame.columns if c.startswith("bout__")]
        keys = [c for c in ("dataset_id", "recording_id", "participant_id", "session_id", "group", "electrode") if c in frame]
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
    return result


def _save_profile_figure(results: list[dict[str, Any]], output: Path) -> None:
    rows = []
    for item in results:
        rows.append({"dataset": item["task"]["dataset"], "group": item["task"]["group"], "profile": item["profile_mean"], "foi": item["foi"]})
    datasets = list(dict.fromkeys(item["dataset"] for item in rows))
    fig, axes = plt.subplots(1, len(datasets), figsize=(4.2 * len(datasets), 4.0), sharey=True, squeeze=False)
    colors = {"Control": "#7f7f7f", "PD": "#d95f02", "PD_OFF": "#7570b3", "PD_ON": "#1b9e77"}
    for axis, dataset in zip(axes.flat, datasets):
        subset = [item for item in rows if item["dataset"] == dataset]
        foi = np.asarray(subset[0]["foi"])
        for group in dict.fromkeys(item["group"] for item in subset):
            profiles = np.asarray([item["profile"] for item in subset if item["group"] == group])
            axis.plot(foi, profiles.T, color=colors.get(group, "#777777"), alpha=0.04, linewidth=0.35)
            axis.plot(foi, np.nanmean(profiles, axis=0), color=colors.get(group, "#777777"), linewidth=2, label=group)
        axis.set_xscale("log"); axis.set_ylim(0, 1); axis.grid(alpha=0.2); axis.set_title(dataset)
        axis.set_xlabel("Frequency (Hz)")
    axes[0, 0].set_ylabel("LAVI")
    axes[-1, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("All-electrode LAVI profiles", y=1.02)
    fig.tight_layout(); fig.savefig(output, dpi=180, bbox_inches="tight"); plt.close(fig)


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
    im, _ = mne.viz.plot_topomap(data, info, axes=output_axis, show=False, contours=0, vlim=(vmin, vmax), cmap=cmap, extrapolate="local")
    output_axis.set_title(title, fontsize=8)
    return True


def _save_topomap_outputs(electrode: pd.DataFrame, output: Path) -> None:
    topo_root = output / "figures" / "topomaps"; topo_root.mkdir(parents=True, exist_ok=True)
    bands = _ordered_bands(dict.fromkeys(electrode["band"]))
    # Group means and contrasts, one compact figure per dataset.
    contrast_frames: dict[str, pd.DataFrame] = {}
    for dataset, frame in electrode.groupby("dataset"):
        groups = list(dict.fromkeys(frame["group"]))
        panels = groups + (["PD_minus_Control"] if {"PD", "Control"}.issubset(groups) else [])
        fig, axes = plt.subplots(len(bands), len(panels), figsize=(3.0 * len(panels), 2.7 * len(bands)), squeeze=False)
        for ri, band in enumerate(bands):
            for ci, group in enumerate(panels):
                subset = frame.loc[frame["band"].eq(band)]
                if group in groups:
                    values = subset.loc[subset["group"].eq(group)].groupby("electrode")["lavi_mean"].mean()
                else:
                    left = subset.loc[subset["group"].eq("PD")].groupby("electrode")["lavi_mean"].mean()
                    right = subset.loc[subset["group"].eq("Control")].groupby("electrode")["lavi_mean"].mean()
                    values = left.subtract(right, fill_value=np.nan)
                _plot_topomap(
                    values,
                    axes[ri, ci],
                    f"{band.title()}\n{group}",
                    0.0 if group in groups else -0.3,
                    1.0 if group in groups else 0.3,
                    "viridis" if group in groups else "RdBu_r",
                )
            if {"PD", "Control"}.issubset(groups):
                contrast_frames[dataset] = frame
        fig.suptitle(f"Rhythmicity topomaps — {dataset}", y=0.995); fig.tight_layout(); fig.savefig(topo_root / f"{dataset}_group_topomaps.png", dpi=180, bbox_inches="tight"); plt.close(fig)
    # Participant galleries are multi-page PDFs to avoid producing hundreds of large PNGs.
    for dataset, frame in electrode.groupby("dataset"):
        pdf_path = topo_root / f"{dataset}_participant_topomaps.pdf"
        with PdfPages(pdf_path) as pdf:
            for participant, participant_frame in frame.groupby("participant_id"):
                fig, axes = plt.subplots(1, len(bands), figsize=(3.0 * len(bands), 2.8), squeeze=False)
                for ci, band in enumerate(bands):
                    values = participant_frame.loc[participant_frame["band"].eq(band)].groupby("electrode")["lavi_mean"].mean()
                    _plot_topomap(values, axes[0, ci], band.title())
                fig.suptitle(f"{dataset} — {participant}", y=1.0); fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

    # One compact cross-dataset contrast figure for quick review.
    datasets = list(dict.fromkeys(electrode["dataset"]))
    fig, axes = plt.subplots(len(bands), len(datasets), figsize=(3.0 * len(datasets), 2.7 * len(bands)), squeeze=False)
    for ci, dataset in enumerate(datasets):
        frame = electrode.loc[electrode["dataset"].eq(dataset)]
        groups = set(frame["group"])
        for ri, band in enumerate(bands):
            subset = frame.loc[frame["band"].eq(band)]
            if {"PD", "Control"}.issubset(groups):
                left = subset.loc[subset["group"].eq("PD")].groupby("electrode")["lavi_mean"].mean()
                right = subset.loc[subset["group"].eq("Control")].groupby("electrode")["lavi_mean"].mean()
                values = left.subtract(right, fill_value=np.nan)
                label = "PD − Control"
            elif {"PD_ON", "PD_OFF"}.issubset(groups):
                left = subset.loc[subset["group"].eq("PD_ON")].groupby("electrode")["lavi_mean"].mean()
                right = subset.loc[subset["group"].eq("PD_OFF")].groupby("electrode")["lavi_mean"].mean()
                values = left.subtract(right, fill_value=np.nan)
                label = "PD-ON − PD-OFF"
            else:
                values = pd.Series(dtype=float); label = "contrast unavailable"
            _plot_topomap(values, axes[ri, ci], f"{dataset}\n{band.title()} {label}", -0.3, 0.3, "RdBu_r")
    fig.suptitle("Rhythmicity topomap contrasts", y=0.995); fig.tight_layout(); fig.savefig(output / "figures" / "rhythmicity_topomap_summary.png", dpi=180, bbox_inches="tight"); plt.close(fig)


def _save_group_effects(participant: pd.DataFrame, output: Path) -> None:
    colors = {"Control": "#7f7f7f", "PD": "#d95f02", "PD_OFF": "#7570b3", "PD_ON": "#1b9e77"}
    datasets = list(dict.fromkeys(participant["dataset"]))
    bands = _ordered_bands(dict.fromkeys(participant["band"]))
    fig, axes = plt.subplots(len(bands), len(datasets), figsize=(4.2 * len(datasets), 2.7 * len(bands)), squeeze=False, sharey=True)
    for ri, band in enumerate(bands):
        for ci, dataset in enumerate(datasets):
            axis = axes[ri, ci]; subset = participant.loc[participant["band"].eq(band) & participant["dataset"].eq(dataset)]
            groups = list(dict.fromkeys(subset["group"])); data = [subset.loc[subset["group"].eq(g), "lavi_mean"].dropna() for g in groups]
            violin = axis.violinplot(data, positions=np.arange(len(groups)), showmeans=True, showextrema=False)
            for body, group in zip(violin["bodies"], groups): body.set_facecolor(colors.get(group, "#777777")); body.set_alpha(0.7)
            axis.set_xticks(np.arange(len(groups)), groups, rotation=35, ha="right", fontsize=8); axis.set_ylim(0, 1); axis.grid(axis="y", alpha=0.2)
            if ci == 0: axis.set_ylabel(f"{band.title()} mean LAVI")
            if ri == 0: axis.set_title(dataset)
    fig.suptitle("Band-level rhythmicity by group", y=1.0); fig.tight_layout(); fig.savefig(output / "lavi_band_group_effects.png", dpi=180, bbox_inches="tight"); plt.close(fig)


def _save_burst_heatmap(correlations: pd.DataFrame, output: Path) -> None:
    if correlations.empty: return
    datasets = list(dict.fromkeys(correlations["dataset"])); bands = _ordered_bands(dict.fromkeys(correlations["band"])); metrics = list(dict.fromkeys(correlations["burst_metric"]))
    fig, axes = plt.subplots(len(datasets), 1, figsize=(10, 2.6 * len(datasets)), squeeze=False)
    finite_rho = correlations["rho"].dropna().to_numpy(float)
    vmax = max(0.2, float(np.nanmax(np.abs(finite_rho)))) if len(finite_rho) else 0.2
    for axis, dataset in zip(axes.flat, datasets):
        subset = correlations.loc[correlations["dataset"].eq(dataset)].pivot(index="band", columns="burst_metric", values="rho").reindex(index=bands, columns=metrics)
        image = axis.imshow(subset.to_numpy(float), cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        axis.set_xticks(range(len(metrics)), [m.replace("_", " ") for m in metrics], rotation=35, ha="right", fontsize=8); axis.set_yticks(range(len(bands)), [b.title() for b in bands]); axis.set_title(dataset)
        for ri in range(len(bands)):
            for ci in range(len(metrics)):
                value = subset.iloc[ri, ci]
                if np.isfinite(value): axis.text(ci, ri, f"{value:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=axes.ravel().tolist(), label="Spearman rho", shrink=0.8); fig.suptitle("LAVI–burst associations", y=1.0); fig.tight_layout(); fig.savefig(output / "lavi_burst_associations.png", dpi=180, bbox_inches="tight"); plt.close(fig)


def run_analysis(config_path: str | Path, *, datasets: list[str] | None = None, recordings: list[str] | None = None, workers: int | None = None, overwrite: bool = False, generate_figures: bool = True) -> dict[str, Any]:
    config = _read_json(Path(config_path)); _validate_config(config)
    output = Path(config["output_dir"]); output.mkdir(parents=True, exist_ok=True)
    if overwrite and (output / "metrics").exists():
        logging.warning("overwrite requested; existing rhythmicity files will be replaced")
    tasks = _load_tasks(config, datasets, output, recordings)
    foi = _frequency_grid(config)
    n_workers = int(workers or config.get("workers", max(1, min(4, os.cpu_count() or 1))))
    results: list[dict[str, Any]] = []
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
        futures = [executor.submit(_process_recording, task, config, foi) for task in tasks]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Rhythmicity recordings"):
            results.append(future.result())
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
    manifest = {"analysis": "all_electrode_lavi_rhythmicity", "config": config, "n_recordings": len(results), "n_electrode_band_rows": len(electrode), "n_participants": int(participant["participant_id"].nunique()), "lavi_frequency_hz": foi.tolist(), "workers": n_workers, "surrogate_reps": int(config["lavi"].get("surrogate_reps", 0)), "abba_mode": "iaaft_95_percentile" if int(config["lavi"].get("surrogate_reps", 0)) > 0 else "channel_median_baseline"}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if generate_figures:
        _save_profile_figure(results, figures_root / "lavi_profiles_by_dataset.png")
        _save_group_effects(participant, figures_root)
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
    print(f"Completed rhythmicity analysis for {manifest['n_recordings']} recordings")


if __name__ == "__main__":
    main()
