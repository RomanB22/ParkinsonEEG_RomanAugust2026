"""Session-aware typical-bout envelope, phase, and shape gallery."""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from core.runtime import configure_runtime

configure_runtime()

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyses.scale_free.typical_bouts import (
    _coverage_summary,
    _plot_coverage_heatmap,
    _plot_electrode,
    _plot_representations,
    _representation_figure,
    _subject_envelopes,
    _write_gallery,
)


GROUP_ORDER = ("HC", "PD_OFF", "PD_ON")


def _run_tasks(tasks: list[tuple[Any, ...]], workers: int) -> list[dict[str, Any]]:
    if workers == 1:
        return [_subject_envelopes(task) for task in tasks]
    results: list[dict[str, Any]] = []
    try:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_subject_envelopes, task) for task in tasks]
            for future in as_completed(futures):
                results.append(future.result())
    except PermissionError:
        results = [_subject_envelopes(task) for task in tasks]
    return results


def generate_typical_bout_gallery(
    *,
    config: dict[str, Any],
    recordings: pd.DataFrame,
    subject_features: pd.DataFrame,
    electrode_features: pd.DataFrame,
    input_epochs: pd.DataFrame,
    output_dir: str | Path,
) -> list[Path]:
    """Render recording-balanced HC and paired-session PD bout shapes."""
    root = Path(output_dir)
    settings = config["typical_bouts"]
    inventory = json.loads(
        (root / "features" / "electrode_inventory.json").read_text(encoding="utf-8")
    )
    electrodes = [str(value) for value in inventory["common_channels"]]
    bands = {
        str(band): config["bands"][str(band)]
        for band in config["ebosc"]["bands"]
    }
    epoch_lookup = input_epochs.set_index("recording_id")["epoch_file"].astype(str)
    episode_root = root / "intermediate" / "episodes"
    tasks: list[tuple[Any, ...]] = []
    for row in recordings.itertuples(index=False):
        recording_id = str(row.recording_id)
        episode_path = episode_root / f"{recording_id}_bout_episodes.csv.gz"
        if recording_id not in epoch_lookup:
            raise FileNotFoundError(f"Missing epoch inventory for {recording_id}")
        if not episode_path.is_file():
            raise FileNotFoundError(f"Missing cached bout episodes: {episode_path}")
        tasks.append(
            (
                recording_id,
                str(row.condition),
                epoch_lookup.loc[recording_id],
                str(episode_path),
                electrodes,
                bands,
                int(settings["bandpass_filter_order"]),
                float(settings["center_window_seconds"]),
                float(config["ebosc"]["edge_padding_sec"]),
            )
        )
    results = _run_tasks(tasks, int(settings["workers"]))
    order = {
        str(recording_id): index
        for index, recording_id in enumerate(recordings["recording_id"])
    }
    results.sort(key=lambda row: order[row["subject_id"]])
    sfreqs = {float(row["sfreq"]) for row in results}
    if len(sfreqs) != 1:
        raise ValueError("Typical-bout inputs use inconsistent sampling frequencies")
    sfreq = sfreqs.pop()
    waveforms = np.stack([row["waveforms"] for row in results])
    phase_phasors = np.stack([row["phase_phasors"] for row in results])
    phase_shapes = np.stack([row["phase_aligned_shapes"] for row in results])
    counts = np.stack([row["bout_counts"] for row in results])
    baselines = np.stack([row["baseline_amplitude_uv"] for row in results])
    amplitude_uv = (waveforms - 1.0) * baselines[..., np.newaxis]
    phase_shapes_uv = phase_shapes * baselines[..., np.newaxis]

    fit = electrode_features.loc[
        electrode_features["duration_variant"].eq("all_retained")
        & electrode_features["family"].eq("aperiodic_qc")
        & electrode_features["metric"].eq("fit_qc_pass")
    ]
    fit_lookup = fit.pivot(
        index="recording_id", columns="electrode", values="value"
    ).reindex(index=recordings["recording_id"], columns=electrodes)
    if fit_lookup.isna().any().any():
        raise ValueError("Fit-QC table does not cover every recording/electrode")
    coverage = subject_features.loc[
        subject_features["duration_variant"].eq("all_retained")
        & subject_features["family"].eq("aperiodic_qc")
        & subject_features["metric"].eq("fit_qc_fraction")
    ].set_index("recording_id")["value"].reindex(recordings["recording_id"])
    subject_table = recordings[["recording_id", "condition"]].rename(
        columns={"recording_id": "subject_id", "condition": "group"}
    ).reset_index(drop=True)
    subject_table["subject_fit_qc_pass"] = coverage.to_numpy(float) >= float(
        config["aperiodic_fit_qc"]["minimum_subject_qc_fraction"]
    )
    subject_table["electrode_fit_qc"] = list(fit_lookup.to_numpy(bool))
    subject_table.attrs["group_order"] = list(GROUP_ORDER)
    subject_table.attrs["group_colors"] = config["plots"]["condition_colors"]

    half_window = int(round(float(settings["center_window_seconds"]) * sfreq))
    times = np.arange(-half_window, half_window + 1) / sfreq
    intermediate = root / "intermediate" / "typical_bouts"
    intermediate.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        intermediate / "recording_electrode_band_envelopes.npz",
        recording_ids=recordings["recording_id"].to_numpy(str),
        participant_ids=recordings["participant_id"].to_numpy(str),
        conditions=recordings["condition"].to_numpy(str),
        electrodes=np.asarray(electrodes),
        bands=np.asarray(list(bands)),
        times_seconds=times,
        normalized_amplitude_envelopes=waveforms,
        baseline_subtracted_amplitude_envelopes_uv=amplitude_uv,
        relative_phase_phasors=phase_phasors,
        phase_aligned_normalized_shapes=phase_shapes,
        phase_aligned_shapes_uv=phase_shapes_uv,
        bout_counts=counts,
        baseline_amplitude_uv=baselines,
        electrode_fit_qc=fit_lookup.to_numpy(bool),
        recording_fit_qc=subject_table["subject_fit_qc_pass"].to_numpy(bool),
    )
    coverage_rows: list[dict[str, Any]] = []
    for recording_index, recording in subject_table.iterrows():
        for electrode_index, electrode in enumerate(electrodes):
            for band_index, band in enumerate(bands):
                coverage_rows.append(
                    {
                        "subject_id": recording["subject_id"],
                        "group": recording["group"],
                        "electrode": electrode,
                        "band": band,
                        "n_bouts_in_typical_average": int(
                            counts[recording_index, electrode_index, band_index]
                        ),
                        "baseline_amplitude_uv": float(
                            baselines[recording_index, electrode_index, band_index]
                        ),
                        "electrode_fit_qc_pass": bool(
                            fit_lookup.iloc[recording_index, electrode_index]
                        ),
                        "subject_fit_qc_pass": bool(
                            recording["subject_fit_qc_pass"]
                        ),
                    }
                )
    coverage_table = pd.DataFrame.from_records(coverage_rows)
    coverage_table.attrs.update(subject_table.attrs)
    domain_metrics = root / "scale_free" / "metrics"
    domain_metrics.mkdir(parents=True, exist_ok=True)
    coverage_table.to_csv(domain_metrics / "typical_bout_coverage.csv", index=False)
    coverage_summary = _coverage_summary(coverage_table, electrodes, list(bands))
    coverage_summary.to_csv(
        domain_metrics / "typical_bout_group_coverage.csv", index=False
    )

    gallery = root / "figures" / "typical_bouts"
    gallery.mkdir(parents=True, exist_ok=True)
    dpi = int(config["plots"]["dpi"])
    paths = [
        gallery / "bout_detection_subject_coverage.png",
        gallery / "bout_count_qc.png",
    ]
    _plot_coverage_heatmap(
        coverage_summary,
        electrodes,
        list(bands),
        "contributing_subject_fraction",
        "Detected-bout recording coverage by condition, band, and electrode",
        "Fraction of eligible recordings with at least one retained bout",
        paths[0],
        dpi,
    )
    _plot_coverage_heatmap(
        coverage_summary,
        electrodes,
        list(bands),
        "median_bouts_per_eligible_subject",
        "Median detected bouts per eligible recording",
        "Median retained bouts per recording",
        paths[1],
        dpi,
    )
    for electrode_index, electrode in enumerate(electrodes):
        safe = electrode.replace("/", "_")
        for policy, directory in (("all", "all_subjects"), ("fit_qc", "fit_qc")):
            path = gallery / directory / f"{safe}.png"
            _plot_electrode(
                amplitude_uv,
                phase_phasors,
                phase_shapes_uv,
                counts,
                subject_table,
                electrode_index,
                electrode,
                list(bands),
                times,
                policy,
                float(settings["confidence_level"]),
                path,
                dpi,
            )
            paths.append(path)
    for policy, filename in (
        ("all", "grand_average_all_subjects.png"),
        ("fit_qc", "grand_average_fit_qc.png"),
    ):
        figure, axes = _representation_figure(len(bands))
        _plot_representations(
            axes,
            amplitude_uv,
            phase_phasors,
            phase_shapes_uv,
            counts,
            subject_table,
            None,
            list(bands),
            times,
            policy,
            float(settings["confidence_level"]),
        )
        label = "all recordings" if policy == "all" else "fit-QC sensitivity"
        figure.suptitle(
            f"Grand-average stereotypical detected bout ({label})\n"
            "envelope, circular phase, and phase-aligned shape; electrodes averaged within recording"
        )
        figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
        path = gallery / filename
        figure.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(figure)
        paths.append(path)
    _write_gallery(gallery, electrodes)
    manifest = {
        "n_participants": int(recordings["participant_id"].nunique()),
        "n_recordings": int(len(recordings)),
        "condition_counts": recordings["condition"].value_counts().to_dict(),
        "n_electrodes": len(electrodes),
        "bands": list(bands),
        "alignment": "detected bout midpoint",
        "window_seconds_each_side": float(settings["center_window_seconds"]),
        "aggregation": "bouts within recording/electrode/band, then equal-weight recording means within condition",
        "pd_sessions": "PD OFF and PD ON remain separately identified recordings from paired participants",
        "n_figures": len(paths),
    }
    (root / "typical_bouts_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return paths
