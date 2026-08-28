"""Subject-balanced envelope, phase, and phase-aligned bout-shape figures."""

from __future__ import annotations

import html
import json
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from core.runtime import configure_runtime

configure_runtime()

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from scipy.signal import hilbert
from scipy.stats import t as student_t

from analyses.ordinal.metrics import filter_epoch_data


def mean_centered_envelope(
    normalized_envelope: np.ndarray,
    episodes: pd.DataFrame,
    *,
    half_window_samples: int,
) -> tuple[np.ndarray, int]:
    """Return an equal-bout mean centered at each detected interval midpoint."""
    envelope = np.asarray(normalized_envelope, dtype=float)
    if envelope.ndim != 2:
        raise ValueError("normalized_envelope must have shape (epochs, samples)")
    window_offsets = np.arange(-int(half_window_samples), int(half_window_samples) + 1)
    accumulated = np.zeros(len(window_offsets), dtype=float)
    retained = 0
    for episode in episodes.itertuples(index=False):
        epoch_index = int(episode.epoch_index)
        center = (int(episode.start_sample) + int(episode.stop_sample_exclusive) - 1) // 2
        indices = center + window_offsets
        if (
            epoch_index < 0
            or epoch_index >= envelope.shape[0]
            or indices[0] < 0
            or indices[-1] >= envelope.shape[1]
        ):
            continue
        segment = envelope[epoch_index, indices]
        if np.all(np.isfinite(segment)):
            accumulated += segment
            retained += 1
    if retained == 0:
        return np.full(len(window_offsets), np.nan), 0
    return accumulated / retained, retained


def mean_centered_analytic(
    normalized_analytic: np.ndarray,
    episodes: pd.DataFrame,
    *,
    half_window_samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Average envelope, relative-phase phasor, and phase-aligned waveform."""
    analytic = np.asarray(normalized_analytic, dtype=np.complex128)
    if analytic.ndim != 2:
        raise ValueError("normalized_analytic must have shape (epochs, samples)")
    window_offsets = np.arange(-int(half_window_samples), int(half_window_samples) + 1)
    envelope_sum = np.zeros(len(window_offsets), dtype=float)
    phasor_sum = np.zeros(len(window_offsets), dtype=np.complex128)
    shape_sum = np.zeros(len(window_offsets), dtype=float)
    retained = 0
    center_offset = int(half_window_samples)
    for episode in episodes.itertuples(index=False):
        epoch_index = int(episode.epoch_index)
        center = (
            int(episode.start_sample) + int(episode.stop_sample_exclusive) - 1
        ) // 2
        indices = center + window_offsets
        if (
            epoch_index < 0
            or epoch_index >= analytic.shape[0]
            or indices[0] < 0
            or indices[-1] >= analytic.shape[1]
        ):
            continue
        segment = analytic[epoch_index, indices]
        if not np.all(np.isfinite(segment)):
            continue
        rotation = np.exp(-1j * np.angle(segment[center_offset]))
        aligned = segment * rotation
        magnitude = np.abs(aligned)
        unit_phase = np.divide(
            aligned,
            magnitude,
            out=np.zeros_like(aligned),
            where=magnitude > 0.0,
        )
        envelope_sum += magnitude
        phasor_sum += unit_phase
        shape_sum += np.real(aligned)
        retained += 1
    if retained == 0:
        missing = np.full(len(window_offsets), np.nan)
        return missing, missing.astype(np.complex128), missing.copy(), 0
    return (
        envelope_sum / retained,
        phasor_sum / retained,
        shape_sum / retained,
        retained,
    )


def _subject_envelopes(task: tuple[Any, ...]) -> dict[str, Any]:
    (
        subject_id,
        group,
        epoch_path,
        episode_path,
        electrodes,
        bands,
        filter_order,
        window_seconds,
        edge_padding_seconds,
    ) = task
    epochs = mne.read_epochs(epoch_path, preload=True, verbose="ERROR")
    picks = [epochs.ch_names.index(electrode) for electrode in electrodes]
    data_uv = epochs.get_data(picks=picks, copy=True) * 1e6
    sfreq = float(epochs.info["sfreq"])
    half_window_samples = int(round(float(window_seconds) * sfreq))
    edge_samples = int(round(float(edge_padding_seconds) * sfreq))
    if half_window_samples < 1:
        raise ValueError("Typical-bout window is shorter than one sample")
    required = {
        "subject_id",
        "group",
        "electrode",
        "band",
        "epoch_index",
        "start_sample",
        "stop_sample_exclusive",
    }
    episodes = pd.read_csv(episode_path, usecols=sorted(required))
    missing = sorted(required - set(episodes.columns))
    if missing:
        raise ValueError(f"{episode_path} is missing episode columns: {missing}")
    if len(episodes) and (
        set(episodes["subject_id"].astype(str)) != {subject_id}
        or set(episodes["group"].astype(str)) != {group}
    ):
        raise ValueError(f"{episode_path} has inconsistent subject/group labels")

    n_times = 2 * half_window_samples + 1
    waveforms = np.full((len(electrodes), len(bands), n_times), np.nan)
    phase_phasors = np.full(
        (len(electrodes), len(bands), n_times), np.nan + 0j
    )
    phase_aligned_shapes = np.full(
        (len(electrodes), len(bands), n_times), np.nan
    )
    counts = np.zeros((len(electrodes), len(bands)), dtype=np.int64)
    baselines = np.full((len(electrodes), len(bands)), np.nan)
    for band_index, (band, limits) in enumerate(bands.items()):
        filtered = filter_epoch_data(
            data_uv,
            sfreq=sfreq,
            low_hz=float(limits[0]),
            high_hz=float(limits[1]),
            order=int(filter_order),
        )
        analytic = hilbert(filtered, axis=-1)
        amplitude = np.abs(analytic)
        interior = (
            amplitude
            if edge_samples == 0
            else amplitude[..., edge_samples:-edge_samples]
        )
        baseline = np.median(interior, axis=(0, 2))
        if not np.all(np.isfinite(baseline)) or np.any(baseline <= 0.0):
            raise RuntimeError(f"{subject_id}/{band}: invalid amplitude baseline")
        normalized_analytic = analytic / baseline[np.newaxis, :, np.newaxis]
        for electrode_index, electrode in enumerate(electrodes):
            selected = episodes.loc[
                episodes["electrode"].eq(electrode) & episodes["band"].eq(band)
            ]
            waveform, phasor, shape, retained = mean_centered_analytic(
                normalized_analytic[:, electrode_index, :],
                selected,
                half_window_samples=half_window_samples,
            )
            waveforms[electrode_index, band_index] = waveform
            phase_phasors[electrode_index, band_index] = phasor
            phase_aligned_shapes[electrode_index, band_index] = shape
            counts[electrode_index, band_index] = retained
            baselines[electrode_index, band_index] = baseline[electrode_index]
    return {
        "subject_id": subject_id,
        "group": group,
        "sfreq": sfreq,
        "waveforms": waveforms,
        "phase_phasors": phase_phasors,
        "phase_aligned_shapes": phase_aligned_shapes,
        "bout_counts": counts,
        "baseline_amplitude_uv": baselines,
    }


def _mean_ci(values: np.ndarray, confidence_level: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    array = np.asarray(values, dtype=float)
    finite_rows = np.all(np.isfinite(array), axis=1)
    array = array[finite_rows]
    if len(array) == 0:
        missing = np.full(values.shape[1], np.nan)
        return missing, missing, missing
    mean = np.mean(array, axis=0)
    if len(array) == 1:
        return mean, mean.copy(), mean.copy()
    sem = np.std(array, axis=0, ddof=1) / np.sqrt(len(array))
    critical = float(student_t.ppf(0.5 + float(confidence_level) / 2.0, len(array) - 1))
    return mean, mean - critical * sem, mean + critical * sem


def _select_group_curves(
    values: np.ndarray,
    bout_counts: np.ndarray,
    subject_table: pd.DataFrame,
    electrode_index: int | None,
    band_index: int,
    policy: str,
    group: str,
) -> tuple[np.ndarray, int]:
    eligible = subject_table["group"].eq(group).to_numpy(
        dtype=bool, copy=True
    )
    if policy == "fit_qc":
        eligible &= subject_table["subject_fit_qc_pass"].to_numpy(dtype=bool)
    if electrode_index is None:
        subject_curves = []
        group_bouts = 0
        for subject_index in np.flatnonzero(eligible):
            electrode_mask = bout_counts[subject_index, :, band_index] > 0
            if policy == "fit_qc":
                electrode_mask &= subject_table.iloc[subject_index]["electrode_fit_qc"]
            curves = values[subject_index, electrode_mask, band_index]
            if len(curves):
                subject_curves.append(np.mean(curves, axis=0))
                group_bouts += int(
                    bout_counts[subject_index, electrode_mask, band_index].sum()
                )
        return np.asarray(subject_curves, dtype=values.dtype), group_bouts
    if policy == "fit_qc":
        electrode_pass = np.asarray(
            [row[electrode_index] for row in subject_table["electrode_fit_qc"]],
            dtype=bool,
        )
        eligible &= electrode_pass
    eligible &= bout_counts[:, electrode_index, band_index] > 0
    selected = values[eligible, electrode_index, band_index]
    group_bouts = int(bout_counts[eligible, electrode_index, band_index].sum())
    return selected, group_bouts


def _plot_scalar_axis(
    axis: plt.Axes,
    values: np.ndarray,
    bout_counts: np.ndarray,
    subject_table: pd.DataFrame,
    electrode_index: int | None,
    band_index: int,
    times: np.ndarray,
    policy: str,
    confidence_level: float,
    *,
    title: str,
    ylabel: str,
    reference: float | None,
    include_counts: bool,
) -> None:
    colors = {"PD": "#D55E00", "Control": "#0072B2"}
    for group in ("PD", "Control"):
        selected, group_bouts = _select_group_curves(
            values,
            bout_counts,
            subject_table,
            electrode_index,
            band_index,
            policy,
            group,
        )
        mean, lower, upper = _mean_ci(selected, confidence_level)
        label = group
        if include_counts:
            label += f" (subjects={len(selected)}, bouts={group_bouts:,})"
        axis.plot(times, mean, color=colors[group], linewidth=1.6, label=label)
        axis.fill_between(times, lower, upper, color=colors[group], alpha=0.20)
    axis.axvline(0.0, color="0.35", linestyle="--", linewidth=0.8)
    if reference is not None:
        axis.axhline(reference, color="0.5", linestyle=":", linewidth=0.8)
    axis.set(title=title, xlabel="Time from bout center (s)", ylabel=ylabel)
    axis.grid(alpha=0.18)
    axis.legend(frameon=False, fontsize=6.5)


def _plot_phase_axis(
    axis: plt.Axes,
    phase_phasors: np.ndarray,
    bout_counts: np.ndarray,
    subject_table: pd.DataFrame,
    electrode_index: int | None,
    band_index: int,
    times: np.ndarray,
    policy: str,
    *,
    title: str,
) -> None:
    colors = {"PD": "#D55E00", "Control": "#0072B2"}
    concentration_axis = axis.twinx()
    for group in ("PD", "Control"):
        selected, _ = _select_group_curves(
            phase_phasors,
            bout_counts,
            subject_table,
            electrode_index,
            band_index,
            policy,
            group,
        )
        finite_rows = np.all(np.isfinite(selected), axis=1)
        selected = selected[finite_rows]
        if len(selected) == 0:
            continue
        group_phasor = np.mean(selected, axis=0)
        mean_phase = np.unwrap(np.angle(group_phasor))
        mean_phase -= mean_phase[len(mean_phase) // 2]
        concentration = np.abs(group_phasor)
        axis.plot(
            times,
            mean_phase / np.pi,
            color=colors[group],
            linewidth=1.5,
            label=f"{group} phase",
        )
        concentration_axis.plot(
            times,
            concentration,
            color=colors[group],
            linewidth=1.1,
            linestyle=":",
            alpha=0.65,
            label=f"{group} R",
        )
    axis.axvline(0.0, color="0.35", linestyle="--", linewidth=0.8)
    axis.axhline(0.0, color="0.5", linestyle=":", linewidth=0.8)
    axis.set(
        title=title,
        xlabel="Time from bout center (s)",
        ylabel="Circular mean phase (π radians)",
    )
    concentration_axis.set(ylabel="Phase consistency R", ylim=(0.0, 1.05))
    axis.grid(alpha=0.18)
    handles, labels = axis.get_legend_handles_labels()
    r_handles, r_labels = concentration_axis.get_legend_handles_labels()
    axis.legend(handles + r_handles, labels + r_labels, frameon=False, fontsize=6.2)


def _plot_representations(
    axes: np.ndarray,
    waveforms: np.ndarray,
    phase_phasors: np.ndarray,
    phase_aligned_shapes: np.ndarray,
    bout_counts: np.ndarray,
    subject_table: pd.DataFrame,
    electrode_index: int | None,
    bands: list[str],
    times: np.ndarray,
    policy: str,
    confidence_level: float,
) -> None:
    for band_index, band in enumerate(bands):
        display = band.replace("_", " ").title()
        _plot_scalar_axis(
            axes[band_index, 0],
            waveforms,
            bout_counts,
            subject_table,
            electrode_index,
            band_index,
            times,
            policy,
            confidence_level,
            title=f"{display} — envelope",
            ylabel="Hilbert amplitude above baseline (µV)",
            reference=0.0,
            include_counts=True,
        )
        _plot_phase_axis(
            axes[band_index, 1],
            phase_phasors,
            bout_counts,
            subject_table,
            electrode_index,
            band_index,
            times,
            policy,
            title=f"{display} — relative Hilbert phase",
        )
        _plot_scalar_axis(
            axes[band_index, 2],
            phase_aligned_shapes,
            bout_counts,
            subject_table,
            electrode_index,
            band_index,
            times,
            policy,
            confidence_level,
            title=f"{display} — phase-aligned shape",
            ylabel="Band-passed voltage (µV)",
            reference=0.0,
            include_counts=False,
        )


def _representation_figure(n_bands: int) -> tuple[plt.Figure, np.ndarray]:
    """Allocate one envelope/phase/shape row for every configured band."""
    if n_bands < 1:
        raise ValueError("At least one bout band is required for the gallery")
    return plt.subplots(
        n_bands,
        3,
        figsize=(18, 4.25 * n_bands),
        squeeze=False,
    )


def _plot_electrode(
    waveforms: np.ndarray,
    phase_phasors: np.ndarray,
    phase_aligned_shapes: np.ndarray,
    bout_counts: np.ndarray,
    subject_table: pd.DataFrame,
    electrode_index: int,
    electrode: str,
    bands: list[str],
    times: np.ndarray,
    policy: str,
    confidence_level: float,
    output_path: Path,
    dpi: int,
) -> None:
    fig, axes = _representation_figure(len(bands))
    _plot_representations(
        axes,
        waveforms,
        phase_phasors,
        phase_aligned_shapes,
        bout_counts,
        subject_table,
        electrode_index,
        bands,
        times,
        policy,
        confidence_level,
    )
    policy_label = "all subjects" if policy == "all" else "fit-QC electrodes and qualified subjects"
    fig.suptitle(
        f"{electrode}: stereotypical detected bout ({policy_label})\n"
        "envelope, circular phase, and phase-aligned shape; subject-balanced summaries"
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)


def _write_gallery(root: Path, electrodes: list[str]) -> None:
    links = []
    for electrode in electrodes:
        safe = electrode.replace("/", "_")
        links.append(
            "<tr>"
            f"<td>{html.escape(electrode)}</td>"
            f'<td><a href="all_subjects/{html.escape(safe)}.png">all subjects</a></td>'
            f'<td><a href="fit_qc/{html.escape(safe)}.png">fit-QC sensitivity</a></td>'
            "</tr>"
        )
    document = (
        "<!doctype html><html><head><meta charset=\"utf-8\">"
        "<title>Typical bout representations</title>"
        "<style>body{font-family:sans-serif;margin:2rem;color:#222}"
        "table{border-collapse:collapse}td,th{border:1px solid #ddd;padding:.45rem}"
        "a{color:#0067a5}</style></head><body>"
        "<h1>Subject-balanced stereotypical bout representations</h1>"
        "<p>Each figure shows the baseline-subtracted Hilbert envelope in "
        "microvolts, circular mean "
        "phase relative to the bout center with phase consistency R, and the "
        "phase-aligned band-pass waveform. Confidence shading is across subject "
        "means, not across bouts.</p>"
        '<p><a href="grand_average_all_subjects.png">Grand average: all subjects</a> · '
        '<a href="grand_average_fit_qc.png">Grand average: fit-QC sensitivity</a></p>'
        '<p>Detection coverage QC: '
        '<a href="bout_detection_subject_coverage.png">subject coverage</a> · '
        '<a href="bout_count_qc.png">median bout counts</a></p>'
        "<table><thead><tr><th>Electrode</th><th>All subjects</th><th>Fit QC</th>"
        f"</tr></thead><tbody>{''.join(links)}</tbody></table></body></html>"
    )
    root.mkdir(parents=True, exist_ok=True)
    (root / "index.html").write_text(document, encoding="utf-8")


def _coverage_summary(
    coverage: pd.DataFrame,
    electrodes: list[str],
    bands: list[str],
) -> pd.DataFrame:
    """Summarize subject support for every group curve and QC policy."""
    rows = []
    for policy in ("all_subjects", "fit_qc"):
        for group in ("PD", "Control"):
            group_rows = coverage.loc[coverage["group"].eq(group)]
            for electrode in electrodes:
                for band in bands:
                    selected = group_rows.loc[
                        group_rows["electrode"].eq(electrode)
                        & group_rows["band"].eq(band)
                    ]
                    if policy == "fit_qc":
                        selected = selected.loc[
                            selected["subject_fit_qc_pass"]
                            & selected["electrode_fit_qc_pass"]
                        ]
                    values = selected["n_bouts_in_typical_average"].to_numpy(dtype=int)
                    contributing = values > 0
                    rows.append(
                        {
                            "policy": policy,
                            "group": group,
                            "electrode": electrode,
                            "band": band,
                            "n_eligible_subjects": int(len(values)),
                            "n_contributing_subjects": int(contributing.sum()),
                            "contributing_subject_fraction": (
                                float(contributing.mean()) if len(values) else math.nan
                            ),
                            "total_bouts": int(values.sum()),
                            "median_bouts_per_eligible_subject": (
                                float(np.median(values)) if len(values) else math.nan
                            ),
                            "mean_bouts_per_eligible_subject": (
                                float(np.mean(values)) if len(values) else math.nan
                            ),
                        }
                    )
    return pd.DataFrame.from_records(rows)


def _plot_coverage_heatmap(
    summary: pd.DataFrame,
    electrodes: list[str],
    bands: list[str],
    value: str,
    title: str,
    colorbar_label: str,
    output_path: Path,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(20, 8), squeeze=False, sharex=True, sharey=True)
    panels = [
        ("all_subjects", "PD"),
        ("all_subjects", "Control"),
        ("fit_qc", "PD"),
        ("fit_qc", "Control"),
    ]
    values = summary[value].to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    vmax = (
        1.0
        if value == "contributing_subject_fraction"
        else float(np.quantile(finite, 0.98))
    )
    vmax = max(vmax, np.finfo(float).eps)
    image_handle = None
    for axis, (policy, group) in zip(axes.flat, panels):
        selected = summary.loc[
            summary["policy"].eq(policy) & summary["group"].eq(group)
        ]
        matrix = (
            selected.pivot(index="band", columns="electrode", values=value)
            .reindex(index=bands, columns=electrodes)
            .to_numpy(dtype=float)
        )
        image_handle = axis.imshow(
            matrix,
            aspect="auto",
            interpolation="nearest",
            vmin=0.0,
            vmax=vmax,
            cmap="viridis",
        )
        policy_label = (
            "all subjects" if policy == "all_subjects" else "fit-QC sensitivity"
        )
        axis.set_title(f"{group} — {policy_label}")
        axis.set_yticks(
            np.arange(len(bands)), [band.replace("_", " ") for band in bands]
        )
        axis.set_xticks(
            np.arange(len(electrodes)), electrodes, rotation=90, fontsize=6
        )
    fig.suptitle(title)
    fig.supxlabel("Electrode")
    fig.supylabel("Frequency band")
    fig.tight_layout(rect=(0.02, 0.04, 0.94, 0.94))
    assert image_handle is not None
    color_axis = fig.add_axes((0.95, 0.14, 0.012, 0.70))
    fig.colorbar(image_handle, cax=color_axis, label=colorbar_label)
    fig.savefig(output_path, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)


def generate_typical_bout_gallery(
    config_path: str | Path = "config/analyses/scale_free.json",
    *,
    workers: int | None = None,
) -> dict[str, Any]:
    """Extract subject means and render all-electrode typical-bout figures."""
    config_path = Path(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    settings = config["typical_bouts"]
    output_root = Path(config["output_dir"])
    participants = pd.read_csv(config["input"]["participants_file"])
    participants = participants.rename(
        columns={"participant_id": "subject_id", "GROUP": "group"}
    )
    electrode_payload = json.loads(
        (output_root / "metrics" / "electrode_sets.json").read_text(encoding="utf-8")
    )
    electrodes = [str(value) for value in electrode_payload["common_electrodes"]]
    bands = {str(name): limits for name, limits in config["bands"].items()}
    epoch_dir = Path(config["input"]["epochs_dir"])
    episode_dir = output_root / "intermediate" / "episodes"
    tasks = []
    for row in participants.itertuples(index=False):
        epoch_matches = list(epoch_dir.glob(f"{row.subject_id}_task-Rest_desc-cleaned_epo.fif"))
        if len(epoch_matches) != 1:
            raise FileNotFoundError(f"Expected one epoch file for {row.subject_id}")
        episode_path = episode_dir / f"{row.subject_id}_bout_episodes.csv.gz"
        if not episode_path.exists():
            raise FileNotFoundError(f"Missing bout episodes: {episode_path}")
        tasks.append(
            (
                str(row.subject_id),
                str(row.group),
                str(epoch_matches[0]),
                str(episode_path),
                electrodes,
                bands,
                int(settings["bandpass_filter_order"]),
                float(settings["center_window_seconds"]),
                float(config["ebosc"]["edge_padding_seconds"]),
            )
        )
    worker_count = int(workers if workers is not None else settings["workers"])
    results: list[dict[str, Any]] = []
    if worker_count == 1:
        results = [_subject_envelopes(task) for task in tasks]
    else:
        try:
            with ProcessPoolExecutor(max_workers=worker_count) as executor:
                futures = [executor.submit(_subject_envelopes, task) for task in tasks]
                for future in as_completed(futures):
                    results.append(future.result())
        except PermissionError:
            results = [_subject_envelopes(task) for task in tasks]
    order = {str(row.subject_id): index for index, row in enumerate(participants.itertuples(index=False))}
    results.sort(key=lambda row: order[row["subject_id"]])
    sfreqs = {float(row["sfreq"]) for row in results}
    if len(sfreqs) != 1:
        raise ValueError("Typical-bout inputs use inconsistent sampling frequencies")
    sfreq = sfreqs.pop()
    waveforms = np.stack([row["waveforms"] for row in results])
    phase_phasors = np.stack([row["phase_phasors"] for row in results])
    phase_aligned_shapes = np.stack(
        [row["phase_aligned_shapes"] for row in results]
    )
    counts = np.stack([row["bout_counts"] for row in results])
    baselines = np.stack([row["baseline_amplitude_uv"] for row in results])
    amplitude_waveforms_uv = (waveforms - 1.0) * baselines[..., np.newaxis]
    phase_aligned_shapes_uv = phase_aligned_shapes * baselines[..., np.newaxis]
    fit = pd.read_csv(output_root / "metrics" / "electrode_aperiodic_metrics.csv")
    fit_lookup = fit.pivot(index="subject_id", columns="electrode", values="specparam_fit_qc_pass")
    fit_lookup = fit_lookup.reindex(index=participants["subject_id"], columns=electrodes)
    if fit_lookup.isna().any().any():
        raise ValueError("Fit-QC table does not cover every subject/electrode")
    coverage = pd.read_csv(output_root / "metrics" / "subject_specparam_fit_failures.csv")
    coverage = coverage.set_index("subject_id").reindex(participants["subject_id"])
    subject_table = participants[["subject_id", "group"]].copy()
    subject_table["subject_fit_qc_pass"] = coverage["subject_fit_qc_pass"].to_numpy(dtype=bool)
    subject_table["electrode_fit_qc"] = list(fit_lookup.to_numpy(dtype=bool))
    half_window_samples = int(round(float(settings["center_window_seconds"]) * sfreq))
    times = np.arange(-half_window_samples, half_window_samples + 1) / sfreq

    intermediate = output_root / "intermediate" / "typical_bouts"
    intermediate.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        intermediate / "subject_electrode_band_envelopes.npz",
        subject_ids=participants["subject_id"].to_numpy(dtype=str),
        groups=participants["group"].to_numpy(dtype=str),
        electrodes=np.asarray(electrodes),
        bands=np.asarray(list(bands)),
        times_seconds=times,
        normalized_amplitude_envelopes=waveforms,
        baseline_subtracted_amplitude_envelopes_uv=amplitude_waveforms_uv,
        relative_phase_phasors=phase_phasors,
        phase_aligned_normalized_shapes=phase_aligned_shapes,
        phase_aligned_shapes_uv=phase_aligned_shapes_uv,
        bout_counts=counts,
        baseline_amplitude_uv=baselines,
        electrode_fit_qc=fit_lookup.to_numpy(dtype=bool),
        subject_fit_qc=subject_table["subject_fit_qc_pass"].to_numpy(dtype=bool),
    )
    coverage_rows = []
    for subject_index, subject in subject_table.iterrows():
        for electrode_index, electrode in enumerate(electrodes):
            for band_index, band in enumerate(bands):
                coverage_rows.append(
                    {
                        "subject_id": subject["subject_id"],
                        "group": subject["group"],
                        "electrode": electrode,
                        "band": band,
                        "n_bouts_in_typical_average": int(counts[subject_index, electrode_index, band_index]),
                        "baseline_amplitude_uv": float(baselines[subject_index, electrode_index, band_index]),
                        "electrode_fit_qc_pass": bool(fit_lookup.iloc[subject_index, electrode_index]),
                        "subject_fit_qc_pass": bool(subject["subject_fit_qc_pass"]),
                    }
                )
    coverage_table = pd.DataFrame.from_records(coverage_rows)
    coverage_table.to_csv(
        output_root / "metrics" / "typical_bout_coverage.csv",
        index=False,
        float_format="%.17g",
    )

    gallery_root = output_root / "figures" / "typical_bouts"
    gallery_root.mkdir(parents=True, exist_ok=True)
    coverage_summary = _coverage_summary(coverage_table, electrodes, list(bands))
    coverage_summary.to_csv(
        output_root / "metrics" / "typical_bout_group_coverage.csv",
        index=False,
        float_format="%.17g",
    )
    _plot_coverage_heatmap(
        coverage_summary,
        electrodes,
        list(bands),
        "contributing_subject_fraction",
        "Detected-bout subject coverage by group, band, and electrode",
        "Fraction of eligible subjects with at least one retained bout",
        gallery_root / "bout_detection_subject_coverage.png",
        int(config["plots"]["dpi"]),
    )
    _plot_coverage_heatmap(
        coverage_summary,
        electrodes,
        list(bands),
        "median_bouts_per_eligible_subject",
        "Median detected bouts per eligible subject",
        "Median retained bouts per subject",
        gallery_root / "bout_count_qc.png",
        int(config["plots"]["dpi"]),
    )
    for electrode_index, electrode in enumerate(electrodes):
        safe = electrode.replace("/", "_")
        for policy, directory in (("all", "all_subjects"), ("fit_qc", "fit_qc")):
            _plot_electrode(
                amplitude_waveforms_uv,
                phase_phasors,
                phase_aligned_shapes_uv,
                counts,
                subject_table,
                electrode_index,
                electrode,
                list(bands),
                times,
                policy,
                float(settings["confidence_level"]),
                gallery_root / directory / f"{safe}.png",
                int(config["plots"]["dpi"]),
            )
    for policy, filename in (
        ("all", "grand_average_all_subjects.png"),
        ("fit_qc", "grand_average_fit_qc.png"),
    ):
        fig, axes = _representation_figure(len(bands))
        _plot_representations(
            axes,
            amplitude_waveforms_uv,
            phase_phasors,
            phase_aligned_shapes_uv,
            counts,
            subject_table,
            None,
            list(bands),
            times,
            policy,
            float(settings["confidence_level"]),
        )
        label = "all subjects" if policy == "all" else "fit-QC sensitivity"
        fig.suptitle(
            f"Grand-average stereotypical detected bout ({label})\n"
            "envelope, circular phase, and phase-aligned shape; electrodes averaged within subject"
        )
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
        fig.savefig(gallery_root / filename, dpi=int(config["plots"]["dpi"]), bbox_inches="tight")
        plt.close(fig)
    _write_gallery(gallery_root, electrodes)
    payload = {
        "n_subjects": int(len(subject_table)),
        "group_counts": subject_table["group"].value_counts().to_dict(),
        "n_fit_qc_subjects": int(subject_table["subject_fit_qc_pass"].sum()),
        "n_electrodes": int(len(electrodes)),
        "bands": list(bands),
        "alignment": "detected bout midpoint",
        "window_seconds_each_side": float(settings["center_window_seconds"]),
        "plot_amplitude_units": "microvolts above the subject/electrode/band median Hilbert baseline",
        "stored_normalization": "subject/electrode/band median Hilbert amplitude in valid epoch interiors",
        "phase": "circular mean relative to each bout center; resultant length R reports consistency",
        "average_shape": "real phase-aligned analytic signal with center phase rotated to zero",
        "aggregation": "bouts within subject/electrode/band, then subjects with equal weight",
        "confidence_interval": (
            f"{100 * float(settings['confidence_level']):g}% Student-t interval across subject means"
        ),
    }
    (output_root / "typical_bouts_manifest.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    return payload
