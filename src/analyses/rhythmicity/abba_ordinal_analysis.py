"""Ordinal H, C, and F in group-specific or Control-defined ABBA bands.

The analysis uses the group-mean ABBA intervals produced by the rhythmicity
pipeline.  Each accepted EEG epoch is band-pass filtered independently.  For
the full-signal estimate, the filtered epochs are concatenated; for the bout
estimate, ABBA-band amplitude bouts are detected in the robust all-electrode
signal and the corresponding pieces of each electrode are concatenated.
Ordinal patterns are therefore allowed to cross concatenation joins, exactly
as requested, and this policy is retained in every output row.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import platform
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.runtime import configure_runtime

configure_runtime()

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import mne
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from tqdm.auto import tqdm

from analyses.ordinal.metrics import (
    filter_epoch_data,
    metrics_from_probabilities,
    ordinal_probabilities,
)
from analyses.rhythmicity.abba_burst_analysis import _detect_bursts, _global_signal
from analyses.rhythmicity.control_band_qc import (
    QC_BAND_DEFINITION,
    control_defined_segments,
    qc_output_root,
)


METRICS = ("entropy", "complexity", "fisher_information")
METRIC_LABELS = {
    "entropy": "Permutation entropy (H)",
    "complexity": "Statistical complexity (C)",
    "fisher_information": "Fisher information (F)",
}
GROUP_COLORS = {
    "Control": "#0072B2",
    "PD": "#D55E00",
    "PD_OFF": "#7570B3",
    "PD_ON": "#009E73",
}
GROUP_MARKERS = {
    "Control": "o",
    "PD": "s",
    "PD_OFF": "^",
    "PD_ON": "D",
}
CONCATENATION_POLICY = "literal_concatenation_patterns_may_cross_joins"


def _safe_name(value: str) -> str:
    return "".join(character if character.isalnum() or character in "-_" else "_" for character in value)


def _comparison_band_name(task: dict[str, Any], band_name: str, direction: str) -> str:
    """Map native ABBA labels onto explicitly comparable interval labels."""
    for alignment in task.get("comparison_band_alignments", []):
        if str(alignment.get("dataset")) != str(task["dataset"]):
            continue
        source = alignment.get("source_band_by_group", {}).get(str(task["group"]))
        if source != str(band_name):
            continue
        required = alignment.get("required_direction")
        if required is not None and str(required) != str(direction):
            raise ValueError(
                f"Configured alignment {alignment['comparison_band_name']} expected "
                f"{required} LAVI for {task['group']} {band_name}, found {direction}"
            )
        return str(alignment["comparison_band_name"])
    return str(band_name)


def _cross_dataset_band_name(canonical_region: str, direction: str) -> str:
    """Name ABBA bands by canonical region and shared LAVI characteristic."""
    return f"{canonical_region}_{direction}"


def _atomic_pickle(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _write_csv(table: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False, float_format="%.17g", compression="infer")


def concatenate_bouts(filtered: np.ndarray, bursts: list[dict[str, float]]) -> np.ndarray:
    """Concatenate detected intervals into one signal per electrode."""
    values = np.asarray(filtered, dtype=float)
    if values.ndim != 3:
        raise ValueError("filtered must have shape (epochs, electrodes, samples)")
    pieces = [
        values[
            int(burst["epoch_index"]),
            :,
            int(burst["start_sample"]):int(burst["stop_sample_exclusive"]),
        ]
        for burst in bursts
    ]
    return np.concatenate(pieces, axis=1) if pieces else np.empty((values.shape[1], 0))


def _electrode_metrics(
    signals: np.ndarray,
    channels: list[str],
    *,
    dx: int,
    tau: int,
    minimum_samples: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for channel, signal in zip(channels, np.asarray(signals, dtype=float)):
        row: dict[str, Any] = {
            "electrode": channel,
            "n_samples": int(len(signal)),
            "n_ordinal_patterns": 0,
        }
        if len(signal) >= max(int(minimum_samples), (int(dx) - 1) * int(tau) + 1):
            probabilities, n_patterns, ties = ordinal_probabilities(
                signal[None, :], dx=dx, tau=tau, tie_precision=None
            )
            computed = metrics_from_probabilities(probabilities, dx=dx)
            row.update({metric: computed[metric] for metric in METRICS})
            row["n_ordinal_patterns"] = int(n_patterns)
            row["n_exact_tied_patterns"] = int(ties)
        else:
            row.update({metric: np.nan for metric in METRICS})
            row["n_exact_tied_patterns"] = 0
        rows.append(row)
    return rows


def _recording_task(task: dict[str, Any]) -> dict[str, Any]:
    epochs = mne.read_epochs(task["epoch_path"], preload=True, verbose="ERROR")
    picks = mne.pick_types(epochs.info, eeg=True, exclude=[])
    if len(picks) == 0:
        raise ValueError(f"No EEG channels in {task['epoch_path']}")
    data = epochs.get_data(picks=picks, copy=True)
    channels = [epochs.ch_names[index] for index in picks]
    sfreq = float(epochs.info["sfreq"])
    detection_signal = _global_signal(data)
    base = {
        key: task[key]
        for key in (
            "dataset", "recording_id", "participant_id", "session_id", "group",
            "medication_state", "moca", "mmse", "updrs", "updrs_source",
            "age_years", "sex",
        )
    }
    recording_rows: list[dict[str, Any]] = []
    electrode_rows: list[dict[str, Any]] = []
    for segment in task["segments"]:
        low_hz, high_hz = float(segment["start_hz"]), float(segment["end_hz"])
        if not 0.0 < low_hz < high_hz < sfreq / 2.0:
            continue
        filtered = filter_epoch_data(
            data,
            sfreq=sfreq,
            low_hz=low_hz,
            high_hz=high_hz,
            order=int(task["filter_order"]),
        )
        filtered_detection = filter_epoch_data(
            detection_signal[:, None, :],
            sfreq=sfreq,
            low_hz=low_hz,
            high_hz=high_hz,
            order=int(task["filter_order"]),
        )[:, 0, :]
        bursts, _, _ = _detect_bursts(
            filtered_detection,
            sfreq,
            low_hz,
            high_hz,
            detection_percentile=float(task["detection_percentile"]),
            boundary_percentile=float(task["boundary_percentile"]),
            minimum_cycles=float(task["minimum_cycles"]),
        )
        scopes = {
            "full_signal": filtered.transpose(1, 0, 2).reshape(len(channels), -1),
            "within_bout": concatenate_bouts(filtered, bursts),
        }
        segment_meta = {
            "segment_index": int(segment["segment_index"]),
            "band_name": str(segment["band_name"]),
            "canonical_region": str(segment["canonical_region"]),
            "direction": str(segment["direction"]),
            "start_hz": low_hz,
            "end_hz": high_hz,
            "peak_hz": float(segment["peak_hz"]),
            "source_group": str(segment.get("source_group", task["group"])),
            "source_band_name": str(segment.get("source_band_name", segment["band_name"])),
            "band_definition": str(segment.get("band_definition", "group_specific_abba")),
        }
        segment_meta["comparison_band_name"] = _comparison_band_name(
            task, segment_meta["band_name"], segment_meta["direction"]
        )
        segment_meta["cross_dataset_band_name"] = _cross_dataset_band_name(
            segment_meta["canonical_region"], segment_meta["direction"]
        )
        for scope, signals in scopes.items():
            metrics = _electrode_metrics(
                signals,
                channels,
                dx=int(task["embedding_dimension"]),
                tau=int(task["delay_samples"]),
                minimum_samples=int(task["minimum_scope_samples"]),
            )
            for row in metrics:
                electrode_rows.append({
                    **base,
                    **segment_meta,
                    "scope": scope,
                    "n_bouts": int(len(bursts)),
                    "concatenation_policy": CONCATENATION_POLICY,
                    **row,
                })
            valid = [row for row in metrics if np.isfinite(row["entropy"])]
            summary = {
                metric: float(np.mean([row[metric] for row in valid])) if valid else np.nan
                for metric in METRICS
            }
            recording_rows.append({
                **base,
                **segment_meta,
                "scope": scope,
                "n_epochs": int(data.shape[0]),
                "n_electrodes": int(len(channels)),
                "n_valid_electrodes": int(len(valid)),
                "n_samples_per_electrode": int(signals.shape[1]),
                "n_bouts": int(len(bursts)),
                "concatenation_policy": CONCATENATION_POLICY,
                **summary,
            })
    return {
        "task_signature": _task_signature(task),
        "recording_rows": recording_rows,
        "electrode_rows": electrode_rows,
    }


def _task_signature(task: dict[str, Any]) -> str:
    epoch_path = Path(str(task["epoch_path"]))
    stat = epoch_path.stat()
    payload = {
        key: task[key]
        for key in (
            "dataset", "recording_id", "group", "segments", "filter_order",
            "detection_percentile", "boundary_percentile", "minimum_cycles",
            "embedding_dimension", "delay_samples", "minimum_scope_samples",
        )
    }
    payload["epoch_path"] = str(epoch_path.resolve())
    payload["epoch_size"] = int(stat.st_size)
    payload["epoch_mtime_ns"] = int(stat.st_mtime_ns)
    encoded = json.dumps(payload, sort_keys=True, allow_nan=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _checkpoint_path(root: Path, dataset: str, recording_id: str) -> Path:
    digest = hashlib.sha1(recording_id.encode("utf-8")).hexdigest()[:10]
    return root / _safe_name(dataset) / f"{_safe_name(recording_id)}_{digest}.pkl"


def _medication_updrs(
    dataset_dir: Path, participant_id: str, session_id: str
) -> tuple[float, str]:
    behavior_dir = dataset_dir / participant_id / session_id / "beh"
    matches = sorted(behavior_dir.glob("*_task-rest_beh.json"))
    if len(matches) != 1:
        return np.nan, ""
    payload = json.loads(matches[0].read_text(encoding="utf-8"))
    questionnaires = payload.get("questionairres", payload.get("questionnaires", {}))
    value = pd.to_numeric(pd.Series([questionnaires.get("Total UPDRS")]), errors="coerce").iloc[0]
    return (float(value) if pd.notna(value) else np.nan), str(matches[0].resolve())


def _refresh_result_metadata(result: dict[str, Any], task: dict[str, Any]) -> None:
    """Attach current clinical metadata without invalidating EEG checkpoints."""
    keys = (
        "participant_id", "session_id", "group", "medication_state", "moca",
        "mmse", "updrs", "updrs_source", "age_years", "sex",
    )
    segments = {
        int(segment["segment_index"]): segment for segment in task["segments"]
    }
    for table_name in ("recording_rows", "electrode_rows"):
        for row in result[table_name]:
            row.update({key: task[key] for key in keys})
            segment = segments[int(row["segment_index"])]
            row["source_group"] = str(segment.get("source_group", task["group"]))
            row["source_band_name"] = str(
                segment.get("source_band_name", segment["band_name"])
            )
            row["band_definition"] = str(
                segment.get("band_definition", "group_specific_abba")
            )
            row["comparison_band_name"] = _comparison_band_name(
                task, str(row["band_name"]), str(row["direction"])
            )
            row["cross_dataset_band_name"] = _cross_dataset_band_name(
                str(row["canonical_region"]), str(row["direction"])
            )


def _participant_metrics(recordings: pd.DataFrame) -> pd.DataFrame:
    if recordings.empty:
        return recordings.copy()
    keys = [
        "dataset", "participant_id", "group", "comparison_band_name",
        "cross_dataset_band_name", "band_name", "canonical_region",
        "direction", "segment_index", "start_hz", "end_hz", "peak_hz", "scope",
        "source_group", "source_band_name", "band_definition",
    ]
    numeric = [*METRICS, "n_bouts", "n_samples_per_electrode"]
    result = recordings.groupby(keys, as_index=False)[numeric].mean(numeric_only=True)
    clinical = ["moca", "mmse", "updrs", "age_years"]
    metadata = recordings.groupby(keys, as_index=False)[clinical].mean(numeric_only=True)
    return result.merge(metadata, on=keys, validate="one_to_one")


def _fdr_bh(values: pd.Series) -> np.ndarray:
    p = values.to_numpy(dtype=float)
    adjusted = np.full(len(p), np.nan)
    valid = np.isfinite(p)
    if not valid.any():
        return adjusted
    selected = p[valid]
    order = np.argsort(selected)
    ranked = selected[order] * len(selected) / np.arange(1, len(selected) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    restored = np.empty_like(ranked)
    restored[order] = np.minimum(ranked, 1.0)
    adjusted[valid] = restored
    return adjusted


def compute_correlations(participant: pd.DataFrame, minimum_n: int = 5) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    comparison_column = (
        "comparison_band_name" if "comparison_band_name" in participant else "band_name"
    )
    for (dataset, band, scope), frame in participant.groupby(["dataset", comparison_column, "scope"]):
        group_sets = [("all", frame)] + [(str(group), selected) for group, selected in frame.groupby("group")]
        pd_frame = frame.loc[frame["group"].astype(str).str.startswith("PD")]
        if len(pd_frame) and set(pd_frame["group"]) != {"PD"}:
            group_sets.append(("PD_combined", pd_frame))
        for group_model, selected in group_sets:
            for outcome in ("moca", "mmse", "updrs"):
                for metric in METRICS:
                    paired = selected[[outcome, metric]].apply(pd.to_numeric, errors="coerce").dropna()
                    if len(paired) >= int(minimum_n) and paired.nunique().min() > 1:
                        test = spearmanr(paired[outcome], paired[metric])
                        rho, p_value = float(test.statistic), float(test.pvalue)
                    else:
                        rho, p_value = np.nan, np.nan
                    rows.append({
                        "dataset": dataset, "comparison_band_name": band, "scope": scope,
                        "group_model": group_model, "outcome": outcome, "metric": metric,
                        "n": int(len(paired)), "spearman_rho": rho, "p_value": p_value,
                    })
        paired_states = frame.loc[frame["group"].isin(["PD_OFF", "PD_ON"])]
        if {"PD_OFF", "PD_ON"}.issubset(set(paired_states["group"])):
            index = "participant_id"
            for metric in METRICS:
                values = paired_states.pivot_table(
                    index=index, columns="group", values=metric, aggfunc="mean"
                ).reindex(columns=["PD_OFF", "PD_ON"])
                values.columns = [f"{column}_metric" for column in values.columns]
                updrs = paired_states.pivot_table(
                    index=index, columns="group", values="updrs", aggfunc="mean"
                ).reindex(columns=["PD_OFF", "PD_ON"])
                updrs.columns = [f"{column}_updrs" for column in updrs.columns]
                joined = values.join(updrs).dropna()
                metric_delta = joined["PD_ON_metric"] - joined["PD_OFF_metric"]
                updrs_delta = joined["PD_ON_updrs"] - joined["PD_OFF_updrs"]
                if len(joined) >= int(minimum_n) and metric_delta.nunique() > 1 and updrs_delta.nunique() > 1:
                    test = spearmanr(updrs_delta, metric_delta)
                    rho, p_value = float(test.statistic), float(test.pvalue)
                else:
                    rho, p_value = np.nan, np.nan
                rows.append({
                    "dataset": dataset, "comparison_band_name": band, "scope": scope,
                    "group_model": "PD_ON_minus_PD_OFF", "outcome": "updrs_change",
                    "metric": metric, "n": int(len(joined)), "spearman_rho": rho,
                    "p_value": p_value,
                })
    result = pd.DataFrame.from_records(rows)
    if not result.empty:
        result["q_fdr_bh"] = _fdr_bh(result["p_value"])
        result["significant_fdr"] = result["q_fdr_bh"] < 0.05
    return result


def _scatter(axis: plt.Axes, frame: pd.DataFrame, outcome: str, metric: str) -> None:
    plotted = False
    for group, selected in frame.groupby("group", sort=False):
        valid = selected[[outcome, metric]].apply(pd.to_numeric, errors="coerce").dropna()
        if valid.empty:
            continue
        plotted = True
        color = GROUP_COLORS.get(str(group), "#666666")
        axis.scatter(
            valid[outcome],
            valid[metric],
            color=color,
            marker=GROUP_MARKERS.get(str(group), "o"),
            label=str(group).replace("PD_", "PD-"),
            s=34,
            alpha=0.82,
        )
        if len(valid) >= 3 and valid[outcome].nunique() > 1:
            coefficients = np.polyfit(valid[outcome], valid[metric], 1)
            grid = np.linspace(valid[outcome].min(), valid[outcome].max(), 50)
            axis.plot(grid, np.polyval(coefficients, grid), color=color, linewidth=1.1)
    axis.set_xlabel(outcome.upper())
    axis.set_ylabel(METRIC_LABELS[metric])
    axis.grid(alpha=0.18)
    if not plotted:
        axis.text(0.5, 0.5, f"No {outcome.upper()} data", ha="center", va="center", transform=axis.transAxes, color="0.45")


def _plot_band(frame: pd.DataFrame, output: Path, dpi: int) -> None:
    dataset = str(frame["dataset"].iloc[0])
    band = str(frame["comparison_band_name"].iloc[0])
    cognitive = "moca" if frame["moca"].notna().any() else "mmse"
    figure, axes = plt.subplots(4, 4, figsize=(18, 15), constrained_layout=True)
    for column, scope in enumerate(("full_signal", "within_bout")):
        selected = frame.loc[frame["scope"].eq(scope)]
        for group, group_frame in selected.groupby("group", sort=False):
            axes[0, column].scatter(
                group_frame["entropy"], group_frame["fisher_information"],
                c=group_frame["complexity"], cmap="viridis", vmin=0.0, vmax=1.0,
                edgecolor=GROUP_COLORS.get(str(group), "#666666"), linewidth=1.0,
                marker=GROUP_MARKERS.get(str(group), "o"),
                s=52, label=str(group).replace("PD_", "PD-"),
            )
        axes[0, column].set_xlabel("Permutation entropy (H)")
        axes[0, column].set_ylabel("Fisher information (F)")
        axes[0, column].set_title("Full filtered signal" if scope == "full_signal" else "Concatenated bouts")
        axes[0, column].grid(alpha=0.18)
        axes[0, column].legend(frameon=False, fontsize=8)
    colorbar = figure.colorbar(
        plt.cm.ScalarMappable(norm=plt.Normalize(0.0, 1.0), cmap="viridis"),
        ax=[axes[0, 0], axes[0, 1]],
        fraction=0.025,
        pad=0.02,
    )
    colorbar.set_label("Statistical complexity (C)")
    axes[0, 2].axis("off")
    axes[0, 3].axis("off")
    limits = frame.groupby("group")[["start_hz", "end_hz", "band_name"]].first()
    directions = frame.groupby("group")["direction"].first()
    limit_text = "\n".join(
        f"{str(group).replace('PD_', 'PD-')}: {row.band_name}, "
        f"{row.start_hz:g}–{row.end_hz:g} Hz ({directions[group]})"
        for group, row in limits.iterrows()
    )
    control_defined = "band_definition" in frame and frame["band_definition"].eq(QC_BAND_DEFINITION).all()
    limit_heading = "Control group-mean ABBA limits" if control_defined else "ABBA group-mean limits"
    axes[0, 2].text(0.0, 1.0, limit_heading + "\n" + limit_text, va="top", fontsize=11)
    axes[0, 3].text(
        0.0,
        1.0,
        "Color fill in H×F panels = C\nEdge color + marker = group\n"
        "Control ○   PD □   PD-OFF △   PD-ON ◇\nClinical lines are descriptive fits",
        va="top",
        fontsize=11,
    )
    for row, metric in enumerate(METRICS, start=1):
        for column, (scope, outcome) in enumerate((
            ("full_signal", cognitive), ("within_bout", cognitive),
            ("full_signal", "updrs"), ("within_bout", "updrs"),
        )):
            _scatter(axes[row, column], frame.loc[frame["scope"].eq(scope)], outcome, metric)
            axes[row, column].set_title(("Full" if scope == "full_signal" else "Bout") + f" × {outcome.upper()}")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    definition = "Control-defined ABBA" if control_defined else "ABBA"
    figure.suptitle(f"{dataset} — {definition} {band}: H, C, F in full signal and bouts", fontsize=16, fontweight="bold")
    if handles:
        figure.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=int(dpi), bbox_inches="tight")
    plt.close(figure)


def _plot_cross_dataset_plane(
    frame: pd.DataFrame,
    *,
    scope: str,
    band: str,
    dataset_order: list[str],
    output: Path,
    dpi: int,
) -> None:
    selected = frame.loc[
        frame["scope"].eq(scope) & frame["cross_dataset_band_name"].eq(band)
    ]
    figure, axes = plt.subplots(
        len(dataset_order),
        2,
        figsize=(11.5, 3.2 * len(dataset_order)),
        sharex="col",
        sharey="col",
        squeeze=False,
        constrained_layout=True,
    )
    for row_index, dataset in enumerate(dataset_order):
        dataset_frame = selected.loc[selected["dataset"].eq(dataset)]
        if dataset_frame.empty:
            for axis in axes[row_index]:
                axis.text(
                    0.5, 0.5, "No matching ABBA interval", ha="center", va="center",
                    transform=axis.transAxes, color="0.45",
                )
                axis.grid(alpha=0.18)
        else:
            for group, group_frame in dataset_frame.groupby("group", sort=False):
                style = {
                    "color": GROUP_COLORS.get(str(group), "#666666"),
                    "marker": GROUP_MARKERS.get(str(group), "o"),
                    "s": 38,
                    "alpha": 0.82,
                    "label": str(group).replace("PD_", "PD-"),
                }
                axes[row_index, 0].scatter(
                    group_frame["entropy"], group_frame["complexity"], **style
                )
                axes[row_index, 1].scatter(
                    group_frame["entropy"], group_frame["fisher_information"], **style
                )
            sources = dataset_frame.groupby("group").agg(
                band_name=("band_name", "first"),
                start_hz=("start_hz", "first"),
                end_hz=("end_hz", "first"),
            )
            source_text = "; ".join(
                f"{str(group).replace('PD_', 'PD-')} {row.band_name} "
                f"{row.start_hz:g}–{row.end_hz:g} Hz"
                for group, row in sources.iterrows()
            )
            axes[row_index, 0].text(
                0.01, 0.98, source_text, transform=axes[row_index, 0].transAxes,
                va="top", fontsize=7, color="0.35",
            )
            for axis in axes[row_index]:
                axis.grid(alpha=0.18)
        axes[row_index, 0].set_ylabel(f"{dataset}\nComplexity (C)")
        axes[row_index, 1].set_ylabel(f"{dataset}\nFisher information (F)")
    axes[0, 0].set_title("H × C plane", fontweight="bold")
    axes[0, 1].set_title("H × F plane", fontweight="bold")
    axes[-1, 0].set_xlabel("Permutation entropy (H)")
    axes[-1, 1].set_xlabel("Permutation entropy (H)")
    present_groups = [
        group for group in ("Control", "PD", "PD_OFF", "PD_ON")
        if group in set(selected["group"])
    ]
    handles = [
        Line2D(
            [], [], linestyle="none", marker=GROUP_MARKERS[group],
            markerfacecolor=GROUP_COLORS[group], markeredgecolor=GROUP_COLORS[group],
            markersize=7, label=group.replace("PD_", "PD-"),
        )
        for group in present_groups
    ]
    scope_label = "Full filtered signal" if scope == "full_signal" else "Concatenated bouts"
    figure.suptitle(
        f"{scope_label} — ABBA {band.replace('_', ' ')} across four datasets",
        fontsize=15,
        fontweight="bold",
    )
    if handles:
        figure.legend(
            handles=handles,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.015),
            ncol=len(handles),
            frameon=False,
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=int(dpi), bbox_inches="tight")
    plt.close(figure)


def _save_cross_dataset_planes(
    participant: pd.DataFrame,
    *,
    dataset_order: list[str],
    output_root: Path,
    dpi: int,
) -> list[Path]:
    dataset_counts = participant.groupby("cross_dataset_band_name")["dataset"].nunique()
    bands = sorted(dataset_counts.loc[dataset_counts.ge(2)].index.astype(str))
    figure_root = output_root / "figures" / "abba_ordinal" / "cross_dataset_planes"
    expected: list[Path] = []
    for scope in ("full_signal", "within_bout"):
        for band in bands:
            path = figure_root / f"{scope}__{_safe_name(band)}.png"
            _plot_cross_dataset_plane(
                participant,
                scope=scope,
                band=band,
                dataset_order=dataset_order,
                output=path,
                dpi=dpi,
            )
            expected.append(path)
    expected_set = set(expected)
    for path in figure_root.glob("*.png"):
        if path not in expected_set:
            path.unlink()
    return expected


def _make_tasks(config: dict[str, Any], datasets: list[str] | None, recordings: list[str] | None, *, control_bands_qc: bool = False) -> dict[str, list[dict[str, Any]]]:
    # Dimension-sensitivity runs write beneath their own output directory but
    # consume the ABBA segments produced by the shared rhythmicity analysis.
    output_root = Path(config.get("source_output_dir", config["output_dir"]))
    segments = pd.read_csv(output_root / "statistics" / "abba_group_mean_segments.csv")
    if control_bands_qc:
        segments = control_defined_segments(segments)
    settings = config["abba_ordinal"]
    allowed_groups = set(str(value) for value in settings["groups"])
    configured_datasets = settings.get("datasets")
    requested_datasets = set(datasets) if datasets else (
        set(str(value) for value in configured_datasets) if configured_datasets else None
    )
    requested_recordings = set(recordings) if recordings else None
    result: dict[str, list[dict[str, Any]]] = {}
    canonical_root = Path(config["global_output_root"]) / "canonical"
    for manifest_path in sorted(canonical_root.glob("*/recordings.csv.gz")):
        dataset = manifest_path.parent.name
        if requested_datasets is not None and dataset not in requested_datasets:
            continue
        manifest = pd.read_csv(manifest_path, low_memory=False)
        manifest = manifest.loc[manifest["group"].astype(str).isin(allowed_groups)]
        if requested_recordings is not None:
            manifest = manifest.loc[manifest["recording_id"].astype(str).isin(requested_recordings)]
        tasks: list[dict[str, Any]] = []
        for row in manifest.to_dict("records"):
            selector = segments["dataset"].eq(dataset) & segments["end_hz"].gt(segments["start_hz"])
            if not control_bands_qc:
                selector &= segments["group"].astype(str).eq(str(row["group"]))
            selected = segments.loc[selector]
            if selected.empty or not Path(str(row["epoch_path"])).is_file():
                continue
            task = {key: row.get(key, np.nan) for key in (
                "recording_id", "participant_id", "session_id", "group", "medication_state",
                "moca", "mmse", "updrs", "age_years", "sex", "epoch_path",
            )}
            task["updrs_source"] = "canonical_recordings.updrs"
            if dataset == "medication_state" and pd.isna(task["updrs"]):
                medication_dir = settings.get("medication_dataset_dir")
                if medication_dir:
                    task["updrs"], task["updrs_source"] = _medication_updrs(
                        Path(str(medication_dir)),
                        str(task["participant_id"]),
                        str(task["session_id"]),
                    )
            task.update({
                "dataset": dataset,
                "segments": selected.to_dict("records"),
                "filter_order": int(settings["filter_order"]),
                "detection_percentile": float(settings["detection_percentile"]),
                "boundary_percentile": float(settings["boundary_percentile"]),
                "minimum_cycles": float(settings["minimum_cycles"]),
                "embedding_dimension": int(settings["embedding_dimension"]),
                "delay_samples": int(settings["delay_samples"]),
                "minimum_scope_samples": int(settings["minimum_scope_samples"]),
                "comparison_band_alignments": [] if control_bands_qc else settings.get("comparison_band_alignments", []),
            })
            tasks.append(task)
        if tasks:
            result[dataset] = tasks
    return result


def run(
    config_path: str | Path = "config/analyses/rhythmicity.json",
    *,
    datasets: list[str] | None = None,
    recordings: list[str] | None = None,
    workers: int | None = None,
    overwrite: bool = False,
    generate_figures: bool = True,
    control_bands_qc: bool = False,
) -> dict[str, int]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    settings = config["abba_ordinal"]
    base_output_root = Path(config["output_dir"])
    output_root = qc_output_root(base_output_root) if control_bands_qc else base_output_root
    checkpoint_root = output_root / "intermediate" / "abba_ordinal_checkpoints"
    tasks_by_dataset = _make_tasks(config, datasets, recordings, control_bands_qc=control_bands_qc)
    if not tasks_by_dataset:
        raise RuntimeError("No recordings matched the requested ABBA ordinal cohort")
    worker_count = int(workers if workers is not None else settings.get("workers", 1))
    all_results: list[dict[str, Any]] = []
    resumed = 0
    for dataset, tasks in tasks_by_dataset.items():
        dataset_results: list[dict[str, Any]] = []
        pending: list[dict[str, Any]] = []
        for task in tasks:
            checkpoint = _checkpoint_path(checkpoint_root, dataset, str(task["recording_id"]))
            if checkpoint.is_file() and not overwrite:
                try:
                    with checkpoint.open("rb") as stream:
                        cached = pickle.load(stream)
                except (OSError, EOFError, pickle.UnpicklingError):
                    cached = {}
                if cached.get("task_signature") == _task_signature(task):
                    _refresh_result_metadata(cached, task)
                    dataset_results.append(cached)
                    resumed += 1
                    continue
            pending.append(task)
        if pending:
            if worker_count == 1:
                iterator = ((_recording_task(task), task) for task in pending)
                for result, task in tqdm(iterator, total=len(pending), desc=f"ABBA HCF {dataset}"):
                    _atomic_pickle(result, _checkpoint_path(checkpoint_root, dataset, str(task["recording_id"])))
                    dataset_results.append(result)
            else:
                try:
                    with ProcessPoolExecutor(max_workers=worker_count) as executor:
                        futures = {executor.submit(_recording_task, task): task for task in pending}
                        for future in tqdm(as_completed(futures), total=len(futures), desc=f"ABBA HCF {dataset}"):
                            task = futures[future]
                            result = future.result()
                            _atomic_pickle(result, _checkpoint_path(checkpoint_root, dataset, str(task["recording_id"])))
                            dataset_results.append(result)
                except PermissionError:
                    with ThreadPoolExecutor(max_workers=worker_count) as executor:
                        futures = {executor.submit(_recording_task, task): task for task in pending}
                        for future in tqdm(as_completed(futures), total=len(futures), desc=f"ABBA HCF {dataset} (threads)"):
                            task = futures[future]
                            result = future.result()
                            _atomic_pickle(result, _checkpoint_path(checkpoint_root, dataset, str(task["recording_id"])))
                            dataset_results.append(result)
        dataset_recordings = pd.DataFrame(row for result in dataset_results for row in result["recording_rows"])
        dataset_electrodes = pd.DataFrame(row for result in dataset_results for row in result["electrode_rows"])
        dataset_recordings = dataset_recordings.sort_values(["recording_id", "segment_index", "scope"])
        dataset_electrodes = dataset_electrodes.sort_values(["recording_id", "segment_index", "scope", "electrode"])
        dataset_dir = output_root / "metrics" / "abba_ordinal_by_dataset" / _safe_name(dataset)
        _write_csv(dataset_recordings, dataset_dir / "recording_metrics.csv.gz")
        _write_csv(dataset_electrodes, dataset_dir / "electrode_metrics.csv.gz")
        all_results.extend(dataset_results)
    recording_table = pd.DataFrame(row for result in all_results for row in result["recording_rows"])
    electrode_table = pd.DataFrame(row for result in all_results for row in result["electrode_rows"])
    recording_table = recording_table.sort_values(["dataset", "recording_id", "segment_index", "scope"])
    electrode_table = electrode_table.sort_values(["dataset", "recording_id", "segment_index", "scope", "electrode"])
    participant_table = _participant_metrics(recording_table)
    participant_table = participant_table.sort_values(["dataset", "participant_id", "segment_index", "scope"])
    correlations = compute_correlations(participant_table, int(settings.get("minimum_correlation_n", 5)))
    metrics_root = output_root / "metrics"
    statistics_root = output_root / "statistics"
    _write_csv(recording_table, metrics_root / "abba_ordinal_recording_metrics.csv.gz")
    _write_csv(electrode_table, metrics_root / "abba_ordinal_electrode_metrics.csv.gz")
    _write_csv(participant_table, metrics_root / "abba_ordinal_participant_metrics.csv.gz")
    _write_csv(correlations, statistics_root / "abba_ordinal_clinical_correlations.csv")
    dataset_figure_count = 0
    cross_dataset_figure_count = 0
    if generate_figures:
        expected_paths: set[Path] = set()
        for (dataset, band), frame in participant_table.groupby(["dataset", "comparison_band_name"], sort=True):
            path = output_root / "figures" / "abba_ordinal" / f"{_safe_name(dataset)}__{_safe_name(band)}.png"
            _plot_band(frame, path, int(settings.get("figure_dpi", 200)))
            expected_paths.add(path)
            dataset_figure_count += 1
        figure_root = output_root / "figures" / "abba_ordinal"
        for dataset in tasks_by_dataset:
            prefix = f"{_safe_name(dataset)}__"
            for path in figure_root.glob(f"{prefix}*.png"):
                if path not in expected_paths:
                    path.unlink()
        configured_order = [
            str(dataset) for dataset in settings.get("datasets", [])
            if str(dataset) in tasks_by_dataset
        ]
        dataset_order = configured_order + [
            dataset for dataset in tasks_by_dataset if dataset not in configured_order
        ]
        cross_dataset_paths = _save_cross_dataset_planes(
            participant_table,
            dataset_order=dataset_order,
            output_root=output_root,
            dpi=int(settings.get("figure_dpi", 200)),
        )
        cross_dataset_figure_count = len(cross_dataset_paths)
    figure_count = dataset_figure_count + cross_dataset_figure_count
    manifest = {
        "analysis": "abba_band_full_and_concatenated_bout_ordinal_hcf",
        "band_definition": QC_BAND_DEFINITION if control_bands_qc else "group_specific_abba",
        "control_bands_qc": bool(control_bands_qc),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "config": settings,
        "concatenation_policy": CONCATENATION_POLICY,
        "datasets": sorted(tasks_by_dataset),
        "n_recordings": int(recording_table[["dataset", "recording_id"]].drop_duplicates().shape[0]),
        "n_participants": int(participant_table[["dataset", "participant_id"]].drop_duplicates().shape[0]),
        "resumed_recordings": int(resumed),
        "figures": int(figure_count),
        "dataset_band_figures": int(dataset_figure_count),
        "cross_dataset_plane_figures": int(cross_dataset_figure_count),
    }
    manifest_path = output_root / "abba_ordinal_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {
        "recordings": manifest["n_recordings"],
        "participants": manifest["n_participants"],
        "figures": figure_count,
        "cross_dataset_plane_figures": cross_dataset_figure_count,
        "resumed": resumed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/analyses/rhythmicity.json"))
    parser.add_argument("--datasets", nargs="*")
    parser.add_argument("--recordings", nargs="*")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-figures", action="store_true")
    parser.add_argument("--control-bands-qc", action="store_true", help="Apply each dataset's Control ABBA limits to every group and write isolated QC outputs")
    args = parser.parse_args()
    summary = run(
        args.config,
        datasets=args.datasets,
        recordings=args.recordings,
        workers=args.workers,
        overwrite=args.overwrite,
        generate_figures=not args.skip_figures,
        control_bands_qc=args.control_bands_qc,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
