"""Plot canonical bands against ABBA segmentation on participant LAVI profiles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import numpy as np
import pandas as pd

from lavi import abba


DEFAULT_BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 50.0),
}
BAND_COLORS = {"delta": "#4C78A8", "theta": "#72B7B2", "alpha": "#F2CF5B", "beta": "#F58518", "gamma": "#E45756"}
ABBA_COLORS = {"high": "#238b45", "low": "#756bb1"}


def _participant_profiles(output_root: Path) -> list[dict]:
    """Return one all-electrode, recording-averaged profile per participant."""
    raw = pd.read_csv(output_root / "metrics" / "electrode_rhythmicity.csv.gz", low_memory=False)
    recording_profiles: list[dict] = []
    for (dataset, recording_id), frame in raw.groupby(["dataset", "recording_id"], sort=False):
        profile_path = output_root / "profiles" / str(dataset) / f"{recording_id}.npz"
        if not profile_path.is_file():
            continue
        with np.load(profile_path, allow_pickle=False) as profile_file:
            profile = np.asarray(profile_file["lavi"], dtype=float)
            foi = np.asarray(profile_file["foi"], dtype=float)
        first = frame.iloc[0]
        recording_profiles.append({"dataset": dataset, "recording_id": recording_id, "participant_id": first["participant_id"], "group": first["group"], "profile": np.nanmean(profile, axis=0), "foi": foi})
    participants: list[dict] = []
    for dataset in dict.fromkeys(item["dataset"] for item in recording_profiles):
        dataset_items = [item for item in recording_profiles if item["dataset"] == dataset]
        # Keep medication ON and OFF recordings as separate participant-state
        # profiles. For the other datasets the group is constant per subject,
        # so this is equivalent to ordinary participant-level averaging.
        participant_states = dict.fromkeys((item["participant_id"], item["group"]) for item in dataset_items)
        for participant_id, group in participant_states:
            items = [item for item in dataset_items if item["participant_id"] == participant_id and item["group"] == group]
            participants.append({"dataset": dataset, "participant_id": participant_id, "group": group, "profile": np.nanmean(np.stack([item["profile"] for item in items]), axis=0), "foi": items[0]["foi"]})
    return participants


def _representative_profiles(output_root: Path) -> list[dict]:
    participants = _participant_profiles(output_root)
    representatives: list[dict] = []
    for dataset in dict.fromkeys(item["dataset"] for item in participants):
        participant_items = [item for item in participants if item["dataset"] == dataset]
        median_profile = np.nanmedian(np.stack([item["profile"] for item in participant_items]), axis=0)
        representative = min(participant_items, key=lambda item: float(np.nanmean((item["profile"] - median_profile) ** 2)))
        representatives.append(representative)
    return representatives


def _dataset_mean_profiles(participants: list[dict]) -> list[dict]:
    """Build one mean participant profile per dataset for the average limits."""
    means: list[dict] = []
    for dataset in dict.fromkeys(item["dataset"] for item in participants):
        items = [item for item in participants if item["dataset"] == dataset]
        means.append({
            "dataset": dataset,
            "participant_id": "dataset_mean_profile",
            "group": "all participants",
            "profile": np.nanmean(np.stack([item["profile"] for item in items]), axis=0),
            "foi": items[0]["foi"],
        })
    return means


def _group_mean_profiles(participants: list[dict]) -> list[dict]:
    """Build one mean participant profile per dataset and diagnostic group."""
    means: list[dict] = []
    for dataset in dict.fromkeys(item["dataset"] for item in participants):
        dataset_items = [item for item in participants if item["dataset"] == dataset]
        for group in dict.fromkeys(item["group"] for item in dataset_items):
            items = [item for item in dataset_items if item["group"] == group]
            means.append({
                "dataset": dataset,
                "participant_id": f"{group}_mean_profile",
                "group": group,
                "profile": np.nanmean(np.stack([item["profile"] for item in items]), axis=0),
                "foi": items[0]["foi"],
            })
    return means


def _segment_rows(profiles: list[dict], source: str, bands: dict[str, tuple[float, float]]) -> pd.DataFrame:
    """Run ABBA and return one row per frequency segment."""
    rows: list[dict] = []
    for item in profiles:
        foi = np.asarray(item["foi"], dtype=float)
        profile = np.asarray(item["profile"], dtype=float)
        borders, _, _ = abba(profile, foi)
        for segment_index, border in enumerate(borders[0]):
            start_hz, end_hz, peak_hz, peak_lavi, sign = float(border[3]), float(border[4]), float(border[5]), float(border[6]), float(border[8])
            if not np.isfinite(start_hz) or not np.isfinite(end_hz):
                continue
            rows.append({
                "dataset": item["dataset"],
                "participant_id": item["participant_id"],
                "group": item["group"],
                "segment_index": segment_index,
                "start_hz": start_hz,
                "end_hz": end_hz,
                "peak_hz": peak_hz,
                "peak_lavi": peak_lavi,
                "direction": "high" if sign > 0 else "low",
                "source": source,
            })
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    # ABBA produces a variable number of intervals. Assign each interval to
    # the canonical region with which it has the largest frequency overlap,
    # then number intervals from low to high within that region. This yields
    # labels such as delta_1/delta_2 without mixing separate ABBA intervals.
    def region(start: float, end: float) -> str:
        overlaps = {name: max(0.0, min(end, high) - max(start, low)) for name, (low, high) in bands.items()}
        best = max(overlaps, key=overlaps.get)
        if overlaps[best] <= 0.0:
            midpoint = np.sqrt(start * end)
            best = min(bands, key=lambda name: abs(np.sqrt(bands[name][0] * bands[name][1]) - midpoint))
        return str(best)

    result["canonical_region"] = [region(float(start), float(end)) for start, end in zip(result["start_hz"], result["end_hz"])]
    key_columns = ["dataset"]
    if source == "group_mean_profile":
        key_columns.append("group")
    elif source == "participant":
        key_columns.extend(["participant_id", "group"])
    for _, indices in result.groupby(key_columns, sort=False).groups.items():
        ordered = result.loc[indices].sort_values(["start_hz", "end_hz"])
        counters: dict[str, int] = {}
        for index in ordered.index:
            region_name = str(result.at[index, "canonical_region"])
            counters[region_name] = counters.get(region_name, 0) + 1
            result.at[index, "band_name"] = f"{region_name}_{counters[region_name]}"
    return result


def _save_figure(representatives: list[dict], bands: dict[str, tuple[float, float]], output: Path) -> pd.DataFrame:
    if not representatives:
        raise RuntimeError("No representative LAVI profiles were found")
    segment_rows: list[dict] = []
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True, sharey=True, squeeze=False)
    for axis, item in zip(axes.flat, representatives):
        foi = np.asarray(item["foi"], dtype=float)
        profile = np.asarray(item["profile"], dtype=float)
        for band, (low, high) in bands.items():
            if high < foi.min() or low > foi.max():
                continue
            visible_low, visible_high = max(low, float(foi.min())), min(high, float(foi.max()))
            axis.axvspan(visible_low, visible_high, color=BAND_COLORS.get(band, "#999999"), alpha=0.10, lw=0, zorder=0)
            axis.text(np.sqrt(visible_low * visible_high), 0.985, band.title(), transform=axis.get_xaxis_transform(), ha="center", va="top", fontsize=8, fontweight="bold", color="#555555")
        borders, _, _ = abba(profile, foi)
        borders = borders[0]
        axis.plot(foi, profile, color="#222222", linewidth=2.2, zorder=3)
        axis.axhline(np.nanmedian(profile), color="#777777", linestyle="--", linewidth=0.9, zorder=2)
        for segment_index, border in enumerate(borders):
            start_hz, end_hz, peak_hz, peak_lavi, sign = float(border[3]), float(border[4]), float(border[5]), float(border[6]), float(border[8])
            direction = "high" if sign > 0 else "low"
            if not np.isfinite(start_hz) or not np.isfinite(end_hz):
                continue
            axis.axvline(start_hz, color=ABBA_COLORS[direction], linewidth=0.55, alpha=0.75, zorder=1)
            axis.axvspan(start_hz, end_hz, color=ABBA_COLORS[direction], alpha=0.06, lw=0, zorder=1)
            # A compact strip marks the ABBA result without obscuring the LAVI curve.
            axis.add_patch(Rectangle((start_hz, 0.015), max(end_hz - start_hz, 0.015), 0.035, facecolor=ABBA_COLORS[direction], edgecolor="none", alpha=0.9, zorder=4))
            if np.isfinite(peak_hz) and np.isfinite(peak_lavi):
                axis.plot(peak_hz, peak_lavi, marker="o", markersize=4.5, markerfacecolor=ABBA_COLORS[direction], markeredgecolor="white", markeredgewidth=0.6, zorder=5)
            segment_rows.append({"dataset": item["dataset"], "participant_id": item["participant_id"], "group": item["group"], "segment_index": segment_index, "start_hz": start_hz, "end_hz": end_hz, "peak_hz": peak_hz, "peak_lavi": peak_lavi, "direction": direction})
        axis.set_xscale("log"); axis.set_xlim(float(foi.min()), float(foi.max())); axis.set_ylim(0, 1); axis.grid(alpha=0.22, which="both")
        axis.set_title(f"{item['dataset']} — representative {item['participant_id']} ({item['group']})", fontsize=11, fontweight="bold")
        axis.set_xlabel("Frequency (Hz)"); axis.set_ylabel("LAVI")
    legend = [Line2D([], [], color="#222222", linewidth=2.2, label="Representative all-electrode LAVI"), Line2D([], [], color="#777777", linestyle="--", linewidth=0.9, label="ABBA median baseline"), Patch(facecolor=ABBA_COLORS["high"], label="ABBA high-rhythmicity segment"), Patch(facecolor=ABBA_COLORS["low"], label="ABBA low-rhythmicity segment")]
    fig.suptitle("Canonical frequency bands versus ABBA segmentation", y=0.99, fontsize=16, fontweight="bold")
    fig.text(0.5, 0.945, "Shaded labels = canonical bands; coloured strips/markers = ABBA segments and their peaks", ha="center", fontsize=9, color="#444444")
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, 0.90), ncol=2, frameon=False, fontsize=9)
    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.08, top=0.80, wspace=0.16, hspace=0.32)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return pd.DataFrame(segment_rows)


def _save_limits_figure(
    participant_segments: pd.DataFrame,
    mean_segments: pd.DataFrame,
    bands: dict[str, tuple[float, float]],
    output: Path,
    participant_counts: dict[str, int] | None = None,
) -> None:
    """Render all participant intervals transparently and mean-profile limits opaquely."""
    datasets = list(dict.fromkeys(mean_segments["dataset"]))
    fig, axes = plt.subplots(1, len(datasets), figsize=(4.6 * len(datasets), 5.5), sharey=True, squeeze=False)
    axes = axes.flat
    # Keep all panels on the same scale and retain the complete canonical
    # definitions, even when ABBA starts above the lowest canonical edge.
    shared_x_min = min(float(low) for low, _ in bands.values())
    shared_x_max = max(float(high) for _, high in bands.values())
    for axis, dataset in zip(axes, datasets):
        participant_subset = participant_segments.loc[participant_segments["dataset"].eq(dataset)]
        mean_subset = mean_segments.loc[mean_segments["dataset"].eq(dataset)]
        # Canonical strip (upper) and labels.
        for band, (low, high) in bands.items():
            visible_low, visible_high = max(low, shared_x_min), min(high, shared_x_max)
            if visible_high <= visible_low:
                continue
            axis.add_patch(Rectangle((visible_low, 0.76), visible_high - visible_low, 0.14, facecolor=BAND_COLORS.get(band, "#999999"), edgecolor="white", linewidth=0.8, alpha=0.65))
            axis.text(
                np.sqrt(visible_low * visible_high),
                0.83,
                f"{band.title()}\n{visible_low:g}–{visible_high:g} Hz",
                ha="center",
                va="center",
                fontsize=7.5,
                fontweight="bold",
                color="#333333",
                linespacing=1.05,
            )
        # Every participant's ABBA interval is drawn with low opacity. This
        # creates a density-like view of how often each frequency is assigned
        # to a high- or low-rhythmicity segment.
        for _, row in participant_subset.iterrows():
            start, end = float(row["start_hz"]), float(row["end_hz"])
            color = ABBA_COLORS.get(str(row["direction"]), "#777777")
            axis.add_patch(Rectangle((start, 0.46), max(end - start, 0.015), 0.18, facecolor=color, edgecolor="none", alpha=0.035, zorder=1))
        # ABBA run on the dataset mean participant profile is the opaque
        # summary limit. It is not a mean of segment indices (which vary in
        # number between participants).
        for _, row in mean_subset.iterrows():
            start, end = float(row["start_hz"]), float(row["end_hz"])
            color = ABBA_COLORS.get(str(row["direction"]), "#777777")
            axis.add_patch(Rectangle((start, 0.22), max(end - start, 0.015), 0.18, facecolor=color, edgecolor="white", linewidth=0.7, alpha=0.95, zorder=3))
            axis.axvline(start, color=color, linewidth=0.8, alpha=0.95, zorder=2)
            axis.text(start, 0.17, f"{start:g}", rotation=90, ha="right", va="top", fontsize=6.5, color=color)
        if not mean_subset.empty:
            last_end = float(mean_subset.iloc[-1]["end_hz"])
            last_color = ABBA_COLORS.get(str(mean_subset.iloc[-1]["direction"]), "#777777")
            axis.axvline(last_end, color=last_color, linewidth=0.8, alpha=0.95, zorder=2)
            axis.text(last_end, 0.17, f"{last_end:g}", rotation=90, ha="right", va="top", fontsize=6.5, color=last_color)
        # In medication-state the same participant contributes ON and OFF
        # profiles, so count participant-state pairs rather than IDs alone.
        n_subjects = participant_subset[["participant_id", "group"]].drop_duplicates().shape[0]
        n_profiles = (participant_counts or {}).get(dataset, n_subjects)
        count_label = f"n = {n_profiles} profiles" if n_profiles == n_subjects else f"n = {n_profiles} profiles\nABBA valid: {n_subjects}"
        axis.text(0.98, 0.97, count_label, transform=axis.transAxes, ha="right", va="top", fontsize=8, color="#555555")
        axis.set_xscale("log"); axis.set_xlim(shared_x_min, shared_x_max); axis.set_ylim(0.08, 1.0)
        axis.set_yticks([0.83, 0.55, 0.31], ["Canonical bands", "Individual ABBA\n(all subjects)", "ABBA on dataset\nmean profile"])
        axis.grid(axis="x", alpha=0.22, which="both")
        axis.set_xlabel("Frequency (Hz)")
        axis.set_title(dataset, fontsize=11, fontweight="bold")
    legend = [
        Patch(facecolor=ABBA_COLORS["high"], alpha=0.10, label="Participant high intervals (transparent)"),
        Patch(facecolor=ABBA_COLORS["low"], alpha=0.10, label="Participant low intervals (transparent)"),
        Patch(facecolor=ABBA_COLORS["high"], alpha=0.95, label="Mean-profile high intervals (opaque)"),
        Patch(facecolor=ABBA_COLORS["low"], alpha=0.95, label="Mean-profile low intervals (opaque)"),
    ]
    fig.suptitle("ABBA band limits across participants", y=0.985, fontsize=15, fontweight="bold")
    fig.text(0.5, 0.925, "Transparent intervals show participant-level variability; opaque intervals are ABBA applied to the mean participant profile", ha="center", fontsize=9, color="#444444")
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, 0.865), ncol=2, frameon=False, fontsize=8.5)
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.18, top=0.72, wspace=0.18)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _save_group_limits_figure(
    participant_segments: pd.DataFrame,
    group_mean_segments: pd.DataFrame,
    bands: dict[str, tuple[float, float]],
    output: Path,
    participant_counts: dict[tuple[str, str], int],
) -> None:
    """Compare ABBA limits between diagnostic groups within each dataset."""
    datasets = list(dict.fromkeys(group_mean_segments["dataset"]))
    fig, axes = plt.subplots(1, len(datasets), figsize=(4.8 * len(datasets), 6.0), sharey=True, squeeze=False)
    axes = axes.flat
    shared_x_min = min(float(low) for low, _ in bands.values())
    shared_x_max = max(float(high) for _, high in bands.values())
    group_order = {"Control": 0, "PD": 1, "PD_OFF": 1, "PD_ON": 2}
    for axis, dataset in zip(axes, datasets):
        groups = sorted(
            group_mean_segments.loc[group_mean_segments["dataset"].eq(dataset), "group"].dropna().unique(),
            key=lambda group: (group_order.get(str(group), 99), str(group)),
        )
        n_groups = len(groups)
        # The top strip is reserved for canonical definitions. Remaining rows
        # are diagnostic groups, with the same vertical positions in every
        # panel so Control/PD/ON/OFF can be compared directly.
        group_positions = np.linspace(0.56, 0.24, n_groups) if n_groups > 1 else np.array([0.40])
        for band, (low, high) in bands.items():
            axis.add_patch(Rectangle((low, 0.78), high - low, 0.14, facecolor=BAND_COLORS.get(band, "#999999"), edgecolor="white", linewidth=0.8, alpha=0.65))
            axis.text(np.sqrt(low * high), 0.85, f"{band.title()}\n{low:g}–{high:g} Hz", ha="center", va="center", fontsize=7.5, fontweight="bold", color="#333333", linespacing=1.05)
        for group, y in zip(groups, group_positions):
            subject_subset = participant_segments.loc[(participant_segments["dataset"].eq(dataset)) & (participant_segments["group"].eq(group))]
            mean_subset = group_mean_segments.loc[(group_mean_segments["dataset"].eq(dataset)) & (group_mean_segments["group"].eq(group))]
            # Subject-level intervals form a translucent distribution behind
            # the group-average segmentation.
            for _, row in subject_subset.iterrows():
                start, end = float(row["start_hz"]), float(row["end_hz"])
                color = ABBA_COLORS.get(str(row["direction"]), "#777777")
                axis.add_patch(Rectangle((start, y - 0.065), max(end - start, 0.015), 0.13, facecolor=color, edgecolor="none", alpha=0.035, zorder=1))
            for _, row in mean_subset.iterrows():
                start, end = float(row["start_hz"]), float(row["end_hz"])
                color = ABBA_COLORS.get(str(row["direction"]), "#777777")
                axis.add_patch(Rectangle((start, y - 0.065), max(end - start, 0.015), 0.13, facecolor=color, edgecolor="white", linewidth=0.7, alpha=0.95, zorder=3))
                # Keep the interval identifier visible in the same figure
                # used to define the burst-analysis labels.  Geometric
                # centering is appropriate because the frequency axis is log.
                if np.isfinite(start) and np.isfinite(end) and end > start:
                    axis.text(np.sqrt(start * end), y + 0.075, str(row.get("band_name", "")), ha="center", va="bottom", fontsize=6.5, fontweight="bold", color=color, rotation=90, clip_on=True, zorder=4)
                axis.axvline(start, color=color, linewidth=0.7, alpha=0.9, zorder=2)
                axis.text(start, y - 0.10, f"{start:g}", rotation=90, ha="right", va="top", fontsize=6.2, color=color)
            if not mean_subset.empty:
                last_end = float(mean_subset.iloc[-1]["end_hz"])
                last_color = ABBA_COLORS.get(str(mean_subset.iloc[-1]["direction"]), "#777777")
                axis.axvline(last_end, color=last_color, linewidth=0.7, alpha=0.9, zorder=2)
                axis.text(last_end, y - 0.10, f"{last_end:g}", rotation=90, ha="right", va="top", fontsize=6.2, color=last_color)
            n_profiles = participant_counts.get((dataset, str(group)), 0)
            n_valid = subject_subset["participant_id"].nunique()
            label = f"{group} (n={n_profiles})" if n_valid == n_profiles else f"{group} (n={n_profiles}; valid={n_valid})"
            axis.text(0.01, y, label, transform=axis.get_yaxis_transform(), ha="left", va="center", fontsize=8.5, fontweight="bold", color="#333333")
        axis.set_xscale("log"); axis.set_xlim(shared_x_min, shared_x_max); axis.set_ylim(0.08, 1.0)
        axis.set_yticks([])
        axis.grid(axis="x", alpha=0.22, which="both")
        axis.set_xlabel("Frequency (Hz)")
        axis.set_title(dataset, fontsize=11, fontweight="bold")
    legend = [
        Patch(facecolor=ABBA_COLORS["high"], alpha=0.10, label="Participant high intervals (transparent)"),
        Patch(facecolor=ABBA_COLORS["low"], alpha=0.10, label="Participant low intervals (transparent)"),
        Patch(facecolor=ABBA_COLORS["high"], alpha=0.95, label="Group-mean high intervals (opaque)"),
        Patch(facecolor=ABBA_COLORS["low"], alpha=0.95, label="Group-mean low intervals (opaque)"),
    ]
    fig.suptitle("ABBA band limits by diagnostic group", y=0.985, fontsize=15, fontweight="bold")
    fig.text(0.5, 0.925, "Control versus PD in standard datasets; Control, PD-OFF, and PD-ON in medication-state", ha="center", fontsize=9, color="#444444")
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, 0.865), ncol=2, frameon=False, fontsize=8.5)
    fig.subplots_adjust(left=0.15, right=0.99, bottom=0.18, top=0.72, wspace=0.18)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run(output_root: Path, config_path: Path | None = None) -> dict[str, int]:
    config = json.loads(config_path.read_text()) if config_path and config_path.is_file() else {}
    bands = {str(name): (float(limits[0]), float(limits[1])) for name, limits in config.get("bands", DEFAULT_BANDS).items()}
    participants = _participant_profiles(output_root)
    representatives = _representative_profiles(output_root)
    figures_root, statistics_root = output_root / "figures", output_root / "statistics"
    figures_root.mkdir(parents=True, exist_ok=True); statistics_root.mkdir(parents=True, exist_ok=True)
    segments = _save_figure(representatives, bands, figures_root / "abba_band_segmentation_comparison.png")
    participant_segments = _segment_rows(participants, source="participant", bands=bands)
    mean_segments = _segment_rows(_dataset_mean_profiles(participants), source="dataset_mean_profile", bands=bands)
    group_mean_segments = _segment_rows(_group_mean_profiles(participants), source="group_mean_profile", bands=bands)
    participant_counts = {dataset: sum(item["dataset"] == dataset for item in participants) for dataset in dict.fromkeys(item["dataset"] for item in participants)}
    _save_limits_figure(participant_segments, mean_segments, bands, figures_root / "abba_band_limits_all_subjects.png", participant_counts)
    # Keep the original filename as a convenient, backwards-compatible alias
    # for downstream notebooks that already reference it.
    _save_limits_figure(participant_segments, mean_segments, bands, figures_root / "abba_band_limits_by_dataset.png", participant_counts)
    segments.to_csv(statistics_root / "abba_representative_segments.csv", index=False)
    participant_segments.to_csv(statistics_root / "abba_subject_segments.csv", index=False)
    mean_segments.to_csv(statistics_root / "abba_dataset_mean_segments.csv", index=False)
    group_counts = {(dataset, str(group)): sum(item["dataset"] == dataset and str(item["group"]) == str(group) for item in participants) for dataset in dict.fromkeys(item["dataset"] for item in participants) for group in dict.fromkeys(item["group"] for item in participants if item["dataset"] == dataset)}
    _save_group_limits_figure(participant_segments, group_mean_segments, bands, figures_root / "abba_band_limits_by_group.png", group_counts)
    group_mean_segments.to_csv(statistics_root / "abba_group_mean_segments.csv", index=False)
    return {"datasets": len(representatives), "participants": len(participants), "segments": len(segments), "subject_segments": len(participant_segments), "group_mean_segments": len(group_mean_segments)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/rhythmicity"))
    parser.add_argument("--config", type=Path, default=Path("config/analyses/rhythmicity.json"))
    args = parser.parse_args()
    print(json.dumps(run(args.output_root, args.config), indent=2))


if __name__ == "__main__":
    main()
