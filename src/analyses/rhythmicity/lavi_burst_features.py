"""Optional LAVI-only frequency-domain burst summaries and violin plots.

This module deliberately does not read the global burst-feature tables.  It
uses the ABBA classification saved with each LAVI profile and treats a
contiguous run of high-rhythmicity frequency bins as a *frequency bout*.
These are not temporal bursts (LAVI profiles are frequency-domain summaries),
so widths are reported in Hz and rates as bouts/Hz rather than seconds or
minutes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from tqdm.auto import tqdm


GROUP_COLORS = {
    "Control": "#7f7f7f",
    "PD": "#d95f02",
    "PD_OFF": "#7570b3",
    "PD_ON": "#1b9e77",
}
DEFAULT_BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 50.0),
}

FEATURES = [
    "lavi_high_bout_count",
    "lavi_high_occupancy",
    "lavi_high_bout_density_hz",
    "lavi_high_bout_width_hz_mean",
    "lavi_high_bout_peak_excess_mean",
    "lavi_high_bout_peak_frequency_hz_mean",
]
FEATURE_LABELS = {
    "lavi_high_bout_count": "High-rhythmicity bout count",
    "lavi_high_occupancy": "High-rhythmicity occupancy",
    "lavi_high_bout_density_hz": "Bout density (per Hz)",
    "lavi_high_bout_width_hz_mean": "Mean bout width (Hz)",
    "lavi_high_bout_peak_excess_mean": "Mean peak excess (LAVI)",
    "lavi_high_bout_peak_frequency_hz_mean": "Mean bout peak frequency (Hz)",
}


def _significance_stars(q_value: float) -> str:
    if not np.isfinite(q_value) or q_value >= 0.05:
        return ""
    if q_value < 0.001:
        return "***"
    if q_value < 0.01:
        return "**"
    return "*"


def _ordered_bands(values: list[str]) -> list[str]:
    return [name for name in DEFAULT_BANDS if name in values] + [name for name in values if name not in DEFAULT_BANDS]


def _segments(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return inclusive contiguous true intervals."""
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return []
    changes = np.diff(np.r_[False, mask, False].astype(int))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1) - 1
    return list(zip(starts.tolist(), ends.tolist()))


def _frequency_edges(freq: np.ndarray) -> np.ndarray:
    if len(freq) == 1:
        half = max(abs(float(freq[0])) * 0.05, 0.05)
        return np.array([freq[0] - half, freq[0] + half], dtype=float)
    edges = np.empty(len(freq) + 1, dtype=float)
    edges[1:-1] = (freq[:-1] + freq[1:]) / 2.0
    edges[0] = freq[0] - (freq[1] - freq[0]) / 2.0
    edges[-1] = freq[-1] + (freq[-1] - freq[-2]) / 2.0
    return edges


def _summarize_band(profile: np.ndarray, sigvect: np.ndarray, freq: np.ndarray, band: tuple[float, float]) -> dict[str, float]:
    low, high = band
    selected = np.isfinite(profile) & np.isfinite(freq) & (freq >= low) & (freq <= high)
    if not selected.any():
        return {name: np.nan for name in FEATURES}
    values = np.asarray(profile[selected], dtype=float)
    status = np.asarray(sigvect[selected], dtype=float)
    frequencies = np.asarray(freq[selected], dtype=float)
    high_mask = status > 0
    bouts = _segments(high_mask)
    band_edges = _frequency_edges(frequencies)
    widths = np.asarray([band_edges[end + 1] - band_edges[start] for start, end in bouts], dtype=float)
    baseline = float(np.nanmedian(profile))
    peaks = np.asarray([np.nanmax(values[start : end + 1]) for start, end in bouts], dtype=float)
    peak_freqs = np.asarray([frequencies[start + int(np.nanargmax(values[start : end + 1]))] for start, end in bouts], dtype=float)
    count = float(len(bouts))
    span = max(float(band_edges[-1] - band_edges[0]), np.finfo(float).eps)
    return {
        "lavi_high_bout_count": count,
        "lavi_high_occupancy": float(np.mean(high_mask)),
        "lavi_high_bout_density_hz": count / span,
        "lavi_high_bout_width_hz_mean": float(np.nanmean(widths)) if len(widths) else np.nan,
        "lavi_high_bout_peak_excess_mean": float(np.nanmean(peaks - baseline)) if len(peaks) else np.nan,
        "lavi_high_bout_peak_frequency_hz_mean": float(np.nanmean(peak_freqs)) if len(peak_freqs) else np.nan,
    }


def _fdr_bh(values: pd.Series | np.ndarray) -> np.ndarray:
    p = np.asarray(values, dtype=float)
    q = np.full(p.shape, np.nan, dtype=float)
    valid = np.isfinite(p)
    if not valid.any():
        return q
    pv = p[valid]
    order = np.argsort(pv)
    ranked = pv[order] * len(pv) / np.arange(1, len(pv) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted = np.empty_like(ranked)
    adjusted[order] = np.minimum(ranked, 1.0)
    q[valid] = adjusted
    return q


def _load_features(output_root: Path, bands: dict[str, tuple[float, float]]) -> pd.DataFrame:
    raw_path = output_root / "metrics" / "electrode_rhythmicity.csv.gz"
    raw = pd.read_csv(raw_path, low_memory=False)
    rows: list[dict[str, Any]] = []
    recordings = raw[["dataset", "recording_id"]].drop_duplicates().to_dict("records")
    for recording in tqdm(recordings, desc="LAVI frequency bouts"):
        dataset, recording_id = recording["dataset"], recording["recording_id"]
        profile_path = output_root / "profiles" / str(dataset) / f"{recording_id}.npz"
        if not profile_path.is_file():
            continue
        metadata = raw.loc[(raw["dataset"].eq(dataset)) & (raw["recording_id"].eq(recording_id))]
        metadata = metadata.drop_duplicates("electrode").set_index("electrode")
        with np.load(profile_path, allow_pickle=False) as profile_file:
            lavi = np.asarray(profile_file["lavi"], dtype=float)
            foi = np.asarray(profile_file["foi"], dtype=float)
            channels = [str(channel) for channel in profile_file["channels"].tolist()]
            if "sigvect" in profile_file:
                sigvect = np.asarray(profile_file["sigvect"], dtype=float)
            else:
                # Older cached profiles predate sigvect being persisted.  The
                # default ABBA mode is deterministic, so reconstruct it here.
                from lavi import abba
                _, _, sigvect_list = abba(lavi, foi)
                sigvect = np.asarray(sigvect_list, dtype=float)
        for channel_index, channel in enumerate(channels):
            if channel not in metadata.index or channel_index >= lavi.shape[0]:
                continue
            meta = metadata.loc[channel]
            for band, limits in bands.items():
                summary = _summarize_band(lavi[channel_index], sigvect[channel_index], foi, limits)
                rows.append({
                    "dataset": dataset,
                    "recording_id": recording_id,
                    "participant_id": meta["participant_id"],
                    "group": meta["group"],
                    "electrode": channel,
                    "band": band,
                    **summary,
                })
    return pd.DataFrame(rows)


def _participant_table(electrode: pd.DataFrame) -> pd.DataFrame:
    if electrode.empty:
        return electrode
    keys = ["dataset", "participant_id", "group", "electrode", "band"]
    participant_electrode = electrode.groupby(keys, as_index=False)[FEATURES].mean(numeric_only=True)
    return participant_electrode.groupby(["dataset", "participant_id", "group", "band"], as_index=False)[FEATURES].mean(numeric_only=True)


def _group_statistics(participant: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (dataset, band), frame in participant.groupby(["dataset", "band"]):
        groups = list(dict.fromkeys(frame["group"]))
        for left_index, group_a in enumerate(groups):
            for group_b in groups[left_index + 1 :]:
                for feature in FEATURES:
                    left = pd.to_numeric(frame.loc[frame["group"].eq(group_a), feature], errors="coerce").dropna().to_numpy(float)
                    right = pd.to_numeric(frame.loc[frame["group"].eq(group_b), feature], errors="coerce").dropna().to_numpy(float)
                    if len(left) < 2 or len(right) < 2 or np.unique(np.r_[left, right]).size < 2:
                        t_value, p_value = np.nan, np.nan
                    else:
                        test = ttest_ind(left, right, equal_var=False, nan_policy="omit")
                        t_value, p_value = float(test.statistic), float(test.pvalue)
                    rows.append({"dataset": dataset, "band": band, "feature": feature, "group_a": group_a, "group_b": group_b, "n_a": len(left), "n_b": len(right), "mean_a": np.nanmean(left) if len(left) else np.nan, "mean_b": np.nanmean(right) if len(right) else np.nan, "t_value": t_value, "p_value": p_value})
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["q_fdr_bh"] = np.nan
    for dataset in result["dataset"].unique():
        mask = result["dataset"].eq(dataset)
        result.loc[mask, "q_fdr_bh"] = _fdr_bh(result.loc[mask, "p_value"])
    result["significant_fdr"] = result["q_fdr_bh"] < 0.05
    return result


def _save_violins(participant: pd.DataFrame, output: Path, bands: list[str], statistics: pd.DataFrame | None = None) -> None:
    if participant.empty:
        return
    datasets = list(dict.fromkeys(participant["dataset"]))
    n_rows, n_cols = len(datasets), len(FEATURES)
    rng = np.random.default_rng(42)
    for band in bands:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.25 * n_cols, 2.85 * n_rows), squeeze=False, sharex=True, sharey="col")
        for row_index, dataset in enumerate(datasets):
            frame = participant.loc[participant["dataset"].eq(dataset) & participant["band"].eq(band)]
            # Standard datasets are Control vs PD. Medication-state data are
            # shown with the independent Control group plus PD-OFF and PD-ON.
            if dataset == "medication_state":
                dataset_groups = [group for group in ("Control", "PD_OFF", "PD_ON") if group in set(frame["group"])]
            else:
                dataset_groups = [group for group in ("Control", "PD") if group in set(frame["group"])]
            for col_index, feature in enumerate(FEATURES):
                axis = axes[row_index, col_index]
                for group_index, group in enumerate(dataset_groups):
                    values = pd.to_numeric(frame.loc[frame["group"].eq(group), feature], errors="coerce").dropna().to_numpy(float)
                    if len(values) > 1 and np.ptp(values) > 0:
                        violin = axis.violinplot([values], positions=[group_index], widths=0.72, showmeans=False, showmedians=True, showextrema=False)
                        body = violin["bodies"][0]
                        body.set_facecolor(GROUP_COLORS.get(group, "#555555")); body.set_edgecolor("white"); body.set_linewidth(0.6); body.set_alpha(0.72)
                    if len(values):
                        jitter = rng.uniform(-0.16, 0.16, size=len(values))
                        axis.scatter(np.full(len(values), group_index) + jitter, values, s=14, color=GROUP_COLORS.get(group, "#555555"), edgecolor="white", linewidth=0.35, alpha=0.9, zorder=3)
                labels = [group.replace("PD_", "PD-") if group.startswith("PD_") else group for group in dataset_groups]
                axis.set_xticks(range(len(dataset_groups)), labels, rotation=25, ha="right", fontsize=7)
                axis.grid(axis="y", alpha=0.22)
                if row_index == 0:
                    axis.set_title(FEATURE_LABELS[feature], fontsize=9, fontweight="bold")
                if col_index == 0:
                    axis.set_ylabel(dataset, fontsize=9, fontweight="bold")
                if row_index == n_rows - 1:
                    axis.set_xlabel("Group", fontsize=8)
                if statistics is not None and len(dataset_groups) >= 2:
                    comparisons = statistics.loc[
                        statistics["dataset"].eq(dataset)
                        & statistics["band"].eq(band)
                        & statistics["feature"].eq(feature)
                        & statistics["group_a"].isin(dataset_groups)
                        & statistics["group_b"].isin(dataset_groups)
                    ].sort_values("q_fdr_bh")
                    group_positions = {group: index for index, group in enumerate(dataset_groups)}
                    bracket_index = 0
                    for _, comparison in comparisons.iterrows():
                        stars = _significance_stars(float(comparison["q_fdr_bh"]))
                        if stars:
                            # Use an x-data/axes-y transform so the annotation
                            # remains visible with the shared column scales.
                            left = group_positions.get(comparison["group_a"])
                            right = group_positions.get(comparison["group_b"])
                            if left is None or right is None:
                                continue
                            left, right = min(left, right), max(left, right)
                            y = 0.78 + 0.08 * bracket_index
                            axis.plot([left, left, right, right], [y - 0.025, y, y, y - 0.025], transform=axis.get_xaxis_transform(), color="#222222", linewidth=0.8, clip_on=False)
                            axis.text((left + right) / 2, y + 0.01, stars, transform=axis.get_xaxis_transform(), ha="center", va="bottom", fontsize=10, fontweight="bold", color="#111111")
                            bracket_index += 1
        legend_groups = [group for group in ("Control", "PD", "PD_OFF", "PD_ON") if group in set(participant["group"])]
        legend = [Patch(facecolor=GROUP_COLORS.get(group, "#555555"), edgecolor="white", label=group.replace("PD_", "PD-") if group.startswith("PD_") else group) for group in legend_groups]
        fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, 1.005), ncol=len(legend), frameon=False, fontsize=9)
        fig.suptitle(f"LAVI-only frequency-domain burst characteristics — {band.title()}", y=1.045, fontsize=15, fontweight="bold")
        fig.text(0.5, 1.018, "Each point is one participant; high-rhythmicity ABBA frequency bouts are summarized within this band", ha="center", fontsize=9, color="#444444")
        fig.subplots_adjust(left=0.06, right=0.995, bottom=0.09, top=0.91, wspace=0.20, hspace=0.40)
        fig.savefig(output / f"lavi_burst_quantity_violins_{band}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def run(output_root: Path, config_path: Path | None = None) -> dict[str, int]:
    config = json.loads(config_path.read_text()) if config_path and config_path.is_file() else {}
    bands = {str(name): (float(limits[0]), float(limits[1])) for name, limits in config.get("bands", DEFAULT_BANDS).items()}
    bands_ordered = _ordered_bands(list(bands))
    electrode = _load_features(output_root, {band: bands[band] for band in bands_ordered})
    participant = _participant_table(electrode)
    metrics_root = output_root / "metrics"; statistics_root = output_root / "statistics"; figures_root = output_root / "figures"
    metrics_root.mkdir(parents=True, exist_ok=True); statistics_root.mkdir(parents=True, exist_ok=True); figures_root.mkdir(parents=True, exist_ok=True)
    electrode.to_csv(metrics_root / "lavi_burst_features_electrode.csv.gz", index=False)
    participant.to_csv(metrics_root / "lavi_burst_features_participant.csv.gz", index=False)
    statistics = _group_statistics(participant)
    statistics.to_csv(statistics_root / "lavi_burst_feature_group_comparisons.csv", index=False)
    _save_violins(participant, figures_root, bands_ordered, statistics)
    return {"electrode_rows": len(electrode), "participant_rows": len(participant), "n_participants": int(participant["participant_id"].nunique()) if not participant.empty else 0}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/rhythmicity"))
    parser.add_argument("--config", type=Path, default=Path("config/analyses/rhythmicity.json"))
    args = parser.parse_args()
    summary = run(args.output_root, args.config)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
