"""Figures that mirror the primary visual outputs of the original dataset pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.runtime import configure_runtime

configure_runtime()

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

from core.plotting import save_figure


CONDITION_ORDER = ("HC", "PD_OFF", "PD_ON")
CONDITION_LABELS = {
    "HC": "Healthy control",
    "PD_OFF": "PD OFF",
    "PD_ON": "PD ON",
}
CONTRAST_ORDER = (
    "PD_OFF_minus_HC",
    "PD_ON_minus_HC",
    "PD_ON_minus_PD_OFF",
)
CONTRAST_LABELS = {
    "PD_OFF_minus_HC": "PD OFF − HC",
    "PD_ON_minus_HC": "PD ON − HC",
    "PD_ON_minus_PD_OFF": "PD ON − OFF",
}
MMSE_MODEL_ORDER = ("HC", "PD_OFF", "PD_ON", "PD_ON_minus_PD_OFF")
MMSE_MODEL_LABELS = {
    "HC": "HC",
    "PD_OFF": "PD OFF",
    "PD_ON": "PD ON",
    "PD_ON_minus_PD_OFF": "PD ON − OFF",
}


@dataclass(frozen=True)
class TopographicMetric:
    family: str
    metric: str
    label: str
    band_policy: str
    display_scale: float = 1.0

    @property
    def token(self) -> str:
        return f"{self.family}_{self.metric}"


# These are the primary spatial quantities shared with the original PSD,
# ordinal, scale-free, and bout pipelines.  Metrics unavailable in ds002778
# (for example bycycle waveform symmetry) are deliberately not fabricated.
TOPOGRAPHIC_METRICS = (
    TopographicMetric("psd", "relative_power", "Relative band power (%)", "all", 100.0),
    TopographicMetric("ordinal", "entropy", "Permutation entropy (H)", "broadband_all"),
    TopographicMetric("ordinal", "complexity", "Statistical complexity (C)", "broadband_all"),
    TopographicMetric("ordinal", "fisher_information", "Fisher information (F)", "broadband_all"),
    TopographicMetric("aperiodic", "aperiodic_offset", "Aperiodic offset", "broadband"),
    TopographicMetric("aperiodic", "aperiodic_exponent", "Aperiodic exponent", "broadband"),
    TopographicMetric("bouts", "oscillatory_occupancy", "Oscillatory occupancy", "ebosc"),
    TopographicMetric("bouts", "bouts_per_minute", "Bouts per minute", "ebosc"),
    TopographicMetric("bouts", "bout_duration_mean_s", "Mean bout duration (s)", "ebosc"),
    TopographicMetric("within_bout_ordinal", "entropy", "Within-bout permutation entropy (H)", "ebosc"),
    TopographicMetric("within_bout_ordinal", "complexity", "Within-bout statistical complexity (C)", "ebosc"),
    TopographicMetric("within_bout_ordinal", "fisher_information", "Within-bout Fisher information (F)", "ebosc"),
    TopographicMetric("periodic_peak", "peak_present", "Periodic-peak prevalence (%)", "specparam", 100.0),
    TopographicMetric("periodic_peak", "peak_power_linear", "Periodic peak power", "specparam"),
)


def _safe_name(value: str) -> str:
    return "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in value
    )


def _band_label(band: str, config: dict[str, Any]) -> str:
    if band == "broadband":
        return "Broadband"
    low, high = config["bands"][band]
    return f"{band.replace('_', ' ').title()}\n{low:g}–{high:g} Hz"


def _band_order(
    specification: TopographicMetric,
    present: set[str],
    config: dict[str, Any],
) -> list[str]:
    configured = [str(value) for value in config["bands"]]
    if specification.band_policy == "broadband":
        candidates = ["broadband"]
    elif specification.band_policy == "broadband_all":
        candidates = ["broadband", *configured]
    elif specification.band_policy == "ebosc":
        candidates = [str(value) for value in config["ebosc"]["bands"]]
    elif specification.band_policy == "specparam":
        minimum = float(config["specparam"]["frequency_range_hz"][0])
        candidates = [
            band
            for band in configured
            if float(config["bands"][band][0]) >= minimum
        ]
    else:
        candidates = configured
    return [band for band in candidates if band in present]


def _make_topomap_info(electrodes: list[str]) -> mne.Info:
    montage = mne.channels.make_standard_montage("biosemi32")
    missing = sorted(set(electrodes) - set(montage.ch_names))
    if missing:
        raise ValueError(f"No BioSemi32 positions are available for: {missing}")
    info = mne.create_info(electrodes, sfreq=250.0, ch_types="eeg")
    info.set_montage(montage)
    return info


def _finite_limits(values: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return 0.0, 1.0
    low, high = np.quantile(finite, [0.02, 0.98])
    low, high = float(low), float(high)
    if np.isclose(low, high):
        padding = max(abs(low) * 0.02, 1e-6)
    else:
        padding = 0.04 * (high - low)
    return low - padding, high + padding


def _symmetric_limit(values: np.ndarray, minimum: float = 0.25) -> float:
    finite = np.abs(np.asarray(values, dtype=float))
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return 1.0
    return max(minimum, float(np.quantile(finite, 0.98)))


def _draw_topomap(
    axis: Any,
    values: np.ndarray,
    info: mne.Info,
    *,
    cmap: str,
    vlim: tuple[float, float],
    significant: np.ndarray | None = None,
) -> Any | None:
    values = np.asarray(values, dtype=float)
    finite = np.flatnonzero(np.isfinite(values))
    if len(finite) < 4:
        axis.axis("off")
        axis.text(
            0.5,
            0.5,
            f"Insufficient finite electrodes\n({len(finite)})",
            ha="center",
            va="center",
        )
        return None
    selected_info = mne.pick_info(info, finite.tolist(), copy=True)
    mask = None
    if significant is not None:
        mask = np.asarray(significant, dtype=bool)[finite]
    image, _ = mne.viz.plot_topomap(
        values[finite],
        selected_info,
        axes=axis,
        show=False,
        sensors=True,
        contours=6 if significant is None else 0,
        cmap=cmap,
        vlim=vlim,
        mask=mask,
        mask_params={
            "marker": "o",
            "linestyle": "None",
            "markerfacecolor": "none",
            "markeredgecolor": "black",
            "linewidth": 1.0,
            "markersize": 6,
        },
    )
    return image


def _attach_recordings(
    table: pd.DataFrame,
    recordings: pd.DataFrame,
) -> pd.DataFrame:
    columns = ["recording_id", "participant_id", "condition", "mmse"]
    return table.merge(recordings[columns], on="recording_id", validate="many_to_one")


def _bootstrap_median(
    values: np.ndarray,
    *,
    n_resamples: int,
    confidence_level: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    center = np.median(values, axis=0)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(n_resamples, len(values)))
    resampled = np.median(values[indices], axis=1)
    tail = (1.0 - confidence_level) / 2.0
    lower, upper = np.quantile(resampled, [tail, 1.0 - tail], axis=0)
    return center, lower, upper


def plot_group_psd_curves(
    subject_psd: pd.DataFrame,
    recordings: pd.DataFrame,
    output_dir: str | Path,
    config: dict[str, Any],
) -> list[Path]:
    """Plot condition PSD confidence bands and the paired medication PSD change."""
    selected = subject_psd.loc[
        subject_psd["duration_variant"].eq("all_retained")
    ].copy()
    attached = _attach_recordings(selected, recordings)
    matrix = attached.pivot(
        index=["recording_id", "participant_id", "condition"],
        columns="frequency_hz",
        values="median_psd_uv2_hz",
    ).sort_index(axis=1)
    frequencies = matrix.columns.to_numpy(float)
    settings = config["statistics"]
    colors = config["plots"]["condition_colors"]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(10.5, 6.2), constrained_layout=True)
    for index, condition in enumerate(CONDITION_ORDER):
        values = matrix.loc[
            matrix.index.get_level_values("condition") == condition
        ].to_numpy(float)
        center, lower, upper = _bootstrap_median(
            values,
            n_resamples=int(settings["bootstrap_resamples"]),
            confidence_level=float(settings["confidence_level"]),
            seed=int(settings["random_seed"]) + index,
        )
        center_db = 10.0 * np.log10(np.maximum(center, np.finfo(float).tiny))
        lower_db = 10.0 * np.log10(np.maximum(lower, np.finfo(float).tiny))
        upper_db = 10.0 * np.log10(np.maximum(upper, np.finfo(float).tiny))
        axis.plot(
            frequencies,
            center_db,
            color=colors[condition],
            linewidth=2.0,
            label=f"{CONDITION_LABELS[condition]} (n={len(values)})",
        )
        axis.fill_between(
            frequencies,
            lower_db,
            upper_db,
            color=colors[condition],
            alpha=0.18,
        )
    axis.set(
        xlabel="Frequency (Hz)",
        ylabel="PSD (dB µV²/Hz)",
        title="Group median PSD with pointwise 95% bootstrap CIs",
        xlim=(float(frequencies.min()), float(frequencies.max())),
    )
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    group_path = output_dir / "group_median_psd_with_ci.png"
    save_figure(figure, group_path, int(config["plots"]["dpi"]))

    pd_matrix = matrix.loc[
        matrix.index.get_level_values("condition").isin(["PD_OFF", "PD_ON"])
    ].reset_index().pivot(
        index="participant_id", columns="condition", values=list(frequencies)
    )
    off = pd_matrix.xs("PD_OFF", axis=1, level="condition").to_numpy(float)
    on = pd_matrix.xs("PD_ON", axis=1, level="condition").to_numpy(float)
    differences_db = 10.0 * np.log10(
        np.maximum(on, np.finfo(float).tiny)
    ) - 10.0 * np.log10(np.maximum(off, np.finfo(float).tiny))
    center, lower, upper = _bootstrap_median(
        differences_db,
        n_resamples=int(settings["bootstrap_resamples"]),
        confidence_level=float(settings["confidence_level"]),
        seed=int(settings["random_seed"]) + 100,
    )
    figure, axis = plt.subplots(figsize=(10.5, 5.5), constrained_layout=True)
    axis.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
    axis.plot(frequencies, center, color="#6A3D9A", linewidth=2.0)
    axis.fill_between(frequencies, lower, upper, color="#6A3D9A", alpha=0.20)
    axis.set(
        xlabel="Frequency (Hz)",
        ylabel="PD ON − OFF PSD (dB)",
        title=f"Paired medication-state PSD change (n={len(differences_db)})",
        xlim=(float(frequencies.min()), float(frequencies.max())),
    )
    axis.grid(alpha=0.2)
    paired_path = output_dir / "paired_pd_on_minus_off_psd_change.png"
    save_figure(figure, paired_path, int(config["plots"]["dpi"]))
    return [group_path, paired_path]


def plot_relative_band_power_summary(
    subject_features: pd.DataFrame,
    recordings: pd.DataFrame,
    path: str | Path,
    config: dict[str, Any],
) -> Path:
    """Mirror the original combined relative-band-power distribution figure."""
    bands = [str(value) for value in config["bands"]]
    selected = subject_features.loc[
        subject_features["duration_variant"].eq("all_retained")
        & subject_features["family"].eq("psd")
        & subject_features["metric"].eq("relative_power")
        & subject_features["band"].isin(bands)
    ]
    attached = _attach_recordings(selected, recordings)
    n_columns = 3
    n_rows = int(np.ceil(len(bands) / n_columns))
    figure, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(4.4 * n_columns, 4.1 * n_rows),
        squeeze=False,
    )
    colors = config["plots"]["condition_colors"]
    rng = np.random.default_rng(int(config["statistics"]["random_seed"]))
    for axis, band in zip(axes.flat, bands):
        table = attached.loc[attached["band"].eq(band)]
        pd_table = table.loc[table["condition"].isin(["PD_OFF", "PD_ON"])]
        paired = pd_table.pivot(
            index="participant_id", columns="condition", values="value"
        ).dropna()
        for _, row in paired.iterrows():
            axis.plot(
                [2, 3],
                100.0 * np.asarray([row["PD_OFF"], row["PD_ON"]], dtype=float),
                color="0.65",
                alpha=0.35,
                linewidth=0.7,
                zorder=1,
            )
        for condition_index, condition in enumerate(CONDITION_ORDER, start=1):
            values = 100.0 * table.loc[
                table["condition"].eq(condition), "value"
            ].dropna().to_numpy(float)
            jitter = rng.uniform(-0.07, 0.07, size=len(values))
            axis.scatter(
                condition_index + jitter,
                values,
                color=colors[condition],
                edgecolor="white",
                linewidth=0.4,
                alpha=0.82,
                s=26,
                zorder=3,
            )
            if len(values):
                median = float(np.median(values))
                axis.plot(
                    [condition_index - 0.17, condition_index + 0.17],
                    [median, median],
                    color="black",
                    linewidth=1.8,
                    zorder=4,
                )
        low, high = config["bands"][band]
        axis.set_xticks([1, 2, 3], ["HC", "PD OFF", "PD ON"])
        axis.set(
            ylabel="Relative power (%)",
            title=f"{band.title()} ({low:g}–{high:g} Hz)",
        )
        axis.grid(axis="y", alpha=0.2)
    for axis in axes.flat[len(bands) :]:
        axis.set_visible(False)
    figure.suptitle("Subject-level relative band power with paired PD trajectories")
    figure.tight_layout()
    path = Path(path)
    save_figure(figure, path, int(config["plots"]["dpi"]))
    return path


def plot_group_mean_topomaps(
    electrode_features: pd.DataFrame,
    recordings: pd.DataFrame,
    output_dir: str | Path,
    config: dict[str, Any],
) -> list[Path]:
    """Plot HC, PD OFF, and PD ON group topographies for every shared metric."""
    primary = electrode_features.loc[
        electrode_features["duration_variant"].eq("all_retained")
    ]
    attached = _attach_recordings(primary, recordings)
    electrodes = list(dict.fromkeys(attached["electrode"].astype(str)))
    info = _make_topomap_info(electrodes)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for specification in TOPOGRAPHIC_METRICS:
        table = attached.loc[
            attached["family"].eq(specification.family)
            & attached["metric"].eq(specification.metric)
        ].copy()
        if table.empty:
            continue
        bands = _band_order(
            specification,
            set(table["band"].astype(str)),
            config,
        )
        if not bands:
            continue
        aggregation = "median" if specification.family == "psd" else "mean"
        grouped = (
            table.loc[table["band"].isin(bands)]
            .groupby(["condition", "band", "electrode"])["value"]
            .agg(aggregation)
            .mul(specification.display_scale)
        )
        figure, axes = plt.subplots(
            len(CONDITION_ORDER),
            len(bands),
            figsize=(3.45 * len(bands), 3.35 * len(CONDITION_ORDER)),
            squeeze=False,
        )
        for column, band in enumerate(bands):
            band_values = grouped.loc[(slice(None), band, slice(None))].to_numpy(float)
            limits = _finite_limits(band_values)
            image = None
            for row, condition in enumerate(CONDITION_ORDER):
                selected = grouped.loc[(condition, band)].reindex(electrodes)
                image = _draw_topomap(
                    axes[row, column],
                    selected.to_numpy(float),
                    info,
                    cmap="viridis",
                    vlim=limits,
                )
                if row == 0:
                    axes[row, column].set_title(_band_label(band, config))
                if column == 0:
                    axes[row, column].text(
                        -0.20,
                        0.5,
                        CONDITION_LABELS[condition],
                        transform=axes[row, column].transAxes,
                        rotation=90,
                        va="center",
                        ha="center",
                        fontweight="bold",
                    )
            if image is not None:
                figure.colorbar(image, ax=axes[:, column].tolist(), shrink=0.55)
        figure.suptitle(
            f"{specification.label}: condition {aggregation} topographies\n"
            "Color limits are shared across conditions within each band"
        )
        figure.subplots_adjust(top=0.89, hspace=0.28, wspace=0.34)
        path = output_dir / f"{_safe_name(specification.token)}_group_means.png"
        save_figure(figure, path, int(config["plots"]["dpi"]))
        paths.append(path)
    return paths


def _plot_inferential_topomaps(
    statistics: pd.DataFrame,
    *,
    panel_column: str,
    panel_order: tuple[str, ...],
    panel_labels: dict[str, str],
    output_dir: str | Path,
    config: dict[str, Any],
    title_suffix: str,
) -> list[Path]:
    selected_primary = statistics.loc[
        statistics["duration_variant"].eq("all_retained")
        & statistics["sensitivity_cohort"].eq("all_participants")
    ]
    electrodes = list(dict.fromkeys(selected_primary["electrode"].astype(str)))
    info = _make_topomap_info(electrodes)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for specification in TOPOGRAPHIC_METRICS:
        table = selected_primary.loc[
            selected_primary["family"].eq(specification.family)
            & selected_primary["metric"].eq(specification.metric)
            & selected_primary[panel_column].isin(panel_order)
        ].copy()
        if table.empty:
            continue
        bands = _band_order(
            specification,
            set(table["band"].astype(str)),
            config,
        )
        if not bands:
            continue
        effect_limit = _symmetric_limit(table["standardized_effect"].to_numpy(float))
        figure, axes = plt.subplots(
            len(panel_order),
            len(bands),
            figsize=(3.5 * len(bands), 3.35 * len(panel_order)),
            squeeze=False,
        )
        image = None
        for row, panel in enumerate(panel_order):
            for column, band in enumerate(bands):
                indexed = table.loc[
                    table[panel_column].eq(panel) & table["band"].eq(band)
                ].drop_duplicates("electrode").set_index("electrode")
                values = indexed["standardized_effect"].reindex(electrodes).to_numpy(float)
                significant = (
                    indexed["fdr_reject"].fillna(False).reindex(electrodes, fill_value=False).to_numpy(bool)
                )
                image = _draw_topomap(
                    axes[row, column],
                    values,
                    info,
                    cmap="RdBu_r",
                    vlim=(-effect_limit, effect_limit),
                    significant=significant,
                )
                if row == 0:
                    axes[row, column].set_title(_band_label(band, config))
                if column == 0:
                    axes[row, column].text(
                        -0.20,
                        0.5,
                        panel_labels[panel],
                        transform=axes[row, column].transAxes,
                        rotation=90,
                        va="center",
                        ha="center",
                        fontweight="bold",
                    )
        if image is not None:
            figure.colorbar(
                image,
                ax=axes.ravel().tolist(),
                shrink=0.48,
                label="Standardized effect",
            )
        figure.suptitle(
            f"{specification.label}: {title_suffix}\n"
            "Black rings mark electrodes surviving the configured BH-FDR correction"
        )
        figure.subplots_adjust(top=0.89, hspace=0.28, wspace=0.28)
        path = output_dir / f"{_safe_name(specification.token)}_{_safe_name(title_suffix)}.png"
        save_figure(figure, path, int(config["plots"]["dpi"]))
        paths.append(path)
    return paths


def plot_condition_contrast_topomaps(
    statistics: pd.DataFrame,
    output_dir: str | Path,
    config: dict[str, Any],
) -> list[Path]:
    """Plot adjusted between-condition and paired medication effect maps."""
    return _plot_inferential_topomaps(
        statistics,
        panel_column="contrast",
        panel_order=CONTRAST_ORDER,
        panel_labels=CONTRAST_LABELS,
        output_dir=output_dir,
        config=config,
        title_suffix="condition contrast effects",
    )


def plot_mmse_topomaps(
    statistics: pd.DataFrame,
    output_dir: str | Path,
    config: dict[str, Any],
) -> list[Path]:
    """Plot age/sex-adjusted MMSE association and medication-change maps."""
    return _plot_inferential_topomaps(
        statistics,
        panel_column="mmse_model",
        panel_order=MMSE_MODEL_ORDER,
        panel_labels=MMSE_MODEL_LABELS,
        output_dir=output_dir,
        config=config,
        title_suffix="MMSE partial Spearman maps",
    )


def plot_comparable_pipeline_figures(
    *,
    subject_features: pd.DataFrame,
    electrode_features: pd.DataFrame,
    subject_psd: pd.DataFrame,
    electrode_condition_statistics: pd.DataFrame,
    electrode_mmse_statistics: pd.DataFrame,
    recordings: pd.DataFrame,
    output_dir: str | Path,
    config: dict[str, Any],
) -> list[Path]:
    """Create the original-pipeline-comparable ds002778 visual battery."""
    output_dir = Path(output_dir)
    paths = plot_group_psd_curves(
        subject_psd,
        recordings,
        output_dir / "psd",
        config,
    )
    paths.append(
        plot_relative_band_power_summary(
            subject_features,
            recordings,
            output_dir / "psd" / "group_relative_band_power_violins.png",
            config,
        )
    )
    if not electrode_features.empty:
        paths.extend(
            plot_group_mean_topomaps(
                electrode_features,
                recordings,
                output_dir / "topomaps" / "group_means",
                config,
            )
        )
    if not electrode_condition_statistics.empty:
        paths.extend(
            plot_condition_contrast_topomaps(
                electrode_condition_statistics,
                output_dir / "topomaps" / "condition_contrasts",
                config,
            )
        )
    if not electrode_mmse_statistics.empty:
        paths.extend(
            plot_mmse_topomaps(
                electrode_mmse_statistics,
                output_dir / "topomaps" / "mmse",
                config,
            )
        )
    return paths
