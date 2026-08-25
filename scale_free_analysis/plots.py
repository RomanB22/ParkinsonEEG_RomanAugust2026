"""Figures for spectral decomposition, bouts, cycles, and group comparisons."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.runtime import configure_runtime

configure_runtime()

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd


def _save(fig, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _plot_topomap(
    axis: Any,
    values: np.ndarray,
    info: Any,
    *,
    cmap: str,
    vlim: tuple[float, float],
) -> Any | None:
    """Plot finite electrodes or label a panel with insufficient coverage."""
    finite_indices = np.flatnonzero(np.isfinite(values))
    if len(finite_indices) < 4:
        axis.set_axis_off()
        axis.text(
            0.5,
            0.5,
            f"Insufficient finite data\n({len(finite_indices)} electrodes)",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
        return None
    finite_info = mne.pick_info(info, finite_indices.tolist(), copy=True)
    image, _ = mne.viz.plot_topomap(
        values[finite_indices],
        finite_info,
        axes=axis,
        show=False,
        sensors=True,
        contours=6,
        cmap=cmap,
        vlim=vlim,
    )
    return image


def plot_spectral_example(example: dict[str, Any], path: Path, dpi: int) -> None:
    """Plot components and signed residuals for one electrode's specparam fit."""
    frequencies = example["frequencies_hz"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].loglog(
        frequencies,
        example["observed_psd_uv2_hz"],
        color="black",
        linewidth=1.2,
        label="Observed PSD",
    )
    axes[0].loglog(
        frequencies,
        example["modeled_psd_uv2_hz"],
        color="#0072B2",
        linewidth=2,
        label="Full specparam model",
    )
    axes[0].loglog(
        frequencies,
        example["aperiodic_psd_uv2_hz"],
        color="#D55E00",
        linewidth=2,
        label="Aperiodic component",
    )
    axes[0].set(xlabel="Frequency (Hz)", ylabel="PSD (µV²/Hz)", title="Spectral decomposition")
    axes[0].grid(alpha=0.2)
    axes[0].legend(frameon=False)

    axes[1].plot(
        frequencies,
        np.log10(example["observed_psd_uv2_hz"]),
        color="black",
        linewidth=1.2,
        label="Observed log power",
    )
    axes[1].plot(
        frequencies,
        np.log10(example["aperiodic_psd_uv2_hz"]),
        color="#D55E00",
        linewidth=2,
        label="Aperiodic log power",
    )
    axes[1].fill_between(
        frequencies,
        np.log10(example["aperiodic_psd_uv2_hz"]),
        np.log10(example["modeled_psd_uv2_hz"]),
        color="#009E73",
        alpha=0.3,
        label="Periodic component",
    )
    axes[1].set(
        xlabel="Frequency (Hz)",
        ylabel="log₁₀ PSD",
        title="Fitted periodic component (not the residual)",
    )
    axes[1].grid(alpha=0.2)
    axes[1].legend(frameon=False)

    observed_log = np.log10(example["observed_psd_uv2_hz"])
    modeled_log = np.log10(example["modeled_psd_uv2_hz"])
    residual = observed_log - modeled_log
    axes[2].axhline(0.0, color="black", linewidth=0.8)
    axes[2].plot(frequencies, residual, color="0.25", linewidth=1.2)
    axes[2].fill_between(
        frequencies,
        0.0,
        residual,
        where=residual >= 0.0,
        color="#009E73",
        alpha=0.35,
        label="Observed above model",
    )
    axes[2].fill_between(
        frequencies,
        0.0,
        residual,
        where=residual < 0.0,
        color="#CC79A7",
        alpha=0.35,
        label="Observed below model",
    )
    axes[2].set(
        xlabel="Frequency (Hz)",
        ylabel="Observed − full model (log₁₀ PSD)",
        title="Signed model residual",
    )
    axes[2].grid(alpha=0.2)
    axes[2].legend(frameon=False, fontsize=8)
    title = f"{example['subject_id']} — {example['electrode']}"
    if example.get("group"):
        title += f" — {example['group']}"
    details = []
    if np.isfinite(float(example.get("aperiodic_exponent", np.nan))):
        details.append(f"exponent={float(example['aperiodic_exponent']):.3f}")
    if np.isfinite(float(example.get("specparam_r_squared", np.nan))):
        details.append(f"R²={float(example['specparam_r_squared']):.3f}")
    if np.isfinite(float(example.get("specparam_error_mae", np.nan))):
        details.append(f"MAE={float(example['specparam_error_mae']):.3f}")
    qc_pass = example.get("specparam_fit_qc_pass")
    if isinstance(qc_pass, (bool, np.bool_)):
        qc_label = "QC PASS" if bool(qc_pass) else "QC FAIL"
        reasons = str(example.get("specparam_fit_qc_reasons", ""))
        details.append(qc_label if bool(qc_pass) else f"{qc_label}: {reasons}")
    if details:
        title += "\n" + ", ".join(details)
    fig.suptitle(title)
    fig.tight_layout()
    _save(fig, path, dpi)


def plot_bout_example(example: dict[str, Any], path: Path, dpi: int) -> None:
    """Plot one epoch, its time-frequency power, and detected bout mask."""
    signal = np.asarray(example["signal_uv"], dtype=float)
    times = np.arange(len(signal)) / float(example["sfreq"])
    mask = np.asarray(example["band_mask"], dtype=bool)
    frequencies = np.asarray(example["wavelet_frequencies_hz"], dtype=float)
    power = np.asarray(example["wavelet_power"], dtype=float)
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axes[0].plot(times, signal, color="0.25", linewidth=1.0, label="Cleaned EEG")
    axes[0].plot(
        times,
        np.where(mask, signal, np.nan),
        color="#D55E00",
        linewidth=1.8,
        label=f"Detected {example['band']} bout",
    )
    axes[0].set(ylabel="Amplitude (µV)", title="eBOSC-detected oscillatory bouts")
    axes[0].legend(frameon=False)
    image = axes[1].pcolormesh(
        times,
        frequencies,
        10.0 * np.log10(np.maximum(power, np.finfo(float).tiny)),
        shading="auto",
        cmap="magma",
    )
    axes[1].contour(
        times,
        frequencies,
        np.broadcast_to(mask, power.shape),
        levels=[0.5],
        colors=["cyan"],
        linewidths=1.0,
    )
    axes[1].set(xlabel="Time within accepted epoch (s)", ylabel="Frequency (Hz)", title="Wavelet power and band-bout outline")
    fig.colorbar(image, ax=axes[1], label="Wavelet power (dB a.u.)")
    fig.suptitle(f"{example['subject_id']} — {example['electrode']} — {example['band']}")
    fig.tight_layout()
    _save(fig, path, dpi)


def plot_cycle_example(example: dict[str, Any], path: Path, dpi: int) -> None:
    """Plot bycycle extrema for cycles retained within detected bouts."""
    signal = np.asarray(example["signal_uv"], dtype=float)
    sfreq = float(example["sfreq"])
    times = np.arange(len(signal)) / sfreq
    cycles = example["cycles"]
    fig, axis = plt.subplots(figsize=(12, 4.5))
    axis.plot(times, signal, color="0.25", linewidth=1.0)
    if len(cycles):
        peaks = cycles["sample_peak"].to_numpy(dtype=int)
        troughs = cycles["sample_last_trough"].to_numpy(dtype=int)
        axis.scatter(times[peaks], signal[peaks], color="#D55E00", s=28, label="Peaks")
        axis.scatter(times[troughs], signal[troughs], color="#0072B2", s=28, label="Troughs")
        for _, row in cycles.iterrows():
            start = int(row["sample_last_trough"])
            stop = int(row["sample_next_trough"])
            axis.axvspan(times[start], times[min(stop, len(times) - 1)], color="#009E73", alpha=0.10)
    axis.set(
        xlabel="Time within accepted epoch (s)",
        ylabel="Amplitude (µV)",
        title="bycycle waveform landmarks inside eBOSC bouts",
    )
    axis.legend(frameon=False)
    fig.suptitle(f"{example['subject_id']} — {example['electrode']} — {example['band']}")
    fig.tight_layout()
    _save(fig, path, dpi)


def plot_group_distributions(
    subject_band_table: pd.DataFrame,
    group_order: list[str],
    colors: dict[str, str],
    band_order: list[str],
    band_labels: dict[str, str],
    output_dir: Path,
    dpi: int,
) -> None:
    """Create PD/Control subject-level distributions for major band metrics."""
    metrics = {
        "oscillatory_occupancy": "Oscillatory occupancy",
        "bouts_per_minute": "Bouts per minute",
        "bout_duration_mean_s": "Mean bout duration (s)",
        "cycle_amplitude_mean_uv": "Mean cycle amplitude (µV)",
        "cycle_frequency_mean_hz": "Mean cycle frequency (Hz)",
        "peak_power_log10": "Periodic peak power (log₁₀ above 1/f)",
    }
    positions = np.arange(len(band_order), dtype=float)
    offsets = np.linspace(-0.18, 0.18, len(group_order))
    for metric, label in metrics.items():
        fig, axis = plt.subplots(figsize=(10, 5.5))
        for group, offset in zip(group_order, offsets):
            for band_index, band in enumerate(band_order):
                values = subject_band_table.loc[
                    subject_band_table["group"].eq(group)
                    & subject_band_table["band"].eq(band),
                    metric,
                ].dropna().to_numpy(dtype=float)
                if len(values):
                    jitter = np.linspace(-0.05, 0.05, len(values))
                    axis.scatter(
                        np.full(len(values), positions[band_index] + offset) + jitter,
                        np.sort(values),
                        s=12,
                        alpha=0.35,
                        color=colors[group],
                    )
                    axis.boxplot(
                        [values],
                        positions=[positions[band_index] + offset],
                        widths=0.28,
                        patch_artist=True,
                        boxprops={"facecolor": colors[group], "alpha": 0.35},
                        medianprops={"color": "black"},
                        whiskerprops={"color": colors[group]},
                        capprops={"color": colors[group]},
                        flierprops={"markersize": 0},
                    )
            axis.scatter([], [], color=colors[group], label=group)
        axis.set_xticks(positions, [band_labels[band] for band in band_order])
        axis.set(ylabel=label, title=f"Subject-level {label.lower()} by group")
        axis.grid(axis="y", alpha=0.2)
        axis.legend(frameon=False)
        fig.tight_layout()
        _save(fig, output_dir / f"group_{metric}.png", dpi)


def plot_aperiodic_topomaps(
    electrode_table: pd.DataFrame,
    info: Any,
    group_order: list[str],
    path: Path,
    dpi: int,
) -> None:
    """Plot group mean aperiodic offset and exponent."""
    metrics = (
        ("aperiodic_offset", "Aperiodic offset"),
        ("aperiodic_exponent", "Aperiodic exponent"),
    )
    fig, axes = plt.subplots(
        len(group_order),
        2,
        figsize=(8, 3.8 * len(group_order)),
        squeeze=False,
    )
    for column, (metric, label) in enumerate(metrics):
        grouped = electrode_table.groupby(["group", "electrode"])[metric].mean()
        all_values = grouped.to_numpy(dtype=float)
        low, high = float(np.nanmin(all_values)), float(np.nanmax(all_values))
        if np.isclose(low, high):
            high = low + max(abs(low) * 0.01, 1e-6)
        for row, group in enumerate(group_order):
            values = grouped.loc[group].reindex(info.ch_names).to_numpy(dtype=float)
            image = _plot_topomap(
                axes[row, column],
                values,
                info,
                cmap="viridis",
                vlim=(low, high),
            )
            axes[row, column].set_title(label)
            if image is not None:
                fig.colorbar(image, ax=axes[row, column], shrink=0.7)
            if column == 0:
                axes[row, column].text(
                    -0.22,
                    0.5,
                    group,
                    transform=axes[row, column].transAxes,
                    rotation=90,
                    va="center",
                    ha="center",
                    fontweight="bold",
                )
    fig.suptitle("Group mean specparam aperiodic topographies")
    fig.tight_layout()
    _save(fig, path, dpi)


def plot_band_topomaps(
    electrode_band_table: pd.DataFrame,
    info: Any,
    group_order: list[str],
    band_order: list[str],
    band_labels: dict[str, str],
    output_dir: Path,
    dpi: int,
) -> None:
    """Plot major band-resolved metrics as group mean scalp maps."""
    metrics = {
        "oscillatory_occupancy": "Oscillatory occupancy",
        "bout_duration_mean_s": "Mean bout duration (s)",
        "cycle_amplitude_mean_uv": "Mean cycle amplitude (µV)",
    }
    for metric, label in metrics.items():
        fig, axes = plt.subplots(
            len(group_order),
            len(band_order),
            figsize=(3.5 * len(band_order), 3.7 * len(group_order)),
            squeeze=False,
        )
        finite = electrode_band_table[metric].to_numpy(dtype=float)
        finite = finite[np.isfinite(finite)]
        low, high = (float(np.min(finite)), float(np.max(finite))) if len(finite) else (0.0, 1.0)
        if np.isclose(low, high):
            high = low + max(abs(low) * 0.01, 1e-6)
        for row, group in enumerate(group_order):
            for column, band in enumerate(band_order):
                selected = electrode_band_table.loc[
                    electrode_band_table["group"].eq(group)
                    & electrode_band_table["band"].eq(band)
                ].groupby("electrode")[metric].mean().reindex(info.ch_names)
                values = selected.to_numpy(dtype=float)
                image = _plot_topomap(
                    axes[row, column],
                    values,
                    info,
                    cmap="magma",
                    vlim=(low, high),
                )
                axes[row, column].set_title(band_labels[band])
                if image is not None:
                    fig.colorbar(image, ax=axes[row, column], shrink=0.66)
                if column == 0:
                    axes[row, column].text(
                        -0.22,
                        0.5,
                        group,
                        transform=axes[row, column].transAxes,
                        rotation=90,
                        va="center",
                        ha="center",
                        fontweight="bold",
                    )
        fig.suptitle(f"Group mean {label.lower()} topographies")
        fig.tight_layout()
        _save(fig, output_dir / f"group_topomap_{metric}.png", dpi)


def plot_effect_sizes(
    comparisons: pd.DataFrame,
    band_order: list[str],
    band_labels: dict[str, str],
    path: Path,
    dpi: int,
) -> None:
    """Plot PD-minus-Control Hedges g for subject-level band comparisons."""
    selected_metrics = [
        "peak_power_log10",
        "oscillatory_occupancy",
        "bouts_per_minute",
        "bout_duration_mean_s",
        "cycle_amplitude_mean_uv",
        "cycle_frequency_mean_hz",
        "rise_decay_symmetry_mean",
        "peak_trough_symmetry_mean",
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), squeeze=False)
    for axis, band in zip(axes.flat, band_order):
        table = comparisons.loc[
            comparisons["band"].eq(band)
            & comparisons["metric"].isin(selected_metrics)
        ].set_index("metric").reindex(selected_metrics)
        effects = table["hedges_g_pd_minus_control"].to_numpy(dtype=float)
        significant = table["fdr_reject"].fillna(False).to_numpy(dtype=bool)
        colors = np.where(significant, "#D55E00", "0.55")
        axis.barh(np.arange(len(selected_metrics)), effects, color=colors)
        axis.axvline(0.0, color="black", linewidth=0.8)
        axis.set_yticks(np.arange(len(selected_metrics)), [name.replace("_", " ") for name in selected_metrics], fontsize=8)
        axis.set(xlabel="Hedges g (PD − Control)", title=band_labels[band])
        axis.grid(axis="x", alpha=0.2)
    fig.suptitle("Subject-level PD vs Control effects; orange passes FDR correction")
    fig.tight_layout()
    _save(fig, path, dpi)
