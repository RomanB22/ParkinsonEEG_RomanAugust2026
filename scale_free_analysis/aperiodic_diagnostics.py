"""Fit-quality auditing and frequency-range sensitivity for specparam models."""

from __future__ import annotations

import logging
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping

from src.runtime import configure_runtime

configure_runtime()

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, ttest_ind

from .metrics import fit_specparam_spectrum
from .plots import plot_spectral_example


QC_COLUMNS = (
    "specparam_residual_bias_log10",
    "specparam_residual_sd_log10",
    "specparam_residual_max_abs_log10",
    "specparam_fit_qc_pass",
    "specparam_fit_qc_reasons",
)


def _save(fig: Any, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)


def _range_id(limits: list[float] | tuple[float, float]) -> str:
    low, high = (float(value) for value in limits)
    return f"{low:g}_{high:g}Hz"


def assess_specparam_fit(
    metrics: Mapping[str, Any],
    observed_power: np.ndarray,
    modeled_power: np.ndarray,
    qc: Mapping[str, Any],
) -> dict[str, Any]:
    """Return signed log-residual metrics and deterministic QC flags."""
    observed = np.asarray(observed_power, dtype=float)
    modeled = np.asarray(modeled_power, dtype=float)
    if observed.shape != modeled.shape or observed.ndim != 1:
        raise ValueError("Observed and modeled spectra must be matching vectors")
    if np.any(observed <= 0.0) or np.any(modeled <= 0.0):
        raise ValueError("Fit-QC spectra must be positive")
    residual = np.log10(observed) - np.log10(modeled)
    exponent_low, exponent_high = (float(value) for value in qc["exponent_range"])
    reasons = []
    if float(metrics["specparam_r_squared"]) < float(qc["minimum_r_squared"]):
        reasons.append("r_squared_below_minimum")
    if float(metrics["specparam_error_mae"]) > float(
        qc["maximum_error_mae_log10"]
    ):
        reasons.append("mae_above_maximum")
    if not exponent_low <= float(metrics["aperiodic_exponent"]) <= exponent_high:
        reasons.append("exponent_outside_range")
    maximum_residual = float(np.max(np.abs(residual)))
    if maximum_residual > float(qc["maximum_absolute_residual_log10"]):
        reasons.append("residual_above_maximum")
    return {
        "specparam_residual_bias_log10": float(np.mean(residual)),
        "specparam_residual_sd_log10": float(np.std(residual)),
        "specparam_residual_max_abs_log10": maximum_residual,
        "specparam_fit_qc_pass": len(reasons) == 0,
        "specparam_fit_qc_reasons": "pass" if not reasons else ";".join(reasons),
    }


def augment_primary_fit_qc(
    spectra_dir: str | Path,
    electrode_metrics: pd.DataFrame,
    qc: Mapping[str, Any],
) -> pd.DataFrame:
    """Attach residual diagnostics and QC flags to every primary fit."""
    spectra_dir = Path(spectra_dir)
    rows = []
    for subject_id, selected in electrode_metrics.groupby("subject_id", sort=False):
        path = spectra_dir / f"{subject_id}_specparam_spectra.npz"
        if not path.exists():
            raise FileNotFoundError(f"Missing saved specparam spectra: {path}")
        with np.load(path, allow_pickle=False) as spectra:
            electrodes = spectra["electrodes"].astype(str).tolist()
            observed = spectra["observed_psd_uv2_hz"]
            modeled = spectra["modeled_psd_uv2_hz"]
        lookup = {electrode: index for index, electrode in enumerate(electrodes)}
        for _, metric_row in selected.iterrows():
            electrode = str(metric_row["electrode"])
            if electrode not in lookup:
                raise ValueError(f"{path}: missing electrode {electrode}")
            index = lookup[electrode]
            rows.append(
                {
                    "subject_id": str(subject_id),
                    "electrode": electrode,
                    **assess_specparam_fit(
                        metric_row,
                        observed[index],
                        modeled[index],
                        qc,
                    ),
                }
            )
    diagnostics = pd.DataFrame.from_records(rows)
    augmented = electrode_metrics.drop(columns=list(QC_COLUMNS), errors="ignore").merge(
        diagnostics,
        on=["subject_id", "electrode"],
        how="left",
        validate="one_to_one",
    )
    if augmented["specparam_fit_qc_pass"].isna().any():
        raise RuntimeError("Some primary specparam fits are missing QC results")
    augmented["specparam_fit_qc_pass"] = augmented[
        "specparam_fit_qc_pass"
    ].astype(bool)
    augmented["aperiodic_exponent_qc"] = augmented["aperiodic_exponent"].where(
        augmented["specparam_fit_qc_pass"]
    )
    return augmented


def _fit_subject_ranges(task: tuple[Any, ...]) -> list[dict[str, Any]]:
    (
        spectra_path,
        subject_rows,
        ranges,
        primary_range,
        specparam_settings,
        qc,
    ) = task
    with np.load(spectra_path, allow_pickle=False) as spectra:
        electrodes = spectra["electrodes"].astype(str).tolist()
        frequencies = spectra["frequencies_hz"].copy()
        observed_primary = spectra["observed_psd_uv2_hz"].copy()
        modeled_primary = spectra["modeled_psd_uv2_hz"].copy()
    metric_lookup = {str(row["electrode"]): row for row in subject_rows}
    rows = []
    for electrode_index, electrode in enumerate(electrodes):
        primary = metric_lookup[electrode]
        for range_order, limits in enumerate(ranges):
            limits = [float(value) for value in limits]
            is_primary = limits == primary_range
            if is_primary:
                metrics = {
                    key: primary[key]
                    for key in (
                        "aperiodic_offset",
                        "aperiodic_exponent",
                        "specparam_r_squared",
                        "specparam_error_mae",
                        "n_detected_peaks",
                    )
                }
                observed = observed_primary[electrode_index]
                modeled = modeled_primary[electrode_index]
            else:
                settings = dict(specparam_settings)
                settings["frequency_range_hz"] = limits
                metrics, _, curves = fit_specparam_spectrum(
                    frequencies,
                    observed_primary[electrode_index],
                    {},
                    settings,
                )
                observed = curves["observed_psd_uv2_hz"]
                modeled = curves["modeled_psd_uv2_hz"]
            rows.append(
                {
                    "subject_id": str(primary["subject_id"]),
                    "group": str(primary["group"]),
                    "electrode": electrode,
                    "fit_range_id": _range_id(limits),
                    "fit_range_order": int(range_order),
                    "fit_fmin_hz": limits[0],
                    "fit_fmax_hz": limits[1],
                    "is_primary_range": bool(is_primary),
                    **metrics,
                    **assess_specparam_fit(metrics, observed, modeled, qc),
                }
            )
    return rows


def fit_range_sensitivity(
    spectra_dir: str | Path,
    primary_metrics: pd.DataFrame,
    specparam_settings: Mapping[str, Any],
    sensitivity: Mapping[str, Any],
    qc: Mapping[str, Any],
    *,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """Fit every subject/electrode over each configured frequency range."""
    spectra_dir = Path(spectra_dir)
    ranges = [
        [float(value) for value in limits]
        for limits in sensitivity["frequency_ranges_hz"]
    ]
    primary_range = [
        float(value) for value in specparam_settings["frequency_range_hz"]
    ]
    if primary_range not in ranges:
        raise ValueError("Aperiodic sensitivity ranges must include the primary range")
    if str(sensitivity["aperiodic_mode"]) != str(
        specparam_settings["aperiodic_mode"]
    ):
        raise ValueError("Range sensitivity must keep the primary aperiodic mode")
    tasks = []
    for subject_id, selected in primary_metrics.groupby("subject_id", sort=False):
        tasks.append(
            (
                str(spectra_dir / f"{subject_id}_specparam_spectra.npz"),
                selected.to_dict(orient="records"),
                ranges,
                primary_range,
                dict(specparam_settings),
                dict(qc),
            )
        )
    rows: list[dict[str, Any]] = []
    workers = int(sensitivity["workers"])
    if workers == 1:
        for index, task in enumerate(tasks, start=1):
            rows.extend(_fit_subject_ranges(task))
            if logger is not None:
                logger.info("Aperiodic range sensitivity [%d/%d]", index, len(tasks))
    else:
        try:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(_fit_subject_ranges, task) for task in tasks]
                for index, future in enumerate(as_completed(futures), start=1):
                    rows.extend(future.result())
                    if logger is not None and (
                        index == len(tasks) or index % 10 == 0
                    ):
                        logger.info(
                            "Aperiodic range sensitivity [%d/%d]",
                            index,
                            len(tasks),
                        )
        except PermissionError:
            if logger is not None:
                logger.warning(
                    "Process workers are unavailable; using deterministic serial fitting"
                )
            rows.clear()
            for index, task in enumerate(tasks, start=1):
                rows.extend(_fit_subject_ranges(task))
                if logger is not None and (
                    index == len(tasks) or index % 10 == 0
                ):
                    logger.info(
                        "Aperiodic range sensitivity [%d/%d]", index, len(tasks)
                    )
    return pd.DataFrame.from_records(rows).sort_values(
        ["fit_range_order", "subject_id", "electrode"]
    ).reset_index(drop=True)


def summarize_range_sensitivity(
    electrode_sensitivity: pd.DataFrame,
    qc: Mapping[str, Any],
) -> pd.DataFrame:
    """Create one all-fit and one QC-qualified exponent summary per subject/range."""
    keys = [
        "subject_id",
        "group",
        "fit_range_id",
        "fit_range_order",
        "fit_fmin_hz",
        "fit_fmax_hz",
        "is_primary_range",
    ]
    rows = []
    for key_values, selected in electrode_sensitivity.groupby(keys, sort=False):
        row = dict(zip(keys, key_values))
        passing = selected.loc[selected["specparam_fit_qc_pass"]]
        total = int(selected["electrode"].nunique())
        n_passing = int(passing["electrode"].nunique())
        fraction = n_passing / total
        subject_pass = fraction >= float(qc["minimum_subject_qc_fraction"])
        row.update(
            {
                "n_electrodes": total,
                "n_qc_pass_electrodes": n_passing,
                "qc_pass_fraction": fraction,
                "subject_fit_qc_pass": subject_pass,
                "aperiodic_exponent_all_mean": float(
                    selected["aperiodic_exponent"].mean()
                ),
                "aperiodic_exponent_all_median": float(
                    selected["aperiodic_exponent"].median()
                ),
                "aperiodic_exponent_qc_mean": (
                    float(passing["aperiodic_exponent"].mean())
                    if len(passing)
                    else np.nan
                ),
                "aperiodic_exponent_qc_qualified": (
                    float(passing["aperiodic_exponent"].mean())
                    if subject_pass and len(passing)
                    else np.nan
                ),
                "specparam_r_squared_mean": float(
                    selected["specparam_r_squared"].mean()
                ),
                "specparam_error_mae_mean": float(
                    selected["specparam_error_mae"].mean()
                ),
            }
        )
        rows.append(row)
    return pd.DataFrame.from_records(rows).sort_values(
        ["fit_range_order", "subject_id"]
    ).reset_index(drop=True)


def _hedges_g(values_a: np.ndarray, values_b: np.ndarray) -> float:
    n_a, n_b = len(values_a), len(values_b)
    if n_a < 2 or n_b < 2:
        return np.nan
    pooled = (
        (n_a - 1) * np.var(values_a, ddof=1)
        + (n_b - 1) * np.var(values_b, ddof=1)
    ) / (n_a + n_b - 2)
    correction = 1.0 - 3.0 / (4.0 * (n_a + n_b) - 9.0)
    return float(
        correction * (np.mean(values_a) - np.mean(values_b)) / np.sqrt(pooled)
    )


def compare_ranges(subject_sensitivity: pd.DataFrame) -> pd.DataFrame:
    """Return transparent unadjusted PD-Control summaries for every range/QC policy."""
    rows = []
    policies = {
        "all_electrode_fits": "aperiodic_exponent_all_mean",
        "qc_qualified_subjects": "aperiodic_exponent_qc_qualified",
    }
    for range_id, selected in subject_sensitivity.groupby("fit_range_id", sort=False):
        for policy, column in policies.items():
            pd_values = selected.loc[selected["group"].eq("PD"), column].dropna().to_numpy()
            control_values = selected.loc[
                selected["group"].eq("Control"), column
            ].dropna().to_numpy()
            if len(pd_values) >= 2 and len(control_values) >= 2:
                welch = ttest_ind(pd_values, control_values, equal_var=False)
                mann = mannwhitneyu(
                    pd_values, control_values, alternative="two-sided"
                )
                welch_t = float(welch.statistic)
                welch_p = float(welch.pvalue)
                mann_u = float(mann.statistic)
                mann_p = float(mann.pvalue)
            else:
                welch_t = welch_p = mann_u = mann_p = np.nan
            first = selected.iloc[0]
            rows.append(
                {
                    "fit_range_id": range_id,
                    "fit_range_order": first["fit_range_order"],
                    "fit_fmin_hz": first["fit_fmin_hz"],
                    "fit_fmax_hz": first["fit_fmax_hz"],
                    "is_primary_range": first["is_primary_range"],
                    "aggregation_policy": policy,
                    "n_pd": len(pd_values),
                    "n_control": len(control_values),
                    "pd_mean": float(np.mean(pd_values)) if len(pd_values) else np.nan,
                    "control_mean": (
                        float(np.mean(control_values)) if len(control_values) else np.nan
                    ),
                    "mean_difference_pd_minus_control": float(
                        np.mean(pd_values) - np.mean(control_values)
                    ) if len(pd_values) and len(control_values) else np.nan,
                    "hedges_g_pd_minus_control": _hedges_g(
                        pd_values, control_values
                    ),
                    "welch_t": welch_t,
                    "welch_p_value": welch_p,
                    "mann_whitney_u": mann_u,
                    "mann_whitney_p_value": mann_p,
                }
            )
    return pd.DataFrame.from_records(rows)


def qc_summary(electrode_metrics: pd.DataFrame) -> pd.DataFrame:
    """Summarize electrode-level primary fit QC by group and overall."""
    rows = []
    groups: list[tuple[str, pd.DataFrame]] = [("All", electrode_metrics)]
    groups.extend(list(electrode_metrics.groupby("group", sort=False)))
    for group, selected in groups:
        reason_counts = selected["specparam_fit_qc_reasons"].value_counts()
        rows.append(
            {
                "group": group,
                "n_fits": len(selected),
                "n_qc_pass": int(selected["specparam_fit_qc_pass"].sum()),
                "qc_pass_fraction": float(selected["specparam_fit_qc_pass"].mean()),
                "r_squared_mean": float(selected["specparam_r_squared"].mean()),
                "r_squared_median": float(selected["specparam_r_squared"].median()),
                "error_mae_mean": float(selected["specparam_error_mae"].mean()),
                "residual_max_abs_mean": float(
                    selected["specparam_residual_max_abs_log10"].mean()
                ),
                "failure_reason_counts": "|".join(
                    f"{reason}:{count}"
                    for reason, count in reason_counts.items()
                    if reason != "pass"
                ),
            }
        )
    return pd.DataFrame.from_records(rows)


def plot_fit_qc_dashboard(
    metrics: pd.DataFrame,
    qc: Mapping[str, Any],
    colors: Mapping[str, str],
    path: Path,
    dpi: int,
) -> None:
    """Show fit-quality distributions, thresholds, and subject QC coverage."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for group, selected in metrics.groupby("group", sort=False):
        color = colors.get(str(group), "0.4")
        axes[0, 0].hist(
            selected["specparam_r_squared"],
            bins=np.linspace(0, 1, 41),
            density=True,
            histtype="step",
            linewidth=1.8,
            color=color,
            label=str(group),
        )
        axes[0, 1].hist(
            selected["specparam_error_mae"],
            bins=40,
            density=True,
            histtype="step",
            linewidth=1.8,
            color=color,
            label=str(group),
        )
        axes[1, 0].hist(
            selected["specparam_residual_max_abs_log10"],
            bins=40,
            density=True,
            histtype="step",
            linewidth=1.8,
            color=color,
            label=str(group),
        )
    axes[0, 0].axvline(float(qc["minimum_r_squared"]), color="black", linestyle="--")
    axes[0, 1].axvline(
        float(qc["maximum_error_mae_log10"]), color="black", linestyle="--"
    )
    axes[1, 0].axvline(
        float(qc["maximum_absolute_residual_log10"]),
        color="black",
        linestyle="--",
    )
    axes[0, 0].set(title="Model R²", xlabel="R²", ylabel="Density")
    axes[0, 1].set(title="Mean absolute log error", xlabel="MAE (log₁₀ power)")
    axes[1, 0].set(
        title="Largest absolute signed residual",
        xlabel="max |observed − full model| (log₁₀ power)",
        ylabel="Density",
    )
    subject_qc = (
        metrics.groupby(["subject_id", "group"])["specparam_fit_qc_pass"]
        .mean()
        .rename("fraction")
        .reset_index()
    )
    rng = np.random.default_rng(0)
    for position, (group, selected) in enumerate(subject_qc.groupby("group", sort=False)):
        values = selected["fraction"].to_numpy(dtype=float)
        axes[1, 1].scatter(
            position + rng.uniform(-0.12, 0.12, len(values)),
            values,
            color=colors.get(str(group), "0.4"),
            alpha=0.55,
            s=20,
            label=str(group),
        )
    axes[1, 1].axhline(
        float(qc["minimum_subject_qc_fraction"]), color="black", linestyle="--"
    )
    axes[1, 1].set_xticks(
        np.arange(subject_qc["group"].nunique()),
        subject_qc["group"].drop_duplicates(),
    )
    axes[1, 1].set(
        title="QC-passing electrode fraction per subject",
        ylabel="Fraction of 60 shared electrodes",
        ylim=(-0.03, 1.03),
    )
    for axis in axes.flat:
        axis.grid(alpha=0.2)
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Specparam fit-quality audit — dashed lines are configured thresholds")
    fig.tight_layout()
    _save(fig, path, dpi)


def plot_range_sensitivity(
    subjects: pd.DataFrame,
    colors: Mapping[str, str],
    path: Path,
    dpi: int,
) -> None:
    """Show exponent and QC stability across fixed-mode frequency ranges."""
    order = (
        subjects[["fit_range_id", "fit_range_order"]]
        .drop_duplicates()
        .sort_values("fit_range_order")["fit_range_id"]
        .tolist()
    )
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    rng = np.random.default_rng(1)
    groups = subjects["group"].drop_duplicates().tolist()
    offsets = np.linspace(-0.16, 0.16, len(groups))
    for group, offset in zip(groups, offsets):
        for position, range_id in enumerate(order):
            selected = subjects.loc[
                subjects["group"].eq(group)
                & subjects["fit_range_id"].eq(range_id)
            ]
            values = selected["aperiodic_exponent_all_mean"].to_numpy(dtype=float)
            axes[0, 0].scatter(
                position + offset + rng.uniform(-0.035, 0.035, len(values)),
                values,
                s=11,
                alpha=0.28,
                color=colors.get(str(group), "0.4"),
            )
            mean = float(np.mean(values))
            sem = (
                float(np.std(values, ddof=1) / np.sqrt(len(values)))
                if len(values) > 1
                else 0.0
            )
            axes[0, 0].errorbar(
                position + offset,
                mean,
                yerr=1.96 * sem,
                fmt="o",
                capsize=3,
                color=colors.get(str(group), "0.4"),
                label=str(group) if position == 0 else None,
            )
            axes[1, 0].scatter(
                position + offset,
                selected["qc_pass_fraction"].mean(),
                color=colors.get(str(group), "0.4"),
                s=55,
                label=str(group) if position == 0 else None,
            )
    axes[0, 0].set_xticks(np.arange(len(order)), order)
    axes[0, 0].set(
        title="All-electrode subject means",
        ylabel="Aperiodic exponent",
    )
    axes[0, 0].legend(frameon=False)
    axes[1, 0].set_xticks(np.arange(len(order)), order)
    axes[1, 0].set(
        title="Mean electrode QC-pass fraction",
        ylabel="QC-pass fraction",
        ylim=(0, 1.03),
    )
    matrix = subjects.pivot(
        index="subject_id",
        columns="fit_range_id",
        values="aperiodic_exponent_all_mean",
    ).reindex(columns=order)
    correlation = matrix.corr(method="spearman")
    image = axes[0, 1].imshow(correlation, vmin=0.0, vmax=1.0, cmap="viridis")
    axes[0, 1].set_xticks(np.arange(len(order)), order, rotation=35, ha="right")
    axes[0, 1].set_yticks(np.arange(len(order)), order)
    axes[0, 1].set_title("Cross-range subject Spearman correlation")
    for row in range(len(order)):
        for column in range(len(order)):
            axes[0, 1].text(
                column,
                row,
                f"{correlation.iloc[row, column]:.2f}",
                ha="center",
                va="center",
                color="white" if correlation.iloc[row, column] < 0.65 else "black",
            )
    fig.colorbar(image, ax=axes[0, 1], shrink=0.75)
    primary = subjects.loc[subjects["is_primary_range"]]
    reasons = (
        primary.assign(failed=~primary["subject_fit_qc_pass"])
        .groupby("group")["failed"]
        .agg(["sum", "count"])
    )
    axes[1, 1].axis("off")
    lines = [
        "Interpretation",
        "",
        "All models use fixed mode and identical peak settings.",
        "Only the fitted frequency range changes.",
        "",
    ]
    for group, row in reasons.iterrows():
        lines.append(
            f"Primary range: {int(row['sum'])}/{int(row['count'])} {group} subjects "
            "fall below the configured 80% electrode-QC coverage."
        )
    axes[1, 1].text(0.0, 1.0, "\n".join(lines), va="top", fontsize=10)
    for axis in (axes[0, 0], axes[1, 0]):
        axis.grid(alpha=0.2)
    fig.suptitle("Aperiodic exponent sensitivity to fitting range")
    fig.tight_layout()
    _save(fig, path, dpi)


def plot_group_median_decomposition(
    spectra_dir: str | Path,
    metrics: pd.DataFrame,
    colors: Mapping[str, str],
    path: Path,
    dpi: int,
) -> None:
    """Plot cohort-level observed, full-model, aperiodic, and residual curves."""
    spectra_dir = Path(spectra_dir)
    subject_curves = []
    frequencies = None
    group_lookup = metrics.drop_duplicates("subject_id").set_index("subject_id")["group"]
    for subject_id in group_lookup.index:
        with np.load(
            spectra_dir / f"{subject_id}_specparam_spectra.npz", allow_pickle=False
        ) as spectra:
            current_frequencies = spectra["frequencies_hz"].copy()
            if frequencies is None:
                frequencies = current_frequencies
            elif not np.array_equal(frequencies, current_frequencies):
                raise ValueError("Saved specparam spectra use inconsistent frequencies")
            subject_curves.append(
                {
                    "subject_id": subject_id,
                    "group": group_lookup.loc[subject_id],
                    "observed": np.median(
                        np.log10(spectra["observed_psd_uv2_hz"]), axis=0
                    ),
                    "modeled": np.median(
                        np.log10(spectra["modeled_psd_uv2_hz"]), axis=0
                    ),
                    "aperiodic": np.median(
                        np.log10(spectra["aperiodic_psd_uv2_hz"]), axis=0
                    ),
                }
            )
    groups = metrics["group"].drop_duplicates().tolist()
    fig, axes = plt.subplots(len(groups), 2, figsize=(14, 4.5 * len(groups)), squeeze=False)
    for row, group in enumerate(groups):
        selected = [item for item in subject_curves if item["group"] == group]
        observed = np.median(np.stack([item["observed"] for item in selected]), axis=0)
        modeled = np.median(np.stack([item["modeled"] for item in selected]), axis=0)
        aperiodic = np.median(np.stack([item["aperiodic"] for item in selected]), axis=0)
        axes[row, 0].plot(frequencies, observed, color="black", label="Observed")
        axes[row, 0].plot(frequencies, modeled, color="#0072B2", label="Full model")
        axes[row, 0].plot(frequencies, aperiodic, color="#D55E00", label="Aperiodic")
        axes[row, 0].set(
            title=f"{group}: median subject/electrode decomposition",
            xlabel="Frequency (Hz)",
            ylabel="log₁₀ PSD",
        )
        residual = observed - modeled
        axes[row, 1].axhline(0.0, color="black", linewidth=0.8)
        axes[row, 1].plot(frequencies, residual, color=colors.get(str(group), "0.4"))
        axes[row, 1].fill_between(
            frequencies, 0.0, residual, where=residual >= 0, color="#009E73", alpha=0.3
        )
        axes[row, 1].fill_between(
            frequencies, 0.0, residual, where=residual < 0, color="#CC79A7", alpha=0.3
        )
        axes[row, 1].set(
            title=f"{group}: observed − full model",
            xlabel="Frequency (Hz)",
            ylabel="Signed residual (log₁₀ PSD)",
        )
        for axis in axes[row]:
            axis.grid(alpha=0.2)
    axes[0, 0].legend(frameon=False)
    fig.suptitle(
        "Cohort-level specparam audit (median across electrodes within subject, then subjects)"
    )
    fig.tight_layout()
    _save(fig, path, dpi)


def run_aperiodic_diagnostics(
    output_dir: str | Path,
    electrode_metrics: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """Run and save the complete primary-QC and fit-range sensitivity audit."""
    output_dir = Path(output_dir)
    spectra_dir = output_dir / "intermediate" / "spectra"
    metrics_dir = output_dir / "metrics"
    figures_dir = output_dir / "figures" / "aperiodic_diagnostics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    qc = config["aperiodic_fit_qc"]
    augmented = augment_primary_fit_qc(spectra_dir, electrode_metrics, qc)
    sensitivity = fit_range_sensitivity(
        spectra_dir,
        augmented,
        config["specparam"],
        config["aperiodic_sensitivity"],
        qc,
        logger=logger,
    )
    subject_sensitivity = summarize_range_sensitivity(sensitivity, qc)
    range_comparisons = compare_ranges(subject_sensitivity)
    summary = qc_summary(augmented)
    augmented.to_csv(
        metrics_dir / "electrode_aperiodic_metrics.csv",
        index=False,
        float_format="%.17g",
    )
    sensitivity.to_csv(
        metrics_dir / "electrode_aperiodic_range_sensitivity.csv.gz",
        index=False,
        float_format="%.17g",
        compression="gzip",
    )
    subject_sensitivity.to_csv(
        metrics_dir / "subject_aperiodic_range_sensitivity.csv",
        index=False,
        float_format="%.17g",
    )
    subject_sensitivity.loc[subject_sensitivity["is_primary_range"]].to_csv(
        metrics_dir / "subject_aperiodic_qc_metrics.csv",
        index=False,
        float_format="%.17g",
    )
    range_comparisons.to_csv(
        metrics_dir / "aperiodic_range_group_comparisons.csv",
        index=False,
        float_format="%.17g",
    )
    summary.to_csv(
        metrics_dir / "specparam_fit_qc_summary.csv",
        index=False,
        float_format="%.17g",
    )
    colors = {
        str(group): str(config["plots"]["group_colors"].get(group, "0.4"))
        for group in augmented["group"].drop_duplicates()
    }
    dpi = int(config["plots"]["dpi"])
    plot_fit_qc_dashboard(
        augmented,
        qc,
        colors,
        figures_dir / "fit_qc_dashboard.png",
        dpi,
    )
    plot_range_sensitivity(
        subject_sensitivity,
        colors,
        figures_dir / "frequency_range_sensitivity.png",
        dpi,
    )
    plot_group_median_decomposition(
        spectra_dir,
        augmented,
        colors,
        figures_dir / "group_median_decomposition_and_residuals.png",
        dpi,
    )
    example_row = augmented.iloc[0]
    example_path = spectra_dir / f"{example_row['subject_id']}_specparam_spectra.npz"
    with np.load(example_path, allow_pickle=False) as spectra:
        electrodes = spectra["electrodes"].astype(str).tolist()
        example_index = electrodes.index(str(example_row["electrode"]))
        example = {
            "subject_id": str(example_row["subject_id"]),
            "group": str(example_row["group"]),
            "electrode": str(example_row["electrode"]),
            "frequencies_hz": spectra["frequencies_hz"].copy(),
            "aperiodic_exponent": float(example_row["aperiodic_exponent"]),
            "specparam_r_squared": float(example_row["specparam_r_squared"]),
            "specparam_error_mae": float(example_row["specparam_error_mae"]),
            "specparam_fit_qc_pass": bool(
                example_row["specparam_fit_qc_pass"]
            ),
            "specparam_fit_qc_reasons": str(
                example_row["specparam_fit_qc_reasons"]
            ),
            **{
                name: spectra[name][example_index].copy()
                for name in (
                    "observed_psd_uv2_hz",
                    "modeled_psd_uv2_hz",
                    "aperiodic_psd_uv2_hz",
                    "periodic_psd_uv2_hz",
                )
            },
        }
    plot_spectral_example(
        example,
        output_dir / "figures" / "examples" / "specparam_decomposition.png",
        dpi,
    )
    return {
        "electrode_metrics": augmented,
        "electrode_sensitivity": sensitivity,
        "subject_sensitivity": subject_sensitivity,
        "range_comparisons": range_comparisons,
        "qc_summary": summary,
    }
