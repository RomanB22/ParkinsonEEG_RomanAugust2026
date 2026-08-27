"""Fit-quality and fixed-versus-knee auditing for specparam models."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping

from core.runtime import configure_runtime

configure_runtime()

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from core.plotting import save_figure as _save
from .plots import plot_spectral_example


QC_COLUMNS = (
    "specparam_residual_bias_log10",
    "specparam_residual_sd_log10",
    "specparam_residual_max_abs_log10",
    "specparam_fit_qc_pass",
    "specparam_fit_qc_reasons",
)


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


def summarize_primary_fit_qc(
    electrode_metrics: pd.DataFrame,
    qc: Mapping[str, Any],
) -> pd.DataFrame:
    """Create one all-fit and QC-qualified 4–50 Hz exponent row per subject."""
    rows = []
    for (subject_id, group), selected in electrode_metrics.groupby(
        ["subject_id", "group"], sort=False
    ):
        passing = selected.loc[selected["specparam_fit_qc_pass"]]
        total = int(selected["electrode"].nunique())
        n_passing = int(passing["electrode"].nunique())
        fraction = n_passing / total
        subject_pass = fraction >= float(qc["minimum_subject_qc_fraction"])
        rows.append(
            {
                "subject_id": str(subject_id),
                "group": str(group),
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
    return pd.DataFrame.from_records(rows).sort_values("subject_id").reset_index(
        drop=True
    )


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


def summarize_model_selection(
    electrode_metrics: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return electrode provenance and one fixed-versus-knee summary per subject."""
    comparison_columns = [
        "subject_id",
        "group",
        "electrode",
        "specparam_aperiodic_mode",
        "specparam_model_selection_criterion",
        "specparam_model_selection_reason",
        "specparam_delta_bic_knee_minus_fixed",
        "knee_model_fit_success",
        "knee_model_eligible",
        "knee_frequency_outlier_within_subject",
        "knee_frequency_zscore_within_subject",
        "fixed_aperiodic_offset",
        "fixed_aperiodic_exponent",
        "fixed_specparam_r_squared",
        "fixed_specparam_error_mae",
        "fixed_specparam_aic",
        "fixed_specparam_bic",
        "knee_aperiodic_offset",
        "knee_aperiodic_knee",
        "knee_aperiodic_exponent",
        "knee_aperiodic_knee_frequency_hz",
        "knee_specparam_r_squared",
        "knee_specparam_error_mae",
        "knee_specparam_aic",
        "knee_specparam_bic",
    ]
    missing = sorted(set(comparison_columns) - set(electrode_metrics.columns))
    if missing:
        raise ValueError(f"Model-comparison metrics are missing columns: {missing}")
    electrode = electrode_metrics[comparison_columns].copy()
    rows = []
    for (subject_id, group), selected in electrode.groupby(
        ["subject_id", "group"], sort=False
    ):
        valid_knees = selected.loc[selected["knee_model_eligible"]]
        rows.append(
            {
                "subject_id": subject_id,
                "group": group,
                "n_electrodes": int(selected["electrode"].nunique()),
                "n_knee_selected": int(
                    selected["specparam_aperiodic_mode"].eq("knee").sum()
                ),
                "fraction_knee_selected": float(
                    selected["specparam_aperiodic_mode"].eq("knee").mean()
                ),
                "n_knee_frequency_outliers": int(
                    selected["knee_frequency_outlier_within_subject"].sum()
                ),
                "fixed_aperiodic_exponent_mean": float(
                    selected["fixed_aperiodic_exponent"].mean()
                ),
                "knee_aperiodic_exponent_mean": float(
                    valid_knees["knee_aperiodic_exponent"].mean()
                ),
                "knee_frequency_hz_mean": float(
                    valid_knees["knee_aperiodic_knee_frequency_hz"].mean()
                ),
                "delta_bic_knee_minus_fixed_mean": float(
                    selected["specparam_delta_bic_knee_minus_fixed"].mean()
                ),
                "fixed_r_squared_mean": float(
                    selected["fixed_specparam_r_squared"].mean()
                ),
                "knee_r_squared_mean": float(
                    selected["knee_specparam_r_squared"].mean()
                ),
            }
        )
    return electrode, pd.DataFrame.from_records(rows)


def plot_model_selection_summary(
    electrode: pd.DataFrame,
    subject: pd.DataFrame,
    colors: Mapping[str, str],
    path: Path,
    dpi: int,
) -> None:
    """Visualize BIC selection, knee frequencies, and group coverage."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    groups = subject["group"].drop_duplicates().astype(str).tolist()
    for group_index, group in enumerate(groups):
        group_subject = subject.loc[subject["group"].eq(group)]
        x = np.full(len(group_subject), group_index, dtype=float)
        jitter = np.linspace(-0.12, 0.12, max(len(group_subject), 1))[: len(group_subject)]
        axes[0, 0].scatter(
            x + jitter,
            group_subject["fraction_knee_selected"],
            color=colors.get(group, "0.4"),
            alpha=0.65,
            s=18,
        )
        group_electrode = electrode.loc[electrode["group"].eq(group)]
        axes[0, 1].hist(
            group_electrode["specparam_delta_bic_knee_minus_fixed"].dropna(),
            bins=40,
            histtype="step",
            density=True,
            linewidth=1.7,
            color=colors.get(group, "0.4"),
            label=group,
        )
        valid = group_electrode.loc[group_electrode["knee_model_eligible"]]
        axes[1, 0].hist(
            valid["knee_aperiodic_knee_frequency_hz"].dropna(),
            bins=40,
            histtype="step",
            density=True,
            linewidth=1.7,
            color=colors.get(group, "0.4"),
            label=group,
        )
    axes[0, 0].set_xticks(range(len(groups)), groups)
    axes[0, 0].set(
        ylabel="Fraction of electrodes",
        title="Knee model selected within each subject",
        ylim=(-0.03, 1.03),
    )
    axes[0, 1].axvline(0.0, color="black", linestyle="--", linewidth=1.0)
    axes[0, 1].set(
        xlabel="BIC(knee) − BIC(fixed)",
        ylabel="Density",
        title="Negative values favor knee",
    )
    axes[1, 0].set(
        xlabel="Knee frequency (Hz)",
        ylabel="Density",
        title="Interpretable knee fits after 2-SD exclusion",
    )
    counts = (
        electrode.groupby(["group", "specparam_aperiodic_mode"], sort=False)
        .size()
        .unstack(fill_value=0)
        .reindex(index=groups, columns=["fixed", "knee"], fill_value=0)
    )
    axes[1, 1].bar(groups, counts["fixed"], color="#999999", label="Fixed")
    axes[1, 1].bar(
        groups,
        counts["knee"],
        bottom=counts["fixed"],
        color="#CC79A7",
        label="Knee",
    )
    axes[1, 1].set(ylabel="Electrode fits", title="BIC-selected threshold model")
    axes[1, 1].legend(frameon=False)
    for axis in axes.flat:
        axis.grid(alpha=0.2)
    axes[0, 1].legend(frameon=False)
    axes[1, 0].legend(frameon=False)
    fig.suptitle("Fixed versus knee specparam audit — both fitted over 4–50 Hz")
    fig.tight_layout()
    _save(fig, path, dpi)


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
    """Run and save the complete 4–50 Hz fit-QC and model-selection audit."""
    output_dir = Path(output_dir)
    spectra_dir = output_dir / "intermediate" / "spectra"
    metrics_dir = output_dir / "metrics"
    figures_dir = output_dir / "figures" / "aperiodic_diagnostics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    qc = config["aperiodic_fit_qc"]
    augmented = augment_primary_fit_qc(spectra_dir, electrode_metrics, qc)
    subject_qc = summarize_primary_fit_qc(augmented, qc)
    summary = qc_summary(augmented)
    model_electrode, model_subject = summarize_model_selection(augmented)
    augmented.to_csv(
        metrics_dir / "electrode_aperiodic_metrics.csv",
        index=False,
        float_format="%.17g",
    )
    subject_qc.to_csv(
        metrics_dir / "subject_aperiodic_qc_metrics.csv",
        index=False,
        float_format="%.17g",
    )
    for retired_path in (
        metrics_dir / "electrode_aperiodic_range_sensitivity.csv.gz",
        metrics_dir / "subject_aperiodic_range_sensitivity.csv",
        metrics_dir / "aperiodic_range_group_comparisons.csv",
        figures_dir / "frequency_range_sensitivity.png",
    ):
        retired_path.unlink(missing_ok=True)
    summary.to_csv(
        metrics_dir / "specparam_fit_qc_summary.csv",
        index=False,
        float_format="%.17g",
    )
    model_electrode.to_csv(
        metrics_dir / "electrode_aperiodic_model_comparison.csv.gz",
        index=False,
        float_format="%.17g",
        compression="gzip",
    )
    model_subject.to_csv(
        metrics_dir / "subject_aperiodic_model_comparison.csv",
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
    plot_model_selection_summary(
        model_electrode,
        model_subject,
        colors,
        figures_dir / "fixed_vs_knee_model_selection.png",
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
            "specparam_aperiodic_mode": str(
                example_row["specparam_aperiodic_mode"]
            ),
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
                    "fixed_aperiodic_psd_uv2_hz",
                    "knee_aperiodic_psd_uv2_hz",
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
        "subject_qc": subject_qc,
        "qc_summary": summary,
        "model_comparison_electrode": model_electrode,
        "model_comparison_subject": model_subject,
    }
