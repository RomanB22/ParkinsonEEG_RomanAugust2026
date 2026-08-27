"""End-to-end whole-head Parkinson severity-axis analysis."""

from __future__ import annotations

import json
import logging
import platform
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any

from core.runtime import configure_runtime

configure_runtime()

import matplotlib
import mne
import numpy as np
import pandas as pd
import scipy

from .features import (
    build_shared_electrode_features,
    load_pd_cohort,
    subject_feature_matrix,
)
from .plots import (
    plot_clinical_axes,
    plot_electrode_selection,
    plot_feature_scatter_pages,
    plot_forest_pages,
)
from .statistics import clinical_axis_association, correlate_progression_features


def load_analysis_config(path: str | Path) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {
        "input", "output_dir", "electrode_scope", "analysis", "features", "plots"
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Missing disease-progression config sections: {missing}")
    scope = config["electrode_scope"]
    if scope.get("policy") != "all_cohort_shared_electrodes":
        raise ValueError("Disease progression must use all cohort-shared electrodes")
    if int(scope.get("expected_count", 0)) < 1:
        raise ValueError("electrode_scope.expected_count must be positive")
    analysis = config["analysis"]
    if analysis.get("group") != "PD":
        raise ValueError("Disease-progression analysis must be restricted to PD")
    if analysis.get("primary_outcome") != "updrs":
        raise ValueError("UPDRS must be the primary progression axis")
    if analysis.get("secondary_outcomes") != ["moca"]:
        raise ValueError("MOCA must be the prespecified complementary axis")
    cognitive_status = analysis.get("cognitive_status", {})
    if cognitive_status.get("impairment_below") != 26 or cognitive_status.get(
        "normal_range"
    ) != [26, 30]:
        raise ValueError("Cognitive status must define impairment <26 and normal 26–30")
    if analysis.get("covariates") != ["age_years", "sex_male"]:
        raise ValueError("Primary correlations must adjust for age and sex")
    if int(analysis["minimum_subjects"]) < 10:
        raise ValueError("analysis.minimum_subjects must be at least 10")
    if int(analysis["bootstrap_resamples"]) < 100:
        raise ValueError("analysis.bootstrap_resamples must be at least 100")
    if not 0.0 < float(analysis["bootstrap_confidence_level"]) < 1.0:
        raise ValueError("Invalid bootstrap confidence level")
    if not 0.0 < float(analysis["fdr_alpha"]) < 1.0:
        raise ValueError("Invalid FDR alpha")
    if analysis.get("fdr_scope") != "within_outcome_feature_family_and_method":
        raise ValueError("Disease-progression FDR scope changed unexpectedly")
    features = config["features"]
    canonical_ordinal = ["delta", "theta", "alpha", "beta", "low_gamma"]
    canonical_bouts = ["theta", "alpha", "low_beta", "high_beta"]
    if features["ordinal_bands"] != canonical_ordinal:
        raise ValueError("Disease progression requires canonical ordinal bands")
    if features["psd_bands"] != canonical_ordinal:
        raise ValueError("Disease progression requires canonical PSD bands")
    if features["bout_bands"] != canonical_bouts:
        raise ValueError("Disease progression requires canonical bout bands")
    if int(features["embedding_dimension"]) != 6 or int(features["delay_samples"]) != 1:
        raise ValueError("Disease progression uses the primary D=6, tau=1 ordinal block")
    if not 0.0 < float(features["minimum_aperiodic_qc_fraction"]) <= 1.0:
        raise ValueError("minimum_aperiodic_qc_fraction must be in (0, 1]")
    if int(config["plots"]["dpi"]) < 50:
        raise ValueError("plots.dpi must be at least 50")
    return config


def _logger(output_dir: Path, overwrite: bool) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("disease_progression")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (
        logging.StreamHandler(),
        logging.FileHandler(
            output_dir / "disease_progression.log", mode="w" if overwrite else "a"
        ),
    ):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def _write_csv(table: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False, float_format="%.17g")


def _topographic_info(config: dict[str, Any], electrodes: list[str]) -> mne.Info:
    files = sorted(Path().glob(str(config["input"]["epoch_example_glob"])))
    if not files:
        raise FileNotFoundError("No epoch file is available for electrode positions")
    epochs = mne.read_epochs(files[0], preload=False, verbose="ERROR")
    missing = sorted(set(electrodes) - set(epochs.ch_names))
    if missing:
        raise ValueError(f"Topographic reference is missing electrodes: {missing}")
    picks = [epochs.ch_names.index(name) for name in electrodes]
    info = mne.pick_info(epochs.info, picks, copy=True)
    info["bads"] = []
    return info


def _format(value: Any, digits: int = 3) -> str:
    return "NA" if not np.isfinite(float(value)) else f"{float(value):.{digits}g}"


def _write_report(
    path: Path,
    cohort: pd.DataFrame,
    dictionary: pd.DataFrame,
    correlations: pd.DataFrame,
    clinical_axes: pd.DataFrame,
    config: dict[str, Any],
    electrodes: list[str],
) -> None:
    adjusted = correlations.loc[
        correlations["method"].eq("partial_spearman_age_sex")
    ].copy()
    primary = adjusted.loc[adjusted["outcome"].eq("updrs")]
    clinical = clinical_axes.loc[
        clinical_axes["method"].eq("partial_spearman_age_sex")
    ].iloc[0]
    cognitive_counts = cohort["cognitive_status"].value_counts()
    lines = [
        "# Whole-head Parkinson disease severity report",
        "",
        "## Scope",
        "",
        (
            "This is a cross-sectional severity analysis within participants with Parkinson "
            "disease. UPDRS is the primary motor-severity axis; MOCA is a complementary "
            "cognitive axis. The word progression describes severity ordering and does not "
            "imply longitudinal change, prediction, or causality."
        ),
        "",
        f"PD participants: {len(cohort)}.",
        f"UPDRS range: {cohort['updrs'].min():g}–{cohort['updrs'].max():g}.",
        f"MOCA range: {cohort['moca'].min():g}–{cohort['moca'].max():g}.",
        (
            "Cognitive status definition: cognitive impairment = MOCA < 26; "
            "cognitively normal = MOCA 26–30."
        ),
        (
            f"PD cognitive impairment: {int(cognitive_counts.get('cognitive_impairment', 0))}; "
            f"PD cognitively normal: {int(cognitive_counts.get('cognitively_normal', 0))}."
        ),
        f"Prespecified EEG features: {len(dictionary)}.",
        f"Cohort-shared electrodes: {len(electrodes)}.",
        "Electrodes: " + ", ".join(electrodes) + ".",
        "Primary estimates are partial Spearman correlations adjusted for age and sex.",
        (
            "BH-FDR is controlled separately within outcome, feature family, and method. "
            "Only canonical, non-overlapping frequency bands are used."
        ),
        "",
        "## Relationship between clinical axes",
        "",
        (
            f"Age/sex-adjusted UPDRS–MOCA partial rho={_format(clinical['estimate'])}, "
            f"p={_format(clinical['p_value'])}, n={int(clinical['n_subjects'])}."
        ),
        "",
        "## UPDRS findings",
        "",
        f"FDR-significant adjusted features: {int(primary['fdr_reject'].sum())} of {len(primary)}.",
        "",
        "| Feature | Family | n | Partial rho | 95% CI | Raw p | Family q | FDR |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    strongest = primary.reindex(primary["estimate"].abs().sort_values(ascending=False).index).head(20)
    for _, row in strongest.iterrows():
        lines.append(
            f"| {row['feature_label']} | {row['family']} | {int(row['n_subjects'])} | "
            f"{_format(row['estimate'])} | [{_format(row['ci_lower'])}, "
            f"{_format(row['ci_upper'])}] | {_format(row['p_value'])} | "
            f"{_format(row['p_fdr_bh'])} | {bool(row['fdr_reject'])} |"
        )
    lines.extend(
        [
            "",
            "## Reading the outputs",
            "",
            (
                "For UPDRS, a positive correlation means the EEG quantity is higher at "
                "greater motor severity. For MOCA, a negative correlation points in the "
                "same worse-disease direction. `progression_aligned_estimate` reverses the "
                "MOCA sign so positive always means higher with worse clinical status."
            ),
            (
                "Use `fdr_reject == True` in `metrics/progression_correlations.csv` for "
                "corrected significance. Scatter plots show raw observations; reported "
                "partial correlations, not the displayed simple regression line, are the "
                "primary statistical result."
            ),
            "",
            "Complete methods are documented in [`METHODS.md`](../METHODS.md).",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_analysis(
    config_path: str | Path,
    *,
    output_dir_override: str | Path | None = None,
    overwrite: bool = False,
    bootstrap_resamples_override: int | None = None,
) -> dict[str, Any]:
    config_path = Path(config_path)
    config = load_analysis_config(config_path)
    if output_dir_override is not None:
        config["output_dir"] = str(output_dir_override)
    if bootstrap_resamples_override is not None:
        if bootstrap_resamples_override < 100:
            raise ValueError("Bootstrap override must be at least 100")
        config["analysis"]["bootstrap_resamples"] = int(bootstrap_resamples_override)
    output_dir = Path(config["output_dir"])
    sentinel = output_dir / "metrics" / "progression_correlations.csv"
    if sentinel.exists() and not overwrite:
        raise FileExistsError(f"Disease-progression outputs exist at {sentinel}")
    logger = _logger(output_dir, overwrite)
    logger.info("Loading PD cohort and resolving all cohort-shared electrodes")
    cohort = load_pd_cohort(config)
    features, dictionary, electrodes = build_shared_electrode_features(config, cohort)
    wide = subject_feature_matrix(cohort, features)
    logger.info("Computing UPDRS-primary and MOCA-secondary correlations")
    correlations = correlate_progression_features(features, dictionary, config)
    clinical_axes = clinical_axis_association(cohort, config)

    metrics_dir = output_dir / "metrics"
    _write_csv(cohort, metrics_dir / "pd_cohort.csv")
    _write_csv(dictionary, metrics_dir / "feature_dictionary.csv")
    _write_csv(features, metrics_dir / "subject_features_long.csv")
    _write_csv(wide, metrics_dir / "subject_feature_matrix.csv")
    _write_csv(correlations, metrics_dir / "progression_correlations.csv")
    _write_csv(clinical_axes, metrics_dir / "clinical_axis_association.csv")
    _write_csv(
        pd.DataFrame(
            {
                "electrode": electrodes,
                "selection_order": np.arange(1, len(electrodes) + 1),
                "role": "all electrodes shared by every subject in the analysis cohort",
            }
        ),
        metrics_dir / "electrode_selection.csv",
    )

    info = _topographic_info(config, electrodes)
    figures_dir = output_dir / "figures"
    dpi = int(config["plots"]["dpi"])
    plot_electrode_selection(info, figures_dir / "electrode_selection.png", dpi)
    plot_clinical_axes(cohort, clinical_axes, figures_dir / "clinical_axes.png", dpi)
    scatter_outputs: list[Path] = []
    forest_outputs: list[Path] = []
    for outcome in [config["analysis"]["primary_outcome"], *config["analysis"]["secondary_outcomes"]]:
        color = str(config["plots"][f"{outcome}_color"])
        scatter_outputs.extend(
            plot_feature_scatter_pages(
                features,
                dictionary,
                correlations,
                outcome=outcome,
                output_dir=figures_dir / "scatter" / outcome,
                dpi=dpi,
                features_per_page=int(config["plots"]["features_per_page"]),
                color=color,
            )
        )
        forest_outputs.extend(
            plot_forest_pages(
                correlations,
                outcome=outcome,
                output_dir=figures_dir / "forest" / outcome,
                dpi=dpi,
                significant_color=str(config["plots"]["significant_color"]),
            )
        )
    _write_report(
        output_dir / "REPORT.md", cohort, dictionary, correlations, clinical_axes,
        config, electrodes
    )

    adjusted = correlations.loc[
        correlations["method"].eq("partial_spearman_age_sex")
    ]
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config_file": str(config_path.resolve()),
        "analysis_config": config,
        "software": {
            "python": platform.python_version(),
            "mne": mne.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "matplotlib": matplotlib.__version__,
            "statsmodels": version("statsmodels"),
        },
        "n_pd_subjects": int(len(cohort)),
        "updrs_range": [float(cohort["updrs"].min()), float(cohort["updrs"].max())],
        "moca_range": [float(cohort["moca"].min()), float(cohort["moca"].max())],
        "cognitive_status_definition": {
            "cognitive_impairment": "MOCA < 26",
            "cognitively_normal": "MOCA 26-30",
        },
        "cognitive_status_counts": cohort["cognitive_status"].value_counts().to_dict(),
        "electrodes": electrodes,
        "n_electrodes": int(len(electrodes)),
        "n_features": int(len(dictionary)),
        "n_adjusted_tests": int(len(adjusted)),
        "adjusted_fdr_rejections": {
            outcome: int(adjusted.loc[adjusted["outcome"].eq(outcome), "fdr_reject"].sum())
            for outcome in adjusted["outcome"].unique()
        },
        "n_scatter_pages": int(len(scatter_outputs)),
        "n_forest_pages": int(len(forest_outputs)),
        "interpretation": (
            "Cross-sectional severity association only; UPDRS is the primary motor axis "
            "and MOCA is complementary. This is not longitudinal progression."
        ),
        "multiplicity": (
            "Benjamini-Hochberg FDR within outcome, feature family, and method; "
            "canonical frequency bands only."
        ),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    logger.info("Disease-progression analysis completed | output=%s", output_dir)
    return manifest
