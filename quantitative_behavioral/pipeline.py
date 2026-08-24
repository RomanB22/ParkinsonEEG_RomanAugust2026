"""Cross-sectional MOCA associations with ordinal and oscillatory-bout features."""

from __future__ import annotations

import json
import logging
import platform
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any

from src.runtime import configure_runtime

configure_runtime()

import matplotlib
import mne
import numpy as np
import pandas as pd
import scipy

from .features import (
    build_electrode_features,
    build_subject_features,
    subject_feature_matrix,
)
from .plots import (
    plot_cohort_audit,
    plot_electrode_topomap_pages,
    plot_family_forest,
    plot_family_heatmap,
    plot_family_scatter_grid,
)
from .statistics import correlate_electrodes, correlate_subject_features


FAMILIES = (
    "ordinal_broadband",
    "ordinal_band",
    "bout_properties",
    "bout_ordinal",
)


def load_analysis_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as stream:
        config = json.load(stream)
    required = {"input", "output_dir", "analysis", "features", "expected", "plots"}
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Missing quantitative-behavioral config sections: {missing}")
    analysis = config["analysis"]
    if analysis.get("primary_group") != "PD":
        raise ValueError("The prespecified primary cohort must be PD")
    if analysis.get("outcome_column") != "MOCA":
        raise ValueError("The prespecified outcome must be MOCA")
    if analysis.get("covariates") != ["AGE", "GENDER"]:
        raise ValueError("Covariates must be prespecified as age and sex")
    if int(analysis["minimum_subjects"]) < 10:
        raise ValueError("analysis.minimum_subjects must be at least 10")
    if int(analysis["bootstrap_resamples"]) < 100:
        raise ValueError("analysis.bootstrap_resamples must be at least 100")
    confidence = float(analysis["bootstrap_confidence_level"])
    if not 0.0 < confidence < 1.0:
        raise ValueError("bootstrap_confidence_level must be between zero and one")
    alpha = float(analysis["fdr_alpha"])
    if not 0.0 < alpha < 1.0:
        raise ValueError("fdr_alpha must be between zero and one")
    if analysis.get("fdr_scope") != "within_feature_family_and_correlation_method":
        raise ValueError("FDR scope must remain within prespecified feature families")
    requested = config["features"]
    regular_metrics = ["entropy", "complexity", "fisher_information"]
    if requested.get("ordinal_metrics") != regular_metrics:
        raise ValueError("Only regular ordinal H, C, and F are supported")
    if requested.get("bout_ordinal_metrics") != regular_metrics:
        raise ValueError("Within-bout ordinal features must be regular H, C, and F")
    if int(config["expected"]["shared_electrodes"]) < 1:
        raise ValueError("expected.shared_electrodes must be positive")
    if int(config["plots"]["dpi"]) < 50:
        raise ValueError("plots.dpi must be at least 50")
    return config


def _configure_logger(output_dir: Path, overwrite: bool) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("quantitative_behavioral")
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(console)
    file_handler = logging.FileHandler(
        output_dir / "quantitative_behavioral.log", mode="w" if overwrite else "a"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def _write_csv(table: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False, float_format="%.17g")


def _validate_upstream_manifests(config: dict[str, Any]) -> dict[str, Any]:
    inputs = config["input"]
    ordinal_manifest_path = Path(inputs["ordinal_subject_file"]).parents[1] / "manifest.json"
    scale_free_manifest_path = Path(inputs["bout_subject_file"]).parents[1] / "manifest.json"
    bout_ordinal_manifest_path = Path(inputs["bout_ordinal_subject_file"]).parents[1] / "manifest.json"
    manifests = {}
    for name, path in (
        ("ordinal", ordinal_manifest_path),
        ("scale_free", scale_free_manifest_path),
        ("bout_ordinal", bout_ordinal_manifest_path),
    ):
        if not path.exists():
            raise FileNotFoundError(f"Missing {name} manifest: {path}")
        manifests[name] = json.loads(path.read_text(encoding="utf-8"))
    expected = config["expected"]
    for name, manifest in manifests.items():
        if int(manifest.get("n_common_electrodes", -1)) != int(expected["shared_electrodes"]):
            raise ValueError(f"{name} manifest does not use the expected shared electrodes")
    ordinal_settings = manifests["ordinal"]["analysis_config"]["ordinal"]
    if int(ordinal_settings["embedding_dimension"]) != int(expected["embedding_dimension"]):
        raise ValueError("Ordinal source has the wrong embedding dimension")
    if int(ordinal_settings["delay_samples"]) != int(expected["delay_samples"]):
        raise ValueError("Ordinal source has the wrong delay")
    bout_ordinal_settings = manifests["bout_ordinal"]["analysis_config"]["ordinal"]
    if int(bout_ordinal_settings["embedding_dimension"]) != int(expected["embedding_dimension"]):
        raise ValueError("Within-bout ordinal source has the wrong embedding dimension")
    if int(bout_ordinal_settings["delay_samples"]) != int(expected["delay_samples"]):
        raise ValueError("Within-bout ordinal source has the wrong delay")
    if bool(manifests["bout_ordinal"].get("renyi_metrics_included", True)):
        raise ValueError("Within-bout source unexpectedly includes Rényi metrics")
    psd = manifests["scale_free"]["analysis_config"]["psd"]
    actual_range = [float(psd["fmin_hz"]), float(psd["fmax_hz"])]
    if actual_range != [float(value) for value in expected["scale_free_psd_range_hz"]]:
        raise ValueError("Scale-free source does not use the expected PSD range")
    return {
        name: {
            "manifest_file": str(path.resolve()),
            "created_utc": manifests[name].get("created_utc"),
            "n_subjects": manifests[name].get("n_subjects"),
            "n_common_electrodes": manifests[name].get("n_common_electrodes"),
        }
        for name, path in (
            ("ordinal", ordinal_manifest_path),
            ("scale_free", scale_free_manifest_path),
            ("bout_ordinal", bout_ordinal_manifest_path),
        )
    }


def _topographic_info(config: dict[str, Any], electrode_order: list[str]) -> Any:
    files = sorted(Path().glob(str(config["input"]["epoch_example_glob"])))
    if not files:
        raise FileNotFoundError("No cleaned epoch file is available for topographic positions")
    epochs = mne.read_epochs(files[0], preload=False, verbose="ERROR")
    missing = sorted(set(electrode_order) - set(epochs.ch_names))
    if missing:
        raise ValueError(f"Topographic reference file is missing electrodes: {missing}")
    picks = [epochs.ch_names.index(name) for name in electrode_order]
    info = mne.pick_info(epochs.info, picks, copy=True)
    info["bads"] = []
    return info


def _write_report(
    path: Path,
    cohort: pd.DataFrame,
    dictionary: pd.DataFrame,
    correlations: pd.DataFrame,
    config: dict[str, Any],
) -> None:
    settings = config["analysis"]
    pd_cohort = cohort.loc[cohort["group"].eq(settings["primary_group"])]
    primary = correlations.loc[correlations["method"].eq("partial_spearman_age_sex")]
    top = primary.reindex(primary["estimate"].abs().sort_values(ascending=False).index).head(12)
    lines = [
        "# Quantitative-behavioral MOCA association report",
        "",
        "## Scope",
        "",
        (
            "This is a cross-sectional association analysis, not a longitudinal measure of "
            "Parkinson disease progression and not evidence of prediction or causality."
        ),
        "",
        f"Primary cohort: {settings['primary_group']} (n={len(pd_cohort)}).",
        f"MOCA range: {pd_cohort['moca'].min():g}–{pd_cohort['moca'].max():g}.",
        f"Prespecified EEG features: {len(dictionary)}.",
        (
            "Primary estimates are partial Spearman correlations after rank-residualizing "
            "MOCA and each EEG feature for age and sex. Unadjusted Spearman estimates are "
            "reported as sensitivity analyses."
        ),
        (
            "Benjamini–Hochberg FDR is controlled separately within each prespecified feature "
            "family and correlation method."
        ),
        "",
        "## Strongest adjusted associations by absolute effect size",
        "",
        "| Feature | Family | n | Partial rho | 95% CI | Raw p | FDR p | FDR reject |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for _, row in top.iterrows():
        lines.append(
            f"| {row['feature_label']} | {row['family']} | {int(row['n_subjects'])} | "
            f"{row['estimate']:.3f} | [{row['ci_lower']:.3f}, {row['ci_upper']:.3f}] | "
            f"{row['p_value']:.4g} | {row['p_fdr_bh']:.4g} | {bool(row['fdr_reject'])} |"
        )
    lines.extend(["", "Complete machine-readable results are in `metrics/subject_level_correlations.csv`.", ""])
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
        if int(bootstrap_resamples_override) < 100:
            raise ValueError("Bootstrap override must be at least 100")
        config["analysis"]["bootstrap_resamples"] = int(bootstrap_resamples_override)
    output_dir = Path(config["output_dir"])
    result_path = output_dir / "metrics" / "subject_level_correlations.csv"
    if result_path.exists() and not overwrite:
        raise FileExistsError(
            f"Quantitative-behavioral outputs exist at {result_path}; rerun with --overwrite"
        )
    logger = _configure_logger(output_dir, overwrite)
    logger.info("Validating upstream manifests and prespecified analysis sources")
    upstream = _validate_upstream_manifests(config)

    cohort, subject_features, dictionary = build_subject_features(config)
    primary_group = str(config["analysis"]["primary_group"])
    primary_n = int(cohort["group"].eq(primary_group).sum())
    if primary_n < int(config["analysis"]["minimum_subjects"]):
        raise ValueError("Primary cohort is smaller than analysis.minimum_subjects")
    logger.info(
        "Subject features ready | subjects=%d | PD=%d | features=%d",
        len(cohort),
        primary_n,
        len(dictionary),
    )
    subject_correlations = correlate_subject_features(subject_features, dictionary, config)
    electrode_features, electrode_order = build_electrode_features(
        config, cohort, dictionary
    )
    electrode_correlations = correlate_electrodes(
        electrode_features, dictionary, config
    )
    wide = subject_feature_matrix(cohort, subject_features)

    metrics_dir = output_dir / "metrics"
    _write_csv(cohort, metrics_dir / "moca_cohort.csv")
    _write_csv(dictionary, metrics_dir / "feature_dictionary.csv")
    _write_csv(subject_features, metrics_dir / "subject_features_long.csv")
    _write_csv(wide, metrics_dir / "analysis_dataset.csv")
    _write_csv(subject_correlations, metrics_dir / "subject_level_correlations.csv")
    _write_csv(electrode_correlations, metrics_dir / "electrode_correlations.csv")
    feature_columns = dictionary["feature_id"].tolist()
    spearman_matrix = wide.loc[wide["group"].eq(primary_group), feature_columns].corr(
        method="spearman"
    )
    spearman_matrix.index.name = "feature_id"
    _write_csv(spearman_matrix.reset_index(), metrics_dir / "pd_feature_spearman_matrix.csv")

    figures_dir = output_dir / "figures"
    dpi = int(config["plots"]["dpi"])
    plot_cohort_audit(
        cohort,
        subject_features,
        dictionary,
        primary_group,
        figures_dir / "audit" / "cohort_and_coverage.png",
        dpi,
    )
    for family in FAMILIES:
        plot_family_forest(
            subject_correlations,
            family,
            figures_dir / "correlations" / f"{family}_forest.png",
            dpi,
        )
        plot_family_heatmap(
            subject_correlations,
            family,
            figures_dir / "correlations" / f"{family}_adjusted_sensitivity_heatmap.png",
            dpi,
        )
        plot_family_scatter_grid(
            subject_features,
            subject_correlations,
            dictionary,
            family,
            primary_group,
            figures_dir / "scatter" / f"{family}_moca_scatter_grid.png",
            dpi,
        )
    info = _topographic_info(config, electrode_order)
    plot_electrode_topomap_pages(
        electrode_correlations,
        dictionary,
        electrode_order,
        info,
        figures_dir / "topomaps",
        dpi,
    )
    _write_report(
        output_dir / "REPORT.md", cohort, dictionary, subject_correlations, config
    )

    primary_results = subject_correlations.loc[
        subject_correlations["method"].eq("partial_spearman_age_sex")
    ]
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config_file": str(config_path.resolve()),
        "analysis_config": config,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "matplotlib": matplotlib.__version__,
            "mne": mne.__version__,
            "statsmodels": version("statsmodels"),
        },
        "upstream": upstream,
        "n_subjects_total": len(cohort),
        "n_primary_pd_subjects": primary_n,
        "moca_missing_primary": int(
            cohort.loc[cohort["group"].eq(primary_group), "moca"].isna().sum()
        ),
        "n_features": len(dictionary),
        "feature_family_counts": dictionary["family"].value_counts().to_dict(),
        "n_subject_level_tests_per_method": len(dictionary),
        "n_primary_fdr_rejections": int(primary_results["fdr_reject"].sum()),
        "n_electrode_tests": len(electrode_correlations),
        "interpretation": (
            "Cross-sectional PD cognition associations only; not longitudinal progression, "
            "causal inference, or validated prediction."
        ),
        "unit_of_analysis": (
            "One subject per correlation. Bout and electrode observations are summarized "
            "within subject before primary inference."
        ),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    logger.info(
        "Completed quantitative-behavioral analysis | PD=%d | features=%d | adjusted FDR rejections=%d",
        primary_n,
        len(dictionary),
        manifest["n_primary_fdr_rejections"],
    )
    return manifest

