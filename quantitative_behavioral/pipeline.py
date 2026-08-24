"""Cross-sectional MOCA associations with ordinal and oscillatory-bout features."""

from __future__ import annotations

import json
import logging
import platform
import shutil
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
    build_dimension_sensitivity_features,
    build_electrode_features,
    build_subject_features,
    subject_feature_matrix,
)
from .plots import (
    plot_cohort_audit,
    plot_dimension_sensitivity_heatmaps,
    plot_dimension_stability_lines,
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

DIMENSION_METRICS = (
    "entropy",
    "complexity",
    "fisher_information",
    "renyi_entropy_alpha_0_5",
    "renyi_complexity_alpha_0_5",
    "renyi_entropy_alpha_0_9",
    "renyi_complexity_alpha_0_9",
    "renyi_entropy_alpha_1_1",
    "renyi_complexity_alpha_1_1",
    "renyi_entropy_alpha_2",
    "renyi_complexity_alpha_2",
    "renyi_entropy_alpha_5",
    "renyi_complexity_alpha_5",
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
    sensitivity = config.get("dimension_sensitivity")
    if not isinstance(sensitivity, dict) or not sensitivity.get("enabled"):
        raise ValueError("The ordinal embedding-dimension sensitivity must be enabled")
    if sensitivity.get("embedding_dimensions") != [3, 4, 5, 6]:
        raise ValueError("Dimension sensitivity must prespecify D=3,4,5,6")
    if int(sensitivity.get("delay_samples", -1)) != 1:
        raise ValueError("Dimension sensitivity must use tau=1")
    if sensitivity.get("metrics") != list(DIMENSION_METRICS):
        raise ValueError(
            "Dimension analysis must include regular H/C/F plus Rényi Hα/Cα at "
            "alpha=0.5, 0.9, 1.1, 2, and 5"
        )
    if (
        sensitivity.get("analysis_block_policy")
        != "one_separate_feature_matrix_per_embedding_dimension"
    ):
        raise ValueError("Each embedding dimension must use a separate feature matrix")
    if (
        sensitivity.get("fdr_scope")
        != "within_each_dimension_across_all_91_features_per_method"
    ):
        raise ValueError("Dimension-analysis FDR must be controlled separately within D")
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
    provenance = {
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
    sensitivity = config["dimension_sensitivity"]
    sensitivity_root = Path(sensitivity["ordinal_output_root"])
    sensitivity_delay = int(sensitivity["delay_samples"])
    for dimension in sensitivity["embedding_dimensions"]:
        name = f"ordinal_D{int(dimension)}_tau{sensitivity_delay}"
        path = sensitivity_root / f"D{int(dimension)}_tau{sensitivity_delay}" / "manifest.json"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing dimension-sensitivity manifest: {path}. Run "
                "bash quantitative_behavioral/prepare_dimension_sensitivity.sh"
            )
        manifest = json.loads(path.read_text(encoding="utf-8"))
        settings = manifest["analysis_config"]["ordinal"]
        if int(settings["embedding_dimension"]) != int(dimension):
            raise ValueError(f"{name} manifest has the wrong embedding dimension")
        if int(settings["delay_samples"]) != sensitivity_delay:
            raise ValueError(f"{name} manifest has the wrong delay")
        if int(manifest.get("n_common_electrodes", -1)) != int(
            config["expected"]["shared_electrodes"]
        ):
            raise ValueError(f"{name} manifest does not use the shared-electrode set")
        provenance[name] = {
            "manifest_file": str(path.resolve()),
            "created_utc": manifest.get("created_utc"),
            "n_subjects": manifest.get("n_subjects"),
            "n_common_electrodes": manifest.get("n_common_electrodes"),
            "figures_generated": manifest.get("figures_generated", True),
        }
    return provenance


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
    dimension_correlations: pd.DataFrame,
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
            "The complete statistical specification, including age/sex-adjusted partial "
            "Spearman equations and FDR scopes, is in [`METHODS.md`](../METHODS.md)."
        ),
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
    dimension_primary = dimension_correlations.loc[
        dimension_correlations["method"].eq("partial_spearman_age_sex")
    ]
    dimensions = [int(value) for value in config["dimension_sensitivity"]["embedding_dimensions"]]
    rejection_counts = {
        dimension: int(
            dimension_primary.loc[
                dimension_primary["embedding_dimension"].eq(dimension), "fdr_reject"
            ].sum()
        )
        for dimension in dimensions
    }
    lines.extend(
        [
            "## Embedding-dimension robustness analysis",
            "",
            (
                "Regular ordinal H, C, and F and Rényi entropy/complexity at alpha=0.5, "
                "0.9, 1.1, 2, and 5 were tested at D=3, 4, 5, and 6 with tau=1 for "
                "broadband and all six ordinal bands."
            ),
            (
                "Each embedding dimension is a separate 91-feature analysis block and has "
                "its own one-row-per-subject feature matrix. BH-FDR is controlled within "
                "each D across its 91 features and separately by correlation method."
            ),
            (
                "D=6 is the primary ordinal block; D=3, D=4, and D=5 are sensitivity "
                "blocks. These blocks are not statistically independent because they reuse "
                "the same participants and EEG signals, so selecting the best D after seeing "
                "the results remains exploratory."
            ),
            "",
            "Adjusted FDR rejections by separate D block:",
            "",
            *[
                f"- D={dimension}: {rejection_counts[dimension]} of 91"
                for dimension in dimensions
            ],
            "",
            (
                "Use `fdr_reject == True` and `p_fdr_bh < 0.05` in "
                "`metrics/dimension_sensitivity_correlations.csv` for corrected statistical "
                "significance within the indicated D block. Similar effect direction and "
                "magnitude across dimensions adds robustness, but is not a separate "
                "significance test."
            ),
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
    logger.info("Building D=3,4,5,6 ordinal sensitivity features at tau=1")
    (
        dimension_features,
        dimension_dictionary,
        dimension_electrode_features,
        dimension_electrode_order,
    ) = build_dimension_sensitivity_features(config, cohort)
    dimension_correlations = correlate_subject_features(
        dimension_features, dimension_dictionary, config
    )
    dimension_electrode_correlations = correlate_electrodes(
        dimension_electrode_features, dimension_dictionary, config
    )

    metrics_dir = output_dir / "metrics"
    # This legacy combined-D matrix could be mistaken for a single model input.
    # D-specific matrices below replace it and are the only dimension-analysis inputs.
    (metrics_dir / "dimension_sensitivity_analysis_dataset.csv").unlink(
        missing_ok=True
    )
    dimension_metrics_dir = metrics_dir / "dimensions"
    if dimension_metrics_dir.exists():
        shutil.rmtree(dimension_metrics_dir)
    _write_csv(cohort, metrics_dir / "moca_cohort.csv")
    _write_csv(dictionary, metrics_dir / "feature_dictionary.csv")
    _write_csv(subject_features, metrics_dir / "subject_features_long.csv")
    _write_csv(wide, metrics_dir / "analysis_dataset.csv")
    _write_csv(subject_correlations, metrics_dir / "subject_level_correlations.csv")
    _write_csv(
        subject_correlations.loc[
            subject_correlations["method"].eq("partial_spearman_age_sex")
            & subject_correlations["fdr_reject"]
        ],
        metrics_dir / "significant_primary_correlations.csv",
    )
    _write_csv(electrode_correlations, metrics_dir / "electrode_correlations.csv")
    _write_csv(
        dimension_dictionary,
        metrics_dir / "dimension_sensitivity_feature_dictionary.csv",
    )
    _write_csv(
        dimension_features,
        metrics_dir / "dimension_sensitivity_subject_features_long.csv",
    )
    _write_csv(
        dimension_correlations,
        metrics_dir / "dimension_sensitivity_correlations.csv",
    )
    _write_csv(
        dimension_correlations.loc[
            dimension_correlations["method"].eq("partial_spearman_age_sex")
            & dimension_correlations["fdr_reject"]
        ],
        metrics_dir / "dimension_sensitivity_significant_correlations.csv",
    )
    _write_csv(
        dimension_electrode_correlations,
        metrics_dir / "dimension_sensitivity_electrode_correlations.csv",
    )
    for dimension in config["dimension_sensitivity"]["embedding_dimensions"]:
        dimension = int(dimension)
        dimension_dir = metrics_dir / "dimensions" / f"D{dimension}"
        selected_dictionary = dimension_dictionary.loc[
            dimension_dictionary["embedding_dimension"].eq(dimension)
        ]
        feature_ids = set(selected_dictionary["feature_id"])
        selected_features = dimension_features.loc[
            dimension_features["feature_id"].isin(feature_ids)
        ]
        selected_correlations = dimension_correlations.loc[
            dimension_correlations["feature_id"].isin(feature_ids)
        ]
        selected_electrode_correlations = dimension_electrode_correlations.loc[
            dimension_electrode_correlations["feature_id"].isin(feature_ids)
        ]
        _write_csv(
            selected_dictionary,
            dimension_dir / "feature_dictionary.csv",
        )
        _write_csv(
            subject_feature_matrix(cohort, selected_features),
            dimension_dir / "analysis_dataset.csv",
        )
        _write_csv(
            selected_correlations,
            dimension_dir / "correlations.csv",
        )
        _write_csv(
            selected_correlations.loc[
                selected_correlations["method"].eq("partial_spearman_age_sex")
                & selected_correlations["fdr_reject"]
            ],
            dimension_dir / "significant_correlations.csv",
        )
        _write_csv(
            selected_electrode_correlations,
            dimension_dir / "electrode_correlations.csv",
        )
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
    sensitivity_figures = figures_dir / "dimension_sensitivity"
    if sensitivity_figures.exists():
        shutil.rmtree(sensitivity_figures)
    plot_dimension_sensitivity_heatmaps(
        dimension_correlations,
        sensitivity_figures / "adjusted_correlation_heatmaps.png",
        dpi,
    )
    plot_dimension_stability_lines(
        dimension_correlations,
        sensitivity_figures / "adjusted_effect_stability.png",
        dpi,
    )
    for dimension in config["dimension_sensitivity"]["embedding_dimensions"]:
        dimension = int(dimension)
        selected_dictionary = dimension_dictionary.loc[
            dimension_dictionary["embedding_dimension"].eq(dimension)
        ]
        feature_ids = set(selected_dictionary["feature_id"])
        selected_correlations = dimension_correlations.loc[
            dimension_correlations["feature_id"].isin(feature_ids)
        ]
        selected_features = dimension_features.loc[
            dimension_features["feature_id"].isin(feature_ids)
        ]
        dimension_family = f"ordinal_D{dimension}"
        plot_family_forest(
            selected_correlations,
            dimension_family,
            sensitivity_figures / f"D{dimension}_forest.png",
            dpi,
        )
        plot_family_heatmap(
            selected_correlations,
            dimension_family,
            sensitivity_figures / f"D{dimension}_adjusted_sensitivity_heatmap.png",
            dpi,
        )
        for quantity_set, quantity_dictionary in selected_dictionary.groupby(
            "quantity_set", sort=False
        ):
            quantity_ids = set(quantity_dictionary["feature_id"])
            plot_family_scatter_grid(
                selected_features.loc[
                    selected_features["feature_id"].isin(quantity_ids)
                ],
                selected_correlations.loc[
                    selected_correlations["feature_id"].isin(quantity_ids)
                ],
                quantity_dictionary,
                dimension_family,
                primary_group,
                sensitivity_figures
                / "scatter"
                / f"D{dimension}_{quantity_set}_moca_scatter_grid.png",
                dpi,
            )
    dimension_info = _topographic_info(config, dimension_electrode_order)
    plot_electrode_topomap_pages(
        dimension_electrode_correlations,
        dimension_dictionary,
        dimension_electrode_order,
        dimension_info,
        sensitivity_figures / "topomaps",
        dpi,
    )
    _write_report(
        output_dir / "REPORT.md",
        cohort,
        dictionary,
        subject_correlations,
        dimension_correlations,
        config,
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
        "dimension_sensitivity": {
            "embedding_dimensions": config["dimension_sensitivity"]["embedding_dimensions"],
            "delay_samples": config["dimension_sensitivity"]["delay_samples"],
            "n_features": len(dimension_dictionary),
            "n_subject_level_tests_per_method": len(dimension_dictionary),
            "n_features_per_dimension": {
                str(int(dimension)): int(
                    dimension_dictionary["embedding_dimension"].eq(int(dimension)).sum()
                )
                for dimension in config["dimension_sensitivity"]["embedding_dimensions"]
            },
            "fdr_scope": config["dimension_sensitivity"]["fdr_scope"],
            "n_primary_fdr_rejections": int(
                dimension_correlations.loc[
                    dimension_correlations["method"].eq("partial_spearman_age_sex"),
                    "fdr_reject",
                ].sum()
            ),
            "fdr_rejections_by_dimension": {
                str(int(dimension)): int(
                    dimension_correlations.loc[
                        dimension_correlations["method"].eq(
                            "partial_spearman_age_sex"
                        )
                        & dimension_correlations["embedding_dimension"].eq(
                            int(dimension)
                        ),
                        "fdr_reject",
                    ].sum()
                )
                for dimension in config["dimension_sensitivity"]["embedding_dimensions"]
            },
            "n_electrode_tests": len(dimension_electrode_correlations),
            "feature_matrix_policy": (
                "One separate 91-feature, one-row-per-subject matrix for each D; "
                "embedding dimensions are never concatenated into one model matrix."
            ),
        },
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
