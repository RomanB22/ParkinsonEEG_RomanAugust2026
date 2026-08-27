"""MOCA sensitivity using fit-QC bout and within-bout ordinal summaries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from analyses.behavioral.features import (
    BAND_LABELS,
    METRIC_LABELS,
    METRIC_UNITS,
    load_cohort,
)
from analyses.behavioral.plots import (
    plot_family_forest,
    plot_family_heatmap,
    plot_family_scatter_grid,
)
from analyses.behavioral.statistics import correlate_subject_features


def _write_csv(table: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False, float_format="%.17g")


def _configured_bout_bands(config: dict[str, Any]) -> list[str]:
    """Return the canonical, non-overlapping bands used for fit-QC inference."""
    bands = [str(value) for value in config["features"]["bout_bands"]]
    if not bands or len(bands) != len(set(bands)):
        raise ValueError("features.bout_bands must contain unique canonical bands")
    return bands


def _append_family(
    rows: list[pd.DataFrame],
    dictionary: list[dict[str, Any]],
    table: pd.DataFrame,
    *,
    metrics: list[str],
    bands: list[str],
    family: str,
    domain: str,
    prefix: str,
    source_file: str | Path,
) -> None:
    for band in bands:
        selected = table.loc[table["band"].eq(band)]
        if selected.empty or selected["subject_id"].duplicated().any():
            raise ValueError(f"{source_file}: invalid subject rows for {band}")
        for metric in metrics:
            feature_id = f"{prefix}_{band}_{metric}"
            rows.append(
                selected[["subject_id", metric]]
                .rename(columns={metric: "value"})
                .assign(feature_id=feature_id)
            )
            band_label = BAND_LABELS.get(band, band.replace("_", " ").title())
            metric_label = METRIC_LABELS[metric]
            label = (
                f"Fit-QC {band_label} within-bout {metric_label}"
                if domain == "bout_ordinal"
                else f"Fit-QC {band_label} {metric_label}"
            )
            dictionary.append(
                {
                    "feature_id": feature_id,
                    "family": family,
                    "domain": domain,
                    "band": band,
                    "metric": metric,
                    "feature_label": label,
                    "unit": METRIC_UNITS[metric],
                    "source_file": str(Path(source_file).resolve()),
                    "analysis_level": (
                        "subject mean across fit-QC electrodes; subject requires "
                        "at least 80% passing shared electrodes"
                    ),
                }
            )


def run_behavioral_fit_qc_sensitivity(
    config_path: str | Path = "config/analyses/behavioral.json",
    *,
    scale_free_qc_subject_file: str | Path = (
        "outputs/full/scale_free/metrics/subject_band_metrics_fit_qc.csv"
    ),
    bout_ordinal_qc_subject_file: str | Path = (
        "outputs/full/bouts/metrics/subject_band_metrics_fit_qc.csv"
    ),
) -> dict[str, Any]:
    """Calculate PD-only MOCA associations in the qualified fit-QC cohort."""
    config_path = Path(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    cohort = load_cohort(config)
    scale_path = Path(scale_free_qc_subject_file)
    ordinal_path = Path(bout_ordinal_qc_subject_file)
    scale = pd.read_csv(scale_path)
    ordinal = pd.read_csv(ordinal_path)
    bands = _configured_bout_bands(config)
    expected_source_bands = set(bands)
    bout_metrics = [str(value) for value in config["features"]["bout_properties"]]
    ordinal_metrics = [
        str(value) for value in config["features"]["bout_ordinal_metrics"]
    ]
    for name, table, metrics in (
        ("fit-QC bout properties", scale, bout_metrics),
        ("fit-QC within-bout ordinal", ordinal, ordinal_metrics),
    ):
        required = {
            "subject_id",
            "group",
            "band",
            "subject_fit_qc_pass",
            "n_fit_qc_electrodes",
            *metrics,
        }
        missing = sorted(required - set(table.columns))
        if missing:
            raise ValueError(f"{name} table is missing columns: {missing}")
        if not table["subject_fit_qc_pass"].all():
            raise ValueError(f"{name} includes nonqualified subjects")
        if set(table["band"].astype(str)) != expected_source_bands:
            raise ValueError(
                f"{name} must contain exactly the configured canonical bout bands: "
                f"{sorted(expected_source_bands)}"
            )
    scale_subjects = set(scale["subject_id"].astype(str))
    ordinal_subjects = set(ordinal["subject_id"].astype(str))
    if scale_subjects != ordinal_subjects:
        raise ValueError("Fit-QC bout sources do not use the same qualified subjects")
    expected_groups = cohort[["subject_id", "group"]]
    for name, table in (
        ("fit-QC bout properties", scale),
        ("fit-QC within-bout ordinal", ordinal),
    ):
        source_groups = table[["subject_id", "group"]].drop_duplicates()
        if source_groups["subject_id"].duplicated().any():
            raise ValueError(f"{name} has inconsistent group labels")
        compared = source_groups.merge(
            expected_groups,
            on="subject_id",
            how="left",
            suffixes=("_source", "_cohort"),
            validate="one_to_one",
        )
        if not compared["group_source"].eq(compared["group_cohort"]).all():
            raise ValueError(f"{name} and cohort group labels disagree")

    rows: list[pd.DataFrame] = []
    dictionary_rows: list[dict[str, Any]] = []
    _append_family(
        rows,
        dictionary_rows,
        scale,
        metrics=bout_metrics,
        bands=bands,
        family="bout_properties_fit_qc",
        domain="bout",
        prefix="bout_fit_qc",
        source_file=scale_path,
    )
    _append_family(
        rows,
        dictionary_rows,
        ordinal,
        metrics=ordinal_metrics,
        bands=bands,
        family="bout_ordinal_fit_qc",
        domain="bout_ordinal",
        prefix="bout_ordinal_fit_qc",
        source_file=ordinal_path,
    )
    features = pd.concat(rows, ignore_index=True)
    dictionary = pd.DataFrame.from_records(dictionary_rows)
    if features.duplicated(["subject_id", "feature_id"]).any():
        raise ValueError("Fit-QC behavioral features contain duplicate rows")
    features = features.merge(
        cohort[["subject_id", "group", "moca", "age_years", "gender", "sex_male"]],
        on="subject_id",
        how="left",
        validate="many_to_one",
    ).merge(dictionary, on="feature_id", how="left", validate="many_to_one")
    correlations = correlate_subject_features(features, dictionary, config)

    output_root = Path(config["output_dir"])
    metrics_dir = output_root / "metrics" / "fit_qc_sensitivity"
    figures_dir = output_root / "figures" / "fit_qc_sensitivity"
    _write_csv(dictionary, metrics_dir / "feature_dictionary.csv")
    _write_csv(features, metrics_dir / "subject_features_long.csv")
    _write_csv(correlations, metrics_dir / "subject_level_correlations.csv")
    primary_group = str(config["analysis"]["primary_group"])
    for family in ("bout_properties_fit_qc", "bout_ordinal_fit_qc"):
        plot_family_forest(
            correlations,
            family,
            figures_dir / f"{family}_forest.png",
            int(config["plots"]["dpi"]),
        )
        plot_family_heatmap(
            correlations,
            family,
            figures_dir / f"{family}_adjusted_sensitivity_heatmap.png",
            int(config["plots"]["dpi"]),
        )
        plot_family_scatter_grid(
            features,
            correlations,
            dictionary,
            family,
            primary_group,
            figures_dir / f"{family}_moca_scatter_grid.png",
            int(config["plots"]["dpi"]),
        )

    adjusted = correlations.loc[
        correlations["method"].eq("partial_spearman_age_sex")
    ]
    significant = adjusted.loc[adjusted["fdr_reject"]]
    result_lines = [
        (
            f"- {row.feature_label}: rho={row.estimate:.3f}, "
            f"95% CI [{row.ci_lower:.3f}, {row.ci_upper:.3f}], "
            f"p={row.p_value:.4g}, BH q={row.p_fdr_bh:.4g}."
        )
        for row in significant.itertuples(index=False)
    ]
    pd_subjects = features.loc[features["group"].eq(primary_group), "subject_id"].nunique()
    report = "\n".join(
        [
            "# Fit-QC bout associations with MOCA",
            "",
            f"Cohort: {primary_group}, n={pd_subjects}; each subject has at least "
            "48/60 passing specparam fits.",
            "",
            "These are sensitivity analyses using only QC-passing electrodes. "
            "They do not replace the all-electrode provenance analysis. Partial "
            "Spearman correlations adjust for age and sex, with BH FDR controlled "
            "separately within the fit-QC bout-property and fit-QC within-bout "
            "ordinal families using only canonical non-overlapping bands.",
            "",
            f"FDR-significant adjusted associations: {len(significant)}/{len(adjusted)}.",
            *result_lines,
            "",
        ]
    )
    (output_root / "FIT_QC_SENSITIVITY.md").write_text(report, encoding="utf-8")
    payload = {
        "config_file": str(config_path.resolve()),
        "n_qualified_subjects": int(len(scale_subjects)),
        "n_qualified_pd_subjects": int(pd_subjects),
        "n_features": int(len(dictionary)),
        "n_adjusted_fdr_significant": int(len(significant)),
        "families": dictionary["family"].value_counts().to_dict(),
        "policy": (
            "PD-only MOCA fit-QC sensitivity; partial Spearman adjusted for age "
            "and sex; BH FDR within each fit-QC feature family."
        ),
    }
    (output_root / "fit_qc_sensitivity_manifest.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    return payload
