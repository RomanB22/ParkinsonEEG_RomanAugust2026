"""End-to-end transparent PD versus Control exploration pipeline."""

from __future__ import annotations

import copy
import json
import logging
import platform
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any

from core.runtime import configure_runtime
from core.frequency_bands import CANONICAL_BOUT_BAND_NAMES

configure_runtime()

import matplotlib
import numpy as np
import pandas as pd
import scipy
import sklearn

from .features import (
    LEAKAGE_EXCLUSIONS,
    build_feature_table,
    discover_completed_sweeps,
)
from .modeling import (
    average_repeated_predictions,
    bootstrap_auc_differences,
    bootstrap_performance,
    fit_final_models,
    run_nested_validation,
)
from .matching import (
    apply_precomputed_control_pd_pairs,
    match_control_pd_pairs,
    remove_demographic_predictors,
)
from .plots import (
    plot_calibration,
    plot_coefficient_stability,
    plot_confusion_matrices,
    plot_demographic_matching,
    plot_entropy_complexity_plane,
    plot_feature_correlations,
    plot_feature_distributions,
    plot_features_vs_age,
    plot_model_performance,
    plot_roc_and_precision_recall,
    plot_sweep_sensitivity,
)


def load_exploration_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as stream:
        config = json.load(stream)
    required = {
        "input",
        "output_dir",
        "primary_ordinal_parameters",
        "ordinal_sweep",
        "ordinal_model_bands",
        "candidate_features",
        "psd_log_ratio",
        "models",
        "validation",
        "plots",
        "demographic_matching",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Missing exploration config sections: {missing}")
    validation = config["validation"]
    if int(validation["outer_folds"]) < 2 or int(validation["inner_folds"]) < 2:
        raise ValueError("Inner and outer cross-validation require at least two folds")
    if int(validation["outer_repeats"]) < 1:
        raise ValueError("outer_repeats must be positive")
    if int(validation["bootstrap_resamples"]) < 20:
        raise ValueError("bootstrap_resamples must be at least 20")
    primary_ordinal = config["primary_ordinal_parameters"]
    if int(primary_ordinal.get("delay_samples", -1)) != 1:
        raise ValueError("The primary ordinal analysis must use tau=1")
    ordinal_sweep = config["ordinal_sweep"]
    if ordinal_sweep.get("expected_dimensions") != [3, 4, 5, 6]:
        raise ValueError("Ordinal sensitivity must prespecify D=3,4,5,6")
    if ordinal_sweep.get("expected_delays") != [1]:
        raise ValueError("Ordinal sensitivity must use only tau=1")
    threshold = float(validation["classification_threshold"])
    if not 0.0 < threshold < 1.0:
        raise ValueError("classification_threshold must lie between zero and one")
    if str(validation.get("threshold_policy", "fixed")) not in {
        "fixed",
        "inner_youden",
    }:
        raise ValueError("threshold_policy must be fixed or inner_youden")
    candidate = config["candidate_features"]
    if candidate.get("renyi_metrics") != [
        "renyi_entropy_alpha_0_1",
        "renyi_complexity_alpha_0_1",
        "renyi_entropy_alpha_10",
        "renyi_complexity_alpha_10",
    ]:
        raise ValueError("Exploration Rényi predictors must remain the alpha endpoints")
    if candidate.get("bout_bands") != list(CANONICAL_BOUT_BAND_NAMES):
        raise ValueError("Exploration bout predictors must use the canonical bands")
    matching = config["demographic_matching"]
    if matching.get("exact_variables") != ["sex_male"]:
        raise ValueError("Demographic matching must require exact sex")
    if matching.get("distance_variable") != "age_years":
        raise ValueError("Demographic matching distance must use age")
    if float(matching.get("maximum_age_difference_years", 0)) <= 0:
        raise ValueError("Demographic matching requires a positive age caliper")
    return config


def _configure_logger(output_dir: Path, overwrite: bool) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("exploration")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (
        logging.StreamHandler(),
        logging.FileHandler(
            output_dir / "exploration.log", mode="w" if overwrite else "a"
        ),
    ):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def _write_csv(table: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False, float_format="%.17g")


def _write_revision_report(
    path: Path,
    performance: pd.DataFrame,
    auc_vs_demographics: pd.DataFrame | None,
    auc_vs_psd: pd.DataFrame,
    permutation_results: pd.DataFrame,
    models: dict[str, dict[str, Any]],
    *,
    cohort_mode: str = "full",
) -> None:
    """Write a concise, uncertainty-aware audit of the revised model set."""
    auc = (
        performance.loc[performance["metric"].eq("roc_auc")]
        .set_index("model")
        .sort_values("estimate", ascending=False)
    )
    vs_demo = (
        auc_vs_demographics.set_index("model")
        if auc_vs_demographics is not None and not auc_vs_demographics.empty
        else pd.DataFrame()
    )
    vs_psd = auc_vs_psd.set_index("model")
    permutation = (
        permutation_results.set_index("model")
        if not permutation_results.empty and "model" in permutation_results
        else pd.DataFrame()
    )

    def interval(row: pd.Series) -> str:
        return f"{row['estimate']:.3f} [{row['ci_lower']:.3f}, {row['ci_upper']:.3f}]"

    def difference(table: pd.DataFrame, model: str) -> str:
        if table.empty:
            return "not included"
        if model not in table.index:
            return "reference"
        row = table.loc[model]
        marker = "†" if float(row["ci_lower"]) > 0 or float(row["ci_upper"]) < 0 else ""
        return (
            f"{float(row['auc_difference']):+.3f} "
            f"[{float(row['ci_lower']):+.3f}, {float(row['ci_upper']):+.3f}]{marker}"
        )

    lines = [
        "# Exploration model revision",
        "",
        f"Cohort mode: **{cohort_mode}**. This report compares the newly available EEG "
        "feature blocks using the same "
        "repeated nested cross-validation splits and ridge-logistic model family. "
        "Values are subject-level out-of-fold ROC AUC with stratified-bootstrap 95% "
        "intervals. A dagger (†) marks a paired AUC-difference interval that excludes zero.",
        "",
        "## Model comparison",
        "",
        "| Model | Predictors | ROC AUC [95% CI] | ΔAUC vs demographics | ΔAUC vs PSD |",
        "|---|---:|---:|---:|---:|",
    ]
    for model, row in auc.iterrows():
        lines.append(
            f"| {models[model]['label']} | {len(models[model]['features'])} | "
            f"{interval(row)} | {difference(vs_demo, model)} | "
            f"{difference(vs_psd, model)} |"
        )

    eeg_models = [
        model
        for model in auc.index
        if model not in {"clinical_extension", "demographics"}
    ]
    best_eeg = max(eeg_models, key=lambda model: float(auc.loc[model, "estimate"]))
    improved_vs_psd = [
        model
        for model in eeg_models
        if model in vs_psd.index and float(vs_psd.loc[model, "ci_lower"]) > 0
    ]
    best_recommendation = (
        f"- {models[best_eeg]['label']} remains the highest-scoring EEG model; retain it as the benchmark."
        if best_eeg == "psd_adjusted"
        else f"- Retain {models[best_eeg]['label']} as the highest-scoring EEG candidate, "
        "without claiming superiority unless its paired interval versus PSD is positive."
    )
    baseline_recommendation = (
        "- Keep demographics as the baseline and PSD + demographics as the EEG benchmark."
        if cohort_mode == "full"
        else "- Age and sex are excluded from every matched-cohort model; PSD is the EEG benchmark."
    )
    lines.extend(
        [
            "",
            "## Conservative interpretation",
            "",
            f"The highest internally validated EEG-only model is **{models[best_eeg]['label']}** "
            f"(AUC {interval(auc.loc[best_eeg])}).",
            "",
        ]
    )
    if improved_vs_psd:
        names = ", ".join(f"**{models[model]['label']}**" for model in improved_vs_psd)
        lines.append(
            f"The following additions have a paired 95% interval above the PSD benchmark: {names}."
        )
    else:
        lines.append(
            "No added EEG feature block has a paired 95% AUC-difference interval wholly "
            "above the PSD benchmark. Any higher point estimate should therefore be treated "
            "as promising but uncertain, rather than a demonstrated improvement."
        )
    lines.extend(
        [
            "",
            "Rényi models use only α=0.1 and α=10 as sensitivity endpoints; the intermediate "
            "α values were excluded because they are highly redundant with H, C, F and with "
            "one another. Embedding dimensions remain separate sensitivity analyses and are "
            "never concatenated into a single feature vector.",
            "",
            "Bout dynamics, within-bout ordinal quantities, typical-bout shapes, and the "
            "aperiodic exponent depend on spectral parameterization or its bout threshold. "
            "Their results remain fit-QC-sensitive even when their cross-validation score is high.",
            "",
            "The clinical-extension model includes MOCA and is not an EEG-only diagnostic model. "
            "All results are internal to this case-control cohort and require external validation.",
            "",
            "## Recommended reporting set",
            "",
            baseline_recommendation,
            best_recommendation,
            "- Report within-bout ordinal, bout dynamics, typical-bout shape, aperiodic, and "
            "Rényi models as prespecified sensitivity blocks rather than combining every quantity.",
            "- Keep the clinical extension separate because MOCA changes the intended use and "
            "is not an EEG feature.",
            "- Do not select a model from the ordinal embedding sweep; use it only to assess "
            "whether conclusions depend on D at fixed tau=1.",
            "",
            "## Permutation tests",
            "",
        ]
    )
    if permutation.empty:
        lines.append("Permutation tests were skipped for this run.")
    else:
        p_column = next(
            (
                column
                for column in permutation.columns
                if column in {"permutation_p", "permutation_p_value"}
                or "p_value" in column
            ),
            None,
        )
        if p_column is None:
            lines.append("See `metrics/permutation_tests.csv` for the complete chance benchmark.")
        else:
            lines.extend(["| Model | Permutation p |", "|---|---:|"])
            for model in auc.index:
                if model in permutation.index:
                    lines.append(
                        f"| {models[model]['label']} | "
                        f"{float(permutation.loc[model, p_column]):.4g} |"
                    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _feature_summary(table: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows = []
    for group, selected in table.groupby("group", sort=False):
        for feature in features:
            values = selected[feature].to_numpy(dtype=float)
            rows.append(
                {
                    "group": group,
                    "feature": feature,
                    "n_subjects": int(len(values)),
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values, ddof=1)),
                    "median": float(np.median(values)),
                    "q25": float(np.quantile(values, 0.25)),
                    "q75": float(np.quantile(values, 0.75)),
                }
            )
    return pd.DataFrame.from_records(rows)


def _coefficient_summary(coefficients: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, feature), selected in coefficients.groupby(
        ["model", "feature"], sort=False
    ):
        values = selected["coefficient_per_sd"].to_numpy(dtype=float)
        positive_fraction = float(np.mean(values > 0.0))
        rows.append(
            {
                "model": model,
                "model_label": selected["model_label"].iloc[0],
                "model_role": selected["model_role"].iloc[0],
                "feature": feature,
                "n_outer_fits": int(len(values)),
                "coefficient_median": float(np.median(values)),
                "coefficient_ci_lower": float(np.quantile(values, 0.025)),
                "coefficient_ci_upper": float(np.quantile(values, 0.975)),
                "positive_fraction": positive_fraction,
                "same_sign_fraction": max(positive_fraction, 1.0 - positive_fraction),
            }
        )
    return pd.DataFrame.from_records(rows)


def _sweep_status(
    config: dict[str, Any], completed: list[dict[str, Any]]
) -> pd.DataFrame:
    completed_lookup = {
        (item["embedding_dimension"], item["delay_samples"]): item["path"]
        for item in completed
    }
    rows = []
    for dimension in config["ordinal_sweep"]["expected_dimensions"]:
        for delay in config["ordinal_sweep"]["expected_delays"]:
            key = (int(dimension), int(delay))
            rows.append(
                {
                    "embedding_dimension": key[0],
                    "delay_samples": key[1],
                    "complete": key in completed_lookup,
                    "metrics_path": completed_lookup.get(key, ""),
                }
            )
    return pd.DataFrame.from_records(rows)


def _run_sweep_sensitivity(
    base_table: pd.DataFrame,
    completed: list[dict[str, Any]],
    config: dict[str, Any],
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    performance_tables = []
    prediction_tables = []
    coefficient_tables = []
    feature_names = [
        "ordinal_global_entropy",
        "ordinal_global_complexity",
        "ordinal_global_fisher_information",
    ]
    for index, item in enumerate(completed, start=1):
        dimension = int(item["embedding_dimension"])
        delay = int(item["delay_samples"])
        label = f"D={dimension}, tau={delay}"
        logger.info(
            "Ordinal sweep sensitivity [%d/%d] | %s",
            index,
            len(completed),
            label,
        )
        ordinal = pd.read_csv(item["path"])
        required = {
            "subject_id",
            "entropy",
            "complexity",
            "fisher_information",
        }
        missing = sorted(required - set(ordinal))
        if missing:
            raise ValueError(f"{item['path']} is missing columns: {missing}")
        ordinal = ordinal.rename(
            columns={
                "entropy": feature_names[0],
                "complexity": feature_names[1],
                "fisher_information": feature_names[2],
            }
        )
        base_columns = ["subject_id", "target_pd"]
        if "cv_group" in base_table:
            base_columns.append("cv_group")
        selected = base_table[base_columns].merge(
            ordinal[["subject_id", *feature_names]],
            on="subject_id",
            how="left",
            validate="one_to_one",
        )
        if selected[feature_names].isna().any().any():
            raise ValueError(f"{label} does not contain every modeling subject")
        model_name = f"D{dimension}_tau{delay}"
        models = {
            model_name: {
                "label": label,
                "role": "ordinal_parameter_sensitivity",
                "features": feature_names,
            }
        }
        predictions, _, coefficients = run_nested_validation(
            selected, models, config["validation"]
        )
        averaged = average_repeated_predictions(predictions)
        performance = bootstrap_performance(
            averaged,
            n_resamples=int(config["validation"]["bootstrap_resamples"]),
            seed=int(config["validation"]["random_seed"]) + dimension * 100 + delay,
        )
        for table in (performance, averaged, coefficients):
            table.insert(0, "delay_samples", delay)
            table.insert(0, "embedding_dimension", dimension)
        performance_tables.append(performance)
        prediction_tables.append(averaged)
        coefficient_tables.append(coefficients)
    empty = pd.DataFrame()
    return (
        pd.concat(performance_tables, ignore_index=True) if performance_tables else empty,
        pd.concat(prediction_tables, ignore_index=True) if prediction_tables else empty,
        pd.concat(coefficient_tables, ignore_index=True) if coefficient_tables else empty,
    )


def run_analysis(
    config_path: str | Path,
    *,
    output_dir_override: str | Path | None = None,
    overwrite: bool = False,
    quick: bool = False,
    skip_sweep: bool = False,
    skip_permutations: bool = False,
    matched_demographics: bool = False,
) -> dict[str, Any]:
    """Run feature assembly, nested validation, final descriptive fits, and plots."""
    config_path = Path(config_path)
    config = load_exploration_config(config_path)
    config = copy.deepcopy(config)
    if output_dir_override is not None:
        config["output_dir"] = str(output_dir_override)
    elif matched_demographics:
        config["output_dir"] = str(config["demographic_matching"]["output_dir"])
    if quick:
        config["validation"]["outer_repeats"] = 2
        config["validation"]["bootstrap_resamples"] = 100
        config["validation"]["permutation_resamples"] = 20
    if skip_permutations:
        config["validation"]["permutation_resamples"] = 0
    output_dir = Path(config["output_dir"])
    sentinel = output_dir / "metrics" / "model_performance.csv"
    if sentinel.exists() and not overwrite:
        raise FileExistsError(
            f"Exploration outputs exist at {sentinel}; rerun with --overwrite"
        )
    logger = _configure_logger(output_dir, overwrite)
    logger.info("Building one-row-per-subject feature table")
    feature_table, provenance = build_feature_table(config)
    pair_table = balance_table = pd.DataFrame()
    if matched_demographics:
        matching_config = config["demographic_matching"]
        precomputed_pairs = matching_config.get("precomputed_pairs_file")
        precomputed_balance = matching_config.get("precomputed_balance_file")
        if bool(precomputed_pairs) != bool(precomputed_balance):
            raise ValueError(
                "Both precomputed_pairs_file and precomputed_balance_file are required"
            )
        if precomputed_pairs:
            feature_table, pair_table, balance_table = (
                apply_precomputed_control_pd_pairs(
                    feature_table,
                    pd.read_csv(precomputed_pairs),
                    pd.read_csv(precomputed_balance),
                    maximum_age_difference_years=float(
                        matching_config["maximum_age_difference_years"]
                    ),
                )
            )
            logger.info("Using validated canonical precomputed demographic pairs")
        else:
            feature_table, pair_table, balance_table = match_control_pd_pairs(
                feature_table,
                maximum_age_difference_years=float(
                    matching_config["maximum_age_difference_years"]
                ),
            )
        models = remove_demographic_predictors(config["models"])
        logger.info(
            "Matched sensitivity cohort | pairs=%d | maximum_age_gap=%.1f years",
            len(pair_table),
            float(pair_table["absolute_age_difference_years"].max()),
        )
    else:
        models = config["models"]
    all_features = list(
        dict.fromkeys(
            feature
            for specification in models.values()
            for feature in specification["features"]
        )
    )
    logger.info(
        "Feature audit passed | subjects=%d | PD=%d | Control=%d | candidate_features=%d",
        len(feature_table),
        int(feature_table["target_pd"].sum()),
        int((1 - feature_table["target_pd"]).sum()),
        len(all_features),
    )

    features_dir = output_dir / "features"
    metrics_dir = output_dir / "metrics"
    cv_dir = output_dir / "cross_validation"
    predictions_dir = output_dir / "predictions"
    models_dir = output_dir / "models"
    figures_dir = output_dir / "figures"
    _write_csv(feature_table, features_dir / "subject_modeling_table.csv")
    _write_csv(provenance, features_dir / "feature_provenance.csv")
    if matched_demographics:
        _write_csv(pair_table, features_dir / "demographic_match_pairs.csv")
        _write_csv(balance_table, features_dir / "demographic_balance.csv")
    _write_csv(
        _feature_summary(feature_table, all_features),
        features_dir / "feature_group_summary.csv",
    )
    _write_csv(
        feature_table[all_features].corr(method="spearman").rename_axis("feature").reset_index(),
        features_dir / "feature_spearman_correlations.csv",
    )

    logger.info(
        "Running repeated nested cross-validation | models=%d | outer=%dx%d | inner=%d",
        len(models),
        int(config["validation"]["outer_folds"]),
        int(config["validation"]["outer_repeats"]),
        int(config["validation"]["inner_folds"]),
    )
    predictions, fold_metrics, fold_coefficients = run_nested_validation(
        feature_table, models, config["validation"]
    )
    averaged_predictions = average_repeated_predictions(predictions)
    performance = bootstrap_performance(
        averaged_predictions,
        n_resamples=int(config["validation"]["bootstrap_resamples"]),
        seed=int(config["validation"]["random_seed"]),
    )
    auc_differences = (
        bootstrap_auc_differences(
            averaged_predictions,
            reference_model="demographics",
            n_resamples=int(config["validation"]["bootstrap_resamples"]),
            seed=int(config["validation"]["random_seed"]) + 1,
        )
        if "demographics" in models
        else pd.DataFrame(
            columns=[
                "model",
                "reference_model",
                "auc_difference",
                "ci_lower",
                "ci_upper",
                "bootstrap_resamples",
            ]
        )
    )
    auc_differences_vs_psd = bootstrap_auc_differences(
        averaged_predictions,
        reference_model="psd_adjusted",
        n_resamples=int(config["validation"]["bootstrap_resamples"]),
        seed=int(config["validation"]["random_seed"]) + 2,
    )
    coefficient_summary = _coefficient_summary(fold_coefficients)
    first_model = next(iter(models))
    fold_assignment_columns = ["repeat", "fold", "subject_id", "target_pd"]
    if "cv_group" in predictions:
        fold_assignment_columns.append("cv_group")
    fold_assignments = predictions.loc[
        predictions["model"].eq(first_model),
        fold_assignment_columns,
    ].sort_values(["repeat", "fold", "subject_id"])
    _write_csv(predictions, predictions_dir / "repeated_outer_predictions.csv")
    _write_csv(averaged_predictions, predictions_dir / "subject_out_of_fold_predictions.csv")
    _write_csv(fold_assignments, cv_dir / "outer_fold_assignments.csv")
    _write_csv(fold_metrics, cv_dir / "outer_fold_metrics.csv")
    _write_csv(fold_coefficients, cv_dir / "outer_fold_coefficients.csv")
    _write_csv(performance, metrics_dir / "model_performance.csv")
    _write_csv(auc_differences, metrics_dir / "auc_differences_vs_demographics.csv")
    _write_csv(auc_differences_vs_psd, metrics_dir / "auc_differences_vs_psd.csv")
    _write_csv(coefficient_summary, metrics_dir / "coefficient_stability.csv")

    logger.info("Fitting descriptive final models and permutation tests")
    final_coefficients, permutation_results = fit_final_models(
        feature_table,
        models,
        config["validation"],
        models_dir,
    )
    _write_csv(final_coefficients, metrics_dir / "final_model_coefficients.csv")
    _write_csv(permutation_results, metrics_dir / "permutation_tests.csv")
    _write_revision_report(
        output_dir / "MODEL_REVISION.md",
        performance,
        auc_differences,
        auc_differences_vs_psd,
        permutation_results,
        models,
        cohort_mode="demographically matched" if matched_demographics else "full",
    )

    completed_sweeps = discover_completed_sweeps(config)
    sweep_status = _sweep_status(config, completed_sweeps)
    _write_csv(sweep_status, metrics_dir / "ordinal_sweep_status.csv")
    sweep_performance = sweep_predictions = sweep_coefficients = pd.DataFrame()
    if completed_sweeps and not skip_sweep:
        sweep_performance, sweep_predictions, sweep_coefficients = _run_sweep_sensitivity(
            feature_table,
            completed_sweeps,
            config,
            logger,
        )
        _write_csv(sweep_performance, metrics_dir / "ordinal_sweep_performance.csv")
        _write_csv(sweep_predictions, predictions_dir / "ordinal_sweep_predictions.csv")
        _write_csv(sweep_coefficients, cv_dir / "ordinal_sweep_coefficients.csv")
    elif skip_sweep:
        logger.info("Skipping ordinal parameter sensitivity by request")
    else:
        logger.info("No completed ordinal parameter sweep outputs were found")

    logger.info("Creating documented feature and validation figures")
    group_order = [
        group
        for group in config["plots"]["group_order"]
        if group in set(feature_table["group"])
    ]
    colors = {
        group: str(config["plots"]["group_colors"].get(group, "0.4"))
        for group in group_order
    }
    model_order = [
        model
        for model in config["plots"]["model_order"]
        if model in models
    ]
    dpi = int(config["plots"]["dpi"])
    plot_feature_distributions(
        feature_table,
        all_features,
        group_order,
        colors,
        figures_dir / "features" / "candidate_feature_distributions.png",
        dpi,
    )
    plot_features_vs_age(
        feature_table,
        all_features,
        group_order,
        colors,
        figures_dir / "features" / "versus_age",
        dpi,
    )
    if matched_demographics:
        plot_demographic_matching(
            pair_table,
            balance_table,
            figures_dir / "features" / "demographic_matching.png",
            dpi,
        )
    plot_entropy_complexity_plane(
        feature_table,
        group_order,
        colors,
        figures_dir / "features" / "ordinal_entropy_complexity_plane.png",
        dpi,
    )
    plot_feature_correlations(
        feature_table,
        all_features,
        figures_dir / "features" / "feature_correlation_heatmap.png",
        dpi,
    )
    plot_model_performance(
        performance,
        model_order,
        figures_dir / "validation" / "model_performance_comparison.png",
        dpi,
    )
    plot_roc_and_precision_recall(
        averaged_predictions,
        model_order,
        figures_dir / "validation" / "roc_and_precision_recall_curves.png",
        dpi,
    )
    plot_calibration(
        averaged_predictions,
        model_order,
        figures_dir / "validation" / "calibration_and_prediction_distributions.png",
        dpi,
    )
    plot_confusion_matrices(
        averaged_predictions,
        model_order,
        figures_dir / "validation" / "confusion_matrices.png",
        dpi,
    )
    for model_name in (
        "psd_adjusted",
        "ordinal_adjusted",
        "ordinal_renyi_adjusted",
        "bout_dynamics_adjusted",
        "bout_ordinal_adjusted",
        "typical_bout_shape_adjusted",
        "multimodal_compact",
        "ordinal_psd_adjusted",
    ):
        plot_coefficient_stability(
            fold_coefficients,
            model_name,
            figures_dir / "explainability" / f"{model_name}_coefficient_stability.png",
            dpi,
        )
    if not sweep_performance.empty:
        plot_sweep_sensitivity(
            sweep_performance.loc[sweep_performance["metric"].eq("roc_auc")],
            figures_dir / "sensitivity" / "ordinal_parameter_sweep_roc_auc.png",
            dpi,
        )

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config_file": str(config_path.resolve()),
        "analysis_config": config,
        "runtime_options": {
            "quick": bool(quick),
            "skip_sweep": bool(skip_sweep),
            "skip_permutations": bool(skip_permutations),
            "overwrite": bool(overwrite),
            "matched_demographics": bool(matched_demographics),
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "matplotlib": matplotlib.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": version("joblib"),
        },
        "n_subjects": int(len(feature_table)),
        "group_counts": feature_table["group"].value_counts().to_dict(),
        "cohort_mode": "demographically_matched" if matched_demographics else "full",
        "demographic_matching": (
            {
                "n_pairs": int(len(pair_table)),
                "exact_variables": config["demographic_matching"]["exact_variables"],
                "distance_variable": config["demographic_matching"]["distance_variable"],
                "algorithm": config["demographic_matching"]["algorithm"],
                "maximum_age_difference_years": float(
                    config["demographic_matching"]["maximum_age_difference_years"]
                ),
                "observed_maximum_age_difference_years": float(
                    pair_table["absolute_age_difference_years"].max()
                ),
                "fold_policy": "Matched pairs remain together in every outer and inner fold.",
                "bootstrap_policy": "Matched pairs are resampled together.",
                "permutation_policy": "PD/Control labels are permuted within matched pairs.",
            }
            if matched_demographics
            else None
        ),
        "outcome": "target_pd: PD=1, Control=0",
        "primary_model": "ordinal_adjusted",
        "primary_ordinal_parameters": config["primary_ordinal_parameters"],
        "leakage_exclusions": LEAKAGE_EXCLUSIONS,
        "feature_policy": (
            "Every model uses one row per subject. Electrode and bout observations are "
            "aggregated within subject before modeling. PSD predictors are prespecified "
            "log2 ratios against gamma. Rényi "
            "models retain only the prespecified low/high Rényi endpoints because intermediate "
            "alphas are extremely redundant. Typical-bout curves are reduced to peak, "
            "half-height width, temporal asymmetry, and relative-phase consistency."
        ),
        "validation_policy": (
            "All scaling and ridge-C selection occur within repeated nested stratified "
            "cross-validation. Reported discrimination uses averaged out-of-fold "
            "predictions; final all-subject fits are descriptive deployment artifacts, "
            "not independent validation."
        ),
        "ordinal_sweep_completed": int(sweep_status["complete"].sum()),
        "ordinal_sweep_expected": int(len(sweep_status)),
        "limitations": (
            "This is internal case-control discrimination in one cohort. It does not "
            "establish clinical diagnostic utility and requires external validation."
        ),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    logger.info("Exploration analysis completed | output=%s", output_dir)
    return manifest
