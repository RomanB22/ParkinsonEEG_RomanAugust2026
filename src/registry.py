"""The complete, explicit stage graph for full and matched analyses."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

from stages import (
    Artifact,
    RunContext,
    Stage,
    python_command,
)


RETIRED_BANDS = ("broad_5_15", "low_gamma", "low_beta", "high_beta")
PRIMARY_FIT = '"specparam_primary_fit_range_id": "4_50Hz"'


def _flag(condition: bool, value: str) -> list[str]:
    return [value] if condition else []


def _python_builder(
    module: str,
    *arguments: str,
    supports_progress: bool = False,
    supports_overwrite: bool = True,
    compute_skip_figures: bool = False,
):
    def build(context: RunContext, replace: bool) -> list[list[str]]:
        command = python_command("-m", module, *arguments)
        if supports_progress and context.no_progress:
            command.append("--no-progress")
        if compute_skip_figures and context.profile_name == "compute":
            command.append("--skip-figures")
        if supports_overwrite and context.overwrite:
            command.append("--overwrite")
        return [command]

    return build


def _sweep_builder(base: str, output: str):
    def build(context: RunContext, replace: bool) -> list[list[str]]:
        command = python_command(
            "-m",
            "sweep",
            "--base-config",
            base,
            "--output-root",
            output,
        )
        if context.no_progress:
            command.append("--no-progress")
        if context.overwrite:
            command.append("--overwrite")
        return [command]

    return build


def _inspection_builder(context: RunContext, replace: bool) -> list[list[str]]:
    return [[
        *python_command("scripts/inspect_dataset.py"),
        "--config",
        "config/preprocessing.yaml",
    ]]


def _preprocessing_tests_builder(
    context: RunContext, replace: bool
) -> list[list[str]]:
    return [[
        *python_command("-m", "unittest", "-v"),
        "tests.test_cleaning",
        "tests.test_config",
        "tests.test_dataset",
        "tests.test_ica",
    ]]


def _clean_builder(context: RunContext, replace: bool) -> list[list[str]]:
    command = [
        *python_command("scripts/run_preprocessing.py"),
        "--config",
        "config/preprocessing.yaml",
        "--workers",
        str(context.preprocessing_workers),
    ]
    if context.no_progress:
        command.append("--no-progress")
    if context.skip_manual_ica_review:
        command.append("--skip-manual-ica-review")
    # A stale runner fingerprint is not permission to refit ICA. Without an
    # explicit public --overwrite, the preprocessing batch runner validates
    # and reuses every complete subject and processes only missing subjects.
    if context.overwrite:
        command.append("--overwrite")
    return [command]


def _ds002778_inspection_builder(
    context: RunContext, replace: bool
) -> list[list[str]]:
    return [[
        *python_command("scripts/inspect_dataset.py"),
        "--config",
        "config/preprocessing_ds002778.yaml",
    ]]


def _ds002778_clean_builder(
    context: RunContext, replace: bool
) -> list[list[str]]:
    command = [
        *python_command("scripts/run_preprocessing.py"),
        "--config",
        "config/preprocessing_ds002778.yaml",
        "--workers",
        str(context.preprocessing_workers),
    ]
    if context.no_progress:
        command.append("--no-progress")
    if context.skip_manual_ica_review:
        command.append("--skip-manual-ica-review")
    if context.overwrite:
        command.append("--overwrite")
    return [command]


def _ds002778_analysis_builder(
    context: RunContext, replace: bool
) -> list[list[str]]:
    command = python_command(
        "-m",
        "analyses.medication.run_ds002778_analysis",
        "--config",
        "config/analyses/ds002778.json",
    )
    if context.overwrite:
        command.append("--overwrite")
    elif Path("outputs/ds002778/features/subject_features_long.csv").is_file():
        # Refresh statistics and the complete figure battery from reusable
        # feature caches when a newly required plot is missing.
        command.append("--statistics-only")
    return [command]


def _complete_cleaned_cohort() -> str | None:
    participants = Path("processed/metadata/subjects.csv")
    epochs = Path("processed/epochs")
    if not participants.is_file():
        return f"missing {participants}"
    try:
        with participants.open(newline="", encoding="utf-8") as stream:
            expected = sum(1 for _ in csv.DictReader(stream))
    except OSError as error:
        return f"cannot read {participants}: {error}"
    actual = len(list(epochs.glob("sub-*_task-Rest_desc-cleaned_epo.fif")))
    if actual != expected:
        return f"cleaned epoch cohort is incomplete ({actual}/{expected} subjects)"
    return None


def _complete_ds002778_cleaned_cohort() -> str | None:
    recordings = Path("processed_ds002778/metadata/recordings.csv")
    epochs = Path("processed_ds002778/epochs")
    if not recordings.is_file():
        return f"missing {recordings}"
    try:
        with recordings.open(newline="", encoding="utf-8") as stream:
            expected = sum(1 for _ in csv.DictReader(stream))
    except OSError as error:
        return f"cannot read {recordings}: {error}"
    actual = len(list(epochs.glob("sub-*_ses-*_task-rest_desc-cleaned_epo.fif")))
    if actual != expected:
        return f"ds002778 cleaned epoch cohort is incomplete ({actual}/{expected} recordings)"
    return None


def _bycycle_builder(config: str, output: str):
    def build(context: RunContext, replace: bool) -> list[list[str]]:
        analysis = python_command(
            "-m",
            "analyses.bycycle.run_bycycle_burst_analysis",
            "--config",
            config,
        )
        if context.no_progress:
            analysis.append("--no-progress")
        if context.overwrite:
            analysis.append("--overwrite")
        figures = python_command(
            "-m",
            "analyses.bycycle.generate_group_figures",
            "--config",
            config,
            "--output-dir",
            output,
        )
        return [analysis, figures]

    return build


def _full_scale_builder(context: RunContext, replace: bool) -> list[list[str]]:
    command = python_command(
        "-m",
        "analyses.scale_free.run_scale_free_analysis",
        "--config",
        "config/analyses/scale_free.json",
        "--skip-specparam-gallery",
    )
    if context.no_progress:
        command.append("--no-progress")
    if context.overwrite:
        command.append("--overwrite")
    return [command]


def _matched_scale_builder(context: RunContext, replace: bool) -> list[list[str]]:
    command = python_command(
        "-m",
        "analyses.scale_free.run_scale_free_analysis",
        "--config",
        "outputs/matched/cohort/configs/scale_free.json",
        "--skip-specparam-gallery",
    )
    if context.no_progress:
        command.append("--no-progress")
    if context.overwrite:
        command.append("--overwrite")
    return [command]


def _fit_qc_builder(matched: bool):
    def build(context: RunContext, replace: bool) -> list[list[str]]:
        command = python_command(
            "-m", "analyses.scale_free.run_fit_qc_sensitivity"
        )
        if matched:
            command.extend(
                [
                    "--scale-free-output",
                    "outputs/matched/scale_free",
                    "--bout-ordinal-output",
                    "outputs/matched/bouts",
                    "--participants",
                    "outputs/matched/cohort/matched_subjects.csv",
                    "--behavioral-config",
                    "outputs/matched/cohort/configs/behavioral.json",
                    "--behavioral-scale-free-qc-subject-file",
                    "outputs/matched/scale_free/metrics/subject_band_metrics_fit_qc.csv",
                    "--behavioral-bout-ordinal-qc-subject-file",
                    "outputs/matched/bouts/metrics/subject_band_metrics_fit_qc.csv",
                ]
            )
        if context.overwrite:
            command.append("--overwrite")
        return [command]

    return build


def _tests_builder(context: RunContext, replace: bool) -> list[list[str]]:
    return [[*python_command("-m", "unittest", "discover", "-s", "tests", "-v")]]


def _a(path: str, *, contains: Iterable[str] = (), excludes: Iterable[str] = ()) -> Artifact:
    return Artifact(Path(path), tuple(contains), tuple(excludes))


def _sweep_artifacts(root: str) -> tuple[Artifact, ...]:
    artifacts: list[Artifact] = []
    for dimension in (3, 4, 5):
        base = f"{root}/D{dimension}_tau1"
        artifacts.extend(
            [
                _a(f"{base}/manifest.json", excludes=RETIRED_BANDS),
                _a(
                    f"{base}/metrics/subject_electrode_mean_metrics.csv",
                    contains=("renyi_entropy_alpha_0_1", "renyi_entropy_alpha_10"),
                ),
                _a(f"{base}/metrics/band_subject_electrode_mean_metrics.csv"),
                _a(f"{base}/metrics/electrode_metrics.csv"),
                _a(f"{base}/metrics/band_electrode_metrics.csv"),
            ]
        )
    return tuple(artifacts)


def build_registry() -> dict[str, Stage]:
    """Return every stage keyed by its stable public identifier."""
    stages = [
        Stage(
            "inspect",
            "Dataset inspection and metadata bootstrap",
            "shared",
            "preprocessing",
            (),
            _inspection_builder,
            (
                _a("processed/metadata/subjects.csv"),
                _a("processed/metadata/dataset_inspection_report.md"),
            ),
            (Path("scripts/inspect_dataset.py"), Path("src/core/metadata.py"), Path("config/preprocessing.yaml")),
        ),
        Stage(
            "preprocessing-tests",
            "Preprocessing validation tests",
            "shared",
            "preprocessing",
            ("inspect",),
            _preprocessing_tests_builder,
            (),
            (Path("tests/test_cleaning.py"), Path("tests/test_config.py"), Path("tests/test_dataset.py"), Path("tests/test_ica.py"), Path("src")),
            always_run=True,
        ),
        Stage(
            "clean",
            "Reviewed ICA cleaning and four-second epoching",
            "shared",
            "preprocessing",
            ("inspect", "preprocessing-tests"),
            _clean_builder,
            (
                _a("processed/metadata/preprocessing_qc.csv"),
            ),
            (Path("scripts/run_preprocessing.py"), Path("src"), Path("config/preprocessing.yaml")),
            extra_validator=_complete_cleaned_cohort,
        ),
        Stage(
            "ds002778.inspect",
            "ds002778 dataset inspection and recording metadata",
            "ds002778",
            "preprocessing",
            (),
            _ds002778_inspection_builder,
            (
                _a("processed_ds002778/metadata/subjects.csv"),
                _a("processed_ds002778/metadata/recordings.csv"),
                _a("processed_ds002778/metadata/dataset_inspection_report.md"),
            ),
            (
                Path("scripts/inspect_dataset.py"),
                Path("src/core/metadata.py"),
                Path("config/preprocessing_ds002778.yaml"),
            ),
        ),
        Stage(
            "ds002778.clean",
            "ds002778 shared cleaning and four-second epoching",
            "ds002778",
            "preprocessing",
            ("ds002778.inspect",),
            _ds002778_clean_builder,
            (_a("processed_ds002778/metadata/preprocessing_qc.csv"),),
            (
                Path("scripts/run_preprocessing.py"),
                Path("src/core"),
                Path("config/preprocessing_ds002778.yaml"),
            ),
            extra_validator=_complete_ds002778_cleaned_cohort,
        ),
        Stage(
            "ds002778.analysis",
            "ds002778 shared EEG features, paired inference, and complete figures",
            "ds002778",
            "analysis",
            ("ds002778.clean",),
            _ds002778_analysis_builder,
            (
                _a(
                    "outputs/ds002778/manifest.json",
                    contains=(
                        '"ordinal_included": true',
                        '"ebosc_bouts_included": true',
                        '"within_bout_ordinal_included": true',
                        '"n_figures":',
                    ),
                ),
                _a("outputs/ds002778/features/subject_features_long.csv"),
                _a("outputs/ds002778/features/electrode_features_long.csv"),
                _a("outputs/ds002778/statistics/condition_contrasts.csv"),
                _a("outputs/ds002778/statistics/mmse_associations.csv"),
                _a("outputs/ds002778/figures/comparable_pipeline/psd/group_median_psd_with_ci.png"),
                _a("outputs/ds002778/figures/comparable_pipeline/psd/paired_pd_on_minus_off_psd_change.png"),
                _a("outputs/ds002778/figures/comparable_pipeline/topomaps/group_means/ordinal_entropy_group_means.png"),
                _a("outputs/ds002778/figures/comparable_pipeline/topomaps/condition_contrasts/psd_relative_power_condition_contrast_effects.png"),
                _a("outputs/ds002778/figures/comparable_pipeline/topomaps/mmse/aperiodic_aperiodic_exponent_MMSE_partial_Spearman_maps.png"),
                _a("outputs/ds002778/psd/manifest.json"),
                _a("outputs/ds002778/psd/metrics/subject_psd.csv"),
                _a("outputs/ds002778/ordinal/manifest.json"),
                _a("outputs/ds002778/scale_free/manifest.json"),
                _a("outputs/ds002778/bouts/manifest.json"),
                _a("outputs/ds002778/behavioral/manifest.json"),
                _a("outputs/ds002778/psd/figures/statistical_battery/violins/psd/psd_subject_violins_page_001.png"),
                _a("outputs/ds002778/ordinal/figures/statistical_battery/group_statistics/ordinal__PD_OFF_minus_HC_forest_page_001.png"),
                _a("outputs/ds002778/behavioral/figures/statistical_battery/correlations/heatmaps/ordinal_mmse_correlation_heatmap.png"),
                _a("outputs/ds002778/psd/figures/comparable_pipeline/topomaps/condition_contrasts/psd_relative_power_condition_contrast_effects.png"),
                _a("outputs/ds002778/bouts/figures/comparable_pipeline/topomaps/condition_contrasts/within_bout_ordinal_entropy_condition_contrast_effects.png"),
            ),
            (
                Path("src/analyses/medication"),
                Path("config/analyses/ds002778.json"),
                Path("src/core/group_statistics.py"),
            ),
        ),
        Stage(
            "full.psd",
            "PSD features and PD/Control statistics",
            "full",
            "features",
            ("clean",),
            _python_builder("analyses.psd.run_psd_analysis"),
            (
                _a("outputs/full/psd/manifest.json", excludes=RETIRED_BANDS),
                _a("outputs/full/psd/metrics/subject_electrode_band_power.csv"),
                _a("outputs/full/psd/metrics/group_subject_statistics.csv"),
            ),
            (Path("src/analyses/psd"), Path("src/core/group_statistics.py")),
        ),
        Stage(
            "full.ordinal",
            "Primary D=6, tau=1 ordinal features",
            "full",
            "features",
            ("clean",),
            _python_builder(
                "analyses.ordinal.run_ordinal_analysis",
                supports_progress=True,
                compute_skip_figures=True,
            ),
            (
                _a(
                    "outputs/full/ordinal/manifest.json",
                    excludes=RETIRED_BANDS,
                ),
                _a(
                    "outputs/full/ordinal/metrics/subject_electrode_mean_metrics.csv",
                    contains=("renyi_entropy_alpha_0_1", "renyi_entropy_alpha_10"),
                ),
                _a("outputs/full/ordinal/metrics/band_subject_electrode_mean_metrics.csv"),
            ),
            (Path("src/analyses/ordinal"), Path("src/core/group_statistics.py")),
            profile_artifacts={
                "paper": (
                    _a("outputs/full/ordinal/figures/topomaps/renyi_alpha_0_1/group_mean_topomaps.png"),
                    _a("outputs/full/ordinal/figures/group_statistics/broadband/entropy_group_statistics.png"),
                ),
                "full-qc": (
                    _a("outputs/full/ordinal/figures/topomaps/renyi_alpha_0_1/group_mean_topomaps.png"),
                    _a("outputs/full/ordinal/figures/group_statistics/broadband/entropy_group_statistics.png"),
                ),
            },
        ),
        Stage(
            "full.ordinal-sweep",
            "Independent D=3,4,5 ordinal sensitivity at tau=1",
            "full",
            "features",
            ("full.ordinal",),
            _sweep_builder("config/analyses/ordinal.json", "outputs/full/ordinal_sweep"),
            _sweep_artifacts("outputs/full/ordinal_sweep"),
            (Path("src/sweep.py"), Path("src/analyses/ordinal")),
        ),
        Stage(
            "full.scale-free",
            "PSD parameterization, aperiodic fits, eBOSC bouts, and cycle features",
            "full",
            "features",
            ("clean", "full.psd"),
            _full_scale_builder,
            (
                _a(
                    "outputs/full/scale_free/manifest.json",
                    contains=(PRIMARY_FIT, '"criterion": "bic"', '"range_sensitivity_enabled": false'),
                    excludes=RETIRED_BANDS,
                ),
                _a("outputs/full/scale_free/metrics/electrode_aperiodic_metrics.csv"),
                _a("outputs/full/scale_free/metrics/electrode_band_metrics.csv"),
            ),
            (Path("src/analyses/scale_free"), Path("src/core/group_statistics.py")),
        ),
        Stage(
            "full.specparam-gallery",
            "Flat all-electrode specparam gallery",
            "full",
            "report",
            ("full.scale-free",),
            _python_builder("analyses.scale_free.generate_specparam_figures"),
            (
                _a("outputs/full/scale_free/figures/specparam_decomposition/index.html"),
                _a("outputs/full/scale_free/figures/specparam_decomposition/figure_index.csv"),
            ),
            (Path("src/analyses/scale_free/specparam_gallery.py"), Path("src/analyses/scale_free/generate_specparam_figures.py")),
        ),
        Stage(
            "full.bycycle",
            "Independent bycycle burst-detector sensitivity",
            "full",
            "optional",
            ("full.scale-free",),
            _bycycle_builder("config/analyses/bycycle.json", "outputs/full/bycycle"),
            (
                _a("outputs/full/bycycle/manifest.json", excludes=RETIRED_BANDS),
                _a("outputs/full/bycycle/metrics/subject_electrode_band_metrics.csv"),
                _a("outputs/full/bycycle/figures/group_comparisons/group_bout_duration_mean_s.png"),
            ),
            (Path("src/analyses/bycycle"),),
        ),
        Stage(
            "full.within-bout-ordinal",
            "Within-bout Shannon ordinal features",
            "full",
            "features",
            ("full.scale-free",),
            _python_builder(
                "analyses.bouts.run_bout_analyses",
                supports_progress=True,
                compute_skip_figures=True,
            ),
            (
                _a(
                    "outputs/full/bouts/manifest.json",
                    contains=(PRIMARY_FIT, '"criterion": "bic"'),
                    excludes=RETIRED_BANDS,
                ),
                _a("outputs/full/bouts/metrics/subject_electrode_band_metrics.csv"),
            ),
            (Path("src/analyses/bouts"),),
            profile_artifacts={
                "paper": (
                    _a("outputs/full/bouts/figures/group_statistics/entropy_group_statistics.png"),
                ),
                "full-qc": (
                    _a("outputs/full/bouts/figures/group_statistics/entropy_group_statistics.png"),
                ),
            },
        ),
        Stage(
            "full.eight-electrode",
            "Prespecified eight-electrode sensitivity view",
            "full",
            "analysis",
            ("full.psd", "full.ordinal", "full.scale-free", "full.within-bout-ordinal"),
            _python_builder("analyses.eight_electrode.run_eight_electrode_analysis"),
            (
                _a("outputs/full/eight_electrode/manifest.json", excludes=RETIRED_BANDS),
                _a("outputs/full/eight_electrode/REPORT.md"),
                _a("outputs/full/eight_electrode/metrics/electrode_selection.csv", contains=("F4", "P8")),
            ),
            (Path("src/analyses/eight_electrode"),),
        ),
        Stage(
            "full.fit-qc",
            "Aperiodic-fit-QC bout and MOCA sensitivity",
            "full",
            "analysis",
            ("full.scale-free", "full.within-bout-ordinal"),
            _fit_qc_builder(False),
            (
                _a("outputs/full/scale_free/fit_qc_sensitivity_manifest.json"),
                _a("outputs/full/scale_free/metrics/subject_band_metrics_fit_qc.csv", excludes=RETIRED_BANDS),
                _a("outputs/full/bouts/metrics/subject_band_metrics_fit_qc.csv", excludes=RETIRED_BANDS),
            ),
            (Path("src/analyses/scale_free/fit_qc_sensitivity.py"), Path("src/analyses/behavioral/fit_qc_sensitivity.py")),
        ),
        Stage(
            "full.typical-bouts",
            "Subject-balanced stereotypical bout QC gallery",
            "full",
            "report",
            ("full.scale-free", "full.fit-qc"),
            _python_builder("analyses.scale_free.generate_typical_bouts"),
            (
                _a("outputs/full/scale_free/typical_bouts_manifest.json", excludes=RETIRED_BANDS),
                _a("outputs/full/scale_free/figures/typical_bouts/index.html"),
                _a("outputs/full/scale_free/figures/typical_bouts/grand_average_all_subjects.png"),
            ),
            (Path("src/analyses/scale_free/typical_bouts.py"),),
        ),
        Stage(
            "full.classification",
            "Transparent PD-versus-Control prediction models",
            "full",
            "model",
            ("full.psd", "full.ordinal", "full.ordinal-sweep", "full.scale-free", "full.within-bout-ordinal", "full.typical-bouts"),
            _python_builder("analyses.exploration.run_exploration"),
            (
                _a("outputs/full/exploration/manifest.json"),
                _a("outputs/full/exploration/features/subject_modeling_table.csv", contains=("ordinal_global_renyi_entropy_alpha_0_1",), excludes=RETIRED_BANDS),
                _a("outputs/full/exploration/MODEL_REVISION.md"),
            ),
            (Path("src/analyses/exploration"),),
        ),
        Stage(
            "full.cognition",
            "MOCA clinical associations",
            "full",
            "analysis",
            ("full.ordinal-sweep", "full.scale-free", "full.within-bout-ordinal"),
            _python_builder("analyses.behavioral.run_quantitative_behavioral"),
            (
                _a("outputs/full/behavioral/manifest.json"),
                _a("outputs/full/behavioral/REPORT.md"),
                _a("outputs/full/behavioral/metrics/feature_dictionary.csv", contains=("aperiodic_exponent",), excludes=RETIRED_BANDS),
            ),
            (Path("src/analyses/behavioral"),),
        ),
        Stage(
            "full.severity",
            "Whole-head UPDRS and MOCA clinical associations",
            "full",
            "analysis",
            ("full.psd", "full.ordinal-sweep", "full.scale-free", "full.within-bout-ordinal"),
            _python_builder("analyses.progression.run_disease_progression"),
            (
                _a("outputs/full/progression/manifest.json"),
                _a("outputs/full/progression/REPORT.md"),
                _a("outputs/full/progression/metrics/progression_correlations.csv"),
            ),
            (Path("src/analyses/progression"),),
        ),
        Stage(
            "full.duration-qc",
            "At-least-60-second accepted-duration sensitivity",
            "full",
            "analysis",
            ("full.classification", "full.cognition"),
            _python_builder("analyses.duration_qc.run_duration_qc_sensitivity"),
            (
                _a("outputs/full/duration_qc/manifest.json", contains=('"minimum_accepted_duration_seconds": 60',)),
                _a("outputs/full/duration_qc/REPORT.md"),
            ),
            (Path("src/analyses/duration_qc"),),
        ),
        Stage(
            "matched.prepare",
            "Canonical exact-sex/optimal-age matched cohort and generated views",
            "matched",
            "cohort",
            ("inspect",),
            _python_builder("analyses.matching.prepare_matched_cohort", supports_overwrite=False),
            (
                _a("outputs/matched/cohort/manifest.json"),
                _a("outputs/matched/cohort/matched_subjects.csv", contains=("match_pair_id",)),
                _a("outputs/matched/cohort/configs/scale_free.json"),
                _a("outputs/matched/cohort/configs/behavioral.json"),
            ),
            (Path("analyses.matching.prepare_matched_cohort"), Path("src/analyses/exploration/matching.py"), Path("config/analyses/psd.json"), Path("config/analyses/ordinal.json"), Path("config/analyses/scale_free.json"), Path("config/analyses/bouts.json"), Path("config/analyses/behavioral.json")),
        ),
        Stage(
            "matched.psd",
            "Matched PSD statistics",
            "matched",
            "analysis",
            ("matched.prepare", "full.psd"),
            _python_builder("analyses.psd.run_psd_analysis", "--config", "outputs/matched/cohort/configs/psd.json"),
            (
                _a("outputs/matched/psd/manifest.json", excludes=RETIRED_BANDS),
                _a("outputs/matched/psd/metrics/group_subject_statistics.csv"),
            ),
            (Path("src/analyses/psd"),),
        ),
        Stage(
            "matched.ordinal",
            "Matched primary ordinal statistics and figures",
            "matched",
            "analysis",
            ("matched.prepare", "full.ordinal"),
            _python_builder(
                "analyses.ordinal.run_ordinal_analysis",
                "--config",
                "outputs/matched/cohort/configs/ordinal.json",
                supports_progress=True,
            ),
            (
                _a("outputs/matched/ordinal/manifest.json", contains=('"mode": "filtered_subject_level_reuse"',), excludes=RETIRED_BANDS),
                _a("outputs/matched/ordinal/metrics/subject_electrode_mean_metrics.csv", contains=("renyi_entropy_alpha_10",)),
            ),
            (Path("src/analyses/ordinal"),),
        ),
        Stage(
            "matched.ordinal-sweep",
            "Matched D=3,4,5 ordinal sensitivity at tau=1",
            "matched",
            "analysis",
            ("matched.prepare", "full.ordinal-sweep", "matched.ordinal"),
            _sweep_builder("outputs/matched/cohort/configs/ordinal.json", "outputs/matched/ordinal_sweep"),
            _sweep_artifacts("outputs/matched/ordinal_sweep"),
            (Path("src/sweep.py"), Path("src/analyses/ordinal")),
        ),
        Stage(
            "matched.scale-free",
            "Matched scale-free and bout-property summaries",
            "matched",
            "analysis",
            ("matched.prepare", "full.scale-free"),
            _matched_scale_builder,
            (
                _a("outputs/matched/scale_free/manifest.json", contains=(PRIMARY_FIT, '"mode": "filtered_subject_level_reuse"'), excludes=RETIRED_BANDS),
                _a("outputs/matched/scale_free/metrics/electrode_aperiodic_metrics.csv"),
            ),
            (Path("src/analyses/scale_free"),),
        ),
        Stage(
            "matched.specparam-gallery",
            "Matched flat all-electrode specparam gallery",
            "matched",
            "report",
            ("matched.scale-free",),
            _python_builder("analyses.scale_free.generate_specparam_figures", "--config", "outputs/matched/cohort/configs/scale_free.json"),
            (
                _a("outputs/matched/scale_free/figures/specparam_decomposition/index.html"),
                _a("outputs/matched/scale_free/figures/specparam_decomposition/figure_index.csv"),
            ),
            (Path("src/analyses/scale_free/specparam_gallery.py"),),
        ),
        Stage(
            "matched.bycycle",
            "Matched independent bycycle sensitivity",
            "matched",
            "optional",
            ("matched.prepare", "matched.scale-free", "full.bycycle"),
            _bycycle_builder("outputs/matched/cohort/configs/bycycle.json", "outputs/matched/bycycle"),
            (
                _a("outputs/matched/bycycle/manifest.json", excludes=RETIRED_BANDS),
                _a("outputs/matched/bycycle/figures/group_comparisons/group_bout_duration_mean_s.png"),
            ),
            (Path("src/analyses/bycycle"),),
        ),
        Stage(
            "matched.within-bout-ordinal",
            "Matched within-bout ordinal statistics",
            "matched",
            "analysis",
            ("matched.prepare", "matched.scale-free", "full.within-bout-ordinal"),
            _python_builder("analyses.bouts.run_bout_analyses", "--config", "outputs/matched/cohort/configs/bouts.json", supports_progress=True),
            (
                _a("outputs/matched/bouts/manifest.json", contains=(PRIMARY_FIT,), excludes=RETIRED_BANDS),
                _a("outputs/matched/bouts/metrics/subject_electrode_band_metrics.csv"),
            ),
            (Path("src/analyses/bouts"),),
        ),
        Stage(
            "matched.eight-electrode",
            "Matched eight-electrode sensitivity view",
            "matched",
            "analysis",
            ("matched.psd", "matched.ordinal", "matched.scale-free", "matched.within-bout-ordinal"),
            _python_builder("analyses.eight_electrode.run_eight_electrode_analysis", "--config", "outputs/matched/cohort/configs/eight_electrode.json"),
            (
                _a("outputs/matched/eight_electrode/manifest.json", excludes=RETIRED_BANDS),
                _a("outputs/matched/eight_electrode/REPORT.md"),
            ),
            (Path("src/analyses/eight_electrode"),),
        ),
        Stage(
            "matched.fit-qc",
            "Matched fit-QC bout and MOCA sensitivity",
            "matched",
            "analysis",
            ("matched.scale-free", "matched.within-bout-ordinal"),
            _fit_qc_builder(True),
            (
                _a("outputs/matched/scale_free/fit_qc_sensitivity_manifest.json"),
                _a("outputs/matched/scale_free/metrics/subject_band_metrics_fit_qc.csv", excludes=RETIRED_BANDS),
            ),
            (Path("src/analyses/scale_free/fit_qc_sensitivity.py"), Path("src/analyses/behavioral/fit_qc_sensitivity.py")),
        ),
        Stage(
            "matched.typical-bouts",
            "Matched stereotypical bout QC gallery",
            "matched",
            "report",
            ("matched.scale-free", "matched.fit-qc"),
            _python_builder("analyses.scale_free.generate_typical_bouts", "--config", "outputs/matched/cohort/configs/scale_free.json"),
            (
                _a("outputs/matched/scale_free/typical_bouts_manifest.json", excludes=RETIRED_BANDS),
                _a("outputs/matched/scale_free/figures/typical_bouts/index.html"),
            ),
            (Path("src/analyses/scale_free/typical_bouts.py"),),
        ),
        Stage(
            "matched.classification",
            "Matched PD-versus-Control prediction models",
            "matched",
            "model",
            ("matched.psd", "matched.ordinal-sweep", "matched.scale-free", "matched.within-bout-ordinal", "matched.typical-bouts"),
            _python_builder("analyses.exploration.run_exploration", "--config", "outputs/matched/cohort/configs/exploration.json", "--matched-demographics"),
            (
                _a("outputs/matched/exploration/manifest.json", contains=("outputs/matched/cohort/matched_subjects.csv",)),
                _a("outputs/matched/exploration/features/subject_modeling_table.csv", excludes=RETIRED_BANDS),
            ),
            (Path("src/analyses/exploration"),),
        ),
        Stage(
            "matched.cognition",
            "Matched MOCA clinical associations",
            "matched",
            "analysis",
            ("matched.ordinal-sweep", "matched.scale-free", "matched.within-bout-ordinal"),
            _python_builder("analyses.behavioral.run_quantitative_behavioral", "--config", "outputs/matched/cohort/configs/behavioral.json"),
            (
                _a("outputs/matched/behavioral/manifest.json"),
                _a("outputs/matched/behavioral/REPORT.md"),
                _a("outputs/matched/behavioral/metrics/feature_dictionary.csv", excludes=RETIRED_BANDS),
            ),
            (Path("src/analyses/behavioral"),),
        ),
        Stage(
            "matched.severity",
            "Matched whole-head UPDRS and MOCA associations",
            "matched",
            "analysis",
            ("matched.psd", "matched.ordinal-sweep", "matched.scale-free", "matched.within-bout-ordinal"),
            _python_builder("analyses.progression.run_disease_progression", "--config", "outputs/matched/cohort/configs/progression.json"),
            (
                _a("outputs/matched/progression/manifest.json"),
                _a("outputs/matched/progression/REPORT.md"),
                _a("outputs/matched/progression/metrics/progression_correlations.csv"),
            ),
            (Path("src/analyses/progression"),),
        ),
        Stage(
            "matched.duration-qc",
            "Matched at-least-60-second duration sensitivity",
            "matched",
            "analysis",
            ("matched.classification", "matched.cognition"),
            _python_builder("analyses.duration_qc.run_duration_qc_sensitivity", "--matched"),
            (
                _a("outputs/matched/duration_qc/manifest.json", contains=('"minimum_accepted_duration_seconds": 60',)),
                _a("outputs/matched/duration_qc/REPORT.md"),
            ),
            (Path("src/analyses/duration_qc"),),
        ),
        Stage(
            "tests",
            "Repository integration tests",
            "shared",
            "validation",
            (),
            _tests_builder,
            (),
            (Path("tests"), Path("src")),
            always_run=True,
        ),
    ]
    registry = {stage.id: stage for stage in stages}
    if len(registry) != len(stages):
        raise RuntimeError("Duplicate stage identifier in registry")
    for stage in stages:
        missing = sorted(set(stage.dependencies) - set(registry))
        if missing:
            raise RuntimeError(f"Stage {stage.id} has unknown dependencies: {missing}")
    return registry
