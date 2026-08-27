"""The complete, explicit stage graph for full and matched analyses."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

from parkinson_eeg.stages import (
    Artifact,
    RunContext,
    Stage,
    python_command,
)


RETIRED_BAND = "broad_5_15"
PRIMARY_FIT = '"specparam_primary_fit_range_id": "4_50Hz"'


def _flag(condition: bool, value: str) -> list[str]:
    return [value] if condition else []


def _python_builder(
    script: str,
    *arguments: str,
    supports_progress: bool = False,
    supports_overwrite: bool = True,
    compute_skip_figures: bool = False,
):
    def build(context: RunContext, replace: bool) -> list[list[str]]:
        command = python_command(script, *arguments)
        if supports_progress and context.no_progress:
            command.append("--no-progress")
        if compute_skip_figures and context.profile_name == "compute":
            command.append("--skip-figures")
        if supports_overwrite and replace:
            command.append("--overwrite")
        return [command]

    return build


def _sweep_builder(base: str, output: str):
    def build(context: RunContext, replace: bool) -> list[list[str]]:
        command = python_command(
            "-m",
            "parkinson_eeg.sweep",
            "--base-config",
            base,
            "--output-root",
            output,
        )
        if context.no_progress:
            command.append("--no-progress")
        if replace:
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
    if replace:
        command.append("--overwrite")
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


def _bycycle_builder(config: str, output: str):
    def build(context: RunContext, replace: bool) -> list[list[str]]:
        analysis = python_command(
            "bycycle_burst_analysis/run_bycycle_burst_analysis.py",
            "--config",
            config,
        )
        if context.no_progress:
            analysis.append("--no-progress")
        if replace:
            analysis.append("--overwrite")
        figures = python_command(
            "bycycle_burst_analysis/generate_group_figures.py",
            "--config",
            config,
            "--output-dir",
            output,
        )
        return [analysis, figures]

    return build


def _full_scale_builder(context: RunContext, replace: bool) -> list[list[str]]:
    command = python_command(
        "scale_free_analysis/run_scale_free_analysis.py",
        "--config",
        "scale_free_analysis/config.json",
        "--skip-specparam-gallery",
    )
    if context.no_progress:
        command.append("--no-progress")
    if replace:
        command.append("--overwrite")
    return [command]


def _matched_scale_builder(context: RunContext, replace: bool) -> list[list[str]]:
    command = python_command(
        "scale_free_analysis/run_scale_free_analysis.py",
        "--config",
        "matched_analysis/processed/configs/scale_free.json",
        "--skip-specparam-gallery",
    )
    if context.no_progress:
        command.append("--no-progress")
    if replace:
        command.append("--overwrite")
    return [command]


def _fit_qc_builder(matched: bool):
    def build(context: RunContext, replace: bool) -> list[list[str]]:
        command = python_command("scale_free_analysis/run_fit_qc_sensitivity.py")
        if matched:
            command.extend(
                [
                    "--scale-free-output",
                    "scale_free_analysis/processed_matched",
                    "--bout-ordinal-output",
                    "bout_analyses/processed_matched",
                    "--participants",
                    "matched_analysis/processed/matched_subjects.csv",
                    "--behavioral-config",
                    "matched_analysis/processed/configs/quantitative_behavioral.json",
                    "--behavioral-scale-free-qc-subject-file",
                    "scale_free_analysis/processed_matched/metrics/subject_band_metrics_fit_qc.csv",
                    "--behavioral-bout-ordinal-qc-subject-file",
                    "bout_analyses/processed_matched/metrics/subject_band_metrics_fit_qc.csv",
                ]
            )
        if replace:
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
                _a(f"{base}/manifest.json", excludes=(RETIRED_BAND,)),
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
            (Path("scripts/inspect_dataset.py"), Path("src/metadata.py"), Path("config/preprocessing.yaml")),
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
            "full.psd",
            "PSD features and PD/Control statistics",
            "full",
            "features",
            ("clean",),
            _python_builder("psd_analysis/run_psd_analysis.py"),
            (
                _a("psd_analysis/processed/manifest.json", excludes=(RETIRED_BAND,)),
                _a("psd_analysis/processed/metrics/subject_electrode_band_power.csv"),
                _a("psd_analysis/processed/metrics/group_subject_statistics.csv"),
            ),
            (Path("psd_analysis"), Path("src/group_statistics.py")),
        ),
        Stage(
            "full.ordinal",
            "Primary D=6, tau=1 ordinal features",
            "full",
            "features",
            ("clean",),
            _python_builder(
                "ordinal_analysis/run_ordinal_analysis.py",
                supports_progress=True,
                compute_skip_figures=True,
            ),
            (
                _a(
                    "ordinal_analysis/processed/manifest.json",
                    contains=('"subject_topomaps_generated": false',),
                    excludes=(RETIRED_BAND,),
                ),
                _a(
                    "ordinal_analysis/processed/metrics/subject_electrode_mean_metrics.csv",
                    contains=("renyi_entropy_alpha_0_1", "renyi_entropy_alpha_10"),
                ),
                _a("ordinal_analysis/processed/metrics/band_subject_electrode_mean_metrics.csv"),
            ),
            (Path("ordinal_analysis"), Path("src/group_statistics.py")),
            profile_artifacts={
                "paper": (
                    _a("ordinal_analysis/processed/figures/topomaps/renyi_alpha_0_1/group_mean_topomaps.png"),
                    _a("ordinal_analysis/processed/figures/group_statistics/broadband/entropy_group_statistics.png"),
                ),
                "full-qc": (
                    _a("ordinal_analysis/processed/figures/topomaps/renyi_alpha_0_1/group_mean_topomaps.png"),
                    _a("ordinal_analysis/processed/figures/group_statistics/broadband/entropy_group_statistics.png"),
                ),
            },
        ),
        Stage(
            "full.ordinal-sweep",
            "Independent D=3,4,5 ordinal sensitivity at tau=1",
            "full",
            "features",
            ("full.ordinal",),
            _sweep_builder("ordinal_analysis/config.json", "ordinal_analysis/parameter_sweep"),
            _sweep_artifacts("ordinal_analysis/parameter_sweep"),
            (Path("parkinson_eeg/sweep.py"), Path("ordinal_analysis")),
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
                    "scale_free_analysis/processed/manifest.json",
                    contains=(PRIMARY_FIT, '"criterion": "bic"', '"range_sensitivity_enabled": false'),
                    excludes=(RETIRED_BAND,),
                ),
                _a("scale_free_analysis/processed/metrics/electrode_aperiodic_metrics.csv"),
                _a("scale_free_analysis/processed/metrics/electrode_band_metrics.csv"),
            ),
            (Path("scale_free_analysis"), Path("src/group_statistics.py")),
        ),
        Stage(
            "full.specparam-gallery",
            "Flat all-electrode specparam gallery",
            "full",
            "report",
            ("full.scale-free",),
            _python_builder("scale_free_analysis/generate_specparam_figures.py"),
            (
                _a("scale_free_analysis/processed/figures/specparam_decomposition/index.html"),
                _a("scale_free_analysis/processed/figures/specparam_decomposition/figure_index.csv"),
            ),
            (Path("scale_free_analysis/specparam_gallery.py"), Path("scale_free_analysis/generate_specparam_figures.py")),
        ),
        Stage(
            "full.bycycle",
            "Independent bycycle burst-detector sensitivity",
            "full",
            "optional",
            ("full.scale-free",),
            _bycycle_builder("bycycle_burst_analysis/config.json", "bycycle_burst_analysis/processed"),
            (
                _a("bycycle_burst_analysis/processed/manifest.json", excludes=(RETIRED_BAND,)),
                _a("bycycle_burst_analysis/processed/metrics/subject_electrode_band_metrics.csv"),
                _a("bycycle_burst_analysis/processed/figures/group_comparisons/group_bout_duration_mean_s.png"),
            ),
            (Path("bycycle_burst_analysis"),),
        ),
        Stage(
            "full.within-bout-ordinal",
            "Within-bout Shannon ordinal features",
            "full",
            "features",
            ("full.scale-free",),
            _python_builder(
                "bout_analyses/run_bout_analyses.py",
                supports_progress=True,
                compute_skip_figures=True,
            ),
            (
                _a(
                    "bout_analyses/processed/manifest.json",
                    contains=(PRIMARY_FIT, '"criterion": "bic"'),
                    excludes=(RETIRED_BAND,),
                ),
                _a("bout_analyses/processed/metrics/subject_electrode_band_metrics.csv"),
            ),
            (Path("bout_analyses"),),
            profile_artifacts={
                "paper": (
                    _a("bout_analyses/processed/figures/group_statistics/entropy_group_statistics.png"),
                ),
                "full-qc": (
                    _a("bout_analyses/processed/figures/group_statistics/entropy_group_statistics.png"),
                ),
            },
        ),
        Stage(
            "full.eight-electrode",
            "Prespecified eight-electrode sensitivity view",
            "full",
            "analysis",
            ("full.psd", "full.ordinal", "full.scale-free", "full.within-bout-ordinal"),
            _python_builder("eight_electrode_analysis/run_eight_electrode_analysis.py"),
            (
                _a("eight_electrode_analysis/processed/manifest.json", excludes=(RETIRED_BAND,)),
                _a("eight_electrode_analysis/processed/REPORT.md"),
                _a("eight_electrode_analysis/processed/metrics/electrode_selection.csv", contains=("F4", "P8")),
            ),
            (Path("eight_electrode_analysis"),),
        ),
        Stage(
            "full.fit-qc",
            "Aperiodic-fit-QC bout and MOCA sensitivity",
            "full",
            "analysis",
            ("full.scale-free", "full.within-bout-ordinal"),
            _fit_qc_builder(False),
            (
                _a("scale_free_analysis/processed/fit_qc_sensitivity_manifest.json"),
                _a("scale_free_analysis/processed/metrics/subject_band_metrics_fit_qc.csv", excludes=(RETIRED_BAND,)),
                _a("bout_analyses/processed/metrics/subject_band_metrics_fit_qc.csv", excludes=(RETIRED_BAND,)),
            ),
            (Path("scale_free_analysis/fit_qc_sensitivity.py"), Path("quantitative_behavioral/fit_qc_sensitivity.py")),
        ),
        Stage(
            "full.typical-bouts",
            "Subject-balanced stereotypical bout QC gallery",
            "full",
            "report",
            ("full.scale-free", "full.fit-qc"),
            _python_builder("scale_free_analysis/generate_typical_bouts.py"),
            (
                _a("scale_free_analysis/processed/typical_bouts_manifest.json", excludes=(RETIRED_BAND,)),
                _a("scale_free_analysis/processed/figures/typical_bouts/index.html"),
                _a("scale_free_analysis/processed/figures/typical_bouts/grand_average_all_subjects.png"),
            ),
            (Path("scale_free_analysis/typical_bouts.py"),),
        ),
        Stage(
            "full.classification",
            "Transparent PD-versus-Control prediction models",
            "full",
            "model",
            ("full.psd", "full.ordinal", "full.ordinal-sweep", "full.scale-free", "full.within-bout-ordinal", "full.typical-bouts"),
            _python_builder("exploration/run_exploration.py"),
            (
                _a("exploration/processed/manifest.json"),
                _a("exploration/processed/features/subject_modeling_table.csv", contains=("ordinal_global_renyi_entropy_alpha_0_1",), excludes=(RETIRED_BAND,)),
                _a("exploration/processed/MODEL_REVISION.md"),
            ),
            (Path("exploration"),),
        ),
        Stage(
            "full.cognition",
            "MOCA clinical associations",
            "full",
            "analysis",
            ("full.ordinal-sweep", "full.scale-free", "full.within-bout-ordinal"),
            _python_builder("quantitative_behavioral/run_quantitative_behavioral.py"),
            (
                _a("quantitative_behavioral/processed/manifest.json"),
                _a("quantitative_behavioral/processed/REPORT.md"),
                _a("quantitative_behavioral/processed/metrics/feature_dictionary.csv", contains=("aperiodic_exponent",), excludes=(RETIRED_BAND,)),
            ),
            (Path("quantitative_behavioral"),),
        ),
        Stage(
            "full.severity",
            "Whole-head UPDRS and MOCA clinical associations",
            "full",
            "analysis",
            ("full.psd", "full.ordinal-sweep", "full.scale-free", "full.within-bout-ordinal"),
            _python_builder("disease_progression/run_disease_progression.py"),
            (
                _a("disease_progression/processed/manifest.json"),
                _a("disease_progression/processed/REPORT.md"),
                _a("disease_progression/processed/metrics/progression_correlations.csv"),
            ),
            (Path("disease_progression"),),
        ),
        Stage(
            "full.duration-qc",
            "At-least-60-second accepted-duration sensitivity",
            "full",
            "analysis",
            ("full.classification", "full.cognition"),
            _python_builder("duration_qc_analysis/run_duration_qc_sensitivity.py"),
            (
                _a("duration_qc_analysis/processed/manifest.json", contains=('"minimum_accepted_duration_seconds": 60',)),
                _a("duration_qc_analysis/processed/REPORT.md"),
            ),
            (Path("duration_qc_analysis"),),
        ),
        Stage(
            "matched.prepare",
            "Canonical exact-sex/optimal-age matched cohort and generated views",
            "matched",
            "cohort",
            ("inspect",),
            _python_builder("matched_analysis/prepare_matched_cohort.py", supports_overwrite=False),
            (
                _a("matched_analysis/processed/manifest.json"),
                _a("matched_analysis/processed/matched_subjects.csv", contains=("match_pair_id",)),
                _a("matched_analysis/processed/configs/scale_free.json"),
                _a("matched_analysis/processed/configs/quantitative_behavioral.json"),
            ),
            (Path("matched_analysis/prepare_matched_cohort.py"), Path("exploration/matching.py"), Path("psd_analysis/config.json"), Path("ordinal_analysis/config.json"), Path("scale_free_analysis/config.json"), Path("bout_analyses/config.json"), Path("quantitative_behavioral/config.json")),
        ),
        Stage(
            "matched.psd",
            "Matched PSD statistics",
            "matched",
            "analysis",
            ("matched.prepare", "full.psd"),
            _python_builder("psd_analysis/run_psd_analysis.py", "--config", "matched_analysis/processed/configs/psd.json"),
            (
                _a("psd_analysis/processed_matched/manifest.json", excludes=(RETIRED_BAND,)),
                _a("psd_analysis/processed_matched/metrics/group_subject_statistics.csv"),
            ),
            (Path("psd_analysis"),),
        ),
        Stage(
            "matched.ordinal",
            "Matched primary ordinal statistics and figures",
            "matched",
            "analysis",
            ("matched.prepare", "full.ordinal"),
            _python_builder("ordinal_analysis/run_ordinal_analysis.py", "--config", "matched_analysis/processed/configs/ordinal.json", supports_progress=True),
            (
                _a("ordinal_analysis/processed_matched/manifest.json", contains=('"mode": "filtered_subject_level_reuse"',), excludes=(RETIRED_BAND,)),
                _a("ordinal_analysis/processed_matched/metrics/subject_electrode_mean_metrics.csv", contains=("renyi_entropy_alpha_10",)),
            ),
            (Path("ordinal_analysis"),),
        ),
        Stage(
            "matched.ordinal-sweep",
            "Matched D=3,4,5 ordinal sensitivity at tau=1",
            "matched",
            "analysis",
            ("matched.prepare", "full.ordinal-sweep", "matched.ordinal"),
            _sweep_builder("matched_analysis/processed/configs/ordinal.json", "ordinal_analysis/parameter_sweep_matched"),
            _sweep_artifacts("ordinal_analysis/parameter_sweep_matched"),
            (Path("parkinson_eeg/sweep.py"), Path("ordinal_analysis")),
        ),
        Stage(
            "matched.scale-free",
            "Matched scale-free and bout-property summaries",
            "matched",
            "analysis",
            ("matched.prepare", "full.scale-free"),
            _matched_scale_builder,
            (
                _a("scale_free_analysis/processed_matched/manifest.json", contains=(PRIMARY_FIT, '"mode": "filtered_subject_level_reuse"'), excludes=(RETIRED_BAND,)),
                _a("scale_free_analysis/processed_matched/metrics/electrode_aperiodic_metrics.csv"),
            ),
            (Path("scale_free_analysis"),),
        ),
        Stage(
            "matched.specparam-gallery",
            "Matched flat all-electrode specparam gallery",
            "matched",
            "report",
            ("matched.scale-free",),
            _python_builder("scale_free_analysis/generate_specparam_figures.py", "--config", "matched_analysis/processed/configs/scale_free.json"),
            (
                _a("scale_free_analysis/processed_matched/figures/specparam_decomposition/index.html"),
                _a("scale_free_analysis/processed_matched/figures/specparam_decomposition/figure_index.csv"),
            ),
            (Path("scale_free_analysis/specparam_gallery.py"),),
        ),
        Stage(
            "matched.bycycle",
            "Matched independent bycycle sensitivity",
            "matched",
            "optional",
            ("matched.prepare", "matched.scale-free", "full.bycycle"),
            _bycycle_builder("matched_analysis/processed/configs/bycycle_burst.json", "bycycle_burst_analysis/processed_matched"),
            (
                _a("bycycle_burst_analysis/processed_matched/manifest.json", excludes=(RETIRED_BAND,)),
                _a("bycycle_burst_analysis/processed_matched/figures/group_comparisons/group_bout_duration_mean_s.png"),
            ),
            (Path("bycycle_burst_analysis"),),
        ),
        Stage(
            "matched.within-bout-ordinal",
            "Matched within-bout ordinal statistics",
            "matched",
            "analysis",
            ("matched.prepare", "matched.scale-free", "full.within-bout-ordinal"),
            _python_builder("bout_analyses/run_bout_analyses.py", "--config", "matched_analysis/processed/configs/bout.json", supports_progress=True),
            (
                _a("bout_analyses/processed_matched/manifest.json", contains=(PRIMARY_FIT,), excludes=(RETIRED_BAND,)),
                _a("bout_analyses/processed_matched/metrics/subject_electrode_band_metrics.csv"),
            ),
            (Path("bout_analyses"),),
        ),
        Stage(
            "matched.eight-electrode",
            "Matched eight-electrode sensitivity view",
            "matched",
            "analysis",
            ("matched.psd", "matched.ordinal", "matched.scale-free", "matched.within-bout-ordinal"),
            _python_builder("eight_electrode_analysis/run_eight_electrode_analysis.py", "--config", "matched_analysis/processed/configs/eight_electrode_analysis.json"),
            (
                _a("eight_electrode_analysis/processed_matched/manifest.json", excludes=(RETIRED_BAND,)),
                _a("eight_electrode_analysis/processed_matched/REPORT.md"),
            ),
            (Path("eight_electrode_analysis"),),
        ),
        Stage(
            "matched.fit-qc",
            "Matched fit-QC bout and MOCA sensitivity",
            "matched",
            "analysis",
            ("matched.scale-free", "matched.within-bout-ordinal"),
            _fit_qc_builder(True),
            (
                _a("scale_free_analysis/processed_matched/fit_qc_sensitivity_manifest.json"),
                _a("scale_free_analysis/processed_matched/metrics/subject_band_metrics_fit_qc.csv", excludes=(RETIRED_BAND,)),
            ),
            (Path("scale_free_analysis/fit_qc_sensitivity.py"), Path("quantitative_behavioral/fit_qc_sensitivity.py")),
        ),
        Stage(
            "matched.typical-bouts",
            "Matched stereotypical bout QC gallery",
            "matched",
            "report",
            ("matched.scale-free", "matched.fit-qc"),
            _python_builder("scale_free_analysis/generate_typical_bouts.py", "--config", "matched_analysis/processed/configs/scale_free.json"),
            (
                _a("scale_free_analysis/processed_matched/typical_bouts_manifest.json", excludes=(RETIRED_BAND,)),
                _a("scale_free_analysis/processed_matched/figures/typical_bouts/index.html"),
            ),
            (Path("scale_free_analysis/typical_bouts.py"),),
        ),
        Stage(
            "matched.classification",
            "Matched PD-versus-Control prediction models",
            "matched",
            "model",
            ("matched.psd", "matched.ordinal-sweep", "matched.scale-free", "matched.within-bout-ordinal", "matched.typical-bouts"),
            _python_builder("exploration/run_exploration.py", "--config", "matched_analysis/processed/configs/exploration.json", "--matched-demographics"),
            (
                _a("exploration/processed_matched/manifest.json", contains=("matched_analysis/processed/matched_subjects.csv",)),
                _a("exploration/processed_matched/features/subject_modeling_table.csv", excludes=(RETIRED_BAND,)),
            ),
            (Path("exploration"),),
        ),
        Stage(
            "matched.cognition",
            "Matched MOCA clinical associations",
            "matched",
            "analysis",
            ("matched.ordinal-sweep", "matched.scale-free", "matched.within-bout-ordinal"),
            _python_builder("quantitative_behavioral/run_quantitative_behavioral.py", "--config", "matched_analysis/processed/configs/quantitative_behavioral.json"),
            (
                _a("quantitative_behavioral/processed_matched/manifest.json"),
                _a("quantitative_behavioral/processed_matched/REPORT.md"),
                _a("quantitative_behavioral/processed_matched/metrics/feature_dictionary.csv", excludes=(RETIRED_BAND,)),
            ),
            (Path("quantitative_behavioral"),),
        ),
        Stage(
            "matched.severity",
            "Matched whole-head UPDRS and MOCA associations",
            "matched",
            "analysis",
            ("matched.psd", "matched.ordinal-sweep", "matched.scale-free", "matched.within-bout-ordinal"),
            _python_builder("disease_progression/run_disease_progression.py", "--config", "matched_analysis/processed/configs/disease_progression.json"),
            (
                _a("disease_progression/processed_matched/manifest.json"),
                _a("disease_progression/processed_matched/REPORT.md"),
                _a("disease_progression/processed_matched/metrics/progression_correlations.csv"),
            ),
            (Path("disease_progression"),),
        ),
        Stage(
            "matched.duration-qc",
            "Matched at-least-60-second duration sensitivity",
            "matched",
            "analysis",
            ("matched.classification", "matched.cognition"),
            _python_builder("duration_qc_analysis/run_duration_qc_sensitivity.py", "--matched"),
            (
                _a("duration_qc_analysis/processed_matched/manifest.json", contains=('"minimum_accepted_duration_seconds": 60',)),
                _a("duration_qc_analysis/processed_matched/REPORT.md"),
            ),
            (Path("duration_qc_analysis"),),
        ),
        Stage(
            "tests",
            "Repository integration tests",
            "shared",
            "validation",
            (),
            _tests_builder,
            (),
            (Path("tests"), Path("parkinson_eeg")),
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
