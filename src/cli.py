"""One human-readable command-line interface for the complete repository."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from config import DEFAULT_CONFIG, PipelineConfig, load_pipeline_config
from runner import PipelineRunner, Selection
from stages import RunContext, command_environment, command_text, python_command


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Public pipeline configuration")
    parser.add_argument("--profile", default="paper", choices=("compute", "paper", "full-qc"))
    parser.add_argument("--cohort", choices=("full", "matched", "both"), default="both")
    parser.add_argument(
        "--dataset",
        default="primary",
        help="Dataset profile from config/pipeline.yaml, or 'both' (default: primary)",
    )
    parser.add_argument("--overwrite", action="store_true", help="Recompute selected stages")
    parser.add_argument(
        "--augment-weighted-entropy",
        action="store_true",
        help="Append weighted entropy to existing primary ordinal and bout outputs",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--skip-sweep", action="store_true")
    parser.add_argument("--skip-exploration", action="store_true", help="Skip classification models")
    parser.add_argument("--skip-matched", action="store_true", help="Compatibility alias for --cohort full")
    parser.add_argument("--include-bycycle-bursts", action="store_true")
    parser.add_argument("--preprocessing-workers", type=int, default=int(os.environ.get("PARKINSON_EEG_PREPROCESSING_WORKERS", "2")))
    parser.add_argument("--skip-manual-ica-review", action="store_true")
    parser.add_argument("--env", help="Conda environment; normally supplied by run_pipeline.sh")
    parser.add_argument("--log-file", help=argparse.SUPPRESS)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="parkinson-eeg",
        description="Run, inspect, or resume the Parkinson resting-state EEG pipeline.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="Clean signals and run the selected analysis profile")
    _common(run)

    analyses = commands.add_parser("analyses", help="Run downstream analyses from existing cleaned epochs")
    _common(analyses)

    plan = commands.add_parser("plan", help="Show the ordered dependency plan and commands")
    _common(plan)

    status = commands.add_parser("status", help="Show freshness of every selected stage")
    _common(status)

    stage = commands.add_parser("stage", help="Run one named stage")
    stage.add_argument("stage_id")
    stage.add_argument("--no-deps", action="store_true", help="Do not resolve dependencies")
    _common(stage)

    review = commands.add_parser("review", help="Generate ICA review material and stop")
    review.add_argument("--config", default=str(DEFAULT_CONFIG))
    review.add_argument(
        "--dataset",
        default="primary",
        help="Dataset profile from config/pipeline.yaml, or 'both'",
    )
    review.add_argument("--overwrite", action="store_true")
    review.add_argument("--no-progress", action="store_true")
    review.add_argument("--preprocessing-workers", type=int, default=int(os.environ.get("PARKINSON_EEG_PREPROCESSING_WORKERS", "2")))
    review.add_argument("--env")
    review.add_argument("--log-file", help=argparse.SUPPRESS)

    preprocess = commands.add_parser(
        "preprocess",
        help="Run the same preprocessing contract for either or both datasets",
    )
    preprocess.add_argument(
        "phase",
        choices=("inspect", "review", "clean", "status"),
        help="Inspect metadata, generate ICA review material, create epochs, or check completeness",
    )
    preprocess.add_argument("--config", default=str(DEFAULT_CONFIG))
    preprocess.add_argument(
        "--dataset",
        default="primary",
        help="Dataset profile from config/pipeline.yaml, or 'both'",
    )
    preprocess.add_argument(
        "--subjects",
        nargs="*",
        help="Optional participant or recording IDs passed to each selected dataset",
    )
    preprocess.add_argument("--workers", type=int, default=int(os.environ.get("PARKINSON_EEG_PREPROCESSING_WORKERS", "2")))
    preprocess.add_argument("--overwrite", action="store_true")
    preprocess.add_argument("--no-progress", action="store_true")
    preprocess.add_argument("--skip-manual-ica-review", action="store_true")
    preprocess.add_argument("--allow-unreviewed", action="store_true")
    preprocess.add_argument("--no-ica-downsampling", action="store_true")
    preprocess.add_argument("--dry-run", action="store_true")
    preprocess.add_argument("--env")
    preprocess.add_argument("--log-file", help=argparse.SUPPRESS)

    commands.add_parser("list", help="List stable stage identifiers")
    commands.add_parser("validate-config", help="Validate shared scientific settings")
    return parser


def _cohorts(args: argparse.Namespace) -> tuple[str, ...]:
    if getattr(args, "skip_matched", False):
        return ("full",)
    value = getattr(args, "cohort", "both")
    if value == "both":
        return ("full", "matched")
    return (value,)


def _context(args: argparse.Namespace, environment: str) -> RunContext:
    workers = int(getattr(args, "preprocessing_workers", 2))
    if workers < 1:
        raise ValueError("--preprocessing-workers must be a positive integer")
    return RunContext(
        project_root=Path.cwd().resolve(),
        environment_name=environment,
        profile_name=getattr(args, "profile", "paper"),
        overwrite=bool(getattr(args, "overwrite", False)),
        no_progress=bool(getattr(args, "no_progress", False)),
        preprocessing_workers=workers,
        skip_manual_ica_review=bool(getattr(args, "skip_manual_ica_review", False)),
    )


def _dataset_names(config: PipelineConfig, selection: str) -> tuple[str, ...]:
    if selection == "both":
        return tuple(config.datasets)
    config.dataset(selection)
    return (selection,)


def _preprocessing_commands(
    args: argparse.Namespace, config: PipelineConfig
) -> list[tuple[str, list[str]]]:
    """Build one shared inspection/review/clean workflow for selected datasets."""
    phase = str(args.phase)
    dataset_names = _dataset_names(config, str(args.dataset))
    commands: list[tuple[str, list[str]]] = []
    if phase == "status":
        return [
            (
                f"Check {config.dataset(name).label}",
                python_command(
                    "scripts/check_preprocessing_outputs.py",
                    "--config",
                    str(config.dataset(name).preprocessing_config),
                ),
            )
            for name in dataset_names
        ]
    for name in dataset_names:
        dataset = config.dataset(name)
        commands.append(
            (
                f"Inspect {dataset.label}",
                python_command(
                    "scripts/inspect_dataset.py",
                    "--config",
                    str(dataset.preprocessing_config),
                ),
            )
        )
    if phase == "inspect":
        return commands

    commands.append(
        (
            "Validate the shared preprocessing implementation",
            python_command(
                "-m",
                "unittest",
                "-v",
                "tests.test_cleaning",
                "tests.test_config",
                "tests.test_dataset",
                "tests.test_ica",
            ),
        )
    )
    for name in dataset_names:
        dataset = config.dataset(name)
        command = [
            *python_command("scripts/run_preprocessing.py"),
            "--config",
            str(dataset.preprocessing_config),
            "--workers",
            str(args.workers),
        ]
        if phase == "review":
            command.append("--review-only")
        if args.subjects:
            command.extend(["--subjects", *args.subjects])
        if args.no_progress:
            command.append("--no-progress")
        if args.overwrite:
            command.append("--overwrite")
        if phase == "clean" and args.skip_manual_ica_review:
            command.append("--skip-manual-ica-review")
        if phase == "clean" and args.allow_unreviewed:
            command.append("--allow-unreviewed")
        if args.no_ica_downsampling:
            command.append("--no-ica-downsampling")
        commands.append((f"{phase.title()} {dataset.label}", command))
    return commands


def _preprocess(
    args: argparse.Namespace, environment: str, config: PipelineConfig
) -> None:
    if args.workers < 1:
        raise ValueError("--workers must be a positive integer")
    if args.phase == "review" and (
        args.skip_manual_ica_review or args.allow_unreviewed
    ):
        raise ValueError(
            "--skip-manual-ica-review and --allow-unreviewed apply only to clean mode"
        )
    context = _context(args, environment)
    for label, command in _preprocessing_commands(args, config):
        print(f"\n[{label}]\n+ {command_text(command)}", flush=True)
        if not args.dry_run:
            subprocess.run(
                command,
                cwd=context.project_root,
                env=command_environment(context),
                check=True,
            )
    if args.dry_run:
        return
    if args.phase == "review":
        names = ", ".join(_dataset_names(config, args.dataset))
        print(f"\nICA review material is ready for: {names}.")
    elif args.phase == "clean":
        print("\nSelected datasets completed the shared preprocessing contract.")


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "list":
        from registry import build_registry

        for stage in build_registry().values():
            print(f"{stage.id:<30} {stage.cohort:<8} {stage.label}")
        return

    config_path = getattr(args, "config", str(DEFAULT_CONFIG))
    try:
        config = load_pipeline_config(config_path)
        if args.command == "validate-config":
            print(f"Configuration valid: {config.path}")
            print("Shared scientific settings agree across all domain configs.")
            return
        environment = getattr(args, "env", None) or os.environ.get(
            "PARKINSON_EEG_CONDA_ENV", config.environment_name
        )
        if args.command in {"review", "preprocess"}:
            if args.command == "review":
                args.phase = "review"
                args.workers = args.preprocessing_workers
                args.subjects = None
                args.skip_manual_ica_review = False
                args.allow_unreviewed = False
                args.no_ica_downsampling = False
                args.dry_run = False
            _preprocess(args, environment, config)
            return

        profile = config.profile(args.profile)
        if args.command == "analyses" and args.augment_weighted_entropy:
            if args.dataset != "primary":
                raise ValueError("--augment-weighted-entropy currently supports --dataset primary only")
            if args.overwrite:
                raise ValueError("Use either --augment-weighted-entropy or --overwrite, not both")
            import importlib.util as _importlib_util
            import sys as _sys

            augment_path = Path.cwd() / "scripts" / "augment_weighted_entropy.py"
            if not augment_path.is_file():
                raise FileNotFoundError(f"Missing augmentation script: {augment_path}")
            augment_spec = _importlib_util.spec_from_file_location(
                "augment_weighted_entropy", augment_path
            )
            if augment_spec is None or augment_spec.loader is None:
                raise RuntimeError(f"Could not load augmentation script: {augment_path}")
            augment_module = _importlib_util.module_from_spec(augment_spec)
            _sys.modules[augment_spec.name] = augment_module
            augment_spec.loader.exec_module(augment_module)
            _sys.argv = ["augment_weighted_entropy", "--cohort", args.cohort]
            augment_module.main()
            return
        context = _context(args, environment)
        runner = PipelineRunner(config, context)
        if args.command == "stage":
            stages = runner.stages_for_one(args.stage_id, include_dependencies=not args.no_deps)
        else:
            selection = Selection(
                cohorts=_cohorts(args),
                include_preprocessing=args.command in {"run", "plan", "status"},
                skip_sweep=args.skip_sweep,
                skip_models=args.skip_exploration,
                skip_tests=args.skip_tests,
                include_bycycle=True if args.include_bycycle_bursts else None,
            )
            selected_datasets = (
                tuple(config.datasets)
                if args.dataset == "both"
                else (args.dataset,)
            )
            stages = []
            if "primary" in selected_datasets:
                stages.extend(runner.stages_for_profile(profile, selection))
            for dataset_name in selected_datasets:
                if dataset_name == "primary":
                    continue
                analysis_stage = config.dataset(dataset_name).analysis_stage
                if analysis_stage is None:
                    raise ValueError(
                        f"Dataset {dataset_name!r} has no downstream analysis stage"
                    )
                dataset_stages = runner.stages_for_one(
                    analysis_stage,
                    include_dependencies=True,
                )
                if args.command == "analyses":
                    from runner import PREPROCESSING_STAGES

                    dataset_stages = [
                        stage
                        for stage in dataset_stages
                        if stage.id not in PREPROCESSING_STAGES
                    ]
                known = {stage.id for stage in stages}
                stages.extend(
                    stage for stage in dataset_stages if stage.id not in known
                )

        if args.command in {"plan", "status"}:
            runner.show(stages, include_commands=args.command == "plan")
            return
        runner.run(
            stages,
            dry_run=args.dry_run,
            analyses_only=args.command == "analyses",
        )
    except (FileNotFoundError, RuntimeError, ValueError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"ERROR: {error}\n")


if __name__ == "__main__":
    main()
