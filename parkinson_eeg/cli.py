"""One human-readable command-line interface for the complete repository."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from parkinson_eeg.config import DEFAULT_CONFIG, load_pipeline_config
from parkinson_eeg.runner import PipelineRunner, Selection
from parkinson_eeg.stages import RunContext, command_environment, command_text, python_command


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Public pipeline configuration")
    parser.add_argument("--profile", default="paper", choices=("compute", "paper", "full-qc"))
    parser.add_argument("--cohort", choices=("full", "matched", "both"), default="both")
    parser.add_argument("--overwrite", action="store_true", help="Recompute selected stages")
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
    review.add_argument("--overwrite", action="store_true")
    review.add_argument("--no-progress", action="store_true")
    review.add_argument("--preprocessing-workers", type=int, default=int(os.environ.get("PARKINSON_EEG_PREPROCESSING_WORKERS", "2")))
    review.add_argument("--env")
    review.add_argument("--log-file", help=argparse.SUPPRESS)

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


def _review(args: argparse.Namespace, environment: str) -> None:
    if getattr(args, "dry_run", False):
        raise ValueError("Review mode creates ICA material and cannot be a dry run")
    context = _context(args, environment)
    commands = [
        python_command("scripts/inspect_dataset.py", "--config", "config/preprocessing.yaml"),
        python_command(
            "-m",
            "unittest",
            "-v",
            "tests.test_cleaning",
            "tests.test_config",
            "tests.test_dataset",
            "tests.test_ica",
        ),
        [
            *python_command("scripts/run_preprocessing.py"),
            "--config",
            "config/preprocessing.yaml",
            "--review-only",
            "--workers",
            str(context.preprocessing_workers),
            *(["--no-progress"] if context.no_progress else []),
            *(["--overwrite"] if context.overwrite else []),
        ],
    ]
    for command in commands:
        print(f"+ {command_text(command)}", flush=True)
        subprocess.run(command, cwd=context.project_root, env=command_environment(context), check=True)
    print("\nICA review material is ready. Confirm decisions before running clean mode.")


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "list":
        from parkinson_eeg.registry import build_registry

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
        if args.command == "review":
            _review(args, environment)
            return

        profile = config.profile(args.profile)
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
            stages = runner.stages_for_profile(profile, selection)

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
