"""Dependency-aware execution for the declarative stage registry."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from parkinson_eeg.config import PipelineConfig, Profile
from parkinson_eeg.registry import build_registry
from parkinson_eeg.stages import (
    RunContext,
    Stage,
    StateStore,
    command_environment,
    command_text,
    inspect_stage_status,
    stage_fingerprint,
)


PREPROCESSING_STAGES = {"inspect", "preprocessing-tests", "clean"}


@dataclass(frozen=True)
class Selection:
    """Optional reductions applied to a named profile."""

    cohorts: tuple[str, ...] = ("full", "matched")
    include_preprocessing: bool = True
    skip_sweep: bool = False
    skip_models: bool = False
    skip_tests: bool = False
    include_bycycle: bool | None = None


FULL_COMPUTE_TARGETS = (
    "full.psd",
    "full.ordinal",
    "full.ordinal-sweep",
    "full.scale-free",
    "full.within-bout-ordinal",
)
FULL_REPORT_TARGETS = (
    "full.specparam-gallery",
    "full.eight-electrode",
    "full.fit-qc",
    "full.typical-bouts",
    "full.classification",
    "full.cognition",
    "full.severity",
    "full.duration-qc",
)
MATCHED_TARGETS = (
    "matched.psd",
    "matched.ordinal",
    "matched.ordinal-sweep",
    "matched.scale-free",
    "matched.specparam-gallery",
    "matched.within-bout-ordinal",
    "matched.eight-electrode",
    "matched.fit-qc",
    "matched.typical-bouts",
    "matched.classification",
    "matched.cognition",
    "matched.severity",
    "matched.duration-qc",
)


def profile_targets(profile: Profile, selection: Selection) -> list[str]:
    """Translate user intent into terminal stages; dependencies are resolved later."""
    targets: list[str] = []
    include_full = "full" in selection.cohorts or "matched" in selection.cohorts
    include_matched = "matched" in selection.cohorts and profile.include_matched
    if include_full:
        targets.extend(FULL_COMPUTE_TARGETS)
        if profile.include_reports and "full" in selection.cohorts:
            targets.extend(FULL_REPORT_TARGETS)
    if include_matched:
        targets.extend(MATCHED_TARGETS)

    include_bycycle = (
        profile.include_bycycle
        if selection.include_bycycle is None
        else selection.include_bycycle
    )
    if include_bycycle and include_full:
        targets.append("full.bycycle")
    if include_bycycle and include_matched:
        targets.append("matched.bycycle")

    if selection.skip_sweep or not profile.include_sweep:
        forbidden = {
            "full.ordinal-sweep",
            "matched.ordinal-sweep",
            "full.classification",
            "matched.classification",
            "full.cognition",
            "matched.cognition",
            "full.severity",
            "matched.severity",
            "full.duration-qc",
            "matched.duration-qc",
        }
        targets = [stage for stage in targets if stage not in forbidden]
    if selection.skip_models or not profile.include_models:
        forbidden = {
            "full.classification",
            "matched.classification",
            "full.duration-qc",
            "matched.duration-qc",
        }
        targets = [stage for stage in targets if stage not in forbidden]
    if profile.include_tests and not selection.skip_tests:
        targets.append("tests")
    return list(dict.fromkeys(targets))


def resolve_dependencies(
    targets: Iterable[str],
    registry: dict[str, Stage],
) -> list[Stage]:
    """Return a stable topological ordering and detect dependency cycles."""
    ordered: list[Stage] = []
    completed: set[str] = set()
    active: set[str] = set()

    def visit(stage_id: str) -> None:
        if stage_id in completed:
            return
        if stage_id in active:
            raise RuntimeError(f"Pipeline dependency cycle involving {stage_id}")
        try:
            stage = registry[stage_id]
        except KeyError as error:
            raise ValueError(f"Unknown pipeline stage: {stage_id}") from error
        active.add(stage_id)
        for dependency in stage.dependencies:
            visit(dependency)
        active.remove(stage_id)
        completed.add(stage_id)
        ordered.append(stage)

    for target in targets:
        visit(target)
    return ordered


class PipelineRunner:
    """Plan, inspect, and execute stages with fail-closed output validation."""

    def __init__(
        self,
        config: PipelineConfig,
        context: RunContext,
        *,
        state_store: StateStore | None = None,
    ) -> None:
        self.config = config
        self.context = context
        self.registry = build_registry()
        self.state_store = state_store or StateStore()

    def stages_for_profile(self, profile: Profile, selection: Selection) -> list[Stage]:
        stages = resolve_dependencies(profile_targets(profile, selection), self.registry)
        if not selection.include_preprocessing:
            stages = [stage for stage in stages if stage.id not in PREPROCESSING_STAGES]
        return stages

    def stages_for_one(self, stage_id: str, *, include_dependencies: bool) -> list[Stage]:
        if include_dependencies:
            return resolve_dependencies([stage_id], self.registry)
        try:
            return [self.registry[stage_id]]
        except KeyError as error:
            raise ValueError(f"Unknown pipeline stage: {stage_id}") from error

    def inspect(self, stages: Sequence[Stage]) -> list[tuple[Stage, str, str, str]]:
        fingerprints: dict[str, str] = {}

        def fingerprint_for(stage_id: str) -> str:
            if stage_id in fingerprints:
                return fingerprints[stage_id]
            stage = self.registry[stage_id]
            for dependency in stage.dependencies:
                fingerprint_for(dependency)
            value = stage_fingerprint(stage, fingerprints)
            fingerprints[stage_id] = value
            return value

        rows: list[tuple[Stage, str, str, str]] = []
        for stage in stages:
            fingerprint = fingerprint_for(stage.id)
            status = inspect_stage_status(
                stage,
                fingerprint,
                self.state_store,
                profile_name=self.context.profile_name,
            )
            rows.append((stage, status.state, status.detail, fingerprint))
        return rows

    def show(self, stages: Sequence[Stage], *, include_commands: bool) -> None:
        rows = self.inspect(stages)
        print(f"Profile: {self.context.profile_name}")
        print(f"Environment: {self.context.environment_name}")
        print(f"Stages: {len(rows)}")
        for index, (stage, state, detail, _) in enumerate(rows, start=1):
            print(f"{index:>2}. [{state.upper():7}] {stage.id:<30} {stage.label}")
            if state not in {"current", "legacy"}:
                print(f"    {detail}")
            if include_commands:
                replace = self.context.overwrite or state in {"missing", "stale"}
                for command in stage.build_commands(self.context, replace):
                    print(f"    + {command_text(command)}")

    def _validate_analysis_prerequisite(self) -> None:
        clean = self.registry["clean"]
        problems = clean.artifact_problems(self.context.profile_name)
        if problems:
            raise RuntimeError(
                "Downstream-only execution requires a complete cleaned cohort: "
                + problems[0]
            )

    def run(
        self,
        stages: Sequence[Stage],
        *,
        dry_run: bool,
        analyses_only: bool = False,
    ) -> None:
        if analyses_only and not dry_run:
            self._validate_analysis_prerequisite()
        total = len(stages)
        for index, stage in enumerate(stages, start=1):
            _, state, detail, fingerprint = self.inspect([stage])[0]
            replace = self.context.overwrite or state in {"missing", "stale"}
            commands = stage.build_commands(self.context, replace)
            print(f"\nSTEP {index}/{total} — {stage.label} [{stage.id}]")
            if not self.context.overwrite and state == "current":
                print("  current output found; skipping")
                continue
            if not self.context.overwrite and state == "legacy" and not stage.always_run:
                print("  complete legacy output found; adopting runner fingerprint")
                if not dry_run:
                    self.state_store.write(
                        stage, fingerprint, commands, adopted=True
                    )
                continue
            if dry_run:
                for command in commands:
                    print(f"  + {command_text(command)}")
                continue
            for command in commands:
                print(f"  + {command_text(command)}", flush=True)
                subprocess.run(
                    command,
                    cwd=self.context.project_root,
                    env=command_environment(self.context),
                    check=True,
                )
            problems = stage.artifact_problems(self.context.profile_name)
            if problems:
                raise RuntimeError(
                    f"Stage {stage.id} exited successfully but output validation failed: "
                    f"{problems[0]}"
                )
            # Output creation changes the stage fingerprint through artifact
            # metadata. Store the post-run value, not the pre-run missing state.
            _, _, _, completed_fingerprint = self.inspect([stage])[0]
            self.state_store.write(
                stage, completed_fingerprint, commands, adopted=False
            )
        if dry_run:
            print("\nDry run complete; no stages or runner state were changed.")
        else:
            print("\nAll selected pipeline stages completed successfully.")
