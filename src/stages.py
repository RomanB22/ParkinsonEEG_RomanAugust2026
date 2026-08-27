"""Declarative pipeline stages and lightweight artifact validation."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Sequence


@dataclass(frozen=True)
class Artifact:
    """One required output with optional semantic text checks."""

    path: Path
    contains: tuple[str, ...] = ()
    excludes: tuple[str, ...] = ()

    def problem(self) -> str | None:
        if not self.path.is_file():
            return f"missing {self.path}"
        if not self.contains and not self.excludes:
            return None
        try:
            content = self.path.read_text(encoding="utf-8", errors="replace")
        except OSError as error:
            return f"cannot read {self.path}: {error}"
        for text in self.contains:
            if text not in content:
                return f"{self.path} lacks {text!r}"
        for text in self.excludes:
            if text in content:
                return f"{self.path} still contains retired value {text!r}"
        return None


@dataclass(frozen=True)
class RunContext:
    """User choices supplied to stage command builders."""

    project_root: Path
    environment_name: str
    profile_name: str
    overwrite: bool = False
    no_progress: bool = False
    preprocessing_workers: int = 2
    skip_manual_ica_review: bool = False


CommandBuilder = Callable[[RunContext, bool], list[list[str]]]
Validator = Callable[[], str | None]


@dataclass(frozen=True)
class Stage:
    """A named unit of work with explicit dependencies and products."""

    id: str
    label: str
    cohort: str
    category: str
    dependencies: tuple[str, ...]
    build_commands: CommandBuilder
    artifacts: tuple[Artifact, ...]
    fingerprint_paths: tuple[Path, ...] = ()
    always_run: bool = False
    extra_validator: Validator | None = None
    profile_artifacts: dict[str, tuple[Artifact, ...]] = field(default_factory=dict)

    def artifact_problems(self, profile_name: str | None = None) -> list[str]:
        artifacts = list(self.artifacts)
        if profile_name is not None:
            artifacts.extend(self.profile_artifacts.get(profile_name, ()))
        problems = [
            problem for artifact in artifacts if (problem := artifact.problem())
        ]
        if not problems and self.extra_validator is not None:
            problem = self.extra_validator()
            if problem:
                problems.append(problem)
        return problems


@dataclass(frozen=True)
class StageStatus:
    state: str
    detail: str
    fingerprint: str


class StateStore:
    """Small runner-owned provenance records, separate from scientific outputs."""

    def __init__(self, root: Path = Path(".pipeline/state")) -> None:
        self.root = root

    def path(self, stage_id: str) -> Path:
        return self.root / f"{stage_id.replace('.', '__')}.json"

    def read(self, stage_id: str) -> dict[str, object] | None:
        path = self.path(stage_id)
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def write(
        self,
        stage: Stage,
        fingerprint: str,
        commands: Sequence[Sequence[str]],
        *,
        adopted: bool,
    ) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "stage_id": stage.id,
            "label": stage.label,
            "cohort": stage.cohort,
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "fingerprint": fingerprint,
            "adopted_existing_outputs": adopted,
            "commands": [list(command) for command in commands],
            "artifacts": [str(artifact.path) for artifact in stage.artifacts],
        }
        self.path(stage.id).write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )


def _iter_files(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
        return
    if not path.is_dir():
        return
    ignored_parts = {
        ".git",
        ".pipeline",
        "__pycache__",
        "processed",
        "processed_matched",
        "parameter_sweep",
        "parameter_sweep_matched",
        "pipeline_logs",
    }
    for candidate in sorted(path.rglob("*")):
        if not candidate.is_file() or any(part in ignored_parts for part in candidate.parts):
            continue
        if candidate.suffix in {".pyc", ".png", ".csv", ".fif", ".npz"}:
            continue
        yield candidate


def stage_fingerprint(stage: Stage, dependency_fingerprints: dict[str, str]) -> str:
    """Hash the stage definition, relevant source/config files, and dependencies."""
    digest = hashlib.sha256()
    digest.update(stage.id.encode())
    digest.update("\0".join(stage.dependencies).encode())
    for dependency in stage.dependencies:
        digest.update(dependency.encode())
        digest.update(dependency_fingerprints.get(dependency, "unknown").encode())
    for root in (Path("config/pipeline.yaml"), *stage.fingerprint_paths):
        for path in _iter_files(root):
            digest.update(str(path).encode())
            digest.update(path.read_bytes())
    # Artifact metadata makes downstream freshness respond to a regenerated
    # upstream product without hashing large scientific arrays on every status.
    for artifact in stage.artifacts:
        digest.update(str(artifact.path).encode())
        if artifact.path.is_file():
            stat = artifact.path.stat()
            digest.update(str(stat.st_size).encode())
            digest.update(str(stat.st_mtime_ns).encode())
    return digest.hexdigest()


def inspect_stage_status(
    stage: Stage,
    fingerprint: str,
    store: StateStore,
    *,
    profile_name: str,
) -> StageStatus:
    problems = stage.artifact_problems(profile_name)
    if problems:
        return StageStatus("missing", problems[0], fingerprint)
    if stage.always_run:
        return StageStatus("always", "validation stage always runs", fingerprint)
    record = store.read(stage.id)
    if record is None:
        return StageStatus("legacy", "complete outputs predate runner state", fingerprint)
    if record.get("fingerprint") != fingerprint:
        return StageStatus("stale", "source, config, or dependency changed", fingerprint)
    return StageStatus("current", "outputs and fingerprint match", fingerprint)


def python_command(script: str, *arguments: str) -> list[str]:
    """Build a command using the interpreter already selected by the wrapper."""
    return [sys.executable, script, *arguments]


def shell_command(script: str, *arguments: str) -> list[str]:
    return ["bash", script, *arguments]


def command_text(command: Sequence[str]) -> str:
    """Render a copy-pasteable command without shell interpolation."""
    import shlex

    return shlex.join(command)


def command_environment(context: RunContext) -> dict[str, str]:
    environment = os.environ.copy()
    environment["PARKINSON_EEG_CONDA_ENV"] = context.environment_name
    return environment
