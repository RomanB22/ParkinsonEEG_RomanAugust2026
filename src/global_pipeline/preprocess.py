"""Run the repository's standard cleaning contract for selected datasets."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from .schema import GlobalConfig


def run_preprocessing(
    config: GlobalConfig,
    dataset_ids: list[str] | tuple[str, ...] | None = None,
    *,
    workers: int = 1,
    overwrite: bool = False,
    skip_manual_ica_review: bool = False,
    allow_unreviewed: bool = False,
    no_progress: bool = False,
    repair_incompatible_ica: bool = False,
    skip_unusable_recordings: bool = True,
) -> None:
    """Execute one standard preprocessing command per selected dataset."""
    if workers < 1:
        raise ValueError("preprocessing workers must be positive")
    if skip_manual_ica_review and allow_unreviewed:
        raise ValueError("Choose either skip_manual_ica_review or allow_unreviewed")
    enabled = {dataset.dataset_id: dataset for dataset in config.enabled_datasets}
    requested = tuple(enabled) if dataset_ids is None else tuple(dataset_ids)
    unknown = sorted(set(requested) - set(enabled))
    if unknown:
        raise ValueError(f"Unknown or disabled dataset(s): {unknown}; enabled choices: {sorted(enabled)}")
    project_root = config.path.parent.parent
    for dataset_id in requested:
        dataset = enabled[dataset_id]
        if dataset.preprocessing_config is None:
            raise ValueError(
                f"Dataset {dataset_id!r} has no preprocessing_config. "
                "Add one before using --preprocess."
            )
        if not dataset.preprocessing_config.is_file():
            raise FileNotFoundError(
                f"Dataset {dataset_id!r}: preprocessing config not found: "
                f"{dataset.preprocessing_config}"
            )
        command = [
            sys.executable,
            "scripts/run_preprocessing.py",
            "--config",
            str(dataset.preprocessing_config),
            "--dataset-dir",
            str(dataset.root),
            "--task",
            dataset.task,
            "--output-dir",
            str(dataset.epochs_dir.parent),
            "--workers",
            str(workers),
        ]
        if dataset.notch_frequency_hz is not None:
            command.extend(["--notch-frequency", str(dataset.notch_frequency_hz)])
        if dataset.auxiliary_names:
            command.extend(["--auxiliary-names", *dataset.auxiliary_names])
        if overwrite:
            command.append("--overwrite")
        elif repair_incompatible_ica:
            command.append("--repair-incompatible-ica")
        if skip_unusable_recordings:
            command.append("--skip-unusable-recordings")
        if skip_manual_ica_review:
            command.append("--skip-manual-ica-review")
        elif allow_unreviewed:
            command.append("--allow-unreviewed")
        if no_progress:
            command.append("--no-progress")
        print(f"\n[Preprocess {dataset_id}]\n+ {' '.join(command)}", flush=True)
        subprocess.run(command, cwd=project_root, check=True)
