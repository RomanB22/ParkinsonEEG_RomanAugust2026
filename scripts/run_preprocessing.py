#!/usr/bin/env python
"""Fail-visible batch runner for all or selected participants."""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tqdm.auto import tqdm

from core.config import (
    is_ica_review_confirmed,
    load_config,
    preprocessing_signature,
    write_ica_review_proposal,
)
from core.dataset import discover_recordings, subject_id_from_path
from core.metadata import expected_channels_from_dataset, update_preprocessing_qc
from core.preprocessing import process_subject


def _subject_output_is_complete(
    output_dir: str | Path,
    subject_id: str,
    *,
    review_only: bool,
    config: dict | None = None,
) -> bool:
    """Return whether the required subject outputs already exist."""
    output_dir = Path(output_dir)
    required = [
        output_dir
        / "ica"
        / f"{subject_id}_task-Rest_desc-preprocessing-ica.fif",
        output_dir / "qc" / subject_id / "decisions.json",
    ]
    if not review_only:
        required.extend(
            [
                output_dir
                / "cleaned_raw"
                / f"{subject_id}_task-Rest_desc-cleaned_raw.fif",
                output_dir
                / "epochs"
                / f"{subject_id}_task-Rest_desc-cleaned_epo.fif",
            ]
        )
    if not all(path.is_file() for path in required):
        return False
    if config is None:
        return True
    try:
        decisions = json.loads(required[1].read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    return decisions.get("preprocessing_signature") == preprocessing_signature(config)


def _record_parallel_ica_proposal(
    config_path: str | Path,
    output_dir: str | Path,
    subject_id: str,
    *,
    automatic: bool,
) -> None:
    """Serialize proposal writes after a worker finishes one participant."""
    decisions_path = Path(output_dir) / "qc" / subject_id / "decisions.json"
    decisions = json.loads(decisions_path.read_text(encoding="utf-8"))
    components = [
        int(component)
        for component in decisions.get("iclabel_proposed_exclusions", [])
    ]
    reasons = {
        int(component): str(reason)
        for component, reason in decisions.get(
            "iclabel_proposal_reasons", {}
        ).items()
    }
    written = write_ica_review_proposal(
        config_path,
        subject_id,
        components,
        reasons,
        automatic=automatic,
    )
    decisions["iclabel_proposal_written_to_config"] = bool(written)
    decisions_path.write_text(
        json.dumps(decisions, indent=2) + "\n", encoding="utf-8"
    )


def _process_one(
    set_path: Path,
    config: dict,
    expected: list[str],
    *,
    review_only: bool,
    require_review: bool,
    no_downsampling: bool,
    overwrite: bool,
    config_path: str | None,
    skip_manual_ica_review: bool,
    console_logging: bool,
):
    """Pickle-friendly worker wrapper around the subject pipeline."""
    return process_subject(
        set_path,
        config,
        expected,
        review_only=review_only,
        require_review=require_review,
        no_downsampling=no_downsampling,
        overwrite=overwrite,
        config_path=config_path,
        skip_manual_ica_review=skip_manual_ica_review,
        console_logging=console_logging,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/preprocessing.yaml")
    parser.add_argument(
        "--output-dir",
        help="Optional preprocessing output override for development pilots",
    )
    parser.add_argument("--subjects", nargs="*", help="Optional participant IDs; default is all recordings")
    parser.add_argument("--review-only", action="store_true")
    review_group = parser.add_mutually_exclusive_group()
    review_group.add_argument("--allow-unreviewed", action="store_true")
    review_group.add_argument(
        "--skip-manual-ica-review",
        action="store_true",
        help="Automatically apply high-confidence ICLabel proposals without visual confirmation",
    )
    parser.add_argument(
        "--no-ica-downsampling",
        "--no-downsampling",
        dest="no_downsampling",
        action="store_true",
        help="Disable any optional extra ICA-only downsampling (default config already keeps ICA at 250 Hz)",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("PARKINSON_EEG_PREPROCESSING_WORKERS", "1")),
        help="Independent subjects to process concurrently (default: 1)",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the participant progress bar",
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.review_only and args.skip_manual_ica_review:
        parser.error("--skip-manual-ica-review is only meaningful during cleaning")
    config = load_config(args.config)
    if args.output_dir:
        config["project"]["output_dir"] = args.output_dir
    dataset_dir = config["project"]["dataset_dir"]
    task = config["project"]["task"]
    recordings = discover_recordings(dataset_dir, task)
    if args.subjects:
        requested = set(args.subjects)
        recordings = [path for path in recordings if subject_id_from_path(path) in requested]
        found = {subject_id_from_path(path) for path in recordings}
        missing = sorted(requested - found)
        if missing:
            raise FileNotFoundError(f"No recording found for: {missing}")

    if not args.review_only and not args.allow_unreviewed and not args.skip_manual_ica_review:
        unreviewed = [
            subject_id_from_path(path)
            for path in recordings
            if not is_ica_review_confirmed(config, subject_id_from_path(path))
        ]
        if unreviewed:
            preview = ", ".join(unreviewed[:10])
            raise SystemExit(
                f"Refusing to clean {len(unreviewed)} unreviewed ICA decompositions ({preview}...). "
                "Run --review-only first, then add an explicit list for every subject."
            )

    expected = expected_channels_from_dataset(dataset_dir, task, config["channels"]["auxiliary_names"])
    if not args.overwrite:
        completed = [
            subject_id_from_path(path)
            for path in recordings
            if _subject_output_is_complete(
                config["project"]["output_dir"],
                subject_id_from_path(path),
                review_only=args.review_only,
                config=config,
            )
        ]
        if completed:
            completed_set = set(completed)
            recordings = [
                path
                for path in recordings
                if subject_id_from_path(path) not in completed_set
            ]
            print(
                f"Reusing complete preprocessing outputs for {len(completed)} "
                "participant(s); processing only missing subjects"
            )

    worker_count = min(args.workers, max(1, len(recordings)))
    # Prevent each worker's BLAS backend from starting its own large thread
    # pool. Subject-level workers provide the parallelism here.
    if worker_count > 1:
        for variable in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            os.environ[variable] = "1"

    rows = []
    progress = tqdm(
        total=len(recordings),
        desc="ICA cleaning" if not args.review_only else "ICA review",
        unit="subject",
        dynamic_ncols=True,
        disable=args.no_progress,
    )
    parallel_active = False

    def finish_result(result) -> None:
        subject_id = result.subject_id
        if parallel_active and (args.review_only or args.skip_manual_ica_review):
            _record_parallel_ica_proposal(
                args.config,
                config["project"]["output_dir"],
                subject_id,
                automatic=args.skip_manual_ica_review and not args.review_only,
            )
        if result.qc_row:
            rows.append(result.qc_row)
            update_preprocessing_qc(config["project"]["output_dir"], rows)
        progress.set_postfix_str(subject_id, refresh=False)
        progress.update()

    common_kwargs = {
        "review_only": args.review_only,
        "require_review": not args.allow_unreviewed,
        "no_downsampling": args.no_downsampling,
        # Complete subjects were filtered above unless --overwrite was given.
        # Rebuild every remaining subject so an interrupted review/clean run
        # cannot collide with a partial ICA or QC file from the prior attempt.
        "overwrite": True,
        "skip_manual_ica_review": args.skip_manual_ica_review,
        "console_logging": args.no_progress,
    }
    try:
        if worker_count == 1:
            for set_path in recordings:
                result = _process_one(
                    set_path,
                    config,
                    expected,
                    config_path=args.config,
                    **common_kwargs,
                )
                finish_result(result)
        else:
            # Workers never edit the shared config. The parent records each
            # completed proposal atomically in finish_result().
            context = multiprocessing.get_context("spawn")
            try:
                executor = ProcessPoolExecutor(
                    max_workers=worker_count, mp_context=context
                )
            except (OSError, PermissionError) as error:
                tqdm.write(
                    "Parallel subject workers are unavailable on this system; "
                    f"continuing serially ({error})."
                )
                for set_path in recordings:
                    result = _process_one(
                        set_path,
                        config,
                        expected,
                        config_path=args.config,
                        **common_kwargs,
                    )
                    finish_result(result)
            else:
                parallel_active = True
                with executor:
                    futures = {
                        executor.submit(
                            _process_one,
                            set_path,
                            config,
                            expected,
                            config_path=None,
                            **common_kwargs,
                        ): subject_id_from_path(set_path)
                        for set_path in recordings
                    }
                    for future in as_completed(futures):
                        try:
                            finish_result(future.result())
                        except Exception:
                            for pending in futures:
                                pending.cancel()
                            raise
    finally:
        progress.close()
    if rows:
        print(f"Completed {len(rows)} participant(s)")


if __name__ == "__main__":
    main()
