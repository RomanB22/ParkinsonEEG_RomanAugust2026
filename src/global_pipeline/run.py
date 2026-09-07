"""CLI for the dataset-agnostic Parkinson resting-state EEG pipeline."""

from __future__ import annotations

import argparse

from .converter import convert_config
from .preprocess import run_preprocessing
from .schema import load_global_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/global_pipeline.json")
    parser.add_argument("--datasets", nargs="+", help="Dataset ids to analyze; defaults to every enabled dataset")
    parser.add_argument("--convert-only", action="store_true")
    parser.add_argument(
        "--analysis-only",
        action="store_true",
        help="Reuse existing cleaned epochs and skip raw-signal preprocessing",
    )
    parser.add_argument("--skip-figures", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--preprocess",
        action="store_true",
        help="Explicit compatibility flag; preprocessing is the default",
    )
    parser.add_argument("--preprocessing-workers", type=int, default=1)
    review = parser.add_mutually_exclusive_group()
    review.add_argument("--skip-manual-ica-review", action="store_true")
    review.add_argument("--allow-unreviewed", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument(
        "--repair-incompatible-ica",
        action="store_true",
        default=True,
        help="Recompute only incompatible saved ICA decompositions while resuming (default)",
    )
    parser.add_argument(
        "--no-repair-incompatible-ica",
        dest="repair_incompatible_ica",
        action="store_false",
        help="Keep the strict preprocessing gate for incompatible ICA files",
    )
    parser.add_argument(
        "--no-skip-unusable-recordings",
        dest="skip_unusable_recordings",
        action="store_false",
        default=True,
        help="Abort instead of recording and skipping degenerate ICA recordings",
    )
    args = parser.parse_args()
    config = load_global_config(args.config)
    if args.convert_only and args.analysis_only:
        parser.error("--convert-only and --analysis-only are mutually exclusive")
    if args.preprocess and args.analysis_only:
        parser.error("--preprocess and --analysis-only are mutually exclusive")
    if not args.analysis_only and not args.convert_only:
        run_preprocessing(
            config,
            dataset_ids=args.datasets,
            workers=args.preprocessing_workers,
            overwrite=args.overwrite,
            skip_manual_ica_review=args.skip_manual_ica_review,
            allow_unreviewed=args.allow_unreviewed,
            no_progress=args.no_progress,
            repair_incompatible_ica=args.repair_incompatible_ica,
            skip_unusable_recordings=args.skip_unusable_recordings,
        )
    if args.convert_only:
        table = convert_config(config, dataset_ids=args.datasets)
        print(f"Converted {len(table)} recordings to {config.output_root / 'canonical'}")
        return
    # Keep metadata conversion usable in a lightweight environment without
    # forcing MNE/ordpy imports until feature computation is requested.
    from .analysis import run_global_pipeline

    manifest = run_global_pipeline(
        config.path,
        skip_figures=args.skip_figures,
        overwrite=args.overwrite,
        dataset_ids=args.datasets,
        show_progress=not args.no_progress,
    )
    print(
        f"Global pipeline complete: {manifest['n_datasets']} datasets, "
        f"{manifest['n_recordings']} recordings, output={manifest['output_root']}"
    )


if __name__ == "__main__":
    main()
