"""CLI for the dataset-agnostic Parkinson resting-state EEG pipeline."""

from __future__ import annotations

import argparse

from .converter import convert_config
from .schema import load_global_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/global_pipeline.json")
    parser.add_argument("--datasets", nargs="+", help="Dataset ids to analyze; defaults to every enabled dataset")
    parser.add_argument("--convert-only", action="store_true")
    parser.add_argument("--skip-figures", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = load_global_config(args.config)
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
    )
    print(
        f"Global pipeline complete: {manifest['n_datasets']} datasets, "
        f"{manifest['n_recordings']} recordings, output={manifest['output_root']}"
    )


if __name__ == "__main__":
    main()
