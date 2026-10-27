"""ABBA-band ordinal sensitivity across independent embedding dimensions."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

from analyses.rhythmicity.abba_ordinal_analysis import run


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _seed_compatible_checkpoints(
    base: dict[str, Any],
    output: Path,
    dimension: int,
    delay_samples: int,
    *,
    control_bands_qc: bool = False,
) -> int:
    """Hard-link checkpoints from the matching legacy single-D run."""
    settings = base["abba_ordinal"]
    if (
        int(settings["embedding_dimension"]) != dimension
        or int(settings["delay_samples"]) != delay_samples
    ):
        return 0
    source_root = Path(base["output_dir"])
    destination_root = output
    if control_bands_qc:
        source_root /= "control_band_qc"
        destination_root /= "control_band_qc"
    source = source_root / "intermediate" / "abba_ordinal_checkpoints"
    destination = destination_root / "intermediate" / "abba_ordinal_checkpoints"
    if not source.is_dir():
        return 0
    destination.mkdir(parents=True, exist_ok=True)
    linked = 0
    for checkpoint in source.rglob("*.pkl"):
        target = destination / checkpoint.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            continue
        try:
            os.link(checkpoint, target)
        except OSError:
            shutil.copy2(checkpoint, target)
        linked += 1
    return linked


def run_sweep(
    base_config: str | Path = "config/analyses/rhythmicity.json",
    output_root: str | Path | None = None,
    *,
    dimensions: tuple[int, ...] | None = None,
    delay_samples: int | None = None,
    datasets: list[str] | None = None,
    recordings: list[str] | None = None,
    workers: int | None = None,
    overwrite: bool = False,
    generate_figures: bool = True,
    control_bands_qc: bool = False,
) -> dict[int, dict[str, int]]:
    source = Path(base_config)
    base = _load(source)
    settings = base["abba_ordinal"]
    selected_dimensions = tuple(
        int(value)
        for value in (
            dimensions
            if dimensions is not None
            else settings.get("embedding_dimensions", [settings["embedding_dimension"]])
        )
    )
    if not selected_dimensions or len(set(selected_dimensions)) != len(selected_dimensions):
        raise ValueError("Embedding dimensions must be a nonempty list without duplicates")
    if any(dimension < 2 or dimension > 7 for dimension in selected_dimensions):
        raise ValueError("Embedding dimensions must be between 2 and 7")
    tau = int(delay_samples if delay_samples is not None else settings["delay_samples"])
    if tau < 1:
        raise ValueError("delay_samples must be positive")
    root = Path(output_root) if output_root is not None else (
        Path(base["output_dir"]) / "abba_ordinal_dimension_sweep"
    )
    summaries: dict[int, dict[str, int]] = {}
    for index, dimension in enumerate(selected_dimensions, start=1):
        output = root / f"D{dimension}_tau{tau}"
        output.mkdir(parents=True, exist_ok=True)
        generated = output / "config.json"
        config = json.loads(json.dumps(base))
        config["source_output_dir"] = str(Path(base["output_dir"]))
        config["output_dir"] = str(output)
        config["abba_ordinal"]["embedding_dimension"] = dimension
        config["abba_ordinal"]["delay_samples"] = tau
        generated.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        seeded = 0 if overwrite else _seed_compatible_checkpoints(
            base,
            output,
            dimension,
            tau,
            control_bands_qc=control_bands_qc,
        )
        print(
            f"[{index}/{len(selected_dimensions)}] ABBA ordinal D={dimension}, "
            f"tau={tau} (seeded checkpoints={seeded})",
            flush=True,
        )
        summaries[dimension] = run(
            generated,
            datasets=datasets,
            recordings=recordings,
            workers=workers,
            overwrite=overwrite,
            generate_figures=generate_figures,
            control_bands_qc=control_bands_qc,
        )
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/analyses/rhythmicity.json")
    parser.add_argument("--output-root")
    parser.add_argument("--dimensions", nargs="+", type=int)
    parser.add_argument("--delay", type=int)
    parser.add_argument("--datasets", nargs="*")
    parser.add_argument("--recordings", nargs="*")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-figures", action="store_true")
    parser.add_argument("--control-bands-qc", action="store_true")
    args = parser.parse_args()
    summary = run_sweep(
        args.config,
        args.output_root,
        dimensions=tuple(args.dimensions) if args.dimensions else None,
        delay_samples=args.delay,
        datasets=args.datasets,
        recordings=args.recordings,
        workers=args.workers,
        overwrite=args.overwrite,
        generate_figures=not args.skip_figures,
        control_bands_qc=args.control_bands_qc,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
