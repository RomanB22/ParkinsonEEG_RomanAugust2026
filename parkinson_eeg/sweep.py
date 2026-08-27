"""Ordinal embedding-dimension sensitivity without shell-generated configs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ordinal_analysis.pipeline import run_analysis


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def run_ordinal_sweep(
    base_config: str | Path,
    output_root: str | Path,
    *,
    dimensions: tuple[int, ...] = (3, 4, 5),
    delay_samples: int = 1,
    overwrite: bool = False,
    show_progress: bool = True,
    generate_figures: bool = False,
) -> None:
    """Run independent D=3–5 analyses; primary D=6 remains separately owned."""
    source = Path(base_config)
    root = Path(output_root)
    base = _load(source)
    for index, dimension in enumerate(dimensions, start=1):
        output = root / f"D{dimension}_tau{delay_samples}"
        generated = output / "config.json"
        config = json.loads(json.dumps(base))
        config["ordinal"]["embedding_dimension"] = dimension
        config["ordinal"]["delay_samples"] = delay_samples
        config["output_dir"] = str(output)
        source_root = config["input"].get("feature_source_sweep_root")
        if source_root:
            config["input"]["feature_source_output_dir"] = str(
                Path(source_root) / f"D{dimension}_tau{delay_samples}"
            )
        output.mkdir(parents=True, exist_ok=True)
        generated.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        print(
            f"[{index}/{len(dimensions)}] Ordinal sensitivity D={dimension}, "
            f"tau={delay_samples}"
        )
        run_analysis(
            generated,
            overwrite=overwrite,
            show_progress=show_progress,
            generate_figures=generate_figures,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", default="ordinal_analysis/config.json")
    parser.add_argument("--output-root", default="ordinal_analysis/parameter_sweep")
    parser.add_argument("--dimensions", nargs="+", type=int, default=[3, 4, 5])
    parser.add_argument("--delay", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--with-figures", action="store_true")
    args = parser.parse_args()
    run_ordinal_sweep(
        args.base_config,
        args.output_root,
        dimensions=tuple(args.dimensions),
        delay_samples=args.delay,
        overwrite=args.overwrite,
        show_progress=not args.no_progress,
        generate_figures=args.with_figures,
    )


if __name__ == "__main__":
    main()

