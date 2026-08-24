#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "$PROJECT_ROOT"

BASE_CONFIG="${ORDINAL_BASE_CONFIG:-ordinal_analysis/config.json}"
OUTPUT_ROOT="${ORDINAL_SWEEP_OUTPUT_ROOT:-ordinal_analysis/parameter_sweep}"
CONDA_ENV="${ORDINAL_CONDA_ENV:-MNE_Roman}"
dimensions=(3 4 5 6)
delay=1

if [[ ! -f "$BASE_CONFIG" ]]; then
    printf 'Base ordinal configuration not found: %s\n' "$BASE_CONFIG" >&2
    exit 1
fi

for dimension in "${dimensions[@]}"; do
    output_dir="${OUTPUT_ROOT}/D${dimension}_tau${delay}"
    config_path="${output_dir}/config.json"
    manifest_path="${output_dir}/manifest.json"
    required_tables=(
        "${output_dir}/metrics/subject_electrode_mean_metrics.csv"
        "${output_dir}/metrics/band_subject_electrode_mean_metrics.csv"
        "${output_dir}/metrics/electrode_metrics.csv"
        "${output_dir}/metrics/band_electrode_metrics.csv"
        "${output_dir}/metrics/electrode_sets.json"
    )
    complete=true
    for required_path in "${required_tables[@]}" "$manifest_path"; do
        if [[ ! -f "$required_path" ]]; then
            complete=false
            break
        fi
    done

    if [[ "$complete" == true ]]; then
        printf 'Reusing complete D=%d, tau=%d metrics: %s\n' \
            "$dimension" "$delay" "$output_dir"
        continue
    fi

    mkdir -p "$output_dir"
    conda run -n "$CONDA_ENV" python -c '
import json
import sys
from pathlib import Path

base_path, config_path, output_dir, dimension, delay = sys.argv[1:]
with Path(base_path).open(encoding="utf-8") as stream:
    config = json.load(stream)
config["ordinal"]["embedding_dimension"] = int(dimension)
config["ordinal"]["delay_samples"] = int(delay)
config["output_dir"] = output_dir
Path(config_path).write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
' "$BASE_CONFIG" "$config_path" "$output_dir" "$dimension" "$delay"

    printf 'Calculating D=%d, tau=%d ordinal metric tables (figures skipped)\n' \
        "$dimension" "$delay"
    bash ordinal_analysis/run_ordinal_analysis.sh \
        --config "$config_path" \
        --skip-figures \
        --no-progress \
        --overwrite
done

printf 'D=3,4,5,6, tau=1 sensitivity inputs are ready under %s\n' "$OUTPUT_ROOT"
