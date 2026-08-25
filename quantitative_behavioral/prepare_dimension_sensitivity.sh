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
overwrite=false

for argument in "$@"; do
    case "$argument" in
        --overwrite)
            overwrite=true
            ;;
        -h|--help)
            cat <<'EOF'
Usage: bash quantitative_behavioral/prepare_dimension_sensitivity.sh [--overwrite]

Prepare metric-only D=3,4,5,6, tau=1 ordinal inputs. Existing tables are
reused only when they include every configured Rényi alpha column.
EOF
            exit 0
            ;;
        *)
            printf 'Unknown option: %s\n' "$argument" >&2
            exit 2
            ;;
    esac
done

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
    if [[ "$overwrite" == true ]]; then
        complete=false
    fi
    for required_path in "${required_tables[@]}" "$manifest_path"; do
        if [[ "$overwrite" == true ]]; then
            break
        fi
        if [[ ! -f "$required_path" ]]; then
            complete=false
            break
        fi
    done

    if [[ "$complete" == true ]]; then
        if ! python3 -c '
import csv
import sys

required = {
    "renyi_entropy_alpha_0_1",
    "renyi_complexity_alpha_0_1",
    "renyi_entropy_alpha_0_5",
    "renyi_complexity_alpha_0_5",
    "renyi_entropy_alpha_5",
    "renyi_complexity_alpha_5",
}
for filename in sys.argv[1:]:
    with open(filename, newline="", encoding="utf-8") as stream:
        columns = set(next(csv.reader(stream)))
    missing = sorted(required - columns)
    if missing:
        print(f"{filename} is missing new Rényi columns: {missing}", file=sys.stderr)
        raise SystemExit(1)
' "${required_tables[0]}" "${required_tables[1]}" \
            "${required_tables[2]}" "${required_tables[3]}"; then
            complete=false
        fi
    fi

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
