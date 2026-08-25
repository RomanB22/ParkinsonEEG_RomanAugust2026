#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

cd "$PROJECT_ROOT"

BASE_CONFIG="${ORDINAL_BASE_CONFIG:-ordinal_analysis/config.json}"
OUTPUT_ROOT="${ORDINAL_SWEEP_OUTPUT_ROOT:-ordinal_analysis/parameter_sweep}"
CONDA_ENV="${ORDINAL_CONDA_ENV:-${PARKINSON_EEG_CONDA_ENV:-MNE_August2026}}"

embedding_dimensions=(3 4 5 6)
delays=(1)
total_runs=$(( ${#embedding_dimensions[@]} * ${#delays[@]} ))
run_number=0
overwrite=false
with_figures=false
forwarded_arguments=()

for argument in "$@"; do
    case "$argument" in
        -h|--help)
            cat <<'EOF'
Usage: bash ordinal_analysis/run_ordinal_parameter_sweep.sh [ANALYSIS_OPTIONS]

Run D={3,4,5,6} with the ordinal delay fixed at tau=1.
ANALYSIS_OPTIONS are forwarded to run_ordinal_analysis.sh; for example:
  --subjects sub-001 sub-101
  --overwrite
  --no-progress
By default sensitivity runs save metric tables without duplicating the full
figure battery. Pass --with-figures to render figures for every D setting.

Environment overrides:
  ORDINAL_BASE_CONFIG       Base JSON configuration
  ORDINAL_SWEEP_OUTPUT_ROOT Root directory for all sweep outputs
  ORDINAL_CONDA_ENV         Conda environment name
EOF
            exit 0
            ;;
        --config|--config=*)
            printf '%s\n' \
                'Do not pass --config to the sweep; use ORDINAL_BASE_CONFIG instead.' >&2
            exit 2
            ;;
        --overwrite)
            overwrite=true
            forwarded_arguments+=("$argument")
            ;;
        --with-figures)
            with_figures=true
            ;;
        *)
            forwarded_arguments+=("$argument")
            ;;
    esac
done

if [[ ! -f "$BASE_CONFIG" ]]; then
    printf 'Base configuration not found: %s\n' "$BASE_CONFIG" >&2
    exit 1
fi

sweep_output_current() {
    local output_dir="$1"
    local table header
    [[ -f "${output_dir}/manifest.json" ]] || return 1
    for table in \
        "${output_dir}/metrics/subject_electrode_mean_metrics.csv" \
        "${output_dir}/metrics/band_subject_electrode_mean_metrics.csv"; do
        [[ -f "$table" ]] || return 1
        IFS= read -r header < "$table"
        [[ "$header" == *"renyi_entropy_alpha_0_1"* ]] || return 1
        [[ "$header" == *"renyi_complexity_alpha_0_1"* ]] || return 1
        [[ "$header" == *"renyi_entropy_alpha_10"* ]] || return 1
        [[ "$header" == *"renyi_complexity_alpha_10"* ]] || return 1
    done
}

for dimension in "${embedding_dimensions[@]}"; do
    for delay in "${delays[@]}"; do
        run_number=$((run_number + 1))
        output_dir="${OUTPUT_ROOT}/D${dimension}_tau${delay}"
        config_path="${output_dir}/config.json"

        if [[ "$overwrite" == false ]] && sweep_output_current "$output_dir"; then
            printf '\n[%d/%d] Reusing completed ordinal analysis with D=%d, tau=%d\n' \
                "$run_number" "$total_runs" "$dimension" "$delay"
            continue
        fi

        mkdir -p "$output_dir"
        if [[ ! -f "$config_path" || "$overwrite" == true ]]; then
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
Path(config_path).write_text(
    json.dumps(config, indent=2) + "\n",
    encoding="utf-8",
)
' "$BASE_CONFIG" "$config_path" "$output_dir" "$dimension" "$delay"
        fi

        printf '\n[%d/%d] Running ordinal analysis with D=%d, tau=%d\n' \
            "$run_number" "$total_runs" "$dimension" "$delay"
        analysis_command=(
            bash "${SCRIPT_DIR}/run_ordinal_analysis.sh"
            --config "$config_path"
            "${forwarded_arguments[@]}"
        )
        if [[ "$with_figures" == false ]]; then
            analysis_command+=(--skip-figures)
        fi
        "${analysis_command[@]}"
    done
done

printf '\nCompleted all %d ordinal embedding-dimension settings at tau=1. Outputs: %s\n' \
    "$total_runs" "$OUTPUT_ROOT"
