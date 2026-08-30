#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
CONDA_ENVIRONMENT="MNE_August2026"
CONDA_COMMAND="${CONDA_EXE:-$(command -v conda)}"

usage() {
  echo "Usage: bash src/analyses/medication/run_ds002778_analysis.sh {metadata|review|preprocess|analyze|full} [arguments...]" >&2
}

if [[ $# -lt 1 ]]; then
  usage
  exit 2
fi

COMMAND="$1"
shift
cd "$PROJECT_ROOT"

run_python() {
  "$CONDA_COMMAND" run --no-capture-output -n "$CONDA_ENVIRONMENT" \
    env PYTHONPATH="$PROJECT_ROOT/src" python "$@"
}

case "$COMMAND" in
  metadata)
    run_python -m analyses.medication.run_ds002778_analysis --metadata-only "$@"
    ;;
  review)
    run_python scripts/run_preprocessing.py \
      --config config/preprocessing_ds002778.yaml --review-only "$@"
    ;;
  preprocess)
    run_python scripts/run_preprocessing.py \
      --config config/preprocessing_ds002778.yaml "$@"
    ;;
  analyze)
    run_python -m analyses.medication.run_ds002778_analysis "$@"
    ;;
  full)
    PREPROCESSING_ARGS=()
    ANALYSIS_ARGS=()
    while [[ $# -gt 0 ]]; do
      case "$1" in
        --workers)
          if [[ $# -lt 2 ]]; then
            echo "--workers requires an integer argument" >&2
            exit 2
          fi
          PREPROCESSING_ARGS+=("$1" "$2")
          shift 2
          ;;
        --skip-manual-ica-review|--allow-unreviewed|--no-ica-downsampling|--no-downsampling|--no-progress)
          PREPROCESSING_ARGS+=("$1")
          shift
          ;;
        --skip-ordinal|--skip-bouts|--skip-figures)
          ANALYSIS_ARGS+=("$1")
          shift
          ;;
        --overwrite)
          PREPROCESSING_ARGS+=("$1")
          ANALYSIS_ARGS+=("$1")
          shift
          ;;
        *)
          echo "Unsupported full-pipeline argument: $1" >&2
          echo "Supported: --workers N, --skip-manual-ica-review, --allow-unreviewed, --no-ica-downsampling, --no-progress, --skip-ordinal, --skip-bouts, --skip-figures, --overwrite" >&2
          exit 2
          ;;
      esac
    done

    echo "[1/3] Auditing the ds002778 cohort and session metadata"
    run_python -m analyses.medication.run_ds002778_analysis --metadata-only

    echo "[2/3] Preprocessing all 46 raw BDF recordings and creating cleaned epochs"
    run_python scripts/run_preprocessing.py \
      --config config/preprocessing_ds002778.yaml "${PREPROCESSING_ARGS[@]}"

    echo "[3/3] Extracting EEG features and running condition/MMSE inference"
    run_python -m analyses.medication.run_ds002778_analysis "${ANALYSIS_ARGS[@]}"
    ;;
  *)
    echo "Unknown command: $COMMAND" >&2
    usage
    exit 2
    ;;
esac
