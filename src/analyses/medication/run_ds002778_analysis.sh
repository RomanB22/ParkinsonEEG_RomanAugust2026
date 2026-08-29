#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
CONDA_ENVIRONMENT="MNE_August2026"
CONDA_COMMAND="${CONDA_EXE:-$(command -v conda)}"

if [[ $# -lt 1 ]]; then
  echo "Usage: bash src/analyses/medication/run_ds002778_analysis.sh {metadata|review|preprocess|analyze} [arguments...]" >&2
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
  *)
    echo "Unknown command: $COMMAND" >&2
    exit 2
    ;;
esac
