#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "$PROJECT_ROOT"

CONDA_ENV="${PARKINSON_EEG_CONDA_ENV:-MNE_August2026}"
bash scripts/ensure_conda_environment.sh --env "$CONDA_ENV"
conda run --no-capture-output -n "$CONDA_ENV" \
    python src/analyses/eight_electrode/run_eight_electrode_analysis.py "$@"
