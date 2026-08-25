#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "$PROJECT_ROOT"

CONDA_ENV="${PARKINSON_EEG_CONDA_ENV:-MNE_August2026}"
exec conda run --no-capture-output -n "$CONDA_ENV" \
    python matched_analysis/prepare_matched_cohort.py "$@"
