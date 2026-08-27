#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

cd "$PROJECT_ROOT"

export MNE_DONTWRITE_HOME=true
export MPLCONFIGDIR="${TMPDIR:-/tmp}/parkinson_eeg_mpl"
export XDG_CACHE_HOME="${TMPDIR:-/tmp}/parkinson_eeg_cache"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

CONDA_ENV="${PARKINSON_EEG_CONDA_ENV:-MNE_August2026}"
exec conda run --no-capture-output -n "$CONDA_ENV" \
    python src/analyses/scale_free/generate_typical_bouts.py "$@"
