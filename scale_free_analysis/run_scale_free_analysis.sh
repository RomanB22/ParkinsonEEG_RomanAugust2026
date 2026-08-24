#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

cd "$PROJECT_ROOT"

export MNE_DONTWRITE_HOME=true
export MPLCONFIGDIR="${TMPDIR:-/tmp}/parkinson_eeg_mpl"
export XDG_CACHE_HOME="${TMPDIR:-/tmp}/parkinson_eeg_cache"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

exec conda run --no-capture-output -n MNE_Roman \
    python scale_free_analysis/run_scale_free_analysis.py "$@"
