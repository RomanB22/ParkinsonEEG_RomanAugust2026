#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

cd "$PROJECT_ROOT"

export MNE_DONTWRITE_HOME=true
export MPLCONFIGDIR="${TMPDIR:-/tmp}/parkinson_eeg_mpl"
export XDG_CACHE_HOME="${TMPDIR:-/tmp}/parkinson_eeg_cache"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

# Conda captures subprocess output by default, which hides tqdm updates until
# the process exits. Stream stdout/stderr directly so interactive terminals
# render the progress bar in real time.
exec conda run --no-capture-output -n MNE_Roman \
    python ordinal_analysis/run_ordinal_analysis.py "$@"
