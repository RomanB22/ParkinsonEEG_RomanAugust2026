#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${LAVI_ENV:-MNE_August2026}"
export NUMBA_DISABLE_JIT="${NUMBA_DISABLE_JIT:-1}"
export MNE_DONTWRITE_HOME="${MNE_DONTWRITE_HOME:-true}"
exec conda run --no-capture-output -n "${ENV_NAME}" \
  python scripts/run_abba_burst_analysis.py "$@"
