#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${LAVI_ENV:-MNE_August2026}"
exec conda run --no-capture-output -n "${ENV_NAME}" \
  python scripts/run_rhythmicity_analysis.py "$@"
