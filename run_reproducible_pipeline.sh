#!/usr/bin/env bash

# Backward-compatible name for the single public runner.
# New documentation uses: bash run_pipeline.sh ...

set -Eeuo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$PROJECT_ROOT/run_pipeline.sh" "$@"
