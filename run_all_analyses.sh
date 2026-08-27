#!/usr/bin/env bash

# Backward-compatible downstream-only entry point.
# New documentation uses: bash run_pipeline.sh analyses ...

set -Eeuo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$PROJECT_ROOT/run_pipeline.sh" analyses "$@"
