#!/usr/bin/env bash

# Backward-compatible matched-cohort view. Full-cohort feature caches are
# reused through explicit dependencies in the central Python stage graph.

set -Eeuo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
exec bash "$PROJECT_ROOT/run_pipeline.sh" analyses --cohort matched "$@"
