#!/usr/bin/env bash

# Ensure the project conda environment exists without modifying an existing one.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

CONDA_ENV="${PARKINSON_EEG_CONDA_ENV:-MNE_August2026}"
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --env)
            [[ $# -ge 2 ]] || { echo "ERROR: --env requires a name" >&2; exit 2; }
            CONDA_ENV="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            echo "Usage: bash scripts/ensure_conda_environment.sh [--env NAME] [--dry-run]"
            exit 0
            ;;
        *)
            echo "ERROR: unknown option: $1" >&2
            exit 2
            ;;
    esac
done

command -v conda >/dev/null 2>&1 || {
    echo "ERROR: conda is not available on PATH" >&2
    exit 1
}

environment_exists() {
    conda env list | awk -v environment_name="$CONDA_ENV" '
        $1 == environment_name { found = 1 }
        END { exit !found }
    '
}

if environment_exists; then
    printf 'Conda environment ready: %s\n' "$CONDA_ENV"
    exit 0
fi

printf 'Conda environment not found: %s\n' "$CONDA_ENV"
printf 'The pinned environment will be created from requirements.txt.\n'
if [[ "$DRY_RUN" == true ]]; then
    printf '  + bash scripts/create_conda_environment.sh --env %q\n' "$CONDA_ENV"
else
    cd "$PROJECT_ROOT"
    bash scripts/create_conda_environment.sh --env "$CONDA_ENV"
fi
