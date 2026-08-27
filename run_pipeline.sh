#!/usr/bin/env bash

# One public entry point. Scientific work is orchestrated by the typed Python
# stage graph; this wrapper only selects the conda interpreter and captures logs.

set -Eeuo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

if [[ $# -eq 0 ]]; then
    set -- --help
fi

ORIGINAL_ARGUMENTS=("$@")
COMMAND="${1:-help}"
case "$COMMAND" in
    help|-h|--help)
        set -- --help
        COMMAND="help"
        ;;
esac
CONDA_ENV="${PARKINSON_EEG_CONDA_ENV:-}"
PIPELINE_CONFIG="config/pipeline.yaml"
PIPELINE_LOG="${PARKINSON_EEG_PIPELINE_LOG:-}"
DRY_RUN=false

arguments=("$@")
for ((index=0; index<${#arguments[@]}; index++)); do
    case "${arguments[$index]}" in
        --env)
            next=$((index + 1))
            [[ $next -lt ${#arguments[@]} ]] || {
                printf 'ERROR: --env requires a name\n' >&2
                exit 2
            }
            CONDA_ENV="${arguments[$next]}"
            ;;
        --config)
            next=$((index + 1))
            [[ $next -lt ${#arguments[@]} ]] || {
                printf 'ERROR: --config requires a path\n' >&2
                exit 2
            }
            PIPELINE_CONFIG="${arguments[$next]}"
            ;;
        --log-file)
            next=$((index + 1))
            [[ $next -lt ${#arguments[@]} ]] || {
                printf 'ERROR: --log-file requires a path\n' >&2
                exit 2
            }
            PIPELINE_LOG="${arguments[$next]}"
            ;;
        --dry-run) DRY_RUN=true ;;
    esac
done

if [[ -z "$CONDA_ENV" ]]; then
    CONDA_ENV="$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["environment"]["name"])' "$PIPELINE_CONFIG")"
fi

export PARKINSON_EEG_CONDA_ENV="$CONDA_ENV"
export CONDA_CHANNELS="${CONDA_CHANNELS:-defaults}"

mutating=false
case "$COMMAND" in
    run|analyses|review|stage) mutating=true ;;
esac

finish_pipeline_log() {
    pipeline_exit_code=$?
    trap - EXIT
    if [[ "$pipeline_exit_code" -eq 0 ]]; then
        pipeline_state="SUCCESS"
    else
        pipeline_state="FAILED"
    fi
    printf '\nPipeline status: %s (exit code %d)\n' "$pipeline_state" "$pipeline_exit_code"
    printf 'Finished: %s\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')"
    printf 'Consolidated log: %s\n' "$PIPELINE_LOG"
    exit "$pipeline_exit_code"
}

if [[ "$mutating" == true && "$DRY_RUN" == false ]]; then
    if [[ -z "$PIPELINE_LOG" ]]; then
        PIPELINE_LOG="pipeline_logs/$(date '+%Y%m%d_%H%M%S')_${COMMAND}.log"
    fi
    if [[ "$PIPELINE_LOG" != /* ]]; then
        PIPELINE_LOG="$PROJECT_ROOT/$PIPELINE_LOG"
    fi
    mkdir -p "$(dirname "$PIPELINE_LOG")"
    : >> "$PIPELINE_LOG"
    exec > >(tee -a "$PIPELINE_LOG") 2>&1
    trap finish_pipeline_log EXIT
    printf 'Consolidated log: %s\n' "$PIPELINE_LOG"
    printf 'Started: %s\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')"
    printf 'Command:'
    printf ' %q' "$0" "${ORIGINAL_ARGUMENTS[@]}"
    printf '\n'
fi

# Read-only commands and dry-runs use the base interpreter because the public
# runner itself has no scientific dependencies. Real work uses the pinned env.
if [[ "$mutating" == false || "$DRY_RUN" == true ]]; then
    exec python3 src/cli.py "$@"
fi

bash scripts/ensure_conda_environment.sh --env "$CONDA_ENV"
conda run --no-capture-output -n "$CONDA_ENV" python src/cli.py "$@"
