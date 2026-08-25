#!/usr/bin/env bash

# Reproduce the complete reviewed cleaning and downstream analysis workflow.
# Manual ICA review remains a deliberate human checkpoint; use `review` first.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODE="${1:-help}"
if [[ $# -gt 0 ]]; then
    shift
fi

OVERWRITE=false
DRY_RUN=false
NO_PROGRESS=false
SKIP_TESTS=false
SKIP_SWEEP=false
SKIP_EXPLORATION=false
SKIP_MANUAL_ICA_REVIEW=false

usage() {
    cat <<'EOF'
Usage:
  bash run_reproducible_pipeline.sh review [options]
  bash run_reproducible_pipeline.sh run [options]

Modes:
  review   Generate ICA review material and stop at the required human checkpoint.
  run      Run reviewed signal cleaning, then every downstream analysis and test.

Options:
  --overwrite                Recompute cleaning and every downstream stage
  --dry-run                  Print downstream commands without executing them
  --no-progress              Disable supported progress bars
  --skip-tests               Skip the downstream repository-test stage
  --skip-sweep               Skip the D/tau ordinal sweep
  --skip-exploration         Skip full and demographically matched prediction models
  --skip-manual-ica-review   Explicitly use automatic ICLabel proposals during cleaning
  -h, --help                 Show this help

Recommended reproducible reviewed workflow:
  bash run_reproducible_pipeline.sh review --overwrite
  # Inspect ICA stages 08–10 and confirm decisions in config/preprocessing.yaml.
  bash run_reproducible_pipeline.sh run --overwrite

The automatic ICA option is recorded in QC and is not the scientific default.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --overwrite) OVERWRITE=true ;;
        --dry-run) DRY_RUN=true ;;
        --no-progress) NO_PROGRESS=true ;;
        --skip-tests) SKIP_TESTS=true ;;
        --skip-sweep) SKIP_SWEEP=true ;;
        --skip-exploration) SKIP_EXPLORATION=true ;;
        --skip-manual-ica-review) SKIP_MANUAL_ICA_REVIEW=true ;;
        -h|--help) usage; exit 0 ;;
        *) printf 'ERROR: unknown option: %s\n' "$1" >&2; usage >&2; exit 2 ;;
    esac
    shift
done

case "$MODE" in
    review|run) ;;
    help|-h|--help) usage; exit 0 ;;
    *) printf 'ERROR: mode must be review or run\n' >&2; usage >&2; exit 2 ;;
esac

if [[ "$MODE" == "review" && "$SKIP_MANUAL_ICA_REVIEW" == true ]]; then
    printf 'ERROR: --skip-manual-ica-review is only valid in run mode\n' >&2
    exit 2
fi
if [[ "$MODE" == "review" && "$DRY_RUN" == true ]]; then
    printf 'ERROR: --dry-run applies to run mode; review mode creates review material\n' >&2
    exit 2
fi

if [[ "$MODE" == "review" ]]; then
    command=(bash scripts/run_full_cleaning.sh review)
    if [[ "$OVERWRITE" == true ]]; then
        command+=(--overwrite)
    fi
    exec "${command[@]}"
fi

[[ -f processed/metadata/subjects.csv ]] || {
    printf 'ERROR: missing processed/metadata/subjects.csv\n' >&2
    exit 1
}

expected_subjects=$(awk -F, 'NR > 1 {count += 1} END {print count + 0}' \
    processed/metadata/subjects.csv)
cleaned_epochs=$(find processed/epochs -maxdepth 1 -type f \
    -name 'sub-*_task-Rest_desc-cleaned_epo.fif' | wc -l | tr -d ' ')
cleaning_current=false
if [[ "$cleaned_epochs" -eq "$expected_subjects" ]] \
    && [[ -f processed/metadata/preprocessing_qc.csv ]]; then
    cleaning_current=true
fi

cleaning_command=(bash scripts/run_full_cleaning.sh clean)
if [[ "$OVERWRITE" == true ]]; then
    cleaning_command+=(--overwrite)
fi
if [[ "$SKIP_MANUAL_ICA_REVIEW" == true ]]; then
    cleaning_command+=(--skip-manual-ica-review)
fi

printf 'Project: %s\n' "$SCRIPT_DIR"
printf 'Expected subjects: %s\n' "$expected_subjects"
if [[ "$DRY_RUN" == true ]]; then
    printf '\n=== Signal cleaning ===\n'
    printf '  +'
    printf ' %q' "${cleaning_command[@]}"
    printf '\n'
elif [[ "$OVERWRITE" == false && "$cleaning_current" == true ]]; then
    printf '\n=== Signal cleaning ===\n  complete cleaned cohort found; resuming downstream\n'
else
    printf '\n=== Signal cleaning ===\n'
    "${cleaning_command[@]}"
fi

analysis_command=(bash run_all_analyses.sh)
if [[ "$OVERWRITE" == true ]]; then
    analysis_command+=(--overwrite)
fi
if [[ "$DRY_RUN" == true ]]; then
    analysis_command+=(--dry-run)
fi
if [[ "$NO_PROGRESS" == true ]]; then
    analysis_command+=(--no-progress)
fi
if [[ "$SKIP_TESTS" == true ]]; then
    analysis_command+=(--skip-tests)
fi
if [[ "$SKIP_SWEEP" == true ]]; then
    analysis_command+=(--skip-sweep)
fi
if [[ "$SKIP_EXPLORATION" == true ]]; then
    analysis_command+=(--skip-exploration)
fi

"${analysis_command[@]}"

printf '\nComplete reproducible pipeline finished successfully.\n'
