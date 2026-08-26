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
SKIP_MATCHED=false
SKIP_MANUAL_ICA_REVIEW=false
INCLUDE_BYCYCLE_BURSTS=false
PROFILE="paper"
PREPROCESSING_WORKERS="${PARKINSON_EEG_PREPROCESSING_WORKERS:-2}"
CONDA_ENV="${PARKINSON_EEG_CONDA_ENV:-MNE_August2026}"

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
  --skip-sweep               Skip D={3,4,5}; primary D=6 still runs at tau=1
  --skip-exploration         Skip full and demographically matched prediction models
  --skip-matched             Skip the complete matched-cohort sensitivity pipeline
  --include-bycycle-bursts   Also run the optional independent bycycle sensitivity
  --profile NAME             compute, paper (default), or full-qc
  --preprocessing-workers N  Concurrent cleaning subjects (default: 2)
  --skip-manual-ica-review   Explicitly use automatic ICLabel proposals during cleaning
  --env NAME                 Conda environment (default: MNE_August2026)
  -h, --help                 Show this help

Recommended reproducible reviewed workflow:
  bash run_reproducible_pipeline.sh review --overwrite
  # Inspect ICA stages 08–10 and confirm decisions in config/preprocessing.yaml.
  bash run_reproducible_pipeline.sh run --profile paper --overwrite

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
        --skip-matched) SKIP_MATCHED=true ;;
        --include-bycycle-bursts) INCLUDE_BYCYCLE_BURSTS=true ;;
        --profile)
            [[ $# -ge 2 ]] || { printf 'ERROR: --profile requires a name\n' >&2; exit 2; }
            PROFILE="$2"
            shift
            ;;
        --preprocessing-workers)
            [[ $# -ge 2 ]] || { printf 'ERROR: --preprocessing-workers requires a number\n' >&2; exit 2; }
            PREPROCESSING_WORKERS="$2"
            shift
            ;;
        --skip-manual-ica-review) SKIP_MANUAL_ICA_REVIEW=true ;;
        --env)
            [[ $# -ge 2 ]] || { printf 'ERROR: --env requires a name\n' >&2; exit 2; }
            CONDA_ENV="$2"
            shift
            ;;
        -h|--help) usage; exit 0 ;;
        *) printf 'ERROR: unknown option: %s\n' "$1" >&2; usage >&2; exit 2 ;;
    esac
    shift
done

if ! [[ "$PREPROCESSING_WORKERS" =~ ^[1-9][0-9]*$ ]]; then
    printf 'ERROR: --preprocessing-workers must be a positive integer\n' >&2
    exit 2
fi

case "$PROFILE" in
    compute|paper) ;;
    full-qc) INCLUDE_BYCYCLE_BURSTS=true ;;
    *) printf 'ERROR: --profile must be compute, paper, or full-qc\n' >&2; exit 2 ;;
esac
if [[ "$PROFILE" == compute && "$INCLUDE_BYCYCLE_BURSTS" == true ]]; then
    printf 'ERROR: compute profile cannot include the optional bycycle analysis\n' >&2
    exit 2
fi

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

export PARKINSON_EEG_CONDA_ENV="$CONDA_ENV"
environment_arguments=(--env "$CONDA_ENV")
if [[ "$DRY_RUN" == true ]]; then environment_arguments+=(--dry-run); fi
bash scripts/ensure_conda_environment.sh "${environment_arguments[@]}"

if [[ "$MODE" == "review" ]]; then
    command=(
        bash scripts/run_full_cleaning.sh review --env "$CONDA_ENV"
        --workers "$PREPROCESSING_WORKERS"
    )
    if [[ "$NO_PROGRESS" == true ]]; then
        command+=(--no-progress)
    fi
    if [[ "$OVERWRITE" == true ]]; then
        command+=(--overwrite)
    fi
    exec "${command[@]}"
fi

if [[ ! -f processed/metadata/subjects.csv ]]; then
    printf '\n=== Dataset inspection and metadata bootstrap ===\n'
    if [[ "$DRY_RUN" == true ]]; then
        printf '  + conda run -n %q python scripts/inspect_dataset.py --config config/preprocessing.yaml\n' "$CONDA_ENV"
        # Dry-run must remain read-only. Use the source participant table only
        # to support the downstream command preview.
        [[ -f dataset/participants.tsv ]] || {
            printf 'ERROR: missing source dataset/participants.tsv\n' >&2
            exit 1
        }
        expected_subjects=$(awk -F'\t' 'NR > 1 {count += 1} END {print count + 0}' \
            dataset/participants.tsv)
    else
        conda run --no-capture-output -n "$CONDA_ENV" \
            python scripts/inspect_dataset.py --config config/preprocessing.yaml
        [[ -f processed/metadata/subjects.csv ]] || {
            printf 'ERROR: dataset inspection did not create processed/metadata/subjects.csv\n' >&2
            exit 1
        }
    fi
fi

if [[ -z "${expected_subjects:-}" ]]; then
    expected_subjects=$(awk -F, 'NR > 1 {count += 1} END {print count + 0}' \
        processed/metadata/subjects.csv)
fi
cleaned_epochs=0
if [[ -d processed/epochs ]]; then
    cleaned_epochs=$(find processed/epochs -maxdepth 1 -type f \
        -name 'sub-*_task-Rest_desc-cleaned_epo.fif' | wc -l | tr -d ' ')
fi
cleaning_current=false
if [[ "$cleaned_epochs" -eq "$expected_subjects" ]] \
    && [[ -f processed/metadata/preprocessing_qc.csv ]]; then
    if conda run -n "$CONDA_ENV" python scripts/check_preprocessing_outputs.py \
        --config config/preprocessing.yaml --quiet >/dev/null 2>&1; then
        cleaning_current=true
    fi
fi

cleaning_command=(
    bash scripts/run_full_cleaning.sh clean --env "$CONDA_ENV"
    --workers "$PREPROCESSING_WORKERS"
)
if [[ "$OVERWRITE" == true ]]; then
    cleaning_command+=(--overwrite)
fi
if [[ "$SKIP_MANUAL_ICA_REVIEW" == true ]]; then
    cleaning_command+=(--skip-manual-ica-review)
fi
if [[ "$NO_PROGRESS" == true ]]; then
    cleaning_command+=(--no-progress)
fi

printf 'Project: %s\n' "$SCRIPT_DIR"
printf 'Expected subjects: %s\n' "$expected_subjects"
cleaning_rebuilt=false
if [[ "$DRY_RUN" == true ]]; then
    printf '\n=== Signal cleaning ===\n'
    printf '  +'
    printf ' %q' "${cleaning_command[@]}"
    printf '\n'
    if [[ "$cleaning_current" == false || "$OVERWRITE" == true ]]; then
        cleaning_rebuilt=true
    fi
elif [[ "$OVERWRITE" == false && "$cleaning_current" == true ]]; then
    printf '\n=== Signal cleaning ===\n  complete cleaned cohort found; resuming downstream\n'
else
    printf '\n=== Signal cleaning ===\n'
    "${cleaning_command[@]}"
    cleaning_rebuilt=true
fi

analysis_command=(bash run_all_analyses.sh --env "$CONDA_ENV")
analysis_command+=(--profile "$PROFILE")
if [[ "$OVERWRITE" == true || "$cleaning_rebuilt" == true ]]; then
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
if [[ "$SKIP_MATCHED" == true ]]; then
    analysis_command+=(--skip-matched)
fi
if [[ "$INCLUDE_BYCYCLE_BURSTS" == true ]]; then
    analysis_command+=(--include-bycycle-bursts)
fi

"${analysis_command[@]}"

printf '\nComplete reproducible pipeline finished successfully.\n'
