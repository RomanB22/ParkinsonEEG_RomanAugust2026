#!/usr/bin/env bash

# Run the Parkinson resting-state EEG preprocessing workflow in MNE_Roman.
#
# ICA review is deliberately a separate stage. ICLabel prefills proposals, but
# the clean stage refuses to start until a person confirms every participant.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

MODE="${1:-help}"
if [[ $# -gt 0 ]]; then
    shift
fi

CONDA_ENV="MNE_Roman"
CONFIG_PATH="config/preprocessing.yaml"
OVERWRITE=false
SKIP_MANUAL_ICA_REVIEW=false

usage() {
    cat <<'EOF'
Usage:
  bash scripts/run_full_cleaning.sh review [options]
  bash scripts/run_full_cleaning.sh clean  [options]
  bash scripts/run_full_cleaning.sh pilot  [options]

Modes:
  review  Inspect/test the dataset, then create 120 Hz ICA review material
          for all 149 participants and prefill ICLabel proposals. No ICA
          components are removed.

  clean   Inspect/test the dataset, then create cleaned continuous EEG and
          accepted epochs for all participants. By default every participant
          needs a confirmed ICA review; the explicit skip flag uses ICLabel.

  pilot   Run the sub-001/sub-101 pilot with reviewed or automatic ICA choices.

Options:
  --config PATH       Configuration file (default: config/preprocessing.yaml)
  --env NAME          Conda environment (default: MNE_Roman)
  --overwrite         Replace previously generated outputs for the same subjects
  --skip-manual-ica-review
                      Automatically apply high-confidence ICLabel proposals.
                      This bypasses visual confirmation and is recorded in QC.
  -h, --help          Show this message

Source EEG is 500 Hz. Filtered, ICA, cleaned, and epoch data are 120 Hz.
Original files under dataset/ are never overwritten.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)
            [[ $# -ge 2 ]] || { echo "ERROR: --config requires a path" >&2; exit 2; }
            CONFIG_PATH="$2"
            shift 2
            ;;
        --env)
            [[ $# -ge 2 ]] || { echo "ERROR: --env requires a conda environment name" >&2; exit 2; }
            CONDA_ENV="$2"
            shift 2
            ;;
        --overwrite)
            OVERWRITE=true
            shift
            ;;
        --skip-manual-ica-review)
            SKIP_MANUAL_ICA_REVIEW=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "ERROR: unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ "$MODE" == "help" || "$MODE" == "-h" || "$MODE" == "--help" ]]; then
    usage
    exit 0
fi

case "$MODE" in
    review|clean|pilot) ;;
    *)
        echo "ERROR: mode must be review, clean, or pilot" >&2
        usage >&2
        exit 2
        ;;
esac

if [[ "$MODE" == "review" && "$SKIP_MANUAL_ICA_REVIEW" == true ]]; then
    echo "ERROR: --skip-manual-ica-review performs automatic removal and is only valid with clean or pilot." >&2
    exit 2
fi

command -v conda >/dev/null 2>&1 || {
    echo "ERROR: conda is not available on PATH" >&2
    exit 1
}

cd "$PROJECT_ROOT"

if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "ERROR: configuration file not found: $CONFIG_PATH" >&2
    exit 1
fi

# Keep MNE and Matplotlib caches out of the user's home. src/runtime.py applies
# the same defaults, but exporting them here also covers library startup.
export MNE_DONTWRITE_HOME=true
export MPLCONFIGDIR="${TMPDIR:-/tmp}/parkinson_eeg_mpl"
export XDG_CACHE_HOME="${TMPDIR:-/tmp}/parkinson_eeg_cache"
mkdir -p "$MPLCONFIGDIR"
mkdir -p "$XDG_CACHE_HOME"

run_python() {
    echo
    printf '+ conda run -n %q python' "$CONDA_ENV"
    printf ' %q' "$@"
    echo
    conda run -n "$CONDA_ENV" python "$@"
}

COMMON_ARGS=(--config "$CONFIG_PATH" --no-ica-downsampling)
if [[ "$OVERWRITE" == true ]]; then
    COMMON_ARGS+=(--overwrite)
fi
if [[ "$SKIP_MANUAL_ICA_REVIEW" == true ]]; then
    COMMON_ARGS+=(--skip-manual-ica-review)
fi

echo "Project:     $PROJECT_ROOT"
echo "Environment: $CONDA_ENV"
echo "Config:      $CONFIG_PATH"
echo "Mode:        $MODE"
echo "Sampling:    source 500 Hz; final and ICA 120 Hz"
echo "Overwrite:   $OVERWRITE"
echo "ICA review:  $([[ "$SKIP_MANUAL_ICA_REVIEW" == true ]] && echo 'automatic ICLabel proposals' || echo 'manual confirmation required')"

echo
echo "STEP 1/3 — Inspect dataset and preserve metadata"
run_python scripts/inspect_dataset.py --config "$CONFIG_PATH"

echo
echo "STEP 2/3 — Run preprocessing validation tests"
# Tests that read downstream analysis products cannot run during a clean
# bootstrap because those products are intentionally created only after this
# stage. The complete repository suite is run by run_all_analyses.sh after all
# downstream outputs exist.
run_python -m unittest -v \
    tests.test_cleaning \
    tests.test_config \
    tests.test_dataset \
    tests.test_ica \
    tests.test_simple_pipeline

echo
case "$MODE" in
    review)
        echo "STEP 3/3 — Generate ICA review material for every participant"
        run_python scripts/run_preprocessing.py --review-only "${COMMON_ARGS[@]}"
        echo
        echo "ICA review material is ready under processed/qc/<subject>/."
        echo "Inspect ranked stages 08–10. ICLabel proposals were prefilled in"
        echo "$CONFIG_PATH. Edit each list/reason as needed, then set that subject's"
        echo "manual_review_confirmed value to true. Use [] when nothing is removed."
        echo "Then run: bash scripts/run_full_cleaning.sh clean --overwrite"
        ;;
    clean)
        if [[ "$SKIP_MANUAL_ICA_REVIEW" == true ]]; then
            echo "STEP 3/3 — Run full cleaning with automatically accepted ICLabel proposals"
            echo "WARNING: visual ICA confirmation is being skipped and this will be recorded in QC."
        else
            echo "STEP 3/3 — Run the reviewed full cleaning pipeline"
        fi
        run_python scripts/run_preprocessing.py "${COMMON_ARGS[@]}"
        echo
        echo "Full cleaning completed. Outputs are under processed/."
        echo "Review processed/metadata/preprocessing_qc.csv before group analysis."
        ;;
    pilot)
        if [[ "$SKIP_MANUAL_ICA_REVIEW" == true ]]; then
            echo "STEP 3/3 — Run PD/Control pilot with automatically accepted ICLabel proposals"
        else
            echo "STEP 3/3 — Run the reviewed PD/Control pilot"
        fi
        run_python scripts/preprocess_test_set.py "${COMMON_ARGS[@]}"
        echo
        echo "Pilot completed for sub-001 and sub-101."
        ;;
esac
