#!/usr/bin/env bash

# Run every downstream analysis on one canonical exact-sex/optimal-age cohort.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "$PROJECT_ROOT"

OVERWRITE=false
DRY_RUN=false
NO_PROGRESS=false
SKIP_SWEEP=false
SKIP_EXPLORATION=false

usage() {
    cat <<'EOF'
Usage: bash matched_analysis/run_matched_analyses.sh [options]

Prepare one canonical exact-sex/optimal-age matched cohort, then run matched:
PSD, ordinal quantities/planes/topomaps, ordinal D/tau inputs, scale-free,
bouts, fit-QC sensitivity, typical bouts, prediction models, and MOCA analyses.

Options:
  --overwrite          Regenerate every matched result
  --dry-run            Print commands only
  --no-progress        Disable supported progress bars
  --skip-sweep         Skip ordinal sensitivity inputs and MOCA analysis
  --skip-exploration   Skip matched prediction models
  -h, --help           Show this help

Signal cleaning is not repeated: it is an independent subject-level operation.
All matched outputs use *_matched directories and cannot replace full-cohort results.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --overwrite) OVERWRITE=true ;;
        --dry-run) DRY_RUN=true ;;
        --no-progress) NO_PROGRESS=true ;;
        --skip-sweep) SKIP_SWEEP=true ;;
        --skip-exploration) SKIP_EXPLORATION=true ;;
        -h|--help) usage; exit 0 ;;
        *) printf 'ERROR: unknown option: %s\n' "$1" >&2; usage >&2; exit 2 ;;
    esac
    shift
done

print_command() {
    printf '  +'
    printf ' %q' "$@"
    printf '\n'
}

execute() {
    print_command "$@"
    if [[ "$DRY_RUN" == false ]]; then
        "$@"
    fi
}

run_stage() {
    local label="$1"
    local sentinel="$2"
    shift 2
    local -a command=("$@")
    printf '\n=== Matched: %s ===\n' "$label"
    if [[ "$OVERWRITE" == false && -f "$sentinel" ]]; then
        printf '  current output found; skipping\n'
        return
    fi
    if [[ "$OVERWRITE" == true || -e "$sentinel" ]]; then
        command+=(--overwrite)
    fi
    execute "${command[@]}"
}

CONFIG_ROOT="matched_analysis/processed/configs"
MATCHED_PARTICIPANTS="matched_analysis/processed/matched_subjects.csv"

printf '\n=== Canonical matched cohort and configs ===\n'
execute bash matched_analysis/prepare_matched_cohort.sh

run_stage "PSD analysis" psd_analysis/processed_matched/manifest.json \
    bash psd_analysis/run_psd_analysis.sh --config "$CONFIG_ROOT/psd.json"

ordinal_command=(
    bash ordinal_analysis/run_ordinal_analysis.sh
    --config "$CONFIG_ROOT/ordinal.json"
)
if [[ "$NO_PROGRESS" == true ]]; then ordinal_command+=(--no-progress); fi
run_stage "ordinal metrics, all-alpha planes, and all-alpha topomaps" \
    ordinal_analysis/processed_matched/manifest.json "${ordinal_command[@]}"

if [[ "$SKIP_SWEEP" == false ]]; then
    sweep_command=(bash ordinal_analysis/run_ordinal_parameter_sweep.sh)
    if [[ "$NO_PROGRESS" == true ]]; then sweep_command+=(--no-progress); fi
    printf '\n=== Matched: ordinal D/tau parameter sweep ===\n'
    if [[ "$OVERWRITE" == false \
        && -f ordinal_analysis/parameter_sweep_matched/D7_tau10/manifest.json ]]; then
        printf '  current output found; skipping\n'
    else
        if [[ "$OVERWRITE" == true ]]; then sweep_command+=(--overwrite); fi
        print_command env \
            ORDINAL_BASE_CONFIG="$CONFIG_ROOT/ordinal.json" \
            ORDINAL_SWEEP_OUTPUT_ROOT=ordinal_analysis/parameter_sweep_matched \
            "${sweep_command[@]}"
        if [[ "$DRY_RUN" == false ]]; then
            env ORDINAL_BASE_CONFIG="$CONFIG_ROOT/ordinal.json" \
                ORDINAL_SWEEP_OUTPUT_ROOT=ordinal_analysis/parameter_sweep_matched \
                "${sweep_command[@]}"
        fi
    fi
else
    printf '\n=== Matched: ordinal D/tau parameter sweep ===\n  skipped by request\n'
fi

scale_command=(
    bash scale_free_analysis/run_scale_free_analysis.sh
    --config "$CONFIG_ROOT/scale_free.json"
)
if [[ "$NO_PROGRESS" == true ]]; then scale_command+=(--no-progress); fi
run_stage "scale-free and bout-property analysis" \
    scale_free_analysis/processed_matched/manifest.json "${scale_command[@]}"

bout_command=(
    bash bout_analyses/run_bout_analyses.sh
    --config "$CONFIG_ROOT/bout.json"
)
if [[ "$NO_PROGRESS" == true ]]; then bout_command+=(--no-progress); fi
run_stage "within-bout ordinal analysis" \
    bout_analyses/processed_matched/manifest.json "${bout_command[@]}"

run_stage "specparam fit-QC bout sensitivity" \
    scale_free_analysis/processed_matched/fit_qc_sensitivity_manifest.json \
    bash scale_free_analysis/run_fit_qc_sensitivity.sh \
    --scale-free-output scale_free_analysis/processed_matched \
    --bout-ordinal-output bout_analyses/processed_matched \
    --participants "$MATCHED_PARTICIPANTS" \
    --behavioral-config "$CONFIG_ROOT/quantitative_behavioral.json" \
    --behavioral-scale-free-qc-subject-file \
        scale_free_analysis/processed_matched/metrics/subject_band_metrics_fit_qc.csv \
    --behavioral-bout-ordinal-qc-subject-file \
        bout_analyses/processed_matched/metrics/subject_band_metrics_fit_qc.csv

run_stage "stereotypical bout gallery and detection QC" \
    scale_free_analysis/processed_matched/typical_bouts_manifest.json \
    bash scale_free_analysis/generate_typical_bouts.sh \
    --config "$CONFIG_ROOT/scale_free.json"

if [[ "$SKIP_EXPLORATION" == false ]]; then
    exploration_command=(
        bash exploration/run_exploration.sh
        --config "$CONFIG_ROOT/exploration.json"
        --matched-demographics
    )
    exploration_manifest=exploration/processed_matched/manifest.json
    if [[ -f "$exploration_manifest" ]] \
        && ! grep -q 'matched_analysis/processed/matched_subjects.csv' \
            "$exploration_manifest"; then
        printf '\n=== Matched: PD-versus-Control prediction models ===\n'
        printf '  legacy matched output used full-cohort feature sources; rebuilding\n'
        exploration_command+=(--overwrite)
        execute "${exploration_command[@]}"
    else
        run_stage "PD-versus-Control prediction models" \
            "$exploration_manifest" "${exploration_command[@]}"
    fi
else
    printf '\n=== Matched: PD-versus-Control prediction models ===\n  skipped by request\n'
fi

if [[ "$SKIP_SWEEP" == false ]]; then
    printf '\n=== Matched: D=3,4,5,6 quantitative-behavioral ordinal inputs ===\n'
    dimension_command=(bash quantitative_behavioral/prepare_dimension_sensitivity.sh)
    if [[ "$OVERWRITE" == true ]]; then dimension_command+=(--overwrite); fi
    print_command env \
        ORDINAL_BASE_CONFIG="$CONFIG_ROOT/ordinal.json" \
        ORDINAL_SWEEP_OUTPUT_ROOT=ordinal_analysis/parameter_sweep_matched \
        "${dimension_command[@]}"
    if [[ "$DRY_RUN" == false ]]; then
        env ORDINAL_BASE_CONFIG="$CONFIG_ROOT/ordinal.json" \
            ORDINAL_SWEEP_OUTPUT_ROOT=ordinal_analysis/parameter_sweep_matched \
            "${dimension_command[@]}"
    fi
    run_stage "MOCA quantitative-behavioral analysis" \
        quantitative_behavioral/processed_matched/manifest.json \
        bash quantitative_behavioral/run_quantitative_behavioral.sh \
        --config "$CONFIG_ROOT/quantitative_behavioral.json"
fi

printf '\nAll matched-cohort analysis stages completed successfully.\n'
