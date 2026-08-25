#!/usr/bin/env bash

# Run every post-cleaning analysis in dependency order.
# EEG cleaning is intentionally excluded because manual ICA confirmation is a
# separate human-reviewed workflow (scripts/run_full_cleaning.sh).

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

OVERWRITE=false
DRY_RUN=false
NO_PROGRESS=false
SKIP_TESTS=false
SKIP_SWEEP=false
SKIP_EXPLORATION=false

usage() {
    cat <<'EOF'
Usage: bash run_all_analyses.sh [options]

Run the complete post-cleaning Parkinson EEG analysis pipeline:
  1. PSD analysis
  2. Primary ordinal analysis
  3. Ordinal D/tau parameter sweep
  4. Scale-free/specparam and bout-property analysis
  5. Within-bout ordinal analysis
  6. Specparam fit-QC bout and within-bout sensitivity
  7. Subject-balanced stereotypical bout gallery and detection QC
  8. Transparent PD-versus-Control exploration models
  9. D=3,4,5,6 quantitative-behavioral ordinal inputs
 10. MOCA quantitative-behavioral analysis

The default is resumable: a current completed stage is skipped. A stage whose
outputs predate the requested Rényi columns is automatically rerun.

Options:
  --overwrite          Rerun and replace every analysis stage
  --dry-run            Print the commands and freshness decisions only
  --no-progress        Disable progress bars where supported
  --skip-tests         Do not run repository validation tests first
  --skip-sweep         Skip the nine-run D/tau ordinal parameter sweep
  --skip-exploration   Skip the PD-versus-Control model exploration
  -h, --help           Show this help

Prerequisite:
  Complete cleaned epochs and metadata under processed/. Cleaning is not
  launched here because ICA review requires an explicit human decision.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --overwrite)
            OVERWRITE=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --no-progress)
            NO_PROGRESS=true
            shift
            ;;
        --skip-tests)
            SKIP_TESTS=true
            shift
            ;;
        --skip-sweep)
            SKIP_SWEEP=true
            shift
            ;;
        --skip-exploration)
            SKIP_EXPLORATION=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            printf 'ERROR: unknown option: %s\n' "$1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

command -v conda >/dev/null 2>&1 || {
    printf 'ERROR: conda is not available on PATH\n' >&2
    exit 1
}

[[ -f processed/metadata/subjects.csv ]] || {
    printf 'ERROR: missing processed/metadata/subjects.csv\n' >&2
    exit 1
}

epoch_count=$(find processed/epochs -maxdepth 1 -type f \
    -name 'sub-*_task-Rest_desc-cleaned_epo.fif' | wc -l | tr -d ' ')
if [[ "$epoch_count" -lt 1 ]]; then
    printf 'ERROR: no cleaned epoch files were found under processed/epochs/\n' >&2
    exit 1
fi

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

header_has_extended_renyi() {
    local filename="$1"
    [[ -f "$filename" ]] || return 1
    local header
    IFS= read -r header < "$filename"
    [[ "$header" == *"renyi_entropy_alpha_0_1"* ]] || return 1
    [[ "$header" == *"renyi_complexity_alpha_0_1"* ]] || return 1
    [[ "$header" == *"renyi_entropy_alpha_0_5"* ]] || return 1
    [[ "$header" == *"renyi_complexity_alpha_0_5"* ]] || return 1
    [[ "$header" == *"renyi_entropy_alpha_5"* ]] || return 1
    [[ "$header" == *"renyi_complexity_alpha_5"* ]] || return 1
}

psd_current() {
    [[ -f psd_analysis/processed/manifest.json ]]
}

ordinal_primary_current() {
    [[ -f ordinal_analysis/processed/manifest.json ]] || return 1
    header_has_extended_renyi \
        ordinal_analysis/processed/metrics/subject_electrode_mean_metrics.csv
    header_has_extended_renyi \
        ordinal_analysis/processed/metrics/band_subject_electrode_mean_metrics.csv
}

ordinal_sweep_current() {
    local dimension delay directory
    for dimension in 4 6 7; do
        for delay in 1 5 10; do
            directory="ordinal_analysis/parameter_sweep/D${dimension}_tau${delay}"
            [[ -f "${directory}/manifest.json" ]] || return 1
            header_has_extended_renyi \
                "${directory}/metrics/subject_electrode_mean_metrics.csv" || return 1
            header_has_extended_renyi \
                "${directory}/metrics/band_subject_electrode_mean_metrics.csv" || return 1
        done
    done
}

scale_free_current() {
    [[ -f scale_free_analysis/processed/manifest.json ]] || return 1
    [[ -f scale_free_analysis/processed/metrics/specparam_fit_qc_summary.csv ]] || return 1
    [[ -f scale_free_analysis/processed/metrics/subject_aperiodic_range_sensitivity.csv ]] || return 1
    [[ -f scale_free_analysis/processed/figures/aperiodic_diagnostics/group_median_decomposition_and_residuals.png ]] || return 1
    grep -q 'specparam_fit_qc' scale_free_analysis/processed/manifest.json || return 1
}

bout_current() {
    [[ -f bout_analyses/processed/manifest.json ]]
}

fit_qc_sensitivity_current() {
    [[ -f scale_free_analysis/processed/fit_qc_sensitivity_manifest.json ]] || return 1
    [[ -f scale_free_analysis/processed/metrics/subject_band_metrics_fit_qc.csv ]] || return 1
    [[ -f scale_free_analysis/processed/metrics/specparam_fit_failure_group_comparison.csv ]] || return 1
    [[ -f bout_analyses/processed/metrics/subject_band_metrics_fit_qc.csv ]] || return 1
    [[ -f bout_analyses/processed/figures/fit_qc_sensitivity/within_bout_ordinal_all_vs_fit_qc.png ]] || return 1
    [[ -f quantitative_behavioral/processed/fit_qc_sensitivity_manifest.json ]] || return 1
    [[ -f quantitative_behavioral/processed/metrics/fit_qc_sensitivity/subject_level_correlations.csv ]] || return 1
}

typical_bouts_current() {
    [[ -f scale_free_analysis/processed/typical_bouts_manifest.json ]] || return 1
    [[ -f scale_free_analysis/processed/metrics/typical_bout_coverage.csv ]] || return 1
    [[ -f scale_free_analysis/processed/metrics/typical_bout_group_coverage.csv ]] || return 1
    [[ -f scale_free_analysis/processed/figures/typical_bouts/index.html ]] || return 1
    [[ -f scale_free_analysis/processed/figures/typical_bouts/grand_average_all_subjects.png ]] || return 1
    [[ -f scale_free_analysis/processed/figures/typical_bouts/grand_average_fit_qc.png ]] || return 1
    [[ -f scale_free_analysis/processed/figures/typical_bouts/bout_detection_subject_coverage.png ]] || return 1
}

exploration_current() {
    [[ -f exploration/processed/manifest.json ]]
}

quantitative_current() {
    [[ -f quantitative_behavioral/processed/manifest.json ]] || return 1
    local primary_dictionary="quantitative_behavioral/processed/metrics/feature_dictionary.csv"
    local dictionary="quantitative_behavioral/processed/metrics/dimension_sensitivity_feature_dictionary.csv"
    [[ -f "$primary_dictionary" ]] || return 1
    [[ -f "$dictionary" ]] || return 1
    grep -q 'aperiodic_exponent' "$primary_dictionary" || return 1
    grep -q 'aperiodic_exponent_qc' "$primary_dictionary" || return 1
    grep -q 'renyi_entropy_alpha_0_1' "$dictionary" || return 1
    grep -q 'renyi_complexity_alpha_0_1' "$dictionary" || return 1
    grep -q 'renyi_entropy_alpha_0_5' "$dictionary" || return 1
    grep -q 'renyi_complexity_alpha_0_5' "$dictionary" || return 1
    grep -q 'renyi_entropy_alpha_5' "$dictionary" || return 1
    grep -q 'renyi_complexity_alpha_5' "$dictionary" || return 1
}

run_stage() {
    local label="$1"
    local validator="$2"
    local sentinel="$3"
    shift 3
    local -a command=("$@")

    printf '\n=== %s ===\n' "$label"
    if [[ "$OVERWRITE" == false ]] && "$validator"; then
        printf '  current output found; skipping\n'
        return
    fi
    if [[ "$OVERWRITE" == true || -e "$sentinel" ]]; then
        command+=(--overwrite)
    fi
    execute "${command[@]}"
}

printf 'Project: %s\n' "$SCRIPT_DIR"
printf 'Cleaned epochs found: %s\n' "$epoch_count"
printf 'Mode: %s\n' "$([[ "$OVERWRITE" == true ]] && printf overwrite || printf resume)"

if [[ "$SKIP_TESTS" == false ]]; then
    printf '\n=== Repository tests ===\n'
    execute conda run -n MNE_Roman python -m unittest discover -s tests
fi

run_stage "PSD analysis" psd_current psd_analysis/processed/manifest.json \
    bash psd_analysis/run_psd_analysis.sh

ordinal_primary_command=(bash ordinal_analysis/run_ordinal_analysis.sh)
if [[ "$NO_PROGRESS" == true ]]; then
    ordinal_primary_command+=(--no-progress)
fi
run_stage "Primary ordinal analysis" ordinal_primary_current \
    ordinal_analysis/processed/manifest.json "${ordinal_primary_command[@]}"

if [[ "$SKIP_SWEEP" == false ]]; then
    ordinal_sweep_command=(bash ordinal_analysis/run_ordinal_parameter_sweep.sh)
    if [[ "$NO_PROGRESS" == true ]]; then
        ordinal_sweep_command+=(--no-progress)
    fi
    run_stage "Ordinal D/tau parameter sweep" ordinal_sweep_current \
        ordinal_analysis/parameter_sweep/D7_tau10/manifest.json \
        "${ordinal_sweep_command[@]}"
else
    printf '\n=== Ordinal D/tau parameter sweep ===\n  skipped by request\n'
fi

scale_free_command=(bash scale_free_analysis/run_scale_free_analysis.sh)
if [[ "$NO_PROGRESS" == true ]]; then
    scale_free_command+=(--no-progress)
fi
run_stage "Scale-free and bout-property analysis" scale_free_current \
    scale_free_analysis/processed/manifest.json "${scale_free_command[@]}"

bout_command=(bash bout_analyses/run_bout_analyses.sh)
if [[ "$NO_PROGRESS" == true ]]; then
    bout_command+=(--no-progress)
fi
run_stage "Within-bout ordinal analysis" bout_current \
    bout_analyses/processed/manifest.json "${bout_command[@]}"

run_stage "Specparam fit-QC bout sensitivity" fit_qc_sensitivity_current \
    scale_free_analysis/processed/fit_qc_sensitivity_manifest.json \
    bash scale_free_analysis/run_fit_qc_sensitivity.sh

run_stage "Stereotypical bout gallery and detection QC" typical_bouts_current \
    scale_free_analysis/processed/typical_bouts_manifest.json \
    bash scale_free_analysis/generate_typical_bouts.sh

if [[ "$SKIP_EXPLORATION" == false ]]; then
    run_stage "PD-versus-Control model exploration" exploration_current \
        exploration/processed/manifest.json \
        bash exploration/run_exploration.sh
else
    printf '\n=== PD-versus-Control model exploration ===\n  skipped by request\n'
fi

printf '\n=== D=3,4,5,6 quantitative-behavioral ordinal inputs ===\n'
dimension_input_command=(bash quantitative_behavioral/prepare_dimension_sensitivity.sh)
if [[ "$OVERWRITE" == true ]]; then
    dimension_input_command+=(--overwrite)
fi
execute "${dimension_input_command[@]}"

run_stage "MOCA quantitative-behavioral analysis" quantitative_current \
    quantitative_behavioral/processed/manifest.json \
    bash quantitative_behavioral/run_quantitative_behavioral.sh

printf '\nAll requested analysis stages completed successfully.\n'
