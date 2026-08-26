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
SKIP_MATCHED=false
CONDA_ENV="${PARKINSON_EEG_CONDA_ENV:-MNE_August2026}"

usage() {
    cat <<'EOF'
Usage: bash run_all_analyses.sh [options]

Run the complete post-cleaning Parkinson EEG analysis pipeline:
  1. PSD analysis
  2. Primary ordinal analysis
  3. Ordinal embedding-dimension sweep at tau=1
  4. Scale-free/specparam and bout-property analysis
  5. Independent bycycle cycle-consistency burst sensitivity analysis
  6. Within-bout ordinal analysis
  7. Eight-electrode non-progression sensitivity battery
  8. Specparam fit-QC bout and within-bout sensitivity
  9. Subject-balanced stereotypical bout gallery and detection QC
 10. Transparent full-cohort PD-versus-Control exploration models
 11. D=3,4,5,6 quantitative-behavioral ordinal inputs
 12. MOCA quantitative-behavioral analysis
 13. Whole-head UPDRS/MOCA disease-severity analysis
 14. Accepted-duration sensitivity requiring at least 60 seconds
 15. The same complete battery on one canonical age/sex-matched cohort

The default is resumable: a current completed stage is skipped. A stage whose
outputs predate the requested Rényi columns is automatically rerun.

Options:
  --overwrite          Rerun and replace every analysis stage
  --dry-run            Print the commands and freshness decisions only
  --no-progress        Disable progress bars where supported
  --skip-tests         Do not run repository integration tests after analyses
  --skip-sweep         Skip the D={3,4,5,6}, tau=1 ordinal sensitivity sweep
  --skip-exploration   Skip the PD-versus-Control model exploration
  --skip-matched       Skip the complete matched-cohort sensitivity pipeline
  --env NAME           Conda environment (default: MNE_August2026)
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
        --skip-matched)
            SKIP_MATCHED=true
            shift
            ;;
        --env)
            [[ $# -ge 2 ]] || { printf 'ERROR: --env requires a name\n' >&2; exit 2; }
            CONDA_ENV="$2"
            shift 2
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

export PARKINSON_EEG_CONDA_ENV="$CONDA_ENV"
environment_arguments=(--env "$CONDA_ENV")
if [[ "$DRY_RUN" == true ]]; then environment_arguments+=(--dry-run); fi
bash scripts/ensure_conda_environment.sh "${environment_arguments[@]}"

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
    [[ "$header" == *"renyi_entropy_alpha_10"* ]] || return 1
    [[ "$header" == *"renyi_complexity_alpha_10"* ]] || return 1
}

has_eight_electrodes() {
    local filename="$1" electrode
    [[ -f "$filename" ]] || return 1
    for electrode in F4 P4 O2 P6 CP2 CP1 PO7 P8; do
        grep -q "^${electrode}," "$filename" || return 1
    done
}

psd_current() {
    [[ -f psd_analysis/processed/manifest.json ]] || return 1
    [[ -f psd_analysis/processed/metrics/group_subject_statistics.csv ]] || return 1
    [[ -f psd_analysis/processed/metrics/group_electrode_statistics.csv ]] || return 1
    [[ -f psd_analysis/processed/figures/group_statistics/relative_band_power_group_statistics.png ]]
}

ordinal_primary_current() {
    [[ -f ordinal_analysis/processed/manifest.json ]] || return 1
    header_has_extended_renyi \
        ordinal_analysis/processed/metrics/subject_electrode_mean_metrics.csv
    header_has_extended_renyi \
        ordinal_analysis/processed/metrics/band_subject_electrode_mean_metrics.csv
    [[ -f ordinal_analysis/processed/figures/topomaps/renyi_alpha_0_1/group_mean_topomaps.png ]] || return 1
    [[ -f ordinal_analysis/processed/figures/topomaps/renyi_alpha_10/group_mean_zscored_topomaps.png ]] || return 1
    [[ -f ordinal_analysis/processed/figures/bands/topomaps/renyi_alpha_10/group_means/alpha_group_mean_topomaps.png ]] || return 1
    [[ -f ordinal_analysis/processed/metrics/group_subject_statistics_broadband.csv ]] || return 1
    [[ -f ordinal_analysis/processed/metrics/group_electrode_statistics_by_band.csv ]] || return 1
    [[ -f ordinal_analysis/processed/figures/group_statistics/broadband/entropy_group_statistics.png ]] || return 1
}

ordinal_sweep_current() {
    local dimension directory
    for dimension in 3 4 5 6; do
        directory="ordinal_analysis/parameter_sweep/D${dimension}_tau1"
        [[ -f "${directory}/manifest.json" ]] || return 1
        header_has_extended_renyi \
            "${directory}/metrics/subject_electrode_mean_metrics.csv" || return 1
        header_has_extended_renyi \
            "${directory}/metrics/band_subject_electrode_mean_metrics.csv" || return 1
        [[ -f "${directory}/metrics/group_subject_statistics_broadband.csv" ]] || return 1
        [[ -f "${directory}/metrics/group_electrode_statistics_by_band.csv" ]] || return 1
    done
}

scale_free_current() {
    [[ -f scale_free_analysis/processed/manifest.json ]] || return 1
    grep -q '"broad_5_15"' scale_free_analysis/processed/manifest.json || return 1
    [[ -f scale_free_analysis/processed/metrics/specparam_fit_qc_summary.csv ]] || return 1
    [[ -f scale_free_analysis/processed/metrics/subject_aperiodic_range_sensitivity.csv ]] || return 1
    [[ -f scale_free_analysis/processed/figures/aperiodic_diagnostics/group_median_decomposition_and_residuals.png ]] || return 1
    grep -q 'specparam_fit_qc' scale_free_analysis/processed/manifest.json || return 1
    [[ -f scale_free_analysis/processed/metrics/group_subject_statistics_aperiodic.csv ]] || return 1
    [[ -f scale_free_analysis/processed/metrics/group_electrode_statistics_periodic_bout.csv ]] || return 1
    [[ -f scale_free_analysis/processed/figures/group_statistics/aperiodic/aperiodic_exponent_group_statistics.png ]] || return 1
}

specparam_gallery_current() {
    local root="scale_free_analysis/processed/figures/specparam_decomposition"
    local metrics="scale_free_analysis/processed/metrics/electrode_aperiodic_metrics.csv"
    local manifest="scale_free_analysis/processed/manifest.json"
    [[ -f "$manifest" && -f "$metrics" && -f "$root/index.html" \
        && -f "$root/figure_index.csv" ]] || return 1
    grep -q 'flat_single_folder_one_all_electrode_png_per_subject' "$manifest" \
        || return 1
    [[ -z "$(find "$root" -mindepth 1 -type d -print -quit)" ]] || return 1
    local expected actual
    expected=$(awk -F, 'NR > 1 { subjects[$1] = 1 } END { print length(subjects) }' "$metrics")
    actual=$(find "$root" -maxdepth 1 -type f -name 'sub-*_all_electrodes.png' | wc -l | tr -d ' ')
    [[ "$actual" -eq "$expected" ]]
}

bycycle_burst_current() {
    [[ -f bycycle_burst_analysis/processed/manifest.json ]] || return 1
    [[ -f bycycle_burst_analysis/processed/metrics/subject_electrode_band_metrics.csv ]] || return 1
    [[ -f bycycle_burst_analysis/processed/metrics/group_subject_statistics.csv ]] || return 1
    [[ -f bycycle_burst_analysis/processed/metrics/group_electrode_statistics.csv ]] || return 1
    [[ -f bycycle_burst_analysis/processed/metrics/detector_event_agreement.csv ]] || return 1
    [[ -f bycycle_burst_analysis/processed/figures/qc/independent_detection_example.png ]] || return 1
    [[ -f bycycle_burst_analysis/processed/figures/agreement/event_mask_dice.png ]] || return 1
    grep -q '"broad_5_15"' bycycle_burst_analysis/processed/manifest.json
}

bycycle_group_figures_current() {
    local root="bycycle_burst_analysis/processed"
    [[ -f "$root/manifest.json" ]] || return 1
    grep -q '"n_subject_average_violin_figures": 12' "$root/manifest.json" \
        || return 1
    [[ -f "$root/figures/group_comparisons/group_bout_duration_mean_s.png" ]] \
        && [[ -f "$root/figures/group_comparisons/group_bouts_per_minute.png" ]]
}

bout_current() {
    [[ -f bout_analyses/processed/manifest.json ]] || return 1
    grep -q '"broad_5_15"' bout_analyses/processed/manifest.json || return 1
    [[ -f bout_analyses/processed/metrics/group_subject_statistics.csv ]] || return 1
    [[ -f bout_analyses/processed/metrics/group_electrode_statistics.csv ]] || return 1
    [[ -f bout_analyses/processed/figures/group_statistics/entropy_group_statistics.png ]]
}

fit_qc_sensitivity_current() {
    [[ -f scale_free_analysis/processed/fit_qc_sensitivity_manifest.json ]] || return 1
    [[ -f scale_free_analysis/processed/metrics/subject_band_metrics_fit_qc.csv ]] || return 1
    [[ -f scale_free_analysis/processed/metrics/specparam_fit_failure_group_comparison.csv ]] || return 1
    [[ -f bout_analyses/processed/metrics/subject_band_metrics_fit_qc.csv ]] || return 1
    [[ -f bout_analyses/processed/figures/fit_qc_sensitivity/within_bout_ordinal_all_vs_fit_qc.png ]] || return 1
    [[ -f quantitative_behavioral/processed/fit_qc_sensitivity_manifest.json ]] || return 1
    [[ -f quantitative_behavioral/processed/metrics/fit_qc_sensitivity/subject_level_correlations.csv ]] || return 1
    grep -q 'broad_5_15' scale_free_analysis/processed/metrics/subject_band_metrics_fit_qc.csv || return 1
    grep -q 'broad_5_15' bout_analyses/processed/metrics/subject_band_metrics_fit_qc.csv || return 1
}

typical_bouts_current() {
    [[ -f scale_free_analysis/processed/typical_bouts_manifest.json ]] || return 1
    [[ -f scale_free_analysis/processed/metrics/typical_bout_coverage.csv ]] || return 1
    [[ -f scale_free_analysis/processed/metrics/typical_bout_group_coverage.csv ]] || return 1
    [[ -f scale_free_analysis/processed/figures/typical_bouts/index.html ]] || return 1
    [[ -f scale_free_analysis/processed/figures/typical_bouts/grand_average_all_subjects.png ]] || return 1
    [[ -f scale_free_analysis/processed/figures/typical_bouts/grand_average_fit_qc.png ]] || return 1
    [[ -f scale_free_analysis/processed/figures/typical_bouts/bout_detection_subject_coverage.png ]] || return 1
    grep -q 'broad_5_15' scale_free_analysis/processed/typical_bouts_manifest.json || return 1
}

exploration_current() {
    [[ -f exploration/processed/manifest.json ]] || return 1
    [[ -f exploration/processed/MODEL_REVISION.md ]] || return 1
    [[ -f exploration/processed/metrics/auc_differences_vs_psd.csv ]] || return 1
    [[ -f exploration/processed/models/typical_bout_shape_adjusted.joblib ]] || return 1
    [[ -f exploration/processed/figures/features/versus_age/age_scatter_page_001.png ]] || return 1
    grep -q 'ordinal_global_renyi_entropy_alpha_0_1' \
        exploration/processed/features/subject_modeling_table.csv || return 1
    grep -q 'ordinal_global_renyi_entropy_alpha_10' \
        exploration/processed/features/subject_modeling_table.csv || return 1
    grep -q 'bout_alpha_oscillatory_occupancy' \
        exploration/processed/features/subject_modeling_table.csv || return 1
    ! grep -q 'broad_5_15' \
        exploration/processed/features/subject_modeling_table.csv || return 1
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
    grep -q 'renyi_entropy_alpha_10' "$dictionary" || return 1
    grep -q 'renyi_complexity_alpha_10' "$dictionary" || return 1
    grep -q 'bout_alpha_bouts_per_minute' "$primary_dictionary" || return 1
    ! grep -q 'broad_5_15' "$primary_dictionary" || return 1
    ! grep -q 'broad_5_15' "$dictionary" || return 1
}

disease_progression_current() {
    [[ -f disease_progression/processed/manifest.json ]] || return 1
    [[ -f disease_progression/processed/REPORT.md ]] || return 1
    [[ -f disease_progression/processed/metrics/progression_correlations.csv ]] || return 1
    [[ -f disease_progression/processed/figures/electrode_selection.png ]] || return 1
    [[ -f disease_progression/processed/figures/scatter/updrs/updrs_scatter_page_001.png ]] || return 1
    [[ "$(($(wc -l < disease_progression/processed/metrics/electrode_selection.csv) - 1))" -eq 60 ]] || return 1
    ! grep -q 'broad_5_15' \
        disease_progression/processed/metrics/feature_dictionary.csv || return 1
}

eight_electrode_current() {
    [[ -f eight_electrode_analysis/processed/manifest.json ]] || return 1
    [[ -f eight_electrode_analysis/processed/REPORT.md ]] || return 1
    [[ -f eight_electrode_analysis/processed/metrics/group_subject_statistics.csv ]] || return 1
    [[ -f eight_electrode_analysis/processed/metrics/group_electrode_statistics.csv ]] || return 1
    has_eight_electrodes \
        eight_electrode_analysis/processed/metrics/electrode_selection.csv || return 1
    ! grep -q 'broad_5_15' \
        eight_electrode_analysis/processed/metrics/feature_dictionary.csv || return 1
}

duration_qc_current() {
    [[ -f duration_qc_analysis/processed/manifest.json ]] || return 1
    grep -q '"minimum_accepted_duration_seconds": 60' \
        duration_qc_analysis/processed/manifest.json || return 1
    [[ -f duration_qc_analysis/processed/REPORT.md ]] || return 1
    [[ -f duration_qc_analysis/processed/metrics/group_comparisons.csv ]] || return 1
    [[ -f duration_qc_analysis/processed/metrics/moca_correlations.csv ]] || return 1
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
    # Reaching this point means the stage is absent, partial, or stale. Always
    # allow the stage runner to replace remnants, even when its manifest was
    # never written (for example after an interrupted analysis).
    command+=(--overwrite)
    execute "${command[@]}"
}

printf 'Project: %s\n' "$SCRIPT_DIR"
printf 'Cleaned epochs found: %s\n' "$epoch_count"
printf 'Mode: %s\n' "$([[ "$OVERWRITE" == true ]] && printf overwrite || printf resume)"

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
    printf '\n=== Ordinal embedding-dimension sweep (tau=1) ===\n'
    if [[ "$OVERWRITE" == false ]] && ordinal_sweep_current; then
        printf '  current output found; skipping\n'
    else
        # A non-current sweep can contain partial D directories without their
        # manifests; rebuild the sweep rather than treating those files as new.
        ordinal_sweep_command+=(--overwrite)
        execute "${ordinal_sweep_command[@]}"
    fi
else
    printf '\n=== Ordinal embedding-dimension sweep (tau=1) ===\n  skipped by request\n'
fi

scale_free_command=(bash scale_free_analysis/run_scale_free_analysis.sh)
if [[ "$NO_PROGRESS" == true ]]; then
    scale_free_command+=(--no-progress)
fi
if [[ "$OVERWRITE" == false \
    && -f scale_free_analysis/processed/figures/specparam_decomposition/index.html \
    && -f scale_free_analysis/processed/manifest.json ]] \
    && ! grep -q '"broad_5_15"' scale_free_analysis/processed/manifest.json; then
    scale_free_command+=(--skip-specparam-gallery)
fi
run_stage "Scale-free and bout-property analysis" scale_free_current \
    scale_free_analysis/processed/manifest.json "${scale_free_command[@]}"

printf '\n=== Flat all-electrode specparam gallery ===\n'
if specparam_gallery_current; then
    printf '  current output found; skipping\n'
else
    execute bash scale_free_analysis/generate_specparam_figures.sh --overwrite
fi

bycycle_burst_command=(bash bycycle_burst_analysis/run_bycycle_burst_analysis.sh)
if [[ "$NO_PROGRESS" == true ]]; then
    bycycle_burst_command+=(--no-progress)
fi
run_stage "Independent bycycle burst-detection sensitivity" bycycle_burst_current \
    bycycle_burst_analysis/processed/manifest.json "${bycycle_burst_command[@]}"

printf '\n=== Independent bycycle subject-average violin figures ===\n'
if bycycle_group_figures_current; then
    printf '  current output found; skipping\n'
else
    execute bash bycycle_burst_analysis/generate_group_figures.sh
fi

bout_command=(bash bout_analyses/run_bout_analyses.sh)
if [[ "$NO_PROGRESS" == true ]]; then
    bout_command+=(--no-progress)
fi
run_stage "Within-bout ordinal analysis" bout_current \
    bout_analyses/processed/manifest.json "${bout_command[@]}"

run_stage "Eight-electrode non-progression sensitivity battery" \
    eight_electrode_current eight_electrode_analysis/processed/manifest.json \
    bash eight_electrode_analysis/run_eight_electrode_analysis.sh

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

run_stage "Whole-head UPDRS/MOCA disease-severity analysis" \
    disease_progression_current disease_progression/processed/manifest.json \
    bash disease_progression/run_disease_progression.sh

if [[ "$SKIP_EXPLORATION" == false ]]; then
    run_stage "accepted-duration QC sensitivity (at least 60 seconds)" \
        duration_qc_current duration_qc_analysis/processed/manifest.json \
        bash duration_qc_analysis/run_duration_qc_sensitivity.sh
else
    printf '\n=== Accepted-duration QC sensitivity ===\n  skipped because exploration was skipped\n'
fi

if [[ "$SKIP_MATCHED" == false ]]; then
    printf '\n=== Complete matched-cohort sensitivity pipeline ===\n'
    matched_command=(bash matched_analysis/run_matched_analyses.sh --env "$CONDA_ENV")
    if [[ "$OVERWRITE" == true ]]; then matched_command+=(--overwrite); fi
    if [[ "$DRY_RUN" == true ]]; then matched_command+=(--dry-run); fi
    if [[ "$NO_PROGRESS" == true ]]; then matched_command+=(--no-progress); fi
    if [[ "$SKIP_SWEEP" == true ]]; then matched_command+=(--skip-sweep); fi
    if [[ "$SKIP_EXPLORATION" == true ]]; then
        matched_command+=(--skip-exploration)
    fi
    execute "${matched_command[@]}"
else
    printf '\n=== Complete matched-cohort sensitivity pipeline ===\n  skipped by request\n'
fi

if [[ "$SKIP_TESTS" == false ]]; then
    printf '\n=== Repository integration tests ===\n'
    execute conda run -n "$CONDA_ENV" python -m unittest discover -s tests
fi

printf '\nAll requested analysis stages completed successfully.\n'
