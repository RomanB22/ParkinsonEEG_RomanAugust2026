# Transparent PD versus Control exploration

This folder contains a conservative, subject-level classification analysis for
the 149-subject resting-EEG cohort. Its purpose is to test whether ordinal
quantities distinguish PD from Control and whether classical PSD or limited
clinical metadata adds information. It is an internally validated exploratory
model, not a clinically validated diagnostic system.

## Prespecified outcome and exclusions

The binary outcome is `PD=1`, `Control=0`. Every model contains exactly one row
per subject. Electrode-level measurements are summarized within subject by the
upstream ordinal and PSD pipelines; electrodes are never treated as independent
participants.

The following metadata are deliberately unavailable to every model:

| Variable | Reason |
|---|---|
| `participant_id` | Join key only |
| `ID` | Administrative identifier that may encode enrollment structure |
| `EEG` | Recording identifier that may encode group or acquisition order |
| `TYPE` | Perfectly reproduces diagnosis in this cohort |
| `UPDRS` | Missing for every Control, so missingness reveals diagnosis |
| `GROUP` | The classification outcome itself |

Age and sex form the demographic baseline. MOCA is used only in the explicitly
labeled clinical-extension model because cognitive status can itself carry
disease-related information.

## Feature definitions

The primary ordinal representation is the prespecified default `D=6, tau=1`
whole-head median H, C, and F output:

- permutation entropy H;
- statistical complexity C;
- Fisher information F.

The larger band-ordinal sensitivity model uses H, C, and F for theta, alpha,
and beta. Delta, low gamma, and the overlapping broad 5–15 Hz band do not enter
that model, limiting dimensionality and overlap.

PSD uses the subject median across shared electrodes. Relative powers are
compositional, so the model does not enter all percentages independently. It
uses four interpretable base-2 log ratios against low gamma:

```text
log2(delta / low_gamma)
log2(theta / low_gamma)
log2(alpha / low_gamma)
log2(beta / low_gamma)
```

A one-unit increase means a doubling of the numerator-to-low-gamma ratio. The
overlapping 5–15 Hz PSD band is excluded.

## Models

All models are standardized L2-regularized logistic regressions. There is no
PCA, automated univariate screening, SMOTE, synthetic subject generation,
random forest, boosting, or neural network.

| Model | Role | Predictors |
|---|---|---|
| Demographics | Baseline | Age, sex |
| Ordinal H/C/F | Primary unadjusted | Global H, C, F |
| Ordinal H/C/F + demographics | Primary | Age, sex, global H, C, F |
| PSD + demographics | Secondary | Age, sex, four PSD log ratios |
| Ordinal + PSD + demographics | Secondary | Global H/C/F, PSD ratios, age, sex |
| Band ordinal + demographics | Sensitivity | Theta/alpha/beta H/C/F, age, sex |
| Clinical extension | Secondary clinical | Ordinal + PSD + age + sex + MOCA |

The ridge strength is the only learned hyperparameter. Standardized
coefficients, native-unit coefficients, odds ratios, and coefficient stability
across validation folds are saved.

## Validation

The default run uses repeated nested stratified cross-validation:

1. Ten repetitions of five outer folds estimate generalization.
2. Four inner folds select ridge `C` using ROC AUC.
3. Scaling is fitted only on each training fold.
4. Every model receives the same outer subject splits.
5. Each subject receives ten genuinely out-of-fold probabilities, which are
   averaged to one final out-of-fold probability.
6. Subject-stratified bootstrap resampling gives 95% uncertainty intervals.
7. Fixed-`C` label-permutation tests provide a chance benchmark.

Reported metrics are ROC AUC, precision-recall AUC, balanced accuracy,
sensitivity, specificity, Brier score, and log loss. Within each outer split,
the decision threshold maximizes Youden's J statistic using only inner
out-of-fold predictions from the outer training subjects. A test subject never
influences its threshold. Model AUC differences use paired subject bootstrap
samples against the demographic baseline.

The all-subject `.joblib` models are descriptive final fits. They are not used
to claim independent performance; only out-of-fold predictions support the
reported validation metrics.

## Ordinal parameter sensitivity

Completed outputs under `ordinal_analysis/parameter_sweep/` are discovered
automatically. The global H/C/F model is refitted independently for each
completed combination of `D={4,6,7}` and `tau={1,5,10}`. The nine
representations are never concatenated and the primary `D=6, tau=1` definition
does not change based on which setting performs best. Missing sweep runs remain
listed in `ordinal_sweep_status.csv`.

## Run

From the repository root:

```bash
bash exploration/run_exploration.sh --overwrite
```

A faster integration check is available:

```bash
bash exploration/run_exploration.sh \
  --output-dir /tmp/pd-exploration-quick \
  --quick \
  --skip-sweep \
  --overwrite
```

`--quick` is for software validation, not final reporting. Configuration and
all random seeds are in [`config.json`](config.json).

## Outputs

```text
exploration/processed/
├── manifest.json
├── exploration.log
├── features/
│   ├── subject_modeling_table.csv
│   ├── feature_provenance.csv
│   ├── feature_group_summary.csv
│   └── feature_spearman_correlations.csv
├── cross_validation/
│   ├── outer_fold_assignments.csv
│   ├── outer_fold_metrics.csv
│   ├── outer_fold_coefficients.csv
│   └── ordinal_sweep_coefficients.csv
├── predictions/
│   ├── repeated_outer_predictions.csv
│   ├── subject_out_of_fold_predictions.csv
│   └── ordinal_sweep_predictions.csv
├── metrics/
│   ├── model_performance.csv
│   ├── auc_differences_vs_demographics.csv
│   ├── coefficient_stability.csv
│   ├── final_model_coefficients.csv
│   ├── permutation_tests.csv
│   ├── ordinal_sweep_status.csv
│   └── ordinal_sweep_performance.csv
├── models/*.joblib
└── figures/
    ├── features/
    │   ├── candidate_feature_distributions.png
    │   ├── ordinal_entropy_complexity_plane.png
    │   └── feature_correlation_heatmap.png
    ├── validation/
    │   ├── model_performance_comparison.png
    │   ├── roc_and_precision_recall_curves.png
    │   ├── calibration_and_prediction_distributions.png
    │   └── confusion_matrices.png
    ├── explainability/*_coefficient_stability.png
    └── sensitivity/ordinal_parameter_sweep_roc_auc.png
```

Sweep-specific files and figures are created when at least one completed sweep
result is available. Every CSV retains full numeric precision. The manifest
records configuration, versions, exclusions, model roles, validation policy,
sweep completeness, and the external-validation limitation.

## Interpretation limits

- Results measure internal discrimination in this case-control cohort.
- Repeated cross-validation reduces but cannot eliminate optimism from working
  with one dataset.
- The clinical extension must not be interpreted as EEG-only performance.
- External cohort validation is required before any diagnostic claim.
- Coefficients describe associations conditional on the included predictors;
  they do not establish causal mechanisms.
