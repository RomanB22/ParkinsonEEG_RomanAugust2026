# Quantitative behavioral associations with MOCA

This pipeline relates Montreal Cognitive Assessment (**MOCA**) scores to
subject-level EEG quantities from the ordinal, scale-free aperiodic/bout, and
within-bout ordinal workflows. It also compares the aperiodic exponent between
Control and PD participants.

The complete statistical specification, including the age/sex-adjusted partial
Spearman calculation, p-values, bootstrap intervals, and FDR scopes, is in
[`METHODS.md`](METHODS.md).

Although the folder addresses Parkinson disease progression scientifically,
the available dataset has one recording and one MOCA score per participant.
The implemented analysis is therefore **cross-sectional**: it can identify
associations with cognitive status among PD participants, but it cannot measure
within-person progression, temporal change, causality, or predictive validity.

## Primary cohort and outcome

- Primary cohort: the 100 PD participants.
- Outcome: MOCA, available for all 100 PD participants (observed range 9–30).
- Primary estimate: partial Spearman correlation after rank-transforming MOCA,
  the EEG feature, age, and sex, then residualizing both ranked MOCA and the
  ranked EEG feature for age and sex.
- Sensitivity estimate: unadjusted Spearman correlation.
- Uncertainty: deterministic 2,000-subject bootstrap percentile intervals.
- Multiplicity: Benjamini–Hochberg FDR controlled separately within each
  prespecified feature family and correlation method.

Every primary analysis contains one row per participant. Electrodes and bouts
are never treated as independent patients.

## Prespecified feature families

The defaults produce 52 transparent inferential features:

| Family | Features | Count |
|---|---|---:|
| Aperiodic | All-fit and QC-qualified fixed-mode specparam exponents over 4–35 Hz | 2 |
| Broadband ordinal | H, C, F from `D=6`, `tau=1` | 3 |
| Band ordinal | H, C, F in delta, theta, alpha, beta, and low gamma | 15 |
| Bout properties | Occupancy, bouts/minute, duration, cycles/bout, and threshold ratio in four canonical eBOSC bands | 20 |
| Within-bout ordinal | H, C, F pooled within the four canonical bout bands | 12 |

The 52-feature primary table retains regular permutation entropy,
statistical complexity, and Fisher information. Rényi quantities are added in
the separate embedding-dimension analysis blocks described below. Within-bout
ordinal quantities remain regular H/C/F only.

The primary ordinal source is the completed 60-electrode `D6_tau1` parameter
sweep. The aperiodic exponent and bout properties come from the scale-free
analysis, which retains a 1–50 Hz PSD but fits the exponent over 4–35 Hz. The
exponent is estimated in fixed mode at each shared
electrode and averaged across those 60 electrodes within each subject. A
separate QC sensitivity retains only subjects with at least 80% QC-passing
electrodes and averages their passing electrodes. Within-bout
ordinal features come from `bout_analyses/`. All upstream manifests are checked
for the expected 60 shared electrodes and analysis parameters before any
correlation is calculated.

A separate fit-QC sensitivity repeats the 20 bout-property and 12 within-bout
ordinal MOCA associations using only passing electrodes in subjects with at
least 48/60 passing fits. It currently contains 70 PD participants. This
32-feature sensitivity is kept out of the primary 52-feature table and receives
its own family-specific BH corrections.

## Separate embedding-dimension analyses

The broadband and band-resolved ordinal quantities are analyzed at **D=3, 4,
5, and 6**, always with **tau=1**. Each D contains regular H/C/F plus Rényi
entropy and complexity at **alpha=0.1, 0.5, 0.9, 1.1, 2, 5, and 10**. The Rényi values come from
`ordpy.renyi_complexity_entropy`; no separate `renyi_entropy` calculation is
used.

Each D is a separate 102-feature analysis block: six signal scopes
(broadband plus five non-overlapping bands) × seventeen quantities. The pipeline writes a distinct
one-row-per-subject matrix for each D and never concatenates the four D blocks
into one model feature vector. BH-FDR is controlled separately within each D
across its 102 features and within each correlation method.

The D blocks are separate analyses, but they are **not statistically
independent**: all four reuse the same participants and EEG recordings and
their ordinal quantities are correlated. `D=6, tau=1` remains the primary
ordinal block; D=3, D=4, and D=5 are robustness/sensitivity blocks. Choosing
the most favorable D after inspecting results must therefore be described as
exploratory. Bout properties do not depend on D, and within-bout ordinal
quantities remain at D=6 because changing them defines a separate bout-pipeline
sensitivity analysis.

## Run

The upstream ordinal, scale-free, and within-bout pipelines must be complete.
First generate any missing D=3,4,5 ordinal sensitivity inputs. The primary D=6
tables are always reused from `ordinal_analysis/processed`; existing complete
sensitivity tables are also reused. Missing runs save metric tables while
intentionally skipping the hundreds of upstream ordinal diagnostic figures:

```bash
bash quantitative_behavioral/prepare_dimension_sensitivity.sh
```

Then run:

```bash
bash quantitative_behavioral/run_quantitative_behavioral.sh --overwrite
```

The repository fit-QC runner regenerates the QC-qualified bout summaries and
their MOCA sensitivity together:

```bash
bash scale_free_analysis/run_fit_qc_sensitivity.sh
```

To execute every post-cleaning analysis in dependency order—including PSD,
ordinal analyses, scale-free/bout analyses, exploration models, and this MOCA
workflow—use the resumable repository-level runner:

```bash
bash run_all_analyses.sh
```

Use `bash run_all_analyses.sh --help` for overwrite, dry-run, progress, and
stage-skip options.

For a faster code/figure pilot while retaining the same participants and
features:

```bash
bash quantitative_behavioral/run_quantitative_behavioral.sh \
  --bootstrap-resamples 100 \
  --output-dir /tmp/quantitative-behavioral-pilot \
  --overwrite
```

The full 2,000-resample run is the reporting default. The pipeline refuses to
replace an existing primary result table unless `--overwrite` is supplied.

## Outputs

```text
quantitative_behavioral/processed/
├── REPORT.md
├── FIT_QC_SENSITIVITY.md
├── manifest.json
├── quantitative_behavioral.log
├── metrics/
│   ├── moca_cohort.csv
│   ├── feature_dictionary.csv
│   ├── subject_features_long.csv
│   ├── analysis_dataset.csv
│   ├── subject_level_correlations.csv
│   ├── aperiodic_exponent_group_comparison.csv
│   ├── aperiodic_exponent_subject_data.csv
│   ├── significant_primary_correlations.csv
│   ├── electrode_correlations.csv
│   ├── pd_feature_spearman_matrix.csv
│   ├── fit_qc_sensitivity/
│   │   ├── feature_dictionary.csv
│   │   ├── subject_features_long.csv
│   │   └── subject_level_correlations.csv
│   ├── dimension_sensitivity_feature_dictionary.csv
│   ├── dimension_sensitivity_subject_features_long.csv
│   ├── dimension_sensitivity_correlations.csv
│   ├── dimension_sensitivity_significant_correlations.csv
│   ├── dimension_sensitivity_electrode_correlations.csv
│   └── dimensions/D<dimension>/
│       ├── feature_dictionary.csv
│       ├── analysis_dataset.csv
│       ├── correlations.csv
│       ├── significant_correlations.csv
│       └── electrode_correlations.csv
└── figures/
    ├── audit/cohort_and_coverage.png
    ├── aperiodic/group_comparison.png
    ├── correlations/
    │   ├── <family>_forest.png
    │   └── <family>_adjusted_sensitivity_heatmap.png
    ├── scatter/<family>_moca_scatter_grid.png
    ├── fit_qc_sensitivity/<fit_qc_family>_*.png
    ├── topomaps/<domain>_<band>_moca_topomaps.png
    └── dimension_sensitivity/
        ├── adjusted_correlation_heatmaps.png
        ├── adjusted_effect_stability.png
        ├── D<dimension>_forest.png
        ├── D<dimension>_adjusted_sensitivity_heatmap.png
        ├── scatter/D<dimension>_<quantity_set>_moca_scatter_grid.png
        └── topomaps/ordinal_D<dimension>_<scope>_moca_topomaps.png
```

## Reading the results

`subject_level_correlations.csv` is the primary result table. For each feature
and method it contains the subject count, correlation estimate, raw p-value,
bootstrap interval, family-specific BH-FDR p-value, and rejection indicator.
The generated `REPORT.md` lists the strongest adjusted relationships by
absolute effect size but does not silently filter the machine-readable table.
The separate `FIT_QC_SENSITIVITY.md` and `metrics/fit_qc_sensitivity/` files
report the QC-qualified bout analyses; they do not replace the primary table.

Scatter grids retain the raw subject observations and annotate the adjusted
correlation. Forest plots show adjusted estimates and bootstrap intervals;
green points indicate family-specific FDR rejection. Adjusted and unadjusted
heatmaps make covariate sensitivity visible.

The aperiodic group comparison uses one mean exponent per subject. Its primary
effect is the PD-minus-Control coefficient from `exponent ~ PD + age + sex`,
with an HC3 robust standard error and confidence interval. Welch's t test,
Mann–Whitney U, and Hedges' g are included as unadjusted sensitivity and effect
size summaries. A parallel `aperiodic_exponent_qc` row reports the formal
fit-QC sensitivity, with BH correction across the two aperiodic analyses. The
PD-only exponent–MOCA relationships are the two aperiodic rows in
`subject_level_correlations.csv`; their raw scatter,
partial Spearman estimate, and electrode-level localization are generated with
the other feature-family figures.

For the D=3,4,5,6 robustness analysis, use
`dimension_sensitivity_correlations.csv`. A correlation is considered
statistically significant only when the primary adjusted row has both
`fdr_reject == True` and `p_fdr_bh < 0.05`. The raw `p_value` is provided for
transparency, but `p_value < 0.05` by itself is not the decision rule after
testing 102 features within that D. The `family` column identifies the relevant
FDR block (`ordinal_D3` through `ordinal_D6`). Bootstrap intervals quantify
effect uncertainty; a stable direction and magnitude across D strengthens the
robustness interpretation without making the D blocks statistically
independent.

For convenience, the two `significant_*.csv` files contain only adjusted,
FDR-rejected primary rows. They retain all effect estimates, intervals, raw
p-values, and corrected p-values; an empty file means that no test survived
the stated correction.

Electrode-level correlations are secondary spatial localization analyses.
They are calculated separately at each of the same 60 electrodes, still using
one observation per PD participant. Their p-values are BH-corrected across the
60 electrodes within each feature. Topomaps show effect sizes, not subject
counts or uncorrected significance.

## Interpretation limits

- MOCA is bounded and cross-sectional; correlation does not imply cognitive
  decline over time.
- Age and sex are the only prespecified covariates. Medication state,
  education, disease duration, and other potential confounders are unavailable.
- Family-wise FDR choices are explicit but do not make exploratory spatial maps
  confirmatory.
- Missing EEG quantities use feature-wise complete cases. No MOCA, age, sex, or
  EEG feature is imputed; the subject count is shown for every test.
- The analysis reports associations, not a fitted clinical prediction model.
