# Quantitative behavioral associations with MOCA

This pipeline relates Montreal Cognitive Assessment (**MOCA**) scores to
subject-level EEG quantities from the ordinal, scale-free bout, and within-bout
ordinal workflows.

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

The defaults produce 53 transparent features:

| Family | Features | Count |
|---|---|---:|
| Broadband ordinal | H, C, F from `D=6`, `tau=1` | 3 |
| Band ordinal | H, C, F in delta, theta, alpha, beta, low gamma, and broad 5–15 Hz | 18 |
| Bout properties | Occupancy, bouts/minute, duration, cycles/bout, and threshold ratio in four eBOSC bands | 20 |
| Within-bout ordinal | H, C, F pooled within detected theta, alpha, low-beta, and high-beta bouts | 12 |

Only regular permutation entropy, statistical complexity, and Fisher
information are selected. Rényi quantities are deliberately excluded.

The primary ordinal source is the completed 60-electrode `D6_tau1` parameter
sweep. Bout properties come from the 1–50 Hz scale-free analysis. Within-bout
ordinal features come from `bout_analyses/`. All upstream manifests are checked
for the expected 60 shared electrodes and analysis parameters before any
correlation is calculated.

## Embedding-dimension sensitivity

The broadband and band-resolved regular ordinal quantities are additionally
analyzed at **D=3, 4, 5, and 6**, always with **tau=1**. This is a robustness
analysis; `D=6, tau=1` remains the prespecified primary ordinal result. Bout
properties do not depend on D, and within-bout ordinal quantities remain at
the primary D=6 setting because changing D there would define a separate bout
pipeline sensitivity analysis.

There are 84 dimension-sensitivity features: four dimensions × seven signal
scopes (broadband plus six bands) × H/C/F. To avoid selecting a favorable D,
BH-FDR is controlled across all 84 features within each correlation method.
Effect stability across D is visualized, but consistency of direction is not
treated as a substitute for corrected statistical significance.

## Run

The upstream ordinal, scale-free, and within-bout pipelines must be complete.
First generate any missing D=3,4,5,6 ordinal inputs. Existing complete D=4 and
D=6 tables are reused; missing runs save all metric tables while intentionally
skipping the hundreds of upstream ordinal diagnostic figures:

```bash
bash quantitative_behavioral/prepare_dimension_sensitivity.sh
```

Then run:

```bash
bash quantitative_behavioral/run_quantitative_behavioral.sh --overwrite
```

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
├── manifest.json
├── quantitative_behavioral.log
├── metrics/
│   ├── moca_cohort.csv
│   ├── feature_dictionary.csv
│   ├── subject_features_long.csv
│   ├── analysis_dataset.csv
│   ├── subject_level_correlations.csv
│   ├── significant_primary_correlations.csv
│   ├── electrode_correlations.csv
│   ├── pd_feature_spearman_matrix.csv
│   ├── dimension_sensitivity_feature_dictionary.csv
│   ├── dimension_sensitivity_subject_features_long.csv
│   ├── dimension_sensitivity_analysis_dataset.csv
│   ├── dimension_sensitivity_correlations.csv
│   ├── dimension_sensitivity_significant_correlations.csv
│   └── dimension_sensitivity_electrode_correlations.csv
└── figures/
    ├── audit/cohort_and_coverage.png
    ├── correlations/
    │   ├── <family>_forest.png
    │   └── <family>_adjusted_sensitivity_heatmap.png
    ├── scatter/<family>_moca_scatter_grid.png
    ├── topomaps/<domain>_<band>_moca_topomaps.png
    └── dimension_sensitivity/
        ├── adjusted_correlation_heatmaps.png
        ├── adjusted_effect_stability.png
        ├── D<dimension>_forest.png
        ├── D<dimension>_moca_scatter_grid.png
        └── topomaps/ordinal_dimension_sensitivity_<scope>_moca_topomaps.png
```

## Reading the results

`subject_level_correlations.csv` is the primary result table. For each feature
and method it contains the subject count, correlation estimate, raw p-value,
bootstrap interval, family-specific BH-FDR p-value, and rejection indicator.
The generated `REPORT.md` lists the strongest adjusted relationships by
absolute effect size but does not silently filter the machine-readable table.

Scatter grids retain the raw subject observations and annotate the adjusted
correlation. Forest plots show adjusted estimates and bootstrap intervals;
green points indicate family-specific FDR rejection. Adjusted and unadjusted
heatmaps make covariate sensitivity visible.

For the D=3,4,5,6 robustness analysis, use
`dimension_sensitivity_correlations.csv`. A correlation is considered
statistically significant only when the primary adjusted row has both
`fdr_reject == True` and `p_fdr_bh < 0.05`. The raw `p_value` is provided for
transparency, but `p_value < 0.05` by itself is not the decision rule after
testing 84 parameter/signal/metric combinations. Bootstrap intervals quantify
effect uncertainty; a stable direction and magnitude across D strengthens the
robustness interpretation.

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
