# Whole-head Parkinson severity analysis

This pipeline relates EEG quantities averaged over **all electrodes shared by
every subject in the analysis cohort** to clinical severity within Parkinson
disease. The canonical full and matched cohorts currently each have 60 shared
electrodes. The actual list is resolved from the ordinal `electrode_sets.json`
provenance file and recorded in every manifest; it is not hard-coded. UPDRS is
primary and MOCA is a complementary cognitive axis.

The analysis is cross-sectional. “Progression axis” means ordering participants
by current severity and does not imply measured change over time.

## Run

```bash
bash disease_progression/run_disease_progression.sh --overwrite
```

The complete post-cleaning runner executes both full and matched versions:

```bash
bash run_all_analyses.sh --overwrite --no-progress
```

## Outputs

```text
disease_progression/processed/
├── REPORT.md
├── manifest.json
├── disease_progression.log
├── metrics/
│   ├── pd_cohort.csv
│   ├── electrode_selection.csv
│   ├── feature_dictionary.csv
│   ├── subject_features_long.csv
│   ├── subject_feature_matrix.csv
│   ├── progression_correlations.csv
│   └── clinical_axis_association.csv
└── figures/
    ├── electrode_selection.png
    ├── clinical_axes.png
    ├── scatter/{updrs,moca}/*_scatter_page_*.png
    └── forest/{updrs,moca}/*_forest_page_*.png
```

Matched sensitivity results are written to `processed_matched/`. The separate
eight-electrode group-comparison battery lives in
[`eight_electrode_analysis/`](../eight_electrode_analysis/README.md).
Statistical details and FDR families are defined in [`METHODS.md`](METHODS.md).
