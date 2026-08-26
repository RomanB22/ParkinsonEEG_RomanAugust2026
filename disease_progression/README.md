# Selected-electrode Parkinson severity analysis

This pipeline relates EEG quantities from exactly **F4, P4, O2, P6, CP2, CP1,
PO7, and P8** to clinical severity within Parkinson disease. UPDRS is primary;
MOCA is a complementary cognitive axis. It generates every raw scatter page,
family forest plots, an electrode-selection diagram, complete feature and
correlation tables, and a readable report.

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

Matched sensitivity results are written to `processed_matched/`. Statistical
details and FDR families are defined in [`METHODS.md`](METHODS.md).
