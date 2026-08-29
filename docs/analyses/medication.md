# ds002778 medication-state and MMSE analysis

This workflow analyzes OpenNeuro `ds002778-1.0.5` without changing the
original 149-participant pipeline. The analysis unit is a recording, while
inference retains the biological participant ID so that PD ON/OFF recordings
remain paired.

## Cohort and interpretation

- 16 healthy controls (`HC`), one recording each.
- 15 Parkinson disease participants with complete `PD_OFF` and `PD_ON`
  recordings.
- 31 people and 46 recordings total.
- MMSE is participant-level, ranges from 26 to 30, and does not vary between
  medication sessions.

MMSE is therefore analyzed continuously as limited cognitive-score variation.
It is not used to claim longitudinal deterioration or to create impaired versus
normal subgroups. No diagnostic classification or machine-learning analysis is
included.

## Commands

Every command runs inside the `MNE_August2026` conda environment.

```bash
# Validate the raw cohort and write the metadata audit.
bash src/analyses/medication/run_ds002778_analysis.sh metadata

# Generate ICA review material for all 46 recordings.
bash src/analyses/medication/run_ds002778_analysis.sh review --workers 4

# After confirming each recording's entries in preprocessing_ds002778.yaml:
bash src/analyses/medication/run_ds002778_analysis.sh preprocess --workers 4

# Compute features, planned contrasts, MMSE associations, and figures.
bash src/analyses/medication/run_ds002778_analysis.sh analyze
```

For an explicitly unattended preprocessing run, ICLabel proposals can be
accepted and recorded as automatic rather than visually confirmed:

```bash
bash src/analyses/medication/run_ds002778_analysis.sh preprocess \
  --skip-manual-ica-review --workers 4
```

Automatic ICA removal is auditable but is not equivalent to visual review.
The analysis command resumes neither partial feature products nor conflicting
outputs; use `--overwrite` deliberately when replacing them. Expensive stages
can be omitted for a technical pilot with `--skip-ordinal` or `--skip-bouts`.

## Preprocessing

The dataset-specific configuration is
`config/preprocessing_ds002778.yaml`. It loads BioSemi BDF at 512 Hz, excludes
EXG1–EXG8 and Status from scalp EEG, retains the 32 named scalp electrodes,
filters 1–100 Hz, applies a 60 Hz notch, resamples to 250 Hz, applies common
average reference for ICA/ICLabel, interpolates detected bad scalp channels,
and creates non-overlapping four-second epochs.

The ds002778 absolute epoch peak-to-peak guard is 500 µV. This was calibrated
against a raw BioSemi pilot: the original dataset's 200 µV ceiling rejected
32/48 epochs despite only two short robust artifact annotations, whereas the
500 µV guard plus the unchanged robust and annotation criteria retained 44/48.
This change is confined to ds002778.

## EEG features

The primary configuration is `config/analyses/ds002778.json`.

- Welch absolute and relative power: delta, theta, alpha, beta, and gamma.
- D=6, tau=1 permutation entropy, statistical complexity, and Fisher
  information, broadband and within canonical bands.
- Electrode-level fixed/knee specparam fits over 4–50 Hz with BIC selection,
  aperiodic quantities, periodic peaks, and a separate QC-qualified exponent
  requiring the configured fit quality and subject electrode coverage.
- eBOSC oscillatory occupancy, rate, duration, cycles, and threshold ratio for
  theta through gamma.
- All retained epochs are primary. A second analysis uses the same number of
  evenly spaced accepted epochs from every recording as a duration sensitivity.

Subject PSD features use the median across the 32 shared electrodes. Ordinal,
aperiodic, peak, and bout subject features use the electrode mean. Both subject
and electrode long tables are retained.

## Inference

The planned contrasts are:

1. `PD_OFF − HC`: disease-associated difference off medication.
2. `PD_ON − HC`: residual difference on medication.
3. `PD_ON − PD_OFF`: within-participant medication difference.

The two HC comparisons use age- and sex-adjusted OLS with HC3 robust standard
errors. ON/OFF uses paired t inference, paired Cohen dz, a Wilcoxon sensitivity,
and a participant bootstrap interval. Primary MMSE inference is partial
Spearman correlation adjusted for age and sex within HC, PD OFF, and PD ON,
plus `(PD ON − PD OFF) versus MMSE`. Age/sex-adjusted HC3 OLS slopes and
unadjusted Spearman correlations are retained as interpretable sensitivities.

Benjamini–Hochberg correction is applied within feature family, contrast/MMSE
model, duration variant, and sensitivity cohort. Every analysis is repeated
after excluding `sub-pd6` and `sub-pd16`, whose participant notes identify
different preprocessing provenance for the ON recording.

## Outputs

Generated data live under `outputs/ds002778/`:

- `metadata/`: canonical participant/recording tables and cohort audit.
- `features/`: subject/electrode features, PSD curves, inputs, and channel inventory.
- `statistics/condition_contrasts.csv`: the three planned contrasts.
- `statistics/mmse_associations.csv`: continuous MMSE models.
- `statistics/electrode_*`: secondary spatial versions with FDR across the
  complete electrode-level family.
- `figures/`: focused condition and MMSE plots.
- `manifest.json`: parameters, software versions, analyzed sample, and scientific cautions.
