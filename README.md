# Parkinson resting-state EEG preprocessing

This repository contains a readable, conservative MNE pipeline for the 149
eyes-open resting EEG recordings described in [Prompt.md](Prompt.md). It cleans
continuous EEG, creates 4-second epochs, and provides independent downstream
PSD, ordinal, and oscillatory-bout analyses.

The original files under `dataset/` are read-only inputs. Complex-pipeline
outputs are placed under `processed/`; minimal-pipeline outputs are isolated
under `simpler/processed/`.

## Verified dataset

- 149 participants: 100 Parkinson disease (`PD`) and 49 `Control`
- EEGLAB `.set` + external `.fdt`
- 500 Hz acquisition
- Pz online reference reported by the sidecars; Pz is not stored as a channel
- 60 EEG channels common to every recording
- Two EEG layouts after auxiliary channels are excluded
- Medication status is not supplied

The complete inspection is in
[`processed/metadata/dataset_inspection_report.md`](processed/metadata/dataset_inspection_report.md).

## Run with the requested environment

From the project root:

```bash
bash scripts/create_conda_environment.sh --run-tests
```

The setup script creates `MNE_Roman` with Python 3.14 and installs the complete
pinned stack from [`requirements.txt`](requirements.txt). If the environment
already exists, it is retained and updated. The equivalent manual validation
commands are:

```bash
conda run -n MNE_Roman python scripts/inspect_dataset.py
conda run -n MNE_Roman python -m unittest discover -s tests -v
```

The code configures non-interactive MNE/Matplotlib caches automatically.

## Cleaning sequence

```text
raw EEG
→ metadata/channel inspection
→ zero-phase 1–50 Hz FIR filter
→ anti-aliased resampling from 500 to 120 Hz
→ conservative bad-channel detection
→ large-transient BAD annotations
→ extended Infomax ICA
→ ICLabel probabilities and artifact-to-brain ranking
→ automatically prefilled candidate exclusions
→ visually confirmed ICA removal
→ interpolation of recorded bad channels only
→ average reference
→ 4-second epochs, no baseline
→ annotation + residual peak-to-peak rejection
→ cleaned FIF files, QC plots, logs, and decision tables
```

Every definition, threshold, rationale, and QC file is described in
[`PREPROCESSING_PIPELINE.md`](PREPROCESSING_PIPELINE.md).

Exact run order, required edits, command-line options, and the meaning of every
configuration value are documented in
[`PIPELINE_USAGE_AND_PARAMETERS.md`](PIPELINE_USAGE_AND_PARAMETERS.md).

## Run one subject

Generate ICA review material while keeping the temporary ICA copy at the final
120 Hz rate:

```bash
conda run -n MNE_Roman python scripts/preprocess_subject.py sub-001 \
  --review-only --no-ica-downsampling --overwrite
```

Review the ranked ICLabel plot and component diagnostics. Edit the prefilled
list and reasons if necessary, then set
`ica.manual_review_confirmed.sub-001` to `true` in
`config/preprocessing.yaml`. After confirmation, create the cleaned outputs:

```bash
conda run -n MNE_Roman python scripts/preprocess_subject.py sub-001 \
  --no-ica-downsampling --overwrite
```

## Reproduce the verified two-subject pilot

The configuration contains reviewed IC000 ocular-artifact decisions for one PD
participant (`sub-001`) and one Control (`sub-101`):

```bash
conda run -n MNE_Roman python scripts/preprocess_test_set.py \
  --no-ica-downsampling --overwrite
```

Verified pilot result:

| Subject | Group | ICA components | Removed | Large-artifact intervals | Epochs retained | Usable EEG |
|---|---|---:|---|---:|---:|---:|
| sub-001 | PD | 35 | IC000 ocular | 0 | 69/70 (98.6%) | 276 s |
| sub-101 | Control | 32 | IC000 ocular | 7 | 57/67 (85.1%) | 228 s |

Both cleaned continuous files are 120 Hz and 1–50 Hz.

## Safely clean all recordings at 120 Hz

The simplest entry point is the two-stage bash runner:

```bash
bash scripts/run_full_cleaning.sh review --overwrite
# Inspect ranked stages 08–10; edit proposals and confirm each subject.
bash scripts/run_full_cleaning.sh clean --overwrite
```

The script performs dataset inspection and validation tests before either
stage. Review mode prefills ICLabel proposals with
`manual_review_confirmed: false`. Clean mode refuses to use them until a person
visually checks the ranked topography, time course, and PSD and changes that
subject's confirmation flag to `true`.

For an explicitly unattended run, the manual confirmation gate can be bypassed:

```bash
bash scripts/run_full_cleaning.sh clean --skip-manual-ica-review --overwrite
```

This option applies the high-confidence ICLabel proposal directly. It records
`ica_selection_mode: automatic_iclabel` and `automatic_ica_removal: true` in QC,
and writes the actual lists under `ica.automatic_exclude_components` without
overwriting prior manual decisions. Because the present ICA input differs from
ICLabel's training reference and bandwidth, the reviewed workflow above remains
the scientific default.

The equivalent individual Python commands are shown below.

First create review material for all subjects:

```bash
conda run -n MNE_Roman python scripts/run_preprocessing.py \
  --review-only --no-ica-downsampling --overwrite
```

Inspect each participant's ranked `08_ica_*`, `09_ica_*`, and `10_ica_*`
figures. Review mode has already filled `ica.manual_exclude_components` and
reasons with high-confidence ICLabel proposals. Edit them when the plots
disagree, use `[]` when nothing should be removed, and set the participant's
`ica.manual_review_confirmed` value to `true`. The final batch command is:

```bash
conda run -n MNE_Roman python scripts/run_preprocessing.py \
  --no-ica-downsampling --overwrite
```

The final command deliberately refuses to start if any participant lacks a
visually confirmed ICA review. A machine proposal alone never removes a
component unless `--skip-manual-ica-review` is explicitly supplied.

`--no-ica-downsampling` does not disable the required 500→120 Hz resampling.
It keeps the temporary ICA copy at 120 Hz instead of reducing that copy once
more to 100 Hz. The older `--no-downsampling` spelling remains an alias.

## Minimal alternative

The independent minimal workflow performs linear detrending, a 59–61 Hz
fourth-order Butterworth band-stop, a 1–50 Hz fourth-order Butterworth
band-pass, and resampling to 120 Hz before extracting 4-second windows. Run its
pilot with:

```bash
bash simpler/run_simple_cleaning.sh pilot --overwrite
```

Its parameters, outputs, and the pilot's strict 100 µV rejection result are
documented in [`simpler/README.md`](simpler/README.md). It writes only under
`simpler/processed/`, so it cannot overwrite the complex pipeline outputs.

## Ordinal analysis

The independent downstream workflow in [`ordinal_analysis/`](ordinal_analysis/README.md)
uses the accepted cleaned epochs to calculate `ordpy` permutation entropy,
statistical complexity, and Fisher information for every participant/electrode
and each participant's mean across electrodes. The full workflow is applied to
the broadband signal and to delta, theta, alpha, beta, low gamma, and the
overlapping 5–15 Hz range. It generates PD/Control violins, H×C and H×F planes,
individual scalp maps, group-mean scalp maps, complete CSV tables, logs, and a
machine-readable provenance manifest.

```bash
bash ordinal_analysis/run_ordinal_analysis.sh --overwrite
```

## PSD group analysis

The separate workflow in [`psd_analysis/`](psd_analysis/README.md) estimates
subject-balanced median Welch spectra, pointwise 95% bootstrap confidence bands
for PD and Control, and group median relative-power topographies normalized to
each subject/electrode's total 1–50 Hz power for delta,
theta, alpha, beta, 30–50 Hz low gamma, and an additional broad 5–15 Hz band.
It also plots subject-level PD-versus-Control relative-power violins for every
band after taking each subject's median across shared electrodes.

```bash
bash psd_analysis/run_psd_analysis.sh --overwrite
```

## Transparent PD versus Control modeling

The documented workflow in [`exploration/`](exploration/README.md) combines
subject-level ordinal H/C/F, conservative PSD log-ratio features, age, sex, and
an explicitly separate MOCA extension using interpretable ridge logistic
regression. Repeated nested cross-validation, out-of-fold predictions,
permutation tests, coefficient stability, model comparisons, and all feature
and validation figures are saved.

```bash
bash exploration/run_exploration.sh --overwrite
```

## Spectral parameterization and oscillatory bouts

The standalone workflow in
[`scale_free_analysis/`](scale_free_analysis/README.md) implements
[`ScaleFree.md`](ScaleFree.md). It combines `specparam` periodic/aperiodic
decomposition, aperiodic-relative eBOSC bout detection, bycycle waveform
features, subject-balanced PD/Control comparisons, saved intermediate results,
and example/group/topographic figures.

```bash
bash scale_free_analysis/run_scale_free_analysis.sh --overwrite
```

## Ordinal analysis inside detected bouts

The independent workflow in [`bout_analyses/`](bout_analyses/README.md) detects
aperiodic-relative eBOSC bouts and converts each time-limited bout to a
boundary-safe ordinal representation. It calculates only regular permutation
entropy, statistical complexity, and Fisher information—no Rényi metrics—and
saves per-bout tables, pooled subject/electrode/band count tensors, diagnostics,
group figures, electrode figures, and individual topomaps.

```bash
bash bout_analyses/run_bout_analyses.sh --overwrite
```

## Quantitative behavioral MOCA analysis

The cross-sectional workflow in
[`quantitative_behavioral/`](quantitative_behavioral/README.md) relates MOCA to
regular broadband/band ordinal H/C/F, eBOSC bout properties, and within-bout
ordinal H/C/F among PD participants. It uses subject-level partial Spearman
correlations adjusted for age and sex, deterministic bootstrap intervals,
prespecified family-specific FDR correction, raw-data scatter grids, forest
plots, sensitivity heatmaps, and secondary spatial maps. Separate D=3,4,5,6
(tau=1) ordinal blocks include regular H/C/F and Rényi entropy/complexity at
alpha=0.5, 0.9, 1.1, 2, and 5. Each D has its own feature matrix and within-D FDR
correction; D=6 is primary and D=3–5 are sensitivity blocks.
The exact age/sex-adjusted partial Spearman method is documented in
[`quantitative_behavioral/METHODS.md`](quantitative_behavioral/METHODS.md).

```bash
bash quantitative_behavioral/prepare_dimension_sensitivity.sh
bash quantitative_behavioral/run_quantitative_behavioral.sh --overwrite
```

## Combined post-cleaning analysis runner

[`run_all_analyses.sh`](run_all_analyses.sh) combines all post-cleaning
analyses in dependency order: PSD, primary ordinal quantities, the D/tau
sweep, scale-free bout properties, within-bout ordinal quantities,
PD-versus-Control exploration models, D-specific MOCA inputs, and the final
quantitative-behavioral analysis. It resumes from valid completed outputs by
default and detects ordinal tables missing the configured Rényi alpha columns.

```bash
bash run_all_analyses.sh
```

Useful controls include `--overwrite`, `--dry-run`, `--no-progress`,
`--skip-sweep`, and `--skip-exploration`. Cleaning is deliberately not invoked:
manual ICA confirmation remains an explicit prerequisite through
`scripts/run_full_cleaning.sh`.

## 60 Hz notch decision

The final required passband is 1–50 Hz. Therefore the configured 60 Hz line
frequency is already outside the retained band, and a second 60 Hz notch would
be redundant. `notch_enabled` is false and the decision is recorded in every
log and QC row. This follows `Prompt.md`'s instruction to avoid unnecessary
filters while still documenting the 60 Hz acquisition environment.

## Main outputs

```text
processed/
├── cleaned_raw/       # final continuous 1–50 Hz, 120 Hz FIF
├── epochs/            # accepted 4-second epochs, 120 Hz, no baseline
├── ica/               # fitted ICA solutions
├── qc/<subject>/      # ordered plots 01–21 + decisions.json
├── metadata/          # inspection, subject decisions, preprocessing_qc.csv
└── logs/              # one readable log per participant
```
