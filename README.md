# Parkinson resting-state EEG preprocessing

This repository contains a readable, conservative MNE pipeline for the 149
eyes-open resting EEG recordings described in [Prompt.md](Prompt.md). It cleans
continuous EEG, creates 4-second epochs, and provides independent downstream
PSD, ordinal, and oscillatory-bout analyses.

The downstream workflow includes a formal specparam fit-QC sensitivity. Failed
1/f fits remain in the provenance outputs but are excluded from parallel bout,
cycle, and within-bout ordinal summaries, which require at least 48/60 passing
electrodes per subject. Run it after both bout pipelines with
`bash scale_free_analysis/run_fit_qc_sensitivity.sh`.

The original files under `dataset/` are read-only inputs. Generated cleaning
outputs are placed under the root `processed/` tree; analysis outputs stay in
their owning analysis folder's `processed/` or `processed_matched/` tree.

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

The setup script creates `MNE_August2026` with Python 3.14 and installs the complete
pinned stack from [`requirements.txt`](requirements.txt). If the environment
already exists, it is retained and updated. The equivalent manual validation
commands are:

```bash
conda run -n MNE_August2026 python scripts/inspect_dataset.py
conda run -n MNE_August2026 python -m unittest discover -s tests -v
```

The code configures non-interactive MNE/Matplotlib caches automatically.

## Cleaning sequence

```text
raw EEG
→ metadata/channel inspection
→ zero-phase 1–100 Hz FIR filter + 60 Hz notch
→ anti-aliased resampling from 500 to 250 Hz
→ conservative bad-channel detection
→ large-transient BAD annotations
→ pre-ICA common-average reference
→ extended Infomax ICA on the same 1–100 Hz, 250 Hz signal
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

Generate ICA review material at the final 250 Hz rate:

```bash
conda run -n MNE_August2026 python scripts/preprocess_subject.py sub-001 \
  --review-only --overwrite
```

Review the ranked ICLabel plot and component diagnostics. Edit the prefilled
list and reasons if necessary, then set
`ica.manual_review_confirmed.sub-001` to `true` in
`config/preprocessing.yaml`. After confirmation, create the cleaned outputs:

```bash
conda run -n MNE_August2026 python scripts/preprocess_subject.py sub-001 \
  --overwrite
```

## Historical verified two-subject pilot

An earlier verified pilot removed IC000 as an ocular artifact for one PD
participant (`sub-001`) and one Control (`sub-101`). The manual ICA maps are
currently reset to empty, so review mode must repopulate and a reviewer must
confirm the decisions before this command can be used without the automatic
ICLabel override:

```bash
conda run -n MNE_August2026 python scripts/preprocess_test_set.py \
  --overwrite
```

Verified pilot result:

| Subject | Group | ICA components | Removed | Large-artifact intervals | Epochs retained | Usable EEG |
|---|---|---:|---|---:|---:|---:|
| sub-001 | PD | 35 | IC000 ocular | 0 | 69/70 (98.6%) | 276 s |
| sub-101 | Control | 32 | IC000 ocular | 7 | 57/67 (85.1%) | 228 s |

Those historical pilot outputs used the superseded 120 Hz, 1–50 Hz contract
and must not be reused. Current runs regenerate 250 Hz, 1–100 Hz outputs.

## Safely clean all recordings at 250 Hz

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
overwriting prior manual decisions. The ICA/ICLabel input now uses CAR and the
recommended 1–100 Hz bandwidth, although visual review remains the scientific
default.

The equivalent individual Python commands are shown below.

First create review material for all subjects:

```bash
conda run -n MNE_August2026 python scripts/run_preprocessing.py \
  --review-only --overwrite
```

Inspect each participant's ranked `08_ica_*`, `09_ica_*`, and `10_ica_*`
figures. Review mode has already filled `ica.manual_exclude_components` and
reasons with high-confidence ICLabel proposals. Edit them when the plots
disagree, use `[]` when nothing should be removed, and set the participant's
`ica.manual_review_confirmed` value to `true`. The final batch command is:

```bash
conda run -n MNE_August2026 python scripts/run_preprocessing.py \
  --overwrite
```

The final command deliberately refuses to start if any participant lacks a
visually confirmed ICA review. A machine proposal alone never removes a
component unless `--skip-manual-ica-review` is explicitly supplied.

`--no-ica-downsampling` remains as a backward-compatible option. Under the
current defaults it has no effect because both the cleaned signal and ICA copy
already run at 250 Hz.

## Ordinal analysis

The independent downstream workflow in [`ordinal_analysis/`](ordinal_analysis/README.md)
uses the accepted cleaned epochs to calculate `ordpy` permutation entropy,
statistical complexity, and Fisher information for every participant/electrode
and each participant's mean across electrodes. The full workflow is applied to
the broadband signal and to delta, theta, alpha, beta, and low gamma. It
generates PD/Control violins, H×C and H×F planes,
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
theta, alpha, beta, and 30–50 Hz low gamma.
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
It also provides an exact-sex, optimal-age matched 49-pair sensitivity analysis
that removes age and sex from the model predictors and keeps pairs intact
through validation, bootstrap, and permutation inference. Candidate quantities
are plotted against age in both cohorts as a descriptive visual audit.

```bash
bash exploration/run_exploration.sh --overwrite
bash exploration/run_exploration.sh --matched-demographics --overwrite
```

## Spectral parameterization and oscillatory bouts

The standalone workflow in
[`scale_free_analysis/`](scale_free_analysis/README.md) implements
[`ScaleFree.md`](ScaleFree.md). It combines `specparam` periodic/aperiodic
decomposition, aperiodic-relative eBOSC bout detection, bycycle waveform
features, subject-balanced PD/Control comparisons, saved intermediate results,
and example/group/topographic figures. A separate gallery aligns detected bouts
at their centers and shows normalized Hilbert envelopes, circular relative
phase with phase-consistency `R`, and phase-aligned average bout shapes for
Control versus PD, per band and electrode, alongside detection-coverage and
fit-QC sensitivity panels.

```bash
bash scale_free_analysis/run_scale_free_analysis.sh --overwrite
bash scale_free_analysis/generate_typical_bouts.sh
```

An orthogonal sensitivity workflow in
[`bycycle_burst_analysis/`](bycycle_burst_analysis/README.md) detects bursts
directly from cycle-to-cycle amplitude, period, and monotonicity consistency.
It does not use the eBOSC mask or a specparam power threshold. It saves the
independent events and cycle features, repeats subject- and electrode-level
PD-versus-Control inference, and plots event-mask and subject-metric agreement
with eBOSC.

```bash
bash bycycle_burst_analysis/run_bycycle_burst_analysis.sh --overwrite
```

This independent detector is intentionally opt-in and is not run by the
default full pipeline. Add `--include-bycycle-bursts` to the full runner when
this sensitivity analysis is required.

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

## PD-versus-Control inference across analyses

PSD, ordinal, scale-free/bout, independent-bycycle-burst, and within-bout ordinal pipelines share a
subject-level inferential layer. Full-cohort models adjust for age and sex;
matched-cohort models preserve `match_pair_id`. Exploratory electrode-wise
tests include both feature-wise spatial FDR and a stricter domain-wide FDR,
with effect and significance topomaps. Methods and output columns are documented in
[`GROUP_STATISTICS.md`](GROUP_STATISTICS.md).

## Quantitative behavioral MOCA analysis

The cross-sectional workflow in
[`quantitative_behavioral/`](quantitative_behavioral/README.md) relates MOCA to
regular broadband/band ordinal H/C/F, eBOSC bout properties, and within-bout
ordinal H/C/F among PD participants. It also tests the BIC-selected fixed/knee
1–50 Hz aperiodic exponent between Control and PD and its association with
MOCA, with formal electrode-fit QC and a 4–35 Hz sensitivity analysis. The
underlying PSD remains 1–50 Hz. It
uses subject-level partial Spearman
correlations adjusted for age and sex, deterministic bootstrap intervals,
prespecified family-specific FDR correction, raw-data scatter grids, forest
plots, sensitivity heatmaps, and secondary spatial maps. Separate D=3,4,5,6
(tau=1) ordinal blocks include regular H/C/F and Rényi entropy/complexity at
alpha=0.1, 0.5, 0.9, 1.1, 2, 5, and 10. Each D has its own feature matrix and within-D FDR
correction; D=6 is primary and D=3–5 are sensitivity blocks.
The exact age/sex-adjusted partial Spearman method is documented in
[`quantitative_behavioral/METHODS.md`](quantitative_behavioral/METHODS.md).

```bash
bash quantitative_behavioral/prepare_dimension_sensitivity.sh
bash quantitative_behavioral/run_quantitative_behavioral.sh --overwrite
```

## Whole-head disease-severity axes

[`disease_progression/`](disease_progression/README.md) recomputes 141 EEG
features over all electrodes shared by the analysis cohort within PD. UPDRS is
the primary motor-severity axis and MOCA is complementary. Age/sex-adjusted
partial Spearman estimates, unadjusted sensitivities, family-specific FDR,
raw scatter pages, forest plots, and full/matched reports are generated. This
is cross-sectional severity analysis, not measured longitudinal progression.

```bash
bash disease_progression/run_disease_progression.sh --overwrite
```

## Eight-electrode sensitivity battery

[`eight_electrode_analysis/`](eight_electrode_analysis/README.md) adds a
separate full/matched sensitivity analysis for PSD, ordinal, aperiodic,
periodic/bout, and within-bout ordinal quantities using exactly F4, P4, O2, P6,
CP2, CP1, PO7, and P8. It recomputes the subject aggregate and electrode-wise
PD-versus-Control inference from stored electrode-level estimates; the
whole-head primary analyses remain unchanged.

## Accepted-duration sensitivity

[`duration_qc_analysis/`](duration_qc_analysis/README.md) retains the primary
four-second epoch definition but recomputes group comparisons, MOCA
associations, and transparent prediction validation after requiring at least
60 seconds of accepted EEG. It uses complete retained pairs in the matched
cohort and generates separate reports and figures without changing primary
results.

## Pipeline entry point

Use [`run_reproducible_pipeline.sh`](run_reproducible_pipeline.sh) as the single
public launcher. It handles environment and metadata bootstrap, cleaning, and
the downstream dependency order. Three profiles keep routine use concise:

```bash
bash run_reproducible_pipeline.sh run --profile compute  # base caches
bash run_reproducible_pipeline.sh run --profile paper    # default report battery
bash run_reproducible_pipeline.sh run --profile full-qc  # paper + bycycle
```

The `paper` and `full-qc` profiles analyze both the full cohort and one
canonical 49-Control/49-PD exact-sex, optimal-age matched cohort. Full outputs
remain under `processed/`; matched outputs use `processed_matched/`.
`run_all_analyses.sh` remains the internal post-cleaning orchestrator and a
supported expert entry point, but is not required for normal use.

Useful controls include `--overwrite`, `--dry-run`, `--no-progress`,
`--log-file PATH`, `--skip-sweep`, `--skip-exploration`, and the opt-in
`--include-bycycle-bursts`. The D=6 primary ordinal calculation is reused by
the dimension analysis; only D={3,4,5} are computed as additional sensitivity
settings. Matched ordinal calculations reuse validated subject-level full-
cohort values and recompute matched summaries, paired tests, and figures.
Real top-level runs automatically mirror stdout and stderr to timestamped files
under `pipeline_logs/`; dry runs create no files.

## Complete cleaning-to-report reproduction

[`run_reproducible_pipeline.sh`](run_reproducible_pipeline.sh) is the single
top-level launcher. It preserves the required manual ICA checkpoint while
covering signal cleaning, every analysis, both exploration cohorts, figures,
fit-QC sensitivity, and repository tests:

```bash
bash run_reproducible_pipeline.sh review --overwrite
# Review ICA stages 08–10 and confirm the decisions.
bash run_reproducible_pipeline.sh run --profile paper --overwrite
```

For a resumable run, omit `--overwrite`. `--dry-run` prints the complete
cleaning and downstream commands. The explicitly non-default
`--skip-manual-ica-review` option applies automatic ICLabel decisions and is
recorded in preprocessing QC.

The compact repository and cache map is in [`PIPELINE_MAP.md`](PIPELINE_MAP.md),
and copy-paste commands are in [`COMMAND.md`](COMMAND.md).

## 60 Hz notch decision

The retained passband is 1–100 Hz, so the 60 Hz line frequency lies inside the
analysis signal. A 2 Hz-wide 60 Hz notch is enabled by default and its
application is recorded in every log and QC row.

## Main outputs

```text
processed/
├── cleaned_raw/       # final continuous 1–100 Hz, 250 Hz FIF
├── epochs/            # accepted 4-second epochs, 250 Hz, no baseline
├── ica/               # fitted ICA solutions
├── qc/<subject>/      # ordered plots 01–21 + decisions.json
├── metadata/          # inspection, subject decisions, preprocessing_qc.csv
└── logs/              # one readable log per participant
```
