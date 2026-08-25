# EEG preprocessing pipeline: definitions, decisions, and QC

This document follows the actual code from raw input to accepted resting-state
epochs. The goal is to preserve spectral slope and oscillatory peaks while
removing only clear contamination. No epoch normalization, baseline correction,
spectral flattening, or `specparam` analysis is performed. The filtered signal
is resampled from the 500 Hz acquisition rate to a final 120 Hz rate.

## 1. Dataset inspection

**Definition.** Dataset inspection checks what was recorded before changing any
signal: file pairs, participant labels, sampling rate, duration, channel names,
reference, line frequency, channel units/types, and coordinate metadata.

**Implementation.** `scripts/inspect_dataset.py` reads all 149 participant
sidecars and channel tables. It also loads one PD and one Control signal for
basic raw trace and PSD checks.

**Observed facts.** There are 100 PD and 49 Control recordings at 500 Hz. The
source sidecars report a Pz online reference, but Pz is not stored as a signal;
CPz is present. Channel type and units are marked `n/a`. MNE's EEGLAB reader
converts the signal to volts. The pipeline saves the unresolved source labels.

**Outputs.** `subjects.csv`, `recordings.csv`, `channel_metadata.csv`,
`channel_availability.csv`, `channel_signatures.csv`, `common_channels.json`,
and `dataset_inspection_report.md` are under `processed/metadata/`.

## 2. Load one subject and preserve provenance

`src.dataset.load_subject()` reads the `.set`; EEGLAB resolves its companion
`.fdt`. It records the absolute input path, original/reference sidecar, original
channel names, auxiliary channels, line frequency, and sampling rate.

Resp/X/Y/Z are classified as auxiliary when present and excluded from the EEG
analysis copy. Their names remain in provenance. No missing EEG channel is
created. Valid EEGLAB electrode positions are preserved. If a recording lacks
positions, standard 10–05 coordinates are assigned only to its recorded
channels so topographies and interpolation remain possible; the coordinate
source is recorded per participant.

## 3. Raw signal and PSD

The raw data are copied before any operation. QC files `01_raw_signal.png` and
`02_raw_psd.png` show representative channels and the unprocessed spectrum.
Preferred channels are Fp1, Fz, Cz, CPz/Pz, and O1; existing alternatives are
selected automatically when needed.

## 4. Filter to the final 1–50 Hz band

**Definition.** A band-pass retains frequencies between its lower and upper
cutoffs. Here, a zero-phase FIR filter retains 1–50 Hz. Zero phase means that
the offline filter does not systematically shift peaks in time.

`src.preprocessing.filter_eeg()` operates on a copy at the 500 Hz acquisition
rate. `src.preprocessing.resample_eeg()` then applies MNE's anti-aliased FFT
resampling to 120 Hz. Because the retained band ends at 50 Hz, the final 60 Hz
Nyquist frequency leaves a 10 Hz guard band. QC files
`03_filtered_signal.png`, `04_filtered_psd.png`, and
`05_raw_vs_filtered.png` use identical channels and trace intervals while
allowing for the different raw and processed time grids.

The 60 Hz notch is disabled because 60 Hz is above the final 50 Hz low-pass.
Applying both would add an unnecessary filter without retaining any additional
analysis bandwidth. This decision is logged per participant.

## 5. Detect bad recorded channels

**Definition.** A bad channel is an electrode that was recorded but is not a
usable measurement. A missing channel was never recorded. These are never
treated as the same thing.

`src.channels.detect_bad_channels()` measures:

- standard deviation and flatness;
- full-recording peak-to-peak amplitude;
- median correlation with other EEG channels;
- 30–50 Hz power relative to 1–30 Hz power.

Metrics use median/MAD robust z-scores. A flat channel is confirmed immediately.
Otherwise, a channel needs at least two independent failures to be marked bad.
Single-test candidates are saved for review but are not silently altered.

Confirmed names stay in `raw.info["bads"]`; they are not deleted. Reasons and
all metrics are saved in `bad_channel_metrics.csv`. `06_bad_channels.png` shows
confirmed recorded bad channels. When none are found, `06_bad_channels.txt`
states this explicitly and lists missing channels separately in metadata.

## 6. Annotate large temporal artifacts

**Definition.** A temporal artifact is a bad time interval, not a bad electrode.
Examples include electrode pops, saturation, very large movement, and extreme
transients.

`src.artifacts.annotate_large_artifacts()` scans 1-second windows every 0.5 s,
excluding confirmed bad channels. A window is marked when either:

- any retained EEG channel exceeds 500 µV peak-to-peak; or
- global median peak-to-peak is at least 100 µV and at least 10 robust MADs
  above the recording's typical window.

Intervals receive 0.25 s padding and overlapping detections are merged. They
are added as `BAD_amplitude` or `BAD_movement` MNE annotations. Samples are not
cut from the continuous recording. ICA fitting and epoch creation honor these
annotations. The exact onset, duration, metric, and reason are saved in
`temporal_artifacts.csv` and shown in `07_artifact_annotations.png`.

## 7. Fit ICA on a temporary copy

**Definition.** Independent component analysis separates reproducible spatial
source patterns. It is used here only for clear physiological artifacts, mainly
ocular activity because the dataset has no dedicated EOG or ECG channels.

`src.ica.fit_ica()` makes a temporary 1–40 Hz copy for numerical stability and
fits extended Infomax with 99% explained PCA variance, random state 42, and a
maximum of 1000 iterations. BAD intervals and bad channels are excluded from
the fit. With `--no-ica-downsampling`, this temporary copy remains at the final
120 Hz; without the option, only the temporary ICA fitting copy is reduced to
100 Hz. The final EEG is always the separate 1–50 Hz, 120 Hz copy. The older
`--no-downsampling` option name is retained as a backward-compatible alias.

## 8. Rank, inspect, and review every ICA component

ICLabel calculates probabilities for brain, muscle artifact, eye blink, heart
beat, line noise, channel noise, and other. The five known artifact classes are
summed into `iclabel_artifact_probability`. Ranking uses the interpretable
contrast `known-artifact probability - brain probability`: artifact-positive
components appear first, uncertain `other` components remain in the middle,
and brain-positive components appear last. Original `IC###` indices never
change. Review material is:

- `08_ica_probabilities.png`: stacked class probabilities in ranked order;
- `08_ica_components_ranked_p*.png`: ranked, indexed topographies;
- `09_ica_sources_ranked_p*.png`: ranked source time courses;
- `10_ica_properties_ranked_p*.png`: ranked topography, 20-second time course,
  and 1–50 Hz PSD;
- `ica_component_scores.csv`: all seven probabilities, artifact rank,
  prediction, candidate flag, and the original frontal/low-frequency metrics.

A candidate is prefilled only when (1) the winning ICLabel class is a known
artifact, (2) the summed known-artifact probability is at least 0.60, and (3)
the strongest individual artifact class is at least 0.30. By default these
thresholds generate review proposals only. The explicit
`--skip-manual-ica-review` option changes them into automatic exclusions.

An ocular component should have a plausible frontal field, blink/eye-movement
time course, and low-frequency spectrum together. An unusual PSD alone is not a
reason for removal. Oscillatory neural components are retained.

Review mode prefills indices and probability-based reasons under
`ica.manual_exclude_components` and `ica.manual_exclude_reasons`. It also writes
`ica.manual_review_confirmed.<subject>: false`. A reviewer must inspect the
ranked evidence, edit the list/reasons where necessary, and change the flag to
`true`. An explicit `[]` records a decision to remove nothing. Confirmed human
decisions are never overwritten by later review runs, and the clean runner
stops before processing any subject whose flag is not `true`.

With `--skip-manual-ica-review`, the pipeline bypasses that confirmation gate
and applies the newly calculated proposal. The exact automatic lists and
probability reasons are written separately under
`ica.automatic_exclude_components` and `ica.automatic_exclude_reasons`; manual
lists and confirmation flags are preserved. `decisions.json` and the batch QC
table record `ica_selection_mode: automatic_iclabel` and
`automatic_ica_removal: true`.

ICLabel was designed for common-average-referenced, approximately 1–100 Hz EEG
fit with extended Infomax. This pipeline matches extended Infomax but gives the
model the exact 1–40 Hz acquisition-reference copy used for ICA fitting.
Probabilities are therefore explicitly advisory, particularly for muscle and
prefrontal classifications. The final signal remains 1–50 Hz at 120 Hz.

## 9. Apply selected ICA removal

ICA is applied to another copy of the annotated 1–50 Hz signal. The fitted
temporary 1–40 Hz copy never replaces the final data. QC includes:

- `11_removed_ica_components.png`, with index, topography, time course, PSD,
  and reason;
- `12_before_after_ica.png`, on identical channels, interval, and scale;
- `13_ica_removed_signal.png`, showing before, after, and their difference.

The comparison interval is chosen where the selected ICA removal has its
largest representative-channel RMS, making over-cleaning easier to see.

## 10. Interpolate confirmed bad recorded channels

**Definition.** Interpolation reconstructs a recorded but unusable channel from
nearby scalp electrodes. It is not an operation for a channel that never
existed.

Only confirmed bad channels with a finite electrode position are interpolated.
After reconstruction they are included in the final average reference. The
original bad decision remains in tables and `decisions.json`. Each interpolation
plot shows before, after, and nearest recorded neighbors (`14_interpolation_*`).
If no interpolation occurs, `14_interpolation.txt` records that fact.

## 11. Apply a consistent final reference

**Definition.** Re-referencing subtracts a common reference signal from every
EEG channel. A consistent reference is essential for comparable scalp spectra.

After bad-channel handling, `src.preprocessing.rereference()` applies an average
reference. The Pz source reference and final average reference are both saved.
`15_reference_comparison.png` shows the effect on identical traces and scale.

## 12. Verify the final continuous signal

The final continuous copy remains 1–50 Hz and 120 Hz and preserves all BAD
annotations. It is saved before epoching. QC includes:

- `16_final_clean_signal.png`;
- `17_raw_vs_clean.png`, with a relatively clean interval and, when available,
  an annotated artifact interval;
- `18_raw_vs_clean_psd.png`, computed with identical Welch settings and no
  independent normalization.

The PSD comparison is essential: inspect broadband slope and alpha/beta peaks,
not just whether a trace appears smoother. Re-referencing legitimately changes
absolute power, so that change is visible rather than hidden.

## 13. Create fixed-length resting epochs

`mne.make_fixed_length_epochs()` creates non-overlapping 4-second epochs. No
baseline correction is used. Four seconds gives approximately 0.25 Hz nominal
Fourier-bin spacing. Epochs that overlap a `BAD_*` annotation are omitted.

The primary cohort is not selected by retained duration. After all primary
subject-level quantities are calculated, `duration_qc_analysis/` reports a
prespecified sensitivity analysis requiring at least 60 seconds, equivalent to
15 retained epochs. It reuses the same subject-level quantities so the check
isolates duration-based subject exclusion rather than changing the shared
electrode set or feature definitions. In the demographic-matched cohort, both
members of a pair are removed if either member fails this threshold.

## 14. Reject residual contaminated epochs

For each remaining epoch, the pipeline calculates maximum channel peak-to-peak
amplitude. It rejects an epoch when any of these configurable conservative rules
is met:

- absolute maximum is at least 200 µV;
- the epoch maximum is at least 8 robust MADs above other epochs;
- a channel's log peak-to-peak is at least 8 robust MADs above that same
  channel's values in other epochs.

The 200 µV absolute rule was tightened from 500 µV after residual frontal blink
activity with approximately 418 µV peak-to-peak amplitude survived ICA and the
robust rules in `sub-014`. Across the first 64 processed subjects, applying the
new cutoff retrospectively marked about 2.8% of otherwise accepted epochs; a
100 µV cutoff would have marked about 24% and was therefore not selected. For
`sub-014`, 26 of 27 previously accepted epochs exceed 200 µV, confirming that
the residual frontal contamination is participant-wide rather than confined to
the example panel; its retained-epoch count must therefore be reviewed after
rerunning. The channel-wise rule catches isolated pops without rejecting a participant merely
because one channel has consistently larger physiological amplitude.
Every initial epoch receives an accepted flag, drop reason, metrics, and trigger
channel in `epoch_rejection.csv`. `19_epoch_rejection.png` shows accepted and
representative rejected examples. `20_final_epoch_psd.png` is computed only
from retained epochs, with no per-epoch normalization.

## 15. Save outputs and participant summary

The pipeline saves:

```text
processed/
├── cleaned_raw/<subject>_task-Rest_desc-cleaned_raw.fif
├── epochs/<subject>_task-Rest_desc-cleaned_epo.fif
├── ica/<subject>_task-Rest_desc-preprocessing-ica.fif
├── qc/<subject>/01...21 + decisions.json
├── metadata/subjects/<subject>/*.csv
├── metadata/preprocessing_qc.csv
└── logs/<subject>.log
```

`21_summary.png`, `decisions.json`, the participant log, and
`preprocessing_qc.csv` expose every important choice: file/group, source and
missing channels, bad-channel reasons, interpolation, annotations, ICA review,
epoch counts, usable duration, references, filters, notch decision, and both
final and temporary ICA sampling rates.

## 16. Reproducible commands

Inspect and test:

```bash
conda run -n MNE_August2026 python -m pip install -r requirements-icalabel.txt
conda run -n MNE_August2026 python scripts/inspect_dataset.py
conda run -n MNE_August2026 python -m unittest discover -s tests -v
```

Review every subject with ICA kept at the final 120 Hz rate:

```bash
conda run -n MNE_August2026 python scripts/run_preprocessing.py \
  --review-only --no-ica-downsampling --overwrite
```

After every prefilled ICA entry has been visually checked and every
`manual_review_confirmed` flag is `true`, clean all recordings at 120 Hz:

```bash
conda run -n MNE_August2026 python scripts/run_preprocessing.py \
  --no-ica-downsampling --overwrite
```

Explicit unattended alternative:

```bash
bash scripts/run_full_cleaning.sh clean --skip-manual-ica-review --overwrite
```

This is not equivalent to human review. It applies the configured ICLabel
thresholds and records the run as automatic throughout the provenance outputs.

Do not use `--allow-unreviewed` for the final scientific dataset. That option is
available only for controlled debugging and conservatively removes no ICA
components from unreviewed recordings.
