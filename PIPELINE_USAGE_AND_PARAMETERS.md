# How to run and configure the EEG cleaning pipeline

This guide answers two practical questions:

1. How should the pipeline be run safely?
2. What does every setting in `config/preprocessing.yaml` mean?

The scientific rationale and each processing stage are described separately in
[`PREPROCESSING_PIPELINE.md`](PREPROCESSING_PIPELINE.md).

## The short answer: what must be updated?

For this dataset, most defaults should **not** be changed. Before cleaning all
149 recordings, update these items:

1. Confirm `project.dataset_dir` and `project.output_dir` if your directory
   layout differs from this repository.
2. Run the ICA `review` stage.
3. Inspect the artifact-ranked ICA plots. The review stage automatically
   prefills `ica.manual_exclude_components` and probability-based reasons.
4. Edit any incorrect proposal, strengthen each reason with the visual evidence
   you confirmed, and set `ica.manual_review_confirmed.<subject>` to `true`.

The current configuration has no confirmed manual reviews. The full cleaning
command therefore refuses manual-mode cleaning until every ICA decision is
recorded, unless the explicit automatic override is used.

Do not change the prespecified 1–100 Hz filter, 60 Hz notch, or artifact
thresholds merely to make traces look smoother. Changes should be justified by
pilot QC and then applied identically to both groups.

## Configuration-file syntax

Although the file ends in `.yaml`, it uses JSON-formatted YAML. JSON is valid
YAML, and using this restricted form avoids requiring the PyYAML package.

Consequences:

- Keys and text values use double quotes.
- Boolean values are lowercase `true` or `false`.
- An absent value is written as `null`.
- Lists use square brackets: `[0, 2]`.
- Do not add comments inside the file.
- Do not leave a trailing comma after the last item.
- ICA reason keys are strings, for example `"0"`, while exclusion-list values
  are integers, for example `[0]`.

The pipeline validates the configuration before loading EEG. It stops if the
final filter is not exactly 1–100 Hz, the 60 Hz notch is disabled, the sampling
rate does not preserve the band, epoch duration is invalid, baseline correction
is requested, or AutoReject is enabled.

## Before the first run

From the project root, confirm that these exist:

```text
config/preprocessing.yaml
dataset/participants.tsv
dataset/sub-001/eeg/sub-001_task-Rest_eeg.set
dataset/sub-001/eeg/sub-001_task-Rest_eeg.fdt
```

Both `.set` and `.fdt` are required. Confirm the conda environment:

```bash
conda run -n MNE_August2026 python -c "import mne; print(mne.__version__)"
```

Show the bash runner's help:

```bash
bash scripts/run_full_cleaning.sh --help
```

## Recommended run sequence

### Step 1: reproduce the two-subject pilot

This is optional if the checked pilot results and QC already exist, but it is a
useful environment check:

```bash
bash scripts/run_full_cleaning.sh pilot --overwrite
```

This runs inspection and tests, then cleans `pilot_subjects` (`sub-001` and
`sub-101`). Source data are read at 500 Hz and final data are saved at 250 Hz.

Inspect at least:

- `processed/qc/sub-001/11_removed_ica_components.png`;
- `processed/qc/sub-001/12_before_after_ica.png`;
- `processed/qc/sub-001/18_raw_vs_clean_psd.png`;
- `processed/qc/sub-001/19_epoch_rejection.png`;
- the equivalent files for `sub-101`;
- `processed/metadata/preprocessing_qc.csv`.

### Step 2: generate ICA review material for all subjects

```bash
bash scripts/run_full_cleaning.sh review --overwrite
```

This command performs dataset inspection, runs all tests, filters and assesses
each recording, applies CAR, fits ICA at 250 Hz, runs ICLabel, writes ranked ICA review
material, and prefills an unconfirmed proposal in the configuration. It stops
before ICA removal, interpolation, final re-referencing, and epoching.

For each participant, inspect:

```text
08_ica_probabilities.png
08_ica_components_ranked_p*.png
09_ica_sources_ranked_p*.png
10_ica_properties_ranked_p*.png
processed/metadata/subjects/<subject>/ica_component_scores.csv
```

All displays and CSV rows are sorted by known-artifact-minus-brain probability,
so uncertain `other` components fall between likely artifacts and likely brain
signals. Original component numbers remain visible and are the numbers used in
the configuration. The score table and prefilled lists are suggestions only.

### Step 3: record every ICA decision

Review and, where necessary, edit the three ICA mappings in
`config/preprocessing.yaml`. A valid confirmed example is:

```json
"manual_exclude_components": {
  "sub-001": [0],
  "sub-002": [],
  "sub-003": [0, 2]
},
"manual_exclude_reasons": {
  "sub-001": {"0": "clear eye-blink component"},
  "sub-002": {},
  "sub-003": {
    "0": "clear eye-blink component",
    "2": "clear horizontal eye-movement component"
  }
},
"manual_review_confirmed": {
  "sub-001": true,
  "sub-002": true,
  "sub-003": true
}
```

Use `[]` to show that a subject was reviewed and no component should be removed.
Every component in an exclusion list must have a reason. A reason such as
“unusual PSD” is insufficient by itself; use converging topography, time-course,
and PSD evidence. Change the confirmation flag from the automatically written
`false` to `true` only after that inspection.

### Step 4: run the reviewed full cleaning

```bash
bash scripts/run_full_cleaning.sh clean --overwrite
```

The runner checks all confirmation flags before processing the first subject.
If any flag is absent or false, it stops and names the unreviewed subjects. This
prevents a machine-prefilled proposal from being treated as a scientific
decision.

The final signal and ICA/ICLabel input are resampled from 500 Hz to 250 Hz after
the 1–100 Hz band-pass and 60 Hz notch. ICA and ICLabel use a common-average
reference calculated after bad-channel detection.

### Step 5: inspect batch-level QC

Do not proceed directly to group spectral analysis. First inspect:

```text
processed/metadata/preprocessing_qc.csv
processed/logs/<subject>.log
processed/qc/<subject>/21_summary.png
processed/metadata/subjects/<subject>/epoch_rejection.csv
```

Look for unusually many bad channels, low epoch retention, short usable
duration, unexpectedly many ICA removals, or subjects with PSD changes that
cannot be explained by the filter/reference/artifact decisions.

## Running selected subjects

The bash runner targets the entire dataset or the configured pilot. For one
participant, use:

```bash
conda run -n MNE_August2026 python scripts/preprocess_subject.py sub-025 \
  --review-only --no-ica-downsampling --overwrite
```

After updating that participant's ICA entry:

```bash
conda run -n MNE_August2026 python scripts/preprocess_subject.py sub-025 \
  --no-ica-downsampling --overwrite
```

For several selected participants:

```bash
conda run -n MNE_August2026 python scripts/run_preprocessing.py \
  --subjects sub-025 sub-026 sub-027 \
  --review-only --no-ica-downsampling --overwrite
```

After reviewing them, omit `--review-only` to create their cleaned files.

## Bash runner parameters

The syntax is:

```bash
bash scripts/run_full_cleaning.sh MODE [OPTIONS]
```

### Modes

| Mode | Meaning |
|---|---|
| `pilot` | Clean only the IDs under `pilot_subjects`. These IDs need completed ICA entries. |
| `review` | Generate ICA review material for every participant without removing components. |
| `clean` | Run the complete cleaning pipeline. Stops unless every participant has an explicit ICA entry. |

### Options

| Option | Default | Meaning |
|---|---|---|
| `--config PATH` | `config/preprocessing.yaml` | Use a different configuration file. Prefer an absolute path if it is outside the project. |
| `--env NAME` | `MNE_August2026` | Conda environment passed to every Python command. |
| `--overwrite` | off | Replace generated outputs for the same subjects. It never overwrites source files under `dataset/`. |
| `--workers N` | `2` | Process independent participants concurrently. Use `1` when memory is constrained. |
| `--no-progress` | off | Hide the participant progress bar and ETA. |
| `--skip-manual-ica-review` | off | Bypass visual confirmation and automatically apply high-confidence ICLabel proposals in `clean` or `pilot` mode. The automatic lists and provenance are saved separately. |
| `-h`, `--help` | — | Print usage and exit. |

The batch runner verifies and reuses complete per-subject outputs, then safely
rebuilds any incomplete participant from the start. An interrupted cohort run
therefore continues with only missing subjects. Use `--overwrite` when
deliberately regenerating every result after a configuration or code change.

## Python runner parameters

These are mainly useful for debugging or selected-subject runs.

| Parameter | Meaning |
|---|---|
| `subject_id` | Positional ID for `preprocess_subject.py`, such as `sub-025`. |
| `--subjects ID ...` | Restrict `run_preprocessing.py` to the listed IDs. Without it, all recordings are selected. |
| `--workers N` | Process independent subjects concurrently. The master runner defaults to two; use one when memory is constrained. Each worker keeps the same per-subject ICA configuration and random seed. |
| `--no-progress` | Hide the participant-level progress bar and ETA. |
| `--config PATH` | Select the configuration file. |
| `--review-only` | Stop after the ICA review files are saved. No cleaned raw/epochs are produced. |
| `--no-ica-downsampling` | Backward-compatible option that disables optional ICA-only downsampling. It has no effect with the default 250 Hz ICA and final rates. The old `--no-downsampling` spelling is an alias. |
| `--overwrite` | Replace generated outputs for the selected participant(s). |
| `--allow-unreviewed` | Debugging only: continue with no ICA removal for an unreviewed subject. Do not use for the final scientific dataset. |
| `--skip-manual-ica-review` | Apply the newly calculated ICLabel proposal without visual confirmation. Mutually exclusive with `--allow-unreviewed`. |

## Meaning of every configuration parameter

### `project`

| Parameter | Current value | Meaning and whether to change it |
|---|---:|---|
| `project.dataset_dir` | `dataset` | Directory containing `participants.tsv` and `sub-*` folders. Change only if the dataset moves. Relative paths are interpreted from the project root when using the bash runner. |
| `project.output_dir` | `processed` | New directory for cleaned FIF, epochs, ICA, QC, metadata, and logs. It must not be the source dataset directory. |
| `project.task` | `Rest` | BIDS task label used to discover `*_task-Rest_eeg.set`. Keep `Rest` for this dataset. It is case-sensitive. |

### `filter`

| Parameter | Current value | Meaning and effect of changing it |
|---|---:|---|
| `filter.l_freq` | `1.0` Hz | Final high-pass cutoff. Locked to 1 Hz by the project specification. Raising it could distort slow activity and aperiodic slope. |
| `filter.h_freq` | `100.0` Hz | Final low-pass cutoff and ICLabel input boundary. |
| `filter.method` | `fir` | MNE filter family. FIR is used for a stable offline zero-phase response. Keep it unchanged unless the entire filter is revalidated. |
| `filter.phase` | `zero` | Applies the FIR without a systematic time shift. Keep `zero`. |
| `filter.notch_enabled` | `true` | Applies the required line-noise notch because 60 Hz is inside the retained band. |
| `filter.notch_freq_hz` | `60.0` Hz | Recorded power-line frequency removed before resampling. |
| `filter.notch_width_hz` | `2.0` Hz | Width of the 60 Hz notch. |
| `filter.reason` | explanatory text | Human-readable reason saved in logs and QC metadata. Update it only if the filter decision changes. |

The cleaned recording is always validated as exactly 1–100 Hz. These settings
describe the scientific analysis band, not only a plotting range.

### `resampling`

| Parameter | Current value | Meaning and effect of changing it |
|---|---:|---|
| `resampling.target_sfreq` | `250.0` Hz | Final sampling rate for cleaned EEG, epochs, ICA, and ICLabel. Its 125 Hz Nyquist frequency leaves a 25 Hz guard band above 100 Hz. |
| `resampling.method` | `fft` | MNE resampling method used by the pipeline. MNE applies anti-aliasing as part of resampling. |
| `resampling.npad` | `auto` | Lets MNE choose FFT padding for efficient resampling. Keep this fixed for reproducible processing. |

Filtering and the notch are performed before resampling. A target at or below
200 Hz would put the 100 Hz cutoff at or above Nyquist and is rejected.

### `channels`

| Parameter | Current value | Meaning and effect of changing it |
|---|---:|---|
| `channels.auxiliary_names` | `Resp`, `X`, `Y`, `Z` | Recorded non-EEG channels excluded from EEG cleaning. Their names remain in provenance. Add a name only after confirming it is not scalp EEG. |
| `channels.flat_std_uv` | `0.1` µV | A channel with full-recording standard deviation below this is flat and immediately confirmed bad. Increasing it makes flat-channel detection more aggressive. |
| `channels.metric_robust_z` | `6.0` | Upper robust-z threshold for variance, peak-to-peak amplitude, and high-frequency-power metrics. Lower values flag more candidates. |
| `channels.correlation_robust_z` | `6.0` | A channel whose median correlation is this many robust deviations below other channels receives a poor-correlation flag. |
| `channels.minimum_independent_flags` | `2` | Non-flat channels need at least this many different failures before being confirmed bad. Increasing it is more conservative; decreasing it risks interpolation of valid EEG. |
| `channels.minimum_median_correlation` | `-0.2` | Absolute lower correlation boundary. A value below this produces a poor-correlation flag even if the robust-z rule is not met. Raising it flags more channels. |

**Robust z-score** means distance from the median divided by a scale derived
from the median absolute deviation (MAD). It is less sensitive to a few extreme
channels than a mean/standard-deviation z-score.

A flagged channel is not immediately deleted. Confirmed bad recorded channels
are kept, excluded from ICA fitting, and interpolated later if they have valid
positions. Missing channels are never interpolated.

### `artifacts`

| Parameter | Current value | Meaning and effect of changing it |
|---|---:|---|
| `artifacts.window_sec` | `1.0` s | Length of each continuous-data window inspected for large transients. |
| `artifacts.step_sec` | `0.5` s | Distance between window starts. Because it is smaller than the window, detections overlap. A smaller step increases temporal coverage and computation. |
| `artifacts.absolute_peak_to_peak_uv` | `500.0` µV | Mark a window `BAD_amplitude` when any retained EEG channel reaches this peak-to-peak range. Lower values reject more time. |
| `artifacts.global_peak_to_peak_robust_z` | `10.0` | Mark globally unusual windows when median channel peak-to-peak is this many robust MADs above typical windows. |
| `artifacts.minimum_global_peak_to_peak_uv` | `100.0` µV | The global robust rule must also exceed this absolute floor, preventing tiny but statistically unusual windows from being marked. |
| `artifacts.padding_sec` | `0.25` s | Extra time added around a detected window to include artifact onset and recovery. Larger padding can reject more 4-second epochs. |
| `artifacts.merge_gap_sec` | `0.25` s | Neighboring detections separated by no more than this are merged into one annotation. |

Peak-to-peak amplitude is `maximum − minimum` within a channel and time window.
These settings add MNE `BAD_*` annotations; they do not delete continuous
samples. Epochs overlapping them are later rejected.

### `ica`

| Parameter | Current value | Meaning and effect of changing it |
|---|---:|---|
| `ica.method` | `infomax` | ICA algorithm used by MNE. Keep it fixed so decompositions are comparable. |
| `ica.extended` | `true` | Enables extended Infomax, which can model both super-Gaussian and sub-Gaussian sources. |
| `ica.n_components` | `0.99` | Retain enough PCA dimensions to explain 99% of variance before ICA. This means 99% variance, **not 99 components**. |
| `ica.random_state` | `42` | Fixed random seed for reproducible ICA initialization. Keep fixed across subjects/reruns. |
| `ica.max_iter` | `1000` | Maximum ICA optimization iterations. Increase only if logs show non-convergence; it may lengthen runtime. |
| `ica.fit_l_freq` | `1.0` Hz | High-pass cutoff shared by the ICA and ICLabel input. |
| `ica.fit_h_freq` | `100.0` Hz | Low-pass cutoff shared by the ICA and ICLabel input. |
| `ica.temporary_resample_sfreq` | `250.0` Hz | ICA/ICLabel sampling rate. It matches the final data, so no extra ICA-only downsampling occurs. |
| `ica.suggestion_frontal_correlation` | `0.30` | Minimum absolute correlation between an ICA source and the available frontal EEG average for an ocular-review suggestion. It never causes automatic removal. |
| `ica.suggestion_frontal_weight_ratio` | `1.5` | Minimum frontal-versus-whole-scalp component-weight ratio for the same suggestion. Both suggestion criteria must pass. |
| `ica.iclabel_enabled` | `true` | Runs MNE-ICALabel and creates the ranked probabilities and proposal. Disabling it removes the new ranking/proposal workflow. |
| `ica.iclabel_backend` | `onnx` | Uses the reproducible ONNX inference backend supplied by `onnxruntime`. |
| `ica.iclabel_artifact_probability_threshold` | `0.60` | Minimum summed probability across muscle, eye, heart, line, and channel artifact classes for a prefilled candidate. |
| `ica.iclabel_minimum_class_probability` | `0.30` | Requires at least one known artifact class to reach this probability. Combined with the 0.60 total-artifact threshold, this retains mixed artifact components whose probability is split across classes. |
| `ica.manual_exclude_components` | subject mapping | Review mode prefills these integer ICA indices. A reviewer edits the list when the plots disagree; use `[]` when nothing is removed. |
| `ica.manual_exclude_reasons` | subject/component mapping | Review mode prefills probability evidence. The reviewer should record the converging visual evidence for each retained exclusion. |
| `ica.manual_review_confirmed` | subject/Boolean mapping | Clean mode requires `true`. Review mode writes `false` and never overwrites an existing `true`. This is the human safety gate. |
| `ica.automatic_exclude_components` | subject mapping | Audit record of the exact ICLabel component list used by a `--skip-manual-ica-review` run. It does not overwrite the manual mapping. |
| `ica.automatic_exclude_reasons` | subject/component mapping | ICLabel labels and probabilities corresponding to each automatically used exclusion. |

ICLabel and the original frontal suggestion metrics are screening aids. There
are no dedicated EOG or ECG channels. The ICA input now matches ICLabel's CAR
and 1–100 Hz assumptions, but component topography, source time course, PSD,
and before/after effects should still be reviewed together.

### `epochs`

| Parameter | Current value | Meaning and effect of changing it |
|---|---:|---|
| `epochs.duration_sec` | `4.0` s | Fixed epoch duration. The nominal Fourier-bin spacing is approximately `1 / duration`, or 0.25 Hz here. Changing it affects both epoch counts and spectral resolution. |
| `epochs.overlap_sec` | `0.0` s | Time shared by consecutive epochs. Zero gives independent, non-overlapping windows. Keep below `duration_sec`. |
| `epochs.peak_to_peak_uv` | `200.0` µV | Reject a remaining epoch if any EEG channel reaches this peak-to-peak value. This stricter residual-artifact guard catches large blinks that survive ICA; a retrospective check marked about 2.8% of otherwise accepted epochs among the first 64 processed subjects. |
| `epochs.robust_z` | `8.0` | Residual rejection threshold used for both the epoch-wide maximum and each channel's log peak-to-peak relative to its other epochs. |
| `epochs.baseline` | `null` | Baseline correction is forbidden for this resting-state spectral pipeline. The validator requires `null`. |
| `epochs.autoreject_enabled` | `false` | AutoReject is not implemented or used. The validator requires `false`; rejection remains explicit and auditable. |

The channel-wise robust rule catches an isolated channel pop without rejecting
all epochs merely because a channel has consistently higher physiological
amplitude. Rejection reasons and trigger channels are saved per epoch.

### `qc`

These parameters change plots only; they do not change the cleaned signal.

| Parameter | Current value | Meaning and effect of changing it |
|---|---:|---|
| `qc.preferred_channels` | `Fp1`, `Fz`, `Cz`, `CPz`, `Pz`, `O1` | Preferred channels for readable trace/comparison plots. Only channels actually present are selected; fallbacks are chosen automatically. |
| `qc.trace_start_sec` | `20.0` s | Start time of the standard representative trace window. Artifact-targeted plots may choose a more informative interval automatically. |
| `qc.trace_duration_sec` | `10.0` s | Duration of standard trace plots and identical-window comparisons. |
| `qc.psd_fmin_hz` | `1.0` Hz | Lower frequency displayed in PSD QC. |
| `qc.psd_fmax_hz` | `80.0` Hz | Upper frequency displayed for raw-data PSD so the original 60 Hz environment is visible. |
| `qc.final_psd_fmax_hz` | `50.0` Hz | Upper frequency displayed for filtered/final/epoch PSD. It should match the final low-pass. |
| `qc.dpi` | `150` | Saved PNG resolution. Higher values increase image dimensions, save time, and disk use. |
| `qc.ica_components_per_page` | `12` | Number of ICA component topographies/source traces per numbered page. It does not alter ICA itself. |

### `pilot_subjects`

| Parameter | Current value | Meaning and effect of changing it |
|---|---:|---|
| `pilot_subjects` | `sub-001`, `sub-101` | IDs used by the `pilot` mode. Keep at least one PD and one Control with explicit ICA review decisions. Changing this does not affect the full batch. |

## How threshold direction changes behavior

Use this summary when evaluating a justified configuration change:

| Change | Typical consequence |
|---|---|
| Lower an amplitude or robust-z threshold | More channels/windows/epochs are flagged; cleaning becomes more aggressive. |
| Raise an amplitude or robust-z threshold | Fewer items are flagged; more contamination may remain. |
| Increase artifact padding | More neighboring time and potentially more epochs are excluded. |
| Lower `minimum_independent_flags` | More channels are interpolated, including a higher risk of valid channels. |
| Add more ICA components manually | More signal is subtracted; neural/spectral removal risk increases. |
| Increase epoch duration | Better nominal frequency resolution, fewer epochs, greater chance an artifact contaminates a whole epoch. |

Never select thresholds independently for PD and Control participants. That
would introduce group-dependent preprocessing bias.

## Output locations and completion checks

A successful full run creates:

```text
processed/
├── cleaned_raw/      final annotated 1–100 Hz, 250 Hz continuous EEG
├── epochs/           accepted 4-second epochs, 250 Hz, no baseline
├── ica/              fitted ICA solutions
├── qc/<subject>/     ordered stages 01–21 and decisions.json
├── metadata/         dataset tables and preprocessing_qc.csv
└── logs/             one log per participant
```

For 149 input recordings, expect 149 cleaned raw FIF files, 149 epoch FIF files,
149 ICA files, 149 subject QC directories, and 149 rows in
`preprocessing_qc.csv`. File counts alone are not enough: inspect retention,
decisions, raw-versus-clean PSDs, and any extreme QC outliers before analysis.
