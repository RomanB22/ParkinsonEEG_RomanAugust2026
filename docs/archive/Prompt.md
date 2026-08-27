# Task: Build a simple resting-state EEG preprocessing pipeline in Python

I have a locally downloaded resting-state EEG dataset containing:

* Parkinson's disease participants
* Healthy controls
* Resting-state, eyes-open recordings

The final scientific goal is to compare:

* periodic / oscillatory activity
* aperiodic / 1/f activity

between Parkinson's disease and control participants.

The dataset is already downloaded locally. **Do not write code to download it.**

Build a simple, readable, modular preprocessing pipeline using primarily **MNE-Python**.

---

# Main requirements

The pipeline should:

1. Inspect the dataset first.
2. Load each subject.
3. Preserve subject and channel metadata.
4. Filter EEG from **1 to 50 Hz**.
5. Detect bad channels.
6. Detect large temporal artifacts.
7. Perform ICA.
8. Identify and remove clear ICA artifact components.
9. Interpolate bad recorded channels when appropriate.
10. Re-reference consistently.
11. Divide the resting recording into fixed-length epochs.
12. Reject clearly contaminated epochs.
13. Save cleaned signals.
14. Save QC information.
15. Generate plots for **every important preprocessing step**.

Do not implement the final periodic/aperiodic analysis yet.

---

# Programming philosophy

Use **KISS** and **SOLID** principles.

The most important requirement is that the code remains easy to understand and modify.

## KISS

Keep everything as simple as possible.

Prefer:

```python
def filter_eeg(raw, config):
    ...
```

over complicated class hierarchies or unnecessary abstractions.

Avoid:

* deep inheritance
* factories
* dependency injection frameworks
* unnecessary design patterns
* excessive abstraction
* overly generic code
* premature optimization

If a normal function is sufficient, use a function.

---

## SOLID

Apply SOLID pragmatically, without over-engineering.

### Single Responsibility Principle

Each function/module should do one main thing.

For example:

```python
load_subject(...)
filter_signal(...)
detect_bad_channels(...)
fit_ica(...)
apply_ica(...)
epoch_signal(...)
plot_psd(...)
save_qc(...)
```

Do not create one giant function such as:

```python
process_everything(...)
```

containing hundreds of lines.

### Open/Closed Principle

Important preprocessing choices should come from configuration rather than requiring modification of the implementation.

For example:

```yaml
filter:
  l_freq: 1.0
  h_freq: 100.0
```

### Interface Segregation

Keep function inputs simple.

For example:

```python
def plot_psd(raw, output_path, title):
    ...
```

rather than passing large objects containing unrelated information.

### Dependency Inversion

Do not hard-code paths or subject IDs deep inside functions.

Pass them as arguments or obtain them from configuration.

---

# Preferred packages

Use primarily:

```python
import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```

Optionally use:

```python
from pyprep.prep_pipeline import PrepPipeline
```

and:

```python
from autoreject import AutoReject
```

if they provide a clear advantage.

Do not add dependencies unless they are useful.

---

# First step: inspect the actual dataset

Before implementing preprocessing, inspect the local files.

Determine:

* file format
* subject IDs
* Parkinson vs Control labels
* sampling frequencies
* recording durations
* channel names
* number of channels
* EEG channels
* reference channels
* auxiliary channels
* whether EOG or ECG channels exist
* whether electrode coordinates exist
* channels shared across subjects
* channels missing in some participants

Do not assume all recordings have exactly the same channels.

Produce a summary table similar to:

```text
subject_id
group
file
sampling_rate
duration_sec
n_channels
n_eeg_channels
channel_names
reference_channels
```

---

# Preserve subject information

Metadata must remain associated with each subject throughout preprocessing.

At minimum preserve:

```text
subject_id
group
original_file
sampling_rate
recording_duration
original_channels
original_reference
```

Later append QC information.

---

# Important distinction between channel types

Keep these categories separate:

```python
good_channels = [...]
bad_channels = [...]
interpolated_channels = [...]
missing_channels = [...]
```

A missing channel is **not** a bad channel.

Do not interpolate a channel merely because another subject contains it.

---

# Suggested project structure

Keep the repository simple.

For example:

```text
eeg_preprocessing/
│
├── config.yaml
├── inspect_dataset.py
├── preprocessing.py
├── ica.py
├── qc.py
├── utils.py
├── run_subject.py
└── run_all.py
```

This structure is only a suggestion.

If fewer files make the code clearer, use fewer files.

---

# Configuration

Do not scatter constants throughout the code.

Use a small YAML configuration file.

Example:

```yaml
filter:
  l_freq: 1.0
  h_freq: 50.0

line_noise:
  frequency: 60
  notch_enabled: true

resampling:
  target_sfreq: 250.0

ica:
  enabled: true
  method: infomax
  n_components: 0.99
  random_state: 42

epochs:
  duration: 4.0
  overlap: 0.0

plots:
  start_sec: 0
  duration_sec: 20
  dpi: 150

bad_channels:
  use_pyprep: false

autoreject:
  enabled: false
```

---

# Preprocessing sequence

Implement approximately this sequence:

```text
RAW EEG
   ↓
channel + metadata inspection
   ↓
1–100 Hz filtering + 60 Hz notch
   ↓
resampling to 250 Hz
   ↓
bad-channel detection
   ↓
gross artifact annotation
   ↓
ICA
   ↓
remove clear ICA artifacts
   ↓
bad-channel interpolation
   ↓
re-reference
   ↓
fixed-length epochs
   ↓
bad-epoch rejection
   ↓
FINAL CLEAN EEG
```

Each important step must have corresponding QC plots.

---

# 1. Load EEG

Use the correct MNE reader based on the actual file format.

Examples include:

```python
raw = mne.io.read_raw_edf(path, preload=True)
```

or:

```python
raw = mne.io.read_raw_brainvision(vhdr_path, preload=True)
```

or another appropriate MNE reader.

Do not assume EDF unless inspection confirms it.

After loading, print/log:

```python
print(raw.info)
print(raw.ch_names)
print(raw.info["sfreq"])
```

---

# 2. Raw EEG plot

Before changing the signal, save a plot of the raw EEG.

During development, MNE can be used interactively:

```python
raw.plot(
    duration=20,
    n_channels=20,
    scalings="auto"
)
```

For batch processing, generate equivalent Matplotlib/MNE figures and save them automatically.

Plot the same representative channels and time window later so before/after comparisons are meaningful.

---

# 3. Raw PSD

Plot the PSD before cleaning.

For example:

```python
spectrum = raw.compute_psd(
    method="welch",
    fmin=1,
    fmax=min(100, raw.info["sfreq"] / 2)
)

spectrum.plot()
```

The raw PSD may extend beyond 50 Hz for QC purposes.

This is useful for checking:

* line noise
* high-frequency contamination
* abnormal channels
* overall spectral shape

---

# 4. Filter from 1 to 50 Hz

The final cleaned EEG must be filtered between:

```text
1 Hz
and
50 Hz
```

Use MNE:

```python
raw_filtered = raw.copy().filter(
    l_freq=1.0,
    h_freq=50.0
)
```

These values are a fixed methodological requirement.

Do not automatically change them.

Do not apply another high-pass filter later to the final signal.

---

# 5. Plot filtered signal

Plot the filtered EEG using the same:

* channels
* time interval
* scale where practical

as the raw plot.

Example:

```python
raw_filtered.plot(
    duration=20,
    n_channels=20,
    scalings="auto"
)
```

Also plot its PSD:

```python
filtered_psd = raw_filtered.compute_psd(
    method="welch",
    fmin=1,
    fmax=100
)

filtered_psd.plot()
```

---

# 6. Raw vs filtered comparison

Generate a direct comparison between:

```text
RAW EEG
vs
1–100 Hz FILTERED EEG + 60 Hz NOTCH
```

Use identical channels and time windows.

The purpose is to visually inspect what the filter changed.

---

# 7. Line noise

The recordings were acquired in the US, so expected power-line frequency is:

```python
LINE_FREQ = 60
```

The final signal is low-pass filtered at 100 Hz, so 60 Hz remains inside the
retained band. Apply the line-noise notch before resampling.

If needed, MNE provides:

```python
raw.notch_filter(freqs=[60])
```

The pipeline should default to:

```yaml
notch_enabled: true
```

because the 60 Hz acquisition line frequency is retained by the 100 Hz low-pass.

---

# 8. Bad-channel detection

Detect clearly problematic channels.

Possible criteria:

* flat signal
* extreme variance
* extreme peak-to-peak amplitude
* excessive noise
* abnormal PSD
* poor correlation with other EEG channels

MNE stores bad channels as:

```python
raw.info["bads"] = ["Fp1", "T8"]
```

Do not delete them immediately.

If useful, optionally use PyPREP:

```python
from pyprep.prep_pipeline import PrepPipeline
```

but keep the default implementation simple unless PyPREP clearly improves the workflow.

Store the reason for every detected bad channel.

Example:

```python
bad_channel_reasons = {
    "Fp1": ["high_variance"],
    "T8": ["flat_signal"]
}
```

---

# 9. Plot bad channels

Plot detected bad channels before interpolation.

For each bad channel, show where practical:

* time series
* PSD
* variance
* peak-to-peak amplitude

Clearly distinguish:

```text
BAD RECORDED CHANNEL
```

from:

```text
MISSING CHANNEL
```

---

# 10. Detect large temporal artifacts

Detect obvious periods containing:

* electrode pops
* saturation
* extreme movement
* very large transient artifacts

Prefer annotations instead of deleting continuous data.

MNE example:

```python
annotations = mne.Annotations(
    onset=[10.0],
    duration=[2.0],
    description=["BAD_movement"]
)

raw.set_annotations(annotations)
```

Preserve these annotations.

---

# 11. Plot temporal artifact annotations

Generate figures showing the signal and marked bad intervals.

For example:

```text
BAD_movement
BAD_amplitude
BAD_saturation
```

I should be able to see why each interval was marked as bad.

---

# 12. ICA

Use ICA to identify clear physiological artifacts.

Example:

```python
from mne.preprocessing import ICA

ica = ICA(
    n_components=0.99,
    method="infomax",
    fit_params=dict(extended=True),
    random_state=42
)

ica.fit(raw_filtered)
```

ICA settings should come from `config.yaml`.

---

# ICA filtering note

The final EEG must remain **1–100 Hz at 250 Hz sampling**.

If ICA requires different filtering for numerical stability, create a temporary copy:

```python
raw_for_ica = raw_filtered.copy()
```

and modify only that copy if necessary.

ICA and ICLabel should use a common-average-referenced copy of the same
1–100 Hz signal. Never replace the final recording with a differently filtered
ICA-only signal.

---

# 13. Plot ICA decomposition

ICA visualization is mandatory.

Use MNE functions such as:

```python
ica.plot_components()
```

to plot component topographies.

Also use:

```python
ica.plot_sources(raw_filtered)
```

to inspect component time courses.

Each component must have a clear index:

```text
IC000
IC001
IC002
...
```

---

# 14. ICA component properties

For every ICA component, provide enough information to inspect it.

Where practical show:

* component number
* topography
* time course
* PSD

MNE provides useful tools such as:

```python
ica.plot_properties(
    raw_filtered,
    picks=[0]
)
```

This is particularly useful because it combines several component diagnostics.

---

# 15. Detect EOG components

If EOG channels exist, use:

```python
eog_indices, eog_scores = ica.find_bads_eog(raw_filtered)
```

Then inspect the suggested components.

Do not automatically assume every suggested component should be removed without QC.

---

# 16. Detect ECG components

If ECG channels exist, use:

```python
ecg_indices, ecg_scores = ica.find_bads_ecg(raw_filtered)
```

Again, inspect the identified components.

---

# 17. ICA rejection

Remove only components that clearly correspond to artifacts such as:

* eye blinks
* eye movements
* ECG
* obvious non-neural contamination

Do not remove components simply because:

* their PSD looks unusual
* they contain oscillatory activity
* they have high power

Be conservative because the final analysis concerns spectral properties.

Store:

```python
ica.exclude = [0, 3]
```

and record the reason for each component.

Example:

```python
ica_rejection = {
    0: "eye blink",
    3: "ECG"
}
```

---

# 18. Plot removed ICA components

Create a dedicated figure showing only components selected for removal.

For each rejected component show:

```text
component index
topography
time course
PSD
reason for rejection
```

If no components are rejected, explicitly report:

```text
No ICA components removed
```

---

# 19. Apply ICA

Apply ICA to a copy:

```python
cleaned = raw_filtered.copy()

ica.apply(cleaned)
```

Do not overwrite the only copy of `raw_filtered`.

---

# 20. Before vs after ICA plot

This plot is mandatory.

Show:

```text
BEFORE ICA
vs
AFTER ICA
```

using identical:

* channels
* time intervals
* scales where possible

Whenever possible, include an interval containing an artifact targeted by ICA.

---

# 21. Plot what ICA removed

Calculate:

```python
removed_signal = (
    raw_filtered.get_data()
    - cleaned.get_data()
)
```

Plot representative channels from:

```text
Before ICA
After ICA
Difference
```

The difference represents the contribution removed by ICA.

This is important for verifying that ICA is not removing too much signal.

---

# 22. Interpolate bad channels

Interpolate only channels that:

* were actually recorded
* were clearly bad
* have valid electrode positions

MNE example:

```python
cleaned.interpolate_bads(
    reset_bads=False
)
```

Do not interpolate channels that were never present in the recording.

---

# 23. Plot interpolation

For every interpolated channel, show:

```text
before interpolation
after interpolation
neighboring channels
```

Record:

```python
interpolated_channels = [...]
```

---

# 24. Re-reference

Apply the same final reference strategy across subjects.

Prefer average reference unless inspection of the actual dataset indicates another reference is more appropriate.

MNE example:

```python
cleaned.set_eeg_reference(
    ref_channels="average"
)
```

Do this after bad-channel handling so a severely noisy electrode does not contaminate the reference.

Store:

```text
original_reference
final_reference
```

---

# 25. Plot re-referencing effect

Plot representative channels:

```text
before reference
vs
after reference
```

using the same time interval.

---

# 26. Final continuous cleaned signal

Before epoching, plot the final continuous signal.

Clearly label it:

```text
FINAL CLEANED EEG — 1–100 Hz + 60 Hz notch, 250 Hz sampling
```

This is the continuous signal that will be used to create resting-state epochs.

---

# 27. Raw vs final cleaned EEG

Generate a final direct comparison:

```text
RAW EEG
vs
FINAL CLEANED EEG
```

Use:

* same channels
* same intervals
* comparable scales

Show at least:

* one relatively clean interval
* one interval where preprocessing removed an artifact

---

# 28. Raw vs cleaned PSD

This is one of the most important QC figures.

Compute PSDs using the same parameters.

Example:

```python
raw_psd = raw.compute_psd(
    method="welch",
    fmin=1,
    fmax=50
)

clean_psd = cleaned.compute_psd(
    method="welch",
    fmin=1,
    fmax=50
)
```

Plot them together.

Do not normalize them independently.

I want to inspect whether cleaning modified:

* broadband spectral slope
* alpha peaks
* beta peaks
* other oscillatory peaks
* overall power

This is particularly important because the next analysis will estimate periodic and aperiodic components.

---

# 29. Epoch the resting EEG

Use fixed-length epochs.

Default:

```python
EPOCH_DURATION = 4.0
```

MNE example:

```python
epochs = mne.make_fixed_length_epochs(
    cleaned,
    duration=4.0,
    overlap=0.0,
    preload=True,
    reject_by_annotation=True
)
```

Do not apply baseline correction.

A 4-second epoch gives approximately:

```text
1 / 4 s = 0.25 Hz
```

nominal Fourier resolution.

---

# 30. Reject residual bad epochs

Reject only clearly contaminated epochs.

Possible criteria:

* extreme peak-to-peak amplitude
* extreme variance
* residual artifacts

MNE provides simple rejection mechanisms such as:

```python
epochs.drop_bad(
    reject={"eeg": 150e-6}
)
```

but thresholds should be configurable rather than blindly fixed.

Optionally support:

```python
from autoreject import AutoReject
```

but keep:

```yaml
autoreject:
  enabled: false
```

as the default unless testing shows it provides a clear benefit.

Prefer rejection of strongly contaminated epochs over aggressive reconstruction.

---

# 31. Plot epoch rejection

Generate plots showing:

* accepted epochs
* rejected epochs
* reason for rejection

Show representative examples of both.

Also report:

```text
initial epochs
rejected epochs
retained epochs
percent retained
usable duration
```

---

# 32. Final cleaned epoch PSD

Compute and plot the PSD from the final accepted epochs.

Example:

```python
psd = epochs.compute_psd(
    method="welch",
    fmin=1,
    fmax=50
)

psd.plot()
```

Do **not** run `specparam` yet.

---

# 33. Plot every processing stage

For every subject, create an ordered QC directory.

For example:

```text
qc/
└── subject_001/
    ├── 01_raw_signal.png
    ├── 02_raw_psd.png
    ├── 03_filtered_signal.png
    ├── 04_filtered_psd.png
    ├── 05_raw_vs_filtered.png
    ├── 06_bad_channels.png
    ├── 07_artifact_annotations.png
    ├── 08_ica_components.png
    ├── 09_ica_sources.png
    ├── 10_ica_properties.png
    ├── 11_removed_ica_components.png
    ├── 12_before_after_ica.png
    ├── 13_ica_removed_signal.png
    ├── 14_interpolation.png
    ├── 15_reference_comparison.png
    ├── 16_final_clean_signal.png
    ├── 17_raw_vs_clean.png
    ├── 18_raw_vs_clean_psd.png
    ├── 19_epoch_rejection.png
    ├── 20_final_epoch_psd.png
    └── 21_summary.png
```

If a stage does not occur for a subject, do not fake a plot.

Instead, record something such as:

```text
No bad channels detected
```

or:

```text
No ICA components removed
```

---

# 34. Keep plotting code separate

Put plotting functions in something like:

```text
qc.py
```

For example:

```python
def plot_raw_signal(...):
    ...

def plot_psd(...):
    ...

def plot_bad_channels(...):
    ...

def plot_ica_components(...):
    ...

def plot_before_after_ica(...):
    ...

def plot_raw_vs_cleaned(...):
    ...

def plot_epoch_rejection(...):
    ...
```

Do not mix large amounts of plotting code into preprocessing functions.

---

# 35. Use consistent plots

Before/after comparisons should use the same:

* channels
* time interval
* sampling units
* PSD parameters
* frequency axis
* amplitude scale where reasonable

Do not independently autoscale plots in ways that make before/after comparisons misleading.

---

# 36. Representative plotting channels

Do not assume every subject contains exactly the same channels.

You can define preferred channels:

```python
PREFERRED_CHANNELS = [
    "Fp1",
    "Fz",
    "Cz",
    "Pz",
    "O1"
]
```

Then select only the ones that actually exist.

For example:

```python
plot_channels = [
    ch for ch in PREFERRED_CHANNELS
    if ch in raw.ch_names
]
```

If too few are available, automatically choose representative EEG channels.

---

# 37. Save plots automatically

Save figures with sufficient resolution.

Example:

```python
fig.savefig(
    output_path,
    dpi=150,
    bbox_inches="tight"
)

plt.close(fig)
```

Do not require manual closing of figures during batch processing.

---

# 38. QC metadata

Create one QC row per participant.

At minimum include:

```text
subject_id
group
original_file

sampling_rate
recording_duration_sec

original_channels
n_original_channels

missing_channels

bad_channels
n_bad_channels

interpolated_channels
n_interpolated_channels

n_ica_components
ica_components_removed
n_ica_components_removed

n_epochs_initial
n_epochs_rejected
n_epochs_retained
percent_epochs_retained

usable_duration_sec

original_reference
final_reference

filter_low_hz
filter_high_hz
```

Save as:

```text
preprocessing_qc.csv
```

---

# 39. Channel availability table

After inspecting all subjects, create a channel availability table.

Example:

```text
             Fp1  Fp2  Fz  Cz  Pz  O1  O2
subject_01     1    1   1   1   1   1   1
subject_02     1    1   1   1   1   0   1
subject_03     1    1   1   1   1   1   1
```

Also determine:

```python
common_channels = [...]
```

but do not delete non-common channels from individual cleaned files.

We may use the common-channel list later for group analysis.

---

# 40. Output structure

Use something simple like:

```text
processed/
│
├── cleaned_raw/
│   ├── subject_001_clean_raw.fif
│   └── ...
│
├── epochs/
│   ├── subject_001_clean-epo.fif
│   └── ...
│
├── ica/
│   ├── subject_001_ica.fif
│   └── ...
│
├── qc/
│   ├── subject_001/
│   └── ...
│
├── metadata/
│   ├── preprocessing_qc.csv
│   ├── channel_availability.csv
│   └── common_channels.json
│
└── logs/
```

Never overwrite the original data.

---

# 41. Logging

Use Python's standard:

```python
import logging
```

instead of relying entirely on `print()`.

A log for one participant might look like:

```text
Loading subject_001
Group: Parkinson
Sampling frequency: 500 Hz
EEG channels: 64

Applying 1–100 Hz filter and 60 Hz notch
Resampling to 250 Hz
Applying pre-ICA common-average reference

Bad channels:
Fp1

Fitting ICA
ICA components: 35

ICA components removed:
IC000 - blink
IC003 - ECG

Interpolated:
Fp1

Final reference:
average

Initial epochs:
120

Rejected:
8

Retained:
112
```

---

# 42. Reproducibility

Use fixed random states where relevant.

Example:

```python
RANDOM_STATE = 42
```

ICA should therefore use:

```python
ICA(
    ...,
    random_state=42
)
```

---

# 43. Do not hide decisions

The pipeline should never silently:

* delete channels
* remove ICA components
* reject epochs
* change references
* alter filter frequencies

Every important decision must be:

1. logged
2. saved in QC metadata
3. visualized when appropriate

---

# 44. Do not modify data destructively

Use copies.

For example:

```python
raw_original = raw.copy()

raw_filtered = raw.copy().filter(
    l_freq=1.0,
    h_freq=50.0
)

cleaned = raw_filtered.copy()
```

Avoid repeatedly modifying the only version of the data.

---

# 45. Scientific constraints

Because the eventual goal is periodic/aperiodic spectral analysis:

Do not:

* filter above 1 Hz on the final data
* low-pass below 100 Hz
* normalize epochs independently
* baseline-correct resting EEG
* aggressively detrend the data
* automatically remove unusual ICA components
* remove oscillations simply because they appear prominent
* apply additional unnecessary notch filters
* perform spectral flattening
* run `specparam` during preprocessing

The final cleaned EEG should remain:

```text
1–100 Hz at 250 Hz sampling, with a 60 Hz notch
```

---

# 46. Implementation strategy

Do not immediately run the entire dataset.

Proceed in this order.

### Step 1

Inspect all files and report the dataset structure.

### Step 2

Compare channel lists and metadata across participants.

### Step 3

Implement loading and basic plotting.

### Step 4

Run preprocessing on:

```text
1 Parkinson participant
1 Control participant
```

### Step 5

Generate the complete QC plots for those two subjects.

### Step 6

Inspect the results for obvious problems.

### Step 7

Only after that, prepare batch processing for all subjects.

---

# 47. Expected pipeline functions

Keep the implementation approximately this simple:

```python
def load_subject(...):
    ...

def standardize_channels(...):
    ...

def filter_eeg(...):
    ...

def detect_bad_channels(...):
    ...

def annotate_artifacts(...):
    ...

def fit_ica(...):
    ...

def identify_ica_artifacts(...):
    ...

def apply_ica(...):
    ...

def interpolate_bad_channels(...):
    ...

def rereference(...):
    ...

def create_epochs(...):
    ...

def reject_bad_epochs(...):
    ...

def save_cleaned_data(...):
    ...

def save_qc_metadata(...):
    ...
```

Do not force everything into classes.

If classes do not clearly simplify the implementation, use functions.

---

# 48. Example high-level workflow

The final subject pipeline should be easy to read.

Something conceptually similar to:

```python
raw = load_subject(subject_path)

plot_raw(raw)

raw_filtered = filter_eeg(raw, config)
plot_filtered(raw_filtered)

bad_channels = detect_bad_channels(raw_filtered, config)
plot_bad_channels(raw_filtered, bad_channels)

annotated = annotate_artifacts(raw_filtered, config)

ica = fit_ica(annotated, config)
plot_ica(ica, annotated)

components_to_remove = identify_ica_artifacts(
    ica,
    annotated,
    config
)

cleaned = apply_ica(
    annotated,
    ica,
    components_to_remove
)

plot_before_after_ica(
    annotated,
    cleaned
)

cleaned = interpolate_bad_channels(cleaned)

cleaned = rereference(cleaned)

plot_raw_vs_cleaned(
    raw,
    cleaned
)

epochs = create_epochs(cleaned, config)

epochs = reject_bad_epochs(
    epochs,
    config
)

plot_final_psd(epochs)

save_outputs(...)
```

The exact API can differ, but the final code should remain approximately this easy to follow.

---

# 49. Final deliverables

Produce:

1. Dataset inspection script.
2. Dataset/channel summary.
3. `config.yaml`.
4. Simple preprocessing functions.
5. ICA functions.
6. QC plotting functions.
7. Single-subject runner.
8. Batch runner.
9. QC figures for one Parkinson and one Control participant.
10. Subject-level QC CSV.
11. Channel-availability CSV.
12. Saved cleaned continuous EEG.
13. Saved cleaned epochs.
14. README explaining how to run everything.

---

# Most important principles

Keep these priorities throughout the implementation:

```text
1. Scientific validity
2. Preserve the spectral properties of the EEG
3. Make every cleaning decision visible
4. Keep all subject/channel metadata
5. Never confuse missing channels with bad channels
6. Plot every important processing stage
7. Keep the code simple
8. Use SOLID pragmatically
9. Follow KISS
10. Avoid over-engineering
```

The pipeline should be easy enough that another scientist can open the code, follow the preprocessing sequence from top to bottom, and understand exactly what happens to the EEG at every stage.
