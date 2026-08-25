### Python packages and environment

Before implementing the analysis, check whether the required packages are already installed in the current Python environment. Do not reinstall packages unnecessarily.

The main required packages are:

```bash
pip install numpy scipy pandas matplotlib mne tensorpac statsmodels
```

Use:

* **mne** — loading and handling EEG data, channels, filtering, epochs, and EEG metadata.
* **tensorpac** — primary package for phase-amplitude coupling analysis, including PAC metrics, comodulograms, and surrogate analyses.
* **numpy** — numerical calculations and array manipulation.
* **scipy** — signal processing, Hilbert transforms, filtering, and statistical utilities.
* **pandas** — subject-level results tables and metadata.
* **matplotlib** — all main figures and diagnostic plots.
* **statsmodels** — regression, group statistics, and mixed-effects models.

Also install:

```bash
pip install seaborn pingouin
```

These are optional:

* **seaborn** — convenient statistical visualization.
* **pingouin** — effect sizes, confidence intervals, correlations, and additional statistical tests.

For oscillatory-bout detection, use the **existing fBOSC implementation/environment already used by the project**. Do not replace or reimplement fBOSC unless necessary.

If waveform-shape analysis is implemented, consider:

```bash
pip install bycycle neurodsp
```

Use:

* **bycycle** — cycle-by-cycle waveform analysis, including rise-decay and peak-trough symmetry.
* **neurodsp** — filtering and oscillatory signal-processing utilities used by bycycle.

Therefore, a complete environment can be installed with:

```bash
pip install \
    numpy \
    scipy \
    pandas \
    matplotlib \
    mne \
    tensorpac \
    statsmodels \
    seaborn \
    pingouin \
    bycycle \
    neurodsp
```

However, keep dependencies minimal. The core PAC analysis should require only:

```text
numpy
scipy
pandas
matplotlib
mne
tensorpac
statsmodels
```

Treat `bycycle`, `neurodsp`, `seaborn`, and `pingouin` as optional dependencies.

### Package usage

Prefer established library implementations over custom implementations.

In particular:

```python
import mne
from tensorpac import Pac
```

should form the basis of the PAC analysis.

Use Tensorpac for:

* Tort's Modulation Index;
* phase × amplitude comodulograms;
* surrogate PAC;
* normalized PAC where appropriate.

Do **not** write a custom PAC implementation unless Tensorpac cannot perform a required part of the analysis.

Use MNE for:

* EEG loading;
* channel selection;
* sampling-frequency handling;
* continuous signal management;
* filtering where appropriate.

Use SciPy only where lower-level signal operations are needed.

For the optional waveform-shape control:

```python
from bycycle.features import compute_features
```

can be used to quantify properties of beta cycles such as:

* rise-decay symmetry;
* peak-trough symmetry;
* amplitude;
* period;
* cycle consistency.

### Reproducibility

At the beginning of the analysis, save the Python and package versions used.

For example, create:

```text
results/environment.txt
```

containing at least:

```text
Python version
NumPy version
SciPy version
MNE version
Tensorpac version
Statsmodels version
Bycycle version, if used
```

Use a fixed random seed for surrogate generation and subsampling.

Do not change package versions automatically if the analysis already runs correctly in the existing project environment.

You are working on a resting-state, eyes-open EEG dataset containing Parkinson’s disease (PD) patients and healthy controls.

Implement a **beta-phase / low-gamma-amplitude phase-amplitude coupling (PAC) analysis** in Python.

The main goal is to determine whether beta–low-gamma PAC differs between PD and Control subjects, and whether any difference is specifically associated with genuine transient beta oscillations rather than simply differences in beta power or nonsinusoidal waveform shape.

Use **MNE-Python** for EEG handling and preferably **Tensorpac** or another well-established PAC implementation.

Keep the code simple, modular, well documented, and easy to inspect. Follow **KISS and SOLID principles**.

### Data assumptions

* EEG has already been cleaned with the existing preprocessing pipeline.
* Signals are filtered approximately from **1–50 Hz**.
* Preserve subject ID, group label, channel information, and relevant metadata.
* Apply exactly the same analysis pipeline to PD and Control subjects.
* Do not modify the existing preprocessing code.
* Add PAC analysis as a separate module or pipeline.

### Channels

Focus primarily on sensorimotor electrodes:

* C3
* Cz
* C4

If available, neighboring central electrodes can also be analyzed.

Do not average electrodes before computing PAC.

### PAC frequency ranges

Primary coupling of interest:

* Phase: **beta, approximately 13–30 Hz**
* Amplitude: **low gamma, approximately 30–50 Hz**

Also calculate a PAC comodulogram across multiple phase and amplitude frequencies.

For example:

```text
Phase frequencies:
10–30 Hz

Amplitude frequencies:
30–50 Hz
```

Use appropriate narrow-band filters around each center frequency rather than filtering the entire 13–30 Hz beta band as one signal for the comodulogram.

### PAC calculation

Use **Tort's Modulation Index** as the primary PAC metric.

For each subject and channel:

1. Filter the continuous signal around each phase frequency.
2. Calculate instantaneous phase using the Hilbert transform.
3. Filter around each amplitude frequency.
4. Calculate the amplitude envelope using the Hilbert transform.
5. Calculate PAC.
6. Store the complete phase × amplitude PAC matrix.

Also calculate a summary beta–low-gamma PAC value for each subject/channel.

### Surrogate normalization

Do not rely only on raw PAC.

Generate surrogate PAC distributions by disrupting the temporal relationship between beta phase and gamma amplitude.

Prefer circular temporal shifts of the gamma amplitude envelope.

Use approximately:

```text
200–500 surrogates
```

per subject/channel.

Use a fixed random seed.

Calculate:

```text
PAC_z =
(PAC_observed - mean(PAC_surrogates))
/
std(PAC_surrogates)
```

Save:

* raw PAC
* surrogate mean
* surrogate standard deviation
* PAC z-score
* empirical significance

Use surrogate-normalized PAC as the main quantity for group comparisons.

### fBOSC integration

Another part of the analysis pipeline identifies oscillatory bouts using **fBOSC**.

Assume that a beta-bout boolean mask is available:

```python
beta_bout_mask
```

where:

```text
True  = sample belongs to a detected beta oscillation
False = sample does not belong to a beta oscillation
```

Calculate at least two PAC measurements.

#### 1. Conventional PAC

```text
PAC_all
```

Calculate PAC using the complete clean resting-state EEG.

#### 2. Beta-bout PAC

```text
PAC_beta_bouts
```

Calculate PAC using only samples belonging to genuine beta oscillatory bouts detected by fBOSC.

Important:

**Filtering and Hilbert transforms must be calculated on the continuous EEG first.**

Do not concatenate beta bouts and then filter them.

Instead:

```python
phase = calculate_beta_phase(continuous_signal)
amplitude = calculate_gamma_amplitude(continuous_signal)

phase_bouts = phase[beta_bout_mask]
amplitude_bouts = amplitude[beta_bout_mask]
```

Then calculate PAC using the selected samples.

### Control for amount of beta activity

PD and Control subjects may have different amounts of beta-bout data.

This could bias PAC estimates.

For each subject calculate:

* beta-bout probability
* total beta-bout duration
* mean beta-bout duration
* number of beta bouts

When comparing PAC_beta_bouts between groups, control for unequal numbers of available samples.

Prefer a repeated subsampling procedure.

For example:

1. Determine a common number of beta-bout samples.
2. Randomly sample this number of beta-bout samples from each subject.
3. Calculate PAC.
4. Repeat many times.
5. Average the resulting PAC estimates.

This prevents subjects with more beta activity from automatically having more stable PAC estimates.

### Control for beta power

For each subject/channel calculate beta power.

Test whether group differences in PAC remain after accounting for beta power.

For example:

```text
PAC ~ Group + BetaPower
```

Also examine:

```text
PAC vs BetaPower
```

within and across groups.

The goal is to distinguish:

```text
PD
→ increased beta power
→ apparent increased PAC
```

from a PAC alteration that cannot be explained simply by beta amplitude.

### Check for spurious PAC due to waveform shape

PAC can appear when beta oscillations are nonsinusoidal.

For example, a 20-Hz nonsinusoidal beta oscillation can generate a harmonic around 40 Hz, producing apparent:

```text
20-Hz phase
×
40-Hz amplitude
```

coupling.

Therefore include basic diagnostics for nonsinusoidal waveform effects.

At minimum:

* identify whether PAC maxima occur at harmonic relationships;
* inspect beta waveform shape in representative subjects;
* compare PAC with beta power;
* compare PAC during detected beta bouts versus the entire signal.

If practical, calculate simple waveform metrics such as:

* peak sharpness
* trough sharpness
* peak–trough asymmetry
* rise–decay asymmetry
* second-harmonic power

Keep waveform analyses separate from the main PAC pipeline.

Do not interpret statistically significant PAC automatically as evidence for interaction between two independent neural oscillators.

### Subject-level output

Create a dataframe containing at least:

```text
subject_id
group
channel
beta_power
beta_bout_probability
beta_bout_duration
beta_bout_count
pac_all
pac_all_z
pac_beta_bouts
pac_beta_bouts_z
```

Save one row per subject/channel.

### Statistical comparisons

Primary comparisons:

```text
PD vs Control
```

for:

```text
PAC_all_z
PAC_beta_bouts_z
```

Also test:

```text
PAC ~ Group + BetaPower
```

If multiple electrodes are included, use a mixed-effects model if appropriate, with subject as a random effect.

For example:

```text
PAC ~ Group + BetaPower + Channel + (1 | Subject)
```

Do not treat electrodes from the same subject as independent subjects.

Report:

* effect size
* confidence interval
* p-value

Correct for multiple comparisons when testing complete PAC comodulograms.

### Important interaction analysis

Explicitly test whether the PD-Control difference depends on whether PAC is measured during beta bouts.

Conceptually test:

```text
PAC ~ Group * PAC_condition
```

where:

```text
PAC_condition = all_signal or beta_bouts
```

The scientifically interesting result would be whether the PD-Control difference becomes stronger, weaker, or disappears when PAC is restricted to genuine beta oscillations.

### Required plots

Generate and save:

1. Example clean EEG from C3/Cz/C4.
2. Beta-filtered signal with beta phase.
3. Low-gamma-filtered signal with gamma envelope.
4. Beta-bout detections over the EEG signal.
5. Phase-binned gamma-amplitude distribution.
6. PAC comodulogram for representative Control subject.
7. PAC comodulogram for representative PD subject.
8. Mean Control PAC comodulogram.
9. Mean PD PAC comodulogram.
10. PD − Control PAC difference map.
11. Example surrogate PAC distribution.
12. PAC_all_z by group.
13. PAC_beta_bouts_z by group.
14. PAC_all_z versus PAC_beta_bouts_z for each subject.
15. PAC versus beta power.
16. PAC versus beta-bout probability.
17. PAC versus mean beta-bout duration.

For group plots, show individual subjects whenever practical rather than only bars.

### Validation

Explicitly check:

* sampling rate;
* sufficient data length;
* filter edge artifacts;
* NaNs;
* missing channels;
* minimum number of beta bouts;
* number of beta-bout samples;
* recording duration;
* unequal numbers of valid samples across subjects;
* stability of PAC estimates across different amounts of data.

Subjects without enough beta-bout data should be clearly flagged rather than silently included.

### Code organization

Keep the implementation modular, for example:

```text
pac_analysis/
    config.py
    filtering.py
    pac.py
    surrogates.py
    bout_pac.py
    waveform.py
    statistics.py
    plotting.py
    run_pac_analysis.py
```

Avoid unnecessary classes or complicated abstractions.

Configuration parameters should be centralized, including:

```text
phase frequencies
amplitude frequencies
filter bandwidths
channels
number of surrogates
minimum beta-bout samples
number of subsampling repetitions
random seed
```

### Outputs

Save:

```text
results/
    subject_level_pac.csv
    pac_matrices/
    surrogate_results/
    figures/
    statistics/
```

Save intermediate results so PAC does not need to be recomputed just to regenerate figures.

### Main scientific question

The analysis should answer:

> **Is beta–low-gamma PAC altered in Parkinson's disease, and is the alteration specifically associated with genuine transient beta oscillatory bouts rather than simply differences in beta power or nonsinusoidal beta waveform shape?**

Prioritize interpretability, robustness, and validation over classification performance.
