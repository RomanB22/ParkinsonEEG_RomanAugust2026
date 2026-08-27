Implement a **Python pipeline to characterize oscillatory activity in cleaned resting-state EEG** and compare **Parkinson’s disease (PD) vs Control** subjects.

Use **MNE-Python, specparam, eBOSC, and bycycle** where appropriate.

### 1. Spectral decomposition — `specparam`

For each subject and channel:

* Compute PSD from the cleaned continuous EEG.
* Use `specparam` to separate **aperiodic (1/f)** and **periodic** components.
* Audit every fit with R², log-error, exponent-range, and signed-residual QC;
  retain failed fits for inspection and report QC-qualified summaries separately.
* Fit both fixed and knee aperiodic models over 4–50 Hz using peak widths of
  1–12 Hz, at most eight peaks, minimum peak height 0, and peak threshold 2.
* Compare candidates with BIC, prefer fixed on ties, and exclude knee
  frequencies more than 2 within-subject SD from the subject mean before model
  selection. Use 4–50 Hz for every aperiodic fit.
* Preserve the selected mode with every exponent: a fixed exponent is the
  single 4–50 Hz slope, whereas a knee exponent is the asymptotic slope above
  the bend.
* Extract:

  * aperiodic exponent and offset
  * oscillatory peak frequency
  * peak power
  * peak bandwidth
* Analyze theta (4–7 Hz), alpha (8–13 Hz), low-beta (13–20 Hz), and high-beta (20–30 Hz).

### 2. Oscillatory bout detection — `eBOSC`

Detect transient oscillatory episodes rather than assuming oscillations are continuously present.

Importantly, define oscillatory power relative to the **aperiodic background estimated with `specparam`**. Use the BIC-selected fixed or knee background for each subject and electrode, so differences in 1/f activity between PD and Control are not incorrectly interpreted as differences in oscillatory activity.

Use the 95th-percentile BOSC/eBOSC chi-square power threshold and require at
least three cycles above threshold at each frequency. Exclude 0.75 seconds at
both edges of every accepted four-second epoch, never allow a detection to
cross an epoch boundary, and collapse contiguous qualifying samples from any
frequency inside a band into a band-level bout.

For each subject/channel/frequency band calculate:

* oscillatory occupancy / Pepisode
* number of bouts per minute
* bout duration
* number of cycles per bout
* inter-bout interval
* bout amplitude/power

### 3. Cycle-by-cycle analysis — `bycycle`

Apply `bycycle` to characterize the detected oscillatory periods.

Extract:

* cycle amplitude
* cycle frequency/period
* amplitude variability
* period variability
* rise/decay symmetry
* peak/trough symmetry

### 4. PD vs Control

Aggregate results at the **subject level** before performing group statistics.

Compare PD vs Control for:

* aperiodic features
* periodic spectral features
* oscillatory bout features
* cycle-level features

Preserve channel information to allow electrode-level and topographic comparisons.

### Visualization

Generate figures for every major analysis step, including:

* PSD and `specparam` decomposition
* Signed observed-minus-full-model residual and fit-QC status for every subject
  and electrode
* Cohort-level fit-QC and fixed-versus-knee selection figures
* example EEG segments with detected bouts highlighted
* time-frequency representation with bout detection
* bout duration and occupancy distributions
* cycle-by-cycle examples
* scalp topographies
* PD vs Control group comparisons
* subject-balanced stereotypical bout envelopes, circular relative Hilbert
  phase, phase consistency, and phase-aligned average shapes per band and
  electrode
* bout-detection coverage and bout-count QC for Control and PD, both before and
  after aperiodic-fit QC

Keep the implementation **simple, modular, well documented, and easy to validate (KISS/SOLID)**. Save intermediate results so every processing and analysis step can be independently inspected.
