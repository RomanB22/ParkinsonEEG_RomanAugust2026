Implement a **Python pipeline to characterize oscillatory activity in cleaned resting-state EEG** and compare **Parkinson’s disease (PD) vs Control** subjects.

Use **MNE-Python, specparam, eBOSC, and bycycle** where appropriate.

### 1. Spectral decomposition — `specparam`

For each subject and channel:

* Compute PSD from the cleaned continuous EEG.
* Use `specparam` to separate **aperiodic (1/f)** and **periodic** components.
* Extract:

  * aperiodic exponent and offset
  * oscillatory peak frequency
  * peak power
  * peak bandwidth
* Analyze theta (4–7 Hz), alpha (8–13 Hz), low-beta (13–20 Hz), and high-beta (20–30 Hz).

### 2. Oscillatory bout detection — `eBOSC`

Detect transient oscillatory episodes rather than assuming oscillations are continuously present.

Importantly, define oscillatory power relative to the **aperiodic background estimated with `specparam`**, so differences in 1/f activity between PD and Control are not incorrectly interpreted as differences in oscillatory activity.

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
* example EEG segments with detected bouts highlighted
* time-frequency representation with bout detection
* bout duration and occupancy distributions
* cycle-by-cycle examples
* scalp topographies
* PD vs Control group comparisons

Keep the implementation **simple, modular, well documented, and easy to validate (KISS/SOLID)**. Save intermediate results so every processing and analysis step can be independently inspected.
