# Oscillatory-bout and cycle-by-cycle analysis

This standalone downstream pipeline implements [`ScaleFree.md`](../ScaleFree.md)
with MNE-Python, `specparam`, the eBOSC Morlet and detection definitions, and
`bycycle`. It never modifies preprocessing results.

## Scientific sequence

The input is the set of accepted four-second cleaned EEG epochs. Using accepted
epochs instead of reconstructing the cleaned raw stream prevents residual
epoch-level rejections from re-entering the analysis. PSD periodograms are
pooled across epochs, but wavelet transforms, bout detection, and cycle
extraction are performed independently within each epoch. Nothing can cross a
rejected gap or artificial epoch join.

Before calculation, the pipeline finds the electrode intersection across all
analyzed subjects. The full cohort currently has 60 shared electrodes; no
union-only electrode contributes to a table, summary, statistic, or figure.

### 1. Spectral parameterization

For each subject/shared electrode, accepted epochs are concatenated in stored
order and passed to one Welch PSD calculation. Non-overlapping four-second Hann
windows produce a 0.25 Hz grid. The linear 1–40 Hz PSD is fitted with
`specparam.SpectralModel` in fixed aperiodic mode.

Saved broadband parameters are:

- aperiodic offset and exponent;
- model R² and mean absolute error;
- number of fitted peaks.

The highest fitted peak in each theta (4–7 Hz), alpha (8–13 Hz), low-beta
(13–20 Hz), and high-beta (20–30 Hz) band supplies center frequency, power, and
bandwidth. A `peak_present` indicator distinguishes a missing peak from a
numerical value.

### 2. Aperiodic-relative eBOSC bouts

The time-frequency transform exactly reproduces the Morlet definition and
full-convolution crop in `ebosc.BOSC.BOSC_tf`, vectorized across epochs so it is
practical and boundary-safe. A test compares its output directly with the
installed eBOSC function.

The power threshold is explicitly based on the `specparam` aperiodic curve:

1. Compute mean eBOSC wavelet power at every analyzed frequency.
2. Map the fitted aperiodic PSD into wavelet-power units using the ratio between
   mean wavelet power and the full fitted `specparam` spectrum.
3. Apply the configured BOSC/eBOSC chi-square percentile to that mapped
   aperiodic background.
4. Require power to remain above threshold for at least three cycles at each
   frequency.
5. Collapse contiguous detected time-frequency samples into band bouts.

The first and last 0.75 seconds of every four-second epoch are excluded from
detection to protect against Morlet edge effects. Occupancy and bouts per minute
use only the remaining valid samples in their denominators.

Per subject/electrode/band outputs include occupancy (Pepisode), bouts per
minute, mean and median duration, cycles per bout, within-epoch inter-bout
interval, wavelet power and amplitude, and threshold ratio (SNR).

### 3. Cycle-by-cycle characterization

`bycycle.compute_shape_features` identifies trough-to-trough cycles inside each
accepted epoch. A cycle is retained only when at least 50% of its samples
overlap an eBOSC band-bout mask. The pipeline records the original bycycle
features and summarizes:

- voltage and analytic-band amplitude;
- period in seconds and frequency in hertz;
- amplitude and period standard deviation and coefficient of variation;
- rise/decay symmetry;
- peak/trough symmetry.

### 4. Subject-level PD vs Control comparisons

Electrode values are averaged first, producing one value per subject for every
broadband or band-resolved feature. Only those subject-level rows enter group
comparisons. Outputs include descriptive summaries, Welch independent-samples
t tests, Mann–Whitney tests, PD-minus-Control Hedges g, and Benjamini–Hochberg
correction across the complete reported Welch-test family. These tests are
exploratory; they do not compensate for clinical covariates not available in
the dataset.

## Run

Install or update the complete environment:

```bash
bash scripts/create_conda_environment.sh --env MNE_Roman --run-tests
```

Run the full cohort and all shared electrodes:

```bash
bash scale_free_analysis/run_scale_free_analysis.sh --overwrite
```

Run a small development pilot without changing the configured output:

```bash
bash scale_free_analysis/run_scale_free_analysis.sh \
  --subjects sub-001 sub-101 \
  --channels Cz Fz \
  --output-dir /tmp/scale-free-pilot \
  --overwrite
```

The full run is compute-intensive because wavelet and bycycle calculations are
performed independently for every accepted epoch, shared electrode, and band.
The progress bar reports completed subject/electrode pairs.

## Outputs

```text
scale_free_analysis/processed/
├── manifest.json
├── scale_free_analysis.log
├── metrics/
│   ├── analyzed_inputs.csv
│   ├── electrode_sets.json
│   ├── electrode_aperiodic_metrics.csv
│   ├── electrode_band_metrics.csv
│   ├── subject_aperiodic_metrics.csv
│   ├── subject_band_metrics.csv
│   ├── group_aperiodic_summary.csv
│   ├── group_band_summary.csv
│   └── pd_control_comparisons.csv
├── intermediate/
│   ├── spectra/sub-*_specparam_spectra.npz
│   ├── thresholds/sub-*_ebosc_thresholds.csv.gz
│   ├── episodes/sub-*_bout_episodes.csv.gz
│   └── cycles/sub-*_bycycle_cycles.csv.gz
└── figures/
    ├── examples/
    │   ├── specparam_decomposition.png
    │   ├── detected_bout_and_time_frequency.png
    │   └── bycycle_waveform_landmarks.png
    ├── group_comparisons/*.png
    └── topomaps/*.png
```

The compressed intermediate files preserve individual bout and cycle rows, the
frequency-specific aperiodic background and threshold, and observed/fitted
spectral curves. Every stage can therefore be inspected without reverse
engineering a final aggregate.

## Validation

Run the dedicated tests:

```bash
conda run -n MNE_Roman python -m unittest discover \
  -s tests -p 'test_scale_free_analysis.py' -v
```

The tests cover configuration, synthetic aperiodic/peak recovery, exact eBOSC
wavelet equivalence, duration and edge rules, bout summaries, and bycycle units
and selection.
