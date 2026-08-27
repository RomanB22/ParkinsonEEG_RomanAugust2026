# Oscillatory-bout and cycle-by-cycle analysis

This standalone downstream pipeline implements [`docs/analyses/scale_free_background.md`](../docs/analyses/scale_free_background.md)
with MNE-Python, `specparam`, the eBOSC Morlet and detection definitions, and
`bycycle`. It never modifies preprocessing results.

Formal PD-versus-Control tests cover aperiodic, canonical periodic, bout, and
cycle quantities. They use subject-level shared-electrode aggregates; the full
cohort adjusts for age and sex and the matched cohort preserves its pairs.
Electrode-wise results are exploratory localization with strict domain-wide
FDR. See
[`../docs/group_statistics.md`](../docs/group_statistics.md) for the common inference
policy.

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
windows produce a 0.25 Hz grid. The PSD covers 1–50 Hz, while both
`specparam.SpectralModel` candidates are fitted over 4–50 Hz. One candidate
uses fixed mode and the other uses knee mode. Both use the paper-aligned
settings: peak widths 1–12 Hz, at most eight
peaks, minimum peak height 0, and peak threshold 2.

In the central pipeline, this identical Welch array is loaded losslessly from
`outputs/full/psd/metrics/subject_electrode_psd.npz` rather than
calculated twice. Subject IDs, electrode order, array shape, finite values, and
the exact 1–50 Hz grid are validated before reuse. A focused standalone run
falls back to the same local Welch calculation when that cache is absent or
incompatible, and records the decision in its manifest.

BIC compares the two candidates for each subject/electrode while penalizing the
knee model's extra parameter. It is calculated on log10-power residuals as
`n × ln(RSS/n) + k × ln(n)`, where `k` includes the aperiodic parameters and
three parameters per fitted Gaussian peak. This avoids selecting knee merely
because its unpenalized R² cannot be worse. Fixed wins ties. Knee frequency is calculated as
`knee^(1/exponent)`; knee frequencies more than 2 SD from that subject's mean
across shared electrodes, non-finite knees, and knees outside 4–50 Hz are
ineligible, causing a transparent fallback to fixed. Both candidate fits,
selection reason, BIC values, and selected model are saved.

The meaning of the selected exponent depends on the selected model. In fixed
mode it is the single slope across 4–50 Hz; in knee mode it is the asymptotic
slope above the bend. The selected exponent is appropriate for constructing
the selected aperiodic background used by eBOSC, but pooled group analyses of
that value must retain the model label and be interpreted cautiously. The
candidate-specific fixed and knee exponents are preserved in the model
comparison tables for transparent sensitivity analyses.

Saved broadband parameters are:

- aperiodic offset and exponent;
- model R² and mean absolute error;
- number of fitted peaks.

Periodic spectral elevations and overlapping higher-frequency shoulders are
represented by the fitted periodic Gaussians. A benchmark allowing peak widths
up to 20 Hz did not materially improve fit quality or exponent stability, so
the primary peak settings remain unchanged rather than being selected after
inspection.

Every primary fit receives formal, configurable QC. A fit passes when R² is at
least 0.90, log-power MAE is at most 0.15, its largest absolute signed residual
is at most 0.75 log10 units, and its exponent lies in 0–3. No fit is silently
deleted: all fits remain in the tables and gallery with a pass/fail flag and
failure reasons. A subject-level QC-qualified exponent is reported separately
only when at least 80% of that subject's 60 shared electrodes pass.

Fit QC is also propagated into a formal downstream sensitivity analysis after
the within-bout ordinal pipeline has completed. It retains only passing
electrodes and requires at least 48/60 passing fits per subject. The original
all-electrode bout, cycle, and within-bout ordinal outputs are preserved; the
QC-qualified versions are written alongside them.

No alternative frequency-range fit is run: every fixed and knee candidate uses
4–50 Hz.

The highest fitted peak in each theta (4–7 Hz), alpha (8–13 Hz), low-beta
(13–20 Hz), and high-beta (20–30 Hz) band supplies center frequency, power,
and bandwidth. A `peak_present` indicator distinguishes a missing peak from a
numerical value.

For visual inspection, the pipeline writes one overview PNG per subject with
all shared-electrode decompositions in the same figure. The PNGs share one flat
folder and a root HTML index. Electrode titles include the selected mode,
exponent, R², and QC status. The separate detailed example also shows the
observed spectrum, full model, aperiodic and fitted periodic components, and
the signed observed-minus-full-model residual.
Observed power may lie below the aperiodic curve because that curve is a fitted
baseline, not a pointwise lower bound; negative residuals are shown explicitly.
These plots reuse the saved fitted curves and never refit specparam.

### 2. Aperiodic-relative eBOSC bouts

The time-frequency transform exactly reproduces the Morlet definition and
full-convolution crop in `ebosc.BOSC.BOSC_tf`, vectorized across epochs so it is
practical and boundary-safe. A test compares its output directly with the
installed eBOSC function.

The power threshold is explicitly based on the BIC-selected fixed or knee
`specparam` aperiodic curve for that subject and electrode:

1. Compute mean eBOSC wavelet power at every analyzed frequency.
2. Map the fitted aperiodic PSD into wavelet-power units using the ratio between
   mean wavelet power and the full fitted `specparam` spectrum.
3. Apply the configured BOSC/eBOSC chi-square percentile to that mapped
   aperiodic background.
4. Require power to remain above threshold for at least three cycles at each
   frequency.
5. Collapse contiguous detected time-frequency samples into band bouts.

Detection requires a technically successful fixed fit. A failed, invalid, or
outlier knee candidate falls back to fixed mode. Formal R², MAE, residual, and
exponent-range QC does **not** gate the primary detector: every technically
fitted electrode remains in the provenance analysis. After detection, the
separate fit-QC sensitivity removes failed-fit electrodes, re-aggregates bout
and within-bout ordinal quantities, and requires at least 80% passing
electrodes for a subject-level QC result. Thus the repository exposes both the
complete primary analysis and the conservative QC-qualified sensitivity.

The first and last 0.75 seconds of every four-second epoch are excluded from
detection to protect against Morlet edge effects. Occupancy and bouts per minute
use only the remaining valid samples in their denominators.

Per subject/electrode/band outputs include occupancy (Pepisode), bouts per
minute, mean and median duration, cycles per bout, within-epoch inter-bout
interval, wavelet power and amplitude, and threshold ratio (SNR).

Bout-detection QC is deliberately visible rather than reduced to a single
pass/fail label:

- only preprocessing-accepted epochs and the 60 cohort-shared electrodes enter
  detection;
- exact edge and minimum-duration rules are tested and every retained interval
  keeps its epoch, sample bounds, duration, peak frequency, power, and threshold
  ratio in the compressed episode tables;
- the detection example overlays the aperiodic-relative threshold and detected
  time-frequency mask;
- counts, occupancy, duration, amplitude, threshold ratio, and missing-bout
  rates are summarized in the quality figures and metrics;
- primary all-fit outputs are accompanied by a fit-QC sensitivity that removes
  failed subject/electrode aperiodic models.

The typical-bout gallery adds a coverage check for every group, band, and
electrode. It reports the eligible and contributing subject counts, total bout
counts, and mean/median bouts per subject. This makes sparse or unequal support
visible before interpreting a group-average curve.

### 3. Subject-balanced stereotypical bout representations

For a visually interpretable average bout, each accepted epoch is band-pass
filtered and converted to its complex Hilbert analytic signal. Each
subject/electrode/band signal is divided by its median Hilbert amplitude over
valid epoch interiors, and every detected bout is aligned to its temporal
midpoint in a ±0.5-second window. Each panel row contains:

- the normalized Hilbert amplitude envelope;
- circular mean Hilbert phase relative to the bout center, with dotted
  resultant length `R` showing phase consistency from 0 to 1;
- the real phase-aligned analytic signal, which provides an average oscillatory
  bout shape without cancellation from arbitrary absolute phase.

For the latter two quantities, every bout is rotated so its analytic phase is
zero at its center. Consequently, the phase and shape panels describe relative
within-bout evolution. They are not evidence of absolute phase locking to an
external event, and the phase-aligned shape is not an event-related potential.

Aggregation is hierarchical: bouts are averaged within each
subject/electrode/band first, and subjects are then averaged with equal weight.
Envelope and phase-aligned-shape shading is a pointwise 95% Student-t confidence
interval across subject means, not across bouts. Phase is circularly averaged
and accompanied by `R` rather than a linear confidence band. The grand-average
figure first averages available electrodes within each subject. A parallel
gallery retains only fit-QC electrodes from subjects meeting the 48/60 fit
criterion. These are descriptive waveform summaries; their pointwise intervals
are not multiplicity-corrected group tests.

### 4. Cycle-by-cycle characterization

`bycycle.compute_shape_features` identifies trough-to-trough cycles inside each
accepted epoch. A cycle is retained only when at least 50% of its samples
overlap an eBOSC band-bout mask. The pipeline records the original bycycle
features and summarizes:

- voltage and analytic-band amplitude;
- period in seconds and frequency in hertz;
- amplitude and period standard deviation and coefficient of variation;
- rise/decay symmetry;
- peak/trough symmetry.

All cycle summaries are retained in the electrode and subject metric tables.
The much larger per-cycle provenance tables are disabled by default to reduce
disk writes and cache size. Set `cache.save_raw_cycle_tables` to `true` in the
configuration only when individual cycle rows are needed for an audit.

### 5. Subject-level PD vs Control comparisons

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
bash scripts/create_conda_environment.sh --env MNE_August2026 --run-tests
```

Run the full cohort and all shared electrodes:

```bash
bash src/analyses/scale_free/run_scale_free_analysis.sh --overwrite
```

If the scale-free analysis already exists, generate or resume the flat gallery
containing exactly one all-electrode figure per subject without rerunning eBOSC
or bycycle:

```bash
bash src/analyses/scale_free/generate_specparam_figures.sh
```

Use `--overwrite` to regenerate existing images. The defaults use four worker
processes and 100 DPI; both can be overridden with `--workers` and `--dpi`.
The legacy `--overwrite-subject-overviews` option remains as an alias for
regenerating these combined figures.

Fit QC and all four range-sensitivity analyses can be regenerated from the
saved spectra without rerunning eBOSC or bycycle:

```bash
bash src/analyses/scale_free/run_aperiodic_diagnostics.sh
bash src/analyses/scale_free/generate_specparam_figures.sh --overwrite
```

After both bout pipelines exist, propagate fit QC without repeating PSD fits,
wavelets, bout detection, or ordinal encoding:

```bash
bash src/analyses/scale_free/run_fit_qc_sensitivity.sh
```

This stage verifies that the independently repeated exponent and R² agree to
numerical precision before re-aggregation. The PD/Control comparison of fit
failure uses one failure fraction per subject and includes an age/sex-adjusted
HC3-robust model; electrodes are not treated as independent observations.

Generate the subject-balanced Control-versus-PD typical-bout gallery from the
saved episode intervals (without repeating bout detection):

```bash
bash src/analyses/scale_free/generate_typical_bouts.sh
```

Run a small development pilot without changing the configured output:

```bash
bash src/analyses/scale_free/run_scale_free_analysis.sh \
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
outputs/full/scale_free/
├── manifest.json
├── scale_free_analysis.log
├── metrics/
│   ├── analyzed_inputs.csv
│   ├── electrode_sets.json
│   ├── electrode_aperiodic_metrics.csv
│   ├── electrode_aperiodic_model_comparison.csv.gz
│   ├── subject_aperiodic_model_comparison.csv
│   ├── subject_aperiodic_qc_metrics.csv
│   ├── specparam_fit_qc_summary.csv
│   ├── subject_specparam_fit_failures.csv
│   ├── specparam_fit_failure_group_comparison.csv
│   ├── electrode_band_metrics.csv
│   ├── electrode_band_metrics_fit_qc.csv
│   ├── specparam_figure_index.csv
│   ├── subject_aperiodic_metrics.csv
│   ├── subject_band_metrics.csv
│   ├── subject_band_metrics_fit_qc.csv
│   ├── bout_property_fit_qc_sensitivity.csv
│   ├── typical_bout_coverage.csv
│   ├── typical_bout_group_coverage.csv
│   ├── group_aperiodic_summary.csv
│   ├── group_band_summary.csv
│   └── pd_control_comparisons.csv
├── intermediate/
│   ├── spectra/sub-*_specparam_spectra.npz
│   ├── thresholds/sub-*_ebosc_thresholds.csv.gz
│   ├── episodes/sub-*_bout_episodes.csv.gz
│   └── cycles/sub-*_bycycle_cycles.csv.gz  # optional; disabled by default
└── figures/
    ├── examples/
    │   ├── specparam_decomposition.png
    │   ├── detected_bout_and_time_frequency.png
    │   └── bycycle_waveform_landmarks.png
    ├── group_comparisons/*.png
    ├── aperiodic_diagnostics/
    │   ├── fit_qc_dashboard.png
    │   ├── fixed_vs_knee_model_selection.png
    │   ├── group_median_decomposition_and_residuals.png
    │   └── fit_failures_by_group.png
    ├── fit_qc_sensitivity/bout_properties_all_vs_fit_qc.png
    ├── typical_bouts/
    │   ├── index.html
    │   ├── grand_average_all_subjects.png
    │   ├── grand_average_fit_qc.png
    │   ├── bout_detection_subject_coverage.png
    │   ├── bout_count_qc.png
    │   ├── all_subjects/<electrode>.png
    │   └── fit_qc/<electrode>.png
    ├── topomaps/*.png
    └── specparam_decomposition/
        ├── index.html
        ├── figure_index.csv
        └── sub-*_<{PD,Control}>_all_electrodes.png
```

The single flat gallery contains one overview figure per subject. Each overview
contains every shared electrode on the same linear-frequency/log-power canvas:
observed PSD in black, the full specparam model in blue, and the aperiodic
component in orange. The fixed candidate is dashed gray and the knee candidate
is dotted magenta; each electrode title identifies the selected mode. Red
electrode labels indicate formal QC failures. No
individual-electrode PNGs or per-subject folders are generated; poor fits stay
visible in the combined figure rather than being hidden.

The compressed intermediate files preserve individual bout rows, the
frequency-specific aperiodic background and threshold, and observed/fitted
spectral curves. Optional raw cycle rows can be enabled for a cycle-level audit;
the default cache retains their complete summaries without the large raw table.

## Validation

Run the dedicated tests:

```bash
conda run -n MNE_August2026 python -m unittest discover \
  -s tests -p 'test_scale_free_analysis.py' -v
```

The tests cover fixed and knee recovery, BIC selection, within-subject knee
outlier handling, exact eBOSC wavelet equivalence, duration and edge rules,
bout summaries, and bycycle units and selection.
