# Group PSD and frequency-band topography analysis

This folder contains a standalone downstream spectral analysis of the accepted,
cleaned EEG epochs. It does not alter the dataset or preprocessing outputs.

## What is calculated

The pipeline first inventories every analyzed subject and restricts the run to
the electrode intersection shared by all of them. For each subject, accepted
four-second epochs are then loaded in their stored temporal order and
concatenated separately for those shared electrodes. The pipeline makes one
Welch call on that `electrodes × concatenated samples` array.
Welch uses non-overlapping four-second Hann windows, so a spectral window never
straddles an artificial join between epochs. At 250 Hz, the 1,000-sample window
produces a 0.25 Hz frequency grid. The default analysis retains 1–50 Hz.

Aggregation is deliberately hierarchical and subject-balanced:

1. Concatenate every accepted epoch in temporal order for each electrode.
2. Calculate one Welch PSD per subject/electrode. Welch averages the
   non-overlapping window periodograms in linear µV²/Hz; there is no
   median-across-epochs step.
3. For the whole-head subject curve, take the median across the electrodes
   present in every analyzed participant.
4. Take the median of those subject curves within PD and Control.
5. Bootstrap subjects within each group to obtain a pointwise 95% percentile
   confidence interval around the group median.
6. Convert the median and confidence bounds to dB only for plotting and the
   explicit dB columns. Averaging or taking medians in dB would answer a
   different question, so all aggregation remains in linear power.

The confidence intervals use 2,000 deterministic resamples by default. They
are pointwise descriptive intervals, not simultaneous confidence bands and not
hypothesis tests.

## Frequency bands

Band power is the trapezoidal integral of each concatenated subject/electrode
linear PSD. Relative band power divides that integral by the same subject and
electrode's total integrated power from 1–50 Hz:

`relative band power = band power / total 1–50 Hz power`

| Band | Limits |
|---|---:|
| Delta | 1–4 Hz |
| Theta | 4–8 Hz |
| Alpha | 8–13 Hz |
| Beta | 13–30 Hz |
| Low gamma | 30–50 Hz |

Endpoints are included. A shared boundary such as 4 Hz is a single point with
zero width; the adjacent trapezoids cover the intervals on either side without
double-counting an interval of power.

Group topomaps show the electrode-wise median across subjects in relative band
power, displayed as a percentage of total 1–50 Hz power. Normalization occurs
for every subject/electrode before the group median is calculated. They use the
same common electrodes used everywhere else in the analysis, so PD and Control
maps are directly comparable. For each band, both groups use the same color
limits. The CSV tables retain absolute band power alongside total and relative
power.

The violin figure first takes the median across the common electrodes for each
subject and band. Each violin therefore contains one independent observation
per subject, with the individual subject values shown as points. PD and Control
are displayed side by side for every configured frequency band.

Formal PD-versus-Control inference is saved at both the subject and electrode
levels. The primary subject test is age/sex-adjusted in the full cohort and
paired by demographic match in the matched cohort. Electrode tests are
exploratory spatial localization and use a strict domain-wide FDR flag. The
configured canonical bands all enter these tests. See
[`../GROUP_STATISTICS.md`](../GROUP_STATISTICS.md) for the complete policy.

## Run

From the repository root:

```bash
bash psd_analysis/run_psd_analysis.sh --overwrite
```

Selected participants can be used for a development run, but at least three per
included group are needed for the group statistics:

```bash
bash psd_analysis/run_psd_analysis.sh \
  --subjects sub-001 sub-002 sub-101 sub-102 \
  --overwrite
```

Configuration is in [`config.json`](config.json). The runner uses the existing
`MNE_August2026` environment and the same non-interactive cache policy as the
preprocessing workflow.

## Outputs

```text
psd_analysis/processed/
├── manifest.json
├── psd_analysis.log
├── metrics/
│   ├── analyzed_inputs.csv
│   ├── subject_electrode_psd.npz
│   ├── subject_global_psd.csv
│   ├── group_median_psd.csv
│   ├── subject_electrode_band_power.csv
│   ├── subject_band_power.csv
│   ├── group_electrode_band_power.csv
│   ├── group_subject_statistics.csv
│   ├── group_electrode_statistics.csv
│   └── electrode_sets.json
└── figures/
    ├── group_median_psd_with_ci.png
    ├── group_median_band_power_topomaps.png
    ├── group_relative_band_power_violins.png
    └── group_statistics/relative_band_power_group_statistics.png
```

`subject_electrode_psd.npz` is the compact lossless spectral array. It contains
subject IDs, group labels, the cohort-wide shared electrodes, frequencies, and
a dense `subjects × shared electrodes × frequencies` PSD array in µV²/Hz.
Source-layout union information is retained separately in
`electrode_sets.json`, but union-only electrodes do not enter PSD or band-power
calculations.

The CSV tables retain 17 significant digits. `manifest.json` records the full
configuration, software versions, aggregation sequence, confidence-interval
definition, electrode sets, frequency resolution, and topomap color limits.

Run validation with:

```bash
conda run -n MNE_August2026 python -m unittest discover -s tests -v
```
