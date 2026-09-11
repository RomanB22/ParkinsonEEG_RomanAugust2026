# Independent all-electrode rhythmicity analysis

This document describes the optional rhythmicity pipeline based on the Python
implementation of **LAVI** (the LAVI toolbox; the user-facing abbreviation is
LAVI, not “LCVI”). It is deliberately independent of the main burst/aperiodic
pipeline: enabling or rerunning it does not alter the canonical recordings or
the existing global metrics.

## What is measured

For every retained recording, every EEG electrode, and a logarithmic frequency
grid (3.2–45 Hz by default), the pipeline computes a LAVI profile. LAVI is a
Lagged Angle Vector Index: it compares the complex wavelet representation at
two time points separated by the configured lag (1.5 cycles here). In
idealized form, the index is the magnitude of the normalized sum of the
complex-vector products across time, so values near one indicate a stable,
phase-consistent oscillatory pattern at that frequency and values near zero
indicate weak or inconsistent lagged phase structure. It is not a power or
amplitude measure. The profile is summarized in the conventional delta (1–4
Hz), theta (4–8 Hz), alpha (8–13 Hz), beta (13–30 Hz), and gamma (30–50 Hz)
bands. Bands are clipped to the available frequency grid, so gamma has bins
only up to 45 Hz with the default configuration.

The profile is computed after concatenating retained epochs within each
electrode, removing the channel mean, and excluding non-EEG channels and
finite/constant channels. All electrodes with a valid standard-10/20 montage
location are retained for topographic summaries. The default ABBA significance
classification uses the LAVI channel median as its baseline. Optional IAAFT/pink
surrogates can be enabled with `lavi.surrogate_reps > 0`; this is slower and is
reported in `manifest.json`.

## Inputs and reproducibility

Inputs are the cleaned canonical epochs under
`outputs/global/canonical/<dataset>/` and the existing recording-level burst
features under `outputs/global/metrics/<dataset>/recording_features.csv.gz`.
The recommended environment is `MNE_August2026`, which contains MNE and the
pinned Python LAVI commit listed in `requirements-rhythmicity.txt`.

Run the independent analysis with:

```bash
conda run -n MNE_August2026 bash scripts/run_rhythmicity_analysis.sh \
  --config config/analyses/rhythmicity.json --workers 4
```

The command is resumable. Existing per-recording files in
`outputs/rhythmicity/profiles/` are loaded unless `--overwrite` is supplied.
`--datasets` and `--recordings` allow a smoke test or a focused rerun, and
`--skip-figures` writes metrics/statistics without rendering figures. A tqdm
progress bar reports recording completion; process workers fall back to threads
on systems where process semaphores are unavailable.

## Optional LAVI-only burst characterization

The standard report compares LAVI with the existing global time-domain burst
table. To obtain an independent characterization using only LAVI, run:

```bash
MNE_DONTWRITE_HOME=true NUMBA_DISABLE_JIT=1 PYTHONPATH=src \
  conda run --no-capture-output -n MNE_August2026 \
  python scripts/run_lavi_burst_features.py
```

This command reads only the saved LAVI profiles and
`electrode_rhythmicity.csv.gz`; it does not read or modify global burst
features. A LAVI profile is a frequency-domain quantity, so “bouts” here mean
contiguous runs of ABBA high-rhythmicity frequency bins within a band. They are
not temporal bursts. The six plotted measures are high-rhythmicity bout count,
high-rhythmicity occupancy, bout density (count/Hz), mean bout width (Hz),
mean peak excess above the profile median, and mean bout peak frequency (Hz).

The command writes electrode- and participant-level tables, pairwise Welch
group tests with Benjamini–Hochberg FDR correction, and
`figures/lavi_burst_quantity_violins_<band>.png` (one figure each for delta,
theta, alpha, beta, and gamma). Violin overlays contain one point per
participant; each figure has all datasets as rows and quantities as columns.
Control-versus-PD groups are shown for the standard datasets, while the
medication-state row contains Control, PD-OFF, and PD-ON. Brackets with
`*`/`**`/`***` indicate FDR-adjusted q < .05/.01/.001.

### Exact LAVI-only quantity calculations

For each recording and electrode, `prepare_lavi` produces a LAVI value on
each frequency-grid bin. ABBA then classifies each bin relative to that
electrode's LAVI median: `+0.5` (high rhythmicity), `-0.5` (low rhythmicity),
or `0` (not retained as a significant ABBA segment when surrogate limits are
used). For a canonical band, only bins whose frequencies fall inside that
band are used. A bout is a maximal contiguous run of `+0.5` bins; therefore
the definition is in frequency, not time.

For a band with (K) available bins and frequency-bin edges (e_i), the
reported quantities are:

1. `lavi_high_bout_count`: number of contiguous high-rhythmicity runs.
2. `lavi_high_occupancy`: (K^{-1}sum_i I(\mathrm{ABBA}_i=+0.5)).
3. `lavi_high_bout_density_hz`: bout count divided by the available band
   span (e_{K}-e_0), in bouts/Hz.
4. `lavi_high_bout_width_hz_mean`: mean of (e_{j+1}-e_i) across bouts
   ([i,j]), the frequency-domain analogue of duration.
5. `lavi_high_bout_peak_excess_mean`: for each bout, its maximum LAVI minus
   the electrode's full-profile median; the reported value is the mean across
   bouts.
6. `lavi_high_bout_peak_frequency_hz_mean`: frequency of each bout's maximum
   LAVI, averaged across bouts.

Recording-level rows are first averaged across repeated sessions for each
participant/electrode/band, then across electrodes for participant-level
violins and group tests. Missing bouts produce zero count/occupancy/density;
width, peak excess, and peak-frequency means are missing rather than zero.
These summaries cannot yield temporal duration (seconds), bouts/minute, or
waveform cycles. Those require a time-resolved burst detector and should not
be inferred from these LAVI frequency bouts.

## Output files

`outputs/rhythmicity/metrics/`

* `electrode_rhythmicity.csv.gz`: one row per recording/electrode/band, with
  LAVI mean, median, peak, peak frequency, and high/low rhythmicity fractions.
* `participant_electrode_rhythmicity.csv.gz`: electrode values averaged across
  recordings for each participant.
* `participant_rhythmicity.csv.gz`: the analysis table used for inference;
  electrode values are averaged within participant and band. Burst quantities
  (bout count, occupancy, rate, duration, amplitude, and cycles) and available
  MoCA/MMSE/UPDRS metadata are merged here.
* `abba_bands.csv.gz`: frequency spans classified as high or low rhythmicity by
  ABBA.

`outputs/rhythmicity/statistics/`

* `lavi_group_comparisons.csv`: Welch two-sample comparisons of participant
  `lavi_mean` between available groups in each dataset and band. It contains
  sample sizes, means/SDs, mean difference, 95% CI, Welch t, degrees of freedom,
  raw p, and Benjamini–Hochberg q. FDR is controlled across bands within each
  dataset/comparison family. The `significant_fdr` flag is true at q < .05;
  `relevant_effect` additionally requires |Hedges g| ≥ 0.5.
* `lavi_electrode_comparisons.csv`: electrode-wise Welch tests with BH-FDR
  correction within each dataset, band, and comparison. Electrodes with q <
  .05 are marked with white circles in the contrast topomaps.
* `lavi_burst_correlations.csv`: Spearman correlations between participant
  `lavi_mean` and each burst quantity, with q values FDR-adjusted within dataset.
  Correlations use |rho| ≥ 0.30 as the practical-relevance threshold.
* `lavi_clinical_correlations.csv`: Spearman correlations for all six LAVI
  quantities (`mean`, `median`, `peak`, `peak_frequency_hz`,
  `high_rhythmicity_fraction`, and `low_rhythmicity_fraction`) against every
  available clinical outcome (MoCA and/or MMSE). Correlations require at least
  five complete, non-constant participant pairs; q values are adjusted across
  bands for each dataset, clinical measure, and LAVI quantity.
* `lavi_relationship_summary.csv`: one combined table with group differences,
  burst correlations, and clinical correlations. Filter
  `significant_and_relevant == True` to obtain relationships that are both
  statistically supported after FDR correction and of a pre-specified minimum
  practical size.

## Figures and how to read them

All figures are saved at 300 dpi in `outputs/rhythmicity/figures/`.

* `lavi_profiles_by_dataset.png` shows group means as bold lines with
  participant-level 95% confidence ribbons. Multiple recordings from one
  participant are averaged before the ribbon is calculated. Colored background
  strips mark canonical bands; the x-axis is logarithmic with explicit
  frequency ticks. Higher LAVI means greater phase/oscillation regularity, not
  greater power. Grey/orange are Control/PD in the standard datasets; grey,
  purple, and green are Control/PD-OFF/PD-ON in medication-state. The ribbons
  are uncertainty in the participant-level group means, not individual traces
  or electrode-wise intervals. Visual separation is descriptive; use the
  FDR-adjusted group-comparison table for significance. Each shaded region is
  labelled with its canonical band name directly in the panel.
* `lavi_band_group_effects.png` shows one violin per group and band. Every dot
  is one participant (deterministically jittered), with the median and extrema
  visible. Each band row uses a focused y-axis shared across datasets, so
  within-band differences are easier to see while cross-dataset comparisons
  remain valid. The small `n=` labels are the participants contributing to that
  panel; q values report the corresponding group tests, and brackets with
  `*`/`**`/`***` mark FDR q < .05/.01/.001.
* `lavi_group_comparisons.png` is a forest-style display of mean differences and
  95% CIs. Zero means no group difference; q values are printed beside each
  estimate.
* `lavi_clinical_correlations.png` shows participant-level scatterplots of band
  mean LAVI against MoCA and MMSE for every dataset and band. Lines are
  ordinary-least-squares fits; each panel reports n, Spearman rho, and FDR q.
  The full set of LAVI quantities is in the CSV table.
* `lavi_burst_associations.png` displays LAVI–burst Spearman rho by band and
  dataset. A `*` marks FDR significance and a `†` marks a practically relevant
  effect; both symbols together indicate a significant, relevant relationship.
  The clinical scatterplots use the same notation next to rho.
* `lavi_burst_quantity_violins_<band>.png` (from the optional LAVI-only command)
  shows the six frequency-domain bout measures for every participant, dataset,
  and group within one band. It does not reuse the global burst detector.
* `figures/topomaps/<dataset>_group_topomaps.png` contains group-average LAVI
  topomaps plus a PD-minus-Control (or medication-state PD-ON-minus-PD-OFF)
  contrast when the corresponding groups exist.
  `rhythmicity_topomap_summary.png` provides compact cross-dataset contrasts.
  Participant-level maps are collected in one PDF per dataset to keep file
  sizes manageable. All maps use smooth cubic interpolation and the viridis
  colour map. Group maps use a robust shared LAVI scale; contrast maps are
  centred on zero with limits estimated from the observed contrast
  distribution. White circles identify electrode-wise FDR-significant
  contrasts (q < .05).

## Statistical interpretation

The inferential unit is the participant, not the electrode, so electrodes do
not inflate the nominal sample size. Welch tests are descriptive group
comparisons and should be interpreted with the reported n, confidence interval,
and FDR-adjusted q rather than raw p alone. Spearman correlations are likewise
associations, not causal effects. Missing clinical scores are omitted pairwise;
datasets with fewer than five complete pairs are reported as unavailable rather
than assigned a misleading zero correlation.

## Current validation status

The implementation has been syntax-checked and exercised on representative
recordings from each dataset, including recordings containing non-EEG channels,
all-electrode topomaps, violin-point overlays, statistical-table generation,
and figure rendering. The full dataset run is resumable and should be launched
with the command above; the manifest records completed and failed recordings so
partial runs remain auditable. No group or clinical conclusion should be drawn
until that full run completes and the resulting CSVs have been reviewed.
