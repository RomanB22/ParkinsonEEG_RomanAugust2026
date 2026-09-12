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

## Canonical bands versus ABBA segmentation

To inspect whether the fixed canonical bands are compatible with a
data-driven segmentation, run:

```bash
scripts/plot_abba_segmentation.sh
```

This produces `figures/abba_band_segmentation_comparison.png` and
`statistics/abba_representative_segments.csv`. It contains one panel per
dataset. The representative is selected as the participant whose
all-electrode, recording-averaged LAVI profile is closest (least-squares
distance) to that dataset's participant-median profile. The black curve is
that participant's mean LAVI profile; the dashed line is its ABBA median
baseline. Shaded labels show the canonical delta/theta/alpha/beta/gamma
intervals, while green and purple strips show ABBA high- and low-rhythmicity
segments and dots mark their peaks.

This figure is a sensitivity/interpretation aid, not a replacement for the
canonical bands used in the inferential tables. ABBA segments are defined on
the frequency grid and relative to the profile baseline; they do not establish
temporal burst durations or universally optimal clinical frequency boundaries.

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
* `abba_band_segmentation_comparison.png` overlays canonical bands and ABBA
  frequency segments for a representative participant in each dataset.
* `abba_band_limits_all_subjects.png` (also written as the backwards-compatible
  `abba_band_limits_by_dataset.png`) presents aligned, side-by-side interval
  strips for every participant. The upper strip in each panel is the fixed
  canonical band definition (with numeric limits); transparent middle strips
  show each participant's ABBA intervals; and opaque lower strips show ABBA
  applied to the mean participant profile for that dataset. All panels share
  one logarithmic frequency scale, so interval widths and shifts can be
  compared directly across datasets. Profiles for which ABBA returned no
  finite interval are reported in the panel count but do not contribute an
  interval overlay.
* `abba_band_limits_by_group.png` stratifies the all-participant result by
  diagnostic group: Control and PD for the standard datasets, and Control,
  PD-OFF, and PD-ON for medication-state. Transparent intervals are the
  participant-level limits within each group; opaque intervals are ABBA applied
  to that group's mean participant profile. The corresponding group-average
  limits are in `statistics/abba_group_mean_segments.csv`.
* `abba_burst_quantity_violins.png` applies the group-specific ABBA limits
  shown in `abba_band_limits_by_group.png` to the cleaned time-domain EEG and
  compares burst duration, cycle count, rate (normalised by interval width),
  occupancy, and peak EEG amplitude in µV. Every ABBA interval is retained
  separately with labels such as `delta_1`, `delta_2`, and so on; the group-mean
  direction (high/low LAVI) is retained as metadata rather than pooled. The
  medication row includes Control, PD-OFF, and PD-ON. Each point is one
  participant-state mean and asterisks mark the strongest FDR-adjusted group
  difference for the same label; the complete pairwise tests remain in the
  statistics CSV. Detection uses the robust standardized all-electrode signal,
  but the reported voltage is measured from an unstandardized highest-RMS
  electrode after broadband artifact-outlier rejection, so it is in physical
  microvolt units without common-average cancellation.
* `abba_burst_shapes.png` shows the average trough-aligned, phase-resolved
  burst shape for each named ABBA interval and population. Curves are first
  averaged within participant and then across participants; ribbons are 95%
  Student-t confidence intervals. The horizontal axis spans three cycles before
  to three cycles after the nearest trough, and the vertical axis is the
  band-passed EEG voltage in µV (not a normalized amplitude).
* Focused versions are also written for `theta_1`, `alpha_1`, `beta_1`,
  `beta_2`, and `gamma_1`: `abba_burst_quantity_violins_<band>.png` and
  `abba_burst_shapes_<band>.png`. Each title reports whether that interval is
  high- or low-LAVI (or whether the direction differs by population). For the
  focused `theta_1` comparison only, medication Control uses its native
  `theta_2` low-LAVI interval; its separate high-LAVI `theta_1` interval is
  not mixed into the PD-OFF/PD-ON comparison. This alignment is documented in
  `statistics/abba_burst_focused_group_comparisons.csv`.

### ABBA temporal-burst sensitivity analysis

Run the independent pipeline with
`bash scripts/run_abba_burst_analysis.sh --workers 4`. It reads the ABBA
group-mean intervals and canonical recording manifests, robustly z-scores each
electrode within epoch, and averages electrodes before filtering. For every
recording, the fixed group-specific ABBA intervals are filtered separately.

To isolate the effect of group-specific band selection, run the Control-band
quality-control analysis with:

```bash
bash scripts/run_abba_burst_analysis.sh --workers 4 --control-bands-qc
```

For every dataset, this selects only the Control group-mean ABBA intervals and
applies those exact frequency limits to Control and all PD groups. Bands use
direction-stable names such as `theta_low`, `alpha_high`, and `beta_high`.
Results are isolated under `outputs/rhythmicity/control_band_qc/`; the metric
tables also retain `source_group`, `source_band_name`, and `band_definition`.
Temporal bursts are runs above the 90th percentile of the band Hilbert
amplitude; boundaries are expanded to the 75th percentile and runs shorter
than two centre-frequency cycles are discarded. Quantities are calculated per
named interval and then averaged within participant, keeping participants as
the independent observations. The shape curves use the nearest trough and a
normalized ±3-cycle window. This is an intentional sensitivity implementation
of the paper's Figure 3B logic, not a replacement for the project's canonical
eBOSC burst pipeline.

### ABBA burst–clinical associations

Run `MPLCONFIGDIR=/tmp/mpl-cache-abba PYTHONPATH=src python scripts/plot_abba_burst_clinical.py`
after the ABBA burst metrics exist. The
script does not reread or refilter EEG: it merges
`metrics/abba_burst_participant_metrics.csv.gz` with the participant clinical
metadata in `outputs/global/canonical/*/recordings.csv.gz`. It writes one
participant-level scatter-plot figure for each selected interval
(`theta_1`, `alpha_1`, `beta_1`, `beta_2`, and `gamma_1`) and each outcome
family:

* `figures/abba_burst_clinical_cognitive_<band>.png` uses MoCA in the three
  standard datasets and MMSE in `medication_state`, with all available groups
  shown in their group colours.
* `figures/abba_burst_clinical_updrs_<band>.png` uses UPDRS and restricts the
  correlation to PD groups (`PD`, `PD-OFF`, and `PD-ON`), because UPDRS is a
  motor-severity scale rather than a control-versus-PD outcome. Medication
  state has no UPDRS values in the canonical metadata and is marked as
  unavailable.

Each panel is one burst quantity versus the clinical score for one dataset;
points are participant-state means, lines are pooled within-dataset least-
squares guides, and annotations report Spearman rho, n, and BH-FDR q. Before
testing, x-axis burst values are screened separately within each panel and
values more than three ordinary standard deviations from that panel's x mean
are excluded. No clinical-score values are removed by this rule. The retained
statistics are in `statistics/abba_burst_clinical_correlations.csv`; every
excluded participant/value is audited in
`statistics/abba_burst_clinical_x_outlier_exclusions.csv`. These are
exploratory, pooled associations: group separation, missing scores, the
outlier rule, and the data-driven ABBA interval definitions can all influence
a correlation, so they should not be interpreted as causal or medication
effects. The corresponding untrimmed sensitivity table is also retained as
`statistics/abba_burst_clinical_correlations_untrimmed.csv` so the impact of
the outlier rule can be checked directly.
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

## ABBA-band ordinal H/C/F pipeline

The ABBA ordinal pipeline calculates permutation entropy (H), statistical
complexity (C), and Fisher information (F) in each valid group-mean ABBA
interval. It produces two estimates per recording and electrode:

- `full_signal`: epochs are filtered independently in the ABBA interval and
  then concatenated before ordinal encoding.
- `within_bout`: temporal amplitude bouts are detected from the robust
  all-electrode signal using the same 90th-percentile detection,
  75th-percentile boundary, and two-cycle rules as the ABBA burst analysis.
  Matching filtered electrode segments are concatenated before ordinal
  encoding.

Both estimates intentionally permit ordinal patterns to cross concatenation
joins. The `concatenation_policy` column makes that requested choice explicit.
Electrode H/C/F values are averaged to recording level and repeated sessions
are averaged to participant level.

Run the configured Control/PD-OFF/PD-ON analysis with:

```bash
MNE_DONTWRITE_HOME=true NUMBA_DISABLE_JIT=1 PYTHONPATH=src \
  conda run --no-capture-output -n MNE_August2026 \
  python scripts/run_abba_ordinal_analysis.py --workers 4
```

The corresponding Control-band QC rerun is:

```bash
MPLCONFIGDIR=/tmp/mpl-cache-rhythmicity PYTHONPATH=src \
  python scripts/run_abba_ordinal_analysis.py --workers 4 --control-bands-qc
```

The configured embedding-dimension sensitivity analysis runs D=4, 5, and 6
independently at tau=1:

```bash
MNE_DONTWRITE_HOME=true NUMBA_DISABLE_JIT=1 PYTHONPATH=src \
  conda run --no-capture-output -n MNE_August2026 \
  python scripts/run_abba_ordinal_sweep.py --workers 4
```

Outputs and generated configs are isolated under
`outputs/rhythmicity/abba_ordinal_dimension_sweep/D<dimension>_tau1/`.
Compatible checkpoints from the former single-D run are reused for D=5.
Add `--control-bands-qc` to run the same D=4/5/6 sweep with each dataset's
Control-group ABBA limits applied unchanged to every population. Those outputs
are isolated in each dimension's `control_band_qc/` subdirectory.

It uses the same dataset-specific Control limits and `{band}_low` /
`{band}_high` notation as the burst QC, with independent checkpoints, metrics,
statistics, figures, and manifest under `outputs/rhythmicity/control_band_qc/`.

The default configuration selects all four canonical datasets and retains
Control, PD, PD-OFF, and PD-ON as distinct populations. Use `--datasets` to
override the configured datasets,
`--recordings ...` for a smoke test, `--skip-figures` for computation only,
or `--overwrite` to ignore compatible checkpoints. Each recording is written
atomically under `outputs/rhythmicity/intermediate/abba_ordinal_checkpoints/`.
After each dataset finishes, its recording and electrode tables are saved
under `metrics/abba_ordinal_by_dataset/<dataset>/`, so an interruption does not
discard a completed dataset.

Final outputs are:

- `metrics/abba_ordinal_recording_metrics.csv.gz`
- `metrics/abba_ordinal_electrode_metrics.csv.gz`
- `metrics/abba_ordinal_participant_metrics.csv.gz`
- `statistics/abba_ordinal_clinical_correlations.csv`
- `figures/abba_ordinal/<dataset>__<ABBA-band>.png`
- `figures/abba_ordinal/cross_dataset_planes/<scope>__<region_direction>.png`
- `abba_ordinal_manifest.json`

The clinical table contains Spearman correlations for MoCA, MMSE, and UPDRS,
both overall and stratified by group when at least five complete observations
exist. For paired medication recordings it also correlates PD ON−OFF H/C/F
changes with ON−OFF UPDRS changes. Each band figure contains separate H×F planes for the full and bout
signals (point fill represents C), plus H/C/F scatterplots against the
available cognitive score (MoCA preferred, MMSE otherwise) and UPDRS.
Population is encoded by both color and marker shape in every panel: circles
for Control, squares for PD, triangles for PD-OFF, and diamonds for PD-ON.
For medication-state theta comparisons, the configured
`theta_low_aligned` comparison deliberately combines Control `theta_2` with
PD-OFF `theta_1` and PD-ON `theta_1`. These are all ABBA low-LAVI intervals.
The native interval remains in `band_name`; the harmonized plotting and
correlation label is stored separately in `comparison_band_name`.
The unmatched high-LAVI Control `theta_1` is labeled
`theta_high_control_only` so it cannot be mistaken for the aligned
cross-population theta comparison.

The cross-dataset plane gallery contains separate figures for `full_signal`
and `within_bout`. Within each scope, one figure is written for every ABBA
region/direction characteristic observed in at least two datasets. Each figure
has four dataset rows and paired H×C and H×F columns with shared axes. Thus the
theta-low figure compares the aligned low-LAVI theta interval in all four
datasets, including the special medication-state mapping above.
The medication-state canonical table does not expose Total UPDRS in its generic
`updrs` field, so this pipeline reads `Total UPDRS` from each session's source
behavior JSON; `updrs_source` records that file in the electrode and recording
outputs.
