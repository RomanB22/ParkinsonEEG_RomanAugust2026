# Ordinal analysis with `ordpy`

This folder is a standalone downstream analysis of the accepted, cleaned EEG
epochs. It does not modify the source dataset, ICA solutions, cleaned FIF files,
or preprocessing decisions.

## Scientific scope

For every available electrode in every participant, the analysis estimates the
Bandt-Pompe ordinal-pattern distribution and uses `ordpy` 1.2.2 to calculate:

- normalized permutation entropy, **H**;
- Jensen-Shannon statistical complexity, **C**;
- discrete Fisher information, **F**.

The same complete analysis is performed on the broadband cleaned epochs and on
six band-pass versions: delta (1–4 Hz), theta (4–8 Hz), alpha (8–13 Hz), beta
(13–30 Hz), low gamma (30–50 Hz), and broad 5–15 Hz. The 5–15 Hz range overlaps
theta, alpha, and the lower edge of beta; it is an additional targeted summary,
not a statistically independent canonical band.

It also calculates one subject-level value for each quantity by taking the
arithmetic mean of that subject's electrode-level values. It deliberately does
not average EEG voltages across electrodes first: the cleaned data use an
average reference, so a spatially averaged waveform would collapse toward zero.

This stage is descriptive. It creates no p-values, multiple-comparison claims,
or classifier results.

## Inputs

The default configuration reads:

```text
processed/epochs/sub-*_task-Rest_desc-cleaned_epo.fif
processed/metadata/subjects.csv
```

The epoch FIF files contain only the 4-second epochs accepted by preprocessing.
Using them prevents rejected temporal artifacts from re-entering the ordinal
analysis. All 149 expected files must exist for the default full run.

## Ordinal parameters and tie policy

The defaults in [`config.json`](config.json) are:

| Parameter | Value | Meaning |
|---|---:|---|
| `embedding_dimension` | `3` | Each ordinal symbol orders three samples. There are `3! = 6` possible patterns. |
| `delay_samples` | `1` | Adjacent elements of a pattern are one 120 Hz sample apart (`1/120 ≈ 8.33 ms`). |
| `tie_precision` | `null` | Uses the `ordpy` default full-precision policy; samples retain their full float64 decimals and are never rounded. |

No noise or jitter is added. Exact equalities that remain at full precision use
`ordpy`'s deterministic `argsort` behavior. Their counts and fractions are
written for every subject/electrode so this edge case is visible rather than
hidden.

Ordinal counts are pooled across accepted epochs, but embeddings that would
cross an epoch boundary are removed. Thus a transition from the end of one
accepted epoch to the beginning of another cannot create an artificial ordinal
pattern. Vectorized NumPy ordering implements the same ordinal symbolization as
`ordpy.ordinal_sequence` without its row-wise overhead. The metric functions
`ordpy.complexity_entropy(..., probs=True)` and
`ordpy.fisher_shannon(..., probs=True)` receive the resulting pooled
probability distribution in lexicographic permutation order.

## Band filtering

Every accepted epoch and electrode is filtered independently with a
fourth-order Butterworth band-pass represented as second-order sections and
applied forward and backward by `scipy.signal.sosfiltfilt`. The result is
zero-phase. Filtering an epoch independently means that neither rejected-data
gaps nor the join between two accepted epochs can generate a filter transition.
Ordinal patterns are then pooled with the same boundary-safe procedure used by
the broadband analysis.

The filter method, order, phase, band limits, and boundary policy are explicit
in [`config.json`](config.json), repeated in every band-electrode table row, and
copied into `manifest.json`. Band limits are also checked against each input
file's Nyquist frequency when filtering begins.

## Run

From the repository root, create or update the complete project environment:

```bash
bash scripts/create_conda_environment.sh
```

Run all participants:

```bash
bash ordinal_analysis/run_ordinal_analysis.sh --overwrite
```

The terminal displays a `tqdm` progress bar covering every subject/analysis
stage and identifies the current subject and stage (loading, broadband, or band
name). Detailed subject/band progress is also retained in
`ordinal_analysis.log`. Use `--no-progress` to suppress the terminal bar in
non-interactive jobs.

For a quick selected-subject run:

```bash
bash ordinal_analysis/run_ordinal_analysis.sh \
  --subjects sub-001 sub-101 --overwrite
```

The runner refuses to replace an existing electrode-metrics table unless
`--overwrite` is supplied. A custom configuration can be selected with
`--config PATH`.

## Outputs

```text
ordinal_analysis/processed/
├── manifest.json
├── ordinal_analysis.log
├── metrics/
│   ├── analyzed_inputs.csv
│   ├── electrode_metrics.csv
│   ├── subject_electrode_mean_metrics.csv
│   ├── group_electrode_summary.csv
│   ├── group_subject_mean_summary.csv
│   ├── band_electrode_metrics.csv
│   ├── band_subject_electrode_mean_metrics.csv
│   ├── group_band_electrode_summary.csv
│   ├── group_band_subject_mean_summary.csv
│   └── electrode_sets.json
└── figures/
    ├── violins/
    │   ├── electrode_entropy_violins.png
    │   ├── electrode_complexity_violins.png
    │   ├── electrode_fisher_information_violins.png
    │   └── subject_electrode_mean_violins.png
    ├── planes/
    │   ├── electrode_hxc_p*.png
    │   ├── electrode_hxf_p*.png
    │   └── subject_electrode_mean_hxc_hxf.png
    ├── topomaps/
    │   ├── group_mean_topomaps.png
    │   ├── group_mean_zscored_topomaps.png
    │   └── subjects/sub-*_ordinal_topomaps.png
    └── bands/
        ├── delta|theta|alpha|beta|low_gamma|broad_5_15/
        │   ├── violins/*.png
        │   └── planes/*.png
        └── topomaps/
            ├── group_means/<band>_group_mean_topomaps.png
            ├── group_means_zscored/<band>_group_mean_zscored_topomaps.png
            └── subjects/sub-*_band_ordinal_topomaps.png
```

### Tables

`electrode_metrics.csv` contains one row per subject/electrode with H, C, F,
sample and epoch counts, the number of ordinal patterns, exact-tie diagnostics,
sampling rate, embedding parameters, and tie policy. Floating-point results are
written with 17 significant digits.

`subject_electrode_mean_metrics.csv` contains one row per participant and the
mean of each metric across their available electrodes. Each participant has one
of two 63-electrode layouts. `electrode_sets.json` records the 66-electrode union
and the 60 electrodes shared by everyone.

`band_electrode_metrics.csv` adds one row per subject, electrode, and band. It
includes H/C/F, pattern and exact-tie diagnostics, numerical band limits, and
filter provenance. `band_subject_electrode_mean_metrics.csv` contains one row
per subject and band, formed by averaging that subject's electrode-level metric
values. The two `group_band_*` tables provide the corresponding descriptive
PD/Control summaries.

The two group-summary tables report sample counts, means, standard deviations,
and medians for PD and Control without inferential testing.

### Figures

- Electrode violin figures compare the subject distributions for PD and Control
  at every electrode in the union.
- Subject-mean violins give each participant one value, preventing participants
  with more electrodes from receiving more weight.
- Per-electrode H×C and H×F pages plot every subject, colored by group.
- Subject-mean planes plot one point per participant.
- Every participant receives a three-panel H/C/F topomap using their actual
  available electrode locations.
- The group figure averages values at the 60 common electrodes. All subject and
  group topomaps use the same full-dataset color limits for a given metric.
- Each band receives the same violin and H×C/H×F products as broadband.
- Each participant also receives one 6-band × 3-metric topomap figure. Scales
  are fixed across participants and groups within each band/metric pair.
- Six group band figures compare PD and Control on the shared electrode set.
- Additional broadband and band-resolved group maps z-score each metric across
  all subjects pooled across groups, separately within every electrode (and
  band). Their diverging, zero-centered scales show standardized group
  deviations rather than absolute metric levels. Constant combinations are
  assigned zero; population standard deviation (`ddof=0`) is used.

## Provenance and validation

`manifest.json` records the complete configuration, software versions, group
counts, electrode sets, tie policy, epoch pooling policy, subject-mean
definition, filter policy, band limits, and broadband/band topomap color limits.
`ordinal_analysis.log` records progress through every subject and every band.

Run all repository tests with:

```bash
conda run -n MNE_Roman python -m unittest discover -s tests -v
```
