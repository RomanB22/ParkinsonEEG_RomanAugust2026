# Ordinal analysis within detected oscillatory bouts

This standalone downstream pipeline detects transient oscillatory bouts and
calculates regular Bandt–Pompe permutation entropy (**H**), Jensen–Shannon
statistical complexity (**C**), and discrete Fisher information (**F**) from
the band-limited EEG samples inside those bouts. It does **not** calculate any
Rényi quantity and does not modify preprocessing or scale-free outputs.

## Scientific sequence

The analysis uses only accepted four-second cleaned epochs and only electrodes
present in every analyzed subject. Its sequence is:

1. Calculate a subject/electrode Welch PSD over 1–50 Hz.
2. Fit the 1–50 Hz PSD with fixed-mode `specparam`.
3. Map the fitted aperiodic spectrum to the eBOSC Morlet-power scale.
4. Detect samples above the 95th-percentile aperiodic-relative power threshold
   for at least three cycles, independently within each accepted epoch.
5. Collapse detections into theta (4–7 Hz), alpha (8–13 Hz), low-beta
   (13–20 Hz), and high-beta (20–30 Hz) bouts.
6. Zero-phase band-pass each complete accepted epoch before extracting the
   corresponding bout intervals. Short bouts are never filtered in isolation.
7. Encode every bout independently into ordinal patterns. Pattern counts are
   pooled only after encoding, so no embedding crosses a bout boundary, epoch
   boundary, rejected interval, or join between bouts.
8. Calculate regular H, C, and F from the pooled distribution for every
   subject/shared-electrode/band. Electrode metrics are then averaged to one
   value per subject/band for descriptive PD/Control figures and summaries.

After this primary all-electrode analysis, the fit-QC sensitivity stage can
exclude failed specparam electrodes and re-aggregate H, C, and F. Formal
QC-qualified summaries require at least 48/60 passing electrodes per subject.
The repeated exponent and R² are checked against the scale-free pipeline to
numerical precision before existing passing-electrode results are reused.

The default ordinal parameters are `D=6` and `tau=1` sample, matching the
prespecified primary configuration in the existing ordinal workflow. The
state space therefore contains `6! = 720` possible patterns.

## Time-limited bouts and sample-size transparency

For `D=6`, `tau=1`, a bout must contain at least six samples to yield one
ordinal pattern. A shorter bout remains in the detection tables but is marked
`analyzable_ordinal_bout=0` and is never joined to another bout to manufacture
patterns. The output reports:

- detected and ordinal-analyzable bout counts;
- short-bout exclusions;
- pooled ordinal-pattern count;
- observed pattern-state count and `D!` state-space coverage;
- exact-tie count and fraction;
- the full sparse or dense count vector in a compressed NPZ file.

Per-bout H/C/F values are saved for inspection, but the primary electrode-level
quantities come from the distribution pooled across all bouts for that
subject/electrode/band. This is more stable than averaging estimates from many
short individual bouts. Subjects or electrodes with no analyzable patterns
receive missing H/C/F values rather than an invented value.

## Run

Run the full cohort and all shared electrodes:

```bash
bash bout_analyses/run_bout_analyses.sh --overwrite
```

Then generate QC-filtered bout-property and within-bout ordinal sensitivity
outputs:

```bash
bash scale_free_analysis/run_fit_qc_sensitivity.sh
```

Run a development pilot without changing the configured output:

```bash
bash bout_analyses/run_bout_analyses.sh \
  --subjects sub-001 sub-101 \
  --channels Fz Cz P3 Oz \
  --output-dir /tmp/bout-analyses-pilot \
  --overwrite
```

`--subjects` and `--channels` restrict a pilot. Requested channels must be
shared by every selected subject. `--no-progress` disables the live progress
bar. The runner refuses to replace an existing primary result table unless
`--overwrite` is supplied.

## Outputs

```text
bout_analyses/processed/
├── manifest.json
├── bout_analyses.log
├── metrics/
│   ├── analyzed_inputs.csv
│   ├── electrode_sets.json
│   ├── subject_electrode_band_metrics.csv
│   ├── subject_band_metrics.csv
│   ├── group_band_summary.csv
│   ├── subject_electrode_band_metrics_fit_qc.csv
│   ├── subject_band_metrics_fit_qc_all_subjects.csv
│   ├── subject_band_metrics_fit_qc.csv
│   ├── group_band_summary_fit_qc.csv
│   ├── pd_control_comparisons_fit_qc.csv
│   ├── within_bout_ordinal_fit_qc_sensitivity.csv
│   ├── bout_duration_records.csv.gz
│   └── example_bout_ordinal_distribution.csv
├── intermediate/
│   ├── episodes/sub-*_bout_episodes.csv.gz
│   ├── thresholds/sub-*_ebosc_thresholds.csv.gz
│   ├── bout_metrics/sub-*_bout_ordinal_metrics.csv.gz
│   └── ordinal_counts/sub-*_ordinal_counts.npz
└── figures/
    ├── steps/
    │   ├── 01_bout_detection.png
    │   └── 02_ordinal_encoding.png
    ├── quality/bout_and_ordinal_diagnostics.png
    ├── fit_qc_sensitivity/within_bout_ordinal_all_vs_fit_qc.png
    ├── group/
    │   ├── subject_metric_violins.png
    │   └── subject_ordinal_planes.png
    ├── electrodes/<band>_electrode_violins.png
    └── topomaps/
        ├── group_mean_topomaps.png
        └── subjects/sub-*_bout_ordinal_topomaps.png
```

Each subject NPZ stores a count tensor shaped
`(shared electrodes, bands, D!)`, together with ordered electrode, band, and
ordinal-pattern labels. Dividing a count vector by its sum exactly reconstructs
the probability distribution supplied to `ordpy`.

## Figure audit trail

- `01_bout_detection.png` shows the fitted spectrum, mapped aperiodic
  background, power threshold, highlighted bout mask, and duration-qualified
  time-frequency detections.
- `02_ordinal_encoding.png` follows one band-limited bout from samples through
  its ordinal-symbol sequence to its probability distribution.
- The quality page exposes bout durations, bout availability, ordinal sample
  sizes, and state-space coverage by band and group.
- Subject-balanced group violins and H×C/H×F planes show H/C/F without treating
  electrodes or bouts as independent participants.
- Electrode violins preserve spatial detail; group and per-subject topomaps
  make spatial patterns inspectable.

These are descriptive outputs. The pipeline deliberately does not turn every
bout into an independent group-statistical observation, which would
pseudoreplicate participants.
