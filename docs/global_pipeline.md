# Global multi-dataset pipeline

`run_global_pipeline.sh` is the dataset-agnostic entry point for the complete
workflow. It preprocesses raw BIDS EEG into cleaned MNE FIF epochs, converts
metadata into one schema, and then runs the common analysis battery. The
converter accepts CSV/TSV participant tables and maps different study column
names into one schema.

The four configured studies are:

- `dataset` → `primary`
- `ds002778-1.0.5` → `medication_state`
- `ds007526-1.0.2`
- `ds008768-1.0.0`

All four are enabled in [`config/global_pipeline.json`](../config/global_pipeline.json).

```bash
bash scripts/ensure_conda_environment.sh --env MNE_August2026
conda run -n MNE_August2026 python -m pip install "xarray>=2024.10.0"
bash run_global_pipeline.sh --config config/global_pipeline.json --convert-only
bash run_global_pipeline.sh --config config/global_pipeline.json --skip-figures
bash run_global_pipeline.sh --config config/global_pipeline.json
```

The last command preprocesses every enabled dataset before analysis. Add
`--overwrite` to recompute existing ICA, cleaned raw files, and epochs. Use
`--analysis-only` only when the cleaned epoch files are already current:

```bash
bash run_global_pipeline.sh --analysis-only
```

During analysis, a progress bar reports completed recordings across all
selected datasets. Datasets are processed sequentially from the fewest to the
largest number of recordings. Within each dataset, recordings can be analyzed
in parallel with one complete recording per worker; the parent process writes
the caches. As soon as a dataset finishes, its metrics, statistics, and all
figures are written. It also indicates whether the current subject result was
reused from the resumable intermediate cache. Use `--no-progress` in batch
logs or non-interactive jobs to disable it.

For example, use eight analysis workers:

```bash
bash run_global_pipeline.sh --analysis-only --analysis-workers 8
```

Dataset-specific intermediate tables are written under
`outputs/global/metrics/<dataset>/` and
`outputs/global/statistics/<dataset>/`; combined tables remain directly under
`metrics/` and `statistics/`.

When resuming after a preprocessing configuration change, the global runner
automatically recomputes only recordings whose saved ICA provenance is stale:

```bash
bash run_global_pipeline.sh \
  --skip-manual-ica-review \
  --preprocessing-workers 10
```

Use `--no-repair-incompatible-ica` to restore the strict gate.

By default, every dataset with `"enabled": true` is analyzed. To select a
subset at the command line, pass dataset IDs from the configuration:

```bash
bash run_global_pipeline.sh --datasets primary
bash run_global_pipeline.sh --datasets primary medication_state
```

This selector does not enable disabled datasets; first set their `enabled`
field to `true` and provide valid paths. An unknown or disabled ID fails before
EEG processing starts.

If cleaned epochs do not exist yet, add `preprocessing_config` to the dataset
entry. The normal command will run it automatically; the explicit equivalent
is:

```bash
bash run_global_pipeline.sh \
  --datasets ds007526-1.0.2 \
  --preprocess \
  --skip-manual-ica-review \
  --preprocessing-workers 4
```

For the primary and medication datasets, preprocessing configs are already
provided. The cleaning command performs filtering, notch filtering, resampling,
ICA/QC, and four-second epoching using the existing repository contract. Manual
ICA review remains the default; `--skip-manual-ica-review` is an explicit
unattended option.

The global runner records and skips recordings with degenerate ICA input, such
as a recording whose artifact annotations leave fewer than two seconds of
usable signal. These exclusions are written to each preprocessing output's
`qc/preprocessing_failures.csv`; all valid recordings continue through the
pipeline. Standalone preprocessing remains fail-fast unless
`--skip-unusable-recordings` is supplied.

The pipeline performs subject-level Welch PSD and relative band power,
4–50 Hz aperiodic spectral fitting with fixed and knee models selected by BIC,
electrode-wise group statistics with Welch/Mann–Whitney tests and BH-FDR, PSD,
aperiodic, and entropy topomaps,
the four requested entropy quantities (H, C, F, and weighted entropy),
eBOSC/wavelet aperiodic-relative oscillatory bouts, within-bout
H/C/F/weighted entropy, and
PD-only correlations with UPDRS, MOCA, and MMSE whenever those outcomes exist.
Both the legacy age/sex-adjusted partial Spearman result and an unadjusted
Spearman sensitivity result are saved.

Outputs are written under `outputs/global/`:

```text
canonical/<dataset>/recordings.csv.gz
metrics/recording_features.csv.gz
metrics/subject_features.csv.gz
metrics/analysis_exclusions.csv.gz
metrics/aperiodic_fit_qc.csv.gz
metrics/aperiodic_subject_qc.csv.gz
statistics/group_statistics.csv.gz
statistics/clinical_correlations.csv.gz
figures/<dataset>/*_topomaps.png
figures/<dataset>/psd_broadband_mean_ci.png
figures/<dataset>/average_detected_bouts_<band>.png
figures/<dataset>/aperiodic_topomaps_broadband.png
figures/<dataset>/*_contrast_*_topomaps.png
figures/<dataset>/scatter_<moca|mmse|updrs>_<bout|within_bout>_<band>.png
figures/<dataset>/subject_violins_<family>_<band|broadband>.png
figures/<dataset>/entropy_hx[cf]_planes_<band>.png
figures/<dataset>/within_bout_entropy_hx[cf]_planes_<band>.png
```

Band-specific figures are written separately for every configured band. The
aperiodic and full-spectrum PSD summaries are broadband figures; the remaining
families use one figure per band.

The analysis unit is one complete cleaned recording at a time. All accepted
four-second epochs for that subject/session/condition are loaded together and
released before the next recording is opened. Recordings are never
concatenated across subjects or medication conditions. PSD uses the
subject-level 4-second Hann Welch spectrum with no overlap. Band filtering,
full-signal ordinal patterns, and eBOSC bout detection preserve epoch
boundaries; ordinal counts are pooled across epochs without creating windows
across rejected-data gaps. `block_epochs` is retained as a compatibility
setting but is no longer an analysis boundary.
Raw samples and bout waveforms are not written to feature tables. Canonical
tables use compressed CSV and do not require a Parquet engine.

After every recording, resumable intermediate files are written under
`outputs/global/intermediate/subjects/<dataset>/`:

```text
<recording>_features.csv.gz
<recording>_permutation_patterns.npz
<recording>_metadata.json
```

The compressed NPZ stores the lexicographic permutation order and exact pooled
pattern-count vectors for boundary-safe full-signal and within-bout analyses, for every
configured embedding dimension (D=3, 4, 5, 6, and 7 by default), together with
the weighted-entropy accumulators, eBOSC settings, and PSD spectrum. It does
not store the raw symbol sequence. On a rerun, a matching complete subject
cache is reused; a changed signal file or analysis configuration automatically
invalidates that subject only. Aggregation, FDR statistics, clinical
correlations, and plotting then run from the compact subject results after all
selected recordings are available.

The PSD figure shows the subject-level median PSD across available EEG
electrodes with a pointwise 95% bootstrap confidence interval across
recordings. This robust summary prevents a small number of high-power
recordings from dominating the figure. Population topomap panels are arranged
vertically by feature (one feature per row, groups across columns), use shared
color limits for each feature, and identify the group, family, band, and metric
(including the embedding dimension where applicable). Contrast topomaps use
the same vertical layout and show `group_b - group_a` on a symmetric scale
centered at zero; white electrode markers identify electrodes whose Welch-test
p-values survive the BH-FDR threshold, which is stated in the figure title.
Clinical
scatter plots use one point per participant and condition, restricted to
PD-labeled groups, and are created only when the corresponding clinical
values are available. Their annotations are unadjusted Spearman associations;
the age/sex-adjusted and unadjusted results remain in
`statistics/clinical_correlations.csv.gz`, with FDR applied separately by
dataset, outcome, feature family, and method.

Subject violin plots first average all available electrode rows for each
participant and group/condition. They therefore show one biological
observation per participant/condition rather than one observation per
electrode. Entropy and within-bout entropy violins use the configured primary
embedding dimension; all other configured metrics are included. Each violin
panel annotates pairwise Welch tests on the subject-level electrode averages,
showing both the raw p-value and the BH-FDR-adjusted q-value. Comparisons are
called significant only when q < 0.05 (or the configured FDR threshold).

`average_detected_bouts_<band>.png` shows the legacy-style subject-balanced average
around the midpoint of each detected eBOSC bout: baseline-subtracted Hilbert
envelope, relative Hilbert phase, and phase-aligned band-passed signal. Bout
waveforms are averaged across electrodes within each recording, then across
recordings; shaded intervals are Student-t 95% intervals across recordings.

Aperiodic QC uses the configured minimum R², maximum log10 MAE, maximum
absolute log10 residual, and exponent range. Failed QC fits are retained in
the QC tables with their reasons and raw diagnostics, but their aperiodic
features are set to missing and they do not contribute to eBOSC detection,
aperiodic statistics, topomaps, or violin plots. The subject QC table reports
the fraction of passing electrodes using the configured 80% threshold.

Entropy-plane figures show H versus complexity (H×C) and H versus Fisher
information (H×F), separately for every frequency band. They are produced for
both full-signal entropy and within-bout entropy, with one electrode-averaged
point per participant and condition, using the configured primary dimension
(D=6 by default). The saved feature table and clinical/statistical tables
contain all configured dimensions D=3–7.
