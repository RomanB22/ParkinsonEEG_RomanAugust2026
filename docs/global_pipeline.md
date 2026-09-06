# Global multi-dataset pipeline

`run_global_pipeline.sh` is the dataset-agnostic entry point for the common
analysis battery. It expects cleaned MNE FIF epochs and a small dataset entry
for each study. The converter accepts CSV/TSV participant tables and maps
different study column names into one schema.

Configure the four studies in [`config/global_pipeline.json`](../config/global_pipeline.json).
The repository currently has two enabled cohorts and two disabled templates,
because only two cleaned cohorts are present in this checkout. Set the paths
and `enabled` to `true` for studies 3 and 4; no analysis code changes are
needed.

```bash
bash run_global_pipeline.sh --config config/global_pipeline.json --convert-only
bash run_global_pipeline.sh --config config/global_pipeline.json --skip-figures
bash run_global_pipeline.sh --config config/global_pipeline.json
```

By default, every dataset with `"enabled": true` is analyzed. To select a
subset at the command line, pass dataset IDs from the configuration:

```bash
bash run_global_pipeline.sh --datasets primary
bash run_global_pipeline.sh --datasets primary medication_state
```

This selector does not enable disabled datasets; first set their `enabled`
field to `true` and provide valid paths. An unknown or disabled ID fails before
EEG processing starts.

The pipeline performs Welch PSD and relative band power, electrode-wise group
statistics with Welch/Mann–Whitney tests and BH-FDR, PSD and entropy topomaps,
the four requested entropy quantities (H, C, F, and weighted entropy),
Hilbert-amplitude oscillatory bouts, within-bout H/C/F/weighted entropy, and
PD-only age/sex-adjusted partial Spearman correlations with UPDRS, MOCA, and
MMSE whenever those outcomes exist.

Outputs are written under `outputs/global/`:

```text
canonical/<dataset>/recordings.csv.gz
metrics/recording_features.csv.gz
metrics/subject_features.csv.gz
statistics/group_statistics.csv.gz
statistics/clinical_correlations.csv.gz
figures/<dataset>/*_topomaps.png
```

Memory is bounded by `block_epochs` (default 16). Epochs are opened with
`preload=False`, read one block at a time, downcast to float32, and discarded
after feature accumulation. Raw samples and bout waveforms are not written to
feature tables. Canonical tables use compressed CSV and do not require a
Parquet engine.
