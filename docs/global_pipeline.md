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
`preload=False`, read into labeled xarray blocks, downcast to float32, and
discarded after feature accumulation. Vectorized NumPy/SciPy kernels operate
on the xarray block data; xarray retains epoch/channel/time labels without
materializing a full study cube. Raw samples and bout waveforms are not
written to feature tables. Canonical tables use compressed CSV and do not
require a Parquet engine.
