# Reproduce the complete analysis

Run the pipeline from ICA review through every full-cohort and matched-cohort
analysis using the reviewed two-step workflow.

## Recommended workflow: manual ICA review

First, generate the ICA review material:

```bash
bash run_reproducible_pipeline.sh review --overwrite
```

Inspect ICA stages 08–10 and confirm the component decisions in
`config/preprocessing.yaml`. Then run signal cleaning and the complete analysis
battery:

```bash
bash run_reproducible_pipeline.sh run --overwrite --no-progress
```

The `run` command includes:

- reviewed ICA signal cleaning;
- PSD and relative band-power analyses;
- primary and D/tau-sensitivity ordinal analyses;
- scale-free/specparam, bout, within-bout ordinal, and fit-QC analyses;
- typical-bout and other diagnostic figures;
- PD-versus-Control exploration models;
- MOCA quantitative-behavioral analyses;
- the complete downstream battery for both the full cohort and the canonical
  age/sex-matched cohort;
- repository validation tests.

The matched pipeline runs by default. Do not pass `--skip-matched` when both
cohorts are required.

## Automatic ICA fallback

To use automatic ICLabel proposals without manual confirmation:

```bash
bash run_reproducible_pipeline.sh run \
  --overwrite \
  --no-progress \
  --skip-manual-ica-review
```

This fallback is recorded in preprocessing QC and is not the recommended
scientific workflow.

## Resume an interrupted run

To reuse completed stages and continue only missing or stale stages, omit
`--overwrite`:

```bash
bash run_reproducible_pipeline.sh run --no-progress
```

Full-cohort outputs are written under each analysis folder's `processed/`
directory. Matched outputs use `processed_matched/`; the matched subject list,
pair assignments, balance table, and generated configs are stored under
`matched_analysis/processed/`.
