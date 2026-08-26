# Reproduce the complete analysis

`run_reproducible_pipeline.sh` is the public entry point. It bootstraps the
conda environment and metadata, performs ICA cleaning, and invokes the internal
downstream orchestrator. Most users do not need to call `run_all_analyses.sh`
directly.

Both entry points check for the default `MNE_August2026` environment. If it is
missing, they create it with Python 3.14 and install the pinned
[`requirements.txt`](requirements.txt) stack. An existing environment is reused
without reinstalling packages. Use `--env NAME` only when an intentional
override is needed. If an interrupted Conda creation left a non-environment
directory at the requested prefix, setup preserves it under a timestamped
`.incomplete.*` name before creating a clean environment; it never deletes the
leftover automatically.

Every real `review` or `run` invocation writes the complete stdout/stderr stream
to a timestamped file under `pipeline_logs/` while continuing to display it in
the terminal. The final lines record `SUCCESS` or `FAILED` and the exit code.
Override the destination when needed:

```bash
bash run_reproducible_pipeline.sh run \
  --log-file pipeline_logs/my_full_run.log \
  --no-progress
```

`--no-progress` is optional, but produces a cleaner text log without progress-
bar redraw characters. After the run, list warnings and errors with:

```bash
rg -n -i "warning|error|traceback|exception|failed" pipeline_logs/*.log
```

Dry runs remain read-only and do not create a log.

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
bash run_reproducible_pipeline.sh run \
  --profile paper \
  --overwrite \
  --no-progress
```

The `run` command includes:

- reviewed ICA signal cleaning;
- PSD and relative band-power analyses;
- primary D=6 ordinal analysis plus D={3,4,5} sensitivity analyses, all at
  tau=1 (D=6 is calculated only once);
- scale-free/specparam, eBOSC bout, within-bout ordinal, and fit-QC analyses;
- a separate eight-electrode sensitivity battery using F4, P4, O2, P6, CP2,
  CP1, PO7, and P8 for non-progression group analyses;
- typical-bout and other diagnostic figures;
- PD-versus-Control exploration models;
- MOCA quantitative-behavioral analyses;
- whole-head UPDRS/MOCA disease-severity associations using all
  cohort-shared electrodes;
- an accepted-duration sensitivity requiring at least 60 seconds (15 retained
  four-second epochs), including group, MOCA, and prediction-model checks;
- the complete downstream battery for both the full cohort and the canonical
  age/sex-matched cohort;
- repository validation tests.

The matched pipeline runs by default. Do not pass `--skip-matched` when both
cohorts are required.

## Choose a profile

The same entry point supports three explicit workloads:

```bash
# Reusable base metrics, minimal figures, full cohort only
bash run_reproducible_pipeline.sh run --profile compute --no-progress

# Complete paper/report battery for full and matched cohorts (default)
bash run_reproducible_pipeline.sh run --profile paper --no-progress

# Paper profile plus the slow independent bycycle detector
bash run_reproducible_pipeline.sh run --profile full-qc --no-progress
```

`compute` stops after PSD, ordinal, scale-free/eBOSC, and within-bout ordinal
metric caches. `paper` adds inference, galleries, models, behavioral analyses,
duration sensitivity, matched analyses, and tests. `full-qc` additionally runs
the independent bycycle sensitivity. All profiles resume current stages unless
`--overwrite` is supplied.

Signal cleaning shows a participant-level progress bar and processes two
independent subjects concurrently by default. Adjust this according to
available memory with `--preprocessing-workers N`. Use one worker for the most
conservative memory footprint; two is the recommended default. Passing
`--no-progress` intentionally hides the cleaning and analysis progress bars.
When `--overwrite` is omitted, complete subject outputs are reused and only
missing or interrupted subjects are processed.

The computationally expensive independent bycycle burst sensitivity is not
part of the default full run. Include it for both cohorts only when requested:

```bash
bash run_reproducible_pipeline.sh run \
  --overwrite \
  --profile full-qc
```

The runner is bootstrappable from the source `dataset/`: if the root
`processed/` directory is absent, dataset inspection first creates
`processed/metadata/` and `processed/qc/dataset_inspection/` before signal
cleaning begins. Preprocessing-only tests run before cleaning; integration
tests that require generated analysis files run at the end of the complete
pipeline.

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

See [`PIPELINE_MAP.md`](PIPELINE_MAP.md) for the dependency graph, ownership of
cached calculations, and focused rerun commands.
