# Commands

Run commands from the repository root.

## Recommended reviewed workflow

```bash
bash run_pipeline.sh review --overwrite
# Inspect ICA QC stages 08–10 and confirm every subject.
bash run_pipeline.sh run --profile paper --no-progress
```

## Unattended reproducibility workflow

```bash
bash run_pipeline.sh run --profile paper \
  --overwrite \
  --skip-manual-ica-review \
  --preprocessing-workers 8 \
  --no-progress
```

## Preview and resume

```bash
bash run_pipeline.sh plan --profile paper
bash run_pipeline.sh status --profile paper
bash run_pipeline.sh run --profile paper
```

The default is resumable. `--overwrite` recomputes every selected stage.

## Downstream analyses

```bash
# Full plus matched paper battery
bash run_pipeline.sh analyses --profile paper

# Full cohort only
bash run_pipeline.sh analyses --profile paper --cohort full

# Matched view only; required full feature caches are resolved automatically
bash run_pipeline.sh analyses --profile paper --cohort matched

# Expensive independent bycycle sensitivity
bash run_pipeline.sh analyses --profile full-qc
```

## Focused stages

```bash
bash run_pipeline.sh list
bash run_pipeline.sh stage full.ordinal
bash run_pipeline.sh stage full.scale-free --overwrite --no-progress
bash run_pipeline.sh stage matched.cognition
```

Use `--no-deps` only when deliberately testing a stage with all inputs already
prepared.

## Environment and logging

The default environment is `MNE_August2026`. Override it with `--env NAME`.
Real runs create timestamped consolidated logs under `pipeline_logs/`:

```bash
bash run_pipeline.sh run --profile paper --log-file pipeline_logs/paper_run.log
```

## Compatibility aliases

These forward directly to the same runner:

```bash
bash run_reproducible_pipeline.sh run --profile paper
bash run_all_analyses.sh --profile paper
bash matched_analysis/run_matched_analyses.sh --profile paper
```
