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

The default is resumable. `--overwrite` recomputes every selected stage, and is
forwarded only when it appears explicitly in the public command. A stale,
missing, or partial stage never receives it automatically. Complete stale
outputs that still pass artifact validation are preserved and adopted; stages
with missing or invalid artifacts execute their non-overwriting resume path.
For preprocessing, an automatically detected stale stage does not imply ICA
refitting: complete subject outputs with a matching preprocessing signature are
reused. A compatible saved ICA from an incomplete/review-only subject is also
loaded instead of refit. ICA is forcibly refit only when `--overwrite` is
explicitly supplied.

Compatible primary ordinal metrics from a completed compute run are validated
and reused, either to resume compute or to add paper figures.

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
bash src/analyses/matching/run_matched_analyses.sh --profile paper
```
