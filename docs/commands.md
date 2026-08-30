# Commands

Run commands from the repository root.

## Unified preprocessing for both datasets

The `preprocess` command selects a dataset profile while keeping the signal
processing and QC implementation identical. Dataset-specific source formats,
task labels, auxiliary channels, ICA decisions, and output roots remain in
their own preprocessing configuration files.

```bash
# Preview the complete shared workflow for all 195 recordings.
bash run_pipeline.sh preprocess clean --dataset both --dry-run

# Inspect source metadata without cleaning signals.
bash run_pipeline.sh preprocess inspect --dataset both

# Confirm that every recording has current, complete outputs.
bash run_pipeline.sh preprocess status --dataset both

# Generate ICA review material for either dataset or both.
bash run_pipeline.sh preprocess review --dataset primary --workers 4
bash run_pipeline.sh preprocess review --dataset ds002778 --workers 4

# After confirming each dataset's ICA decisions, run all preprocessing/QC steps.
bash run_pipeline.sh preprocess clean --dataset both --workers 8 --no-progress

# Explicit unattended alternative; the automatic choice is recorded in provenance.
bash run_pipeline.sh preprocess clean --dataset both \
  --skip-manual-ica-review --workers 8 --no-progress
```

Use `--subjects` with participant IDs for the primary dataset or recording IDs
such as `sub-pd3_ses-off` for ds002778. Outputs remain separated under
`processed/` and `processed_ds002778/`.

## Unified full analysis and plotting

Dataset selection also applies to the downstream stage graph. The primary
dataset retains its full/matched design, while ds002778 uses the same shared
PSD, ordinal, aperiodic/specparam, and eBOSC feature families with paired
ON/OFF inference and MMSE models appropriate to its repeated sessions.

```bash
# Preview every selected stage and exact command.
bash run_pipeline.sh plan --profile full-qc --dataset both

# Run or resume preprocessing, every analysis, and all figure products.
bash run_pipeline.sh run --profile full-qc --dataset both --no-progress

# Downstream only, requiring complete cleaned epochs for both datasets.
bash run_pipeline.sh analyses --profile full-qc --dataset both --no-progress

# Inspect freshness across both dataset workflows.
bash run_pipeline.sh status --profile full-qc --dataset both
```

The shared feature definitions are identical where the datasets support them;
the inferential design is not forced to be identical. In particular,
ds002778 preserves within-person PD ON/OFF pairing rather than treating its 46
recordings as independent subjects.

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
