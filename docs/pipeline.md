# Pipeline stages

Use `bash run_pipeline.sh list` for the live list of stable stage identifiers
and `bash run_pipeline.sh plan` for the exact ordered commands.

```text
inspect -> preprocessing tests -> ICA cleaning + epochs
                                  |-> PSD
                                  |-> ordinal D=6 -> ordinal D=3,4,5 sweep
                                  |-> scale-free -> within-bout ordinal
                                                    |
                                                    |-> group statistics
                                                    |-> fit-QC sensitivity
                                                    |-> typical-bout gallery
                                                    |-> classification
                                                    |-> MOCA associations
                                                    |-> UPDRS/MOCA associations

full-cohort features -> canonical demographic matching
                     -> matched cohort summaries and inference
```

## Profiles

| Profile | Purpose | Matched | Figures/reports | Bycycle |
|---|---|---:|---:|---:|
| `compute` | Reusable primary feature caches | No | Minimal | No |
| `paper` | Complete paper battery | Yes | Yes | No |
| `full-qc` | Paper battery plus independent burst sensitivity | Yes | Yes | Yes |

The independent bycycle detector is intentionally optional because it is a
slow sensitivity analysis rather than the primary bout definition.

## Resumption

The default behavior is resumable. For each stage the runner reports:

- `CURRENT`: artifacts and runner fingerprint agree;
- `LEGACY`: complete artifacts exist but predate the refactored runner;
- `STALE`: source, configuration, or an upstream dependency changed;
- `MISSING`: one or more required artifacts are absent;
- `ALWAYS`: a validation stage that intentionally reruns.

Use `--overwrite` only when every selected stage should be recomputed. The
runner never adds `--overwrite` merely because a stage is stale, missing, or
partial. Without explicit permission, each stage may safely resume/reuse its
compatible outputs or stop and request an explicit overwrite. A stale stage
whose required artifacts still pass validation is preserved and adopted under
the current runner fingerprint; invalid or missing artifacts are never adopted.

## Human ICA checkpoint

ICA review remains deliberately separate:

```bash
bash run_pipeline.sh review --overwrite
# Inspect ICA QC and confirm decisions in config/preprocessing.yaml.
bash run_pipeline.sh run --profile paper
```

For an explicitly unattended reproducibility run,
`--skip-manual-ica-review` applies the recorded high-confidence ICLabel
proposals and marks that choice in preprocessing provenance.
