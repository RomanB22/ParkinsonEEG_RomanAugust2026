# Pipeline map

## One public command

```bash
bash run_reproducible_pipeline.sh review --overwrite
# Review and confirm ICA decisions.
bash run_reproducible_pipeline.sh run --profile paper --no-progress
```

Profiles:

| Profile | Purpose | Cohorts | Optional bycycle |
|---|---|---|---|
| `compute` | Build reusable primary feature caches with minimal figures | Full only | No |
| `paper` | Complete figures, inference, models, behavioral reports, and tests | Full + matched | No |
| `full-qc` | Complete paper battery plus independent burst-detector sensitivity | Full + matched | Yes |

Runs are resumable by default. Add `--overwrite` only when intentionally
replacing current products. `run_all_analyses.sh` is the internal downstream
orchestrator; individual analysis scripts remain available for focused reruns.
Preprocessing uses two subject-level workers by default and displays completed
subjects plus an ETA. Set `--preprocessing-workers 1` when memory is limited.
Each real top-level run also writes a consolidated timestamped stdout/stderr log
under `pipeline_logs/`; use `--log-file PATH` to override its location.

## Data flow and ownership

```text
dataset/
  -> preprocessing -> processed/epochs + processed/metadata
       -> PSD
       -> ordinal D=6 (primary)
       -> ordinal D=3,4,5 (sensitivity only)
       -> scale-free: PSD fit -> specparam -> wavelets -> eBOSC -> bouts
              -> within-bout ordinal (reuses scale-free episodes/thresholds)
       -> exploration / MOCA / UPDRS / eight-electrode / duration reports

full-cohort subject features
  -> validated filtering by matched subject IDs
       -> matched summaries, paired inference, FDR, figures, and reports
```

Each expensive calculation has one owner:

| Calculation | Owner | Reused by |
|---|---|---|
| Cleaned four-second epochs | preprocessing | every analysis |
| D=6 ordinal metrics | primary ordinal analysis | dimension and behavioral analyses |
| D=3–5 ordinal metrics | ordinal sensitivity sweep | dimension and behavioral analyses |
| PSD/specparam/wavelet/eBOSC detections | scale-free analysis | within-bout ordinal analysis |
| Subject-level ordinal and scale-free values | compatible full-cohort caches | matched analyses |

Cache reuse is fail-closed: parameter settings, subjects, complete row grids,
and shared electrodes must match. Statistical aggregation is never reused
across cohorts.

## Storage policy

- Scale-free cycle summaries are always calculated. Raw per-cycle tables are
  omitted by default because they previously dominated cache size; set
  `cache.save_raw_cycle_tables` to `true` only for cycle-level auditing.
- Within-bout episode and threshold paths are symbolic links to the scale-free
  cache, avoiding a second copy.
- Full-cohort products use `processed/`; matched products use
  `processed_matched/`; ordinal sensitivity products use `parameter_sweep/`
  and `parameter_sweep_matched/`.
- The independent bycycle detector is intentionally opt-in because it is a
  sensitivity analysis and one of the slowest stages.

## Focused reruns

```bash
# Downstream only, from existing cleaned epochs
bash run_all_analyses.sh --profile paper --no-progress

# Primary D=6 ordinal metrics and figures
bash ordinal_analysis/run_ordinal_analysis.sh --overwrite --no-progress

# D=3,4,5 sensitivity tables; D=6 is reused from the primary output
bash ordinal_analysis/run_ordinal_parameter_sweep.sh --overwrite --no-progress

# Scale-free fit and eBOSC cache, then fast within-bout ordinal metrics
bash scale_free_analysis/run_scale_free_analysis.sh --overwrite --no-progress
bash bout_analyses/run_bout_analyses.sh --overwrite --no-progress

# Matched downstream battery
bash matched_analysis/run_matched_analyses.sh --no-progress
```

Scientific definitions and output schemas remain in each analysis folder's
`README.md`. Copy-paste end-to-end commands are in `COMMAND.md`.
