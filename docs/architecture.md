# Repository architecture

The repository separates orchestration from scientific calculations.

```text
run_pipeline.sh
  -> parkinson_eeg/cli.py
       -> config.py       validated public settings and profiles
       -> registry.py     explicit stage/dependency definitions
       -> runner.py       resume, status, dry-run, and execution
       -> stages.py       artifact checks and provenance fingerprints

scientific domains
  preprocessing          src/ and scripts/
  PSD                    psd_analysis/
  ordinal                ordinal_analysis/
  aperiodic + eBOSC      scale_free_analysis/
  within-bout ordinal    bout_analyses/
  independent detector   bycycle_burst_analysis/
  classification         exploration/
  clinical associations  quantitative_behavioral/ and disease_progression/

shared infrastructure
  src/analysis_io.py     participant and epoch discovery
  src/group_statistics.py
  src/group_statistics_plots.py
  src/plotting.py        figure output and visual policy
```

The domain packages remain self-contained so a researcher can read one method
without understanding the complete pipeline. Their `run_*.py` files are
internal stage entry points. Most users should only call `run_pipeline.sh`.

## Calculation ownership

Expensive features are calculated once and reused:

| Product | Owner | Views and consumers |
|---|---|---|
| Cleaned four-second epochs | preprocessing | every domain |
| PSD and band power | PSD | group, matched, eight-electrode, models, clinical |
| D=6 ordinal quantities | primary ordinal | group, matched, models, clinical |
| D=3–5 quantities | ordinal sweep | dimension sensitivity only |
| Specparam fits and eBOSC episodes | scale-free | bouts, fit QC, clinical, models |
| Within-bout quantities | within-bout ordinal | group, matched, clinical, models |

Matched analyses reuse compatible subject/electrode feature caches. They
recalculate cohort aggregation, paired inference, multiplicity correction,
figures, and reports. The eight-electrode and duration-QC analyses are views of
the same feature tables rather than independent signal-processing pipelines.

## Stage contract

Every registered stage declares:

- a stable identifier;
- a human-readable label;
- cohort and category;
- upstream dependencies;
- exact commands;
- required output artifacts;
- source and configuration paths that determine freshness.

After a successful stage, `.pipeline/state/` records its fingerprint. Existing
outputs created before this runner are adopted once when their artifact checks
pass. A source, configuration, or dependency change marks the stage stale.
Partial and stale stages are always replaced rather than appended to.

`.pipeline/` contains generated orchestration state, not scientific results,
and is ignored by Git.

## Adding an analysis

1. Put calculations in the closest scientific domain package.
2. Expose one small Python entry point.
3. Add a declarative `Stage` in `parkinson_eeg/registry.py`.
4. Declare all real dependencies and required artifacts.
5. Add focused numerical and stage-graph tests.
6. Document the method in that domain's README and the outputs in
   `docs/outputs.md`.

Do not add another top-level shell orchestrator or copy cohort-specific feature
calculations.

