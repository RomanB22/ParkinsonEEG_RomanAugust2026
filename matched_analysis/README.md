# Canonical matched-cohort analysis

This layer repeats the downstream analysis battery on one deterministic cohort
matched on the measured demographics age and sex. It does not alter or replace
the full-cohort results.

## Matching method

Every Control is paired to one unique PD participant of the same recorded sex.
Within each sex stratum, the Hungarian linear-assignment algorithm minimizes the
total absolute age difference without replacement. A pair must be within the
prespecified five-year age caliper. The current data yield 49 pairs (98 subjects),
with all 49 Controls retained.

The generated files under `matched_analysis/processed/` are the single source of
truth for every matched pipeline:

- `matched_subjects.csv`: full metadata for the selected subjects plus pair ID;
- `demographic_match_pairs.csv`: auditable Control–PD pair assignments;
- `demographic_balance.csv`: age and sex balance before and after matching;
- `subject_ids.txt`: the exact subject list;
- `configs/*.json`: generated pipeline configurations;
- `manifest.json`: matching settings, counts, paths, and subject IDs.

Matching balances measured age and sex between diagnosis groups. It does not
prove that age/sex confounding or unmeasured confounding has been eliminated.
For MOCA correlations within PD, age and sex adjustment is retained because
between-group matching does not remove within-PD associations with those
variables.

## Run it

```bash
bash matched_analysis/run_matched_analyses.sh
```

Use `--overwrite` to rebuild all matched products. The master scripts run both
cohorts automatically:

```bash
bash run_all_analyses.sh
bash run_reproducible_pipeline.sh run
```

Matched outputs are kept alongside full outputs in `processed_matched/` folders,
including PSD, ordinal metrics/planes/topomaps, scale-free fits and bouts,
within-bout ordinal analyses, prediction exploration, and behavioral analyses.
The matched ordinal sensitivity grid is stored in
`ordinal_analysis/parameter_sweep_matched/`.

The final matched sensitivity requires at least 60 accepted EEG seconds. A
complete demographic pair is retained only when both members pass; outputs are
written to `duration_qc_analysis/processed_matched/`. This preserves pairing
while leaving the primary matched cohort unchanged.
