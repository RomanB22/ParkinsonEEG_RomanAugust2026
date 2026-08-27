# Canonical matched-cohort analysis

This layer repeats the downstream analysis battery on one deterministic cohort
matched on the measured demographics age and sex. It does not alter or replace
the full-cohort results.

The matched battery also includes the whole-head PD-only UPDRS/MOCA severity
analysis under `outputs/matched/progression/`, plus a distinct
eight-electrode group-comparison sensitivity under
`outputs/matched/eight_electrode/`. Disease progression uses all
cohort-shared electrodes and the PD members retained by the canonical match;
within-PD correlations are not pairwise tests. The eight-electrode group
comparison preserves Control–PD pairing.

## Matching method

Every Control is paired to one unique PD participant of the same recorded sex.
Within each sex stratum, the Hungarian linear-assignment algorithm minimizes the
total absolute age difference without replacement. A pair must be within the
prespecified five-year age caliper. The current data yield 49 pairs (98 subjects),
with all 49 Controls retained.

The generated files under `outputs/matched/cohort/` are the single source of
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
bash run_pipeline.sh analyses --profile paper --cohort matched
```

The independent bycycle burst sensitivity is optional and skipped by default.
Include it explicitly with:

```bash
bash run_pipeline.sh analyses --profile full-qc --cohort matched
```

Use `--overwrite` to rebuild all matched products. The paper profile runs both
cohorts automatically:

```bash
bash run_pipeline.sh analyses --profile paper
```

Matched outputs are kept alongside full outputs in `processed_matched/` folders,
including PSD, ordinal metrics/planes/topomaps, scale-free fits and bouts,
within-bout ordinal analyses, the eight-electrode sensitivity battery,
prediction exploration, and behavioral analyses.
The matched ordinal sensitivity grid is stored in
`outputs/matched/ordinal_sweep/`.

Ordinal H/C/F and Rényi values are subject-level feature calculations, so the
matched pipeline filters them from compatible full-cohort D=3–6 caches instead
of reading and filtering the same EEG again. Before reuse it verifies the
complete subject/electrode grids, exact shared-electrode order, embedding
parameters, bands, and filter configuration. Cohort summaries, paired
statistics, FDR correction, and all matched figures are always recalculated.
The matched scale-free stage applies the same guarded reuse to subject-level
fits, detections, and cycle summaries, then reruns matched fit-QC summaries,
paired inference, FDR, and figures. The within-bout ordinal stage consumes that
matched cache rather than repeating PSD, specparam, wavelet, or eBOSC work.

The final matched sensitivity requires at least 60 accepted EEG seconds. A
complete demographic pair is retained only when both members pass; outputs are
written to `outputs/matched/duration_qc/`. This preserves pairing
while leaving the primary matched cohort unchanged.
