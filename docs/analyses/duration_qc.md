# Accepted-duration QC sensitivity

The primary pipeline keeps non-overlapping 4-second epochs. This layer asks
whether conclusions remain similar after requiring at least 60 seconds, or 15
accepted epochs, per participant.

It deliberately reuses the primary subject-level quantities. Recomputing EEG
features after removing subjects could change the shared-electrode set and
would confound duration exclusion with a different feature definition.

The sensitivity layer recomputes:

- conservative PD-versus-Control comparisons, with age/sex-adjusted HC3 OLS
  in the full cohort and paired tests in the demographic-matched cohort;
- age/sex-adjusted and unadjusted MOCA correlations with the same family-level
  BH corrections used by `src/analyses/behavioral/`;
- nested-CV transparent prediction models without refitting any EEG quantity.

For the matched cohort, a complete pair is retained only when both its Control
and PD member pass duration QC.

Run after the ordinary exploration and quantitative-behavioral pipelines:

```bash
bash src/analyses/duration_qc/run_duration_qc_sensitivity.sh --overwrite
bash src/analyses/duration_qc/run_duration_qc_sensitivity.sh --matched --overwrite
```

Reports are written to `outputs/full/duration_qc/REPORT.md` and
`outputs/matched/duration_qc/REPORT.md`.

