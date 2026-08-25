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
  BH corrections used by `quantitative_behavioral/`;
- nested-CV transparent prediction models without refitting any EEG quantity.

For the matched cohort, a complete pair is retained only when both its Control
and PD member pass duration QC.

Run after the ordinary exploration and quantitative-behavioral pipelines:

```bash
bash duration_qc_analysis/run_duration_qc_sensitivity.sh --overwrite
bash duration_qc_analysis/run_duration_qc_sensitivity.sh --matched --overwrite
```

Reports are written to `duration_qc_analysis/processed/REPORT.md` and
`duration_qc_analysis/processed_matched/REPORT.md`.

