# Eight-electrode sensitivity analysis

This battery repeats the PD-versus-Control inference for PSD relative power,
broadband and band-limited ordinal quantities, aperiodic properties, periodic
and bout properties, and within-bout ordinal quantities using exactly **F4,
P4, O2, P6, CP2, CP1, PO7, and P8**.

It is an additional sensitivity analysis. It does not replace the primary
whole-head outputs, and it is separate from `src/analyses/progression/`, which uses
all cohort-shared electrodes.

```bash
bash src/analyses/eight_electrode/run_eight_electrode_analysis.sh --overwrite
```

Results are written to `processed/`; the canonical matched runner writes an
independent paired analysis to `processed_matched/`. Every feature has a raw
group-distribution panel (with Control–PD pair lines in the matched cohort), a
subject-effect summary, and an eight-electrode effect heatmap. `REPORT.md`
summarizes corrected discoveries. Subject-level FDR is controlled within each
analysis domain. Electrode-wise tables include strict domain-wide FDR and
secondary within-feature localization FDR.
