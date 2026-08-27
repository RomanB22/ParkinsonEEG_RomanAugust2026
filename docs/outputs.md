# Output ownership

Output paths remain next to their scientific owners to preserve existing links
and report provenance.

| Domain | Full cohort | Matched cohort |
|---|---|---|
| Cleaning and metadata | `processed/` | shared subject-level cleaning |
| PSD | `psd_analysis/processed/` | `psd_analysis/processed_matched/` |
| Ordinal primary | `ordinal_analysis/processed/` | `ordinal_analysis/processed_matched/` |
| Ordinal D sensitivity | `ordinal_analysis/parameter_sweep/` | `ordinal_analysis/parameter_sweep_matched/` |
| Scale-free and bouts | `scale_free_analysis/processed/` | `scale_free_analysis/processed_matched/` |
| Within-bout ordinal | `bout_analyses/processed/` | `bout_analyses/processed_matched/` |
| Classification | `exploration/processed/` | `exploration/processed_matched/` |
| MOCA | `quantitative_behavioral/processed/` | `quantitative_behavioral/processed_matched/` |
| UPDRS/MOCA | `disease_progression/processed/` | `disease_progression/processed_matched/` |
| Eight-electrode view | `eight_electrode_analysis/processed/` | `eight_electrode_analysis/processed_matched/` |
| Duration QC | `duration_qc_analysis/processed/` | `duration_qc_analysis/processed_matched/` |

Every domain writes a `manifest.json`, metric tables, figures, and a log or
report where applicable. `.pipeline/state/` only records orchestration
fingerprints. Timestamped consolidated stdout/stderr logs are written to
`pipeline_logs/`.

Generated paths are ignored by Git. Original data remain read-only inputs.

