# Output ownership

Output paths remain next to their scientific owners to preserve existing links
and report provenance.

| Domain | Full cohort | Matched cohort |
|---|---|---|
| Cleaning and metadata | `processed/` | shared subject-level cleaning |
| PSD | `outputs/full/psd/` | `outputs/matched/psd/` |
| Ordinal primary | `outputs/full/ordinal/` | `outputs/matched/ordinal/` |
| Ordinal D sensitivity | `outputs/full/ordinal_sweep/` | `outputs/matched/ordinal_sweep/` |
| Scale-free and bouts | `outputs/full/scale_free/` | `outputs/matched/scale_free/` |
| Within-bout ordinal | `outputs/full/bouts/` | `outputs/matched/bouts/` |
| Classification | `outputs/full/exploration/` | `outputs/matched/exploration/` |
| MOCA | `outputs/full/behavioral/` | `outputs/matched/behavioral/` |
| UPDRS/MOCA | `outputs/full/progression/` | `outputs/matched/progression/` |
| Eight-electrode view | `outputs/full/eight_electrode/` | `outputs/matched/eight_electrode/` |
| Duration QC | `outputs/full/duration_qc/` | `outputs/matched/duration_qc/` |

The sessioned ds002778 workflow follows the same domain ownership under
`outputs/ds002778/`: `psd/`, `ordinal/`, `scale_free/`, `bouts/`, and
`behavioral/`. Each contains its own `manifest.json`, `metrics/`, and
`figures/`. The recording-aware canonical long tables remain in
`outputs/ds002778/features/` for paired cross-domain ON/OFF models.
The comprehensive violin, condition-effect forest, MMSE forest, and
correlation-heatmap battery is also routed into the corresponding domain
`figures/statistical_battery/` directories.

Every domain writes a `manifest.json`, metric tables, figures, and a log or
report where applicable. `.pipeline/state/` only records orchestration
fingerprints. Timestamped consolidated stdout/stderr logs are written to
`pipeline_logs/`.

Generated paths are ignored by Git. Original data remain read-only inputs.
