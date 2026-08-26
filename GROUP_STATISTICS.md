# PD-versus-Control group statistics

PSD, broadband and band-resolved ordinal quantities, scale-free/periodic/bout
properties, independent bycycle burst properties, and within-bout ordinal
quantities use one shared inference policy.
The subject is always the independent biological unit; electrodes and bouts are
never treated as independent participants.

## Primary subject-level inference

Each metric is first aggregated across the electrodes shared by every analyzed
subject. PSD relative power uses the median, matching its descriptive violin
plot. Ordinal, scale-free, bout, and within-bout ordinal quantities use the
arithmetic mean, matching their existing subject summaries.

- Full cohort: `metric ~ PD + age + sex`, fit by ordinary least squares with
  HC3 heteroskedasticity-robust standard errors. The reported PD coefficient is
  PD minus Control at equal age and sex.
- Demographically matched cohort: a paired t test uses `match_pair_id`. A paired
  Wilcoxon p-value is saved as a sensitivity result. Pairing is not discarded.
- Descriptive columns also report group means, medians, standard deviations,
the unadjusted mean difference, Hedges g, Welch's test, and Mann–Whitney's
test. These are not substituted for the primary model.

The primary p-values receive Benjamini–Hochberg FDR correction across all
metric-by-band tests in their declared analysis domain. Broadband ordinal,
band-resolved ordinal, aperiodic, periodic/bout, PSD, and within-bout analyses
are separate, scientifically interpretable domains. Independent bycycle burst
tests are also one separate sensitivity domain and are not pooled with eBOSC
p-values. The overlapping 5–15 Hz
band remains in descriptive plots and tables but is excluded from formal group
inference because it overlaps the canonical bands.

## Electrode-wise inference

An identical model is fit separately at every shared electrode. These tests are
spatial localization analyses, not extra independent subjects. Two corrected
p-values are saved:

1. `primary_p_fdr_bh_within_feature`: BH across electrodes for one
   metric/band, useful as secondary localization.
2. `primary_p_fdr_bh_domain`: BH across every electrode-by-metric-by-band test
   in the domain. `primary_fdr_reject_domain` is the conservative formal flag
   used for rings on the statistical topomaps.

Topomaps show PD-minus-Control standardized effects and `-log10(domain q)`.
For the full cohort, the plotted effect is the age/sex-adjusted PD coefficient
divided by the outcome standard deviation; matched maps use paired Cohen dz.
The sign convention is consistent throughout: positive values mean higher in
PD. Confidence intervals, raw p-values, both q-values, sample sizes, effect
sizes, and model descriptions remain available in CSV form even when no result
passes FDR.

## Output locations

- `psd_analysis/processed/metrics/group_{subject,electrode}_statistics.csv`
- `ordinal_analysis/processed/metrics/group_*_statistics_*.csv`
- `scale_free_analysis/processed/metrics/group_*_statistics_*.csv`
- `bycycle_burst_analysis/processed/metrics/group_{subject,electrode}_statistics.csv`
- `bout_analyses/processed/metrics/group_{subject,electrode}_statistics.csv`

Matched results use the corresponding `processed_matched/` directories. Figures
are under each analysis folder's `figures/group_statistics/` directory.
