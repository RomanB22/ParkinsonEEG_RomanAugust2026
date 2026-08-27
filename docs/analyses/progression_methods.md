# Disease-severity association methods

## Cohort and interpretation

The analysis is restricted to participants labeled PD with complete UPDRS,
MOCA, age, and sex metadata. UPDRS is the prespecified primary motor-severity
axis and MOCA is a complementary cognitive axis. Because the dataset is
cross-sectional, this is a disease-severity analysis—not a longitudinal rate
of progression, prognostic model, or causal analysis.

Cognitive impairment is defined descriptively as MOCA < 26 and cognitively
normal as MOCA 26–30. The derived status is exported and the boundary is shown
in MOCA figures. Inferential associations continue to use continuous MOCA;
the status does not create an additional hypothesis-test family.

## Electrode scope

Every feature is recalculated from the complete electrode intersection across
the analysis cohort. The ordered list is read from the D=6, tau=1 ordinal
`electrode_sets.json` provenance file. The pipeline fails if any requested
subject, source, or band does not contain that complete set. There are
currently 60 electrodes in both canonical cohorts.

Ordinal, aperiodic, bout, and within-bout ordinal features are averaged across
the shared electrodes. Relative PSD band power uses the median, matching the
primary PSD analysis. A QC-qualified exponent is reported only when at least
80% of shared electrodes pass formal specparam fit QC (48/60 in the current
cohorts).

The primary ordinal block is D=6, tau=1. It contains regular H/C/F and Rényi
entropy/complexity for alpha=0.1, 0.5, 0.9, 1.1, 2, 5, and 10, both broadband
and in the five canonical ordinal bands. PSD, bout properties, and within-bout
H/C/F use their canonical non-overlapping bands.

## Statistical model

For each EEG feature and outcome, the primary estimate is an age/sex-adjusted
partial Spearman correlation:

1. Rank the EEG feature, clinical outcome, age, and binary sex covariate using
   average ranks for ties.
2. Regress the ranked EEG feature on ranked age and sex and retain residuals.
3. Regress the ranked clinical outcome on the same covariates and retain
   residuals.
4. Pearson-correlate the two residual vectors. The resulting coefficient is
   the partial Spearman rho.

The p-value uses the residual-correlation t statistic and its covariate-adjusted
degrees of freedom. A deterministic subject bootstrap supplies percentile 95%
confidence intervals. Unadjusted Spearman correlations are saved as
sensitivity analyses.

For UPDRS, positive rho means the quantity increases with worse motor
severity. MOCA has the opposite clinical direction, so the output also contains
`progression_aligned_estimate = -rho` for MOCA. This sign alignment aids visual
comparison and is not a new statistical test or composite clinical score.

## Multiplicity

Benjamini–Hochberg FDR is controlled separately within each combination of:

- clinical outcome (UPDRS or MOCA);
- prespecified feature family (ordinal, PSD, aperiodic, bout, within-bout
  ordinal); and
- method (adjusted primary or unadjusted sensitivity).

Use `fdr_reject == True` and `p_fdr_bh < 0.05` for corrected significance in
that declared family. Raw p-values and all non-significant results remain in
the output to avoid selective reporting.

## Full and matched cohorts

The full analysis uses all PD participants. The matched sensitivity uses only
the PD participants retained by the canonical PD/Control demographic match.
The Control members and match-pair IDs do not enter a within-PD severity
correlation; therefore this analysis is not paired. Its value is sensitivity to
the smaller, demographically restricted PD sample.
