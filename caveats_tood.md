# LAVI-defined frequency bands: proposal, novelty, and caveats

## Proposed idea

Parkinson's disease (PD) may change the frequency location of an oscillatory
process without changing its broader rhythmicity class. Therefore, instead of
assuming that identical numerical frequency ranges represent identical neural
processes in PD and controls, we can compare **rhythmicity-defined functional
homologs**: corresponding high- or low-LAVI intervals that may occupy somewhat
different frequency ranges in the two populations.

A suitable formulation is:

> Parkinson's disease may alter the frequency location of a rhythmic process
> without changing its functional rhythmicity class. We therefore compare
> homologous rhythmicity-defined intervals rather than assuming that identical
> nominal frequency ranges represent identical neural processes in PD and
> controls.

These intervals should not be described simply as "the same frequency band."
Two intervals are plausible functional homologs only when they share more than
an absolute LAVI value. Matching should consider:

- High- versus low-LAVI direction.
- Ordinal location in the rhythmicity spectrum.
- A corresponding landmark, such as the high-LAVI alpha peak.
- Preferably, similar scalp topography and behavioral or clinical associations.

Equal LAVI values at two unrelated frequencies do not by themselves establish
physiological equivalence.

## Novelty assessment

The general idea of using LAVI and ABBA to define spectral intervals from
rhythmicity rather than conventional power-based frequency boundaries is not,
by itself, novel. The foundational LAVI/ABBA study already describes automated
high- and low-rhythmicity band-border detection at the individual level and
comparison of rhythmic architecture across individuals and datasets:

- [Universal rhythmic architecture uncovers two modes of neural dynamics](https://www.nature.com/articles/s41467-026-73553-8)

Individualized frequency bands based on empirical spectral landmarks also
predate LAVI, particularly individual alpha frequency:

- [Toward a reliable, automated method of individual alpha frequency quantification](https://pubmed.ncbi.nlm.nih.gov/29357113/)

The potentially novel contribution is the **PD-specific, multi-cohort
application**, especially if it:

1. Identifies conserved and disease-shifted rhythmicity landmarks across four
   scalp-EEG datasets.
2. Treats spectral slowing as a displacement or deformation of rhythmic
   architecture rather than only a redistribution of power between fixed
   bands.
3. Separates trait-like PD effects from medication-state effects.
4. Demonstrates external, leave-one-dataset-out reproducibility.
5. Shows clinical or behavioral relevance beyond conventional bands.

PD-related alpha slowing is already documented, including reduced peak alpha
frequency despite conventional definition of alpha as 8--13 Hz:

- [Peak alpha frequency and alpha power spectral density as vulnerability markers of cognitive impairment in Parkinson's disease](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2025.1575815/full)

Thus, the defensible novelty claim is not merely "individualized bands." A
stronger framing is:

> A multi-cohort, LAVI-derived atlas of conserved and disease-shifted
> rhythmicity landmarks in Parkinson's disease.

## Evidence in the current results

The group-mean ABBA results in
`outputs/rhythmicity/statistics/abba_group_mean_segments.csv` already suggest a
reproducible but shifted high-LAVI alpha interval:

- Control intervals begin at approximately 7.0--7.9 Hz and peak at
  approximately 9.3--9.9 Hz.
- PD and PD-OFF intervals begin at approximately 5.9--6.6 Hz and peak at
  approximately 8.3--9.3 Hz.
- Their upper boundaries remain approximately 11.7--12.3 Hz.

This is compatible with a leftward displacement or deformation of a conserved
rhythmicity structure. It should not yet be interpreted as proof of a new
frequency-band standard.

## Can the four datasets define new PD bands?

The four datasets can support a **candidate multi-cohort PD rhythmicity atlas**,
but not yet a universal or clinical standard. A standard would require
prospective independent validation, test--retest reliability, robustness to
recording conditions, and evidence that the new definitions improve inference
over conventional bands.

The datasets should not simply be concatenated before calculating one grand
mean profile because:

- The sample sizes are unequal, so the largest dataset would dominate.
- Site, hardware, montage, preprocessing, vigilance, and cohort composition can
  be confounded with diagnosis.
- PD-ON and PD-OFF observations from the same person are repeated measures, not
  independent participants.
- ABBA applied to a grand-mean LAVI profile is not mathematically equivalent to
  averaging participant-level ABBA boundaries.
- A mean boundary can conceal multimodal or heterogeneous participant-level
  intervals.

Controls and PD participants can have separate **descriptive atlases**, but
using separate group-specific bands for a direct group comparison changes the
frequencies being measured and therefore changes the estimand.

## Recommended derivation workflow

1. Compute the LAVI profile and ABBA intervals separately for every
   participant.
2. Use the participant as the statistical unit. Average repeated recordings as
   appropriate before cross-sectional inference.
3. Use PD-OFF as the primary unmedicated disease-state definition; retain PD-ON
   as a paired medication analysis or validation condition.
4. Match intervals across participants using LAVI direction, ordinal position,
   and an alpha rhythmicity anchor rather than conventional-band overlap alone.
5. Estimate start frequency, end frequency, and peak frequency on a log-frequency
   scale using a hierarchical model or random-effects meta-analysis, with
   dataset/site modeled explicitly.
6. Report both the central estimate and between-participant/between-dataset
   prediction intervals. Do not report only a mean boundary.
7. Use balanced or random-effects cohort weighting rather than allowing the
   largest cohort to determine the result.
8. Perform leave-one-dataset-out validation: derive and freeze the atlas using
   three datasets, then evaluate it in the fourth.
9. Compare candidate and conventional bands on reproducibility, test--retest
   stability, PD--control discrimination, medication sensitivity, clinical
   associations, and incremental predictive value.

Potential covariates or sources of heterogeneity include age, sex, cognitive
status, disease severity, medication state, vigilance/drowsiness, recording
duration, montage, reference, sampling rate, and site.

## Three analyses with different interpretations

### 1. Group-specific native bands

Derive a PD atlas and a control atlas separately. This is appropriate for
describing each population's native rhythmic architecture and formally testing
boundary or peak-frequency shifts.

It is generally not appropriate to filter each group using its own interval and
then interpret a difference in raw power, amplitude, or burst rate as though
both groups had been measured over the same frequencies.

### 2. Common reference bands

Derive intervals without using the diagnostic comparison and then apply the
same frozen limits to both groups. Two defensible choices are:

- Control-derived reference intervals applied to controls and PD.
- Diagnosis-blind intervals derived in an independent training sample or fold.

The existing analysis under `outputs/rhythmicity/control_band_qc/` implements
the important control-reference sensitivity: each dataset's control-derived
limits are applied unchanged to its control and PD groups.

### 3. Functional-homolog analysis

Use each participant's or population's corresponding rhythmicity interval and
compare quantities designed to remain interpretable after frequency shifts,
such as occupancy, cycle count, within-band normalized peak position, waveform
shape, or frequency-normalized event rate.

This analysis asks whether corresponding rhythmic processes differ even when
their physical frequency ranges have shifted. Boundary displacement should be
reported as a separate outcome rather than absorbed invisibly into the band
definition.

## Circularity and validation caveats

- Do not use diagnosis labels to select the most discriminating boundaries and
  then test those same boundaries on the same participants. Use nested
  cross-validation or an independent discovery/validation split.
- Do not use a clinical outcome to choose bands and then report an uncorrected
  association with that outcome in the same data.
- If group-specific intervals are used, clearly state that the comparison is
  between functional homologs, not identical spectral content.
- Predefine the rules for missing, split, merged, or direction-inconsistent
  ABBA intervals.
- Quantify how often the proposed homolog exists in each group and dataset;
  absence of a band may itself be biologically meaningful and should not be
  silently excluded.
- Validate that results are not driven by alpha slowing, drowsiness, unequal
  data duration, or differences in signal-to-noise ratio.

## Current surrogate limitation

The current rhythmicity configuration uses `surrogate_reps: 0`. Consequently,
the existing ABBA divisions are classifications relative to the LAVI-profile
median, not surrogate-supported statistically significant band boundaries.

Before presenting an atlas or proposed standard, the band-definition stage
should be repeated with an adequate number of appropriate phase-randomized or
IAAFT/pink-noise surrogate realizations. Approximately 200 or more realizations
would align with the foundational LAVI/ABBA approach, subject to a formal
precision and computation assessment.

## Recommended language for a manuscript

Prefer:

> We derived candidate rhythmicity-defined functional homologs and evaluated
> their reproducibility across four PD EEG cohorts.

or:

> We constructed and externally validated a candidate multi-cohort atlas of PD
> rhythmicity landmarks.

Avoid, without additional validation:

> We established new standard frequency bands for Parkinson's disease.

The strongest report would present conventional-band results, common
control-reference results, and functional-homolog results together. Agreement
across these analyses would provide substantially stronger evidence than any
single band-definition strategy.
