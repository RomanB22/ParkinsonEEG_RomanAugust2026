# Statistical methods for the MOCA association analysis

## Purpose and inferential scope

This pipeline tests cross-sectional associations between Montreal Cognitive
Assessment (MOCA) scores and subject-level EEG features among participants with
Parkinson disease (PD). It does not estimate longitudinal disease progression,
causality, or out-of-sample clinical prediction.

The unit of inference is the participant. Electrode observations and detected
bouts are summarized within each participant before correlations are
calculated; they are never treated as independent patients.

The implementation is in [`statistics.py`](statistics.py), feature construction
is in [`features.py`](features.py), and all prespecified settings are in
[`config.json`](config.json).

## Aperiodic exponent construction and diagnostic-group comparison

The scale-free workflow calculates PSDs over 1–50 Hz and fits fixed-mode
specparam models over 4–35 Hz. For each
participant it exports the aperiodic exponent at every one of the 60 electrodes
shared by all participants. The primary exponent is the arithmetic mean across
those electrodes, giving one value per participant.

Every electrode fit is audited against four configurable criteria: R² at least
0.90, log10-power MAE at most 0.15, maximum absolute signed residual at most
0.75 log10 units, and exponent within 0–3. All 8,940 fits remain visible. The
QC sensitivity averages only passing electrodes and includes a participant
only when at least 80% (48/60) electrodes pass. Fixed-mode 3–35, 4–40, and
3–40 Hz fits retain identical peak settings and assess fitting-range stability;
4–35 Hz remains primary. Delta is retained in descriptive PSD analyses but is
not used to estimate the aperiodic exponent.

The Control-versus-PD analysis uses all 149 participants. Its primary model is:

```text
aperiodic exponent = beta0 + betaPD I(PD) + betaage age + betasex sex_male + error
```

`betaPD` is the adjusted PD-minus-Control difference in native exponent units.
Inference uses an HC3 heteroskedasticity-robust standard error, two-sided
p-value, and 95% confidence interval. The table also reports group means,
standard deviations and medians, the raw mean difference, Welch's unequal-
variance t test, Mann–Whitney U test, and Hedges' g. These latter quantities are
unadjusted sensitivity/descriptive summaries. The primary all-fit group
comparison and QC-qualified sensitivity are jointly BH-corrected as a
two-analysis aperiodic family.

The exponent–MOCA question is separate: it uses PD participants only and the
same partial Spearman procedure described below. The all-fit and QC-qualified
exponent tests form a two-feature family for BH-FDR. Electrode-level
exponent–MOCA maps are secondary localization results.

## Cohort, outcome, and covariates

- Primary cohort: PD participants only.
- Outcome: MOCA score.
- Descriptive cognitive status: cognitive impairment for MOCA < 26 and
  cognitively normal for MOCA 26–30. This category is exported and marked in
  figures, but the association models retain continuous MOCA to avoid losing
  information at an arbitrary dichotomy.
- Covariates: age in years and recorded sex.
- Sex coding: `sex_male = 1` for `M` and `sex_male = 0` for `F`.
- Minimum sample size: 30 complete participants for a reported estimate.
- Missing data: feature-wise complete cases; no values are imputed.

Every test uses one row per participant containing the EEG feature, MOCA, age,
and sex. A participant is excluded from a particular test only if one of those
four values is missing. The output column `n_subjects` reports the resulting
sample size.

## Primary age/sex-adjusted correlation

The primary statistic is a **partial Spearman correlation adjusted for age and
sex**. It is implemented by rank-transforming and residualizing both MOCA and
the EEG feature.

For one EEG feature, let:

- `X` be the EEG feature across participants;
- `Y` be MOCA;
- `A` be age;
- `S` be the binary sex indicator.

### 1. Rank every variable

The pipeline calculates average ranks, so tied values receive their mean rank:

```text
RX = rank(X)
RY = rank(Y)
RA = rank(A)
RS = rank(S)
```

Ranking the binary sex indicator is an affine recoding of the same two groups,
so its residualization effect is equivalent to using the 0/1 indicator when
both groups are represented.

### 2. Residualize ranked EEG and ranked MOCA

The covariate design matrix contains an intercept, ranked age, and ranked sex:

```text
Z = [1, RA, RS]
```

Two ordinary least-squares regressions are fitted:

```text
RX = Z βX + eX
RY = Z βY + eY
```

Equivalently, with `Z+` denoting the least-squares pseudoinverse and
`HZ = Z Z+` denoting the projection onto the covariate space:

```text
eX = (I - HZ) RX
eY = (I - HZ) RY
```

Both sides are adjusted. Residualizing MOCA alone, or the EEG feature alone,
would not implement the partial correlation used here.

### 3. Correlate the residual ranks

The adjusted coefficient is the Pearson correlation between the two residual
vectors:

```text
partial Spearman rho = cor(eX, eY)
```

Interpretation is conditional and monotonic: a positive value means that
participants with a higher EEG-feature rank than expected for their age and
sex also tend to have a higher MOCA rank than expected for their age and sex.
A negative value indicates the opposite direction.

This procedure controls additive rank-scale associations with age and sex. It
does not automatically control education, medication state, disease duration,
age-by-sex interactions, or other unavailable confounders.

## P-value for the adjusted coefficient

For `n` complete participants, adjusted correlation `r`, and effective
covariate rank `k`, the pipeline calculates:

```text
t = r × sqrt((n - k - 2) / (1 - r²))
degrees of freedom = n - k - 2
```

The p-value is the two-sided tail probability from the Student t distribution.
With non-collinear age and sex covariates, `k=2`, so the usual degrees of
freedom are `n-4`. The implementation calculates `k` from the numerical rank of
the design matrix instead of assuming it is always two.

The raw p-value tests a zero partial rank correlation for that individual
feature. It is not the final multiplicity-adjusted decision criterion.

## Unadjusted sensitivity analysis

Every subject-level feature also receives a conventional unadjusted Spearman
correlation:

```text
unadjusted Spearman rho = cor(rank(X), rank(Y))
```

These rows have `method = spearman_unadjusted`. They show how strongly the
result depends on age/sex adjustment but are not the primary inference. Primary
rows have `method = partial_spearman_age_sex`.

## Bootstrap confidence intervals

Uncertainty is quantified with a deterministic participant-level percentile
bootstrap:

1. Sample `n` participant indices with replacement.
2. Keep each sampled participant's EEG feature, MOCA, age, and sex together.
3. Recalculate ranks, both residual regressions, and the correlation from the
   resampled data.
4. Repeat 2,000 times for the reporting run.
5. Use the 2.5th and 97.5th percentiles as the 95% confidence interval.

The random seed is prespecified and receives deterministic feature- and
method-specific offsets. At least 80% of requested bootstrap estimates must be
finite; otherwise the interval is reported as missing. Output columns record
both requested and valid bootstrap counts.

The bootstrap interval describes effect-size uncertainty. It can exclude zero
while the FDR-adjusted test remains non-significant because the interval is not
itself adjusted for all tested features.

## Multiple-comparison correction

Benjamini-Hochberg false-discovery-rate (BH-FDR) correction is applied at
`alpha=0.05`. Adjusted and unadjusted methods are corrected separately.

### Primary 52-feature analysis

BH-FDR is applied within each prespecified feature family:

| Family | Tests per method |
|---|---:|
| All-fit and QC-qualified aperiodic exponent | 2 |
| Broadband regular ordinal quantities | 3 |
| Band-resolved regular ordinal quantities | 15 |
| Bout properties | 20 |
| Within-bout regular ordinal quantities | 12 |

### Fit-QC bout sensitivity

The fit-QC sensitivity is a separate 32-feature analysis restricted to
subjects with at least 48/60 passing specparam fits. Only passing-electrode
values contribute to the subject mean. It contains 20 band-resolved bout
properties and 12 within-bout H/C/F quantities. Partial Spearman correlations
use the same age/sex rank-residualization and subject bootstrap as the primary
analysis. BH-FDR is controlled separately within its 20-feature bout-property
family and 12-feature within-bout ordinal family, and not pooled with the
primary 52-feature analysis. Broad 5–15 Hz is generated upstream for descriptive
plots and QC but excluded from all association tests.

### Separate embedding-dimension analyses

Each embedding dimension is a separate 102-feature analysis block:

```text
6 signal scopes × 17 ordinal quantities = 102 tests per D and method
```

The seventeen quantities are regular entropy, complexity, and Fisher information,
plus Rényi entropy and complexity at `alpha=0.1`, `0.5`, `0.9`, `1.1`, `2`, `5`, and `10`. BH-FDR is
applied separately within `ordinal_D3`, `ordinal_D4`, `ordinal_D5`, and
`ordinal_D6`.

The four D blocks are stored as separate feature matrices and are not
concatenated into one model input. They are nevertheless statistically
dependent because they contain the same participants and are derived from the
same EEG recordings. D=6 is the primary ordinal block; choosing the best result
from D=3–6 after inspecting all blocks is exploratory and is not licensed by
the within-D corrections.

### Electrode-level localization

Electrode correlations are secondary spatial analyses. For each EEG feature,
the 60 electrode p-values are BH-corrected within that feature. These results
use `p_fdr_bh_within_feature` and `fdr_reject_within_feature`; they should not
be interpreted as independent participant-level replications.

## Significance decision rule

For a primary subject-level association, use the row satisfying:

```text
method == "partial_spearman_age_sex"
fdr_reject == True
p_fdr_bh < 0.05
```

The columns have the following roles:

| Column | Meaning |
|---|---|
| `estimate` | Adjusted or unadjusted Spearman coefficient |
| `ci_lower`, `ci_upper` | Participant-bootstrap confidence interval |
| `p_value` | Raw two-sided p-value for that feature |
| `p_fdr_bh` | BH-FDR-adjusted p-value within the stated family/D block |
| `fdr_reject` | Formal subject-level significance indicator |
| `n_subjects` | Complete participants used for that test |

A raw `p_value < 0.05` is described as nominal or suggestive when
`fdr_reject=False`; it is not reported as statistically significant after the
prespecified multiplicity correction.

## Reproducibility

Run the documented analysis with:

```bash
bash quantitative_behavioral/prepare_dimension_sensitivity.sh
bash quantitative_behavioral/run_quantitative_behavioral.sh --overwrite
```

The complete subject-level results are written to
`processed/metrics/subject_level_correlations.csv` and
`processed/metrics/dimension_sensitivity_correlations.csv`. Each D-specific
matrix and result table is stored under `processed/metrics/dimensions/D<value>/`.
The generated `processed/manifest.json` records the configuration, software
versions, input manifests, feature counts, FDR scopes, and bootstrap settings.
