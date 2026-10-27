# Cross-dataset interpretation of resting-state EEG bout findings

## Central interpretation

The most coherent story across the two resting-state EEG datasets is:

> Parkinson's disease is associated with a greater burden and persistence of
> resting-state theta bouts, whereas cognitive status is more closely related
> to the temporal organization within those bouts.

Both datasets compare healthy controls with participants with Parkinson's
disease. The primary dataset contains 100 PD participants and 49 healthy
controls. The independent `ds002778` dataset contains 15 PD participants,
recorded both OFF and ON medication, and 16 healthy controls.

## Summary of the evidence

| Finding | Primary dataset | `ds002778` | Interpretation |
|---|---|---|---|
| Theta oscillatory occupancy | Higher in PD and FDR-significant in both the full and demographically matched cohorts | Higher in PD OFF and PD ON, with large standardized effects but without FDR significance | Strongest directionally replicated disease-associated effect |
| Theta mean bout duration | Longer in PD and FDR-significant in both the full and matched cohorts | Longer in PD OFF and PD ON, but without FDR significance | Supports more persistent theta activity in PD |
| Theta within-bout entropy and complexity | Higher values were associated with better MOCA performance within PD (`rho` approximately 0.33, FDR `q=0.0056`) | Associations with MMSE were positive within PD but nonsignificant | Candidate cognitive marker, but not yet independently replicated |
| Theta within-bout Fisher information | No consistent cognitive or diagnostic relationship | No MMSE relationship | Does not currently appear to be a robust marker |
| Beta bouts per minute | Lower in PD, with the strongest evidence in the full cohort and weaker evidence after demographic matching | Lower in PD OFF but approximately equal to HC in PD ON; nonsignificant | Possible medication-sensitive secondary effect |
| Delta entropy, complexity, and Fisher information versus motor severity | Not tested against Total UPDRS | No significant OFF, ON, or paired-change association with Total UPDRS | No evidence that these delta ordinal quantities track motor severity in the replication dataset |

## Disease-associated theta-bout burden

The most reproducible physiological pattern is an increase in theta-bout
expression in PD. In the primary dataset, PD participants had both greater
theta oscillatory occupancy and longer theta bouts. These effects survived
FDR correction in the full cohort and in the demographically matched cohort.

The independent `ds002778` dataset showed the same direction and relatively
large standardized effects in both PD medication conditions. The effects did
not survive FDR correction in that dataset, which contains only 15 PD and 16
healthy-control participants. Its results therefore provide directional
support rather than formal independent statistical replication.

Together, the datasets suggest that resting-state PD physiology is
characterized by more frequent or persistent entry into theta-dominated
oscillatory states.

## Within-bout organization and cognition

Theta within-bout entropy and statistical complexity did not behave as robust
HC-versus-PD diagnostic markers. Their adjusted group differences were small
or sensitive to cohort selection and did not consistently survive FDR
correction.

Instead, these quantities were related to cognitive status within the large PD
cohort. Higher theta within-bout entropy and complexity were associated with
better MOCA performance after adjustment for age and sex. Both correlations
were approximately `rho=0.33` and survived FDR correction (`q=0.0056`). The
associations also survived the duration-QC and fit-QC sensitivity analyses.

The demographically matched PD subset retained nearly the same effect size
(`rho` approximately 0.315), although it did not survive FDR correction after
the PD sample decreased from 100 to 49. This pattern is more consistent with a
loss of power than with a reversal of the effect.

No significant corresponding association was observed with MMSE in
`ds002778`. This is not a strong contradiction because the primary PD cohort
spans MOCA scores from 9 to 30, whereas `ds002778` contains only 15 PD
participants with MMSE scores restricted to 26 through 30. The restricted
MMSE range provides little variation with which to test cognitive severity.
The PD entropy and complexity estimates in `ds002778` nevertheless remained
positive, matching the direction in the primary cohort.

Thus, theta within-bout entropy and complexity should be presented as
promising cognitive correlates that require validation in an independent
cohort with a broader distribution of cognitive impairment. Entropy and
complexity are also mathematically related properties of the same ordinal
distribution and should not be interpreted as fully independent biological
signals.

## Medication-state findings

The paired OFF-versus-ON design in `ds002778` helps distinguish stable
disease-associated features from state-sensitive EEG quantities. Dopaminergic
medication significantly reduced theta within-bout entropy, complexity, and
Fisher information relative to the OFF condition. In contrast, theta
occupancy and duration remained elevated and beta bout rate moved numerically
toward the healthy-control level.

This pattern suggests that:

- theta-bout burden may be a relatively stable PD-associated physiological
  feature;
- within-bout ordinal organization may be sensitive to medication or current
  physiological state; and
- beta event rate may also be medication-sensitive, although its evidence is
  less consistent across datasets.

The medication-related reduction in entropy or complexity must not be
interpreted as cognitive worsening. MMSE was participant-level and did not
vary between OFF and ON sessions, so the analysis does not test an acute
cognitive response to medication.

## Total UPDRS and delta ordinal structure in `ds002778`

Total UPDRS is a session-specific motor-severity measure in this dataset. All
15 PD participants have scores in both states: 20–58 OFF medication and 16–54
ON medication. Healthy controls were excluded because they do not have a
comparable session-specific Total UPDRS measure. The primary correlations are
partial Spearman correlations adjusted for age and sex. OFF and ON models use
same-session EEG and Total UPDRS values; the change model correlates paired
ON−OFF EEG changes with paired ON−OFF Total UPDRS changes. The reported `q`
values use Benjamini–Hochberg FDR correction within each feature family and
model.

Delta entropy, complexity, and Fisher information refer here to the regular
1–4 Hz ordinal-band quantities. They are not within-bout estimates because
the eBOSC bout analysis starts at theta (4 Hz).

| Delta quantity | PD OFF: rho, p, q | PD ON: rho, p, q | ON−OFF change: rho, p, q |
|---|---:|---:|---:|
| Permutation entropy | −0.110, 0.721, 0.727 | 0.120, 0.696, 0.867 | −0.350, 0.241, 0.927 |
| Statistical complexity | −0.110, 0.721, 0.727 | 0.129, 0.674, 0.867 | −0.321, 0.285, 0.927 |
| Fisher information | 0.197, 0.520, 0.727 | −0.187, 0.542, 0.867 | 0.167, 0.585, 0.927 |

None of these nine focused delta tests was significant before or after FDR
correction. The negative change correlations for delta entropy and complexity
are moderate in magnitude, but with only 15 paired observations they are too
imprecise to support a motor-severity claim. Their similarity also reflects
that entropy and complexity are derived from the same ordinal distribution.

The previously selected bout quantities were also tested against Total UPDRS:

| Focused quantity | PD OFF: rho, p, q | PD ON: rho, p, q | ON−OFF change: rho, p, q |
|---|---:|---:|---:|
| Theta within-bout entropy | 0.250, 0.410, 0.702 | 0.056, 0.855, 0.855 | 0.381, 0.199, 0.391 |
| Theta within-bout complexity | 0.283, 0.350, 0.699 | 0.087, 0.778, 0.848 | 0.378, 0.203, 0.391 |
| Theta within-bout Fisher information | 0.129, 0.674, 0.809 | −0.180, 0.556, 0.790 | 0.256, 0.399, 0.479 |
| Theta oscillatory occupancy | 0.329, 0.272, 0.863 | 0.324, 0.280, 0.602 | −0.167, 0.586, 0.823 |
| Theta mean bout duration | 0.048, 0.877, 0.877 | 0.230, 0.449, 0.672 | −0.069, 0.823, 0.823 |
| Beta bouts per minute | −0.139, 0.651, 0.863 | 0.031, 0.920, 0.956 | −0.263, 0.385, 0.769 |

None of these 18 focused bout tests survived FDR correction. Thus, the most
coherent interpretation is that the focused EEG quantities characterize PD
and medication-related physiology more clearly than they track concurrent
motor severity in this small cohort. A broad, exploratory screen did yield
FDR-positive associations for ON−OFF change in aperiodic fit-QC fraction and
gamma within-bout entropy; the former is a quality-control measure and the
latter was not a prespecified focus, so neither changes the focused delta
conclusion.

## Integrated model

The combined findings motivate separating two components of oscillatory
activity:

1. **Bout quantity and persistence:** Theta occupancy and duration provide the
   clearest disease-associated signal. They differentiate PD from healthy
   controls but do not show a reliable linear relationship with MOCA or MMSE.
2. **Within-bout temporal organization:** Theta entropy and complexity are not
   stable diagnostic markers, but they may index cognitive status within PD
   and are sensitive to medication state.

Beta bouts per minute may provide a secondary third component. Beta bout rate
was lower in PD in the primary dataset, but the evidence weakened after
matching and was not reproduced statistically in `ds002778`. The numerical
increase from PD OFF to PD ON suggests possible medication sensitivity, but
this should remain an exploratory interpretation.

## Proposed manuscript conclusion

> Across two resting-state EEG cohorts, Parkinson's disease was characterized
> primarily by increased theta-bout occupancy and duration, suggesting a shift
> toward more persistent low-frequency oscillatory states. The internal
> ordinal structure of theta bouts was less useful for distinguishing PD from
> healthy controls but was associated with cognitive performance in the larger
> PD cohort. Medication-state effects in the independent cohort further
> suggest that within-bout complexity is state-sensitive, whereas theta-bout
> burden may represent a more stable disease-associated alteration. These
> results motivate separating the quantity of pathological oscillatory
> activity from its within-bout temporal organization.

## Interpretation limits

- The primary cognitive analysis is cross-sectional and cannot establish
  cognitive decline, progression, or causality.
- The two datasets use different cognitive instruments: MOCA in the primary
  cohort and MMSE in `ds002778`.
- The very narrow MMSE range and small `ds002778` sample substantially limit
  independent cognitive replication.
- Medication state, education, and disease duration are unavailable as
  covariates in the primary dataset.
- The datasets differ in electrode count, recording duration, acquisition
  hardware, and other protocol details.
- Directional agreement in a small cohort should not be described as formal
  replication when its corrected tests are nonsignificant.
- The results support physiological interpretation and hypothesis generation,
  not diagnostic classification or individual-level clinical prediction.

## Recommended hierarchy of claims

1. **Primary cross-dataset claim:** PD is associated with increased
   resting-state theta-bout occupancy and duration.
2. **Promising cognitive claim:** Higher theta within-bout entropy and
   complexity are associated with better cognitive performance in the large
   PD cohort.
3. **State-sensitivity claim:** Within-bout ordinal structure is modulated by
   medication and may not represent a fixed disease trait.
4. **Secondary exploratory claim:** Reduced beta bout rate may be related to PD
   and medication state, but its cross-dataset evidence is weaker.
