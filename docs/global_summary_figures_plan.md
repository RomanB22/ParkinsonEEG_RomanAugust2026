# Global Summary Figures Plan

The existing per-dataset figures remain available for detailed inspection. The
figures below are a smaller presentation set designed to communicate the main
conclusions quickly.

## Main figures

### 1. Cross-dataset replication summary

`cross_dataset_summary.png`

A matrix with the three PD–Control datasets plus a medication-state column and
the main features as rows:

- theta relative power;
- theta burst count;
- theta burst occupancy;
- beta relative power;
- alpha entropy.

Each cell shows the direction and percentage of significant electrodes. The
medication-state column summarizes the Control versus PD-OFF/PD-ON contrasts
and is explicitly not treated as a single PD group. A companion bar shows how
many columns have a broad effect (at least 50% of electrodes significant).

### 2. Replicated spectral topographies

`replicated_spectral_topomaps.png`

Two consistency maps show the number of independent datasets in which each
electrode is significant:

- theta relative power;
- beta relative power.

The maps directly answer where the strongest effects are replicated, rather
than repeating three very similar effect maps. The figure uses a simple
schematic electrode layout; the original detailed topomaps remain available
for precise electrode-level inspection.

### 3. Theta-burst effect sizes

`theta_burst_effects.png`

A horizontal forest plot for theta burst count, burst rate, and occupancy.
Each of the first three datasets has a point estimate and an approximate 95%
confidence interval for the standardized PD–Control difference. The
medication-state dataset adds separate PD-OFF–Control and PD-ON–Control
estimates. A vertical zero line makes the direction and consistency immediately
visible.

### 4. Clinical replication

`theta_fisher_moca_replication.png`

Four side-by-side PD-only scatter plots showing theta Fisher information at
D=4 against clinical score. The first three panels use MoCA; the
medication-state panel uses MMSE because MoCA is not available in that cohort.
The panels use common axes and annotate sample size and Spearman rho. FDR q is
shown for the first three datasets; the medication panel reports the minimum
group-wise q-value for PD-OFF and PD-ON. This displays the replicated tendency
without silently mixing clinical scales.

### 5. Medication-state result

`medication_state_effects.png`

A compact forest plot for PD-ON minus PD-OFF differences in the principal
theta, beta, burst, and aperiodic features. Confidence intervals and a zero
line emphasize that this small cohort did not produce a clear medication
effect. This figure is useful as a secondary or supplementary figure.

### 6. Theta Fisher–alpha power association

`theta_fisher_alpha_power_association.png`

Four side-by-side participant-level scatter plots showing the association
between theta Fisher information at D=4 and alpha relative power. The first
three panels show Control and PD; the medication-state panel distinguishes
Control, PD-OFF, and PD-ON. Each panel reports the sample size, Spearman rho,
and unadjusted p-value.

`theta_fisher_absolute_theta_power_association.png` repeats this analysis with
absolute theta power, helping distinguish a relationship with theta amplitude
from one driven by relative-power normalization.

### 7. Alpha relative-power distributions and clinical association

`alpha_relative_power_distributions.png` is a four-row violin plot comparing
alpha relative power across groups, with corrected significance brackets.
`alpha_power_moca_replication.png` shows alpha relative power versus MoCA in
the first three datasets and versus MMSE in the medication-state cohort. The
scatter panels report Spearman rho and FDR q-values.

### 8. Strongest UPDRS association

`updrs_strongest_association.png`

A PD-only scatter plot for the strongest saved unadjusted UPDRS association,
selected by the smallest within-dataset FDR q-value. In the current outputs
this is beta burst occupancy versus UPDRS-III in ds008768. The plot explicitly
labels the result as exploratory and single-dataset because it is not
replicated across cohorts.

### 9. Within-bout clinical associations

`within_bout_clinical_associations.png`

A four-panel figure showing the strongest saved within-bout association for
selected clinical outcomes. The current panels include the two FDR-significant
MoCA findings (primary and ds007526), plus the strongest available UPDRS and
MMSE findings. Significant results are green; the strongest UPDRS/MMSE panels
are gray because they do not survive FDR correction.

### 10. Cross-dataset participant distributions

`cross_dataset_distributions.png`

A four-row by five-column violin grid for the quantities in the cross-dataset
summary: theta relative power, theta burst count, theta burst occupancy, beta
relative power, and alpha entropy. The first three rows show Control versus PD;
the medication-state row shows Control, PD-OFF, and PD-ON. Stars mark
participant-level Welch tests that survive
Benjamini–Hochberg FDR correction within the corresponding dataset and figure.

### 11. Theta H/C/F distributions at D=4

`theta_hcf_D4_distributions.png`

A four-row by three-column violin grid for theta entropy (H), complexity (C),
and Fisher information (F) at D=4. The first three rows show Control versus PD;
the medication-state row shows Control, PD-OFF, and PD-ON. Stars use the same
within-dataset Benjamini–Hochberg correction across the three displayed H/C/F
features.

### 12. Side-by-side PSD comparison

`psd_control_pd_side_by_side.png`

A horizontal composition of the saved full PSD comparisons for all four
datasets. The first three panels show Control and PD; the medication-state
panel shows Control, PD-OFF, and PD-ON. Each panel retains the median curves
with 95% bootstrap confidence bands and the original frequency range.

### 13. Participant age distributions

`age_group_histograms.png`

Four side-by-side histograms compare participant ages in the Primary, ds007526,
ds008768, and medication-state datasets. The first three panels separate
Control and PD counts; the medication-state panel separates Control, PD-OFF,
and PD-ON. All panels use common five-year bins, while y-axis limits adapt to
cohort size so the smaller medication-state cohort remains readable.

## Design rules

- Keep detailed band-by-band violins, entropy planes, and contrast topomaps as
  supplementary outputs.
- Use the same dataset colors in every figure.
- Use common axes and color scales whenever panels are comparable.
- Report effect sizes and uncertainty rather than displaying many raw p-values.
- Show FDR-significant electrodes as annotations or outlines, not as separate
  dense significance panels.
- Treat the clinical plots as associations, not causal effects.
- Do not include knee or knee-frequency parameters in the summary figures.
