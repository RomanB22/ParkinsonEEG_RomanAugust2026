# Global Summary Figures Plan

The existing per-dataset figures remain available for detailed inspection. The
figures below are a smaller presentation set designed to communicate the main
conclusions quickly.

## Main figures

### 1. Cross-dataset replication summary

`cross_dataset_summary.png`

A matrix with datasets as columns and the main features as rows:

- theta relative power;
- theta burst count;
- theta burst occupancy;
- beta relative power;
- alpha entropy.

Each cell shows the direction of the PD–Control effect, the percentage of
tested electrodes that are FDR-significant, and the count behind that
percentage, for example `↑ 95%` and `63/66`. A companion bar shows how many
datasets have a broad effect (at least 50% of electrodes significant). This
makes replication and dataset dependence visible without mixing effect-size
units with significance counts.

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
Each dataset has a point estimate and an approximate 95% confidence interval
for the standardized PD–Control difference. A vertical zero line makes the
direction and consistency immediately visible.

### 4. Clinical replication

`theta_fisher_moca_replication.png`

Three side-by-side PD-only scatter plots showing theta Fisher information at
D=4 against MoCA. The panels use common axes and annotate sample size,
Spearman rho, and FDR q. This directly displays the replicated negative
association while avoiding a page of exploratory scatter plots.

### 5. Medication-state result

`medication_state_effects.png`

A compact forest plot for PD-ON minus PD-OFF differences in the principal
theta, beta, burst, and aperiodic features. Confidence intervals and a zero
line emphasize that this small cohort did not produce a clear medication
effect. This figure is useful as a secondary or supplementary figure.

### 6. Strongest UPDRS association

`updrs_strongest_association.png`

A PD-only scatter plot for the strongest saved unadjusted UPDRS association,
selected by the smallest within-dataset FDR q-value. In the current outputs
this is beta burst occupancy versus UPDRS-III in ds008768. The plot explicitly
labels the result as exploratory and single-dataset because it is not
replicated across cohorts.

### 7. Within-bout clinical associations

`within_bout_clinical_associations.png`

A four-panel figure showing the strongest saved within-bout association for
selected clinical outcomes. The current panels include the two FDR-significant
MoCA findings (primary and ds007526), plus the strongest available UPDRS and
MMSE findings. Significant results are green; the strongest UPDRS/MMSE panels
are gray because they do not survive FDR correction.

### 8. Cross-dataset participant distributions

`cross_dataset_distributions.png`

A three-row by five-column violin grid for the quantities in the cross-dataset
summary: theta relative power, theta burst count, theta burst occupancy, beta
relative power, and alpha entropy. Each row is one dataset and each cell shows
Control versus PD participant-level distributions with the raw participant
points overlaid. Stars mark participant-level Welch tests that survive
Benjamini–Hochberg FDR correction within the corresponding dataset and figure.

### 9. Theta H/C/F distributions at D=4

`theta_hcf_D4_distributions.png`

A three-row by three-column violin grid for theta entropy (H), complexity (C),
and Fisher information (F) at D=4. Each row is one dataset and each cell shows
Control versus PD participant-level distributions. Stars use the same
within-dataset Benjamini–Hochberg correction across the three displayed H/C/F
features.

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
