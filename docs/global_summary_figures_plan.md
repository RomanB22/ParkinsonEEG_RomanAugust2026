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
- broadband aperiodic offset;
- alpha entropy.

Cell color shows the signed median standardized group difference across
electrodes. Each cell also reports the number of FDR-significant electrodes,
for example `63/66`. This is the main overview figure: it shows which findings
replicate and which are dataset-dependent.

### 2. Replicated spectral topographies

`replicated_spectral_topomaps.png`

Three dataset rows and two feature columns:

- theta relative power;
- beta relative power.

All panels use a common effect scale. FDR-significant electrodes are outlined.
The figure uses a simple schematic electrode layout so that the broad spatial
distribution is visible without presenting all of the detailed topomap pages.

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
