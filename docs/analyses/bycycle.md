# Independent bycycle burst detector

This folder is a sensitivity analysis for the aperiodic-relative eBOSC detector
in `src/analyses/scale_free/`. It detects bursts directly from consecutive-cycle
consistency using `bycycle`; no specparam curve, wavelet-power threshold, or
eBOSC mask is used to decide whether a cycle is bursting.

The prespecified criteria are amplitude fraction > 0.30, amplitude consistency
> 0.50, period consistency > 0.50, monotonicity > 0.80, and at least three
consecutive qualifying cycles. Each accepted four-second epoch is analyzed
separately, with a 0.5-second edge margin. Thresholds are fixed in
`config.json` and must not be tuned using PD-versus-Control significance.

All analyses use only electrodes shared by every included subject. Theta
(4–8 Hz), alpha (8–13 Hz), beta (13–30 Hz), and gamma (30–50 Hz) enter
detection, figures, and formal inference. Delta is excluded from bout detection
to match the 4–50 Hz aperiodic fit used by the primary eBOSC analysis.

Run the complete full-cohort stage with:

```bash
bash src/analyses/bycycle/run_bycycle_burst_analysis.sh --overwrite
```

This sensitivity analysis is not run by the default paper profile. To opt into
both full- and matched-cohort runs:

```bash
bash run_pipeline.sh analyses --profile full-qc
```

Important outputs are:

- `metrics/subject_electrode_band_metrics.csv`: detector results at the finest
  inferential unit retained by the pipeline.
- `metrics/subject_band_metrics.csv`: subject summaries across shared electrodes.
- `metrics/group_subject_statistics.csv`: primary age/sex-adjusted full-cohort
  tests, or paired tests when the matched configuration is used.
- `metrics/group_electrode_statistics.csv`: exploratory localization with
  domain-wide and within-feature BH-FDR columns.
- `metrics/detector_event_agreement.csv`: time-mask Dice/Jaccard agreement with
  eBOSC.
- `metrics/detector_metric_agreement.csv`: subject-summary Spearman agreement.
- `intermediate/episodes/`: independently detected bout boundaries.
- `intermediate/cycles/`: cycles classified as bursting, including all
  consistency features used by the decision.
- `figures/qc/`: example detections and detection coverage.
- `figures/agreement/`: eBOSC-versus-bycycle agreement figures.
- `figures/group_comparisons/group_<metric>.png`: one violin figure per
  detector quantity. Every point is one subject after averaging that quantity
  across all cohort-shared electrodes; the panels separate frequency bands
  and diagnostic groups.
- `figures/group_statistics/`: electrode-level effect and strict-FDR maps.

If detection metrics already exist, regenerate only these subject-average
group figures without repeating burst detection:

```bash
bash src/analyses/bycycle/generate_group_figures.sh
```

Event agreement is not expected to be perfect because the methods define bouts
differently. Scientific robustness should be judged mainly from stability of
effect direction, effect size, and uncertainty across the two detectors—not by
requiring both methods to cross an arbitrary p-value threshold.
