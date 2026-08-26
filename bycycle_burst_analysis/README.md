# Independent bycycle burst detector

This folder is a sensitivity analysis for the aperiodic-relative eBOSC detector
in `scale_free_analysis/`. It detects bursts directly from consecutive-cycle
consistency using `bycycle`; no specparam curve, wavelet-power threshold, or
eBOSC mask is used to decide whether a cycle is bursting.

The prespecified criteria are amplitude fraction > 0.30, amplitude consistency
> 0.50, period consistency > 0.50, monotonicity > 0.80, and at least three
consecutive qualifying cycles. Each accepted four-second epoch is analyzed
separately, with a 0.5-second edge margin. Thresholds are fixed in
`config.json` and must not be tuned using PD-versus-Control significance.

All analyses use only electrodes shared by every included subject. Theta,
alpha, low-beta, and high-beta enter formal inference. The overlapping 5–15 Hz
band is saved and plotted only as a descriptive visualization.

Run the complete full-cohort stage with:

```bash
bash bycycle_burst_analysis/run_bycycle_burst_analysis.sh --overwrite
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
- `figures/group_statistics/`: electrode-level effect and strict-FDR maps.

Event agreement is not expected to be perfect because the methods define bouts
differently. Scientific robustness should be judged mainly from stability of
effect direction, effect size, and uncertainty across the two detectors—not by
requiring both methods to cross an arbitrary p-value threshold.
