# Configuration

`config/pipeline.yaml` is the public source of truth for settings shared across
analyses. It is JSON-formatted YAML so the lightweight runner does not require
PyYAML.

It defines:

- the conda environment and central input paths;
- PSD and aperiodic fitting ranges;
- ordinal dimensions, tau, and Rényi alphas;
- bout bands;
- the prespecified eight electrodes;
- scalar colormap and FDR alpha;
- bounded compute, paper, and full-QC profiles.

Domain configurations such as `config/analyses/scale_free.json` retain the
complete method-specific parameters. This makes each scientific module
self-documenting. On every CLI invocation, the public loader checks that all
duplicated cross-domain values agree and fails before computation if they do
not.

Validate configuration without creating outputs:

```bash
bash run_pipeline.sh validate-config
```

Important current invariants are:

- PSD: 1–50 Hz;
- specparam fixed and knee candidates: 4–50 Hz, selected by BIC;
- ordinal: D=6 primary, D=3–5 independent sensitivity, tau=1 only;
- bout bands: theta, alpha, low beta, and high beta;
- no overlapping 5–15 Hz band in inferential feature tables;
- scalar continuous colormap: `viridis`;
- prespecified FDR alpha: 0.05.

Preprocessing remains fully specified in `config/preprocessing.yaml`; see
`docs/preprocessing.md` and `docs/pipeline_parameters.md`.

