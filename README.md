# Parkinson resting-state EEG

This repository implements a conservative, reproducible analysis of 149
eyes-open resting EEG recordings: 100 Parkinson disease participants and 49
controls. It includes reviewed ICA cleaning, PSD, ordinal quantities,
fixed/knee spectral parameterization, oscillatory bouts, clinical associations,
and transparent prediction models.

## Start here

There is one public entry point:

```bash
# See the complete dependency plan without changing files.
bash run_pipeline.sh plan --profile paper

# Generate ICA review material.
bash run_pipeline.sh review --overwrite

# After visually confirming ICA decisions, run the paper pipeline.
bash run_pipeline.sh run --profile paper
```

The runner creates or reuses the `MNE_August2026` conda environment, resumes
current stages, and writes a consolidated log under `pipeline_logs/`. It never
passes `--overwrite` to a stage unless that flag was explicitly included in the
public command.

For an explicitly unattended run:

```bash
bash run_pipeline.sh run --profile paper \
  --skip-manual-ica-review \
  --preprocessing-workers 8 \
  --no-progress
```

Automatic ICA removal is recorded in provenance and is not the preferred
scientific workflow.

## Useful commands

```bash
# Downstream only, using existing cleaned epochs
bash run_pipeline.sh analyses --profile paper

# One stage and its dependencies
bash run_pipeline.sh stage full.scale-free

# Only matched-cohort results, reusing full feature caches
bash run_pipeline.sh analyses --profile paper --cohort matched

# Include independent bycycle burst-detection sensitivity
bash run_pipeline.sh analyses --profile full-qc

# Inspect freshness and available stage names
bash run_pipeline.sh status --profile paper
bash run_pipeline.sh list

# Validate shared scientific configuration
bash run_pipeline.sh validate-config
```

`run_reproducible_pipeline.sh`, `run_all_analyses.sh`, and
the legacy matched runner remain as thin compatibility aliases.
They contain no pipeline logic.

## Scientific scope

- Preprocessing: 1–100 Hz, 60 Hz notch, 250 Hz, CAR for ICA/ICLabel, extended
  Infomax, interpolation of detected bad channels, four-second epochs.
- PSD: Welch spectra and relative power from 1–50 Hz.
- Ordinal: D=6 primary; D=3–5 independent sensitivities; tau=1 only; Shannon,
  Fisher, and Rényi quantities.
- Aperiodic fits: fixed and knee specparam candidates over 4–50 Hz, selected by
  BIC, with formal fit QC.
- Canonical bands: delta 1–4, theta 4–8, alpha 8–13, beta 13–30, and gamma
  30–50 Hz across the repository.
- Bouts: aperiodic-relative eBOSC detection uses the canonical theta through
  gamma subset (4–50 Hz); bycycle is an optional independent check. Delta is
  excluded because the aperiodic spectra are fitted over 4–50 Hz.
- Clinical analyses: age/sex-adjusted MOCA and UPDRS associations plus full and
  demographically matched cohort views.
- Classification: deliberately conservative, explainable feature sets and
  nested validation.

The overlapping exploratory 5–15 Hz band is not part of inferential feature
tables.

## Documentation

- [Commands](docs/commands.md)
- [Architecture](docs/architecture.md)
- [Pipeline and profiles](docs/pipeline.md)
- [Configuration](docs/configuration.md)
- [Output map](docs/outputs.md)
- [Development](docs/development.md)
- [Preprocessing method](docs/preprocessing.md)
- [Detailed preprocessing parameters](docs/pipeline_parameters.md)
- [Group statistics](docs/group_statistics.md)
- [Analysis methods](docs/analyses/README.md)

Original files under `dataset/` are read-only inputs. Generated full-cohort and
matched results live under `outputs/` and are ignored by Git.
