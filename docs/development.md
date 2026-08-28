# Development and verification

Run fast structural checks:

```bash
bash run_pipeline.sh validate-config
bash run_pipeline.sh plan --profile compute --cohort full
bash -n run_pipeline.sh src/analyses/matching/run_matched_analyses.sh
python -m unittest -v tests.test_pipeline_resume
```

Run the complete test suite inside the pinned environment:

```bash
conda run --no-capture-output -n MNE_August2026 \
  python -m unittest discover -s tests -v
```

Scientific refactors should preserve output keys, row grids, subjects,
electrodes, and numerical results within an explicitly justified floating-point
tolerance. Do not combine structural refactoring with parameter changes.

Shared code and orchestration belongs in `src/`. Plot functions should render
prepared tables; statistics should not be hidden inside plotting code.
