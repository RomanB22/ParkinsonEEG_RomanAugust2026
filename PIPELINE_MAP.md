# Pipeline map

The live pipeline graph is defined once in `parkinson_eeg/registry.py`.

```bash
bash run_pipeline.sh list
bash run_pipeline.sh plan --profile paper
bash run_pipeline.sh status --profile paper
```

See [docs/pipeline.md](docs/pipeline.md) for the dependency diagram and
[docs/architecture.md](docs/architecture.md) for calculation ownership and the
stage contract. See [docs/outputs.md](docs/outputs.md) for all result paths.
