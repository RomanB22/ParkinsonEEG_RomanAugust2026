"""Typed loading and validation for the public pipeline configuration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.frequency_bands import (
    CANONICAL_BOUT_BAND_NAMES,
    validate_frequency_bands,
)


DEFAULT_CONFIG = Path("config/pipeline.yaml")
VALID_COHORTS = ("full", "matched")


@dataclass(frozen=True)
class DatasetProfile:
    """Paths and labels for one dataset using the shared preprocessing contract."""

    name: str
    label: str
    preprocessing_config: Path
    participants_file: Path
    epochs_dir: Path
    epoch_glob: str
    analysis_stage: str | None

    @classmethod
    def from_dict(cls, name: str, value: dict[str, Any]) -> "DatasetProfile":
        required = {
            "label",
            "preprocessing_config",
            "participants_file",
            "epochs_dir",
            "epoch_glob",
        }
        missing = sorted(required - set(value))
        if missing:
            raise ValueError(f"Dataset {name!r} is missing keys: {missing}")
        return cls(
            name=name,
            label=str(value["label"]),
            preprocessing_config=Path(value["preprocessing_config"]),
            participants_file=Path(value["participants_file"]),
            epochs_dir=Path(value["epochs_dir"]),
            epoch_glob=str(value["epoch_glob"]),
            analysis_stage=(
                str(value["analysis_stage"])
                if value.get("analysis_stage") is not None
                else None
            ),
        )


@dataclass(frozen=True)
class Profile:
    """A named, bounded collection of pipeline capabilities."""

    name: str
    description: str
    include_sweep: bool
    include_reports: bool
    include_models: bool
    include_matched: bool
    include_tests: bool
    include_bycycle: bool

    @classmethod
    def from_dict(cls, name: str, value: dict[str, Any]) -> "Profile":
        required = {
            "description",
            "include_sweep",
            "include_reports",
            "include_models",
            "include_matched",
            "include_tests",
            "include_bycycle",
        }
        missing = sorted(required - set(value))
        if missing:
            raise ValueError(f"Profile {name!r} is missing keys: {missing}")
        return cls(
            name=name,
            description=str(value["description"]),
            include_sweep=bool(value["include_sweep"]),
            include_reports=bool(value["include_reports"]),
            include_models=bool(value["include_models"]),
            include_matched=bool(value["include_matched"]),
            include_tests=bool(value["include_tests"]),
            include_bycycle=bool(value["include_bycycle"]),
        )


@dataclass(frozen=True)
class ScientificDefaults:
    """Cross-analysis settings that must not silently drift."""

    psd_range_hz: tuple[float, float]
    aperiodic_fit_range_hz: tuple[float, float]
    ordinal_dimensions: tuple[int, ...]
    ordinal_delay_samples: tuple[int, ...]
    renyi_alphas: tuple[float, ...]
    frequency_bands: dict[str, tuple[float, float]]
    eight_electrodes: tuple[str, ...]
    scalar_colormap: str
    fdr_alpha: float

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ScientificDefaults":
        def pair(name: str) -> tuple[float, float]:
            raw = value.get(name)
            if not isinstance(raw, list) or len(raw) != 2:
                raise ValueError(f"science.{name} must contain exactly two numbers")
            result = (float(raw[0]), float(raw[1]))
            if result[0] >= result[1]:
                raise ValueError(f"science.{name} must be increasing")
            return result

        bands = {
            str(name): (float(limits[0]), float(limits[1]))
            for name, limits in value.get("frequency_bands", {}).items()
        }
        if not bands:
            raise ValueError("science.frequency_bands cannot be empty")
        validate_frequency_bands(bands, context="science.frequency_bands")
        dimensions = tuple(int(item) for item in value.get("ordinal_dimensions", []))
        delays = tuple(int(item) for item in value.get("ordinal_delay_samples", []))
        alphas = tuple(float(item) for item in value.get("renyi_alphas", []))
        electrodes = tuple(str(item) for item in value.get("eight_electrodes", []))
        if dimensions != (3, 4, 5, 6):
            raise ValueError("The documented ordinal dimensions must be [3, 4, 5, 6]")
        if delays != (1,):
            raise ValueError("Ordinal analyses are restricted to tau=1")
        if len(electrodes) != 8 or len(set(electrodes)) != 8:
            raise ValueError("science.eight_electrodes must contain eight unique names")
        fdr_alpha = float(value.get("fdr_alpha", 0.05))
        if not 0.0 < fdr_alpha < 1.0:
            raise ValueError("science.fdr_alpha must be between zero and one")
        return cls(
            psd_range_hz=pair("psd_range_hz"),
            aperiodic_fit_range_hz=pair("aperiodic_fit_range_hz"),
            ordinal_dimensions=dimensions,
            ordinal_delay_samples=delays,
            renyi_alphas=alphas,
            frequency_bands=bands,
            eight_electrodes=electrodes,
            scalar_colormap=str(value.get("scalar_colormap", "viridis")),
            fdr_alpha=fdr_alpha,
        )


@dataclass(frozen=True)
class PipelineConfig:
    """Complete public configuration for orchestration."""

    path: Path
    schema_version: int
    environment_name: str
    preprocessing_config: Path
    participants_file: Path
    epochs_dir: Path
    epoch_glob: str
    datasets: dict[str, DatasetProfile]
    science: ScientificDefaults
    profiles: dict[str, Profile]

    def profile(self, name: str) -> Profile:
        try:
            return self.profiles[name]
        except KeyError as error:
            choices = ", ".join(sorted(self.profiles))
            raise ValueError(f"Unknown profile {name!r}; choose one of: {choices}") from error

    def dataset(self, name: str) -> DatasetProfile:
        try:
            return self.datasets[name]
        except KeyError as error:
            choices = ", ".join(sorted(self.datasets))
            raise ValueError(f"Unknown dataset {name!r}; choose one of: {choices}") from error


def load_pipeline_config(path: str | Path = DEFAULT_CONFIG) -> PipelineConfig:
    """Load JSON-formatted YAML without adding a YAML dependency."""
    config_path = Path(path)
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Pipeline configuration not found: {config_path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON-formatted YAML in {config_path}: {error}") from error

    if int(raw.get("schema_version", 0)) != 1:
        raise ValueError("config/pipeline.yaml must use schema_version 1")
    paths = raw.get("paths", {})
    profiles = {
        str(name): Profile.from_dict(str(name), value)
        for name, value in raw.get("profiles", {}).items()
    }
    if not profiles:
        raise ValueError("At least one pipeline profile is required")
    dataset_values = raw.get("datasets")
    if dataset_values is None:
        dataset_values = {
            "primary": {
                "label": "Primary Parkinson/control dataset",
                "preprocessing_config": paths.get(
                    "preprocessing_config", "config/preprocessing.yaml"
                ),
                "participants_file": paths.get(
                    "participants_file", "processed/metadata/subjects.csv"
                ),
                "epochs_dir": paths.get("epochs_dir", "processed/epochs"),
                "epoch_glob": paths.get(
                    "epoch_glob", "sub-*_task-Rest_desc-cleaned_epo.fif"
                ),
                "analysis_stage": None,
            }
        }
    datasets = {
        str(name): DatasetProfile.from_dict(str(name), value)
        for name, value in dataset_values.items()
    }
    if "primary" not in datasets:
        raise ValueError("config/pipeline.yaml datasets must define 'primary'")
    primary = datasets["primary"]
    result = PipelineConfig(
        path=config_path,
        schema_version=1,
        environment_name=str(raw.get("environment", {}).get("name", "MNE_August2026")),
        preprocessing_config=primary.preprocessing_config,
        participants_file=primary.participants_file,
        epochs_dir=primary.epochs_dir,
        epoch_glob=primary.epoch_glob,
        datasets=datasets,
        science=ScientificDefaults.from_dict(raw.get("science", {})),
        profiles=profiles,
    )
    validate_scientific_configs(result)
    return result


def _json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _numeric_pair(value: Any) -> tuple[float, float]:
    return float(value[0]), float(value[1])


def validate_scientific_configs(config: PipelineConfig) -> None:
    """Fail early when duplicated domain configs disagree with public defaults.

    Domain-specific files are retained because they document each method in a
    self-contained way.  The public configuration is authoritative for values
    shared across domains, and this audit prevents the files from drifting.
    """
    problems: list[str] = []
    files = {
        "psd": Path("config/analyses/psd.json"),
        "ordinal": Path("config/analyses/ordinal.json"),
        "scale_free": Path("config/analyses/scale_free.json"),
        "bout": Path("config/analyses/bouts.json"),
        "bycycle": Path("config/analyses/bycycle.json"),
        "eight": Path("config/analyses/eight_electrode.json"),
        "quantitative": Path("config/analyses/behavioral.json"),
        "disease": Path("config/analyses/progression.json"),
        "duration": Path("config/analyses/duration_qc.json"),
    }
    missing = [str(path) for path in files.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing domain configuration files: {missing}")
    values = {name: _json(path) for name, path in files.items()}

    for name in ("psd", "scale_free", "bout"):
        section = values[name]["psd"]
        observed = (float(section["fmin_hz"]), float(section["fmax_hz"]))
        if observed != config.science.psd_range_hz:
            problems.append(f"{files[name]} PSD range is {observed}")
    for name in ("scale_free", "bout"):
        observed = _numeric_pair(values[name]["specparam"]["frequency_range_hz"])
        if observed != config.science.aperiodic_fit_range_hz:
            problems.append(f"{files[name]} aperiodic fit range is {observed}")
    primary = values["ordinal"]["ordinal"]
    if int(primary["embedding_dimension"]) != max(config.science.ordinal_dimensions):
        problems.append("config/analyses/ordinal.json must contain primary D=6")
    if int(primary["delay_samples"]) != config.science.ordinal_delay_samples[0]:
        problems.append("config/analyses/ordinal.json must contain tau=1")
    for name in ("psd", "ordinal"):
        observed = {
            band: _numeric_pair(limits)
            for band, limits in values[name]["bands"].items()
        }
        if observed != config.science.frequency_bands:
            problems.append(
                f"{files[name]} frequency bands disagree with pipeline.yaml"
            )
    expected_bout_bands = {
        name: config.science.frequency_bands[name]
        for name in CANONICAL_BOUT_BAND_NAMES
    }
    for name in ("scale_free", "bout", "bycycle"):
        observed = {
            band: _numeric_pair(limits)
            for band, limits in values[name]["bands"].items()
        }
        if observed != expected_bout_bands:
            problems.append(
                f"{files[name]} bout bands disagree with pipeline.yaml"
            )
    if tuple(values["eight"]["electrodes"]) != config.science.eight_electrodes:
        problems.append("config/analyses/eight_electrode.json electrode order disagrees")
    fdr_locations = {
        "psd": "statistics",
        "ordinal": "statistics",
        "scale_free": "statistics",
        "bout": "statistics",
        "bycycle": "statistics",
        "eight": "statistics",
        "quantitative": "analysis",
        "disease": "analysis",
        "duration": "statistics",
    }
    for name, section in fdr_locations.items():
        observed = float(values[name][section]["fdr_alpha"])
        if observed != config.science.fdr_alpha:
            problems.append(f"{files[name]} fdr_alpha is {observed}")
    dimension_metrics = set(
        values["quantitative"]["dimension_sensitivity"]["metrics"]
    )
    for alpha in config.science.renyi_alphas:
        token = f"{alpha:g}".replace(".", "_")
        for quantity in ("entropy", "complexity"):
            expected = f"renyi_{quantity}_alpha_{token}"
            if expected not in dimension_metrics:
                problems.append(
                    f"config/analyses/behavioral.json lacks {expected}"
                )
    if config.science.scalar_colormap != "viridis":
        problems.append("The repository-wide scalar colormap must be viridis")
    if problems:
        detail = "\n  - ".join(problems)
        raise ValueError(f"Shared scientific configuration drift detected:\n  - {detail}")
