"""Canonical configuration and recording schema for multiple EEG datasets."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


CANONICAL_COLUMNS = (
    "dataset_id",
    "recording_id",
    "participant_id",
    "session_id",
    "epoch_path",
    "raw_path",
    "group",
    "medication_state",
    "age_years",
    "sex",
    "updrs",
    "moca",
    "mmse",
)

_SUBJECT_RE = re.compile(r"(sub-[A-Za-z0-9]+)")
_SESSION_RE = re.compile(r"(ses-[A-Za-z0-9]+)")


def _nullable_float(value: Any) -> float | None:
    if value is None or pd.isna(value) or str(value).strip() == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if pd.notna(number) else None


def _clean_text(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def subject_from_path(path: str | Path) -> str:
    match = _SUBJECT_RE.search(str(path))
    if not match:
        raise ValueError(f"Cannot infer BIDS participant from {path}")
    return match.group(1)


def session_from_path(path: str | Path) -> str | None:
    match = _SESSION_RE.search(str(path))
    return match.group(1) if match else None


def recording_from_path(path: str | Path) -> str:
    subject = subject_from_path(path)
    session = session_from_path(path)
    return f"{subject}_{session}" if session else subject


def normalize_group(value: Any, participant_id: str) -> str:
    """Map common study labels to ``Control`` or a PD condition label."""
    text = (_clean_text(value) or "").lower().replace("-", "_").replace(" ", "_")
    if text in {"control", "controls", "healthy", "hc", "healthy_control", "healthy_controls"}:
        return "Control"
    if "pd" in text or "parkinson" in text:
        if "off" in text:
            return "PD_OFF"
        if "on" in text:
            return "PD_ON"
        return "PD"
    inferred = participant_id.removeprefix("sub-").lower()
    if inferred.startswith(("hc", "control", "healthy")):
        return "Control"
    if inferred.startswith("pd"):
        return "PD"
    return _clean_text(value) or "Unknown"


def medication_state(group: str, value: Any = None) -> str | None:
    text = (_clean_text(value) or group).lower()
    if "off" in text:
        return "OFF"
    if "on" in text:
        return "ON"
    return None


@dataclass(frozen=True)
class DatasetConfig:
    """One source dataset in the global configuration."""

    dataset_id: str
    root: Path
    metadata: Path | None
    epochs_dir: Path
    epoch_glob: str
    raw_glob: str | None = None
    preprocessing_config: Path | None = None
    enabled: bool = True
    columns: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: dict[str, Any], base_dir: Path) -> "DatasetConfig":
        dataset_id = str(value.get("id", "")).strip()
        if not dataset_id:
            raise ValueError("Every global dataset requires a non-empty id")

        def resolve(raw: Any, *, allow_none: bool = False) -> Path | None:
            if raw is None and allow_none:
                return None
            if raw is None:
                raise ValueError(f"Dataset {dataset_id!r} is missing a path")
            path = Path(str(raw))
            return path if path.is_absolute() else base_dir / path

        epochs_dir = resolve(value.get("epochs_dir", "processed/epochs"))
        metadata = resolve(value.get("metadata"), allow_none=True)
        preprocessing_config = resolve(value.get("preprocessing_config"), allow_none=True)
        return cls(
            dataset_id=dataset_id,
            root=resolve(value.get("root", ".")),
            metadata=metadata,
            epochs_dir=epochs_dir,
            epoch_glob=str(value.get("epoch_glob", "sub-*_task-rest_desc-cleaned_epo.fif")),
            raw_glob=(str(value["raw_glob"]) if value.get("raw_glob") else None),
            preprocessing_config=preprocessing_config,
            enabled=bool(value.get("enabled", True)),
            columns={str(key): str(item) for key, item in value.get("columns", {}).items()},
        )


@dataclass(frozen=True)
class CanonicalRecording:
    dataset_id: str
    recording_id: str
    participant_id: str
    session_id: str | None
    epoch_path: Path
    raw_path: Path | None
    group: str
    medication_state: str | None
    age_years: float | None
    sex: str | None
    updrs: float | None
    moca: float | None
    mmse: float | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "recording_id": self.recording_id,
            "participant_id": self.participant_id,
            "session_id": self.session_id or "",
            "epoch_path": str(self.epoch_path),
            "raw_path": str(self.raw_path) if self.raw_path else "",
            "group": self.group,
            "medication_state": self.medication_state or "",
            "age_years": self.age_years,
            "sex": self.sex or "",
            "updrs": self.updrs,
            "moca": self.moca,
            "mmse": self.mmse,
        }


@dataclass(frozen=True)
class GlobalConfig:
    path: Path
    output_root: Path
    datasets: tuple[DatasetConfig, ...]
    bands: dict[str, tuple[float, float]]
    block_epochs: int = 16
    embedding_dimension: int = 6
    delay_samples: int = 1
    bout_threshold_percentile: float = 95.0
    bout_minimum_cycles: float = 3.0
    fdr_alpha: float = 0.05
    random_seed: int = 20260826

    @property
    def enabled_datasets(self) -> tuple[DatasetConfig, ...]:
        return tuple(dataset for dataset in self.datasets if dataset.enabled)


def load_global_config(path: str | Path) -> GlobalConfig:
    config_path = Path(path).resolve()
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    if int(raw.get("schema_version", 0)) != 1:
        raise ValueError("global pipeline config must use schema_version 1")
    base_dir = config_path.parent.parent
    output_root = Path(str(raw.get("output_root", "outputs/global")))
    if not output_root.is_absolute():
        output_root = base_dir / output_root
    bands = {
        str(name): (float(bounds[0]), float(bounds[1]))
        for name, bounds in raw.get("bands", {}).items()
    }
    if not bands:
        raise ValueError("global pipeline requires at least one frequency band")
    for name, (low, high) in bands.items():
        if not 0.0 < low < high:
            raise ValueError(f"Invalid frequency band {name!r}: {(low, high)}")
    datasets = tuple(
        DatasetConfig.from_dict(item, base_dir)
        for item in raw.get("datasets", [])
    )
    if not datasets:
        raise ValueError("global pipeline requires at least one dataset")
    if len({dataset.dataset_id for dataset in datasets}) != len(datasets):
        raise ValueError("global dataset ids must be unique")
    block_epochs = int(raw.get("block_epochs", 16))
    dx = int(raw.get("embedding_dimension", 6))
    tau = int(raw.get("delay_samples", 1))
    if block_epochs < 1 or not 2 <= dx <= 7 or tau < 1:
        raise ValueError("block_epochs must be positive, dx must be 2..7, and delay >= 1")
    percentile = float(raw.get("bout_threshold_percentile", 95.0))
    if not 50.0 < percentile < 100.0:
        raise ValueError("bout_threshold_percentile must be between 50 and 100")
    alpha = float(raw.get("fdr_alpha", 0.05))
    if not 0.0 < alpha < 1.0:
        raise ValueError("fdr_alpha must be between zero and one")
    return GlobalConfig(
        path=config_path,
        output_root=output_root,
        datasets=datasets,
        bands=bands,
        block_epochs=block_epochs,
        embedding_dimension=dx,
        delay_samples=tau,
        bout_threshold_percentile=percentile,
        bout_minimum_cycles=float(raw.get("bout_minimum_cycles", 3.0)),
        fdr_alpha=alpha,
        random_seed=int(raw.get("random_seed", 20260826)),
    )


def read_canonical_table(path: str | Path) -> pd.DataFrame:
    """Read and validate the compact converter output."""
    table = pd.read_csv(path, compression="infer")
    missing = sorted(set(CANONICAL_COLUMNS) - set(table.columns))
    if missing:
        raise ValueError(f"Canonical recording table is missing columns: {missing}")
    if table["recording_id"].duplicated().any():
        duplicate = table.loc[table["recording_id"].duplicated(), "recording_id"].iloc[0]
        raise ValueError(f"Duplicate canonical recording_id: {duplicate}")
    if table.empty:
        raise ValueError(f"Canonical recording table is empty: {path}")
    return table
