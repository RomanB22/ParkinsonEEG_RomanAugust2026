"""Canonical configuration and recording schema for multiple EEG datasets."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, replace
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

DEFAULT_APERIODIC_SETTINGS = {
    "aperiodic_mode": "best_bic",
    "aperiodic_modes": ["fixed", "knee"],
    "model_selection_criterion": "bic",
    "frequency_range_hz": [4.0, 50.0],
    "peak_width_limits_hz": [1.0, 12.0],
    "max_n_peaks": 8,
    "min_peak_height": 0.0,
    "peak_threshold": 2.0,
}

DEFAULT_APERIODIC_QC_SETTINGS = {
    "minimum_r_squared": 0.9,
    "maximum_error_mae_log10": 0.15,
    "maximum_absolute_residual_log10": 0.75,
    "exponent_range": [0.0, 3.0],
    "minimum_subject_qc_fraction": 0.8,
}

DEFAULT_EBOSC_SETTINGS = {
    "frequency_min_hz": 4.0,
    "frequency_max_hz": 50.0,
    "frequency_step_hz": 1.0,
    "wavenumber": 6.0,
    "power_percentile": 0.95,
    "minimum_cycles": 3.0,
    "edge_padding_seconds": 0.75,
    "figure_window_seconds": 0.5,
}

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
    session_metadata_glob: str | None
    epochs_dir: Path
    epoch_glob: str
    task: str = "Rest"
    raw_glob: str | None = None
    preprocessing_config: Path | None = None
    notch_frequency_hz: float | None = None
    auxiliary_names: tuple[str, ...] = ()
    enabled: bool = True
    columns: dict[str, str] = field(default_factory=dict)
    analysis: dict[str, Any] = field(default_factory=dict)

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
            session_metadata_glob=(
                str(value["session_metadata_glob"])
                if value.get("session_metadata_glob")
                else None
            ),
            epochs_dir=epochs_dir,
            epoch_glob=str(value.get("epoch_glob", "sub-*_task-rest_desc-cleaned_epo.fif")),
            task=str(value.get("task", "Rest")),
            raw_glob=(str(value["raw_glob"]) if value.get("raw_glob") else None),
            preprocessing_config=preprocessing_config,
            notch_frequency_hz=(
                float(value["notch_frequency_hz"])
                if value.get("notch_frequency_hz") is not None
                else None
            ),
            auxiliary_names=tuple(str(item) for item in value.get("auxiliary_names", [])),
            enabled=bool(value.get("enabled", True)),
            columns={str(key): str(item) for key, item in value.get("columns", {}).items()},
            analysis=dict(value.get("analysis", {})),
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
    permutation_dimensions: tuple[int, ...] = (3, 4, 5, 6, 7)
    aperiodic_settings: dict[str, Any] = field(
        default_factory=lambda: dict(DEFAULT_APERIODIC_SETTINGS)
    )
    aperiodic_qc_settings: dict[str, Any] = field(
        default_factory=lambda: dict(DEFAULT_APERIODIC_QC_SETTINGS)
    )
    ebosc_settings: dict[str, Any] = field(
        default_factory=lambda: dict(DEFAULT_EBOSC_SETTINGS)
    )
    delay_samples: int = 1
    bout_threshold_percentile: float = 95.0
    bout_minimum_cycles: float = 3.0
    fdr_alpha: float = 0.05
    random_seed: int = 20260826

    @property
    def enabled_datasets(self) -> tuple[DatasetConfig, ...]:
        return tuple(dataset for dataset in self.datasets if dataset.enabled)

    def for_dataset(self, dataset_id: str) -> "GlobalConfig":
        """Return analysis settings with the selected dataset's overrides."""
        dataset = next(
            (item for item in self.datasets if item.dataset_id == dataset_id), None
        )
        if dataset is None or not dataset.analysis:
            return self
        overrides = dataset.analysis
        bands = dict(self.bands)
        if "bands" in overrides:
            bands.update({
                str(name): (float(bounds[0]), float(bounds[1]))
                for name, bounds in overrides["bands"].items()
            })
        aperiodic = dict(self.aperiodic_settings)
        aperiodic.update(overrides.get("aperiodic", {}))
        ebosc = dict(self.ebosc_settings)
        ebosc.update(overrides.get("ebosc", {}))
        fit_low, fit_high = (float(value) for value in aperiodic["frequency_range_hz"])
        if not 0.0 < fit_low < fit_high:
            raise ValueError(f"Invalid aperiodic fit range for dataset {dataset_id!r}")
        for name, (low, high) in bands.items():
            if not 0.0 < float(low) < float(high):
                raise ValueError(f"Invalid band {name!r} for dataset {dataset_id!r}")
        if not 0.0 < float(ebosc["frequency_min_hz"]) < float(ebosc["frequency_max_hz"]):
            raise ValueError(f"Invalid eBOSC frequency range for dataset {dataset_id!r}")
        return replace(
            self,
            bands=bands,
            aperiodic_settings=aperiodic,
            ebosc_settings=ebosc,
        )


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
    dimensions = tuple(int(value) for value in raw.get("permutation_dimensions", (3, 4, 5, 6, 7)))
    tau = int(raw.get("delay_samples", 1))
    if block_epochs < 1 or not 2 <= dx <= 7 or tau < 1:
        raise ValueError("block_epochs must be positive, dx must be 2..7, and delay >= 1")
    if not dimensions or any(not 2 <= dimension <= 7 for dimension in dimensions):
        raise ValueError("permutation_dimensions must contain values from 2 through 7")
    if len(set(dimensions)) != len(dimensions):
        raise ValueError("permutation_dimensions must not contain duplicates")
    aperiodic = dict(raw.get("aperiodic", {}))
    defaults = dict(DEFAULT_APERIODIC_SETTINGS)
    defaults.update(aperiodic)
    if defaults["aperiodic_modes"] != ["fixed", "knee"]:
        raise ValueError("aperiodic.aperiodic_modes must be ['fixed', 'knee']")
    if defaults["model_selection_criterion"] != "bic":
        raise ValueError("aperiodic.model_selection_criterion must be 'bic'")
    fit_range = [float(value) for value in defaults["frequency_range_hz"]]
    if len(fit_range) != 2 or not 0.0 < fit_range[0] < fit_range[1]:
        raise ValueError("aperiodic.frequency_range_hz must be an increasing positive range")
    aperiodic_qc = dict(DEFAULT_APERIODIC_QC_SETTINGS)
    aperiodic_qc.update(raw.get("aperiodic_fit_qc", {}))
    exponent_range = [float(value) for value in aperiodic_qc["exponent_range"]]
    if not 0.0 <= float(aperiodic_qc["minimum_r_squared"]) <= 1.0:
        raise ValueError("aperiodic_fit_qc.minimum_r_squared must be in [0, 1]")
    if float(aperiodic_qc["maximum_error_mae_log10"]) <= 0.0:
        raise ValueError("aperiodic_fit_qc.maximum_error_mae_log10 must be positive")
    if float(aperiodic_qc["maximum_absolute_residual_log10"]) <= 0.0:
        raise ValueError("aperiodic_fit_qc.maximum_absolute_residual_log10 must be positive")
    if len(exponent_range) != 2 or exponent_range[0] >= exponent_range[1]:
        raise ValueError("aperiodic_fit_qc.exponent_range must increase")
    if not 0.0 < float(aperiodic_qc["minimum_subject_qc_fraction"]) <= 1.0:
        raise ValueError("aperiodic_fit_qc.minimum_subject_qc_fraction must be in (0, 1]")
    ebosc = dict(DEFAULT_EBOSC_SETTINGS)
    ebosc.update(raw.get("ebosc", {}))
    # Accept the pre-eBOSC global names when loading an older configuration.
    if "ebosc" not in raw or "power_percentile" not in raw["ebosc"]:
        if "bout_threshold_percentile" in raw:
            ebosc["power_percentile"] = float(raw["bout_threshold_percentile"]) / 100.0
    if "ebosc" not in raw or "minimum_cycles" not in raw["ebosc"]:
        if "bout_minimum_cycles" in raw:
            ebosc["minimum_cycles"] = float(raw["bout_minimum_cycles"])
    if not 0.0 < float(ebosc["frequency_min_hz"]) < float(ebosc["frequency_max_hz"]):
        raise ValueError("Invalid eBOSC frequency range")
    if float(ebosc["frequency_step_hz"]) <= 0.0 or float(ebosc["wavenumber"]) <= 0.0:
        raise ValueError("eBOSC frequency step and wavenumber must be positive")
    if not 0.0 < float(ebosc["power_percentile"]) < 1.0:
        raise ValueError("ebosc.power_percentile must be between zero and one")
    if float(ebosc["minimum_cycles"]) <= 0.0 or float(ebosc["edge_padding_seconds"]) < 0.0:
        raise ValueError("Invalid eBOSC duration or edge-padding setting")
    if float(ebosc["figure_window_seconds"]) <= 0.0:
        raise ValueError("ebosc.figure_window_seconds must be positive")
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
        permutation_dimensions=dimensions,
        aperiodic_settings=defaults,
        aperiodic_qc_settings=aperiodic_qc,
        ebosc_settings=ebosc,
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
    key_columns = ["dataset_id", "recording_id"]
    if table.duplicated(key_columns).any():
        duplicate = table.loc[table.duplicated(key_columns), key_columns].iloc[0].to_dict()
        raise ValueError(
            "Duplicate canonical recording key: "
            f"{duplicate['dataset_id']}/{duplicate['recording_id']}"
        )
    if table.empty:
        raise ValueError(f"Canonical recording table is empty: {path}")
    return table
