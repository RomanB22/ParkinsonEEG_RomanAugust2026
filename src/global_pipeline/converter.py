"""Convert heterogeneous BIDS-like study metadata to the canonical schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .schema import (
    CANONICAL_COLUMNS,
    CanonicalRecording,
    DatasetConfig,
    GlobalConfig,
    medication_state,
    normalize_group,
    recording_from_path,
    session_from_path,
    subject_from_path,
)


def _read_metadata(dataset: DatasetConfig) -> pd.DataFrame:
    if dataset.metadata is None or not dataset.metadata.exists():
        return pd.DataFrame()
    separator = "\t" if dataset.metadata.suffix.lower() in {".tsv", ".txt"} else ","
    return pd.read_csv(dataset.metadata, sep=separator, dtype=str)


def _metadata_index(metadata: pd.DataFrame) -> dict[str, dict[str, Any]]:
    if metadata.empty:
        return {}
    id_column = next(
        (column for column in ("participant_id", "subject_id", "recording_id", "ID", "id") if column in metadata),
        None,
    )
    if id_column is None:
        return {}
    result: dict[str, dict[str, Any]] = {}
    for row in metadata.to_dict(orient="records"):
        key = str(row.get(id_column, "")).strip()
        if key and not key.startswith("sub-"):
            key = f"sub-{key}"
        if key:
            result[key] = row
    return result


def _value(row: dict[str, Any], columns: dict[str, str], canonical: str) -> Any:
    source = columns.get(canonical)
    if source and source in row:
        return row[source]
    aliases = {
        "group": ("group", "GROUP", "condition", "diagnosis", "diagnostic_group", "status"),
        "medication_state": ("medication_state", "medication", "state", "condition"),
        "age_years": ("age_years", "age", "AGE"),
        "sex": ("sex", "SEX", "gender", "GENDER"),
        "updrs": ("updrs", "UPDRS", "updrs_total", "UPDRS_total"),
        "moca": ("moca", "MOCA", "moca_total", "MoCA"),
        "mmse": ("mmse", "MMSE", "mmse_total"),
    }
    return next((row[name] for name in aliases.get(canonical, ()) if name in row), None)


def _raw_lookup(dataset: DatasetConfig) -> dict[str, Path]:
    if not dataset.raw_glob:
        return {}
    paths = sorted(dataset.root.glob(dataset.raw_glob))
    return {recording_from_path(path): path for path in paths}


def convert_dataset(dataset: DatasetConfig) -> list[CanonicalRecording]:
    if not dataset.epochs_dir.exists():
        raise FileNotFoundError(
            f"Dataset {dataset.dataset_id!r}: epochs directory does not exist: "
            f"{dataset.epochs_dir}. Set enabled=false for unused templates, "
            "or add preprocessing_config and run with --preprocess."
        )
    epoch_paths = sorted(dataset.epochs_dir.glob(dataset.epoch_glob))
    if not epoch_paths:
        raise FileNotFoundError(
            f"Dataset {dataset.dataset_id!r}: no cleaned epochs matched "
            f"{dataset.epochs_dir / dataset.epoch_glob}"
        )
    metadata = _metadata_index(_read_metadata(dataset))
    raw_paths = _raw_lookup(dataset)
    records: list[CanonicalRecording] = []
    for epoch_path in epoch_paths:
        participant_id = subject_from_path(epoch_path)
        session_id = session_from_path(epoch_path)
        recording_id = recording_from_path(epoch_path)
        row = metadata.get(recording_id, metadata.get(participant_id, {}))
        group = normalize_group(_value(row, dataset.columns, "group"), participant_id)
        # Some medication-state datasets keep diagnosis in participants.tsv
        # and encode ON/OFF only in the BIDS session entity.
        session_text = (session_id or "").lower()
        if group == "PD" and "off" in session_text:
            group = "PD_OFF"
        elif group == "PD" and "on" in session_text:
            group = "PD_ON"
        records.append(
            CanonicalRecording(
                dataset_id=dataset.dataset_id,
                recording_id=recording_id,
                participant_id=participant_id,
                session_id=session_id,
                epoch_path=epoch_path.resolve(),
                raw_path=raw_paths.get(recording_id),
                group=group,
                medication_state=medication_state(group, _value(row, dataset.columns, "medication_state")),
                age_years=_as_float(_value(row, dataset.columns, "age_years")),
                sex=_clean(_value(row, dataset.columns, "sex")),
                updrs=_as_float(_value(row, dataset.columns, "updrs")),
                moca=_as_float(_value(row, dataset.columns, "moca")),
                mmse=_as_float(_value(row, dataset.columns, "mmse")),
            )
        )
    return records


def _clean(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    value = str(value).strip()
    return value or None


def _as_float(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value) or str(value).strip() == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def convert_config(
    config: GlobalConfig,
    dataset_ids: list[str] | tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Write one compressed canonical table per dataset and one global table."""
    canonical_root = config.output_root / "canonical"
    canonical_root.mkdir(parents=True, exist_ok=True)
    enabled = {dataset.dataset_id: dataset for dataset in config.enabled_datasets}
    selected_ids = tuple(enabled) if dataset_ids is None else tuple(dataset_ids)
    unknown = sorted(set(selected_ids) - set(enabled))
    if unknown:
        choices = ", ".join(sorted(enabled)) or "none"
        raise ValueError(f"Unknown or disabled dataset(s): {unknown}; enabled choices: {choices}")
    all_records: list[dict[str, Any]] = []
    dataset_manifest: list[dict[str, Any]] = []
    for dataset_id in selected_ids:
        dataset = enabled[dataset_id]
        records = convert_dataset(dataset)
        table = pd.DataFrame.from_records([record.as_dict() for record in records], columns=CANONICAL_COLUMNS)
        output_dir = canonical_root / dataset.dataset_id
        output_dir.mkdir(parents=True, exist_ok=True)
        table.to_csv(output_dir / "recordings.csv.gz", index=False, compression="gzip")
        (output_dir / "dataset.json").write_text(
            json.dumps(
                {
                    "dataset_id": dataset.dataset_id,
                    "n_recordings": len(table),
                    "groups": table["group"].value_counts().to_dict(),
                    "columns": list(CANONICAL_COLUMNS),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        all_records.extend(table.to_dict(orient="records"))
        dataset_manifest.append({"dataset_id": dataset.dataset_id, "n_recordings": len(table)})
    if not all_records:
        raise ValueError("No enabled datasets produced canonical recordings")
    result = pd.DataFrame.from_records(all_records, columns=CANONICAL_COLUMNS)
    result.to_csv(canonical_root / "recordings.csv.gz", index=False, compression="gzip")
    (canonical_root / "manifest.json").write_text(
        json.dumps({"schema_version": 1, "datasets": dataset_manifest}, indent=2),
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/global_pipeline.json")
    parser.add_argument("--datasets", nargs="+", help="Dataset ids to convert; defaults to every enabled dataset")
    args = parser.parse_args()
    from .schema import load_global_config

    config = load_global_config(args.config)
    table = convert_config(config, dataset_ids=args.datasets)
    print(f"Converted {len(table)} recordings from {table['dataset_id'].nunique()} datasets")


if __name__ == "__main__":
    main()
