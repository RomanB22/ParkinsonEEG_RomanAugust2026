"""Tests for the dataset-agnostic input converter."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from global_pipeline.converter import convert_config
from global_pipeline.schema import load_global_config, read_canonical_table


class GlobalPipelineInputTests(unittest.TestCase):
    def test_converter_normalizes_session_conditions_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            epochs = root / "epochs"
            epochs.mkdir()
            for name in (
                "sub-pd01_ses-off_task-rest_desc-cleaned_epo.fif",
                "sub-pd01_ses-on_task-rest_desc-cleaned_epo.fif",
                "sub-hc01_ses-hc_task-rest_desc-cleaned_epo.fif",
            ):
                (epochs / name).touch()
            metadata = root / "participants.tsv"
            pd.DataFrame(
                [
                    {"participant_id": "sub-pd01", "diagnosis": "PD", "age": "71", "sex": "F", "mmse": "28"},
                    {"participant_id": "sub-hc01", "diagnosis": "Control", "age": "69", "sex": "M", "mmse": "30"},
                ]
            ).to_csv(metadata, sep="\t", index=False)
            config_path = root / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "output_root": "out",
                        "bands": {"alpha": [8, 13]},
                        "datasets": [
                            {
                                "id": "study",
                                "metadata": str(metadata),
                                "epochs_dir": str(epochs),
                                "epoch_glob": "*.fif",
                                "columns": {"group": "diagnosis", "age_years": "age", "sex": "sex", "mmse": "mmse"},
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            config = load_global_config(config_path)
            result = convert_config(config)
            self.assertEqual(set(result["group"]), {"PD_OFF", "PD_ON", "Control"})
            self.assertEqual(result["recording_id"].nunique(), 3)
            self.assertEqual(result.loc[result["group"].eq("PD_OFF"), "mmse"].iloc[0], 28.0)
            canonical = read_canonical_table(config.output_root / "canonical" / "recordings.csv.gz")
            self.assertEqual(set(canonical.columns), set(result.columns))

    def test_four_dataset_configuration_is_supported_without_hard_coded_names(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            datasets = []
            for index in range(4):
                folder = root / f"dataset_{index + 1}"
                (folder / "epochs").mkdir(parents=True)
                (folder / "epochs" / f"sub-hc{index + 1}_task-rest_desc-cleaned_epo.fif").touch()
                datasets.append(
                    {
                        "id": f"study_{index + 1}",
                        "epochs_dir": str(folder / "epochs"),
                        "epoch_glob": "*.fif",
                    }
                )
            path = root / "config.json"
            path.write_text(
                json.dumps({"schema_version": 1, "bands": {"alpha": [8, 13]}, "datasets": datasets}),
                encoding="utf-8",
            )
            config = load_global_config(path)
            self.assertEqual(len(config.enabled_datasets), 4)
            result = convert_config(config)
            self.assertEqual(result["dataset_id"].nunique(), 4)
            selected = convert_config(config, dataset_ids=["study_2"])
            self.assertEqual(set(selected["dataset_id"]), {"study_2"})


if __name__ == "__main__":
    unittest.main()
