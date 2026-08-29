import unittest

import numpy as np
import pandas as pd

from analyses.medication.cohort import build_cohort
from analyses.medication.pipeline import load_analysis_config
from analyses.medication.statistics import (
    compute_condition_statistics,
    compute_mmse_statistics,
)
from core.dataset import (
    discover_recordings,
    load_subject,
    recording_id_from_path,
    session_id_from_path,
    subject_id_from_path,
)
from core.config import load_config as load_preprocessing_config


class MedicationDatasetTests(unittest.TestCase):
    def test_session_aware_bids_identifiers(self):
        path = "ds002778-1.0.5/sub-pd3/ses-off/eeg/sub-pd3_ses-off_task-rest_eeg.bdf"
        self.assertEqual(subject_id_from_path(path), "sub-pd3")
        self.assertEqual(session_id_from_path(path), "ses-off")
        self.assertEqual(recording_id_from_path(path), "sub-pd3_ses-off")

    def test_complete_local_cohort(self):
        participants, recordings = build_cohort(
            "ds002778-1.0.5",
            expected_counts={"HC": 16, "PD_OFF": 15, "PD_ON": 15},
        )
        self.assertEqual(len(participants), 31)
        self.assertEqual(len(recordings), 46)
        self.assertEqual(recordings["participant_id"].nunique(), 31)
        self.assertEqual(int(participants["mmse"].min()), 26)
        self.assertEqual(int(participants["mmse"].max()), 30)
        self.assertEqual(
            set(
                participants.loc[
                    participants["provenance_sensitivity_exclusion"],
                    "participant_id",
                ]
            ),
            {"sub-pd6", "sub-pd16"},
        )

    def test_bdf_discovery_preserves_all_sessions(self):
        recordings = discover_recordings("ds002778-1.0.5", "rest")
        identifiers = {recording_id_from_path(path) for path in recordings}
        self.assertEqual(len(recordings), 46)
        self.assertIn("sub-pd3_ses-off", identifiers)
        self.assertIn("sub-pd3_ses-on", identifiers)
        self.assertIn("sub-hc1_ses-hc", identifiers)

    def test_bdf_loader_retains_only_32_scalp_electrodes(self):
        path = next(
            path
            for path in discover_recordings("ds002778-1.0.5", "rest")
            if recording_id_from_path(path) == "sub-hc1_ses-hc"
        )
        auxiliary = [
            *(f"EXG{index}" for index in range(1, 9)),
            "Status",
        ]
        raw, provenance = load_subject(path, auxiliary)
        self.assertEqual(len(raw.ch_names), 32)
        self.assertIn("Pz", raw.ch_names)
        self.assertNotIn("EXG1", raw.ch_names)
        self.assertEqual(set(provenance["dropped_auxiliary_channels"]), set(auxiliary))

    def test_analysis_config_is_prespecified(self):
        config = load_analysis_config("config/analyses/ds002778.json")
        self.assertEqual(config["statistics"]["minimum_pairs"], 10)
        self.assertEqual(config["ordinal"]["embedding_dimension"], 6)

    def test_biosemi_preprocessing_contract_is_dataset_specific(self):
        config = load_preprocessing_config("config/preprocessing_ds002778.yaml")
        self.assertEqual(config["project"]["task"], "rest")
        self.assertEqual(config["epochs"]["peak_to_peak_uv"], 500.0)
        self.assertEqual(len(config["channels"]["auxiliary_names"]), 9)
        self.assertIn("Status", config["channels"]["auxiliary_names"])


class MedicationStatisticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        recording_rows = []
        feature_rows = []
        for index in range(16):
            participant_id = f"sub-hc{index + 1}"
            recording_id = f"{participant_id}_ses-hc"
            mmse = 26 + index % 5
            recording_rows.append(
                {
                    "recording_id": recording_id,
                    "participant_id": participant_id,
                    "condition": "HC",
                    "age_years": 52 + index,
                    "sex_male": index % 2,
                    "mmse": mmse,
                    "provenance_sensitivity_exclusion": False,
                }
            )
            feature_rows.append(cls._feature(recording_id, 0.05 * index + 0.02 * (index % 3)))
        for index in range(15):
            participant_id = f"sub-pd{index + 1}"
            mmse = 26 + index % 5
            age = 51 + index
            off = 2.5 + 0.04 * index + 0.2 * (mmse - 28) + 0.03 * (index % 2)
            delta = -0.8 + 0.4 * (mmse - 28) + 0.015 * (index % 3)
            for condition, session, value in (
                ("PD_OFF", "ses-off", off),
                ("PD_ON", "ses-on", off + delta),
            ):
                recording_id = f"{participant_id}_{session}"
                recording_rows.append(
                    {
                        "recording_id": recording_id,
                        "participant_id": participant_id,
                        "condition": condition,
                        "age_years": age,
                        "sex_male": index % 2,
                        "mmse": mmse,
                        "provenance_sensitivity_exclusion": index in {5, 14},
                    }
                )
                feature_rows.append(cls._feature(recording_id, value))
        cls.recordings = pd.DataFrame.from_records(recording_rows)
        cls.features = pd.DataFrame.from_records(feature_rows)
        cls.config = {
            "statistics": {
                "minimum_per_condition": 10,
                "minimum_pairs": 10,
                "minimum_mmse_participants": 10,
                "confidence_level": 0.95,
                "bootstrap_resamples": 200,
                "random_seed": 42,
                "fdr_alpha": 0.05,
            }
        }

    @staticmethod
    def _feature(recording_id, value):
        return {
            "recording_id": recording_id,
            "duration_variant": "all_retained",
            "feature_id": "psd_beta_relative_power",
            "family": "psd",
            "domain": "psd",
            "band": "beta",
            "metric": "relative_power",
            "value": value,
        }

    def test_condition_contrasts_keep_pd_sessions_paired(self):
        result = compute_condition_statistics(
            self.features, self.recordings, self.config
        )
        primary = result.loc[result["sensitivity_cohort"].eq("all_participants")]
        self.assertEqual(set(primary["contrast"]), {
            "PD_OFF_minus_HC",
            "PD_ON_minus_HC",
            "PD_ON_minus_PD_OFF",
        })
        paired = primary.loc[primary["contrast"].eq("PD_ON_minus_PD_OFF")].iloc[0]
        self.assertEqual(paired["n_pairs"], 15)
        self.assertLess(paired["effect"], 0.0)
        self.assertEqual(paired["analysis_status"], "ok")

    def test_mmse_model_detects_medication_delta_slope(self):
        result = compute_mmse_statistics(self.features, self.recordings, self.config)
        row = result.loc[
            result["sensitivity_cohort"].eq("all_participants")
            & result["mmse_model"].eq("PD_ON_minus_PD_OFF")
        ].iloc[0]
        self.assertEqual(row["n_participants"], 15)
        self.assertEqual(row["analysis_status"], "ok")
        self.assertAlmostEqual(row["mmse_slope"], 0.4, delta=0.04)
        self.assertTrue(np.isfinite(row["statistic"]))
        self.assertTrue(np.isfinite(row["primary_p_value"]))


if __name__ == "__main__":
    unittest.main()
