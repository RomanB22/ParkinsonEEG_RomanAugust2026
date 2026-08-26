import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.ica import _add_iclabel_scores, proposed_ica_exclusions
from src.config import load_config, preprocessing_signature
from scripts.run_preprocessing import (
    _record_parallel_ica_proposal,
    _subject_output_is_complete,
)


class IcaLabelScoringTests(unittest.TestCase):
    def setUp(self):
        self.config = {
            "iclabel_artifact_probability_threshold": 0.60,
            "iclabel_minimum_class_probability": 0.30,
        }
        self.base_scores = pd.DataFrame(
            {
                "component": [0, 1, 2, 3],
                "label": ["IC000", "IC001", "IC002", "IC003"],
                "suggested_ocular_review": [True, False, False, False],
            }
        )

    def test_components_are_ranked_artifact_to_brain(self):
        # Columns: brain, muscle, eye, heart, line, channel, other.
        probabilities = np.asarray(
            [
                [0.02, 0.03, 0.92, 0.01, 0.00, 0.01, 0.01],
                [0.90, 0.02, 0.02, 0.01, 0.00, 0.01, 0.04],
                [0.20, 0.55, 0.10, 0.02, 0.01, 0.02, 0.10],
                [0.03, 0.01, 0.01, 0.00, 0.00, 0.00, 0.95],
            ]
        )
        ranked = _add_iclabel_scores(self.base_scores, probabilities, self.config)
        self.assertEqual(ranked["component"].tolist(), [0, 2, 3, 1])
        self.assertEqual(ranked["artifact_rank"].tolist(), [1, 2, 3, 4])
        self.assertEqual(ranked.loc[0, "iclabel_predicted_label"], "eye blink")

    def test_known_artifacts_above_60_percent_total_are_proposed(self):
        probabilities = np.asarray(
            [
                [0.02, 0.03, 0.92, 0.01, 0.00, 0.01, 0.01],
                [0.90, 0.02, 0.02, 0.01, 0.00, 0.01, 0.04],
                [0.20, 0.55, 0.10, 0.02, 0.01, 0.02, 0.10],
                [0.03, 0.01, 0.01, 0.00, 0.00, 0.00, 0.95],
            ]
        )
        ranked = _add_iclabel_scores(self.base_scores, probabilities, self.config)
        components, reasons = proposed_ica_exclusions(ranked)
        self.assertEqual(components, [0, 2])
        self.assertIn("eye blink", reasons[0])
        self.assertTrue(ranked.loc[ranked["component"] == 0, "proposed_exclusion"].item())
        self.assertTrue(ranked.loc[ranked["component"] == 2, "proposed_exclusion"].item())
        self.assertFalse(ranked["automatic_removal"].any())

    def test_mixed_artifact_is_proposed_when_total_probability_is_high(self):
        probabilities = np.asarray(
            [
                [0.01, 0.40, 0.46, 0.01, 0.00, 0.08, 0.04],
                [0.90, 0.02, 0.02, 0.01, 0.00, 0.01, 0.04],
                [0.20, 0.25, 0.25, 0.10, 0.05, 0.05, 0.10],
                [0.03, 0.01, 0.01, 0.00, 0.00, 0.00, 0.95],
            ]
        )
        ranked = _add_iclabel_scores(self.base_scores, probabilities, self.config)
        mixed = ranked.loc[ranked["component"] == 0].iloc[0]
        self.assertAlmostEqual(mixed["iclabel_artifact_probability"], 0.95)
        self.assertAlmostEqual(mixed["iclabel_strongest_artifact_probability"], 0.46)
        self.assertTrue(mixed["proposed_exclusion"])

    def test_parallel_automatic_proposal_is_serially_recorded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "ica": {
                            "manual_review_confirmed": {},
                            "manual_exclude_components": {},
                            "manual_exclude_reasons": {},
                        }
                    }
                ),
                encoding="utf-8",
            )
            decisions_path = root / "output" / "qc" / "sub-001" / "decisions.json"
            decisions_path.parent.mkdir(parents=True)
            decisions_path.write_text(
                json.dumps(
                    {
                        "iclabel_proposed_exclusions": [2, 5],
                        "iclabel_proposal_reasons": {
                            "2": "eye blink",
                            "5": "muscle artifact",
                        },
                        "iclabel_proposal_written_to_config": False,
                    }
                ),
                encoding="utf-8",
            )
            _record_parallel_ica_proposal(
                config_path,
                root / "output",
                "sub-001",
                automatic=True,
            )
            updated = json.loads(config_path.read_text(encoding="utf-8"))
            self.assertEqual(
                updated["ica"]["automatic_exclude_components"]["sub-001"],
                [2, 5],
            )
            decisions = json.loads(decisions_path.read_text(encoding="utf-8"))
            self.assertTrue(decisions["iclabel_proposal_written_to_config"])

    def test_resume_requires_all_outputs_for_the_requested_stage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subject_id = "sub-001"
            ica_path = (
                root
                / "ica"
                / f"{subject_id}_task-Rest_desc-preprocessing-ica.fif"
            )
            decisions_path = root / "qc" / subject_id / "decisions.json"
            ica_path.parent.mkdir(parents=True)
            decisions_path.parent.mkdir(parents=True)
            ica_path.touch()
            decisions_path.touch()

            self.assertTrue(
                _subject_output_is_complete(root, subject_id, review_only=True)
            )
            self.assertFalse(
                _subject_output_is_complete(root, subject_id, review_only=False)
            )

            cleaned_path = (
                root
                / "cleaned_raw"
                / f"{subject_id}_task-Rest_desc-cleaned_raw.fif"
            )
            epochs_path = (
                root
                / "epochs"
                / f"{subject_id}_task-Rest_desc-cleaned_epo.fif"
            )
            cleaned_path.parent.mkdir(parents=True)
            epochs_path.parent.mkdir(parents=True)
            cleaned_path.touch()
            epochs_path.touch()
            self.assertTrue(
                _subject_output_is_complete(root, subject_id, review_only=False)
            )

    def test_resume_rejects_outputs_from_an_old_preprocessing_contract(self):
        config = load_config("config/preprocessing.yaml")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subject_id = "sub-001"
            required = (
                root / "ica" / f"{subject_id}_task-Rest_desc-preprocessing-ica.fif",
                root / "qc" / subject_id / "decisions.json",
            )
            for path in required:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            required[1].write_text(
                json.dumps({"preprocessing_signature": "old-contract"}),
                encoding="utf-8",
            )
            self.assertFalse(
                _subject_output_is_complete(
                    root, subject_id, review_only=True, config=config
                )
            )
            required[1].write_text(
                json.dumps(
                    {"preprocessing_signature": preprocessing_signature(config)}
                ),
                encoding="utf-8",
            )
            self.assertTrue(
                _subject_output_is_complete(
                    root, subject_id, review_only=True, config=config
                )
            )


if __name__ == "__main__":
    unittest.main()
