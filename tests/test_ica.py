import unittest

import numpy as np
import pandas as pd

from src.ica import _add_iclabel_scores, proposed_ica_exclusions


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


if __name__ == "__main__":
    unittest.main()
