"""Boundary tests for the shared descriptive MoCA classification."""

from __future__ import annotations

import unittest

import pandas as pd

from core.cognitive_status import classify_moca


class CognitiveStatusTests(unittest.TestCase):
    def test_25_is_impaired_and_26_through_30_are_normal(self) -> None:
        result = classify_moca(pd.Series([0, 25, 26, 30]))
        self.assertEqual(
            result.tolist(),
            [
                "cognitive_impairment",
                "cognitive_impairment",
                "cognitively_normal",
                "cognitively_normal",
            ],
        )

    def test_scores_above_30_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            classify_moca(pd.Series([31]))


if __name__ == "__main__":
    unittest.main()
