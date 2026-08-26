"""Tests for the prespecified eight-electrode severity-axis pipeline."""

from __future__ import annotations

import unittest

import pandas as pd

from disease_progression.features import (
    build_selected_electrode_features,
    load_pd_cohort,
)
from disease_progression.pipeline import EXPECTED_ELECTRODES, load_analysis_config
from disease_progression.statistics import correlate_progression_features


class DiseaseProgressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_analysis_config("disease_progression/config.json")

    def test_config_prespecifies_electrodes_outcomes_and_nonoverlapping_bands(self) -> None:
        self.assertEqual(self.config["electrodes"], EXPECTED_ELECTRODES)
        self.assertEqual(self.config["analysis"]["primary_outcome"], "updrs")
        self.assertEqual(self.config["analysis"]["secondary_outcomes"], ["moca"])
        self.assertEqual(self.config["analysis"]["covariates"], ["age_years", "sex_male"])
        self.assertNotIn("broad_5_15", self.config["features"]["ordinal_bands"])
        self.assertNotIn("broad_5_15", self.config["features"]["psd_bands"])
        self.assertNotIn("broad_5_15", self.config["features"]["bout_bands"])

    def test_real_feature_matrix_uses_eight_electrodes_and_one_row_per_subject(self) -> None:
        cohort = load_pd_cohort(self.config)
        features, dictionary = build_selected_electrode_features(self.config, cohort)
        self.assertEqual(len(cohort), 100)
        self.assertEqual(len(dictionary), 141)
        self.assertEqual(len(features), len(cohort) * len(dictionary))
        self.assertFalse(features.duplicated(["subject_id", "feature_id"]).any())
        self.assertTrue(features["n_electrodes_contributing"].between(0, 8).all())
        self.assertFalse(dictionary["feature_id"].str.contains("broad_5_15").any())

    def test_statistics_keep_updrs_and_moca_as_separate_axes(self) -> None:
        subjects = pd.DataFrame(
            {
                "subject_id": [f"sub-{index:03d}" for index in range(40)],
                "group": "PD",
                "value": list(range(40)),
                "feature_id": "test_feature",
                "age_years": [60 + index % 10 for index in range(40)],
                "sex_male": [index % 2 for index in range(40)],
                "updrs": list(range(1, 41)),
                "moca": list(range(40, 0, -1)),
            }
        )
        dictionary = pd.DataFrame.from_records(
            [
                {
                    "feature_id": "test_feature",
                    "feature_label": "Test",
                    "family": "test",
                    "domain": "test",
                    "band": "broadband",
                    "metric": "test",
                    "unit": "unitless",
                    "aggregation": "mean",
                    "source_file": "synthetic",
                }
            ]
        )
        config = {
            "analysis": {
                **self.config["analysis"],
                "bootstrap_resamples": 100,
            }
        }
        result = correlate_progression_features(subjects, dictionary, config)
        self.assertEqual(len(result), 4)
        adjusted = result.loc[result["method"].eq("partial_spearman_age_sex")]
        updrs = adjusted.loc[adjusted["outcome"].eq("updrs")].iloc[0]
        moca = adjusted.loc[adjusted["outcome"].eq("moca")].iloc[0]
        self.assertGreater(updrs["estimate"], 0.9)
        self.assertLess(moca["estimate"], -0.9)
        self.assertGreater(updrs["progression_aligned_estimate"], 0.9)
        self.assertGreater(moca["progression_aligned_estimate"], 0.9)


if __name__ == "__main__":
    unittest.main()
