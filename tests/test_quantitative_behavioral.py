import unittest

import numpy as np
import pandas as pd

from quantitative_behavioral.features import (
    build_dimension_sensitivity_features,
    build_subject_features,
)
from quantitative_behavioral.pipeline import load_analysis_config
from quantitative_behavioral.statistics import (
    correlate_subject_features,
    fdr_bh,
    partial_spearman,
    unadjusted_spearman,
)


class QuantitativeBehavioralTests(unittest.TestCase):
    def test_config_prespecifies_pd_moca_and_regular_ordinal_metrics(self):
        config = load_analysis_config("quantitative_behavioral/config.json")
        self.assertEqual(config["analysis"]["primary_group"], "PD")
        self.assertEqual(config["analysis"]["outcome_column"], "MOCA")
        self.assertEqual(config["analysis"]["covariates"], ["AGE", "GENDER"])
        self.assertEqual(
            config["features"]["ordinal_metrics"],
            ["entropy", "complexity", "fisher_information"],
        )
        self.assertEqual(config["expected"]["shared_electrodes"], 60)
        self.assertEqual(
            config["dimension_sensitivity"]["embedding_dimensions"], [3, 4, 5, 6]
        )
        self.assertEqual(config["dimension_sensitivity"]["delay_samples"], 1)
        self.assertEqual(
            config["dimension_sensitivity"]["fdr_scope"],
            "within_each_dimension_across_all_63_features_per_method",
        )

    def test_partial_spearman_removes_age_confounding(self):
        rng = np.random.default_rng(42)
        age = np.linspace(45.0, 85.0, 500)
        x = age + rng.normal(0.0, 3.0, len(age))
        y = age + rng.normal(0.0, 3.0, len(age))
        sex = np.tile([0.0, 1.0], len(age) // 2)
        raw, _ = unadjusted_spearman(x, y)
        adjusted, _ = partial_spearman(x, y, np.column_stack([age, sex]))
        self.assertGreater(raw, 0.8)
        self.assertLess(abs(adjusted), 0.15)

    def test_partial_spearman_retains_independent_monotonic_signal(self):
        rng = np.random.default_rng(7)
        age = rng.uniform(50.0, 85.0, 300)
        sex = rng.integers(0, 2, len(age)).astype(float)
        x = rng.normal(size=len(age))
        y = 2.5 * x + 0.4 * age + rng.normal(0.0, 0.2, len(age))
        adjusted, p_value = partial_spearman(x, y, np.column_stack([age, sex]))
        self.assertGreater(adjusted, 0.9)
        self.assertLess(p_value, 1e-20)

    def test_fdr_is_applied_only_to_finite_p_values(self):
        adjusted, rejected = fdr_bh(
            np.asarray([0.001, 0.02, 0.5, np.nan]), alpha=0.05
        )
        self.assertTrue(rejected[0])
        self.assertTrue(rejected[1])
        self.assertFalse(rejected[2])
        self.assertFalse(rejected[3])
        self.assertTrue(np.isnan(adjusted[3]))

    def test_subject_correlations_use_one_row_per_subject(self):
        rng = np.random.default_rng(3)
        n_subjects = 100
        feature = rng.normal(size=n_subjects)
        table = pd.DataFrame(
            {
                "subject_id": [f"sub-{index:03d}" for index in range(n_subjects)],
                "group": "PD",
                "moca": 24.0 + feature + rng.normal(0.0, 1.0, n_subjects),
                "age_years": rng.integers(48, 87, n_subjects),
                "sex_male": rng.integers(0, 2, n_subjects),
                "value": feature,
                "feature_id": "ordinal_broadband_entropy",
            }
        )
        dictionary = pd.DataFrame(
            {
                "feature_id": ["ordinal_broadband_entropy"],
                "family": ["ordinal_broadband"],
                "domain": ["ordinal"],
                "band": ["broadband"],
                "metric": ["entropy"],
                "feature_label": ["Broadband permutation entropy H"],
                "unit": ["normalized"],
                "source_file": ["synthetic"],
                "analysis_level": ["subject_mean_across_shared_electrodes"],
            }
        )
        config = load_analysis_config("quantitative_behavioral/config.json")
        config["analysis"]["bootstrap_resamples"] = 100
        result = correlate_subject_features(table, dictionary, config)
        self.assertEqual(len(result), 2)
        self.assertTrue(result["n_subjects"].eq(100).all())
        self.assertEqual(
            set(result["method"]),
            {"partial_spearman_age_sex", "spearman_unadjusted"},
        )

    def test_real_feature_table_is_subject_balanced_and_excludes_renyi(self):
        config = load_analysis_config("quantitative_behavioral/config.json")
        cohort, features, dictionary = build_subject_features(config)
        self.assertEqual(len(cohort), 149)
        self.assertEqual(int(cohort["group"].eq("PD").sum()), 100)
        self.assertEqual(len(dictionary), 53)
        self.assertFalse(dictionary["feature_id"].str.contains("renyi").any())
        self.assertFalse(features.duplicated(["subject_id", "feature_id"]).any())
        pd_features = features.loc[features["group"].eq("PD")]
        self.assertEqual(len(pd_features), 100 * 53)
        self.assertTrue(pd_features["value"].notna().all())

    def test_dimension_blocks_have_63_balanced_regular_and_renyi_features(self):
        config = load_analysis_config("quantitative_behavioral/config.json")
        cohort, _, _ = build_subject_features(config)
        features, dictionary, electrode_features, electrode_order = (
            build_dimension_sensitivity_features(config, cohort)
        )
        self.assertEqual(len(dictionary), 252)
        self.assertEqual(set(dictionary["embedding_dimension"]), {3, 4, 5, 6})
        self.assertEqual(set(dictionary["delay_samples"]), {1})
        self.assertEqual(dictionary.groupby("embedding_dimension").size().to_dict(), {
            3: 63,
            4: 63,
            5: 63,
            6: 63,
        })
        self.assertEqual(
            set(dictionary["family"]),
            {"ordinal_D3", "ordinal_D4", "ordinal_D5", "ordinal_D6"},
        )
        self.assertEqual(int(dictionary["feature_id"].str.contains("renyi").sum()), 168)
        self.assertEqual(
            set(dictionary.loc[dictionary["quantity_set"].ne("regular"), "renyi_alpha"]),
            {0.9, 1.1, 2.0},
        )
        self.assertEqual(len(electrode_order), 60)
        self.assertFalse(features.duplicated(["subject_id", "feature_id"]).any())
        self.assertFalse(
            electrode_features.duplicated(
                ["subject_id", "electrode", "feature_id"]
            ).any()
        )
        self.assertEqual(
            len(features.loc[features["group"].eq("PD")]), 100 * 252
        )


if __name__ == "__main__":
    unittest.main()
