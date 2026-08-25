import json
import unittest

import numpy as np
import pandas as pd

from exploration.features import (
    FORBIDDEN_MODEL_COLUMNS,
    build_feature_table,
    summarize_typical_bout_shapes,
    validate_model_features,
)
from exploration.modeling import (
    average_repeated_predictions,
    bootstrap_performance,
    run_nested_validation,
)
from exploration.matching import match_control_pd_pairs, remove_demographic_predictors
from exploration.pipeline import load_exploration_config


class ExplorationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_exploration_config("exploration/config.json")

    def test_real_feature_table_is_one_row_per_subject_and_leakage_free(self):
        table, provenance = build_feature_table(self.config)
        self.assertEqual(len(table), 149)
        self.assertEqual(table["subject_id"].nunique(), 149)
        self.assertEqual(table["target_pd"].value_counts().to_dict(), {1: 100, 0: 49})
        self.assertFalse({"ID", "EEG", "TYPE", "UPDRS"} & set(table.columns))
        self.assertTrue(np.isfinite(table["ordinal_global_entropy"]).all())
        self.assertTrue(
            np.isfinite(table["ordinal_global_renyi_entropy_alpha_0_1"]).all()
        )
        self.assertTrue(np.isfinite(table["aperiodic_exponent"]).all())
        self.assertTrue(np.isfinite(table["bout_alpha_bouts_per_minute"]).all())
        self.assertTrue(
            np.isfinite(table["typical_alpha_envelope_peak_ratio"]).all()
        )
        self.assertEqual(
            set(provenance.loc[~provenance["included"], "feature"]),
            {"participant_id", "ID", "EEG", "TYPE", "UPDRS", "GROUP"},
        )

    def test_no_model_contains_forbidden_or_overlapping_psd_features(self):
        table, _ = build_feature_table(self.config)
        validate_model_features(table, self.config["models"])
        for specification in self.config["models"].values():
            features = set(specification["features"])
            self.assertFalse(features & FORBIDDEN_MODEL_COLUMNS)
            self.assertFalse(any("broad_5_15" in feature for feature in features))

    def test_nested_validation_returns_repeated_out_of_fold_predictions(self):
        rng = np.random.default_rng(7)
        n_subjects = 60
        first = rng.normal(size=n_subjects)
        truth = (first + rng.normal(scale=0.7, size=n_subjects) > 0.0).astype(int)
        table = pd.DataFrame(
            {
                "subject_id": [f"sub-{index:03d}" for index in range(n_subjects)],
                "target_pd": truth,
                "first": first,
                "second": rng.normal(size=n_subjects),
            }
        )
        models = {
            "test": {
                "label": "Test model",
                "role": "test",
                "features": ["first", "second"],
            }
        }
        validation = {
            "outer_folds": 3,
            "outer_repeats": 2,
            "inner_folds": 2,
            "c_grid": [0.1, 1.0],
            "classification_threshold": 0.5,
            "random_seed": 11,
            "primary_metric": "roc_auc",
        }
        predictions, metrics, coefficients = run_nested_validation(
            table, models, validation
        )
        self.assertEqual(len(predictions), n_subjects * 2)
        self.assertTrue((predictions.groupby("subject_id").size() == 2).all())
        self.assertEqual(len(metrics), 3 * 2 * 7)
        self.assertEqual(len(coefficients), 3 * 2 * 2)
        averaged = average_repeated_predictions(predictions)
        self.assertEqual(len(averaged), n_subjects)
        performance = bootstrap_performance(averaged, n_resamples=30, seed=3)
        self.assertEqual(set(performance["metric"]), {
            "roc_auc",
            "average_precision",
            "balanced_accuracy",
            "sensitivity",
            "specificity",
            "brier_score",
            "log_loss",
        })

    def test_primary_model_is_prespecified_d6_tau1(self):
        self.assertEqual(
            self.config["primary_ordinal_parameters"],
            {"embedding_dimension": 6, "delay_samples": 1},
        )
        self.assertEqual(
            self.config["models"]["ordinal_adjusted"]["role"], "primary"
        )

    def test_typical_bout_reduction_is_complete_and_bounded(self):
        bands = self.config["candidate_features"]["bout_bands"]
        table = summarize_typical_bout_shapes(
            self.config["input"]["typical_bout_file"], bands
        )
        self.assertEqual(len(table), 149)
        self.assertEqual(table["subject_id"].nunique(), 149)
        for band in bands:
            self.assertTrue((table[f"typical_{band}_envelope_peak_ratio"] > 1.0).all())
            self.assertTrue(
                (table[f"typical_{band}_envelope_half_height_width_s"] > 0.0).all()
            )
            self.assertTrue(
                table[f"typical_{band}_envelope_asymmetry"].between(-1.0, 1.0).all()
            )
            self.assertTrue(
                table[f"typical_{band}_relative_phase_consistency"].between(0.0, 1.0).all()
            )

    def test_demographic_matching_is_exact_balanced_and_pair_grouped(self):
        table, _ = build_feature_table(self.config)
        matched, pairs, balance = match_control_pd_pairs(
            table,
            maximum_age_difference_years=5.0,
        )
        self.assertEqual(len(pairs), 49)
        self.assertEqual(len(matched), 98)
        self.assertEqual(matched["target_pd"].value_counts().to_dict(), {0: 49, 1: 49})
        self.assertLessEqual(pairs["absolute_age_difference_years"].max(), 5.0)
        self.assertTrue((matched.groupby("cv_group")["target_pd"].nunique() == 2).all())
        matched_balance = balance.loc[balance["cohort"].eq("matched")]
        self.assertTrue(
            (matched_balance["standardized_mean_difference_pd_minus_control"].abs() < 0.01).all()
        )
        models = remove_demographic_predictors(self.config["models"])
        self.assertNotIn("demographics", models)
        self.assertFalse(
            any(
                feature in {"age_years", "sex_male"}
                for specification in models.values()
                for feature in specification["features"]
            )
        )

    def test_matched_nested_validation_keeps_pairs_together(self):
        rng = np.random.default_rng(19)
        pair_ids = np.repeat([f"pair-{index:02d}" for index in range(20)], 2)
        truth = np.tile([0, 1], 20)
        table = pd.DataFrame(
            {
                "subject_id": [f"sub-{index:03d}" for index in range(40)],
                "target_pd": truth,
                "cv_group": pair_ids,
                "feature": truth + rng.normal(scale=0.8, size=40),
            }
        )
        models = {"paired": {"label": "Paired", "role": "test", "features": ["feature"]}}
        validation = {
            "outer_folds": 4,
            "outer_repeats": 2,
            "inner_folds": 3,
            "c_grid": [0.1, 1.0],
            "classification_threshold": 0.5,
            "threshold_policy": "inner_youden",
            "random_seed": 23,
            "primary_metric": "roc_auc",
        }
        predictions, _, _ = run_nested_validation(table, models, validation)
        pair_folds = predictions.groupby(["repeat", "cv_group"])["fold"].nunique()
        self.assertTrue((pair_folds == 1).all())


if __name__ == "__main__":
    unittest.main()
