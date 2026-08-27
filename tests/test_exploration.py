import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from exploration.features import (
    FORBIDDEN_MODEL_COLUMNS,
    build_feature_table,
    discover_completed_sweeps,
    summarize_typical_bout_shapes,
    validate_model_features,
)
from exploration.modeling import (
    average_repeated_predictions,
    bootstrap_performance,
    run_nested_validation,
)
from exploration.matching import (
    apply_precomputed_control_pd_pairs,
    match_control_pd_pairs,
    remove_demographic_predictors,
)
from exploration.pipeline import load_exploration_config
from matched_analysis.prepare_matched_cohort import prepare_matched_cohort


HAS_GENERATED_FEATURES = Path("processed/metadata/subjects.csv").is_file() and Path(
    "ordinal_analysis/processed/metrics/subject_electrode_mean_metrics.csv"
).is_file()
HAS_TYPICAL_BOUTS = Path(
    "scale_free_analysis/processed/intermediate/typical_bouts/subject_electrode_band_envelopes.npz"
).is_file()


class ExplorationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_exploration_config("exploration/config.json")

    @unittest.skipUnless(HAS_GENERATED_FEATURES, "requires generated feature caches")
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

    @unittest.skipUnless(HAS_GENERATED_FEATURES, "requires generated feature caches")
    def test_no_model_contains_forbidden_features(self):
        table, _ = build_feature_table(self.config)
        validate_model_features(table, self.config["models"])
        for specification in self.config["models"].values():
            features = set(specification["features"])
            self.assertFalse(features & FORBIDDEN_MODEL_COLUMNS)
            self.assertFalse(any("broad_5_15" in feature for feature in features))
        self.assertNotIn(
            "descriptive_only_bout_bands", self.config["candidate_features"]
        )
        self.assertEqual(
            self.config["candidate_features"]["bout_bands"],
            ["theta", "alpha", "low_beta", "high_beta"],
        )

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
        self.assertEqual(
            self.config["ordinal_sweep"],
            {"expected_dimensions": [3, 4, 5, 6], "expected_delays": [1]},
        )

    def test_sweep_discovery_ignores_legacy_nonunit_delays(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for setting in ("D2_tau1", "D3_tau1", "D4_tau1", "D4_tau5", "D6_tau10"):
                metrics = root / setting / "metrics"
                metrics.mkdir(parents=True)
                (metrics / "subject_electrode_mean_metrics.csv").touch()
            config = json.loads(json.dumps(self.config))
            config["input"]["ordinal_sweep_root"] = str(root)
            completed = discover_completed_sweeps(config)
            self.assertEqual(
                [(row["embedding_dimension"], row["delay_samples"]) for row in completed],
                [(3, 1), (4, 1)],
            )

    @unittest.skipUnless(HAS_TYPICAL_BOUTS, "requires generated typical-bout cache")
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

    @unittest.skipUnless(HAS_GENERATED_FEATURES, "requires generated feature caches")
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
        self.assertNotIn("ordinal_core", models)
        self.assertFalse(
            any(
                feature in {"age_years", "sex_male"}
                for specification in models.values()
                for feature in specification["features"]
            )
        )

    @unittest.skipUnless(HAS_GENERATED_FEATURES, "requires generated participant metadata")
    def test_canonical_matched_manifest_drives_every_pipeline_config(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = prepare_matched_cohort(output_root=directory)
            self.assertEqual(manifest["n_pairs"], 49)
            self.assertEqual(manifest["n_subjects"], 98)
            matched_path = str(Path(directory) / "matched_subjects.csv")
            for config_path in manifest["generated_configs"].values():
                config = json.loads(Path(config_path).read_text(encoding="utf-8"))
                self.assertEqual(config["input"]["participants_file"], matched_path)
                self.assertTrue(config["output_dir"].endswith("processed_matched"))
                if Path(config_path).name == "exploration.json":
                    self.assertEqual(
                        config["demographic_matching"]["precomputed_pairs_file"],
                        str(Path(directory) / "demographic_match_pairs.csv"),
                    )
                if Path(config_path).name == "bycycle_burst.json":
                    self.assertEqual(
                        config["input"]["reference_ebosc_output_dir"],
                        "scale_free_analysis/processed_matched",
                    )
                if Path(config_path).name == "ordinal.json":
                    self.assertEqual(
                        config["input"]["feature_source_output_dir"],
                        "ordinal_analysis/processed",
                    )
                    self.assertEqual(
                        config["input"]["feature_source_sweep_root"],
                        "ordinal_analysis/parameter_sweep",
                    )
                if Path(config_path).name == "scale_free.json":
                    self.assertEqual(
                        config["input"]["feature_source_output_dir"],
                        "scale_free_analysis/processed",
                    )

    @unittest.skipUnless(HAS_GENERATED_FEATURES, "requires generated feature caches")
    def test_precomputed_pairs_are_validated_without_double_matching(self):
        table, _ = build_feature_table(self.config)
        matched, pairs, balance = match_control_pd_pairs(
            table,
            maximum_age_difference_years=5.0,
        )
        reapplied, reused_pairs, reused_balance = apply_precomputed_control_pd_pairs(
            matched,
            pairs,
            balance,
            maximum_age_difference_years=5.0,
        )
        self.assertEqual(len(reapplied), 98)
        self.assertTrue(reapplied["cv_group"].eq(reapplied["match_pair_id"]).all())
        pd.testing.assert_frame_equal(
            reused_pairs.reset_index(drop=True), pairs.reset_index(drop=True)
        )
        pd.testing.assert_frame_equal(
            reused_balance.reset_index(drop=True), balance.reset_index(drop=True)
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
