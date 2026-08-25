import unittest

import numpy as np
import pandas as pd

from duration_qc_analysis.pipeline import (
    compare_group_features,
    load_duration_config,
    select_duration_cohort,
)


class DurationQCSensitivityTests(unittest.TestCase):
    def setUp(self):
        self.modeling = pd.DataFrame(
            {
                "subject_id": [f"c{index}" for index in range(4)]
                + [f"p{index}" for index in range(4)],
                "group": ["Control"] * 4 + ["PD"] * 4,
                "target_pd": [0] * 4 + [1] * 4,
                "age_years": [60, 61, 62, 63, 60, 61, 62, 63],
                "sex_male": [0, 1, 0, 1, 0, 1, 0, 1],
                "ordinal_global_entropy": [
                    0.40,
                    0.45,
                    0.50,
                    0.55,
                    0.60,
                    0.65,
                    0.70,
                    0.75,
                ],
            }
        )
        self.qc = self.modeling[["subject_id", "group"]].assign(
            usable_duration_sec=[120, 56, 124, 128, 120, 124, 40, 128],
            n_epochs_retained=[30, 14, 31, 32, 30, 31, 10, 32],
            percent_epochs_retained=[100, 50, 100, 100, 100, 100, 40, 100],
        )

    def test_config_prespecifies_four_second_epochs_and_sixty_seconds(self):
        config = load_duration_config("duration_qc_analysis/config.json")
        self.assertEqual(config["preprocessing_epoch_duration_seconds"], 4.0)
        self.assertEqual(config["minimum_accepted_duration_seconds"], 60.0)

    def test_full_cohort_excludes_only_short_recordings(self):
        qualified, audit, pairs = select_duration_cohort(
            self.modeling, self.qc, minimum_seconds=60.0
        )
        self.assertEqual(
            set(qualified["subject_id"]),
            {"c0", "c2", "c3", "p0", "p1", "p3"},
        )
        self.assertEqual(
            set(audit.loc[~audit["analysis_included"], "subject_id"]),
            {"c1", "p2"},
        )
        self.assertTrue(pairs.empty)

    def test_matched_cohort_removes_both_members_of_failed_pairs(self):
        pair_table = pd.DataFrame(
            {
                "match_pair_id": [f"pair-{index}" for index in range(4)],
                "control_subject_id": [f"c{index}" for index in range(4)],
                "pd_subject_id": [f"p{index}" for index in range(4)],
            }
        )
        qualified, audit, pairs = select_duration_cohort(
            self.modeling,
            self.qc,
            minimum_seconds=60.0,
            pair_table=pair_table,
        )
        self.assertEqual(set(pairs["match_pair_id"]), {"pair-0", "pair-3"})
        self.assertEqual(set(qualified["subject_id"]), {"c0", "p0", "c3", "p3"})
        self.assertTrue(
            qualified.groupby("match_pair_id")["target_pd"].nunique().eq(2).all()
        )
        self.assertEqual(
            set(
                audit.loc[
                    audit["exclusion_reason"].eq(
                        "matched_partner_below_60_seconds"
                    ),
                    "subject_id",
                ]
            ),
            {"c2", "p1"},
        )

    def test_group_comparison_uses_age_sex_adjustment_and_family_fdr(self):
        comparison = compare_group_features(
            self.modeling,
            ["ordinal_global_entropy"],
            matched=False,
            confidence_level=0.95,
            fdr_alpha=0.05,
        )
        self.assertEqual(len(comparison), 1)
        self.assertEqual(comparison.loc[0, "family"], "ordinal")
        self.assertIn("age + sex", comparison.loc[0, "primary_model"])
        self.assertTrue(np.isfinite(comparison.loc[0, "primary_effect"]))


if __name__ == "__main__":
    unittest.main()
