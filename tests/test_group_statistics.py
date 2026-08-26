"""Unit tests for consistent full-cohort and matched group inference."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.group_statistics import compute_group_statistics, fdr_bh


class GroupStatisticsTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(17)
        participant_rows = []
        metric_rows = []
        for group, offset in (("Control", 0.0), ("PD", 0.4)):
            for index in range(8):
                subject_id = f"{group[0]}{index:02d}"
                participant_rows.append(
                    {
                        "participant_id": subject_id,
                        "GROUP": group,
                        "AGE": 60 + index,
                        "GENDER": "M" if index % 2 else "F",
                    }
                )
                for band in ("alpha", "beta"):
                    for electrode in ("Fz", "Cz", "Pz", "Oz"):
                        metric_rows.append(
                            {
                                "subject_id": subject_id,
                                "group": group,
                                "band": band,
                                "electrode": electrode,
                                "value": offset + rng.normal(),
                            }
                        )
        self.participants = pd.DataFrame.from_records(participant_rows)
        self.metrics = pd.DataFrame.from_records(metric_rows)

    def test_bh_preserves_missing_values_and_order(self) -> None:
        adjusted, rejected = fdr_bh(np.array([0.04, np.nan, 0.001, 0.2]))
        self.assertTrue(np.isnan(adjusted[1]))
        self.assertAlmostEqual(adjusted[0], 0.06)
        self.assertAlmostEqual(adjusted[2], 0.003)
        self.assertEqual(rejected.tolist(), [False, False, True, False])

    def test_full_cohort_uses_adjusted_subject_models(self) -> None:
        subject, electrode = compute_group_statistics(
            self.metrics,
            self.participants,
            metrics=("value",),
            strata=("band",),
            domain="synthetic",
        )
        self.assertEqual(len(subject), 2)
        self.assertEqual(len(electrode), 8)
        self.assertTrue(
            subject["inference_design"].eq("full_cohort_age_sex_adjusted").all()
        )
        self.assertIn("primary_p_fdr_bh_domain", subject)
        self.assertIn("primary_p_fdr_bh_within_feature", electrode)
        self.assertIn("primary_p_fdr_bh_domain", electrode)

    def test_matched_cohort_preserves_pairing(self) -> None:
        participants = self.participants.copy()
        participants["match_pair_id"] = [f"pair-{index}" for index in range(8)] * 2
        subject, electrode = compute_group_statistics(
            self.metrics,
            participants,
            metrics=("value",),
            strata=("band",),
            domain="synthetic",
        )
        self.assertTrue(
            subject["inference_design"].eq("demographic_matched_pairs").all()
        )
        self.assertTrue(subject["n_pairs"].eq(8).all())
        self.assertTrue(electrode["n_pairs"].eq(8).all())
        self.assertTrue(subject["paired_wilcoxon_p_value"].notna().all())

    def test_duplicate_subject_electrode_rows_are_rejected(self) -> None:
        duplicated = pd.concat([self.metrics, self.metrics.iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "duplicate subject/electrode"):
            compute_group_statistics(
                duplicated,
                self.participants,
                metrics=("value",),
                strata=("band",),
                domain="synthetic",
            )

    def test_all_group_domains_exclude_overlapping_display_band(self) -> None:
        for relative_path in (
            "psd_analysis/config.json",
            "ordinal_analysis/config.json",
            "scale_free_analysis/config.json",
            "bout_analyses/config.json",
        ):
            config = json.loads(Path(relative_path).read_text(encoding="utf-8"))
            with self.subTest(config=relative_path):
                self.assertEqual(config["statistics"]["fdr_alpha"], 0.05)
                self.assertEqual(
                    config["statistics"]["exclude_bands"], ["broad_5_15"]
                )


if __name__ == "__main__":
    unittest.main()
