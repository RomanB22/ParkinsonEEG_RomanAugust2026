"""Unit tests for ABBA-band full-signal and concatenated-bout H/C/F."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from analyses.rhythmicity.abba_ordinal_analysis import (
    CONCATENATION_POLICY,
    _electrode_metrics,
    compute_correlations,
    concatenate_bouts,
)


class AbbaOrdinalAnalysisTests(unittest.TestCase):
    def test_concatenate_bouts_uses_requested_order_and_boundaries(self) -> None:
        filtered = np.arange(2 * 2 * 8, dtype=float).reshape(2, 2, 8)
        bursts = [
            {"epoch_index": 0, "start_sample": 2, "stop_sample_exclusive": 5},
            {"epoch_index": 1, "start_sample": 4, "stop_sample_exclusive": 7},
        ]
        result = concatenate_bouts(filtered, bursts)
        expected = np.concatenate([filtered[0, :, 2:5], filtered[1, :, 4:7]], axis=1)
        np.testing.assert_array_equal(result, expected)
        self.assertEqual(result.shape, (2, 6))

    def test_empty_bouts_preserve_electrode_axis(self) -> None:
        result = concatenate_bouts(np.zeros((3, 4, 20)), [])
        self.assertEqual(result.shape, (4, 0))

    def test_electrode_metrics_return_finite_h_c_f(self) -> None:
        rng = np.random.default_rng(14)
        rows = _electrode_metrics(
            rng.normal(size=(2, 500)),
            ["C3", "C4"],
            dx=3,
            tau=1,
            minimum_samples=20,
        )
        self.assertEqual([row["electrode"] for row in rows], ["C3", "C4"])
        for row in rows:
            self.assertTrue(np.isfinite(row["entropy"]))
            self.assertTrue(np.isfinite(row["complexity"]))
            self.assertTrue(np.isfinite(row["fisher_information"]))
            self.assertEqual(row["n_ordinal_patterns"], 498)

    def test_policy_explicitly_allows_cross_join_patterns(self) -> None:
        self.assertIn("may_cross_joins", CONCATENATION_POLICY)

    def test_paired_change_tolerates_one_state_with_all_missing_metrics(self) -> None:
        rows = []
        for index in range(6):
            for group in ("PD_OFF", "PD_ON"):
                rows.append({
                    "dataset": "medication_state",
                    "participant_id": f"sub-{index}",
                    "group": group,
                    "band_name": "theta_1",
                    "scope": "within_bout",
                    "moca": np.nan,
                    "mmse": 25 + index,
                    "updrs": 40 + index + (group == "PD_ON"),
                    "entropy": np.nan if group == "PD_ON" else index / 10,
                    "complexity": index / 20,
                    "fisher_information": index / 30,
                })
        result = compute_correlations(pd.DataFrame(rows), minimum_n=5)
        selected = result.loc[
            result["group_model"].eq("PD_ON_minus_PD_OFF")
            & result["metric"].eq("entropy")
        ].iloc[0]
        self.assertEqual(selected["n"], 0)
        self.assertTrue(np.isnan(selected["spearman_rho"]))


if __name__ == "__main__":
    unittest.main()
