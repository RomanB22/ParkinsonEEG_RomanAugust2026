import json
import unittest

import numpy as np
import ordpy
import pandas as pd

from ordinal_analysis.metrics import (
    analyze_epoch_data,
    band_subject_electrode_means,
    filter_epoch_data,
    ordinal_probabilities,
    subject_electrode_means,
)
from ordinal_analysis.pipeline import load_analysis_config
from ordinal_analysis.plots import electrode_metric_zscores


class OrdinalMetricTests(unittest.TestCase):
    def test_electrode_zscores_pool_groups_within_band_and_electrode(self):
        table = pd.DataFrame(
            {
                "subject_id": ["s1", "s2", "s3", "s4"] * 2,
                "group": ["PD", "PD", "Control", "Control"] * 2,
                "band": ["delta"] * 4 + ["theta"] * 4,
                "electrode": ["Fz"] * 8,
                "entropy": [1.0, 2.0, 3.0, 4.0, 10.0, 12.0, 14.0, 16.0],
                "complexity": [4.0, 3.0, 2.0, 1.0, 16.0, 14.0, 12.0, 10.0],
                "fisher_information": [2.0] * 8,
            }
        )
        standardized = electrode_metric_zscores(table, strata=("band",))
        for _, selected in standardized.groupby(["band", "electrode"]):
            self.assertAlmostEqual(selected["entropy"].mean(), 0.0)
            self.assertAlmostEqual(selected["entropy"].std(ddof=0), 1.0)
            self.assertAlmostEqual(selected["complexity"].mean(), 0.0)
            self.assertAlmostEqual(selected["complexity"].std(ddof=0), 1.0)
            np.testing.assert_array_equal(selected["fisher_information"], 0.0)
        delta = standardized.loc[standardized["band"].eq("delta")]
        self.assertLess(delta.loc[delta["group"].eq("PD"), "entropy"].mean(), 0.0)
        self.assertGreater(
            delta.loc[delta["group"].eq("Control"), "entropy"].mean(), 0.0
        )

    def test_single_epoch_probabilities_match_ordpy(self):
        data = np.asarray([[4.0, 7.0, 9.0, 10.0, 6.0, 11.0, 3.0]])
        _, expected = ordpy.ordinal_distribution(
            data[0], dx=3, return_missing=True, ordered=True, tie_precision=None
        )
        actual, n_patterns, n_ties = ordinal_probabilities(data, dx=3, tau=1)
        np.testing.assert_allclose(actual, expected)
        self.assertEqual(n_patterns, 5)
        self.assertEqual(n_ties, 0)

    def test_patterns_never_cross_epoch_boundaries(self):
        data = np.asarray(
            [
                [0.0, 1.0, 2.0, 3.0],
                [3.0, 2.0, 1.0, 0.0],
            ]
        )
        probabilities, n_patterns, _ = ordinal_probabilities(data, dx=3, tau=1)
        self.assertEqual(n_patterns, 4)
        self.assertAlmostEqual(probabilities[0], 0.5)
        self.assertAlmostEqual(probabilities[-1], 0.5)
        self.assertEqual(np.count_nonzero(probabilities), 2)

    def test_full_precision_is_not_rounded(self):
        data = np.asarray([[1.001, 1.003, 1.002, 1.004, 1.000]])
        full, _, full_ties = ordinal_probabilities(data, dx=3, tie_precision=None)
        rounded, _, rounded_ties = ordinal_probabilities(data, dx=3, tie_precision=2)
        self.assertFalse(np.allclose(full, rounded))
        self.assertEqual(full_ties, 0)
        self.assertGreater(rounded_ties, 0)

    def test_subject_mean_averages_metrics_not_voltage(self):
        rng = np.random.default_rng(42)
        data = rng.normal(size=(3, 2, 40))
        metrics = analyze_epoch_data(
            data,
            ["Fz", "Cz"],
            subject_id="sub-test",
            group="PD",
            sfreq=120.0,
        )
        means = subject_electrode_means(metrics)
        self.assertEqual(means.loc[0, "n_electrodes"], 2)
        self.assertAlmostEqual(means.loc[0, "entropy"], metrics["entropy"].mean())
        self.assertAlmostEqual(means.loc[0, "complexity"], metrics["complexity"].mean())
        self.assertAlmostEqual(
            means.loc[0, "fisher_information"], metrics["fisher_information"].mean()
        )

    def test_band_filter_is_epoch_local_and_frequency_selective(self):
        sfreq = 120.0
        times = np.arange(480) / sfreq
        target = np.sin(2 * np.pi * 10.0 * times)
        outside = np.sin(2 * np.pi * 35.0 * times)
        data = np.stack([target + outside, target - outside])[:, None, :]
        filtered = filter_epoch_data(
            data, sfreq=sfreq, low_hz=8.0, high_hz=13.0, order=4
        )
        target_projection = np.abs(np.vdot(filtered[0, 0], target))
        outside_projection = np.abs(np.vdot(filtered[0, 0], outside))
        self.assertGreater(target_projection, 50.0 * outside_projection)

        changed_second_epoch = data.copy()
        changed_second_epoch[1] *= 1000.0
        changed = filter_epoch_data(
            changed_second_epoch, sfreq=sfreq, low_hz=8.0, high_hz=13.0, order=4
        )
        np.testing.assert_allclose(filtered[0], changed[0], rtol=0.0, atol=0.0)

    def test_band_subject_mean_preserves_band_identity(self):
        table = pd.DataFrame(
            {
                "subject_id": ["sub-001"] * 4,
                "group": ["PD"] * 4,
                "band": ["delta", "delta", "theta", "theta"],
                "band_low_hz": [1.0, 1.0, 4.0, 4.0],
                "band_high_hz": [4.0, 4.0, 8.0, 8.0],
                "electrode": ["Fz", "Cz", "Fz", "Cz"],
                "entropy": [0.1, 0.3, 0.6, 0.8],
                "complexity": [0.2, 0.4, 0.5, 0.7],
                "fisher_information": [0.3, 0.5, 0.4, 0.6],
            }
        )
        means = band_subject_electrode_means(table)
        self.assertEqual(means["band"].tolist(), ["delta", "theta"])
        self.assertEqual(means["n_electrodes"].tolist(), [2, 2])
        self.assertAlmostEqual(means.loc[0, "entropy"], 0.2)
        self.assertAlmostEqual(means.loc[1, "entropy"], 0.7)

    def test_config_preserves_all_signal_decimals(self):
        config = load_analysis_config("ordinal_analysis/config.json")
        self.assertIsNone(config["ordinal"]["tie_precision"])
        self.assertEqual(
            list(config["bands"]),
            ["delta", "theta", "alpha", "beta", "low_gamma", "broad_5_15"],
        )
        self.assertEqual(config["bands"]["broad_5_15"], [5.0, 15.0])
        self.assertEqual(config["band_filter"]["order"], 4)
        with open("ordinal_analysis/config.json", encoding="utf-8") as stream:
            self.assertIsNone(json.load(stream)["ordinal"]["tie_precision"])


if __name__ == "__main__":
    unittest.main()
