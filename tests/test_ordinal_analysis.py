import json
import unittest

import numpy as np
import ordpy
import pandas as pd

from ordinal_analysis.metrics import (
    analyze_epoch_data,
    ordinal_probabilities,
    subject_electrode_means,
)
from ordinal_analysis.pipeline import load_analysis_config


class OrdinalMetricTests(unittest.TestCase):
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

    def test_config_preserves_all_signal_decimals(self):
        config = load_analysis_config("ordinal_analysis/config.json")
        self.assertIsNone(config["ordinal"]["tie_precision"])
        with open("ordinal_analysis/config.json", encoding="utf-8") as stream:
            self.assertIsNone(json.load(stream)["ordinal"]["tie_precision"])


if __name__ == "__main__":
    unittest.main()
