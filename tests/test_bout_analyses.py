import unittest
import warnings

import numpy as np
import ordpy
import pandas as pd

from analyses.bouts.metrics import (
    METRICS,
    analyze_bout_segments,
    ordinal_counts,
    shannon_metrics_from_counts,
)
from analyses.bouts.pipeline import load_analysis_config


class BoutAnalysesTests(unittest.TestCase):
    def test_config_uses_regular_metrics_and_selected_aperiodic_fit(self):
        config = load_analysis_config("config/analyses/bouts.json")
        self.assertEqual(config["psd"], {"fmin_hz": 1.0, "fmax_hz": 50.0})
        self.assertEqual(config["specparam"]["frequency_range_hz"], [4.0, 50.0])
        self.assertEqual(config["specparam"]["aperiodic_modes"], ["fixed", "knee"])
        self.assertEqual(config["specparam"]["model_selection_criterion"], "bic")
        self.assertEqual(config["ordinal"]["embedding_dimension"], 6)
        self.assertEqual(config["ordinal"]["delay_samples"], 1)
        self.assertEqual(METRICS, ("entropy", "complexity", "fisher_information"))
        self.assertEqual(
            list(config["bands"]),
            ["theta", "alpha", "low_beta", "high_beta"],
        )
        self.assertEqual(config["statistics"]["exclude_bands"], [])
        self.assertEqual(
            config["input"]["scale_free_output_dir"],
            "outputs/full/scale_free",
        )
        self.assertTrue(config["cache"]["reuse_scale_free_detection"])

    def test_counts_match_ordpy_for_one_bout(self):
        signal = np.asarray([3.0, 1.0, 4.0, 1.5, 5.0, 9.0, 2.0])
        counts, ties = ordinal_counts(signal, dx=3, tau=1)
        expected = ordpy.ordinal_distribution(
            signal, dx=3, taux=1, return_missing=True, ordered=True
        )[1]
        np.testing.assert_allclose(counts / counts.sum(), expected)
        self.assertEqual(int(counts.sum()), len(signal) - 2)
        self.assertEqual(ties, 0)

    def test_bout_pooling_never_crosses_boundaries(self):
        epochs = np.asarray([[1.0, 2.0, 3.0, 9.0, 8.0, 7.0]])
        episodes = pd.DataFrame(
            {
                "epoch_index": [0, 0],
                "start_sample": [0, 3],
                "stop_sample_exclusive": [3, 6],
            }
        )
        pooled, summary, bouts, _ = analyze_bout_segments(
            epochs, episodes, dx=3, tau=1
        )
        self.assertEqual(int(pooled.sum()), 2)
        self.assertEqual(summary["n_ordinal_patterns"], 2)
        self.assertEqual(summary["n_detected_bouts"], 2)
        self.assertEqual(summary["n_analyzable_ordinal_bouts"], 2)
        self.assertEqual(len(bouts), 2)
        concatenated_counts, _ = ordinal_counts(epochs[0], dx=3, tau=1)
        self.assertEqual(int(concatenated_counts.sum()), 4)

    def test_short_bouts_are_reported_not_joined(self):
        epochs = np.arange(20.0)[np.newaxis, :]
        episodes = pd.DataFrame(
            {
                "epoch_index": [0, 0],
                "start_sample": [0, 5],
                "stop_sample_exclusive": [5, 12],
            }
        )
        pooled, summary, bouts, _ = analyze_bout_segments(
            epochs, episodes, dx=6, tau=1
        )
        self.assertEqual(summary["n_short_bouts_excluded"], 1)
        self.assertEqual(summary["n_analyzable_ordinal_bouts"], 1)
        self.assertEqual(int(pooled.sum()), 2)
        self.assertEqual(bouts["analyzable_ordinal_bout"].tolist(), [0, 1])

    def test_regular_metrics_match_ordpy_without_renyi_outputs(self):
        counts = np.asarray([8, 3, 2, 4, 5, 7], dtype=np.int64)
        actual = shannon_metrics_from_counts(counts, dx=3)
        probabilities = counts / counts.sum()
        entropy, complexity = ordpy.complexity_entropy(probabilities, dx=3, probs=True)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Be mindful the correct calculation of Fisher information.*",
                category=UserWarning,
            )
            _, fisher = ordpy.fisher_shannon(probabilities, dx=3, probs=True)
        self.assertEqual(set(actual), set(METRICS))
        self.assertAlmostEqual(actual["entropy"], entropy)
        self.assertAlmostEqual(actual["complexity"], complexity)
        self.assertAlmostEqual(actual["fisher_information"], fisher)


if __name__ == "__main__":
    unittest.main()
