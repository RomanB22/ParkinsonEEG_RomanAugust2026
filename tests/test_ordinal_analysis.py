import json
import itertools
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import ordpy
import pandas as pd

from ordinal_analysis.metrics import (
    METRICS,
    RENYI_ALPHA_METRICS,
    analyze_epoch_data,
    band_subject_electrode_means,
    filter_epoch_data,
    metrics_from_probabilities,
    ordinal_probabilities,
    subject_electrode_means,
)
from ordinal_analysis.pipeline import (
    _load_reusable_subject_metrics,
    load_analysis_config,
)
from ordinal_analysis.plots import electrode_metric_zscores


class OrdinalMetricTests(unittest.TestCase):
    def test_renyi_alpha_grid_includes_requested_extremes(self):
        self.assertEqual(
            tuple(alpha for alpha, _, _ in RENYI_ALPHA_METRICS),
            (0.1, 0.5, 0.9, 1.1, 2.0, 5.0, 10.0),
        )

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

    def test_electrode_zscores_accept_one_renyi_alpha_pair(self):
        entropy_metric = "renyi_entropy_alpha_10"
        complexity_metric = "renyi_complexity_alpha_10"
        table = pd.DataFrame(
            {
                "electrode": ["Fz"] * 4,
                entropy_metric: [0.1, 0.2, 0.3, 0.4],
                complexity_metric: [0.4, 0.3, 0.2, 0.1],
            }
        )
        standardized = electrode_metric_zscores(
            table, metrics=(entropy_metric, complexity_metric)
        )
        self.assertAlmostEqual(standardized[entropy_metric].mean(), 0.0)
        self.assertAlmostEqual(standardized[entropy_metric].std(ddof=0), 1.0)
        self.assertAlmostEqual(standardized[complexity_metric].mean(), 0.0)
        self.assertAlmostEqual(standardized[complexity_metric].std(ddof=0), 1.0)

    def test_single_epoch_probabilities_match_ordpy(self):
        data = np.asarray([[4.0, 7.0, 9.0, 10.0, 6.0, 11.0, 3.0]])
        _, expected = ordpy.ordinal_distribution(
            data[0], dx=3, return_missing=True, ordered=True, tie_precision=None
        )
        actual, n_patterns, n_ties = ordinal_probabilities(data, dx=3, tau=1)
        np.testing.assert_allclose(actual, expected)
        self.assertEqual(n_patterns, 5)
        self.assertEqual(n_ties, 0)

    def test_vectorized_permutation_counts_match_reference_for_all_dimensions(self):
        rng = np.random.default_rng(20260824)
        for dimension in range(3, 8):
            data = rng.normal(size=(4, 80))
            actual, n_patterns, _ = ordinal_probabilities(
                data, dx=dimension, tau=1
            )
            windows = np.lib.stride_tricks.sliding_window_view(
                data, dimension, axis=1
            )
            symbols = np.argsort(windows, axis=-1).reshape(-1, dimension)
            lookup = {
                permutation: index
                for index, permutation in enumerate(
                    itertools.permutations(range(dimension))
                )
            }
            counts = np.zeros(math.factorial(dimension), dtype=np.int64)
            for symbol in symbols:
                counts[lookup[tuple(int(value) for value in symbol)]] += 1
            np.testing.assert_array_equal(actual, counts / counts.sum())
            self.assertEqual(n_patterns, int(counts.sum()))

    def test_renyi_complexity_entropy_matches_ordpy_for_all_alphas(self):
        probabilities = np.asarray([1 / 3, 1 / 15, 4 / 15, 2 / 15, 1 / 5, 0])
        actual = metrics_from_probabilities(probabilities, dx=3)
        for alpha, entropy_metric, complexity_metric in RENYI_ALPHA_METRICS:
            expected_entropy, expected_complexity = ordpy.renyi_complexity_entropy(
                probabilities,
                alpha=alpha,
                dx=3,
                probs=True,
            )
            self.assertAlmostEqual(actual[entropy_metric], expected_entropy)
            self.assertAlmostEqual(actual[complexity_metric], expected_complexity)

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
        for metric in METRICS:
            self.assertIn(metric, metrics)
            self.assertAlmostEqual(means.loc[0, metric], metrics[metric].mean())

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
        for metric in METRICS[3:]:
            table[metric] = [0.2, 0.4, 0.6, 0.8]
        means = band_subject_electrode_means(table)
        self.assertEqual(means["band"].tolist(), ["delta", "theta"])
        self.assertEqual(means["n_electrodes"].tolist(), [2, 2])
        self.assertAlmostEqual(means.loc[0, "entropy"], 0.2)
        self.assertAlmostEqual(means.loc[1, "entropy"], 0.7)

    def test_config_preserves_all_signal_decimals(self):
        config = load_analysis_config("ordinal_analysis/config.json")
        self.assertEqual(config["ordinal"]["delay_samples"], 1)
        self.assertIsNone(config["ordinal"]["tie_precision"])
        self.assertEqual(
            list(config["bands"]),
            ["delta", "theta", "alpha", "beta", "low_gamma", "broad_5_15"],
        )
        self.assertEqual(config["bands"]["broad_5_15"], [5.0, 15.0])
        self.assertEqual(config["band_filter"]["order"], 4)
        with open("ordinal_analysis/config.json", encoding="utf-8") as stream:
            self.assertIsNone(json.load(stream)["ordinal"]["tie_precision"])

    def test_config_rejects_nonunit_delay(self):
        with open("ordinal_analysis/config.json", encoding="utf-8") as stream:
            config = json.load(stream)
        config["ordinal"]["delay_samples"] = 5
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must be 1"):
                load_analysis_config(path)

    def test_reusable_metrics_are_filtered_and_require_complete_grids(self):
        config = load_analysis_config("ordinal_analysis/config.json")
        subjects = ["sub-001", "sub-002"]
        electrodes = ["Fz", "Cz"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metrics_dir = root / "metrics"
            metrics_dir.mkdir()
            (root / "manifest.json").write_text(
                json.dumps({"analysis_config": config}), encoding="utf-8"
            )
            (metrics_dir / "electrode_sets.json").write_text(
                json.dumps({"common_electrodes": electrodes}), encoding="utf-8"
            )
            broadband = pd.DataFrame(
                [
                    {
                        "subject_id": subject,
                        "group": "old",
                        "electrode": electrode,
                        "entropy": 0.5,
                    }
                    for subject in subjects + ["sub-extra"]
                    for electrode in electrodes
                ]
            )
            band = pd.DataFrame(
                [
                    {
                        "subject_id": subject,
                        "group": "old",
                        "band": band_name,
                        "electrode": electrode,
                        "entropy": 0.5,
                    }
                    for subject in subjects + ["sub-extra"]
                    for band_name in config["bands"]
                    for electrode in electrodes
                ]
            )
            inputs = pd.DataFrame(
                {
                    "subject_id": subjects + ["sub-extra"],
                    "group": ["old"] * 3,
                }
            )
            broadband.to_csv(metrics_dir / "electrode_metrics.csv", index=False)
            band.to_csv(metrics_dir / "band_electrode_metrics.csv", index=False)
            inputs.to_csv(metrics_dir / "analyzed_inputs.csv", index=False)

            reused, reused_band, reused_inputs, provenance = (
                _load_reusable_subject_metrics(
                    root,
                    config=config,
                    expected_subjects=subjects,
                    common_channels=electrodes,
                    groups={"sub-001": "PD", "sub-002": "Control"},
                )
            )
            self.assertEqual(len(reused), 4)
            self.assertEqual(len(reused_band), 4 * len(config["bands"]))
            self.assertEqual(len(reused_inputs), 2)
            self.assertEqual(set(reused["group"]), {"PD", "Control"})
            self.assertEqual(provenance["mode"], "filtered_subject_level_reuse")


if __name__ == "__main__":
    unittest.main()
