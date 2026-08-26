"""Tests for the independent cycle-consistency burst detector."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from bycycle_burst_analysis.detector import detect_epoch_bursts, summarize_detection
from bycycle_burst_analysis.plots import plot_subject_average_violins


class BycycleBurstAnalysisTests(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = {
            "center_extrema": "peak",
            "amplitude_fraction_threshold": 0.3,
            "amplitude_consistency_threshold": 0.5,
            "period_consistency_threshold": 0.5,
            "monotonicity_threshold": 0.8,
            "minimum_consecutive_cycles": 3,
            "edge_padding_seconds": 0.5,
        }

    def test_independent_detector_enforces_edges_and_minimum_cycles(self) -> None:
        sfreq = 250.0
        time = np.arange(0.0, 4.0, 1.0 / sfreq)
        signal = 20.0 * np.sin(2.0 * np.pi * 10.0 * time)
        signal += np.random.default_rng(4).normal(scale=0.4, size=len(time))
        cycles, events, mask = detect_epoch_bursts(
            signal,
            sfreq=sfreq,
            band_limits=(8.0, 13.0),
            settings=self.settings,
        )
        edge = int(0.5 * sfreq)
        self.assertFalse(mask[:edge].any())
        self.assertFalse(mask[-edge:].any())
        self.assertTrue((events["n_cycles"] >= 3).all())
        self.assertEqual(int(cycles["is_burst"].sum()), int(events["n_cycles"].sum()))

    def test_empty_detection_has_zero_rate_and_undefined_duration(self) -> None:
        summary = summarize_detection(
            pd.DataFrame(), pd.DataFrame(), analyzed_duration_s=60.0
        )
        self.assertEqual(summary["n_bouts"], 0)
        self.assertEqual(summary["bouts_per_minute"], 0.0)
        self.assertEqual(summary["oscillatory_occupancy"], 0.0)
        self.assertTrue(np.isnan(summary["bout_duration_mean_s"]))

    def test_config_uses_only_canonical_non_overlapping_bands(self) -> None:
        config = json.loads(
            Path("bycycle_burst_analysis/config.json").read_text(encoding="utf-8")
        )
        self.assertEqual(config["detector"]["method"], "bycycle_cycle_consistency")
        self.assertEqual(
            list(config["bands"]),
            ["theta", "alpha", "low_beta", "high_beta"],
        )
        self.assertEqual(config["statistics"]["exclude_bands"], [])
        self.assertEqual(config["detector"]["minimum_consecutive_cycles"], 3)

    def test_subject_average_violins_create_one_file_per_metric(self) -> None:
        rows = []
        for group, offset in (("PD", 0.2), ("Control", 0.0)):
            for subject_index in range(4):
                for band_index, band in enumerate(("theta", "alpha")):
                    rows.append(
                        {
                            "subject_id": f"{group}-{subject_index}",
                            "group": group,
                            "band": band,
                            "n_electrodes": 60,
                            "bouts_per_minute": offset + band_index + subject_index / 10,
                            "bout_duration_mean_s": 0.3 + offset + subject_index / 100,
                        }
                    )
        with tempfile.TemporaryDirectory() as directory:
            outputs = plot_subject_average_violins(
                pd.DataFrame.from_records(rows),
                metrics=["bouts_per_minute", "bout_duration_mean_s"],
                bands=["theta", "alpha"],
                group_order=["PD", "Control"],
                colors={"PD": "#D55E00", "Control": "#0072B2"},
                band_labels={"theta": "Theta", "alpha": "Alpha"},
                output_dir=Path(directory),
                dpi=50,
            )
            self.assertEqual(len(outputs), 2)
            self.assertTrue(all(path.exists() for path in outputs))


if __name__ == "__main__":
    unittest.main()
