import json
import unittest

import numpy as np
from ebosc.BOSC import BOSC_tf
from specparam.sim import sim_power_spectrum

from scale_free_analysis.metrics import (
    cycles_within_bouts,
    detect_frequency_episodes,
    ebosc_wavelet_power,
    extract_band_bouts,
    fit_specparam_spectrum,
    summarize_bouts,
    summarize_cycles,
)
from scale_free_analysis.pipeline import load_analysis_config


class ScaleFreeAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open("scale_free_analysis/config.json", encoding="utf-8") as stream:
            cls.config = json.load(stream)

    def test_config_has_requested_frequency_bands(self):
        config = load_analysis_config("scale_free_analysis/config.json")
        self.assertEqual(
            config["bands"],
            {
                "theta": [4.0, 7.0],
                "alpha": [8.0, 13.0],
                "low_beta": [13.0, 20.0],
                "high_beta": [20.0, 30.0],
            },
        )

    def test_specparam_recovers_aperiodic_and_alpha_peak(self):
        frequencies, power = sim_power_spectrum(
            [1, 40],
            {"fixed": [1.0, 1.5]},
            {"gaussian": [10.0, 0.5, 2.0]},
            nlv=0.005,
            freq_res=0.25,
        )
        aperiodic, bands, curves = fit_specparam_spectrum(
            frequencies,
            power,
            self.config["bands"],
            self.config["specparam"],
        )
        self.assertAlmostEqual(aperiodic["aperiodic_exponent"], 1.5, delta=0.1)
        alpha = next(row for row in bands if row["band"] == "alpha")
        self.assertEqual(alpha["peak_present"], 1)
        self.assertAlmostEqual(alpha["peak_frequency_hz"], 10.0, delta=0.5)
        self.assertTrue(np.all(curves["aperiodic_psd_uv2_hz"] > 0.0))

    def test_vectorized_wavelets_match_ebosc(self):
        sfreq = 120.0
        times = np.arange(480) / sfreq
        signal = np.sin(2.0 * np.pi * 10.0 * times)
        frequencies = np.asarray([6.0, 10.0, 20.0])
        actual = ebosc_wavelet_power(
            signal[np.newaxis, :],
            sfreq=sfreq,
            frequencies=frequencies,
            wavenumber=6.0,
        )[0]
        expected, _, _ = BOSC_tf(signal, frequencies, sfreq, 6.0)
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)

    def test_detection_enforces_duration_and_epoch_edges(self):
        power = np.zeros((3, 1, 30), dtype=float)
        power[0, 0, 5:15] = 3.0
        power[1, 0, 5:8] = 3.0
        power[2, 0, :7] = 3.0
        detected = detect_frequency_episodes(
            power,
            sfreq=10.0,
            frequencies=np.asarray([5.0]),
            thresholds=np.asarray([2.0]),
            minimum_cycles=3.0,
            edge_padding_samples=2,
        )
        self.assertTrue(detected[0, 0, 5:15].all())
        self.assertFalse(detected[1].any())
        self.assertFalse(detected[2].any())
        self.assertFalse(detected[..., :2].any())

    def test_bout_extraction_and_summary(self):
        detected = np.zeros((1, 2, 20), dtype=bool)
        detected[0, 0, 2:6] = True
        detected[0, 1, 10:15] = True
        power = np.where(detected, 8.0, 1.0)
        episodes, mask = extract_band_bouts(
            detected,
            power,
            np.asarray([2.0, 2.0]),
            np.asarray([8.0, 10.0]),
            band="alpha",
            band_limits=(8.0, 13.0),
            sfreq=10.0,
        )
        summary = summarize_bouts(episodes, mask, sfreq=10.0)
        self.assertEqual(summary["n_bouts"], 2)
        self.assertAlmostEqual(summary["oscillatory_occupancy"], 9 / 20)
        self.assertAlmostEqual(episodes.loc[1, "inter_bout_interval_s"], 0.4)

    def test_bycycle_features_are_restricted_to_bout_mask(self):
        sfreq = 120.0
        times = np.arange(480) / sfreq
        signal = 20.0 * np.sin(2.0 * np.pi * 10.0 * times)
        full_mask = np.ones(480, dtype=bool)
        cycles = cycles_within_bouts(
            signal,
            full_mask,
            sfreq=sfreq,
            band_limits=(8.0, 13.0),
            minimum_overlap=0.5,
        )
        summary = summarize_cycles(cycles)
        self.assertGreater(summary["n_cycles"], 20)
        self.assertAlmostEqual(summary["cycle_frequency_mean_hz"], 10.0, delta=0.5)
        self.assertAlmostEqual(summary["rise_decay_symmetry_mean"], 0.5, delta=0.1)


if __name__ == "__main__":
    unittest.main()
