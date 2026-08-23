import unittest

import numpy as np
from scipy.signal import welch

from psd_analysis.metrics import (
    bootstrap_median_ci,
    compute_subject_electrode_psd,
    integrate_bands,
)
from psd_analysis.pipeline import load_psd_config


class PsdAnalysisTests(unittest.TestCase):
    def test_welch_psd_finds_ten_hz_signal(self):
        sfreq = 120.0
        times = np.arange(480) / sfreq
        signal = np.sin(2 * np.pi * 10.0 * times) * 10e-6
        data = np.tile(signal, (4, 2, 1))
        frequencies, psd = compute_subject_electrode_psd(data, sfreq)
        peak = frequencies[np.argmax(psd[0])]
        self.assertAlmostEqual(peak, 10.0)
        self.assertEqual(psd.shape, (2, 197))

    def test_subject_psd_uses_concatenated_epochs(self):
        sfreq = 120.0
        times = np.arange(480) / sfreq
        amplitudes = np.asarray([1.0, 2.0, 100.0]) * 1e-6
        data = np.asarray(
            [[amplitude * np.sin(2 * np.pi * 10.0 * times)] for amplitude in amplitudes]
        )
        frequencies, psd = compute_subject_electrode_psd(data, sfreq)
        concatenated = data.transpose(1, 0, 2).reshape(1, -1)
        expected_frequencies, expected = welch(
            concatenated,
            fs=sfreq,
            window="hann",
            nperseg=480,
            noverlap=0,
            nfft=480,
            detrend="constant",
            scaling="density",
            axis=-1,
            average="mean",
        )
        mask = (expected_frequencies >= 1.0) & (expected_frequencies <= 50.0)
        np.testing.assert_array_equal(frequencies, expected_frequencies[mask])
        np.testing.assert_allclose(psd, expected[:, mask] * 1e12)

        # A median of the three separate epoch PSDs would equal the middle
        # amplitude's PSD and must not be mistaken for concatenated Welch.
        _, middle_psd = compute_subject_electrode_psd(data[1:2], sfreq)
        ten_hz = np.flatnonzero(frequencies == 10.0).item()
        self.assertGreater(psd[0, ten_hz], middle_psd[0, ten_hz])

    def test_band_power_integrates_linear_density(self):
        frequencies = np.arange(1.0, 10.25, 0.25)
        psd = np.full((2, len(frequencies)), 2.0)
        powers = integrate_bands(frequencies, psd, {"test": (2.0, 6.0)})
        np.testing.assert_allclose(powers["test"], [8.0, 8.0])

    def test_bootstrap_median_is_reproducible(self):
        values = np.arange(30.0).reshape(10, 3)
        first = bootstrap_median_ci(values, n_resamples=200, seed=7)
        second = bootstrap_median_ci(values, n_resamples=200, seed=7)
        for left, right in zip(first, second):
            np.testing.assert_array_equal(left, right)
        np.testing.assert_array_equal(first[0], np.median(values, axis=0))
        self.assertTrue(np.all(first[1] <= first[0]))
        self.assertTrue(np.all(first[0] <= first[2]))

    def test_default_config_has_expected_bands_and_bootstrap(self):
        config = load_psd_config("psd_analysis/config.json")
        self.assertEqual(
            list(config["bands"]),
            ["delta", "theta", "alpha", "beta", "low_gamma", "broad_5_15"],
        )
        self.assertEqual(config["bands"]["broad_5_15"], [5.0, 15.0])
        self.assertEqual(config["bootstrap"]["confidence_level"], 0.95)
        self.assertEqual(config["bootstrap"]["n_resamples"], 2000)


if __name__ == "__main__":
    unittest.main()
