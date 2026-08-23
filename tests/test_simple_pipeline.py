import unittest

import mne
import numpy as np

from simpler.pipeline import (
    detrend_filter_resample,
    extract_scale_and_mask_windows,
    load_simple_config,
)


class SimplePipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_simple_config("simpler/config.json")

    def test_requested_band_and_sampling_rate(self):
        self.assertEqual(self.config["bandpass_low_hz"], 1.0)
        self.assertEqual(self.config["bandpass_high_hz"], 50.0)
        self.assertEqual(self.config["target_sfreq"], 120.0)
        self.assertGreater(self.config["target_sfreq"] / 2, self.config["bandpass_high_hz"])

    def test_filter_and_resample(self):
        sfreq = 500.0
        times = np.arange(int(10 * sfreq)) / sfreq
        signal_uv = 10 * np.sin(2 * np.pi * 10 * times) + 5 * np.sin(2 * np.pi * 60 * times)
        signal_uv += np.linspace(-20, 20, len(times))
        raw = mne.io.RawArray(
            np.vstack([signal_uv, signal_uv * 0.8]) * 1e-6,
            mne.create_info(["Fz", "Cz"], sfreq, "eeg"),
            verbose="ERROR",
        )
        processed = detrend_filter_resample(raw, self.config)
        self.assertEqual(processed.info["sfreq"], 120.0)
        self.assertEqual(processed.info["highpass"], 1.0)
        self.assertEqual(processed.info["lowpass"], 50.0)

    def test_rejected_window_is_masked_placeholder(self):
        sfreq = 120.0
        data_uv = np.zeros((2, int(8 * sfreq)))
        data_uv[0, int(5 * sfreq)] = 150.0
        raw = mne.io.RawArray(
            data_uv * 1e-6,
            mne.create_info(["Fz", "Cz"], sfreq, "eeg"),
            verbose="ERROR",
        )
        result = extract_scale_and_mask_windows(raw, self.config)
        np.testing.assert_array_equal(result.valid_mask, [True, False])
        self.assertTrue(np.all(result.scaled_windows[1] == 0.0))


if __name__ == "__main__":
    unittest.main()

