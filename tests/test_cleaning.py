import unittest
import logging

import mne
import numpy as np

from src.artifacts import annotate_large_artifacts, create_and_reject_epochs
from src.channels import detect_bad_channels
from src.config import load_config
from src.preprocessing import filter_eeg, rereference, resample_eeg


def synthetic_raw(data_uv, sfreq=100.0):
    names = ["Fp1", "Fz", "Cz", "O1"][: data_uv.shape[0]]
    info = mne.create_info(names, sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data_uv * 1e-6, info, verbose="ERROR")
    raw.set_montage(mne.channels.make_standard_montage("standard_1020"), verbose="ERROR")
    return raw


class CleaningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config("config/preprocessing.yaml")

    def test_flat_channel_is_bad_and_not_deleted(self):
        rng = np.random.default_rng(42)
        data = rng.normal(scale=10.0, size=(4, 2000))
        data[0] = 0.0
        raw = synthetic_raw(data)
        result = detect_bad_channels(raw, self.config["channels"])
        self.assertIn("Fp1", result.bad_channels)
        self.assertIn("flat_signal", result.reasons["Fp1"])
        self.assertEqual(len(raw.ch_names), 4)

    def test_complex_pipeline_resamples_to_final_rate(self):
        rng = np.random.default_rng(42)
        raw = synthetic_raw(rng.normal(scale=10.0, size=(4, 5000)), sfreq=500.0)
        result = resample_eeg(
            raw,
            self.config["resampling"],
            logging.getLogger("test.resampling"),
        )
        self.assertEqual(raw.info["sfreq"], 500.0)
        self.assertEqual(result.info["sfreq"], 250.0)
        self.assertEqual(result.n_times, 2500)

    def test_iclabel_input_contract_is_notched_bandlimited_and_car(self):
        rng = np.random.default_rng(42)
        raw = synthetic_raw(rng.normal(scale=10.0, size=(4, 5000)), sfreq=500.0)
        logger = logging.getLogger("test.iclabel_input")
        filtered, notch_applied, _ = filter_eeg(
            raw, self.config["filter"], logger
        )
        prepared = rereference(
            resample_eeg(filtered, self.config["resampling"], logger),
            logger,
            stage="test",
        )
        self.assertTrue(notch_applied)
        self.assertEqual(prepared.info["highpass"], 1.0)
        self.assertEqual(prepared.info["lowpass"], 100.0)
        self.assertEqual(prepared.info["sfreq"], 250.0)
        np.testing.assert_allclose(
            prepared.get_data(picks="eeg").mean(axis=0),
            0.0,
            atol=1e-15,
        )

    def test_large_transient_becomes_annotation(self):
        rng = np.random.default_rng(42)
        data = rng.normal(scale=10.0, size=(4, 2000))
        data[1, 500:600] += 1000.0
        raw = synthetic_raw(data)
        annotated, table = annotate_large_artifacts(raw, self.config["artifacts"])
        self.assertFalse(table.empty)
        self.assertTrue(any("BAD_amplitude" in description for description in annotated.annotations.description))
        self.assertEqual(annotated.n_times, raw.n_times)

    def test_epoch_rejection_preserves_clean_epochs(self):
        rng = np.random.default_rng(42)
        data = rng.normal(scale=10.0, size=(4, 1200))
        data[2, 400:800] += np.sin(np.linspace(0, 20 * np.pi, 400)) * 600.0
        raw = synthetic_raw(data)
        result = create_and_reject_epochs(raw, self.config["epochs"])
        self.assertEqual(result.n_initial, 3)
        self.assertGreaterEqual(result.n_rejected, 1)
        self.assertGreaterEqual(result.n_retained, 1)

    def test_residual_blink_above_200_uv_is_rejected(self):
        rng = np.random.default_rng(42)
        data = rng.normal(scale=5.0, size=(4, 1200))
        # A frontal transient confined to the first 4-second epoch. Its
        # peak-to-peak amplitude is safely above the configured 200 µV guard.
        data[0, 120:180] += np.hanning(60) * 300.0
        raw = synthetic_raw(data)
        result = create_and_reject_epochs(raw, self.config["epochs"])
        first = result.rejection_table.loc[0]
        self.assertFalse(first["accepted"])
        self.assertIn("BAD_peak_to_peak", first["reasons"])
        self.assertIn("absolute_peak_to_peak", first["amplitude_reason"])
        self.assertGreaterEqual(first["max_peak_to_peak_uv"], 200.0)


if __name__ == "__main__":
    unittest.main()
