import copy
from pathlib import Path
import json
import tempfile
import unittest

from core.config import (
    is_ica_review_confirmed,
    load_config,
    subject_manual_ica,
    write_ica_review_proposal,
)


class ConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config(Path("config/preprocessing.yaml"))

    def test_final_band_is_fixed(self):
        self.assertEqual(self.config["filter"]["l_freq"], 1.0)
        self.assertEqual(self.config["filter"]["h_freq"], 100.0)

    def test_60_hz_notch_is_enabled_inside_retained_band(self):
        self.assertTrue(self.config["filter"]["notch_enabled"])
        self.assertEqual(self.config["filter"]["notch_freq_hz"], 60.0)
        self.assertLess(
            self.config["filter"]["notch_freq_hz"],
            self.config["filter"]["h_freq"],
        )

    def test_final_sampling_rate_has_nyquist_guard_band(self):
        target = self.config["resampling"]["target_sfreq"]
        high = self.config["filter"]["h_freq"]
        self.assertEqual(target, 250.0)
        self.assertGreater(target / 2.0, high)

    def test_ica_and_iclabel_use_full_car_compatible_band(self):
        self.assertEqual(self.config["ica"]["fit_l_freq"], 1.0)
        self.assertEqual(self.config["ica"]["fit_h_freq"], 100.0)
        self.assertEqual(self.config["ica"]["temporary_resample_sfreq"], 250.0)

    def test_manual_ica_review_state_starts_empty(self):
        components, reasons = subject_manual_ica(self.config, "sub-001")
        self.assertEqual(components, [])
        self.assertEqual(reasons, {})
        self.assertFalse(is_ica_review_confirmed(self.config, "sub-001"))

    def test_review_proposal_prefills_but_remains_unconfirmed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "preprocessing.yaml"
            path.write_text(json.dumps(self.config), encoding="utf-8")
            written = write_ica_review_proposal(
                path,
                "sub-002",
                [3],
                {3: "ICLabel proposal requiring visual confirmation"},
            )
            updated = load_config(path)
        self.assertTrue(written)
        self.assertEqual(updated["ica"]["manual_exclude_components"]["sub-002"], [3])
        self.assertFalse(is_ica_review_confirmed(updated, "sub-002"))

    def test_review_proposal_never_overwrites_confirmed_decision(self):
        config = copy.deepcopy(self.config)
        config["ica"]["manual_exclude_components"]["sub-001"] = [0]
        config["ica"]["manual_exclude_reasons"]["sub-001"] = {
            "0": "reviewed ocular artifact"
        }
        config["ica"]["manual_review_confirmed"]["sub-001"] = True
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "preprocessing.yaml"
            path.write_text(json.dumps(config), encoding="utf-8")
            written = write_ica_review_proposal(
                path,
                "sub-001",
                [7],
                {7: "machine proposal"},
            )
            updated = load_config(path)
        self.assertFalse(written)
        self.assertEqual(updated["ica"]["manual_exclude_components"]["sub-001"], [0])

    def test_automatic_selection_is_recorded_separately_from_manual_review(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "preprocessing.yaml"
            path.write_text(json.dumps(self.config), encoding="utf-8")
            written = write_ica_review_proposal(
                path,
                "sub-001",
                [4, 5],
                {4: "automatic eye proposal", 5: "automatic muscle proposal"},
                automatic=True,
            )
            updated = load_config(path)
        self.assertTrue(written)
        self.assertNotIn("sub-001", updated["ica"]["manual_exclude_components"])
        self.assertEqual(updated["ica"]["automatic_exclude_components"]["sub-001"], [4, 5])


if __name__ == "__main__":
    unittest.main()
