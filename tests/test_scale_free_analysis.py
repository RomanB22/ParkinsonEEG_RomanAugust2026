import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from ebosc.BOSC import BOSC_tf
from specparam.sim import sim_power_spectrum

from scale_free_analysis.aperiodic_diagnostics import assess_specparam_fit
from scale_free_analysis.fit_qc_sensitivity import _fit_coverage
from scale_free_analysis.metrics import (
    cycles_within_bouts,
    detect_frequency_episodes,
    ebosc_wavelet_power,
    extract_band_bouts,
    fit_specparam_spectrum,
    summarize_bouts,
    summarize_cycles,
)
from scale_free_analysis.pipeline import (
    _load_reusable_subject_features,
    load_analysis_config,
)
from scale_free_analysis.specparam_gallery import generate_specparam_gallery
from scale_free_analysis.typical_bouts import (
    _representation_figure,
    mean_centered_analytic,
    mean_centered_envelope,
)


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

    def test_typical_bout_gallery_allocates_one_row_per_band(self):
        figure, axes = _representation_figure(len(self.config["bands"]))
        try:
            self.assertEqual(axes.shape, (4, 3))
        finally:
            import matplotlib.pyplot as plt

            plt.close(figure)

    def test_config_keeps_full_psd_and_uses_reliable_aperiodic_range(self):
        config = load_analysis_config("scale_free_analysis/config.json")
        self.assertEqual(config["psd"]["fmin_hz"], 1.0)
        self.assertEqual(config["psd"]["fmax_hz"], 50.0)
        self.assertEqual(config["specparam"]["frequency_range_hz"], [4.0, 35.0])
        self.assertEqual(config["aperiodic_fit_qc"]["minimum_r_squared"], 0.9)
        self.assertEqual(
            config["aperiodic_sensitivity"]["frequency_ranges_hz"],
            [[4.0, 35.0], [3.0, 35.0], [4.0, 40.0], [3.0, 40.0]],
        )
        self.assertFalse(config["cache"]["save_raw_cycle_tables"])

    def test_matched_cache_filters_complete_subject_features_and_links_inputs(self):
        config = load_analysis_config("scale_free_analysis/config.json")
        subjects = ["sub-001", "sub-002"]
        electrodes = ["Fz", "Cz"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            output = root / "matched"
            metrics = source / "metrics"
            metrics.mkdir(parents=True)
            (source / "manifest.json").write_text(
                json.dumps({"analysis_config": config}), encoding="utf-8"
            )
            (metrics / "electrode_sets.json").write_text(
                json.dumps({"common_electrodes": electrodes}), encoding="utf-8"
            )
            pd.DataFrame(
                [
                    {
                        "subject_id": subject,
                        "group": "old",
                        "electrode": electrode,
                        "aperiodic_exponent": 1.0,
                    }
                    for subject in subjects + ["sub-extra"]
                    for electrode in electrodes
                ]
            ).to_csv(metrics / "electrode_aperiodic_metrics.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "subject_id": subject,
                        "group": "old",
                        "electrode": electrode,
                        "band": band,
                        "n_bouts": 1,
                    }
                    for subject in subjects + ["sub-extra"]
                    for electrode in electrodes
                    for band in config["bands"]
                ]
            ).to_csv(metrics / "electrode_band_metrics.csv", index=False)
            pd.DataFrame(
                {
                    "subject_id": subjects + ["sub-extra"],
                    "group": ["old"] * 3,
                }
            ).to_csv(metrics / "analyzed_inputs.csv", index=False)
            for subject in subjects:
                for subdirectory, suffix in (
                    ("spectra", "specparam_spectra.npz"),
                    ("episodes", "bout_episodes.csv.gz"),
                    ("thresholds", "ebosc_thresholds.csv.gz"),
                ):
                    path = source / "intermediate" / subdirectory / f"{subject}_{suffix}"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.touch()

            aperiodic, bands, inputs, provenance = _load_reusable_subject_features(
                source,
                output,
                config=config,
                expected_subjects=subjects,
                common_channels=electrodes,
                groups={"sub-001": "PD", "sub-002": "Control"},
            )
            self.assertEqual(len(aperiodic), 4)
            self.assertEqual(len(bands), 4 * len(config["bands"]))
            self.assertEqual(len(inputs), 2)
            self.assertEqual(provenance["mode"], "filtered_subject_level_reuse")
            linked = output / "intermediate" / "episodes" / "sub-001_bout_episodes.csv.gz"
            self.assertTrue(linked.is_symlink())
            self.assertTrue(linked.exists())

    def test_specparam_recovers_aperiodic_and_alpha_peak(self):
        np.random.seed(0)
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

    def test_fit_qc_uses_signed_residuals_and_flags_low_r_squared(self):
        observed = np.asarray([10.0, 5.0, 2.0, 1.0])
        modeled = np.asarray([8.0, 5.0, 2.5, 1.0])
        metrics = {
            "aperiodic_exponent": 1.2,
            "specparam_r_squared": 0.85,
            "specparam_error_mae": 0.05,
        }
        result = assess_specparam_fit(
            metrics,
            observed,
            modeled,
            self.config["aperiodic_fit_qc"],
        )
        self.assertFalse(result["specparam_fit_qc_pass"])
        self.assertIn("r_squared_below_minimum", result["specparam_fit_qc_reasons"])
        self.assertGreater(result["specparam_residual_max_abs_log10"], 0.0)

    def test_fit_qc_coverage_requires_configured_electrode_fraction(self):
        fits = pd.DataFrame(
            {
                "subject_id": ["sub-001"] * 4 + ["sub-002"] * 4,
                "group": ["PD"] * 4 + ["Control"] * 4,
                "electrode": ["Fz", "Cz", "Pz", "Oz"] * 2,
                "specparam_fit_qc_pass": [True, True, True, False]
                + [True, True, False, False],
            }
        )
        coverage = _fit_coverage(fits, 0.75).set_index("subject_id")
        self.assertTrue(coverage.loc["sub-001", "subject_fit_qc_pass"])
        self.assertFalse(coverage.loc["sub-002", "subject_fit_qc_pass"])
        self.assertEqual(coverage.loc["sub-001", "n_fit_qc_electrodes"], 3)
        self.assertAlmostEqual(
            coverage.loc["sub-002", "fit_failure_fraction"], 0.5
        )

    def test_typical_bout_envelope_is_center_aligned_and_bout_balanced(self):
        envelope = np.ones((2, 21), dtype=float)
        envelope[0, 8:13] = np.asarray([2.0, 3.0, 5.0, 3.0, 2.0])
        envelope[1, 3:8] = np.asarray([4.0, 5.0, 7.0, 5.0, 4.0])
        episodes = pd.DataFrame(
            {
                "epoch_index": [0, 1],
                "start_sample": [8, 3],
                "stop_sample_exclusive": [13, 8],
            }
        )
        mean, count = mean_centered_envelope(
            envelope, episodes, half_window_samples=2
        )
        np.testing.assert_allclose(mean, [3.0, 4.0, 6.0, 4.0, 3.0])
        self.assertEqual(count, 2)

    def test_typical_bout_phase_alignment_prevents_waveform_cancellation(self):
        analytic = np.ones((2, 21), dtype=np.complex128)
        amplitude = np.asarray([1.0, 2.0, 3.0, 2.0, 1.0])
        phase = np.asarray([-np.pi, -np.pi / 2.0, 0.0, np.pi / 2.0, np.pi])
        analytic[0, 8:13] = amplitude * np.exp(1j * phase)
        analytic[1, 3:8] = amplitude * np.exp(1j * (phase + 0.7))
        episodes = pd.DataFrame(
            {
                "epoch_index": [0, 1],
                "start_sample": [8, 3],
                "stop_sample_exclusive": [13, 8],
            }
        )
        envelope, phasor, shape, count = mean_centered_analytic(
            analytic, episodes, half_window_samples=2
        )
        np.testing.assert_allclose(envelope, amplitude)
        np.testing.assert_allclose(phasor, np.exp(1j * phase), atol=1e-12)
        np.testing.assert_allclose(shape, [-1.0, 0.0, 3.0, 0.0, -1.0], atol=1e-12)
        self.assertEqual(count, 2)

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

    def test_specparam_gallery_renders_one_flat_all_electrode_figure_per_subject(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            spectra_dir = root / "spectra"
            spectra_dir.mkdir()
            frequencies = np.arange(1.0, 40.25, 0.25)
            aperiodic = np.stack([10.0 / frequencies, 8.0 / frequencies])
            modeled = aperiodic.copy()
            observed = modeled * (1.0 + 0.2 * np.exp(-((frequencies - 10.0) ** 2)))
            np.savez_compressed(
                spectra_dir / "sub-001_specparam_spectra.npz",
                electrodes=np.asarray(["Fz", "Cz"]),
                frequencies_hz=frequencies,
                observed_psd_uv2_hz=observed,
                modeled_psd_uv2_hz=modeled,
                aperiodic_psd_uv2_hz=aperiodic,
                periodic_psd_uv2_hz=observed - modeled,
            )
            metrics = pd.DataFrame(
                {
                    "subject_id": ["sub-001", "sub-001"],
                    "group": ["PD", "PD"],
                    "electrode": ["Fz", "Cz"],
                    "aperiodic_offset": [1.0, 0.9],
                    "aperiodic_exponent": [1.0, 1.0],
                    "specparam_r_squared": [0.98, 0.97],
                    "specparam_error_mae": [0.01, 0.02],
                }
            )
            gallery_root = root / "gallery"
            legacy = gallery_root / "PD" / "sub-001"
            legacy.mkdir(parents=True)
            (legacy / "Fz.png").touch()
            (gallery_root / "sub-999_PD_all_electrodes.png").touch()
            index = generate_specparam_gallery(
                spectra_dir,
                metrics,
                gallery_root,
                dpi=60,
                workers=1,
            )
            self.assertEqual(len(index), 1)
            self.assertTrue(
                (gallery_root / "sub-001_PD_all_electrodes.png").exists()
            )
            self.assertEqual(index["subject_figure_path"].nunique(), 1)
            self.assertEqual(index.loc[0, "n_electrodes"], 2)
            self.assertFalse((gallery_root / "PD").exists())
            self.assertEqual(
                sorted(path.name for path in gallery_root.glob("*.png")),
                ["sub-001_PD_all_electrodes.png"],
            )
            self.assertTrue((gallery_root / "index.html").exists())


if __name__ == "__main__":
    unittest.main()
