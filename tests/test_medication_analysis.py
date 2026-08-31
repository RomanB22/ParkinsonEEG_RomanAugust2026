import unittest
import warnings
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from analyses.medication.cohort import build_cohort
from analyses.medication.comparison_plots import (
    TOPOGRAPHIC_METRICS,
    plot_group_mean_topomaps,
    plot_group_psd_curves,
)
from analyses.medication.pipeline import load_analysis_config
from analyses.medication.plots import (
    FOCUSED_BOUT_MMSE_FEATURES,
    FOCUSED_DELTA_UPDRS_FEATURES,
    select_focused_bout_mmse_rows,
    select_focused_delta_updrs_rows,
)
from analyses.medication.statistical_plots import plot_all_feature_violins
from analyses.medication.statistics import (
    _paired_contrast,
    compute_condition_statistics,
    compute_mmse_statistics,
    compute_updrs_statistics,
)
from core.dataset import (
    discover_recordings,
    load_subject,
    recording_id_from_path,
    session_id_from_path,
    subject_id_from_path,
)
from core.config import load_config as load_preprocessing_config


class MedicationDatasetTests(unittest.TestCase):
    def test_session_aware_bids_identifiers(self):
        path = "ds002778-1.0.5/sub-pd3/ses-off/eeg/sub-pd3_ses-off_task-rest_eeg.bdf"
        self.assertEqual(subject_id_from_path(path), "sub-pd3")
        self.assertEqual(session_id_from_path(path), "ses-off")
        self.assertEqual(recording_id_from_path(path), "sub-pd3_ses-off")

    def test_complete_local_cohort(self):
        participants, recordings = build_cohort(
            "ds002778-1.0.5",
            expected_counts={"HC": 16, "PD_OFF": 15, "PD_ON": 15},
        )
        self.assertEqual(len(participants), 31)
        self.assertEqual(len(recordings), 46)
        self.assertEqual(recordings["participant_id"].nunique(), 31)
        self.assertEqual(int(participants["mmse"].min()), 26)
        self.assertEqual(int(participants["mmse"].max()), 30)
        off_updrs = recordings.loc[
            recordings["condition"].eq("PD_OFF"), "total_updrs"
        ]
        on_updrs = recordings.loc[
            recordings["condition"].eq("PD_ON"), "total_updrs"
        ]
        self.assertEqual((int(off_updrs.min()), int(off_updrs.max())), (20, 58))
        self.assertEqual((int(on_updrs.min()), int(on_updrs.max())), (16, 54))
        self.assertEqual(
            set(
                participants.loc[
                    participants["provenance_sensitivity_exclusion"],
                    "participant_id",
                ]
            ),
            {"sub-pd6", "sub-pd16"},
        )

    def test_bdf_discovery_preserves_all_sessions(self):
        recordings = discover_recordings("ds002778-1.0.5", "rest")
        identifiers = {recording_id_from_path(path) for path in recordings}
        self.assertEqual(len(recordings), 46)
        self.assertIn("sub-pd3_ses-off", identifiers)
        self.assertIn("sub-pd3_ses-on", identifiers)
        self.assertIn("sub-hc1_ses-hc", identifiers)

    def test_bdf_loader_retains_only_32_scalp_electrodes(self):
        path = next(
            path
            for path in discover_recordings("ds002778-1.0.5", "rest")
            if recording_id_from_path(path) == "sub-hc1_ses-hc"
        )
        auxiliary = [
            *(f"EXG{index}" for index in range(1, 9)),
            "Status",
        ]
        raw, provenance = load_subject(path, auxiliary)
        self.assertEqual(len(raw.ch_names), 32)
        self.assertIn("Pz", raw.ch_names)
        self.assertNotIn("EXG1", raw.ch_names)
        self.assertEqual(set(provenance["dropped_auxiliary_channels"]), set(auxiliary))

    def test_analysis_config_is_prespecified(self):
        config = load_analysis_config("config/analyses/ds002778.json")
        self.assertEqual(config["statistics"]["minimum_pairs"], 10)
        self.assertEqual(config["statistics"]["minimum_updrs_participants"], 10)
        self.assertEqual(config["ordinal"]["embedding_dimension"], 6)
        self.assertTrue(config["within_bout_ordinal"]["enabled"])
        self.assertEqual(
            config["within_bout_ordinal"]["metrics"],
            ["entropy", "complexity", "fisher_information"],
        )
        self.assertEqual(config["typical_bouts"]["center_window_seconds"], 0.5)
        self.assertEqual(
            config["typical_bouts"]["representations"],
            [
                "amplitude_envelope",
                "relative_phase_and_resultant_length",
                "phase_aligned_shape",
            ],
        )

    def test_biosemi_preprocessing_contract_is_dataset_specific(self):
        config = load_preprocessing_config("config/preprocessing_ds002778.yaml")
        self.assertEqual(config["project"]["task"], "rest")
        self.assertEqual(config["epochs"]["peak_to_peak_uv"], 500.0)
        self.assertEqual(len(config["channels"]["auxiliary_names"]), 9)
        self.assertIn("Status", config["channels"]["auxiliary_names"])


class MedicationStatisticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        recording_rows = []
        feature_rows = []
        for index in range(16):
            participant_id = f"sub-hc{index + 1}"
            recording_id = f"{participant_id}_ses-hc"
            mmse = 26 + index % 5
            recording_rows.append(
                {
                    "recording_id": recording_id,
                    "participant_id": participant_id,
                    "condition": "HC",
                    "age_years": 52 + index,
                    "sex_male": index % 2,
                    "mmse": mmse,
                    "total_updrs": np.nan,
                    "provenance_sensitivity_exclusion": False,
                }
            )
            feature_rows.append(cls._feature(recording_id, 0.05 * index + 0.02 * (index % 3)))
        for index in range(15):
            participant_id = f"sub-pd{index + 1}"
            mmse = 26 + index % 5
            age = 51 + index
            off = 2.5 + 0.04 * index + 0.2 * (mmse - 28) + 0.03 * (index % 2)
            delta = -0.8 + 0.4 * (mmse - 28) + 0.015 * (index % 3)
            off_updrs = 40.0 + index
            updrs_delta = 5.0 * delta
            for condition, session, value, total_updrs in (
                ("PD_OFF", "ses-off", off, off_updrs),
                ("PD_ON", "ses-on", off + delta, off_updrs + updrs_delta),
            ):
                recording_id = f"{participant_id}_{session}"
                recording_rows.append(
                    {
                        "recording_id": recording_id,
                        "participant_id": participant_id,
                        "condition": condition,
                        "age_years": age,
                        "sex_male": index % 2,
                        "mmse": mmse,
                        "total_updrs": total_updrs,
                        "provenance_sensitivity_exclusion": index in {5, 14},
                    }
                )
                feature_rows.append(cls._feature(recording_id, value))
        cls.recordings = pd.DataFrame.from_records(recording_rows)
        cls.features = pd.DataFrame.from_records(feature_rows)
        cls.config = {
            "statistics": {
                "minimum_per_condition": 10,
                "minimum_pairs": 10,
                "minimum_mmse_participants": 10,
                "minimum_updrs_participants": 10,
                "confidence_level": 0.95,
                "bootstrap_resamples": 200,
                "random_seed": 42,
                "fdr_alpha": 0.05,
            }
        }

    @staticmethod
    def _feature(recording_id, value):
        return {
            "recording_id": recording_id,
            "duration_variant": "all_retained",
            "feature_id": "psd_beta_relative_power",
            "family": "psd",
            "domain": "psd",
            "band": "beta",
            "metric": "relative_power",
            "value": value,
        }

    def test_condition_contrasts_keep_pd_sessions_paired(self):
        result = compute_condition_statistics(
            self.features, self.recordings, self.config
        )
        primary = result.loc[result["sensitivity_cohort"].eq("all_participants")]
        self.assertEqual(set(primary["contrast"]), {
            "PD_OFF_minus_HC",
            "PD_ON_minus_HC",
            "PD_ON_minus_PD_OFF",
        })
        paired = primary.loc[primary["contrast"].eq("PD_ON_minus_PD_OFF")].iloc[0]
        self.assertEqual(paired["n_pairs"], 15)
        self.assertLess(paired["effect"], 0.0)
        self.assertEqual(paired["analysis_status"], "ok")

    def test_mmse_model_detects_medication_delta_slope(self):
        result = compute_mmse_statistics(self.features, self.recordings, self.config)
        row = result.loc[
            result["sensitivity_cohort"].eq("all_participants")
            & result["mmse_model"].eq("PD_ON_minus_PD_OFF")
        ].iloc[0]
        self.assertEqual(row["n_participants"], 15)
        self.assertEqual(row["analysis_status"], "ok")
        self.assertAlmostEqual(row["mmse_slope"], 0.4, delta=0.04)
        self.assertTrue(np.isfinite(row["statistic"]))
        self.assertTrue(np.isfinite(row["primary_p_value"]))

    def test_updrs_model_preserves_paired_session_changes(self):
        result = compute_updrs_statistics(
            self.features, self.recordings, self.config
        )
        primary = result.loc[
            result["sensitivity_cohort"].eq("all_participants")
        ]
        self.assertEqual(
            set(primary["updrs_model"]),
            {"PD_OFF", "PD_ON", "PD_ON_minus_PD_OFF"},
        )
        self.assertNotIn(
            "insufficient_mmse_information", set(result["analysis_status"])
        )
        change = primary.loc[
            primary["updrs_model"].eq("PD_ON_minus_PD_OFF")
        ].iloc[0]
        self.assertEqual(change["n_participants"], 15)
        self.assertEqual(change["analysis_status"], "ok")
        self.assertGreater(change["statistic"], 0.99)
        self.assertLess(change["primary_p_value"], 0.001)

    def test_all_zero_paired_differences_are_a_valid_null_without_warnings(self):
        rows = []
        for index in range(10):
            participant_id = f"sub-pd{index + 1}"
            for condition in ("PD_OFF", "PD_ON"):
                rows.append(
                    {
                        "participant_id": participant_id,
                        "condition": condition,
                        "value": 2.0,
                    }
                )
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            result = _paired_contrast(
                pd.DataFrame.from_records(rows),
                minimum_pairs=10,
                confidence_level=0.95,
                bootstrap_resamples=200,
                seed=42,
            )
        self.assertEqual(result["analysis_status"], "ok")
        self.assertEqual(result["effect"], 0.0)
        self.assertEqual(result["statistic"], 0.0)
        self.assertEqual(result["primary_p_value"], 1.0)
        self.assertEqual(result["wilcoxon_statistic"], 0.0)
        self.assertEqual(result["wilcoxon_p_value"], 1.0)

    def test_focused_bout_mmse_selection_includes_six_requested_features(self):
        rows = [
            {"family": family, "band": band, "metric": metric}
            for family, band, metric, _ in FOCUSED_BOUT_MMSE_FEATURES
        ]
        rows.append(
            {"family": "bouts", "band": "gamma", "metric": "bouts_per_minute"}
        )
        selected = select_focused_bout_mmse_rows(pd.DataFrame.from_records(rows))
        self.assertEqual(len(selected), 6)
        self.assertEqual(
            set(map(tuple, selected[["family", "band", "metric"]].to_numpy())),
            {
                (family, band, metric)
                for family, band, metric, _ in FOCUSED_BOUT_MMSE_FEATURES
            },
        )

    def test_focused_delta_updrs_selection_includes_h_c_and_f(self):
        rows = [
            {"family": family, "band": band, "metric": metric}
            for family, band, metric, _ in FOCUSED_DELTA_UPDRS_FEATURES
        ]
        rows.append({"family": "ordinal", "band": "theta", "metric": "entropy"})
        selected = select_focused_delta_updrs_rows(
            pd.DataFrame.from_records(rows)
        )
        self.assertEqual(len(selected), 3)


class MedicationComparisonFigureTests(unittest.TestCase):
    @staticmethod
    def _config():
        return {
            "bands": {"delta": [1.0, 4.0]},
            "ebosc": {"bands": []},
            "specparam": {"frequency_range_hz": [4.0, 50.0]},
            "statistics": {
                "bootstrap_resamples": 100,
                "confidence_level": 0.95,
                "random_seed": 42,
            },
            "plots": {
                "dpi": 50,
                "condition_colors": {
                    "HC": "#0072B2",
                    "PD_OFF": "#D55E00",
                    "PD_ON": "#009E73",
                },
            },
        }

    @staticmethod
    def _recordings():
        rows = []
        for participant in ("sub-hc1", "sub-hc2"):
            rows.append(
                {
                    "recording_id": f"{participant}_ses-hc",
                    "participant_id": participant,
                    "condition": "HC",
                    "mmse": 29,
                }
            )
        for participant in ("sub-pd1", "sub-pd2"):
            for condition, session in (("PD_OFF", "off"), ("PD_ON", "on")):
                rows.append(
                    {
                        "recording_id": f"{participant}_ses-{session}",
                        "participant_id": participant,
                        "condition": condition,
                        "mmse": 28,
                    }
                )
        return pd.DataFrame.from_records(rows)

    def test_violin_battery_keeps_all_missing_declared_features(self):
        recordings = self._recordings()
        features = pd.DataFrame.from_records(
            {
                "recording_id": row.recording_id,
                "duration_variant": "all_retained",
                "feature_id": "psd_gamma_relative_power",
                "family": "psd",
                "band": "gamma",
                "metric": "relative_power",
                "value": np.nan,
            }
            for row in recordings.itertuples()
        )
        with TemporaryDirectory() as temporary_directory:
            paths = plot_all_feature_violins(
                features,
                recordings,
                temporary_directory,
                self._config(),
            )
            self.assertEqual(len(paths), 1)
            self.assertTrue(paths[0].is_file())

    def test_primary_topographic_battery_covers_original_feature_families(self):
        families = {specification.family for specification in TOPOGRAPHIC_METRICS}
        self.assertTrue(
            {"psd", "ordinal", "aperiodic", "bouts", "periodic_peak"}.issubset(
                families
            )
        )

    def test_group_psd_and_topomap_figures_are_written(self):
        recordings = self._recordings()
        psd_rows = []
        feature_rows = []
        montage_channels = [
            "Fp1", "AF3", "F7", "F3", "FC1", "FC5", "T7", "C3",
            "CP1", "CP5", "P7", "P3", "Pz", "PO3", "O1", "Oz",
            "O2", "PO4", "P4", "P8", "CP6", "CP2", "C4", "T8",
            "FC6", "FC2", "F4", "F8", "AF4", "Fp2", "Fz", "Cz",
        ]
        for recording_index, row in recordings.iterrows():
            for frequency in (1.0, 2.0, 3.0):
                psd_rows.append(
                    {
                        "recording_id": row["recording_id"],
                        "duration_variant": "all_retained",
                        "frequency_hz": frequency,
                        "median_psd_uv2_hz": 1.0 + recording_index + frequency,
                    }
                )
            for electrode_index, electrode in enumerate(montage_channels):
                feature_rows.append(
                    {
                        "recording_id": row["recording_id"],
                        "duration_variant": "all_retained",
                        "electrode": electrode,
                        "family": "psd",
                        "metric": "relative_power",
                        "band": "delta",
                        "value": 0.1 + 0.001 * electrode_index + 0.01 * recording_index,
                    }
                )
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            psd_paths = plot_group_psd_curves(
                pd.DataFrame.from_records(psd_rows),
                recordings,
                root / "psd",
                self._config(),
            )
            topomap_paths = plot_group_mean_topomaps(
                pd.DataFrame.from_records(feature_rows),
                recordings,
                root / "topomaps",
                self._config(),
            )
            self.assertEqual(len(psd_paths), 2)
            self.assertEqual(len(topomap_paths), 1)
            self.assertTrue(all(path.is_file() for path in [*psd_paths, *topomap_paths]))


if __name__ == "__main__":
    unittest.main()
