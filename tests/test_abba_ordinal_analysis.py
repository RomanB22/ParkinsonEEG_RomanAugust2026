"""Unit tests for ABBA-band full-signal and concatenated-bout H/C/F."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from analyses.rhythmicity.abba_ordinal_analysis import (
    CONCATENATION_POLICY,
    _comparison_band_name,
    _cross_dataset_band_name,
    _electrode_metrics,
    _refresh_result_metadata,
    compute_correlations,
    concatenate_bouts,
)
from analyses.rhythmicity.control_band_qc import (
    QC_BAND_DEFINITION,
    control_defined_segments,
)
from analyses.rhythmicity.abba_ordinal_sweep import _seed_compatible_checkpoints


class AbbaOrdinalAnalysisTests(unittest.TestCase):
    def test_checkpoint_refresh_backfills_band_provenance(self) -> None:
        row = {
            "segment_index": 2,
            "band_name": "theta_1",
            "canonical_region": "theta",
            "direction": "low",
        }
        result = {"recording_rows": [row.copy()], "electrode_rows": [row.copy()]}
        task = {
            "participant_id": "sub-001",
            "session_id": "",
            "group": "PD",
            "medication_state": "",
            "moca": 25.0,
            "mmse": 27.0,
            "updrs": 30.0,
            "updrs_source": "table.tsv",
            "age_years": 70.0,
            "sex": "M",
            "dataset": "study",
            "comparison_band_alignments": [],
            "segments": [{
                "segment_index": 2,
                "band_name": "theta_1",
                "source_group": "Control",
                "source_band_name": "theta_2",
                "band_definition": "control_defined_abba",
            }],
        }
        _refresh_result_metadata(result, task)
        for refreshed in (result["recording_rows"][0], result["electrode_rows"][0]):
            self.assertEqual(refreshed["source_group"], "Control")
            self.assertEqual(refreshed["source_band_name"], "theta_2")
            self.assertEqual(refreshed["band_definition"], "control_defined_abba")

    def test_sweep_reuses_only_matching_legacy_dimension_checkpoints(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            checkpoint = source / "intermediate" / "abba_ordinal_checkpoints" / "study" / "one.pkl"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"checkpoint")
            base = {
                "output_dir": str(source),
                "abba_ordinal": {"embedding_dimension": 5, "delay_samples": 1},
            }
            target = root / "target"
            self.assertEqual(_seed_compatible_checkpoints(base, target, 4, 1), 0)
            self.assertEqual(_seed_compatible_checkpoints(base, target, 5, 1), 1)
            self.assertEqual(
                (target / "intermediate" / "abba_ordinal_checkpoints" / "study" / "one.pkl").read_bytes(),
                b"checkpoint",
            )

    def test_sweep_reuses_matching_control_band_checkpoints(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            checkpoint = source / "control_band_qc" / "intermediate" / "abba_ordinal_checkpoints" / "study" / "one.pkl"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"control checkpoint")
            base = {
                "output_dir": str(source),
                "abba_ordinal": {"embedding_dimension": 5, "delay_samples": 1},
            }
            target = root / "target"
            self.assertEqual(
                _seed_compatible_checkpoints(
                    base, target, 5, 1, control_bands_qc=True
                ),
                1,
            )
            copied = target / "control_band_qc" / "intermediate" / "abba_ordinal_checkpoints" / "study" / "one.pkl"
            self.assertEqual(copied.read_bytes(), b"control checkpoint")

    def test_control_defined_segments_use_direction_labels_and_only_control_limits(self) -> None:
        segments = pd.DataFrame([
            {"dataset": "study", "group": "Control", "band_name": "theta_2", "canonical_region": "theta", "direction": "low", "start_hz": 4.5, "end_hz": 7.5},
            {"dataset": "study", "group": "Control", "band_name": "alpha_1", "canonical_region": "alpha", "direction": "high", "start_hz": 8.0, "end_hz": 12.5},
            {"dataset": "study", "group": "PD", "band_name": "theta_1", "canonical_region": "theta", "direction": "low", "start_hz": 3.5, "end_hz": 6.0},
            {"dataset": "study", "group": "Control", "band_name": "gamma_2", "canonical_region": "gamma", "direction": "high", "start_hz": 45.0, "end_hz": 45.0},
        ])
        result = control_defined_segments(segments)
        self.assertEqual(result["band_name"].tolist(), ["theta_low", "alpha_high"])
        self.assertEqual(result["source_band_name"].tolist(), ["theta_2", "alpha_1"])
        self.assertEqual(set(result["source_group"]), {"Control"})
        self.assertEqual(set(result["band_definition"]), {QC_BAND_DEFINITION})

    def test_concatenate_bouts_uses_requested_order_and_boundaries(self) -> None:
        filtered = np.arange(2 * 2 * 8, dtype=float).reshape(2, 2, 8)
        bursts = [
            {"epoch_index": 0, "start_sample": 2, "stop_sample_exclusive": 5},
            {"epoch_index": 1, "start_sample": 4, "stop_sample_exclusive": 7},
        ]
        result = concatenate_bouts(filtered, bursts)
        expected = np.concatenate([filtered[0, :, 2:5], filtered[1, :, 4:7]], axis=1)
        np.testing.assert_array_equal(result, expected)
        self.assertEqual(result.shape, (2, 6))

    def test_empty_bouts_preserve_electrode_axis(self) -> None:
        result = concatenate_bouts(np.zeros((3, 4, 20)), [])
        self.assertEqual(result.shape, (4, 0))

    def test_electrode_metrics_return_finite_h_c_f(self) -> None:
        rng = np.random.default_rng(14)
        rows = _electrode_metrics(
            rng.normal(size=(2, 500)),
            ["C3", "C4"],
            dx=3,
            tau=1,
            minimum_samples=20,
        )
        self.assertEqual([row["electrode"] for row in rows], ["C3", "C4"])
        for row in rows:
            self.assertTrue(np.isfinite(row["entropy"]))
            self.assertTrue(np.isfinite(row["complexity"]))
            self.assertTrue(np.isfinite(row["fisher_information"]))
            self.assertEqual(row["n_ordinal_patterns"], 498)

    def test_policy_explicitly_allows_cross_join_patterns(self) -> None:
        self.assertIn("may_cross_joins", CONCATENATION_POLICY)

    def test_medication_low_theta_alignment_uses_control_theta_2(self) -> None:
        alignment = [{
            "dataset": "medication_state",
            "comparison_band_name": "theta_low_aligned",
            "source_band_by_group": {
                "Control": "theta_2", "PD_OFF": "theta_1", "PD_ON": "theta_1"
            },
            "required_direction": "low",
        }]
        for group, source in (("Control", "theta_2"), ("PD_OFF", "theta_1"), ("PD_ON", "theta_1")):
            task = {"dataset": "medication_state", "group": group, "comparison_band_alignments": alignment}
            self.assertEqual(_comparison_band_name(task, source, "low"), "theta_low_aligned")
        control = {"dataset": "medication_state", "group": "Control", "comparison_band_alignments": alignment}
        self.assertEqual(_comparison_band_name(control, "theta_1", "high"), "theta_1")

    def test_control_high_theta_can_be_labeled_as_noncomparison(self) -> None:
        task = {
            "dataset": "medication_state",
            "group": "Control",
            "comparison_band_alignments": [{
                "dataset": "medication_state",
                "comparison_band_name": "theta_high_control_only",
                "source_band_by_group": {"Control": "theta_1"},
                "required_direction": "high",
            }],
        }
        self.assertEqual(
            _comparison_band_name(task, "theta_1", "high"),
            "theta_high_control_only",
        )

    def test_cross_dataset_band_uses_region_and_lavi_direction(self) -> None:
        self.assertEqual(_cross_dataset_band_name("theta", "low"), "theta_low")
        self.assertEqual(_cross_dataset_band_name("beta", "high"), "beta_high")

    def test_paired_change_tolerates_one_state_with_all_missing_metrics(self) -> None:
        rows = []
        for index in range(6):
            for group in ("PD_OFF", "PD_ON"):
                rows.append({
                    "dataset": "medication_state",
                    "participant_id": f"sub-{index}",
                    "group": group,
                    "band_name": "theta_1",
                    "scope": "within_bout",
                    "moca": np.nan,
                    "mmse": 25 + index,
                    "updrs": 40 + index + (group == "PD_ON"),
                    "entropy": np.nan if group == "PD_ON" else index / 10,
                    "complexity": index / 20,
                    "fisher_information": index / 30,
                })
        result = compute_correlations(pd.DataFrame(rows), minimum_n=5)
        selected = result.loc[
            result["group_model"].eq("PD_ON_minus_PD_OFF")
            & result["metric"].eq("entropy")
        ].iloc[0]
        self.assertEqual(selected["n"], 0)
        self.assertTrue(np.isnan(selected["spearman_rho"]))


if __name__ == "__main__":
    unittest.main()
