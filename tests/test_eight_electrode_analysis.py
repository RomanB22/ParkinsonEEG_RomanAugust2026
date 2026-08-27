"""Tests for the separate eight-electrode sensitivity battery."""

from __future__ import annotations

import unittest
from pathlib import Path

from analyses.eight_electrode.pipeline import (
    ELECTRODES,
    build_analysis_tables,
    load_analysis_config,
)


class EightElectrodeAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_analysis_config("config/analyses/eight_electrode.json")

    def test_scope_is_exact_and_uses_canonical_bands(self) -> None:
        self.assertEqual(self.config["electrodes"], ELECTRODES)
        tested = set().union(
            self.config["bands"]["psd"],
            self.config["bands"]["ordinal"],
            self.config["bands"]["bout"],
        )
        self.assertEqual(
            tested,
            {"delta", "theta", "alpha", "beta", "low_gamma", "low_beta", "high_beta"},
        )

    @unittest.skipUnless(
        Path("processed/metadata/subjects.csv").is_file(),
        "requires generated feature caches",
    )
    def test_real_tables_cover_all_domains_and_exact_electrode_set(self) -> None:
        values, dictionary, subject, electrode = build_analysis_tables(self.config)
        self.assertEqual(len(dictionary), 235)
        self.assertEqual(len(values), 149 * 235)
        self.assertEqual(len(subject), 235)
        self.assertEqual(len(electrode), 235 * 8)
        self.assertTrue(values["n_electrodes_contributing"].between(0, 8).all())
        self.assertEqual(set(electrode["electrode"]), set(ELECTRODES))
        self.assertFalse(dictionary["feature_id"].str.contains("broad_5_15").any())
        self.assertTrue(
            {
                "psd_relative_power", "ordinal_broadband", "ordinal_band",
                "aperiodic", "periodic_bout", "within_bout_ordinal",
            }.issubset(set(dictionary["domain"]))
        )


if __name__ == "__main__":
    unittest.main()
