"""Regression checks for interruption-safe analysis resumption."""

from __future__ import annotations

import re
import unittest
from pathlib import Path


class PipelineResumeTests(unittest.TestCase):
    def _run_stage_body(self, filename: str) -> str:
        source = Path(filename).read_text(encoding="utf-8")
        match = re.search(r"\nrun_stage\(\) \{(?P<body>.*?)\n\}", source, re.DOTALL)
        self.assertIsNotNone(match, f"run_stage was not found in {filename}")
        return match.group("body")

    def test_noncurrent_full_stage_always_replaces_partial_output(self) -> None:
        body = self._run_stage_body("run_all_analyses.sh")
        self.assertIn("command+=(--overwrite)", body)
        self.assertNotIn('-e "$sentinel"', body)

    def test_noncurrent_matched_stage_always_replaces_partial_output(self) -> None:
        body = self._run_stage_body("matched_analysis/run_matched_analyses.sh")
        self.assertIn("command+=(--overwrite)", body)
        self.assertNotIn('-e "$sentinel"', body)

    def test_noncurrent_ordinal_sweeps_are_overwritten(self) -> None:
        full = Path("run_all_analyses.sh").read_text(encoding="utf-8")
        matched = Path("matched_analysis/run_matched_analyses.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("ordinal_sweep_command+=(--overwrite)", full)
        self.assertIn("sweep_command+=(--overwrite)", matched)

    def test_independent_bycycle_stage_is_in_full_and_matched_runs(self) -> None:
        full = Path("run_all_analyses.sh").read_text(encoding="utf-8")
        matched = Path("matched_analysis/run_matched_analyses.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("bycycle_burst_analysis/run_bycycle_burst_analysis.sh", full)
        self.assertIn("bycycle_burst_analysis/run_bycycle_burst_analysis.sh", matched)
        self.assertIn("$CONFIG_ROOT/bycycle_burst.json", matched)

    def test_independent_bycycle_stage_is_opt_in_from_every_runner(self) -> None:
        wrapper = Path("run_reproducible_pipeline.sh").read_text(encoding="utf-8")
        full = Path("run_all_analyses.sh").read_text(encoding="utf-8")
        matched = Path("matched_analysis/run_matched_analyses.sh").read_text(
            encoding="utf-8"
        )
        for source in (wrapper, full, matched):
            self.assertIn("INCLUDE_BYCYCLE_BURSTS=false", source)
            self.assertIn("--include-bycycle-bursts", source)
        self.assertIn(
            'if [[ "$INCLUDE_BYCYCLE_BURSTS" == true ]]; then', full
        )
        self.assertIn(
            'if [[ "$INCLUDE_BYCYCLE_BURSTS" == true ]]; then', matched
        )

    def test_public_runner_exposes_bounded_profiles(self) -> None:
        wrapper = Path("run_reproducible_pipeline.sh").read_text(encoding="utf-8")
        downstream = Path("run_all_analyses.sh").read_text(encoding="utf-8")
        for source in (wrapper, downstream):
            self.assertIn("--profile", source)
            self.assertIn("compute", source)
            self.assertIn("paper", source)
            self.assertIn("full-qc", source)
        self.assertIn('PROFILE="paper"', wrapper)
        self.assertIn('PROFILE="paper"', downstream)

    def test_dimension_sweep_does_not_recompute_primary_d6(self) -> None:
        sweep = Path("ordinal_analysis/run_ordinal_parameter_sweep.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("embedding_dimensions=(3 4 5)", sweep)
        self.assertNotIn("embedding_dimensions=(3 4 5 6)", sweep)
        self.assertIn("feature_source_sweep_root", sweep)

    def test_preprocessing_validation_does_not_require_absent_optional_workflow(self) -> None:
        for filename in (
            "scripts/run_full_cleaning.sh",
            "scripts/create_conda_environment.sh",
        ):
            source = Path(filename).read_text(encoding="utf-8")
            self.assertNotIn("tests.test_simple_pipeline", source)

    def test_master_runner_exposes_parallel_preprocessing_and_progress(self) -> None:
        wrapper = Path("run_reproducible_pipeline.sh").read_text(encoding="utf-8")
        cleaning = Path("scripts/run_full_cleaning.sh").read_text(encoding="utf-8")
        batch = Path("scripts/run_preprocessing.py").read_text(encoding="utf-8")
        self.assertIn("--preprocessing-workers", wrapper)
        self.assertIn("--workers", cleaning)
        self.assertIn("conda run --no-capture-output", cleaning)
        self.assertIn("ProcessPoolExecutor", batch)
        self.assertIn('desc="ICA cleaning"', batch)
        self.assertIn("check_preprocessing_outputs.py", wrapper)
        self.assertIn('"$cleaning_rebuilt" == true', wrapper)


if __name__ == "__main__":
    unittest.main()
