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


if __name__ == "__main__":
    unittest.main()
