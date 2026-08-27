"""Tests for bounded cleanup of outputs from retired analysis bands."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from core.output_cleanup import remove_retired_band_outputs


class OutputCleanupTests(unittest.TestCase):
    def test_removes_only_retired_band_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory)
            canonical = output / "figures" / "alpha" / "plot.png"
            canonical.parent.mkdir(parents=True)
            canonical.touch()
            retired_directory = output / "figures" / "broad_5_15"
            retired_directory.mkdir()
            (retired_directory / "plot.png").touch()
            retired_file = output / "metrics_broad_5_15.csv"
            retired_file.touch()

            removed = remove_retired_band_outputs(output)

            self.assertTrue(canonical.exists())
            self.assertFalse(retired_directory.exists())
            self.assertFalse(retired_file.exists())
            self.assertEqual(len(removed), 2)


if __name__ == "__main__":
    unittest.main()
