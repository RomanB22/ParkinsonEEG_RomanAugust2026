"""Regression checks for the refactored public pipeline contract."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from parkinson_eeg.config import load_pipeline_config
from parkinson_eeg.registry import build_registry
from parkinson_eeg.runner import PipelineRunner, Selection, profile_targets
from parkinson_eeg.stages import RunContext, StateStore


class PipelineRefactorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_pipeline_config()
        cls.registry = build_registry()

    def test_one_public_runner_and_thin_compatibility_aliases(self) -> None:
        public = Path("run_pipeline.sh").read_text(encoding="utf-8")
        self.assertIn("python -m parkinson_eeg", public)
        for filename in (
            "run_reproducible_pipeline.sh",
            "run_all_analyses.sh",
            "matched_analysis/run_matched_analyses.sh",
        ):
            source = Path(filename).read_text(encoding="utf-8")
            self.assertLess(len(source.splitlines()), 15)
            self.assertIn("run_pipeline.sh", source)

    def test_public_config_locks_scientific_invariants(self) -> None:
        science = self.config.science
        self.assertEqual(science.psd_range_hz, (1.0, 50.0))
        self.assertEqual(science.aperiodic_fit_range_hz, (4.0, 50.0))
        self.assertEqual(science.ordinal_dimensions, (3, 4, 5, 6))
        self.assertEqual(science.ordinal_delay_samples, (1,))
        self.assertEqual(science.scalar_colormap, "viridis")
        self.assertNotIn("broad_5_15", science.bout_bands)

    def test_stage_identifiers_are_unique_and_dependencies_exist(self) -> None:
        self.assertEqual(len(self.registry), 33)
        for stage in self.registry.values():
            self.assertTrue(stage.label)
            for dependency in stage.dependencies:
                self.assertIn(dependency, self.registry)

    def test_compute_profile_is_bounded_and_bycycle_is_opt_in(self) -> None:
        compute = self.config.profile("compute")
        targets = profile_targets(compute, Selection(cohorts=("full",)))
        self.assertIn("full.psd", targets)
        self.assertIn("full.within-bout-ordinal", targets)
        self.assertNotIn("full.classification", targets)
        self.assertNotIn("full.bycycle", targets)

        full_qc = self.config.profile("full-qc")
        qc_targets = profile_targets(full_qc, Selection())
        self.assertIn("full.bycycle", qc_targets)
        self.assertIn("matched.bycycle", qc_targets)

    def test_matched_stages_reuse_full_feature_owners(self) -> None:
        self.assertIn("full.ordinal", self.registry["matched.ordinal"].dependencies)
        self.assertIn("full.scale-free", self.registry["matched.scale-free"].dependencies)
        self.assertIn(
            "full.within-bout-ordinal",
            self.registry["matched.within-bout-ordinal"].dependencies,
        )

    def test_stale_or_missing_stage_commands_replace_partial_outputs(self) -> None:
        context = RunContext(
            project_root=Path.cwd(),
            environment_name="MNE_August2026",
            profile_name="paper",
        )
        command = self.registry["full.ordinal"].build_commands(context, True)[0]
        self.assertIn("--overwrite", command)

    def test_status_inspection_does_not_write_runner_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state")
            context = RunContext(
                project_root=Path.cwd(),
                environment_name="MNE_August2026",
                profile_name="compute",
            )
            runner = PipelineRunner(self.config, context, state_store=store)
            stages = runner.stages_for_profile(
                self.config.profile("compute"),
                Selection(cohorts=("full",), include_preprocessing=False),
            )
            runner.inspect(stages)
            self.assertFalse(store.root.exists())

    def test_sweep_owns_only_d3_to_d5_at_tau_one(self) -> None:
        command = self.registry["full.ordinal-sweep"].build_commands(
            RunContext(Path.cwd(), "MNE_August2026", "paper"), True
        )[0]
        self.assertIn("parkinson_eeg.sweep", command)
        source = Path("parkinson_eeg/sweep.py").read_text(encoding="utf-8")
        self.assertIn("dimensions: tuple[int, ...] = (3, 4, 5)", source)
        self.assertIn("delay_samples: int = 1", source)

    def test_preprocessing_validation_excludes_optional_untracked_workflow(self) -> None:
        command = self.registry["preprocessing-tests"].build_commands(
            RunContext(Path.cwd(), "MNE_August2026", "paper"), False
        )[0]
        self.assertNotIn("tests.test_simple_pipeline", command)


if __name__ == "__main__":
    unittest.main()
