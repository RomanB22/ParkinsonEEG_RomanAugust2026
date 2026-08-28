"""Regression checks for the refactored public pipeline contract."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from config import load_pipeline_config
from registry import build_registry
from runner import PipelineRunner, Selection, profile_targets
from stages import RunContext, StateStore
from stages import Stage


class PipelineRefactorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_pipeline_config()
        cls.registry = build_registry()

    def test_one_public_runner_and_thin_compatibility_aliases(self) -> None:
        public = Path("run_pipeline.sh").read_text(encoding="utf-8")
        for filename in (
            "run_reproducible_pipeline.sh",
            "run_all_analyses.sh",
            "src/analyses/matching/run_matched_analyses.sh",
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
        self.assertEqual(
            science.frequency_bands,
            {
                "delta": (1.0, 4.0),
                "theta": (4.0, 8.0),
                "alpha": (8.0, 13.0),
                "beta": (13.0, 30.0),
                "gamma": (30.0, 50.0),
            },
        )

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

    def test_no_stage_command_receives_implicit_overwrite(self) -> None:
        context = RunContext(
            project_root=Path.cwd(),
            environment_name="MNE_August2026",
            profile_name="paper",
        )
        for stage_id, stage in self.registry.items():
            for command in stage.build_commands(context, True):
                self.assertNotIn(
                    "--overwrite",
                    command,
                    msg=f"{stage_id} added an implicit --overwrite",
                )

        explicit = RunContext(
            project_root=Path.cwd(),
            environment_name="MNE_August2026",
            profile_name="paper",
            overwrite=True,
        )
        for stage_id in (
            "clean",
            "full.psd",
            "full.ordinal",
            "full.ordinal-sweep",
            "full.scale-free",
            "full.bycycle",
            "full.fit-qc",
        ):
            commands = self.registry[stage_id].build_commands(explicit, True)
            self.assertTrue(
                any("--overwrite" in command for command in commands),
                msg=f"{stage_id} ignored explicit --overwrite",
            )

    def test_stale_clean_does_not_implicitly_refit_ica(self) -> None:
        resumable = RunContext(
            project_root=Path.cwd(),
            environment_name="MNE_August2026",
            profile_name="compute",
            skip_manual_ica_review=True,
        )
        command = self.registry["clean"].build_commands(resumable, True)[0]
        self.assertIn("--skip-manual-ica-review", command)
        self.assertNotIn("--overwrite", command)

        explicit_overwrite = RunContext(
            project_root=Path.cwd(),
            environment_name="MNE_August2026",
            profile_name="compute",
            overwrite=True,
            skip_manual_ica_review=True,
        )
        command = self.registry["clean"].build_commands(
            explicit_overwrite, True
        )[0]
        self.assertIn("--overwrite", command)

    def test_compute_ordinal_manifest_does_not_require_figure_only_fields(self) -> None:
        manifest = next(
            artifact
            for artifact in self.registry["full.ordinal"].artifacts
            if artifact.path.name == "manifest.json"
        )
        self.assertNotIn('"subject_topomaps_generated": false', manifest.contains)
        self.assertIn("paper", self.registry["full.ordinal"].profile_artifacts)

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

    def test_complete_stale_stage_is_adopted_without_execution(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state")
            context = RunContext(
                project_root=Path.cwd(),
                environment_name="MNE_August2026",
                profile_name="paper",
            )
            runner = PipelineRunner(self.config, context, state_store=store)
            builder = Mock(return_value=[["command-that-must-not-run"]])
            stage = Stage(
                "test.stale",
                "Complete stale test output",
                "full",
                "test",
                (),
                builder,
                (),
            )
            runner.inspect = Mock(
                return_value=[(stage, "stale", "source changed", "new-fingerprint")]
            )

            runner.run([stage], dry_run=False)

            builder.assert_called_once_with(context, True)
            record = store.read(stage.id)
            self.assertIsNotNone(record)
            self.assertEqual(record["fingerprint"], "new-fingerprint")
            self.assertTrue(record["adopted_existing_outputs"])

    def test_sweep_owns_only_d3_to_d5_at_tau_one(self) -> None:
        command = self.registry["full.ordinal-sweep"].build_commands(
            RunContext(Path.cwd(), "MNE_August2026", "paper"), True
        )[0]
        self.assertIn("sweep", command)
        source = Path("sweep.py").read_text(encoding="utf-8")
        self.assertIn("dimensions: tuple[int, ...] = (3, 4, 5)", source)
        self.assertIn("delay_samples: int = 1", source)

    def test_preprocessing_validation_excludes_optional_untracked_workflow(self) -> None:
        command = self.registry["preprocessing-tests"].build_commands(
            RunContext(Path.cwd(), "MNE_August2026", "paper"), False
        )[0]
        self.assertNotIn("tests.test_simple_pipeline", command)


if __name__ == "__main__":
    unittest.main()
