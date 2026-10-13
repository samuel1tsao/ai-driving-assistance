"""Profile reruns obey saved experiments; analyze and reuse never train."""

import copy
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import nbformat
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_fog_research import load_notebook_runtime  # noqa: E402

workflow = load_notebook_runtime()
describe_run_identity_changes = workflow.describe_run_identity_changes


class ProfileContinuation(unittest.TestCase):
    def setUp(self):
        runtime = ROOT / ".runtime/profile-continuation-tests"
        runtime.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=runtime)
        self.addCleanup(self.temporary.cleanup)
        self.original_directory = Path.cwd()
        self.addCleanup(os.chdir, self.original_directory)
        self.folder = Path(self.temporary.name)
        notebook = nbformat.read(ROOT / "notebooks/fog_density_consolidated.ipynb", 4)
        namespace = {}
        for cell in notebook.cells:
            if cell.metadata.get("fogStudyConstants") or cell.metadata.get(
                "fogStudySettings"
            ):
                exec(cell.source, namespace)
        self.settings = namespace["SETTINGS"]
        self.settings.update(
            runtime_root=str(self.folder / "runtime"),
            drive_mount=str(self.folder / "drive"),
        )
        self.settings["profile"].update(name="partial-2k", initialization="pretrained")
        self.settings["data"].update(
            location={"kind": "path", "path": str(self.folder)}, low_disk=False
        )
        self.settings["output"].update(
            location={"kind": "path", "path": str(self.folder)},
            run_name="partial-2k-v1",
            export_name="reports",
        )
        self.settings["training"].update(device="cpu", amp=False)

    def existing_run(self, complete):
        configuration, paths = workflow.resolve_study_configuration(
            copy.deepcopy(self.settings)
        )
        recorded = copy.deepcopy(configuration)
        recorded.pop("density_sampling", None)
        recorded.pop("overwrite_existing", None)
        recorded["profile_scope"].pop("density_sampling", None)
        recorded["profile_scope"].pop("overwrite_existing", None)
        recorded["profile_scope"].pop("version", None)
        (paths["run"] / "settings.json").write_text(
            json.dumps({"config": recorded, "versions": {"torch": "historical"}})
        )
        for model in recorded["checkpoints"]:
            for arm in recorded["arms"]:
                directory = paths["run"] / f"{model}-{arm}-42"
                directory.mkdir()
                (directory / "status.json").write_text(
                    json.dumps({"status": "complete" if complete else "interrupted"})
                )
                (directory / "results.csv").write_text("model,mAP50_95\nfixture,0.5\n")
        pd.DataFrame([{"status": "complete"}]).to_csv(
            paths["run"] / "status.csv", index=False
        )
        return recorded, paths["run"]

    def test_finished_run_reuses_without_data_weights_identity_or_training(self):
        recorded, output = self.existing_run(True)
        configuration, paths = workflow.resolve_study_configuration(
            copy.deepcopy(self.settings)
        )
        self.assertEqual(configuration, recorded)
        self.assertEqual(workflow.SETTINGS["profile"]["density_sampling"], "natural")
        with (
            patch.object(
                workflow,
                "prepare_profile_dataset_subset",
                side_effect=AssertionError("no extraction"),
            ),
            patch.object(
                workflow,
                "load_and_validate_manifest",
                side_effect=AssertionError("no image loading"),
            ),
            patch.object(
                workflow,
                "run_training_preflight",
                side_effect=AssertionError("no training preflight"),
            ),
            patch.object(
                workflow,
                "run_configured_detector_study",
                side_effect=AssertionError("no training"),
            ),
            patch.object(
                workflow,
                "execute_analysis",
                return_value=pd.DataFrame([{"mAP50_95": 0.5}]),
            ),
        ):
            workflow.prepare_fog_data()
            rows = workflow.load_and_validate_dataset_annotations(configuration)
            self.assertEqual(rows, [])
            checks = workflow.display_training_readiness_checks(configuration, rows)
            results, status = workflow.train_or_analyze_configured_study(
                "train", configuration, rows, checks
            )
            workflow.verify_subset_completion_and_checkpoint_reuse(results, rows)
            self.assertFalse(results.empty)
        self.assertEqual(paths["run"], output)
        self.assertEqual(
            json.loads((output / "settings.json").read_text())["config"], recorded
        )

    def test_unfinished_run_resumes_saved_sampling_and_budget(self):
        recorded, output = self.existing_run(False)
        changed_defaults = copy.deepcopy(self.settings)
        changed_defaults["profile"].update(
            epochs=99, density_epochs=99, density_sampling="balanced_bins"
        )
        configuration, paths = workflow.resolve_study_configuration(changed_defaults)
        self.assertEqual(configuration, recorded)
        self.assertFalse(workflow.should_reuse_completed_training(configuration))
        with patch.object(
            workflow,
            "run_configured_detector_study",
            return_value=(pd.DataFrame(), None),
        ) as train:
            workflow.train_or_analyze_configured_study(
                "train", configuration, [], pd.DataFrame([{"ready": True}])
            )
            self.assertEqual(train.call_args.args[1], recorded)

    def test_analyze_routes_only_to_saved_reports_and_unknown_action_never_trains(self):
        configuration, paths = workflow.resolve_study_configuration(
            copy.deepcopy(self.settings)
        )
        with (
            patch.object(
                workflow,
                "run_configured_detector_study",
                side_effect=AssertionError("no training"),
            ),
            patch.object(
                workflow,
                "run_checkpoint_evaluation",
                side_effect=AssertionError("no inference"),
            ),
            patch.object(workflow, "execute_analysis", return_value=pd.DataFrame()),
        ):
            results, status = workflow.train_or_analyze_configured_study(
                " ANALYZE ", configuration, None, None
            )
            self.assertIsNone(status)
            with self.assertRaisesRegex(ValueError, "unknown actions"):
                workflow.train_or_analyze_configured_study(
                    "typo", configuration, [], None
                )

    def test_retrain_allocates_fresh_sibling_and_preserves_previous_run(self):
        recorded, previous = self.existing_run(True)
        prior_bytes = (previous / "settings.json").read_bytes()
        settings = copy.deepcopy(self.settings)
        settings["retrain"] = True
        configuration, paths = workflow.resolve_study_configuration(settings)
        self.assertNotEqual(paths["run"], previous)
        self.assertEqual(paths["run"].parent, previous.parent)
        self.assertEqual(configuration["density_sampling"], "natural")
        self.assertEqual(configuration["initialization"], "pretrained")
        self.assertEqual(
            configuration["density_checkpoint"],
            str(self.folder / "colab-full-study" / "density-42" / "best.pt"),
        )
        self.assertFalse((paths["run"] / "settings.json").exists())
        self.assertEqual((previous / "settings.json").read_bytes(), prior_bytes)
        self.assertFalse(workflow.should_reuse_completed_training(configuration))
        settings = copy.deepcopy(self.settings)
        settings["output"]["run_name"] = paths["run"].name
        continued, continued_paths = workflow.resolve_study_configuration(settings)
        self.assertEqual(continued_paths["run"], paths["run"])

    def test_retrain_is_boolean_and_only_for_training(self):
        for action, flag in [("analyze", True), ("evaluate", True), ("train", "yes")]:
            settings = copy.deepcopy(self.settings)
            settings.update(action=action, retrain=flag)
            with self.assertRaisesRegex(ValueError, "retrain"):
                workflow.resolve_study_configuration(settings)

    def test_legacy_full_configuration_keeps_full_continuation_identity(self):
        self.settings["profile"].update(name="full", initialization="existing_full_run")
        self.settings["output"]["run_name"] = "full-run"
        configuration, paths = workflow.resolve_study_configuration(
            copy.deepcopy(self.settings)
        )
        configuration.pop("profile_name")
        configuration.pop("initialization")
        (paths["run"] / "settings.json").write_text(
            json.dumps({"config": configuration})
        )
        restored, _ = workflow.resolve_study_configuration(copy.deepcopy(self.settings))
        self.assertEqual(restored["profile_name"], "full")
        self.assertEqual(restored["initialization"], "existing_full_run")

    def test_mismatch_message_identifies_package_and_sampling(self):
        message = describe_run_identity_changes(
            {"config": {"density_sampling": "natural"}, "versions": {"torch": "old"}},
            {
                "config": {"density_sampling": "balanced_bins"},
                "versions": {"torch": "new"},
            },
        )
        self.assertIn("config.density_sampling", message)
        self.assertIn("versions.torch", message)
        self.assertIn("natural", message)
        self.assertIn("balanced_bins", message)


class FrozenPredictorSetup(unittest.TestCase):
    def setUp(self):
        runtime = ROOT / ".runtime/predictor-tests"
        runtime.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=runtime)
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)
        self.settings = {
            "action": workflow.ACTION_TRAIN,
            "profile": {"name": workflow.PROFILE_2K},
            "density_predictor": {"seed": workflow.DEFAULT_SEED},
        }
        self.config = {
            "output": str(self.folder / "run"),
            "density_checkpoint": str(self.folder / "predictor.pt"),
            "profile_name": workflow.PROFILE_2K,
            "arms": list(workflow.DETECTION_ARMS),
            "device": "cpu",
        }
        for name, value in (
            ("SETTINGS", self.settings),
            ("FOG_DENSITY_MODEL", None),
            ("FOG_DENSITY_MODEL_KEY", None),
        ):
            patcher = patch.object(workflow, name, value, create=True)
            patcher.start()
            self.addCleanup(patcher.stop)

    def save_predictor(self, path, bias=0.0):
        model = workflow.FogDensityRegressor()
        with torch.no_grad():
            model.features[-1].weight.zero_()
            model.features[-1].bias.fill_(bias)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict()}, path)
        return model

    def test_load_freezes_exact_weights_and_reuses_the_predictor(self):
        path = Path(self.config["density_checkpoint"])
        expected = self.save_predictor(path)
        model = workflow.load_fog_density_model(path, "cpu")
        self.assertFalse(model.training)
        self.assertTrue(
            all(not parameter.requires_grad for parameter in model.parameters())
        )
        for key, tensor in expected.state_dict().items():
            torch.testing.assert_close(tensor, model.state_dict()[key])
        images = torch.rand(2, 3, 32, 32, requires_grad=True)
        beta = workflow.get_fog_density(images)
        self.assertEqual(tuple(beta.shape), (2, 1))
        self.assertFalse(beta.requires_grad)
        self.assertTrue(((beta >= 0) & (beta <= workflow.BETA_SCALE)).all())
        self.assertIs(model, workflow.load_fog_density_model(path, "cpu"))

    def test_replaced_checkpoint_reloads_weights_at_the_same_path(self):
        path = Path(self.config["density_checkpoint"])
        self.save_predictor(path)
        previous = workflow.load_fog_density_model(path, "cpu")
        stamp = path.stat().st_mtime_ns
        self.save_predictor(path, bias=2.0)
        os.utime(path, ns=(stamp + 1000000000, stamp + 1000000000))
        current = workflow.load_fog_density_model(path, "cpu")
        self.assertIsNot(current, previous)
        self.assertAlmostEqual(current.features[-1].bias.item(), 2.0)

    def test_full_run_without_saved_predictor_leaves_setup_to_training(self):
        self.settings["profile"] = {"name": workflow.PROFILE_FULL}
        self.config["profile_name"] = workflow.PROFILE_FULL
        self.config["density_checkpoint"] = None
        self.assertIsNone(workflow.resolve_density_predictor_checkpoint(self.config))

    def test_evaluation_selects_the_source_runs_saved_predictor(self):
        source = self.folder / "source"
        path = (
            source
            / f"{workflow.DENSITY_RUN_PREFIX}-{workflow.DEFAULT_SEED}"
            / workflow.BEST_CHECKPOINT
        )
        self.save_predictor(path, bias=1.0)
        (source / workflow.RUN_SETTINGS_FILE).write_text(
            json.dumps({"config": {"density_checkpoint": None}})
        )
        self.settings["action"] = workflow.ACTION_EVALUATE
        self.config["evaluation_source"] = str(source)
        self.assertEqual(
            workflow.resolve_density_predictor_checkpoint(self.config), path
        )
        model = workflow.load_fog_density_model(path, "cpu")
        self.assertAlmostEqual(model.features[-1].bias.item(), 1.0)

    def test_shared_artifact_names_control_completed_run_detection(self):
        output = Path(self.config["output"])
        directory = (
            output / f"{workflow.YOLO_SMALL}-{workflow.ARM_RGB}-{workflow.DEFAULT_SEED}"
        )
        directory.mkdir(parents=True)
        (output / "experiment.json").write_text("{}")
        (directory / "metrics.csv").write_text("score\n0.5\n")
        (directory / "state.json").write_text('{"status":"complete"}')
        self.config.update(
            checkpoints={workflow.YOLO_SMALL: "unused"},
            arms=[workflow.ARM_RGB],
            seeds=[workflow.DEFAULT_SEED],
        )
        with (
            patch.object(workflow, "RUN_SETTINGS_FILE", "experiment.json"),
            patch.object(workflow, "RESULTS_FILE", "metrics.csv"),
            patch.object(workflow, "STATUS_FILE", "state.json"),
        ):
            self.assertTrue(workflow.should_reuse_completed_training(self.config))

    def test_renamed_resume_file_still_requires_a_run_identity(self):
        output = self.folder / "unidentified"
        directory = output / "comparison"
        directory.mkdir(parents=True)
        (directory / "resume.pt").write_bytes(b"prior-checkpoint")
        with (
            patch.object(workflow, "LAST_CHECKPOINT", "resume.pt"),
            self.assertRaisesRegex(ValueError, "profile identity"),
        ):
            workflow.validate_and_record_run_identity(
                [],
                {"profile_name": workflow.PROFILE_2K, "density_checkpoint": None},
                output,
            )
        self.assertFalse((output / workflow.RUN_SETTINGS_FILE).exists())


if __name__ == "__main__":
    unittest.main()
