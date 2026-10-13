"""Offline matched-scene fixtures test mechanics, never scientific findings."""

import copy
import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import nbformat
import pandas as pd
import torch
from filelock import FileLock, Timeout
from PIL import Image
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_fog_research import load_notebook_runtime  # noqa: E402

training = analytics = load_notebook_runtime()
prepare_cityscapes_manifest = training.prepare_cityscapes_manifest
load_and_validate_manifest = training.load_and_validate_manifest


class FixtureDetector:
    calls = 0

    def __init__(self, name, *args, **kwargs):
        self.name = name
        self.model = nn.Linear(1, 1)

    def predict_objects(self, image, row):
        FixtureDetector.calls += 1
        return [
            {"box": box, "label": label, "score": 0.9}
            for box, label in zip(row["boxes"], row["labels"])
        ]


class FixtureAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, image, beta):
        return image * self.scale


class FixtureDensity(nn.Module):
    def __init__(self):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(0.01))

    def forward(self, image):
        return self.beta.reshape(1, 1)


class MatchedFogEvaluation(unittest.TestCase):
    def setUp(self):
        runtime = ROOT / ".runtime/matched-fog-tests"
        runtime.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=runtime)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        for index in range(2):
            base = f"frankfurt_00000{index}_000001"
            for folder in ("leftImg8bit", "leftImg8bit_foggy", "gtFine"):
                (self.root / folder / "val/frankfurt").mkdir(
                    parents=True, exist_ok=True
                )
            Image.new("RGB", (32, 32), "white").save(
                self.root / "leftImg8bit/val/frankfurt" / f"{base}_leftImg8bit.png"
            )
            for beta in (0.005, 0.01, 0.02):
                Image.new("RGB", (32, 32), "gray").save(
                    self.root
                    / "leftImg8bit_foggy/val/frankfurt"
                    / f"{base}_leftImg8bit_foggy_beta_{beta}.png"
                )
            mask = np.zeros((32, 32), dtype=np.uint16)
            mask[5:17, 4:14] = 26001
            Image.fromarray(mask).save(
                self.root / "gtFine/val/frankfurt" / f"{base}_gtFine_instanceIds.png"
            )
        self.rows = load_and_validate_manifest(
            prepare_cityscapes_manifest(
                self.root, self.root / "city.json", with_detection_labels=True
            )
        )

    def source_run(self):
        source = self.root / "training"
        source.mkdir()
        config = training.default_training_configuration(self.root)
        config.update(
            output=str(self.root / "evaluation"),
            evaluation_source=str(source),
            size=32,
            device="cpu",
            amp=False,
            seeds=[42],
            evaluation_split="test",
            evaluation_max_pairs=1,
        )
        (source / "settings.json").write_text(json.dumps({"config": config}))
        (source / "input-manifest.json").write_text("[]")
        for model in config["checkpoints"]:
            for arm in config["arms"]:
                directory = source / f"{model}-{arm}-42"
                directory.mkdir()
                torch.save(
                    {
                        "model": FixtureDetector(model).model.state_dict(),
                        "adapter": FixtureAdapter().state_dict(),
                    },
                    directory / "best.pt",
                )
        directory = source / "density-42"
        directory.mkdir()
        torch.save({"model": FixtureDensity().state_dict()}, directory / "best.pt")
        return source, config

    def test_boxes_pairs_and_deterministic_scene_cap(self):
        self.assertEqual(len(self.rows), 8)
        self.assertEqual(self.rows[0]["boxes"], [[4, 5, 14, 17]])
        selected = training.validate_matched_evaluation(self.rows, [], "test", 1)
        reversed_selection = training.validate_matched_evaluation(
            self.rows[::-1], [], "test", 1
        )
        self.assertEqual(selected, reversed_selection)
        self.assertEqual(len(selected), 4)
        self.assertEqual({row["beta"] for row in selected}, {0, 0.005, 0.01, 0.02})

    def test_missing_variant_label_mismatch_and_leakage_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            training.validate_matched_evaluation(self.rows[1:], [], "test")
        bad = copy.deepcopy(self.rows)
        bad[0]["boxes"] = []
        with self.assertRaisesRegex(ValueError, "ground truth"):
            training.validate_matched_evaluation(bad, [], "test")
        bad = copy.deepcopy(self.rows)
        bad[0]["detection_labeled"] = False
        with self.assertRaisesRegex(ValueError, "gtFine"):
            training.validate_matched_evaluation(bad, [], "test")
        fitted = [{**self.rows[0], "split": "density_train"}]
        with self.assertRaisesRegex(ValueError, "overlap"):
            training.validate_matched_evaluation(self.rows, fitted, "test")
        for cap in (0, -1, True, "100"):
            with self.assertRaises(ValueError):
                training.validate_matched_evaluation(self.rows, [], "test", cap)

    def test_missing_clear_or_mask_never_becomes_detection_negative(self):
        clear = next((self.root / "leftImg8bit").rglob("*.png"))
        clear.unlink()
        with self.assertRaisesRegex(FileNotFoundError, "clear"):
            prepare_cityscapes_manifest(self.root, self.root / "bad.json", True)
        self.assertFalse((self.root / "bad.json").exists())

    def test_effect_direction_and_one_seed_has_no_interval(self):
        paired = pd.DataFrame(
            [
                dict(
                    dataset="foggy_cityscapes",
                    model="YOLO11n",
                    arm="predicted_density",
                    seed=seed,
                    density=density,
                    delta_mAP=effect,
                    images=4,
                )
                for seed in [42]
                for density, effect in [("0.0", -0.02), ("foggy", 0.05)]
            ]
        )
        analytics.export_hypothesis_tables(self.root, paired)
        effects = pd.read_csv(self.root / "fog-vs-clear.csv")
        self.assertAlmostEqual(effects.fog_advantage.iloc[0], 0.07)
        self.assertTrue(effects.supports_direction_in_this_seed.iloc[0])
        summary = pd.read_csv(self.root / "hypothesis-summary.csv")
        self.assertTrue(summary.delta_mAP_fog_ci_low.isna().all())
        repeated = pd.concat([paired.assign(seed=seed) for seed in (42, 43, 44)])
        analytics.export_hypothesis_tables(self.root, repeated)
        summary = pd.read_csv(self.root / "hypothesis-summary.csv")
        self.assertAlmostEqual(summary.delta_mAP_fog_ci_low.iloc[0], 0.05)

    def test_six_comparisons_eval_only_reuse_identity_and_source_lock(self):
        source, config = self.source_run()
        originals = {
            str(path): (path.read_bytes(), path.stat().st_mtime_ns)
            for path in source.rglob("*.pt")
        }
        with (
            patch.object(training, "ObjectDetector", FixtureDetector),
            patch.object(training, "FogAdapter", FixtureAdapter),
            patch.object(training, "FogDensityRegressor", FixtureDensity),
            patch.object(
                training,
                "fit_object_detector",
                side_effect=AssertionError("must not train"),
            ),
            patch.object(
                training,
                "fit_fog_density_model",
                side_effect=AssertionError("must not train"),
            ),
        ):
            results, status = training.run_checkpoint_evaluation(self.rows, config)
            self.assertEqual(len(status), 6)
            self.assertTrue(status.status.eq("complete").all())
            self.assertTrue(results[results.density == "all"].mAP50_95.gt(0.99).all())
            calls = FixtureDetector.calls
            training.run_checkpoint_evaluation(self.rows, config)
            self.assertEqual(calls, FixtureDetector.calls)
            with FileLock(str(source / "run.lock"), timeout=0):
                with self.assertRaises(Timeout):
                    training.run_checkpoint_evaluation(self.rows, config)
            changed = dict(config, repeats=2)
            with self.assertRaisesRegex(ValueError, "configuration"):
                training.run_checkpoint_evaluation(self.rows, changed)
        for path, (content, timestamp) in originals.items():
            self.assertEqual(Path(path).read_bytes(), content)
            self.assertEqual(Path(path).stat().st_mtime_ns, timestamp)
        output = Path(config["output"])
        for filename in (
            "fog-vs-clear.csv",
            "hypothesis-summary.csv",
            "dataset-audit.csv",
            "density-estimates.csv",
        ):
            self.assertTrue((output / filename).is_file(), filename)
        qa = ROOT / ".runtime/hypothesis-qa"
        qa.mkdir(parents=True, exist_ok=True)
        for path in (output / "figures").glob("*.png"):
            shutil.copy2(path, qa / path.name)

    def test_mismatched_size_and_missing_arm_reject_before_metadata(self):
        source, config = self.source_run()
        output = Path(config["output"])
        with self.assertRaisesRegex(ValueError, "image size"):
            training.run_checkpoint_evaluation(self.rows, dict(config, size=64))
        self.assertFalse((output / "settings.json").exists())
        next(source.glob("*-rgb_control-42/best.pt")).unlink()
        with self.assertRaisesRegex(FileNotFoundError, "trained arm"):
            training.run_checkpoint_evaluation(self.rows, config)
        self.assertFalse((output / "settings.json").exists())
        with self.assertRaisesRegex(ValueError, "separate"):
            training.run_checkpoint_evaluation(
                self.rows, dict(config, output=str(source))
            )

    def test_setup_cleanup_obeys_lock_and_preserves_current_input_audit(self):
        output = self.root / "setup-check-1"
        output.mkdir()
        audit = output / "subset-selection.json"
        audit.write_text('{"current": true}')
        old = output / "old.pt"
        old.write_bytes(b"old checkpoint")
        config = training.default_training_configuration(self.root)
        config.update(
            output=str(output),
            profile_name="setup",
            overwrite_existing=True,
            checkpoints={"YOLO11n": "unused"},
            arms=["rgb_control"],
            seeds=[42],
        )
        with (
            patch.object(
                training,
                "run_training_preflight",
                return_value=pd.DataFrame([{"ready": True}]),
            ),
            patch.object(training, "validate_and_record_run_identity"),
            patch.object(
                training, "prepare_density_estimator_for_seed", return_value=None
            ),
            patch.object(
                training, "run_detector_comparison", return_value={"status": "complete"}
            ),
            patch.object(
                analytics, "generate_study_analytics", return_value=pd.DataFrame()
            ),
        ):
            with FileLock(str(output / "run.lock"), timeout=0):
                with self.assertRaises(Timeout):
                    training.run_configured_detector_study([], config)
            self.assertEqual(old.read_bytes(), b"old checkpoint")
            training.run_configured_detector_study([], config)
            self.assertFalse(old.exists())
            self.assertEqual(audit.read_text(), '{"current": true}')

    def test_interrupted_evaluation_recovers_prediction_prefix(self):
        source, config = self.source_run()
        original_predict = FixtureDetector.predict_objects
        calls = []

        def interrupt_after_first_record(detector, image, row):
            calls.append(row["id"])
            if len(calls) == 5:
                raise KeyboardInterrupt("fixture interruption")
            return original_predict(detector, image, row)

        with (
            patch.object(training, "ObjectDetector", FixtureDetector),
            patch.object(training, "FogAdapter", FixtureAdapter),
            patch.object(training, "FogDensityRegressor", FixtureDensity),
        ):
            with patch.object(
                FixtureDetector, "predict_objects", interrupt_after_first_record
            ):
                with self.assertRaises(KeyboardInterrupt):
                    training.run_checkpoint_evaluation(self.rows, config)
            prediction_path = next(Path(config["output"]).glob("*/*-predictions.jsonl"))
            prefix = prediction_path.read_bytes()
            self.assertEqual(len(prefix.splitlines()), 1)
            results, status = training.run_checkpoint_evaluation(self.rows, config)
            self.assertTrue(prediction_path.read_bytes().startswith(prefix))
            self.assertEqual(len(status), 6)
            self.assertFalse(results.empty)

    def test_notebook_evaluation_executes_in_reading_order(self):
        source, config = self.source_run()
        notebook = nbformat.read(ROOT / "notebooks/fog_density_consolidated.ipynb", 4)
        namespace = {}
        original_directory = Path.cwd()
        original_paths = sys.path[:]
        try:
            for cell in notebook.cells:
                if cell.cell_type != "code":
                    continue
                code = cell.source
                if "subprocess.check_call" in code:
                    continue
                with (
                    contextlib.redirect_stdout(io.StringIO()),
                    contextlib.redirect_stderr(io.StringIO()),
                ):
                    exec(compile(code, "notebook fixture cell", "exec"), namespace)
                if 'SETTINGS["density_predictor"] =' in code:
                    namespace["SETTINGS"]["density_predictor"].update(
                        checkpoint=str(source / "density-42" / "best.pt"),
                        source_run="fixture",
                    )
                if cell.metadata.get("fogStudySettings"):
                    settings = namespace["SETTINGS"]
                    settings.update(
                        action="evaluate", runtime_root=str(self.root / "runtime")
                    )
                    settings["drive_mount"] = str(self.root / "drive")
                    settings["data"].update(
                        location={"kind": "path", "path": str(self.root)},
                        low_disk=False,
                    )
                    settings["output"]["location"] = {
                        "kind": "path",
                        "path": str(self.root),
                    }
                    settings["training"].update(
                        device="cpu", amp=False, seeds=[42], size=32
                    )
                    settings["profile"]["size"] = 32
                    settings["density_predictor"] = {
                        "checkpoint": str(source / "density-42" / "best.pt"),
                        "source_run": "fixture",
                        "seed": 42,
                    }
                    settings["evaluation"].update(
                        source_run=str(source),
                        cityscapes_root=str(self.root),
                        output_name="notebook-evaluation",
                        max_pairs=1,
                    )
                if (
                    "resolve_study_configuration(SETTINGS)" in code
                    or cell.metadata.get("fogStudySection") == "predictor"
                ):
                    namespace.update(
                        ObjectDetector=FixtureDetector,
                        FogAdapter=FixtureAdapter,
                        FogDensityRegressor=FixtureDensity,
                    )
            self.assertEqual(len(namespace["status"]), 6)
            self.assertTrue(Path(namespace["archive_path"]).is_file())
            self.assertEqual(
                Path(namespace["TRAINING_CONFIGURATION"]["output"]),
                self.root / "notebook-evaluation",
            )
        finally:
            os.chdir(original_directory)
            sys.path[:] = original_paths


if __name__ == "__main__":
    unittest.main()
