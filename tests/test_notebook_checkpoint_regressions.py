"""Exercise notebook checkpoint contracts without downloading models or using a GPU."""

import importlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_RUNTIME = PROJECT_ROOT / ".runtime" / "checkpoint-regressions"
SUPPORT_DIRECTORY = PROJECT_ROOT / "notebooks" / "_support"


class SmallDetector:
    name = "YOLO11n"

    def __init__(self):
        self.model = nn.Sequential(nn.Linear(1, 1), nn.Dropout(0.2))

    def training_mode(self):
        self.model.train()

    def loss(self, image, annotation):
        expected = image.new_tensor([[annotation["target"]]])
        return torch.nn.functional.mse_loss(self.model(image), expected)


class SmallAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, image, density):
        return image * self.scale


class CheckpointRegressions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        TEST_RUNTIME.mkdir(parents=True, exist_ok=True)
        sys.path.insert(0, str(SUPPORT_DIRECTORY))
        cls.training = importlib.import_module("fog_training")

    @classmethod
    def tearDownClass(cls):
        sys.path.remove(str(SUPPORT_DIRECTORY))

    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory(dir=TEST_RUNTIME)
        self.addCleanup(self.temporary_directory.cleanup)
        self.training_rows = [
            {"id": str(index), "value": index / 10 + 0.1, "target": index / 20}
            for index in range(5)
        ]
        self.partitions = {
            "detector_train": self.training_rows,
            "val": self.training_rows[:2],
        }
        self.configuration = {
            "device": "cpu",
            "amp": False,
            "epochs": 2,
            "size": 32,
            "accumulate": 2,
            "save_every": 1,
            "learning_rates": {"YOLO11n": 0.01},
            "adapter_lr": 0.01,
        }
        self.addCleanup(patch.stopall)
        patch.object(self.training, "tqdm", lambda values, **kwargs: values).start()
        patch.object(self.training, "beta_for", lambda *args: torch.zeros(1, 1)).start()

    def make_models(self):
        torch.manual_seed(42)
        return SmallDetector(), SmallAdapter()

    def train(self, directory, interrupt_at=None):
        directory.mkdir(parents=True, exist_ok=True)
        detector, adapter = self.make_models()
        image_call_count = 0

        def load_test_image(annotation, size, device):
            nonlocal image_call_count
            image_call_count += 1
            if image_call_count == interrupt_at:
                raise KeyboardInterrupt("Simulated runtime interruption")
            return torch.tensor([[annotation["value"]]])

        with patch.object(self.training, "image_tensor", load_test_image):
            self.training.fit_detector(
                detector,
                adapter,
                None,
                self.partitions,
                "rgb_control",
                self.configuration,
                directory,
                42,
            )
        return torch.load(directory / "last.pt", weights_only=True)

    def assert_saved_states_equal(self, expected, actual):
        if isinstance(expected, torch.Tensor):
            torch.testing.assert_close(expected, actual, rtol=0, atol=0)
        elif isinstance(expected, dict):
            self.assertEqual(expected.keys(), actual.keys())
            for name in expected:
                self.assert_saved_states_equal(expected[name], actual[name])
        elif isinstance(expected, list):
            self.assertEqual(len(expected), len(actual))
            for expected_item, actual_item in zip(expected, actual):
                self.assert_saved_states_equal(expected_item, actual_item)
        else:
            self.assertEqual(expected, actual)

    def test_interrupt_resume_matches_uninterrupted_weights_optimizer_rng_and_history(
        self,
    ):
        uninterrupted = self.train(
            Path(self.temporary_directory.name) / "uninterrupted"
        )
        resumed_directory = Path(self.temporary_directory.name) / "resumed"
        with self.assertRaises(KeyboardInterrupt):
            self.train(resumed_directory, interrupt_at=4)
        interrupted = torch.load(resumed_directory / "last.pt", weights_only=True)
        self.assertEqual(interrupted["step"], 2)
        self.assertEqual(interrupted["running_count"], 2)
        self.assertEqual(interrupted["checkpoint_reason"], "interrupt")
        resumed = self.train(resumed_directory)
        self.assert_saved_states_equal(uninterrupted, resumed)

    def test_periodic_checkpoint_resumes_after_uncatchable_runtime_loss(self):
        uninterrupted = self.train(
            Path(self.temporary_directory.name) / "uninterrupted"
        )
        crash_directory = Path(self.temporary_directory.name) / "crash"
        original_save = self.training.save_detector_training_checkpoint

        def simulate_process_loss(session, improved=False, reason="periodic"):
            original_save(session, improved, reason)
            if reason == "periodic":
                raise RuntimeError("Simulated process loss after persisted checkpoint")

        with patch.object(
            self.training, "save_detector_training_checkpoint", simulate_process_loss
        ):
            with self.assertRaisesRegex(RuntimeError, "process loss"):
                self.train(crash_directory)
        resumed = self.train(crash_directory)
        self.assert_saved_states_equal(uninterrupted, resumed)

    def test_existing_checkpoint_schema_and_completed_resume_are_preserved(self):
        directory = Path(self.temporary_directory.name) / "completed"
        saved = self.train(directory)
        self.assertEqual(
            set(saved),
            {
                "kind",
                "model",
                "adapter",
                "optimizer",
                "scaler",
                "epoch",
                "step",
                "best_loss",
                "history",
                "running_sum",
                "running_count",
                "rng",
                "cuda_rng",
                "checkpoint_reason",
            },
        )
        timestamp = (directory / "last.pt").stat().st_mtime_ns
        with patch.object(
            self.training,
            "train_one_detector_epoch",
            side_effect=AssertionError("Must not retrain"),
        ):
            reused = self.train(directory)
        self.assertEqual(timestamp, (directory / "last.pt").stat().st_mtime_ns)
        self.assert_saved_states_equal(saved, reused)

    def test_incompatible_configuration_is_rejected_without_replacing_checkpoint(self):
        directory = Path(self.temporary_directory.name) / "identity"
        directory.mkdir()
        signature = {
            "engine": 4,
            "config": {"size": 320},
            "inputs": "dataset",
            "density_sha256": None,
            "versions": {},
            "source_sha256": {},
        }
        (directory / "settings.json").write_text(json.dumps(signature))
        checkpoint = directory / "last.pt"
        checkpoint.write_bytes(b"existing checkpoint")
        changed = {**signature, "config": {"size": 640}}
        with patch.object(self.training, "build_run_signature", return_value=changed):
            with self.assertRaisesRegex(ValueError, "configuration"):
                self.training.validate_and_record_run_identity([], {}, directory)
        self.assertEqual(checkpoint.read_bytes(), b"existing checkpoint")
        self.assertEqual(
            json.loads((directory / "settings.json").read_text()), signature
        )

    def test_training_matches_original_reference_results(self):
        expected = json.loads(
            (
                PROJECT_ROOT / "tests/fixtures/pre-refactor-detector-checkpoint.json"
            ).read_text(encoding="utf-8")
        )
        actual = self.train(Path(self.temporary_directory.name) / "reference")

        def compare(reference, result):
            if isinstance(result, torch.Tensor):
                torch.testing.assert_close(
                    torch.tensor(reference, dtype=result.dtype),
                    result,
                    rtol=1e-6,
                    atol=1e-7,
                )
            elif isinstance(reference, dict):
                for key, value in result.items():
                    compare(reference[str(key)], value)
            elif isinstance(reference, list):
                self.assertEqual(len(reference), len(result))
                for left, right in zip(reference, result):
                    compare(left, right)
            elif isinstance(reference, float):
                self.assertAlmostEqual(reference, result, places=6)
            else:
                self.assertEqual(reference, result)

        for key, value in expected.items():
            compare(value, actual[key])

    def test_partial_prediction_record_is_removed_without_repeating_inference(self):
        predictions = Path(self.temporary_directory.name) / "predictions.jsonl"
        record = {"id": "0", "prediction": []}
        complete = json.dumps(record).encode() + b"\n"
        predictions.write_bytes(complete + b'{"id":')
        with patch.object(
            self.training,
            "predict_one_image_with_timing",
            side_effect=AssertionError("Must reuse prediction"),
        ):
            result = self.training.evaluate_resumable(
                SmallDetector(),
                None,
                None,
                "rgb_control",
                self.training_rows[:1],
                self.configuration,
                predictions,
            )
        self.assertEqual(result, [record])
        self.assertEqual(predictions.read_bytes(), complete)

    def test_prediction_order_mismatch_is_rejected(self):
        predictions = Path(self.temporary_directory.name) / "predictions.jsonl"
        predictions.write_text(json.dumps({"id": "wrong"}) + "\n")
        with self.assertRaisesRegex(ValueError, "input order"):
            self.training.read_completed_prediction_records(
                predictions, self.training_rows
            )

    def test_failed_atomic_save_preserves_existing_checkpoint(self):
        checkpoint = Path(self.temporary_directory.name) / "last.pt"
        checkpoint.write_bytes(b"previous completed checkpoint")
        with patch.object(
            self.training.torch, "save", side_effect=OSError("disk full")
        ):
            with self.assertRaisesRegex(OSError, "disk full"):
                self.training.save_torch(checkpoint, {"model": {}})
        self.assertEqual(checkpoint.read_bytes(), b"previous completed checkpoint")

    def test_density_checkpoint_resumes_after_process_loss(self):
        class SmallDensityEstimator(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = nn.Linear(1, 1)

            def forward(self, image):
                return self.layer(image)

        partitions = {
            "density_train": [
                {
                    **row,
                    "dataset": "foggy_cityscapes",
                    "group": str(index),
                    "beta": 0.01 + index * 0.001,
                }
                for index, row in enumerate(self.training_rows)
            ],
            "density_val": [{**row, "beta": 0.01} for row in self.training_rows[:2]],
        }
        configuration = {**self.configuration, "density_epochs": 2}

        def train_density(directory):
            with (
                patch.object(self.training, "DensityCNN", SmallDensityEstimator),
                patch.object(
                    self.training,
                    "image_tensor",
                    lambda row, *args: torch.tensor([[row["value"]]]),
                ),
            ):
                self.training.fit_density(partitions, configuration, directory, 42)
            return torch.load(directory / "last.pt", weights_only=True)

        expected = train_density(
            Path(self.temporary_directory.name) / "density-complete"
        )
        interrupted = Path(self.temporary_directory.name) / "density-interrupted"
        original_save = self.training.save_density_training_checkpoint

        def save_then_crash(session, improved=False):
            original_save(session, improved)
            raise RuntimeError("process loss")

        with patch.object(
            self.training, "save_density_training_checkpoint", save_then_crash
        ):
            with self.assertRaisesRegex(RuntimeError, "process loss"):
                train_density(interrupted)
        self.assert_saved_states_equal(expected, train_density(interrupted))

    def test_second_writer_is_rejected_before_run_metadata_changes(self):
        from filelock import FileLock, Timeout
        import pandas as pd

        output = Path(self.temporary_directory.name) / "locked-run"
        output.mkdir()
        with (
            FileLock(str(output / "run.lock")),
            patch.object(
                self.training, "preflight", return_value=pd.DataFrame({"ready": [True]})
            ),
            patch.object(
                self.training,
                "validate_and_record_run_identity",
                side_effect=AssertionError("Must not write while locked"),
            ),
        ):
            with self.assertRaises(Timeout):
                self.training.run_training(
                    [], {**self.configuration, "output": str(output)}
                )

    def test_overall_progress_counts_prior_epochs_and_current_saved_steps(self):
        progress = self.training.DetectorTrainingProgress(epoch=3, step=2)
        result = self.training.summarize_cumulative_training_progress(progress, 10, 5)
        self.assertEqual(result["completed_image_steps"], 17)
        self.assertEqual(result["overall_training_percent"], 34.0)
        progress.epoch, progress.step = 10, 0
        self.assertEqual(
            self.training.summarize_cumulative_training_progress(progress, 10, 5)[
                "overall_training_percent"
            ],
            100.0,
        )


if __name__ == "__main__":
    unittest.main()
