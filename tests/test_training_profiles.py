"""Validate independent profile selection and run identity contracts."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.run_fog_research import load_notebook_runtime  # noqa: E402

notebook = load_notebook_runtime()
PROFILE_SETTINGS = notebook.PROFILE_SETTINGS
add_balanced_bin_rows = notebook.add_balanced_bin_rows
density_bin, epoch_indices, epoch_size = (
    notebook.density_bin,
    notebook.epoch_indices,
    notebook.epoch_size,
)
select_profile_rows = notebook.select_profile_rows
validate_and_record_run_identity = notebook.validate_and_record_run_identity


def annotation(identifier, split, group, dataset="nuscenes"):
    return {
        "id": identifier,
        "image": identifier,
        "dataset": dataset,
        "split": split,
        "group": group,
        "boxes": [],
        "labels": [],
        "beta": None,
    }


def metric_row(dataset, density, density_bin_name, score, images):
    return {
        "model": "YOLO11n",
        "dataset": dataset,
        "arm": "rgb_control",
        "seed": 42,
        "density": density,
        "density_bin": density_bin_name,
        "distribution": "natural",
        "images": images,
        "mAP50_95": score,
        "AP50": score,
        "AR100": score,
        "precision_at_025": score,
        "recall_at_025": score,
        "mean_ms": 10.0,
        "p50_ms": 10.0,
        "p95_ms": 11.0,
        "serial_fps": 100.0,
        "density_mean_ms": 1.0,
    }


class TrainingProfiles(unittest.TestCase):
    def test_profiles_cover_all_models_and_have_independent_scopes(self):
        expected_models = ["YOLO11n", "YOLO11m", "DETR"]
        for name, profile in PROFILE_SETTINGS.items():
            self.assertEqual(profile["models"], expected_models)
            self.assertIn(profile["evaluation_split"], {"val", "test"})
            if name == "full":
                self.assertEqual(profile["evaluation_split"], "test")
            else:
                self.assertEqual(profile["evaluation_split"], "val")
            self.assertIn(profile["density_sampling"], {"natural", "balanced_bins"})
        self.assertTrue(PROFILE_SETTINGS["setup"]["overwrite_existing"])
        self.assertEqual(PROFILE_SETTINGS["setup"]["run_name"], "setup-check-1")
        self.assertFalse(PROFILE_SETTINGS["partial-2k"]["overwrite_existing"])
        self.assertFalse(PROFILE_SETTINGS["partial-5k"]["overwrite_existing"])
        self.assertFalse(PROFILE_SETTINGS["full"]["overwrite_existing"])
        self.assertNotRegex(PROFILE_SETTINGS["partial-2k"]["run_name"], r"-v\d+$")
        self.assertNotRegex(PROFILE_SETTINGS["partial-5k"]["run_name"], r"-v\d+$")

    def test_balanced_density_epoch_equalizes_known_bins_deterministically(self):
        rows = [
            {"beta": 0.0},
            {"beta": 0.005},
            {"beta": 0.005},
            {"beta": 0.01},
            {"beta": 0.02},
        ]
        first = epoch_indices(rows, 42, 0, balanced=True)
        second = epoch_indices(rows, 42, 0, balanced=True)
        self.assertEqual(first, second)
        labels = [density_bin(rows[index]["beta"]) for index in first]
        self.assertEqual(
            {label: labels.count(label) for label in set(labels)},
            {"clear": 2, "light": 2, "moderate": 2, "dense": 2},
        )
        self.assertEqual(epoch_size(rows, balanced=True), 8)

    def test_balanced_analysis_rows_do_not_pool_datasets(self):
        rows = pd.DataFrame(
            [
                metric_row("foggy_cityscapes", "0.005", "light", 0.2, 3),
                metric_row("foggy_cityscapes", "0.02", "dense", 0.4, 1),
                metric_row("nuscenes", "unknown", "unknown", 0.8, 100),
            ]
        )
        result = add_balanced_bin_rows(rows)
        balanced = result[result.density == "balanced"]
        self.assertEqual(len(balanced), 1)
        self.assertEqual(balanced.iloc[0].dataset, "foggy_cityscapes")
        self.assertAlmostEqual(balanced.iloc[0].mAP50_95, 0.3)

    def test_partial_selection_is_deterministic_scene_safe_and_excludes_final_test(
        self,
    ):
        rows = []
        for group in range(8):
            rows.extend(
                annotation(f"train-{group}-{frame}", "detector_train", f"scene-{group}")
                for frame in range(3)
            )
        rows.extend(
            annotation(f"val-{index}", "val", "tuning-scene") for index in range(4)
        )
        rows.append(annotation("test-0", "test", "test-scene"))
        first = select_profile_rows(
            rows, "partial-2k", 42, {"detector_train": 6, "density_train": 6, "val": 2}
        )
        second = select_profile_rows(
            rows, "partial-2k", 42, {"detector_train": 6, "density_train": 6, "val": 2}
        )
        self.assertEqual([row["id"] for row in first], [row["id"] for row in second])
        self.assertNotIn("test-0", {row["id"] for row in first})
        selected_groups = {
            row["group"] for row in first if row["split"] == "detector_train"
        }
        for group in selected_groups:
            self.assertEqual(sum(row["group"] == group for row in first), 3)

    def test_partial_profiles_share_fixed_tuning_ids(self):
        rows = [
            annotation(f"val-{index}", "val", f"tuning-{index // 2}")
            for index in range(8)
        ]
        limits = {"detector_train": 2, "density_train": 2, "val": 6}
        first = select_profile_rows(rows, "partial-2k", 42, limits)
        second = select_profile_rows(rows, "partial-5k", 42, limits)
        first_ids = [row["id"] for row in first if row["split"] == "val"]
        second_ids = [row["id"] for row in second if row["split"] == "val"]
        self.assertEqual(first_ids, second_ids)

    def test_changed_profile_identity_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / ".runtime") as temporary:
            directory = Path(temporary)
            base = {
                "profile_name": "partial-2k",
                "initialization": "pretrained",
                "output": str(directory),
                "epochs": 5,
                "density_epochs": 5,
                "seeds": [42],
                "arms": ["rgb_control", "predicted_density"],
                "size": 640,
                "batch_size": 1,
                "effective_batch_size": 4,
                "accumulate": 4,
                "loader": {
                    "num_workers": 0,
                    "prefetch_factor": None,
                    "pin_memory": False,
                },
                "save_every": 200,
                "repeats": 1,
                "learning_rates": {"YOLO11n": 1e-4},
                "adapter_lr": 1e-3,
                "amp": False,
                "device": "cpu",
                "density_checkpoint": None,
                "checkpoints": {"YOLO11n": "yolo11n.pt"},
                "evaluation_split": "val",
            }
            validate_and_record_run_identity([], base, directory)
            changed = {**base, "profile_name": "partial-5k"}
            with self.assertRaisesRegex(ValueError, "configuration"):
                validate_and_record_run_identity([], changed, directory)
            self.assertEqual(
                json.loads((directory / "settings.json").read_text())["profile"],
                "partial-2k",
            )

    def test_partial_profile_rejects_unidentified_existing_checkpoint(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / ".runtime") as temporary:
            directory = Path(temporary)
            (directory / "YOLO11n-rgb_control-42").mkdir()
            (directory / "YOLO11n-rgb_control-42" / "last.pt").write_bytes(
                b"checkpoint"
            )
            configuration = {
                "profile_name": "partial-2k",
                "initialization": "pretrained",
                "output": str(directory),
                "epochs": 5,
                "density_epochs": 5,
                "seeds": [42],
                "arms": ["rgb_control"],
                "size": 640,
                "batch_size": 1,
                "effective_batch_size": 4,
                "accumulate": 4,
                "loader": {
                    "num_workers": 0,
                    "prefetch_factor": None,
                    "pin_memory": False,
                },
                "save_every": 200,
                "repeats": 1,
                "learning_rates": {"YOLO11n": 1e-4},
                "adapter_lr": 1e-3,
                "amp": False,
                "device": "cpu",
                "density_checkpoint": None,
                "checkpoints": {"YOLO11n": "yolo11n.pt"},
                "evaluation_split": "val",
            }
            with self.assertRaisesRegex(ValueError, "profile identity"):
                validate_and_record_run_identity([], configuration, directory)


if __name__ == "__main__":
    unittest.main()
