"""Bounded real-detector integration tests; all outputs are TEST FIXTURES, not study results.

Uses real nuScenes RGB/boxes but arbitrary beta targets solely to exercise regression.
Tests full pretrained YOLO11n/YOLO11m/DETR training, both arms, resume and analytics.
"""

import argparse
import copy
import json
import os
import tempfile
from pathlib import Path

from run_fog_research import load_notebook_runtime

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".ultralytics"))
os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")


def main():
    import ijson
    import torch

    notebook = load_notebook_runtime()
    (
        default_training_configuration,
        run_configured_detector_study,
        validate_training_inputs,
        create_training_data_partitions,
        get_fog_model_training_data,
        resolve_density_predictor_checkpoint,
        fit_fog_density_model,
    ) = (
        notebook.default_training_configuration,
        notebook.run_configured_detector_study,
        notebook.validate_training_inputs,
        notebook.create_training_data_partitions,
        notebook.get_fog_model_training_data,
        notebook.resolve_density_predictor_checkpoint,
        notebook.fit_fog_density_model,
    )
    ObjectDetector, load_manifest = (
        notebook.ObjectDetector,
        notebook.load_manifest,
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args()
    selected, seen = [], set()
    with (ROOT / "data/fog/nuscenes_all.json").open("rb") as handle:
        for row in ijson.items(handle, "item", use_float=True):
            if row["group"] not in seen and len(row["boxes"]) >= 2:
                selected.append(row)
                seen.add(row["group"])
                if len(selected) == 8:
                    break
    assert len(selected) == 8
    # Fixture partitions are independently grouped. Density targets below are
    # intentionally artificial and never exported as measurements of real fog.
    splits = [
        "density_train",
        "density_train",
        "detector_train",
        "detector_train",
        "val",
        "test",
        "val",
        "test",
    ]
    fixture = []
    for index, (row, split) in enumerate(zip(selected, splits)):
        row = copy.deepcopy(row)
        row["split"] = split
        if index in (0, 1, 6, 7):
            row["beta"] = [0.005, 0.02][index % 2]
            row["detection_labeled"] = False
        fixture.append(row)
    with tempfile.TemporaryDirectory(
        prefix="fog-training-check-", dir=ROOT / "reports/fog"
    ) as folder:
        folder = Path(folder)
        manifest = folder / "fixture.json"
        manifest.write_text(json.dumps(fixture), encoding="utf-8")
        (folder / "fixture.coverage.json").write_text(
            json.dumps(
                {
                    "expected_images": len(fixture),
                    "available_images": len(fixture),
                    "missing_images": [],
                    "missing_channels": [],
                    "complete": True,
                }
            ),
            encoding="utf-8",
        )
        rows = load_manifest(manifest)
        config = default_training_configuration(ROOT)
        config.update(
            output=str(folder / "run"),
            manifest=str(manifest),
            city_manifest="",
            seeds=[7],
            epochs=1,
            density_epochs=1,
            size=args.size,
            accumulate=2,
            save_every=1,
            amp=args.amp,
        )
        validate_training_inputs(config, rows)
        assert len(create_training_data_partitions(rows)["detector_train"]) == 2
        assert (
            len(create_training_data_partitions(rows)["test"]) == 1
        )  # unlabeled density row cannot become a negative detection image
        fog_model = fit_fog_density_model(
            get_fog_model_training_data(rows),
            config,
            resolve_density_predictor_checkpoint(config),
            config["seeds"][0],
        )
        results, status = run_configured_detector_study(rows, config, fog_model)
        assert len(status) == 6 and (status.status == "complete").all()
        for name in config["checkpoints"]:
            model = ObjectDetector(name, config["checkpoints"][name], "cpu")
            original = model.model.state_dict()
            for arm in config["arms"]:
                directory = folder / "run" / f"{name}-{arm}-7"
                saved = torch.load(
                    directory / "last.pt", map_location="cpu", weights_only=True
                )
                changed = any(
                    not torch.equal(saved["model"][key], value)
                    for key, value in original.items()
                    if value.is_floating_point()
                )
                assert changed, f"{name}/{arm}: detector weights did not train"
                assert saved["epoch"] == 1
                assert (directory / "nuscenes-predictions.jsonl").is_file()
                exported = torch.load(
                    directory / "trained-model.pt",
                    map_location="cpu",
                    weights_only=True,
                )
                assert exported["kind"] == "fog-detector-export-v1"
            del model
        mtimes = {
            p: str(p.stat().st_mtime_ns) for p in (folder / "run").glob("*/last.pt")
        }
        run_configured_detector_study(rows, config, fog_model)
        assert all(str(p.stat().st_mtime_ns) == old for p, old in mtimes.items()), (
            "Completed resume retrained a model"
        )
        altered = copy.deepcopy(config)
        altered["epochs"] = 2
        try:
            run_configured_detector_study(rows, altered, fog_model)
        except ValueError as error:
            assert "changed" in str(error)
        else:
            raise AssertionError("Stale configuration accepted")
        # Save only verification artifacts, explicitly named as such.
        import shutil

        qa = ROOT / "reports/fog/verification"
        qa.mkdir(exist_ok=True)
        for path in (folder / "run" / "figures").glob("*.png"):
            shutil.copy2(path, qa / f"TEST-ONLY-{path.name}")
        (qa / "checks.json").write_text(
            json.dumps(
                {
                    "status": "passed",
                    "test_only": True,
                    "input_size": args.size,
                    "amp": args.amp,
                    "models": list(config["checkpoints"]),
                    "arms": config["arms"],
                    "checks": [
                        "density-only isolation",
                        "full detector updates",
                        "six train/eval pipelines",
                        "completed-run resume",
                        "configuration mismatch rejection",
                        "saved metrics and charts",
                    ],
                    "warning": "Tiny fixture; beta targets were arbitrary testing values. No research conclusions.",
                },
                indent=2,
            )
        )
    print(
        "PASS: six pretrained model/arm pipelines trained, evaluated, exported analytics and resumed. TEST ONLY."
    )


if __name__ == "__main__":
    main()
