"""One entry point for preflight, preparation, full training, resume and saved-run analysis."""

import argparse
import json
import os
import sys
from functools import lru_cache
from pathlib import Path
from types import ModuleType

import nbformat

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".ultralytics"))
os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")


@lru_cache(maxsize=1)
def load_notebook_runtime():
    """Execute the notebook's definition cells, without running the study workflow."""
    path = ROOT / "notebooks/fog_density_study.ipynb"
    notebook = nbformat.read(path, as_version=4)
    runtime = ModuleType("fog_study_notebook")
    runtime.__file__ = str(path)
    sys.modules[runtime.__name__] = runtime
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == "code" and cell.metadata.get("fogStudyDefinitions"):
            exec(compile(cell.source, f"{path}:cell-{index}", "exec"), runtime.__dict__)
    return runtime


def load_rows(config):
    load_and_validate_manifest = load_notebook_runtime().load_and_validate_manifest
    rows = []
    for name in ("manifest", "city_manifest"):
        path = config.get(name)
        if path and Path(path).is_file():
            print(f"Validate {path}", flush=True)
            rows.extend(load_and_validate_manifest(path))
    return rows


def main():
    runtime = load_notebook_runtime()
    (
        default_training_configuration,
        run_training_preflight,
        run_configured_detector_study,
        get_fog_model_training_data,
        resolve_density_predictor_checkpoint,
        fit_fog_density_model,
    ) = (
        runtime.default_training_configuration,
        runtime.run_training_preflight,
        runtime.run_configured_detector_study,
        runtime.get_fog_model_training_data,
        runtime.resolve_density_predictor_checkpoint,
        runtime.fit_fog_density_model,
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=["preflight", "prepare", "download-models", "train", "analyze"],
    )
    parser.add_argument("--config", type=Path, help="JSON overrides for default_config")
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="Train all selected detectors without density labels",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--download-fog",
        action="store_true",
        help="Download the ~29.1 GB Kaggle source used in Untitled2.ipynb",
    )
    parser.add_argument(
        "--fog-root",
        type=Path,
        help="Existing Foggy Cityscapes or Kaggle cache directory",
    )
    args = parser.parse_args()
    config = default_training_configuration(ROOT)
    if args.config:
        config.update(json.loads(args.config.read_text(encoding="utf-8")))
    if args.baseline_only:
        config["arms"] = ["rgb_control"]
    if args.output:
        config["output"] = str(args.output.resolve())
    if args.command == "prepare":
        prepare_nuscenes_manifest = runtime.prepare_nuscenes_manifest
        prepare_nuscenes_manifest(
            ROOT / "data/depth/datasets/nuscenes", config["manifest"]
        )
        prepare_cityscapes_manifest, resolve_foggy_images = (
            runtime.prepare_cityscapes_manifest,
            runtime.resolve_foggy_images,
        )
        city = resolve_foggy_images(ROOT, args.fog_root, download=args.download_fog)
        if city is not None and not Path(config["city_manifest"]).exists():
            prepare_cityscapes_manifest(city, config["city_manifest"])
        return
    if args.command == "download-models":
        ObjectDetector = runtime.ObjectDetector
        for name, checkpoint in config["checkpoints"].items():
            print(f"Load pretrained {name}", flush=True)
            detector = ObjectDetector(name, checkpoint, "cpu")
            print(
                f"{name}: {sum(p.numel() for p in detector.model.parameters()):,} parameters",
                flush=True,
            )
            del detector
        return
    if args.command == "analyze":
        generate_study_analytics = runtime.generate_study_analytics
        results = generate_study_analytics(config["output"])
        print(results.to_string(index=False))
        return
    rows = load_rows(config)
    checks = run_training_preflight(config, rows)
    print(checks.to_string(index=False))
    if not checks.ready.all():
        raise SystemExit(2)
    if args.command == "train":
        checkpoint = resolve_density_predictor_checkpoint(config)
        fog_model = fit_fog_density_model(
            get_fog_model_training_data(rows), config, checkpoint, config["seeds"][0]
        )
        run_configured_detector_study(rows, config, fog_model)


if __name__ == "__main__":
    main()
