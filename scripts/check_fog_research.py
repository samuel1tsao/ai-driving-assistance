"""Focused offline checks; synthetic fixtures validate code, never research claims."""

import copy
import json
import tempfile
from pathlib import Path

import nbformat
import numpy as np
import torch
from PIL import Image

from run_fog_research import load_notebook_runtime

ROOT = Path(__file__).resolve().parents[1]

notebook = load_notebook_runtime()
(
    load_manifest,
    calculate_coco_detection_metrics,
    FogAdapter,
    get_density_feature,
) = (
    notebook.load_manifest,
    notebook.calculate_coco_detection_metrics,
    notebook.FogAdapter,
    notebook.get_density_feature,
)
prepare_cityscapes_manifest, resolve_foggy_images = (
    notebook.prepare_cityscapes_manifest,
    notebook.resolve_foggy_images,
)
create_training_data_partitions, clipped_bounds = (
    notebook.create_training_data_partitions,
    notebook.clipped_bounds,
)


def main():
    with tempfile.TemporaryDirectory() as directory:
        folder = Path(directory)
        assert clipped_bounds(np.array([[-2, 0.5], [0.5, -2], [-2, -2]]), 1, 1) is None
        fog_root = folder / "nested" / "leftImg8bit_foggy"
        for split, city in [("train", "aachen"), ("val", "frankfurt")]:
            destination = fog_root / split / city
            destination.mkdir(parents=True)
            for beta in (0.005, 0.01, 0.02):
                Image.new("RGB", (32, 32)).save(
                    destination
                    / f"{city}_000000_000001_leftImg8bit_foggy_beta_{beta}.png"
                )
        assert (
            resolve_foggy_images(folder, explicit=folder / "nested")
            == folder / "nested"
        )
        city_rows = load_manifest(
            prepare_cityscapes_manifest(folder, folder / "city.json")
        )
        assert len(city_rows) == 6 and all(
            not r["detection_labeled"] for r in city_rows
        )
        assert (
            not create_training_data_partitions(city_rows)["test"]
            and len(create_training_data_partitions(city_rows)["density_test"]) == 3
        )
        Image.new("RGB", (32, 32)).save(folder / "image.png")
        Image.new("RGB", (32, 32), color="white").save(folder / "other.png")
        row = {
            "id": "a",
            "image": "image.png",
            "dataset": "foggy_cityscapes",
            "group": "scene",
            "split": "test",
            "beta": 0.01,
            "boxes": [[2, 2, 15, 15]],
            "labels": [1],
        }

        def load(rows):
            path = folder / "manifest.json"
            path.write_text(json.dumps(rows))
            return load_manifest(path)

        valid = load([row])
        for field, value in [
            ("beta", float("nan")),
            ("labels", [100]),
            ("boxes", [[15, 2, 2, 15]]),
            ("iscrowd", []),
            ("image", "missing.png"),
        ]:
            bad = {**row, field: value}
            try:
                load([bad])
            except (ValueError, FileNotFoundError):
                pass
            else:
                raise AssertionError(f"Accepted invalid {field}")
        for bad_rows in (
            [row, row],
            [row, {**row, "id": "b", "image": "other.png", "split": "detector_train"}],
        ):
            try:
                load(bad_rows)
            except ValueError:
                pass
            else:
                raise AssertionError("Accepted duplication or split leakage")
        exact = [
            {
                "id": "a",
                "prediction": [{"box": [2, 2, 15, 15], "label": 1, "score": 0.9}],
            }
        ]
        assert (
            abs(calculate_coco_detection_metrics(valid, exact)["mAP50_95"] - 1) < 1e-6
        )
        assert (
            calculate_coco_detection_metrics(valid, [{"id": "a", "prediction": []}])[
                "mAP50_95"
            ]
            == 0
        )
        wrong = copy.deepcopy(exact)
        wrong[0]["prediction"][0]["label"] = 2
        assert calculate_coco_detection_metrics(valid, wrong)["mAP50_95"] == 0
        x = torch.rand(1, 3, 32, 32)
        adapter = FogAdapter()
        assert torch.equal(adapter(x, torch.tensor([[0.01]])), x)
        with torch.no_grad():
            adapter.output.weight.fill_(0.1)
        assert not torch.equal(
            adapter(x, torch.tensor([[0.005]])), adapter(x, torch.tensor([[0.02]]))
        )
        assert get_density_feature("rgb_control", x, row, None).item() == 0
        try:
            get_density_feature("oracle_density", x, {**row, "beta": None}, None)
        except ValueError:
            pass
        else:
            raise AssertionError("Oracle silently used unknown beta")
    notebook = nbformat.read(ROOT / "notebooks/fog_density_study.ipynb", as_version=4)
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            compile(cell.source, "<notebook>", "exec")
    plan = (ROOT / "docs/fog-density-research-plan.md").read_text(encoding="utf-8")
    assert len(plan.split("\n", 1)[1].strip()) <= 2000
    print(
        "Passed: density-only Kaggle layout, convex-hull projection, input guards, split leakage, exact/empty/wrong AP, scalar conditioning, oracle guard, notebook schema/code, plan length."
    )


if __name__ == "__main__":
    main()
