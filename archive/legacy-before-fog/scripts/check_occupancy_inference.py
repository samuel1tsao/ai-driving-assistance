"""Check frame alignment, adapter guards, cache files and saved prediction contracts."""
import json
import sys
import tempfile
import subprocess
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks/_support"))
import occupancy
import occupancy_inference as inference

samples = inference.prepared_samples(ROOT, 100)
manifest = json.loads((ROOT / "data/occupancy/inputs/nuscenes/frames.json").read_text())
validation = occupancy.validation_scenes(ROOT)
assert len({s["id"] for s in samples}) == len(samples)
assert all(s["id"].split("/")[0] in validation for s in samples)
assert all(len(f["cams"]) == 6 for f in manifest["frames"])
assert all(
    (ROOT / c["data_path"]).is_file() for f in manifest["frames"] for c in f["cams"].values()
)
for limit in (-1, 0, 101):
    try:
        inference.prepared_samples(ROOT, limit)
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid sample count accepted")
for invalid in ([dict(samples[0], dataset="occ3d_waymo")], list(reversed(samples))):
    with patch("occupancy_inference.subprocess.run") as command:
        try:
            inference.run_model(ROOT, inference.MODELS[0], invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("Unsupported or misaligned frames accepted")
        command.assert_not_called()

for model in inference.MODELS:
    found = [occupancy.prediction_path(ROOT, model, s) for s in samples]
    present = [p for p in found if p.exists()]
    for path in present:
        with np.load(path, allow_pickle=False) as data:
            prediction = data["occupancy"]
            assert prediction.shape == (200, 200, 16)
            assert np.issubdtype(prediction.dtype, np.integer)
            assert prediction.min() >= 0 and prediction.max() < 18
            assert np.isfinite(data["inference_ms"]) and data["inference_ms"] > 0
            assert len(str(data["fingerprint"])) == 64
    print(model["name"], len(present), "valid prediction files")

with tempfile.TemporaryDirectory(dir=ROOT / ".runtime") as temporary:
    folder = Path(temporary)
    target = folder / "frames.json"
    first = folder / "frames.partial"
    first.write_text("verified")
    first.replace(target)
    assert target.read_text() == "verified" and not first.exists()
    lock = folder / ".running"
    lock.open("x").close()
    try:
        lock.open("x")
    except FileExistsError:
        pass
    else:
        raise AssertionError("Concurrent lock acquisition succeeded")
    (folder / 'data/occupancy').mkdir(parents=True)
    notebook_lock = folder / 'data/occupancy/.notebook-running'
    with patch('occupancy_inference.prepared_samples', return_value=samples):
        with patch('occupancy_inference.subprocess.run', side_effect=[
            SimpleNamespace(returncode=0), subprocess.TimeoutExpired('test', 3600)
        ]):
            try:
                inference.run_model(folder, inference.MODELS[0], samples)
            except RuntimeError as error:
                assert 'lock is retained' in str(error)
            else:
                raise AssertionError('Timeout was not reported')
        assert notebook_lock.exists()
        with patch('occupancy_inference.subprocess.run', return_value=SimpleNamespace(returncode=0)):
            try:
                inference.run_model(folder, inference.MODELS[0], samples)
            except FileExistsError:
                pass
            else:
                raise AssertionError('A second benchmark bypassed the retained lock')
print("Frame joins, validation split, wrong-dataset/order guards and file contracts passed.")
