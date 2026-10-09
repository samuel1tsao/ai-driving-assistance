"""Run official FlashOCC on prepared camera frames inside the Linux GPU container."""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from mmcv import Config
from mmcv.runner import load_checkpoint

ROOT = Path("/workspace")
REPO = ROOT / "notebooks/_runtime/FlashOCC"
sys.path.insert(0, str(REPO))
os.chdir(str(REPO))
import projects.mmdet3d_plugin
from mmdet3d.models import build_model
from projects.mmdet3d_plugin.datasets.pipelines.loading import PrepareImageInputs

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--variant", choices=["m0", "m1"], default="m1")
parser.add_argument("--limit", type=int, default=100)
parser.add_argument("--refresh", action="store_true")
args = parser.parse_args()
weights = ROOT / "data/models/occupancy" / ("flashocc-" + args.variant + ".pth")
if args.variant == "m1":
    weights = ROOT / "data/models/occupancy/flashocc-r50-256x704.pth"
manifest = ROOT / "data/occupancy/inputs/nuscenes/frames.json"
frames = json.loads(manifest.read_text())["frames"][: args.limit]
assert frames, "No prepared frames."
assert weights.is_file(), "Official pretrained checkpoint is missing."
assert torch.cuda.is_available(), "A CUDA GPU is required."
fingerprint = hashlib.sha256(weights.read_bytes() + manifest.read_bytes()).hexdigest()
config_name = "flashocc-r50-M0.py" if args.variant == "m0" else "flashocc-r50.py"
config = Config.fromfile(str(REPO / "projects/configs/flashocc" / config_name))
config.model.img_backbone.pretrained = None
model = build_model(config.model, test_cfg=config.get("test_cfg"))
load_checkpoint(model, str(weights), map_location="cpu", strict=True)
model.cuda().eval()
prepare = PrepareImageInputs(data_config=config.data_config, is_train=False, sequential=False)
output_root = ROOT / "data/occupancy/predictions/occ3d_nuscenes" / ("flashocc_" + args.variant)
output_root.mkdir(parents=True, exist_ok=True)
lock = output_root / ".running"
lock_handle = lock.open("x")
lock_handle.close()
try:
    warmed = False
    with torch.no_grad():
        for index, frame in enumerate(frames):
            output = output_root / (frame["id"] + ".npz")
            if output.exists() and not args.refresh:
                with np.load(str(output), allow_pickle=False) as cached:
                    if str(cached["fingerprint"]) == fingerprint:
                        print("Cached {}/{}".format(index + 1, len(frames)), flush=True)
                        continue
            for camera in frame["cams"].values():
                camera["data_path"] = str(ROOT / camera["data_path"])
            inputs = prepare.get_inputs({"curr": frame}) + (torch.eye(3),)
            inputs = [tensor.unsqueeze(0).cuda() for tensor in inputs]
            if not warmed:
                for _ in range(3):
                    model.simple_test(None, [{}], img=inputs)
                torch.cuda.synchronize()
                warmed = True
            torch.cuda.synchronize()
            start = time.perf_counter()
            prediction = model.simple_test(None, [{}], img=inputs)[0]
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - start) * 1000
            if torch.is_tensor(prediction):
                prediction = prediction.cpu().numpy()
            assert prediction.shape == (200, 200, 16), "Unexpected output grid."
            assert prediction.min() >= 0 and prediction.max() < 18, "Wrong class taxonomy."
            output.parent.mkdir(parents=True, exist_ok=True)
            temporary = output.with_suffix(".partial")
            with temporary.open("wb") as dest:
                np.savez_compressed(
                    dest,
                    occupancy=prediction.astype(np.uint8),
                    inference_ms=elapsed,
                    fingerprint=fingerprint,
                    gpu=torch.cuda.get_device_name(),
                    variant=args.variant,
                )
            temporary.replace(output)
            print(
                "{} {}/{}: {:.1f} ms".format(args.variant, index + 1, len(frames), elapsed),
                flush=True,
            )
finally:
    lock.unlink()
