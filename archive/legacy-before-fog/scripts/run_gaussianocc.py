"""Evaluate GaussianOcc occupancy; omit image rendering, which does not affect occupancy logits."""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pyquaternion import Quaternion
from torchvision.transforms.functional import to_tensor

ROOT = Path("/workspace")
REPO = ROOT / "notebooks/_runtime/GaussianOcc"
sys.path.insert(0, str(REPO))
os.chdir(str(REPO))
from options import MonodepthOptions
from networks import Encoder_res101, VolumeDecoder
from utils import geom, basic


class OccupancyOnly(VolumeDecoder):
    def get_density(self, *args, **kwargs):
        return None, None, None, None

    def feature2vox_simple(self, features, intrinsics, poses, pack, unpack):
        height, width = features.shape[-2:]
        scaled = geom.scale_intrinsics(intrinsics, width / self.opt.width, height / self.opt.height)
        inverse = geom.safe_inverse(poses)
        total = count = None
        for index in range(len(features)):
            projection = scaled[index : index + 1] @ inverse[index : index + 1]
            volume = self.vox_util.unproject_image_to_mem(
                features[index : index + 1],
                projection,
                inverse[index : index + 1],
                self.Z,
                self.Y,
                self.X,
            )
            mask = (volume.abs() > 0).float()
            if total is None:
                total, count = volume * mask, mask
            else:
                total.add_(volume * mask)
                count.add_(mask)
        return (total / (basic.EPS + count)).permute(0, 1, 4, 3, 2)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--limit", type=int, default=100)
parser.add_argument("--refresh", action="store_true")
args = parser.parse_args()
sys.argv = [sys.argv[0], "--config", str(REPO / "configs/nusc-sem-gs.txt"), "--eval_only"]
opt = MonodepthOptions().parse()
opt.B = 1
opt.semantic_sample_ratio = 1.0
manifest = ROOT / "data/occupancy/inputs/nuscenes/frames.json"
frames = json.loads(manifest.read_text())["frames"][: args.limit]
weights = ROOT / "data/models/occupancy/gaussianocc"
fingerprint = hashlib.sha256(
    manifest.read_bytes()
    + (weights / "encoder.pth").read_bytes()
    + (weights / "depth.pth").read_bytes()
).hexdigest()
models = {
    "encoder": Encoder_res101(opt.input_channel, network_type=opt.encoder),
    "depth": OccupancyOnly(opt),
}
for name, model in models.items():
    state = torch.load(str(weights / (name + ".pth")), map_location="cpu")
    expected = model.state_dict()
    state = {k: v for k, v in state.items() if k in expected}
    model.load_state_dict(state, strict=True)
    model.cuda().eval()
order = [
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_BACK_LEFT",
    "CAM_BACK",
    "CAM_BACK_RIGHT",
    "CAM_FRONT_RIGHT",
]
output_root = ROOT / "data/occupancy/predictions/occ3d_nuscenes/gaussianocc"
output_root.mkdir(parents=True, exist_ok=True)
lock = output_root / ".running"
lock.open("x").close()
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
            images, intrinsics, poses = [], [], []
            for name in order:
                camera = frame["cams"][name]
                with Image.open(str(ROOT / camera["data_path"])) as image:
                    width, height = image.size
                    images.append(
                        to_tensor(
                            image.convert("RGB").resize((opt.width, opt.height), Image.LANCZOS)
                        )
                    )
                intrinsic = np.eye(4, dtype=np.float32)
                intrinsic[:3, :3] = camera["cam_intrinsic"]
                intrinsic[0, :] *= opt.width / width
                intrinsic[1, :] *= opt.height / height
                pose = np.eye(4, dtype=np.float32)
                pose[:3, :3] = Quaternion(camera["sensor2ego_rotation"]).rotation_matrix
                pose[:3, 3] = camera["sensor2ego_translation"]
                intrinsics.append(intrinsic)
                poses.append(pose)
            color = torch.stack(images).cuda()
            data = {
                "pose_spatial": torch.tensor(np.stack(poses)).cuda(),
                ("K", 0, 0): torch.tensor(np.stack(intrinsics)).cuda(),
            }
            if not warmed:
                print("Checking streamed aggregation against the official method.", flush=True)
                decoder = models["depth"]
                features = models["encoder"](color)
                original_shape = decoder.Z, decoder.Y, decoder.X
                decoder.Z, decoder.Y, decoder.X = 4, 8, 8
                pack = lambda x: basic.pack_seqdim(x, 1)
                unpack = lambda x: basic.unpack_seqdim(x, 1)
                official = VolumeDecoder.feature2vox_simple(
                    decoder, features[0], data["K", 0, 0], data["pose_spatial"], pack, unpack
                )
                streamed = decoder.feature2vox_simple(
                    features[0], data["K", 0, 0], data["pose_spatial"], pack, unpack
                )
                assert (
                    torch.count_nonzero(official) > 0
                ), "Aggregation check has no visible features."
                assert torch.allclose(official, streamed, atol=1e-5, rtol=1e-5)
                decoder.Z, decoder.Y, decoder.X = original_shape
                for _ in range(3):
                    print("Warming up GaussianOcc.", flush=True)
                    models["depth"](models["encoder"](color), data, outputs={}, is_train=False)
                torch.cuda.synchronize()
                warmed = True
            torch.cuda.synchronize()
            start = time.perf_counter()
            result = models["depth"](models["encoder"](color), data, outputs={}, is_train=False)
            prediction = result["pred_occ_logits"][0].argmax(0)
            prediction = prediction.cpu().numpy()
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - start) * 1000
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
                    precision="FP32",
                )
            temporary.replace(output)
            print(
                "GaussianOcc {}/{}: {:.1f} ms".format(index + 1, len(frames), elapsed), flush=True
            )
finally:
    lock.unlink()
