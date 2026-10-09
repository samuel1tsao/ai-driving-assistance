"""Prepare matched six-camera inputs; extract only selected images, never whole archives."""
import argparse
import json
import shutil
import sys
import tarfile
from pathlib import Path

import ijson

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks/_support"))
from occupancy import load_samples

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--metadata", type=Path, required=True)
parser.add_argument("--images", type=Path, required=True)
parser.add_argument("--limit", type=int, default=100)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()
samples = load_samples(ROOT, args.limit, args.seed, "occ3d_nuscenes")
selected = {s["id"].split("/")[1]: s["id"] for s in samples}
folder = ROOT / "data/occupancy/inputs/nuscenes"
folder.mkdir(parents=True, exist_ok=True)
tables = {"sample_data", "calibrated_sensor", "sensor", "ego_pose"}
print("Reading original nuScenes calibration and frame metadata.", flush=True)
with tarfile.open(args.metadata, "r|gz") as archive:
    for member in archive:
        name = Path(member.name).stem
        if member.isfile() and name in tables and member.name.endswith(".json"):
            output = folder / (name + ".json")
            if not output.exists():
                temporary = output.with_suffix(".partial")
                with temporary.open("wb") as dest:
                    shutil.copyfileobj(archive.extractfile(member), dest)
                temporary.replace(output)

records = {}
for name in ("sensor", "calibrated_sensor"):
    with (folder / (name + ".json")).open("rb") as source:
        records[name] = {r["token"]: r for r in ijson.items(source, "item", use_float=True)}
with (folder / "sample_data.json").open("rb") as source:
    camera_records = [
        r
        for r in ijson.items(source, "item", use_float=True)
        if r["sample_token"] in selected and r["is_key_frame"] and r["filename"].endswith(".jpg")
    ]
pose_ids = {r["ego_pose_token"] for r in camera_records}
with (folder / "ego_pose.json").open("rb") as source:
    poses = {
        r["token"]: r for r in ijson.items(source, "item", use_float=True) if r["token"] in pose_ids
    }
frames = {token: {"id": identity, "cams": {}} for token, identity in selected.items()}
needed = {}
for record in camera_records:
    calibration = records["calibrated_sensor"][record["calibrated_sensor_token"]]
    channel = records["sensor"][calibration["sensor_token"]]["channel"]
    pose = poses[record["ego_pose_token"]]
    relative = Path("images") / channel / Path(record["filename"]).name
    needed[relative.name] = folder / relative
    frames[record["sample_token"]]["cams"][channel] = {
        "data_path": (folder / relative).relative_to(ROOT).as_posix(),
        "cam_intrinsic": calibration["camera_intrinsic"],
        "sensor2ego_rotation": calibration["rotation"],
        "sensor2ego_translation": calibration["translation"],
        "ego2global_rotation": pose["rotation"],
        "ego2global_translation": pose["translation"],
    }
assert all(len(f["cams"]) == 6 for f in frames.values()), "Missing calibrated camera views."
missing = {name: dest for name, dest in needed.items() if not dest.exists()}
print(f"Extracting {len(missing)} selected images from the downloaded archive.", flush=True)
if missing:
    with tarfile.open(args.images, "r|gz") as archive:
        for member in archive:
            name = Path(member.name).name
            if member.isfile() and name in missing:
                output = missing.pop(name)
                output.parent.mkdir(parents=True, exist_ok=True)
                temporary = output.with_suffix(".partial")
                with temporary.open("wb") as dest:
                    shutil.copyfileobj(archive.extractfile(member), dest)
                temporary.replace(output)
                if not missing:
                    break
assert not missing, f"{len(missing)} camera images not found; no complete manifest written."
manifest = {
    "dataset": "occ3d_nuscenes",
    "seed": args.seed,
    "metadata_archive": str(args.metadata.resolve()),
    "image_archive": str(args.images.resolve()),
    "frames": list(frames.values()),
}
output = folder / "frames.json"
temporary = output.with_suffix(".partial")
temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
temporary.replace(output)
print(f"Ready: {len(frames)} matched frames, {len(needed)} camera images. {output}")
