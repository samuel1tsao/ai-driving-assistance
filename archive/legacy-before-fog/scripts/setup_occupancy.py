"""Recreate the project-specific research runtime without changing host environment variables."""
import subprocess
from pathlib import Path

import gdown

ROOT = Path(__file__).resolve().parents[1]
DOCKER = "C:/Program Files/Docker/Docker/resources/bin/docker.exe"
CONTAINER = "driving-flashocc"
IMAGE = "pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel"
REPOS = [
    (
        "FlashOCC",
        "https://github.com/Yzichen/FlashOCC.git",
        "449dc7947d6549ccb72a99d6f6d8c02f166ec0dd",
    ),
    (
        "FlashOCC/mmdetection3d",
        "https://github.com/open-mmlab/mmdetection3d.git",
        "c9541b0db89498fdea5cafd05b7b17f7b625b858",
    ),
    (
        "GaussianOcc",
        "https://github.com/GANWANSHUI/GaussianOcc.git",
        "0e90118ceef5d16a9d3e4fd132c6bd525054d708",
    ),
]
WEIGHTS = {
    "flashocc-r50-256x704.pth": "1k9BzXB2nRyvXhqf7GQx3XNSej6Oq6I-B",
    "flashocc-m0.pth": "14my3jdqiIv6VIrkozQ6-ruEcBOPVlWGJ",
    "gaussianocc/encoder.pth": "10qpkqB4GQEgAHrYMdESfr60vcRpEw086",
    "gaussianocc/depth.pth": "1E1GHcm4oOgx_T2YX9_aoiKn1xCpNvvUD",
}


def run(command):
    subprocess.run(command, check=True, cwd=ROOT)


print("Checking Docker Desktop. Start it before running this setup.", flush=True)
run([DOCKER, "info", "--format", "{{.ServerVersion}}"])
for name, url, revision in REPOS:
    folder = ROOT / "notebooks/_runtime" / name
    if not folder.exists():
        run(["git", "clone", url, str(folder)])
        run(["git", "-C", str(folder), "checkout", revision])
    actual = subprocess.check_output(["git", "-C", str(folder), "rev-parse", "HEAD"], text=True)
    if actual.strip() != revision:
        raise RuntimeError(f"{name}: unexpected revision; existing files were not overwritten.")
syntax = ROOT / "notebooks/_runtime/FlashOCC/projects/mmdet3d_plugin/models/detectors/bevdet_occ.py"
source = syntax.read_bytes()
syntax.write_bytes(
    source.replace(b"return pred_occ_label, *outs[1:]", b"return (pred_occ_label, *outs[1:])")
)
pose = ROOT / "notebooks/_runtime/GaussianOcc/networks/pose_decoder.py"
source = pose.read_text(encoding="utf-8")
optional_import = "from pytorch3d.transforms.rotation_conversions import matrix_to_axis_angle"
if "\n" + optional_import + "\n" in source:
    source = source.replace("\n" + optional_import + "\n", "\n")
    source = source.replace(
        "        if coarse_RT is not None:\n",
        "        if coarse_RT is not None:\n            " + optional_import + "\n",
    )
    pose.write_text(source, encoding="utf-8")
for name, identifier in WEIGHTS.items():
    output = ROOT / "data/models/occupancy" / name
    output.parent.mkdir(parents=True, exist_ok=True)
    if not output.is_file():
        temporary = output.with_suffix(".partial")
        if not gdown.download(id=identifier, output=str(temporary), quiet=False):
            raise RuntimeError("Checkpoint download failed: " + name)
        temporary.replace(output)
inspection = subprocess.run([DOCKER, "inspect", CONTAINER], capture_output=True)
if inspection.returncode:
    run(
        [
            DOCKER,
            "run",
            "-d",
            "--gpus",
            "all",
            "--name",
            CONTAINER,
            "--shm-size",
            "2g",
            "--mount",
            f"type=bind,source={ROOT},target=/workspace",
            "-w",
            "/workspace/notebooks/_runtime/FlashOCC",
            IMAGE,
            "sleep",
            "infinity",
        ]
    )
run([DOCKER, "start", CONTAINER])
base = [DOCKER, "exec", CONTAINER, "python", "-m", "pip", "install"]
run(
    base
    + [
        "numpy==1.21.6",
        "setuptools==59.5.0",
        "wheel",
        "ninja",
        "mmcv-full==1.5.3",
        "-f",
        "https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html",
    ]
)
run(
    base
    + [
        "mmdet==2.25.1",
        "mmsegmentation==0.25.0",
        "nuscenes-devkit==1.1.9",
        "lyft_dataset_sdk==0.0.8",
        "networkx==2.2",
        "numba==0.53.0",
        "plyfile==0.7.4",
        "scikit-image==0.19.3",
        "tensorboard==2.11.2",
        "trimesh==2.35.39",
        "yapf==0.40.1",
        "timm==0.6.13",
        "einops==0.6.1",
        "configargparse",
        "torch_efficient_distloss",
    ]
)
run(base + ["--force-reinstall", "--no-deps", "opencv-python-headless==4.6.0.66"])
run(
    base
    + [
        "--no-deps",
        "--no-build-isolation",
        "-e",
        "/workspace/notebooks/_runtime/FlashOCC/mmdetection3d",
    ]
)
run(
    [
        DOCKER,
        "exec",
        "-w",
        "/workspace/notebooks/_runtime/FlashOCC/projects",
        CONTAINER,
        "python",
        "setup.py",
        "build_ext",
        "--inplace",
    ]
)
for name in (
    "diff-gaussian-rasterization-confidence",
    "diff-gaussian-rasterization-confidence-semantic",
    "simple-knn",
):
    run(
        base
        + [
            "--no-deps",
            "--no-build-isolation",
            "/workspace/notebooks/_runtime/GaussianOcc/submodule/" + name,
        ]
    )
print("Runtime ready. Prepare camera inputs, then run the occupancy notebook.", flush=True)
