"""Small notebook-facing adapter for the project's Linux GPU runners."""
import json
import subprocess
from pathlib import Path

DOCKER = Path("C:/Program Files/Docker/Docker/resources/bin/docker.exe")
CONTAINER = "driving-flashocc"
MODELS = [
    {
        "name": "FlashOCC M0",
        "slug": "flashocc_m0",
        "runner": "flashocc",
        "variant": "m0",
        "description": "Supervised ResNet-50; 128-channel BEV/head and 1 m depth bins.",
    },
    {
        "name": "FlashOCC M1",
        "slug": "flashocc_m1",
        "runner": "flashocc",
        "variant": "m1",
        "description": "Same backbone and camera resolution; 256-channel BEV/head, 0.5 m depth bins.",
    },
    {
        "name": "GaussianOcc",
        "slug": "gaussianocc",
        "runner": "gaussianocc",
        "description": "Self-supervised occupancy with Gaussian rendering during training; ResNet-101.",
    },
]


def prepared_samples(root, limit=100):
    manifest = Path(root) / "data/occupancy/inputs/nuscenes/frames.json"
    frames = json.loads(manifest.read_text())["frames"]
    if limit is not None and (not isinstance(limit, int) or limit < 1):
        raise ValueError("Sample count must be a positive integer or None.")
    if limit is not None and limit > len(frames):
        raise ValueError(f"Only {len(frames)} camera frames prepared. Prepare more inputs first.")
    frames = frames[:limit]
    if not frames:
        raise ValueError("Choose at least one prepared frame.")
    from occupancy import validation_scenes

    allowed = validation_scenes(root)
    if any(f["id"].split("/")[0] not in allowed for f in frames):
        raise ValueError("Prepared manifest contains a non-validation scene.")
    labels = Path(root) / "data/occupancy/datasets/occ3d_nuscenes/trainval/gts"
    samples = [
        {
            "id": frame["id"],
            "label": labels / frame["id"] / "labels.npz",
            "dataset": "occ3d_nuscenes",
        }
        for frame in frames
    ]
    if not all(s["label"].is_file() for s in samples):
        raise FileNotFoundError("Some prepared camera frames have no occupancy labels.")
    print(f"Same {len(samples)} held-out frames, six calibrated camera images each.")
    return samples


def run_model(root, spec, samples, refresh=False):
    if not samples or any(s["dataset"] != "occ3d_nuscenes" for s in samples):
        raise ValueError("These checkpoints support nuScenes; Waymo needs compatible weights.")
    expected = prepared_samples(root, len(samples))
    if [s["id"] for s in expected] != [s["id"] for s in samples]:
        raise ValueError("Runner inputs and evaluation labels must have identical frame IDs.")
    if not DOCKER.is_file():
        raise FileNotFoundError(f"Docker CLI missing at {DOCKER}")
    state = subprocess.run([str(DOCKER), "start", CONTAINER], capture_output=True, text=True)
    if state.returncode:
        raise RuntimeError(
            "Start Docker Desktop; run scripts/setup_occupancy.ps1 if needed.\n"
            + state.stderr[-1500:]
        )
    command = [
        str(DOCKER),
        "exec",
        CONTAINER,
        "python",
        "/workspace/scripts/run_" + spec["runner"] + ".py",
        "--limit",
        str(len(samples)),
    ]
    if "variant" in spec:
        command += ["--variant", spec["variant"]]
    if refresh:
        command.append("--refresh")
    log = Path(root) / "reports/occupancy" / (spec["slug"] + "-runner.log")
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f"Running {spec['name']}; detailed output: {log.relative_to(root)}", flush=True)
    lock = Path(root) / "data/occupancy/.notebook-running"
    lock.open("x").close()
    timed_out = False
    try:
        with log.open("w", encoding="utf-8") as output:
            result = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT, timeout=3600)
    except subprocess.TimeoutExpired as error:
        timed_out = True
        raise RuntimeError(
            "Runner timed out. The benchmark lock is retained because Docker may "
            "still be computing; inspect the container before retrying."
        ) from error
    finally:
        if not timed_out:
            lock.unlink()
    if result.returncode:
        raise RuntimeError(log.read_text(encoding="utf-8")[-1800:])
    print(f"{spec['name']}: predictions ready.")
