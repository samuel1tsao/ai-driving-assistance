"""Prepare the complete nuScenes trainval keyframe manifest."""

import argparse
from pathlib import Path

from run_fog_research import load_notebook_runtime

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=ROOT / "data/depth/datasets/nuscenes"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "data/fog/nuscenes_all.json"
    )
    args = parser.parse_args()
    prepare_nuscenes_manifest = load_notebook_runtime().prepare_nuscenes_manifest
    prepare_nuscenes_manifest(args.root, args.output)
