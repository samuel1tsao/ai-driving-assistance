"""Prepare the complete nuScenes trainval keyframe manifest."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'notebooks/_support'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT/'data/depth/datasets/nuscenes')
    parser.add_argument('--output', type=Path, default=ROOT/'data/fog/nuscenes_all.json')
    args = parser.parse_args()
    from fog_prepare import prepare_nuscenes
    prepare_nuscenes(args.root, args.output)
