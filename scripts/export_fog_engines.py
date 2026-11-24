"""Build TensorRT engines from the Cityscapes YOLO11n checkpoints.

The training checkpoints store a raw detector state dict, so this loads those
weights into YOLO11n and exports a static 640 fp16 engine on this GPU. An
engine only runs on the GPU that built it.
"""
import os
import sys
from pathlib import Path

import torch
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'reports' / 'fog' / 'ia-yolo-5070'
OUT = RUNS / 'engines'
MODES = ('baseline', 'adaptive')


def export_mode(mode):
    checkpoint = RUNS / mode / 'best.pt'
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    destination = OUT / f'{mode}.pt'
    OUT.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(ROOT / 'yolo11n.pt'))
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model.model.load_state_dict(state['model'])
    model.save(destination)
    exported = YOLO(str(destination))
    engine = exported.export(format='engine', imgsz=640, half=True, device=0, simplify=True)
    if mode == 'adaptive':
        torch.save(state['adapter'], OUT / 'adaptive-adapter.pt')
    print(f'{mode}: {engine}')
    return engine


def main():
    os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT / '.ultralytics'))
    if not torch.cuda.is_available():
        raise SystemExit('CUDA is required to build a TensorRT engine.')
    for mode in MODES:
        export_mode(mode)


if __name__ == '__main__':
    sys.exit(main())
