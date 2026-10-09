"""Batch-one local prediction timing; synthetic input is explicitly not accuracy."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import time

ROOT = Path(__file__).resolve().parents[2]
(ROOT / '.ultralytics').mkdir(exist_ok=True)
os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT / '.ultralytics'))

import cv2
import numpy as np
import psutil
import torch
import ultralytics
from ultralytics import YOLO


def profile(checkpoint, frame, args):
    model = YOLO(checkpoint)
    options = dict(device=args.device, imgsz=args.imgsz, verbose=False, quantize=32, rect=False)
    for _ in range(args.warmup):
        model.predict(frame, **options)
    cuda = args.device != 'cpu'
    if cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    elapsed, stages = [], []
    for _ in range(args.frames):
        if cuda:
            torch.cuda.synchronize()
        start = time.perf_counter()
        result = model.predict(frame, **options)[0]
        if cuda:
            torch.cuda.synchronize()
        elapsed.append((time.perf_counter() - start) * 1000)
        stages.append(result.speed)
    row = dict(model=checkpoint, mean_ms=float(np.mean(elapsed)),
               p50_ms=float(np.percentile(elapsed, 50)), p95_ms=float(np.percentile(elapsed, 95)),
               serial_prediction_fps=1000 / float(np.mean(elapsed)),
               mean_stages_ms={key: float(np.mean([s[key] for s in stages])) for key in stages[0]},
               process_rss_mb=psutil.Process().memory_info().rss / 1024**2,
               peak_torch_allocated_mb=torch.cuda.max_memory_allocated() / 1024**2 if cuda else None,
               peak_torch_reserved_mb=torch.cuda.max_memory_reserved() / 1024**2 if cuda else None)
    path = Path(checkpoint)
    row['checkpoint_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
    del model
    gc.collect()
    if cuda:
        torch.cuda.empty_cache()
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='+', default=['yolov8n.pt', 'yolo11n.pt', 'yolo26n.pt', 'yolo26s.pt'])
    parser.add_argument('--image', type=Path)
    parser.add_argument('--frames', type=int, default=100)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--device', default='0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output', type=Path, default=ROOT / 'reports/local_profile.json')
    args = parser.parse_args()
    if args.frames < 1 or args.warmup < 1 or args.imgsz < 32:
        parser.error('frames and warmup must be positive; imgsz must be at least 32')
    if args.device != 'cpu' and not torch.cuda.is_available():
        parser.error('CUDA requested but unavailable')
    frame = cv2.imread(str(args.image)) if args.image else np.zeros((720, 1280, 3), dtype=np.uint8)
    if frame is None:
        parser.error('Cannot decode input image')
    report = dict(input=str(args.image) if args.image else 'synthetic_black_1280x720',
                  scope='Prediction only; excludes capture, transport, depth, tracking and alerts. No accuracy measurement.',
                  python=platform.python_version(), torch=torch.__version__, ultralytics=ultralytics.__version__,
                  gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                  device=args.device, precision='FP32', imgsz=args.imgsz, rect=False,
                  head='PyTorch default or exported artifact metadata', frames=args.frames, warmup=args.warmup, results=[])
    for checkpoint in args.models:
        report['results'].append(profile(checkpoint, frame, args))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps(report['results'][-1]), flush=True)


if __name__ == '__main__':
    main()
