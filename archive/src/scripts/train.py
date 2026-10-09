"""Fine-tune on an explicitly supplied YOLO dataset; no implicit dataset download."""
import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
(ROOT / '.ultralytics').mkdir(exist_ok=True)
os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT / '.ultralytics'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--model', default='yolo26n.pt')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--device', default='0')
    args = parser.parse_args()
    if not args.data.is_file() or args.epochs < 1 or args.batch < 1:
        parser.error('Supply an existing data YAML and positive epochs/batch')
    from ultralytics import YOLO
    YOLO(args.model).train(data=str(args.data.resolve()), epochs=args.epochs,
                          batch=args.batch, device=args.device, imgsz=640, seed=42,
                          workers=0, project=str(ROOT / 'runs/train'), name=Path(args.model).stem,
                          deterministic=True, patience=10)


if __name__ == '__main__':
    main()
