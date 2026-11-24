"""Train YOLO11n on Foggy Cityscapes with either image filters or a fog-value FiLM adapter.

Prepare reads Cityscapes gtFine instance masks. Adaptive mode is Image-Adaptive
YOLO. Baseline mode is YOLO11n alone. film-beta feeds each image's real fog
level through FiLM. film-zero trains the same adapter with that value fixed at 0.

Examples, from the repository root:

  .\\.venv\\Scripts\\python.exe scripts\\train_ia_yolo.py train --mode film-beta
  .\\.venv\\Scripts\\python.exe scripts\\train_ia_yolo.py train --mode film-zero
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT / '.ultralytics'))
os.environ.setdefault('HF_HOME', str(ROOT / '.cache' / 'huggingface'))
sys.path.insert(0, str(ROOT / 'notebooks' / '_support'))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=('prepare', 'train'))
    parser.add_argument('--fog-root', type=Path, help='Directory that contains leftImg8bit_foggy')
    parser.add_argument('--gtfine', type=Path, help='Cityscapes gtFine directory')
    parser.add_argument('--manifest', type=Path,
                        default=ROOT / 'reports' / 'fog' / 'ia-yolo-5070' / 'cityscapes-detection.json')
    parser.add_argument('--output', type=Path, help='Run directory. Defaults to reports/fog/ia-yolo-5070/<mode>')
    parser.add_argument('--mode', choices=('adaptive', 'baseline', 'film-beta', 'film-zero'),
                        default='adaptive')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--image-size', type=int, default=640)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--limit', type=int, help='Smoke test: cap each split and skip the final mAP')
    parser.add_argument('--fresh', action='store_true', help='Ignore last.pt and start at epoch 1')
    parser.add_argument('--allow-cpu', action='store_true')
    args = parser.parse_args()

    from ia_yolo import (
        build_detection_manifest, default_fog_root, evaluate_detector, gtfine_instructions,
        image_root_of, load_detection_manifest, resolve_gtfine, train_detector,
    )

    fog_root = Path(args.fog_root).resolve() if args.fog_root else default_fog_root(ROOT)
    if fog_root is None or not fog_root.is_dir():
        raise SystemExit('Foggy Cityscapes was not found. Pass --fog-root, or keep the Kaggle cache '
                         'under fog-colab-data/.cache/kagglehub.')
    print(f'Foggy images: {image_root_of(fog_root)}', flush=True)
    gtfine = resolve_gtfine(fog_root, args.gtfine)
    if gtfine is None and (ROOT / 'gtFine').is_dir():
        gtfine = resolve_gtfine(ROOT)
    if args.command == 'prepare':
        if gtfine is None:
            raise SystemExit(gtfine_instructions(fog_root))
        build_detection_manifest(fog_root, gtfine, args.manifest)
        print(f'Wrote {args.manifest}', flush=True)
        return
    if not args.manifest.is_file():
        if gtfine is None:
            raise SystemExit(gtfine_instructions(fog_root))
        build_detection_manifest(fog_root, gtfine, args.manifest)
    rows = load_detection_manifest(args.manifest)
    device = 'cuda' if not args.allow_cpu else 'cpu'
    output = args.output or (ROOT / 'reports' / 'fog' / 'ia-yolo-5070' / args.mode)
    try:
        detector, adapter = train_detector(
            rows, output, mode=args.mode, epochs=args.epochs, image_size=args.image_size,
            batch_size=args.batch_size, seed=args.seed, device=device, limit=args.limit,
            fresh=args.fresh)
        if args.limit:
            print('Smoke test finished. Omit --limit for the full run and the beta mAP table.', flush=True)
            return
        test_rows = [row for row in rows if row['split'] == 'test']
        table = evaluate_detector(detector, adapter, test_rows, device, args.image_size, output,
                                  mode=args.mode)
        print(json.dumps(table, indent=2), flush=True)
        (output / 'status.json').write_text(json.dumps({'status': 'complete', 'mode': args.mode}),
                                            encoding='utf-8')
    except Exception as error:
        output.mkdir(parents=True, exist_ok=True)
        (output / 'status.json').write_text(
            json.dumps({'status': 'error', 'mode': args.mode, 'reason': f'{type(error).__name__}: {error}'}),
            encoding='utf-8')
        raise


if __name__ == '__main__':
    main()
