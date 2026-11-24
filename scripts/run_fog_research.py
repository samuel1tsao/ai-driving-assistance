"""One entry point for preflight, preparation, full training, resume and saved-run analysis."""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT/'.ultralytics'))
os.environ.setdefault('HF_HOME', str(ROOT/'.cache/huggingface'))
os.environ.setdefault('HF_HUB_DISABLE_PROGRESS_BARS', '1')
sys.path.insert(0, str(ROOT/'notebooks/_support'))


def load_rows(config):
    from fog_research import read_manifest
    rows = []
    for name in ('manifest','city_manifest'):
        path = config.get(name)
        if path and Path(path).is_file():
            print(f'Validate {path}', flush=True)
            rows.extend(read_manifest(path))
    return rows


def main():
    from fog_training import default_config, preflight, run_training
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['preflight','prepare','download-models','train','analyze'])
    parser.add_argument('--config', type=Path, help='JSON overrides for default_config')
    parser.add_argument('--baseline-only', action='store_true', help='Train all selected detectors without density labels')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--download-fog', action='store_true', help='Download the ~29.1 GB Kaggle source used in Untitled2.ipynb')
    parser.add_argument('--fog-root', type=Path, help='Existing Foggy Cityscapes or Kaggle cache directory')
    args = parser.parse_args()
    config = default_config(ROOT)
    if args.config:
        config.update(json.loads(args.config.read_text(encoding='utf-8')))
    if args.baseline_only:
        config['arms'] = ['rgb_control']
    if args.output:
        config['output'] = str(args.output.resolve())
    if args.command == 'prepare':
        from fog_prepare import prepare_nuscenes
        prepare_nuscenes(ROOT/'data/depth/datasets/nuscenes', config['manifest'])
        from fog_data import cityscapes_manifest, resolve_foggy_images
        city = resolve_foggy_images(ROOT, args.fog_root, download=args.download_fog)
        if city is not None and not Path(config['city_manifest']).exists():
            cityscapes_manifest(city, config['city_manifest'])
        return
    if args.command == 'download-models':
        import torch
        from fog_research import Detector
        for name, checkpoint in config['checkpoints'].items():
            print(f'Load pretrained {name}', flush=True)
            detector = Detector(name, checkpoint, 'cpu')
            print(f'{name}: {sum(p.numel() for p in detector.model.parameters()):,} parameters', flush=True)
            del detector
        return
    if args.command == 'analyze':
        from fog_analytics import export_analytics
        results = export_analytics(config['output'])
        print(results.to_string(index=False))
        return
    rows = load_rows(config)
    checks = preflight(config, rows)
    print(checks.to_string(index=False))
    if not checks.ready.all():
        raise SystemExit(2)
    if args.command == 'train':
        run_training(rows, config)


if __name__ == '__main__':
    main()
