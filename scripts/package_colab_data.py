"""Package only the prepared study's referenced images and portable manifests."""
import argparse
import json
import os
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def inventory(root, include_fog=False):
    root = root.resolve()
    parents = {}
    manifests = ['nuscenes_all.json'] + (['cityscapes.json'] if include_fog else [])
    files, documents = {}, {}
    for name in manifests:
        source = root / 'data/fog' / name
        rows = json.loads(source.read_text(encoding='utf-8'))
        for index, row in enumerate(rows, 1):
            original = source.parent / row['image']
            parent = parents.get(original.parent)
            if parent is None:
                parent = original.parent.resolve()
                parents[original.parent] = parent
            image = parent / original.name
            if not image.is_file():
                raise FileNotFoundError(image)
            relative = image.relative_to(root).as_posix()
            files[relative] = image
            row['image'] = os.path.relpath(relative, 'data/fog').replace('\\', '/')
            if index % 25000 == 0:
                print(f'Checked {index:,}/{len(rows):,} images in {name}', flush=True)
        documents['data/fog/' + name] = json.dumps(rows)
    coverage = root / 'data/fog/nuscenes_all.coverage.json'
    report = json.loads(coverage.read_text(encoding='utf-8'))
    if not report.get('complete'):
        raise ValueError('Prepare complete nuScenes coverage before packaging.')
    documents['data/fog/nuscenes_all.coverage.json'] = json.dumps(report)
    return files, documents


def package(root, output, include_fog=False, dry_run=False):
    files, documents = inventory(root, include_fog)
    size = sum(p.stat().st_size for p in files.values())
    print(f'{len(files):,} images; {size / 2**30:.2f} GiB before ZIP overhead', flush=True)
    if dry_run:
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation preserves any previously generated bundle.
    with zipfile.ZipFile(output, 'x', compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for index, (relative, path) in enumerate(files.items(), 1):
            archive.write(path, relative)
            if index % 10000 == 0:
                print(f'Packed {index:,}/{len(files):,}', flush=True)
        for relative, content in documents.items():
            archive.writestr(relative, content)
    print(output, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'dist/fog-colab-data.zip')
    parser.add_argument('--include-fog', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    package(ROOT, args.output, args.include_fog, args.dry_run)
