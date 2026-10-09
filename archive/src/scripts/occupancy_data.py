"""Dataset adapters for the shared occupancy evaluation notebook.

The official occupancy projects use different directory layouts, label keys,
free-space IDs, and visibility masks. This module converts their ground-truth
NPZ files to one small manifest contract without changing the original data.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    display_name: str
    free_id: int
    class_ids: tuple[int, ...]
    ignore_ids: tuple[int, ...]
    voxel_size_m: tuple[float, float, float]
    axis_range_m: tuple[float, float, float, float, float, float]
    notes: str


DATASET_SPECS = {
    'occ3d_nuscenes': DatasetSpec(
        'occ3d_nuscenes',
        'Occ3D-nuScenes',
        17,
        tuple(range(18)),
        (),
        (0.4, 0.4, 0.4),
        (-40.0, -40.0, -1.0, 40.0, 40.0, 5.4),
        'labels.npz: semantics + mask_camera; free is class 17',
    ),
    'occ3d_waymo': DatasetSpec(
        'occ3d_waymo',
        'Occ3D-Waymo',
        15,
        tuple(range(16)),
        (),
        (0.4, 0.4, 0.4),
        (-40.0, -40.0, -1.0, 40.0, 40.0, 5.4),
        '*_04.npz: voxel_label + final_voxel_state; free is class 15',
    ),
    'openoccupancy': DatasetSpec(
        'openoccupancy',
        'OpenOccupancy (OpenOcc v2)',
        0,
        tuple(range(17)),
        (0,),
        (0.4, 0.4, 0.4),
        (-40.0, -40.0, -1.0, 40.0, 40.0, 5.4),
        'labels.npz: semantics; class 0 is empty/noise for semantic mIoU',
    ),
    'nuscenes_occupancy': DatasetSpec(
        'nuscenes_occupancy',
        'nuScenes-Occupancy v0.1',
        0,
        tuple(range(17)),
        (0,),
        (0.2, 0.2, 0.2),
        (-51.2, -51.2, -5.0, 51.2, 51.2, 3.0),
        'dense 0.2m labels; class 0 is empty/noise for semantic mIoU',
    ),
}


def get_dataset_spec(dataset: str) -> DatasetSpec:
    """Return the documented geometry and label policy for one dataset."""

    try:
        return DATASET_SPECS[dataset]
    except KeyError as error:
        choices = ', '.join(sorted(DATASET_SPECS))
        raise ValueError(f'Unknown occupancy dataset {dataset!r}; choose {choices}') from error


def _metadata_paths(root: Path, split: str) -> list[Path]:
    """Read official OpenOccupancy/nuScenes-Occupancy split metadata when present."""

    candidates = [
        root / f'nuscenes_infos_{split}_occ.pkl',
        root / f'nuscenes_infos_{split}.pkl',
        root.parent / f'nuscenes_infos_{split}_occ.pkl',
        root.parent / f'nuscenes_infos_{split}.pkl',
    ]
    metadata = next((path for path in candidates if path.is_file()), None)
    if metadata is None:
        return []
    with metadata.open('rb') as handle:
        payload = pickle.load(handle)

    paths: list[Path] = []

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if key in {'occ_path', 'gt_path', 'label_path'} and isinstance(child, str):
                    candidate = Path(child)
                    paths.append(candidate if candidate.is_absolute() else root / candidate)
                else:
                    visit(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                visit(child)

    visit(payload)
    return [path.resolve() for path in paths if path.suffix == '.npz']


def discover_label_files(root: Path, dataset: str, split: str = 'val') -> list[Path]:
    """Find official ground-truth files without silently mixing train and val."""

    root = root.expanduser().resolve()
    if not root.is_dir():
        return []
    if dataset == 'occ3d_nuscenes':
        trainval = root / 'trainval'
        container = trainval if trainval.is_dir() else root
        annotation_path = container / 'annotations.json'
        scene_names = None
        if split in {'train', 'val'} and annotation_path.is_file():
            metadata = json.loads(annotation_path.read_text(encoding='utf-8'))
            scene_names = set(metadata.get(f'{split}_split', []))
        files = sorted((container / 'gts').glob('*/*/labels.npz'))
        if scene_names is not None:
            files = [path for path in files if path.parts[-3] in scene_names]
        if files:
            return files
        return sorted({path for pattern in ('scene_*/*.npz', 'scene_*/*/*.npz') for path in root.glob(pattern)})
    if dataset == 'occ3d_waymo':
        split_dir = {'train': 'training', 'val': 'validation', 'test': 'test'}.get(split, split)
        files = sorted((root / split_dir).glob('*/*.npz'))
        coarse = [path for path in files if path.name.endswith('_04.npz')]
        if coarse:
            return coarse
        files = [path for path in files if not path.stem.endswith('_04')]
        return files or sorted({path for pattern in ('scene_*/*.npz', 'scene_*/*/*.npz') for path in root.glob(pattern)})
    if split != 'all':
        metadata_files = _metadata_paths(root, split)
        if metadata_files:
            return sorted(path for path in metadata_files if path.is_file())
        mini_files = sorted({path for pattern in ('scene_*/*.npz', 'scene_*/*/*.npz') for path in root.glob(pattern)})
        if mini_files:
            return mini_files
        raise ValueError(
            f'Cannot prove the {split!r} split for {dataset}; use split="all" or provide '
            'the official *_infos_<split>_occ.pkl file.'
        )
    patterns = ['openocc_v2/**/labels.npz', 'openocc_v2.1/**/labels.npz', '**/labels.npz']
    if dataset == 'openoccupancy':
        patterns.append('scene_*/*.npz')
        patterns.append('scene_*/*/*.npz')
    found = {path.resolve() for pattern in patterns for path in root.glob(pattern)}
    return sorted(found)


def load_label_file(path: Path, dataset: str) -> tuple[np.ndarray, np.ndarray]:
    """Load labels and the official evaluation visibility mask."""

    spec = get_dataset_spec(dataset)
    if path.suffix == '.npy':
        semantics = np.load(path).astype(np.int32)
        return semantics, np.ones(semantics.shape, dtype=bool)
    payload = np.load(path)
    if dataset == 'occ3d_waymo' and 'occ_label' not in payload:
        semantics = payload['voxel_label'].astype(np.int32)
        valid = payload['final_voxel_state'].astype(bool)
        return semantics, valid
    key = (
        'semantics'
        if 'semantics' in payload
        else 'occupancy'
        if 'occupancy' in payload
        else 'occ_label'
    )
    if key not in payload:
        raise ValueError(f'{path} has no semantics/occupancy array')
    semantics = payload[key].astype(np.int32)
    if 'occ_mask_camera' in payload:
        valid = payload['occ_mask_camera'].astype(bool)
    elif dataset == 'occ3d_nuscenes' and 'mask_camera' in payload:
        valid = payload['mask_camera'].astype(bool)
    elif 'valid_mask' in payload:
        valid = payload['valid_mask'].astype(bool)
    else:
        valid = np.ones(semantics.shape, dtype=bool)
    if valid.shape != semantics.shape:
        raise ValueError(f'{path} visibility mask {valid.shape} does not match {semantics.shape}')
    return semantics, valid


def _frame_parts(path: Path, dataset: str) -> tuple[str, str]:
    if dataset == 'occ3d_nuscenes' and path.name == 'labels.npz':
        return path.parts[-3], path.parts[-2]
    if dataset == 'occ3d_waymo' and path.name.endswith('_04.npz'):
        return path.parts[-2], path.stem.removesuffix('_04')
    return path.parent.name, path.stem


def build_manifest(
    root: Path,
    dataset: str,
    split: str,
    output_manifest: Path,
    normalized_dir: Path,
    limit: int | None = None,
) -> 'list[dict[str, str]]':
    """Normalize labels and write a manifest consumed by the notebook."""

    import csv

    files = discover_label_files(root, dataset, split)
    if limit is not None:
        files = files[:limit]
    if not files:
        return []
    spec = get_dataset_spec(dataset)
    rows: list[dict[str, str]] = []
    normalized_dir.mkdir(parents=True, exist_ok=True)
    for source in files:
        scene, frame = _frame_parts(source, dataset)
        normalized = normalized_dir / dataset / split / scene / f'{frame}.npz'
        normalized.parent.mkdir(parents=True, exist_ok=True)
        if not normalized.is_file():
            semantics, valid = load_label_file(source, dataset)
            np.savez_compressed(normalized, occupancy=semantics, valid_mask=valid)
        rows.append(
            {
                'dataset': dataset,
                'split': split,
                'scene': scene,
                'frame': frame,
                'ground_truth': str(normalized.relative_to(output_manifest.parent)),
                'source_label': str(source),
                'free_id': str(spec.free_id),
                'voxel_size_m': ','.join(map(str, spec.voxel_size_m)),
            }
        )
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with output_manifest.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def iter_npz_paths(value: Iterable[Path]) -> Iterable[Path]:
    """Yield only existing NPZ prediction paths for a manifest helper."""

    return (path for path in value if path.is_file() and path.suffix == '.npz')
