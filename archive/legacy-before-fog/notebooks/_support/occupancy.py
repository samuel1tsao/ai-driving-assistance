"""Occ3D validation: separate nuScenes/Waymo taxonomies and visibility masks."""
import ast
import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from common import select_samples, latency_summary

MODELS = [
    {'name': 'FlashOcc', 'slug': 'flashocc', 'description': 'Efficient camera occupancy model using channel-to-height conversion.'},
    {'name': 'OccFormer', 'slug': 'occformer', 'description': 'Transformer with local and global paths for 3D scene features.'},
    {'name': 'GaussianOcc', 'slug': 'gaussianocc', 'description': 'Gaussian representation with self-supervised geometry learning.'},
]
SPLIT_URL = 'https://raw.githubusercontent.com/nutonomy/nuscenes-devkit/master/python-sdk/nuscenes/utils/splits.py'
CLASS_NAMES = ['other', 'barrier', 'bicycle', 'bus', 'car', 'construction vehicle',
               'motorcycle', 'pedestrian', 'traffic cone', 'trailer', 'truck',
               'drivable surface', 'other flat', 'sidewalk', 'terrain', 'manmade',
               'vegetation', 'free']
WAYMO_CLASSES = ['general object', 'vehicle', 'pedestrian', 'sign', 'cyclist',
                 'traffic light', 'pole', 'construction cone', 'bicycle', 'motorcycle',
                 'building', 'vegetation', 'tree trunk', 'road', 'walkable', 'free']


def class_names(dataset):
    if dataset == 'occ3d_nuscenes': return CLASS_NAMES
    if dataset == 'occ3d_waymo': return WAYMO_CLASSES
    raise ValueError(f'Unsupported dataset: {dataset}')


def validation_scenes(root):
    cache = Path(root) / 'data/occupancy/nuscenes_val_scenes.json'
    if not cache.exists():
        # Parse the literal list; never execute downloaded Python.
        source = urllib.request.urlopen(SPLIT_URL, timeout=30).read().decode()
        for node in ast.parse(source).body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'val' for t in node.targets):
                scenes = ast.literal_eval(node.value)
                break
        else:
            raise ValueError('Official nuScenes validation scene list was not found.')
        if len(scenes) != 150:
            raise ValueError('Unexpected nuScenes validation split size.')
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps({'source': SPLIT_URL, 'scenes': scenes}, indent=2))
    return set(json.loads(cache.read_text())['scenes'])


def load_samples(root, limit=100, seed=42, dataset='occ3d_nuscenes'):
    class_names(dataset)
    if dataset == 'occ3d_nuscenes':
        scenes = validation_scenes(root)
        folder = Path(root) / 'data/occupancy/datasets/occ3d_nuscenes/trainval/gts'
        samples = [{'id': f'{p.parent.parent.name}/{p.parent.name}', 'label': p, 'dataset':dataset}
                   for p in folder.glob('*/*/labels.npz') if p.parent.parent.name in scenes]
        expected = 150
    else:
        folder = Path(root) / 'data/occupancy/datasets/occ3d_waymo/validation'
        samples = [{'id':f'{p.parent.name}/{p.stem.removesuffix("_04")}', 'label':p, 'dataset':dataset}
                   for p in folder.glob('*/*_04.npz') if p.parent.name in {f'{i:03}' for i in range(202)}]
        expected = 202
    scene_count=len({s['id'].split('/')[0] for s in samples})
    print(f'{dataset}: {len(samples):,} labels from {scene_count}/{expected} validation scenes.')
    if scene_count < expected: print('Partial local dataset: results describe only the available scenes.')
    chosen = select_samples(samples, limit, seed)
    print(f'Using the same {len(chosen)} frames for every model. Training scenes excluded.')
    return chosen


def read_target(sample):
    with np.load(sample['label'], allow_pickle=False) as data:
        if sample.get('dataset','occ3d_nuscenes') == 'occ3d_waymo':
            target = np.where(data['voxel_label']==23,15,data['voxel_label']).astype(np.uint8)
            valid = data['final_voxel_state'].astype(bool) & data['infov'].astype(bool)
        else:
            target = data['semantics']
            valid = data['mask_camera'].astype(bool)
    if target.shape != (200, 200, 16) or valid.shape != target.shape:
        raise ValueError('Expected 0.4 m labels and visibility mask shaped 200 × 200 × 16.')
    return target, valid


def confusion_counts(target, prediction, valid, dataset='occ3d_nuscenes'):
    count = len(class_names(dataset))
    if target.shape != prediction.shape or valid.shape != target.shape:
        raise ValueError('Prediction, target and visibility mask shapes must match.')
    if not np.issubdtype(prediction.dtype, np.integer):
        raise ValueError('Predictions must be integer class IDs, not probabilities.')
    actual, predicted = target[valid], prediction[valid]
    if np.any((predicted < 0) | (predicted >= count)) or np.any((actual < 0) | (actual >= count)):
        raise ValueError(f'{dataset} normalized class IDs must be in 0..{count-1}.')
    return np.bincount(actual.astype(np.int64) * count + predicted, minlength=count*count).reshape(count, count)


def summarize_counts(matrix):
    total = matrix.sum()
    if not total:
        raise ValueError('No camera-visible voxels to score.')
    diagonal = matrix.diagonal()
    union = matrix.sum(0) + matrix.sum(1) - diagonal
    free = matrix.shape[0]-1
    iou = np.divide(diagonal, union, out=np.full(len(diagonal), np.nan), where=union > 0)
    tp = matrix[:free, :free].sum()
    def ratio(n, d):
        return float(n / d) if d else np.nan
    return {'mIoU': float(np.nanmean(iou[:free])),
            'occupied_precision': ratio(tp, matrix[:, :free].sum()),
            'occupied_recall': ratio(tp, matrix[:free, :].sum()),
            'free_space_recall': ratio(matrix[free, free], matrix[free, :].sum()),
            'visible_voxels': int(total)}


def prediction_path(root, spec, sample):
    return Path(root) / 'data/occupancy/predictions' / sample.get('dataset','occ3d_nuscenes') / spec['slug'] / (sample['id'] + '.npz')


def evaluate_model(root, spec, samples):
    dataset=samples[0].get('dataset','occ3d_nuscenes') if samples else 'occ3d_nuscenes'
    if any(s.get('dataset','occ3d_nuscenes')!=dataset for s in samples):
        raise ValueError('Do not mix datasets in a comparison.')
    paths = [prediction_path(root, spec, s) for s in samples]
    found = sum(p.is_file() for p in paths)
    row = {'model': spec['name'], 'dataset':dataset, 'description': spec['description'], 'frames': len(samples),
           'predictions_found': found, **latency_summary([])}
    if not samples or found != len(samples):
        return {**row, 'status': 'needs_inference', 'reason': f'{found}/{len(samples)} predictions. Run the official model and export occupancy arrays; see README.'}
    count=len(class_names(dataset))
    counts = np.zeros((count, count), dtype=np.int64)
    times = []
    try:
        for sample, path in zip(samples, paths):
            target, valid = read_target(sample)
            with np.load(path, allow_pickle=False) as data:
                counts += confusion_counts(target, data['occupancy'], valid, dataset)
                if 'inference_ms' in data:
                    ms = float(data['inference_ms'])
                    if not np.isfinite(ms) or ms <= 0:
                        raise ValueError('inference_ms must be a finite positive measurement.')
                    times.append(ms)
        return {**row, 'status': 'ok', **summarize_counts(counts),
                **latency_summary(times if len(times) == len(samples) else []),
                'timing_source': 'exported runner timings; compare only matching hardware/settings'}
    except (ValueError, KeyError, OSError) as error:
        return {**row, 'status': 'error', 'reason': str(error)}


def show_examples(root, specs, samples, count=3, seed=42, show_inputs=False):
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm
    dataset=samples[0].get('dataset','occ3d_nuscenes') if samples else 'occ3d_nuscenes'
    names=class_names(dataset)
    cmap = plt.get_cmap('tab20', len(names)).copy()
    cmap.set_bad('#dedede')
    norm = BoundaryNorm(np.arange(-.5, len(names)+.5), len(names))
    manifest = Path(root) / 'data/occupancy/inputs/nuscenes/frames.json'
    frames = {}
    if show_inputs and dataset == 'occ3d_nuscenes' and manifest.exists():
        frames = {f['id']: f for f in json.loads(manifest.read_text())['frames']}
    for sample in select_samples(samples, count, seed):
        if sample['id'] in frames:
            from PIL import Image
            figure, cameras = plt.subplots(2, 3, figsize=(15, 6))
            order = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                     'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
            for axis, name in zip(cameras.flat, order):
                with Image.open(Path(root) / frames[sample['id']]['cams'][name]['data_path']) as image:
                    axis.imshow(image)
                axis.set_title(name.replace('CAM_', '').replace('_', ' ').title())
                axis.axis('off')
            figure.suptitle(f"Input camera views: {sample['id']}", fontsize=10)
            figure.tight_layout()
            plt.show()
        target, valid = read_target(sample)
        z = 4  # voxel center = 0.8 m above ego origin; range starts at -1 m
        available = [(s, prediction_path(root, s, sample)) for s in specs if prediction_path(root, s, sample).is_file()]
        fig, axes = plt.subplots(1, 1 + len(available), figsize=(6 * (1 + len(available)), 5), squeeze=False)
        fig.subplots_adjust(top=.76, bottom=.12)
        images = [('Ground truth', target)]
        for spec, path in available:
            with np.load(path, allow_pickle=False) as data:
                prediction = data['occupancy']
            confusion_counts(target, prediction, valid, dataset)
            images.append((spec['name'], prediction))
        for axis, (title, volume) in zip(axes[0], images):
            masked = np.ma.masked_where(~valid[..., z], volume[..., z])
            im = axis.imshow(masked.T, origin='lower', extent=(-40,40,-40,40), cmap=cmap, norm=norm)
            axis.set(title=title, xlabel='Ego x (m)', ylabel='Ego y (m)')
        colorbar = fig.colorbar(im, ax=list(axes[0]), ticks=range(len(names)), shrink=.9)
        colorbar.ax.set_yticklabels(names, fontsize=8)
        scene, frame = sample['id'].split('/')
        fig.suptitle(f'{dataset}: {scene} / {frame[:12]}\nz = 0.8 m; grey = excluded by visibility mask', fontsize=11)
        plt.show()
    if samples:
        print('Metrics use all visible heights. These examples show only one horizontal slice.')
