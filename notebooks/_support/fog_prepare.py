"""Resumable nuScenes preparation without loading millions of sweep records into RAM."""
import hashlib
import json
import os
import sqlite3
from collections import Counter
from pathlib import Path

import ijson
import numpy as np
from filelock import FileLock
from pyquaternion import Quaternion
from shapely.geometry import MultiPoint, box as rectangle
from tqdm.auto import tqdm

CAMERAS = ('CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT')
SCHEMA = 3


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temporary, path)


def stream(path):
    with Path(path).open('rb') as handle:
        yield from ijson.items(handle, 'item', use_float=True)


def source_signature(root, version):
    names = ('scene', 'sample', 'sample_data', 'sample_annotation', 'ego_pose',
             'calibrated_sensor', 'sensor', 'instance', 'category')
    return {'schema': SCHEMA, 'root': str(root), 'version': version,
            'tables': {name: [((root/version)/f'{name}.json').stat().st_size,
                              ((root/version)/f'{name}.json').stat().st_mtime_ns] for name in names}}


def metadata_index(root, version, cache):
    signature = source_signature(root, version)
    db_path = cache / 'metadata.sqlite'
    stamp = cache / 'metadata-source.json'
    if db_path.is_file() and stamp.is_file() and json.loads(stamp.read_text()) == signature:
        return db_path
    temp = cache / 'metadata-building.sqlite'
    # Recreate only this explicitly named disposable database after an interrupted import.
    if temp.exists():
        temp.unlink()
    connection = sqlite3.connect(temp)
    try:
        connection.execute('PRAGMA journal_mode=OFF')
        connection.execute('PRAGMA synchronous=OFF')
        connection.execute('CREATE TABLE records(kind TEXT, token TEXT, parent TEXT, payload TEXT, PRIMARY KEY(kind,token))')
        camera_poses = set()
        for kind in signature['tables']:
            batch = []
            for row in tqdm(stream(root/version/f'{kind}.json'), desc=f'Index {kind}', unit=' rows', mininterval=2):
                if kind == 'sample_data':
                    if not row['is_key_frame'] or not row['filename'].startswith('samples/CAM_'):
                        continue
                    camera_poses.add(row['ego_pose_token'])
                if kind == 'ego_pose' and row['token'] not in camera_poses:
                    continue
                parent = row.get('sample_token', row.get('scene_token', ''))
                batch.append((kind, row['token'], parent, json.dumps(row)))
                if len(batch) >= 5000:
                    connection.executemany('INSERT INTO records VALUES(?,?,?,?)', batch)
                    batch.clear()
            if batch:
                connection.executemany('INSERT INTO records VALUES(?,?,?,?)', batch)
            connection.commit()
        connection.execute('CREATE INDEX by_parent ON records(kind,parent)')
        connection.commit()
    finally:
        connection.close()
    os.replace(temp, db_path)
    atomic_json(stamp, signature)
    return db_path


def clipped_bounds(points, width, height):
    """Clip the projected convex hull, not its bounding rectangle (not equivalent)."""
    if len(points) < 3 or not np.isfinite(points).all():
        return None
    lo, hi = points.min(axis=0), points.max(axis=0)
    if hi[0] <= 0 or hi[1] <= 0 or lo[0] >= width or lo[1] >= height:
        return None
    hull = MultiPoint(points).convex_hull.intersection(rectangle(0, 0, width, height))
    return list(hull.bounds) if not hull.is_empty and hull.area > 0 else None


def prepare_nuscenes(root, output, version='v1.0-trainval'):
    """Build complete scene shards with progress; resume after interruption.

    Official train: deterministic 90% train / 10% tuning by scene. Official val:
    held-out test. Every labeled keyframe and all six cameras are accounted for.
    Missing files are recorded and block a full-data claim unless explicitly allowed.
    """
    from nuscenes.utils.splits import create_splits_scenes
    from nuscenes.utils.data_classes import Box
    from fog_research import CLASSES
    root, output = Path(root).resolve(), Path(output).resolve()
    cache = output.parent / (output.stem + '-preparation')
    cache.mkdir(parents=True, exist_ok=True)
    with FileLock(str(cache/'prepare.lock'), timeout=0):
        db = metadata_index(root, version, cache)
        connection = sqlite3.connect(db)
        connection.execute('PRAGMA cache_size=-64000')
        def table(kind):
            return {row['token']: row for (payload,) in connection.execute(
                'SELECT payload FROM records WHERE kind=?', (kind,)) for row in [json.loads(payload)]}
        def children(kind, parent):
            return [json.loads(p) for (p,) in connection.execute(
                'SELECT payload FROM records WHERE kind=? AND parent=?', (kind, parent))]
        def get(kind, token):
            return json.loads(connection.execute('SELECT payload FROM records WHERE kind=? AND token=?',
                                                  (kind, token)).fetchone()[0])
        sensors, calibrated = table('sensor'), table('calibrated_sensor')
        categories, instances = table('category'), table('instance')
        scenes = table('scene')
        splits = create_splits_scenes()
        train_names = set(splits['mini_train' if version == 'v1.0-mini' else 'train'])
        test_names = set(splits['mini_val' if version == 'v1.0-mini' else 'val'])
        ordered = sorted([s for s in scenes.values() if s['name'] in train_names | test_names], key=lambda r: r['name'])
        report = {'source': source_signature(root, version), 'expected_scenes': len(train_names | test_names),
                  'selected_scenes': len(ordered),
                  'expected_images': 0, 'available_images': 0, 'missing_images': [], 'missing_channels': []}
        shard_paths = []
        for scene in tqdm(ordered, desc='Project scenes (resumable)', mininterval=2):
            shard = cache / f'{scene["name"]}.json'
            if shard.exists():
                data = json.loads(shard.read_text(encoding='utf-8'))
                # Recompute scenes that previously had missing images: newly downloaded
                # files become eligible on the next preparation run.
                reusable = data.get('source') == report['source'] and not data['missing_images']
                reusable = reusable and all(Path(r['image']).is_file() for r in data['rows'])
            else:
                reusable = False
            if not reusable:
                bucket = int(hashlib.sha256(scene['name'].encode()).hexdigest()[:8], 16) % 10
                split = 'test' if scene['name'] in test_names else ('val' if bucket == 0 else 'detector_train')
                data = {'rows': [], 'expected_images': 0, 'missing_images': [], 'missing_channels': [], 'source': report['source']}
                for sample in sorted(children('sample', scene['token']), key=lambda r: r['timestamp']):
                    world_boxes = []
                    for ann in children('sample_annotation', sample['token']):
                        category = categories[instances[ann['instance_token']]['category_token']]['name']
                        cls = ('person' if category.startswith('human.pedestrian.') and
                               category.rsplit('.', 1)[-1] not in ('stroller', 'wheelchair', 'personal_mobility') else
                               'bus' if category.startswith('vehicle.bus.') else
                               {'vehicle.car': 'car', 'vehicle.truck': 'truck', 'vehicle.bicycle': 'bicycle',
                                'vehicle.motorcycle': 'motorcycle'}.get(category))
                        if cls is not None:
                            box = Box(ann['translation'], ann['size'], Quaternion(ann['rotation']))
                            world_boxes.append((box.corners(), CLASSES.index(cls)))
                    available_channels = set()
                    for sd in children('sample_data', sample['token']):
                        cs = calibrated[sd['calibrated_sensor_token']]
                        channel = sensors[cs['sensor_token']]['channel']
                        if channel not in CAMERAS:
                            continue
                        available_channels.add(channel)
                        image = root / sd['filename']
                        if not image.is_file():
                            data['missing_images'].append(str(image))
                            continue
                        ego = get('ego_pose', sd['ego_pose_token'])
                        rot_ego = Quaternion(ego['rotation']).rotation_matrix.T
                        rot_cam = Quaternion(cs['rotation']).rotation_matrix.T
                        rotation = rot_cam @ rot_ego
                        translation = -rotation @ np.array(ego['translation']) - rot_cam @ np.array(cs['translation'])
                        intrinsic = np.array(cs['camera_intrinsic'])
                        boxes, labels = [], []
                        for corners, cls in world_boxes:
                            camera = rotation @ corners + translation[:, None]
                            camera = camera[:, camera[2] > 0]
                            if camera.shape[1] < 3:
                                continue
                            projected = intrinsic @ camera
                            points = (projected[:2] / projected[2]).T
                            bounds = clipped_bounds(points, sd['width'], sd['height'])
                            if bounds is not None:
                                boxes.append(bounds); labels.append(cls)
                        data['rows'].append({'id': f'nuscenes/{sd["token"]}', 'image': str(image),
                            'dataset': 'nuscenes', 'group': scene['name'], 'split': split, 'beta': None,
                            'camera': channel, 'width': sd['width'], 'height': sd['height'],
                            'boxes': boxes, 'labels': labels, 'iscrowd': [0]*len(boxes),
                            'label_policy': 'projected 3D boxes, all visibility levels; 2D proxy benchmark'})
                    data['expected_images'] += len(CAMERAS)
                    data['missing_channels'].extend(f'{sample["token"]}/{c}' for c in set(CAMERAS)-available_channels)
                atomic_json(shard, data)
            report['expected_images'] += data['expected_images']
            report['available_images'] += len(data['rows'])
            report['missing_images'].extend(data['missing_images'])
            report['missing_channels'].extend(data['missing_channels'])
            shard_paths.append(shard)
            if len(shard_paths) % 25 == 0:
                atomic_json(output.with_suffix('.coverage.json'), report)
        connection.close()
        complete = (len(ordered) == len(train_names | test_names) and not report['missing_images']
                    and not report['missing_channels'])
        report['complete'] = complete
        atomic_json(output.with_suffix('.coverage.json'), report)
        if not complete:
            raise RuntimeError(f'nuScenes coverage incomplete: {report["available_images"]}/{report["expected_images"]} images. '
                               f'See {output.with_suffix(".coverage.json")}. Download missing images, then rerun; '
                               'completed scenes resume after the remaining sensor archives are extracted.')
        temporary = output.with_suffix('.json.tmp')
        counts = Counter()
        with temporary.open('w', encoding='utf-8') as handle:
            handle.write('[')
            first = True
            for shard in shard_paths:
                for row in json.loads(shard.read_text(encoding='utf-8'))['rows']:
                    if not first:
                        handle.write(',\n')
                    handle.write(json.dumps(row)); first = False
                    counts[row['split']] += 1
            handle.write(']')
        os.replace(temporary, output)
        print(f'Prepared {report["available_images"]} images; complete={complete}; splits={dict(counts)}', flush=True)
        return output
