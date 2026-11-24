"""Dataset entry points; nuScenes preparation lives in fog_prepare."""
import hashlib
from pathlib import Path
import numpy as np
from PIL import Image
from fog_research import CLASSES


def resolve_foggy_images(project_root, explicit=None, download=False):
    """Reuse Windows/Linux Kaggle caches from the user's downloader; optional same-source download."""
    import os
    project_root = Path(project_root).resolve()
    candidates = [Path(explicit)] if explicit else []
    candidates += [project_root/'data/fog/cityscapes',
                   Path.home()/'Downloads'/'foggy-cityscapes',
                   Path.home()/'Downloads'/'Foggy Cityscapes',
                   project_root/'.cache/kagglehub/datasets/minhducnguyen9705/foggy-cityscapes/versions/1',
                   Path.home()/'.cache/kagglehub/datasets/minhducnguyen9705/foggy-cityscapes/versions/1']
    for path in candidates:
        if path.is_dir() and ((path/'leftImg8bit_foggy').is_dir() or (path/'train').is_dir()
                              or next(path.rglob('leftImg8bit_foggy'), None) is not None):
            return path
    if not download:
        return None
    # Keep downloads inside this workspace. Version 1 matches Untitled2.ipynb.
    os.environ['KAGGLEHUB_CACHE'] = str(project_root/'.cache/kagglehub')
    try:
        import kagglehub
    except ImportError as error:
        raise ImportError('Install requirements-fog-research.txt to enable the optional Kaggle download') from error
    print('Downloading the same Foggy Cityscapes v1 source as Untitled2.ipynb (~29.1 GB archive plus extraction).', flush=True)
    return Path(kagglehub.dataset_download('minhducnguyen9705/foggy-cityscapes/versions/1'))


def cityscapes_manifest(root, output, with_detection_labels=False):
    """Kaggle/official images suffice for density supervision; gtFine is optional.

    Rows without gtFine are explicitly density-only, never negative detection images.
    Set with_detection_labels=True for the optional Foggy Cityscapes detection benchmark.
    """
    root, output = Path(root).resolve(), Path(output)
    if output.exists():
        raise FileExistsError(f'{output} exists. Reuse it or choose a new output path.')
    city_ids = {24:'person', 25:'person', 26:'car', 27:'truck', 28:'bus', 31:'train', 32:'motorcycle', 33:'bicycle'}
    rows = []
    from tqdm.auto import tqdm
    # Kaggle extracts may nest leftImg8bit_foggy below additional directories.
    image_roots = ([root/'leftImg8bit_foggy'] if (root/'leftImg8bit_foggy').is_dir() else
                   ([root] if (root/'train').is_dir() else list(root.rglob('leftImg8bit_foggy'))))
    if len(image_roots) != 1:
        raise ValueError(f'Expected one Foggy Cityscapes image root under {root}; found {len(image_roots)}')
    image_root = image_roots[0]
    for split in ('train','val'):
        paths = sorted((image_root/split).glob('*/*_leftImg8bit_foggy_beta_*.png'))
        cache = {}
        for image in tqdm(paths, desc=f'Foggy Cityscapes {split}', mininterval=2):
            base, suffix = image.name.split('_leftImg8bit_foggy_beta_')
            beta = float(suffix.removesuffix('.png'))
            mask_path = root/'gtFine'/split/image.parent.name/f'{base}_gtFine_instanceIds.png'
            if base not in cache:
                if with_detection_labels and not mask_path.is_file():
                    raise FileNotFoundError(f'Missing instance labels: {mask_path}')
                boxes, labels, crowds = [], [], []
                with Image.open(image) as source:
                    width, height = source.size
                if with_detection_labels:
                    with Image.open(mask_path) as source:
                        mask = np.array(source)
                    for instance in np.unique(mask):
                        category = int(instance//1000 if instance >= 1000 else instance)
                        if category in city_ids:
                            yy, xx = np.where(mask == instance)
                            boxes.append([int(xx.min()),int(yy.min()),int(xx.max())+1,int(yy.max())+1])
                            labels.append(CLASSES.index(city_ids[category])); crowds.append(int(instance < 1000))
                cache[base] = boxes, labels, crowds, width, height
            boxes, labels, crowds, width, height = cache[base]
            group = '_'.join(base.split('_')[:2])
            bucket = int(hashlib.sha256(group.encode()).hexdigest()[:8],16) % 10
            if with_detection_labels:
                assigned = 'test' if split == 'val' else ('density_train' if bucket < 2 else 'val' if bucket < 4 else 'detector_train')
            else:
                assigned = 'test' if split == 'val' else ('val' if bucket < 2 else 'density_train')
            rows.append({'id':f'city/{base}/{beta}', 'image':str(image),'dataset':'foggy_cityscapes',
                         'group':group,'split':assigned,'beta':beta,'boxes':boxes,'labels':labels,'iscrowd':crowds,
                         'width':width,'height':height,'annotation_source':str(mask_path) if with_detection_labels else None,
                         'detection_labeled':with_detection_labels,
                         'label_policy':'instance-mask bounding rectangles; rider merged into person' if with_detection_labels else 'beta-only; not a detection target'})
    if not rows:
        raise FileNotFoundError(f'No official Foggy Cityscapes images under {root}/leftImg8bit_foggy')
    seen = set()
    for row in list(rows):
        base = row['id'].split('/')[1]
        split = 'val' if row['split'] == 'test' else 'train'
        clear = root/'leftImg8bit'/split/base.split('_')[0]/f'{base}_leftImg8bit.png'
        if clear.is_file() and base not in seen:
            rows.append({**row,'id':f'city/{base}/clear','image':str(clear),'beta':0.0})
            seen.add(base)
    from fog_prepare import atomic_json
    atomic_json(output, rows)
    return output


def nuscenes_manifest(root, output, version='v1.0-trainval'):
    from fog_prepare import prepare_nuscenes
    return prepare_nuscenes(root, output, version=version)
