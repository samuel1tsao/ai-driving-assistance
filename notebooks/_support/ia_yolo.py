"""Image-Adaptive YOLO on Foggy Cityscapes.

Liu, Liu, Fan, and Zhang, AAAI 2022: a small CNN predicts the parameters of
differentiable image filters, those filters enhance the frame, and the detector
is trained from the detection loss alone. The filter set and the 15-parameter
predictor match that paper. The detector here is YOLO11n, the lightweight model
in the project proposal, rather than the paper's YOLOv3.

Filter order is defog, white balance, gamma, contrast, tone, sharpen.
Parameters are predicted from a 256x256 view and applied at detector resolution.
"""
import csv
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F


FILTER_PARAMETERS = 15
TONE_LEVELS = 8
PREDICTOR_PARAMETERS = 164943


def gaussian_kernel(size=5, sigma=1.0):
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    curve = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    curve = curve / curve.sum()
    kernel = curve[:, None] * curve[None, :]
    return kernel.view(1, 1, size, size).repeat(3, 1, 1, 1)


def decode_filter_parameters(raw):
    """Map unconstrained CNN outputs onto the paper's filter ranges.

    Zero outputs are white balance 1, gamma 1, and a flat tone curve. The
    defog, contrast, and sharpen biases are initialized below zero so a new
    predictor leaves the image nearly unchanged.
    """
    if raw.shape[-1] != FILTER_PARAMETERS:
        raise ValueError(f'Expected {FILTER_PARAMETERS} filter parameters')
    omega = torch.sigmoid(raw[:, 0:1])
    white_balance = torch.exp(torch.tanh(raw[:, 1:4]))
    gamma = torch.exp(torch.tanh(raw[:, 4:5]))
    contrast = torch.sigmoid(raw[:, 5:6])
    tone = F.softplus(raw[:, 6:14]) + 1e-3
    sharpen = F.softplus(raw[:, 14:15])
    return omega, white_balance, gamma, contrast, tone, sharpen


def atmospheric_light(images, pixels=1000):
    """Average the RGB values at the haziest dark-channel locations.

    This estimate is not a learned parameter. Gradients flow through the
    transmission formula, not through the pixel selection.
    """
    with torch.no_grad():
        dark = images.min(dim=1).values
        local = -F.max_pool2d(-dark[:, None], 15, 1, 7)[:, 0]
        flat = local.flatten(1)
        count = min(pixels, flat.shape[1])
        index = flat.topk(count, dim=1).indices
        colors = images.flatten(2).gather(2, index[:, None].expand(-1, 3, -1))
        return colors.mean(dim=2).view(-1, 3, 1, 1)


def defog(images, omega):
    airlight = atmospheric_light(images).clamp(min=0.1)
    normalized = images / airlight
    dark = normalized.min(dim=1, keepdim=True).values
    local = -F.max_pool2d(-dark, 15, 1, 7)
    transmission = (1 - omega.view(-1, 1, 1, 1) * local).clamp(0.1, 1)
    recovered = (images - airlight) / transmission + airlight
    return recovered.clamp(0, 1)


def contrast_filter(images, alpha):
    red, green, blue = images[:, 0:1], images[:, 1:2], images[:, 2:3]
    luminance = (0.27 * red + 0.67 * green + 0.06 * blue).clamp(0, 1)
    enhanced_luminance = 0.5 * (1 - torch.cos(torch.pi * luminance))
    enhanced = images * enhanced_luminance / luminance.clamp(min=1e-4)
    mix = alpha.view(-1, 1, 1, 1)
    return mix * enhanced + (1 - mix) * images


def tone_filter(images, tone):
    """Monotonic piecewise-linear curve from Hu et al., shared across channels."""
    total = tone.sum(dim=1).view(-1, 1, 1, 1).clamp(min=1e-4)
    mapped = images.new_zeros(images.shape)
    source = images.clamp(0, 1) * TONE_LEVELS
    for index in range(TONE_LEVELS):
        weight = tone[:, index].view(-1, 1, 1, 1)
        mapped = mapped + (source - index).clamp(0, 1) * weight
    return mapped / total


def sharpen_filter(images, amount, blur):
    smooth = F.conv2d(images, blur, padding=blur.shape[-1] // 2, groups=3)
    return images + amount.view(-1, 1, 1, 1) * (images - smooth)


def apply_filters(images, raw, blur):
    omega, white_balance, gamma, contrast, tone, sharpen = decode_filter_parameters(raw)
    filtered = defog(images, omega)
    filtered = (filtered * white_balance.view(-1, 3, 1, 1)).clamp(0, 1)
    filtered = filtered.clamp(min=1e-4).pow(gamma.view(-1, 1, 1, 1))
    filtered = contrast_filter(filtered, contrast).clamp(0, 1)
    filtered = tone_filter(filtered, tone).clamp(0, 1)
    return sharpen_filter(filtered, sharpen, blur).clamp(0, 1)


class ParameterPredictor(nn.Module):
    """Five stride-2 convolutions and two linear layers, about 165K weights."""

    def __init__(self):
        super().__init__()
        channels = (3, 16, 32, 32, 32, 32)
        blocks = []
        for incoming, outgoing in zip(channels, channels[1:]):
            blocks.append(nn.Conv2d(incoming, outgoing, 3, stride=2, padding=1))
            blocks.append(nn.LeakyReLU(0.1, inplace=True))
        self.convolution = nn.Sequential(*blocks)
        self.hidden = nn.Linear(32 * 8 * 8, 64)
        self.output = nn.Linear(64, FILTER_PARAMETERS)
        nn.init.zeros_(self.output.weight)
        bias = torch.zeros(FILTER_PARAMETERS)
        bias[0] = -4
        bias[5] = -4
        bias[14] = -4
        self.output.bias.data.copy_(bias)

    def forward(self, images):
        preview = F.interpolate(images, (256, 256), mode='bilinear', align_corners=False)
        features = self.convolution(preview).flatten(1)
        return self.output(F.leaky_relu(self.hidden(features), 0.1))


class ImageAdaptive(nn.Module):
    def __init__(self):
        super().__init__()
        self.predictor = ParameterPredictor()
        self.register_buffer('blur', gaussian_kernel(), persistent=False)

    def forward(self, images):
        return apply_filters(images, self.predictor(images), self.blur)


def default_fog_root(project_root):
    project_root = Path(project_root)
    candidates = [
        project_root / 'fog-colab-data/.cache/kagglehub/datasets/minhducnguyen9705/foggy-cityscapes/versions/1',
        project_root / '.cache/kagglehub/datasets/minhducnguyen9705/foggy-cityscapes/versions/1',
    ]
    for path in candidates:
        if (path / 'leftImg8bit_foggy').is_dir():
            return path.resolve()
    from fog_data import resolve_foggy_images
    found = resolve_foggy_images(project_root)
    return None if found is None else Path(found).resolve()


def image_root_of(fog_root):
    fog_root = Path(fog_root)
    direct = fog_root / 'leftImg8bit_foggy'
    if direct.is_dir():
        return direct
    found = [path for path in fog_root.rglob('leftImg8bit_foggy') if path.is_dir()]
    if len(found) != 1:
        raise FileNotFoundError(f'Expected one leftImg8bit_foggy directory under {fog_root}')
    return found[0]


def normalize_gtfine(path):
    path = Path(path)
    if (path / 'train').is_dir():
        return path
    nested = path / 'gtFine'
    if (nested / 'train').is_dir():
        return nested
    return path


def resolve_gtfine(fog_root, explicit=None):
    if explicit:
        path = normalize_gtfine(explicit)
        if not path.is_dir():
            raise FileNotFoundError(path)
        return path.resolve()
    sibling = Path(fog_root) / 'gtFine'
    if sibling.is_dir():
        return normalize_gtfine(sibling).resolve()
    return None


def gtfine_instructions(fog_root):
    example = Path(fog_root) / 'gtFine' / 'train' / 'aachen' / (
        'aachen_000000_000019_gtFine_instanceIds.png')
    return (
        'Foggy Cityscapes images are present, but this Kaggle copy has no bounding boxes.\n'
        'Download gtFine_trainvaltest.zip from https://www.cityscapes-dataset.com/downloads/\n'
        '(free academic registration; this is the label pack, not the 11 GB image pack).\n'
        f'Unzip it so this file exists:\n  {example}\n'
        'Then run prepare again. Do not re-download the foggy images.'
    )


def _boxes_for_mask(mask_path, label_of):
    with Image.open(mask_path) as source:
        mask = np.array(source)
    boxes, labels, crowds = [], [], []
    for instance in np.unique(mask):
        if int(instance) == 0:
            continue
        category = int(instance // 1000 if instance >= 1000 else instance)
        if category not in label_of:
            continue
        rows, columns = np.where(mask == instance)
        boxes.append([int(columns.min()), int(rows.min()),
                      int(columns.max()) + 1, int(rows.max()) + 1])
        labels.append(label_of[category])
        crowds.append(int(instance < 1000))
    return boxes, labels, crowds


def build_detection_manifest(fog_root, gtfine, output):
    """Official train scenes are split 80/20 into train and tune. Official val is the test."""
    from fog_prepare import atomic_json
    from fog_research import CLASSES
    from tqdm.auto import tqdm

    fog_root, gtfine, output = Path(fog_root), Path(gtfine), Path(output)
    images = image_root_of(fog_root)
    label_of = {24: CLASSES.index('person'), 25: CLASSES.index('person'),
                26: CLASSES.index('car'), 27: CLASSES.index('truck'),
                28: CLASSES.index('bus'), 31: CLASSES.index('train'),
                32: CLASSES.index('motorcycle'), 33: CLASSES.index('bicycle')}
    rows, cache = [], {}
    for official in ('train', 'val'):
        paths = sorted((images / official).glob('*/*_leftImg8bit_foggy_beta_*.png'))
        if not paths:
            raise FileNotFoundError(f'No foggy images in {images / official}')
        for image in tqdm(paths, desc=f'Foggy Cityscapes {official}', mininterval=2):
            marker = '_leftImg8bit_foggy_beta_'
            base, suffix = image.name.split(marker)
            beta = float(suffix.removesuffix('.png'))
            mask_path = gtfine / official / image.parent.name / f'{base}_gtFine_instanceIds.png'
            if base not in cache:
                if not mask_path.is_file():
                    raise FileNotFoundError(
                        f'Missing instance mask: {mask_path}\n{gtfine_instructions(fog_root)}')
                with Image.open(image) as source:
                    width, height = source.size
                cache[base] = (*_boxes_for_mask(mask_path, label_of), width, height, official)
            boxes, labels, crowds, width, height, split_name = cache[base]
            group = '_'.join(base.split('_')[:2])
            if split_name == 'val':
                assigned = 'test'
            else:
                bucket = int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) % 10
                assigned = 'val' if bucket < 2 else 'train'
            rows.append({
                'id': f'city/{base}/{beta}', 'image': image.relative_to(fog_root).as_posix(),
                'dataset': 'foggy_cityscapes', 'group': group, 'split': assigned,
                'beta': beta, 'boxes': boxes, 'labels': labels, 'iscrowd': crowds,
                'width': width, 'height': height, 'official_split': split_name,
            })
    if not rows or not any(row['boxes'] for row in rows):
        raise ValueError('No detection boxes were read from gtFine instance masks')
    counts = {}
    for row in rows:
        counts[row['split']] = counts.get(row['split'], 0) + 1
    betas = sorted({row['beta'] for row in rows})
    print(f'Scenes: {counts}; fog betas {betas}; '
          f'mean boxes {sum(len(row["boxes"]) for row in rows) / len(rows):.2f}', flush=True)
    payload = {'fog_root': str(fog_root.resolve()), 'gtfine': str(gtfine.resolve()),
               'rows': rows}
    atomic_json(output, payload)
    return output


def load_detection_manifest(path):
    payload = json.loads(Path(path).read_text(encoding='utf-8'))
    root = Path(payload['fog_root'])
    rows = []
    for row in payload['rows']:
        image = Path(row['image'])
        if not image.is_file():
            image = root / row['image']
        if not image.is_file():
            raise FileNotFoundError(image)
        rows.append({**row, 'image': str(image.resolve())})
    return rows


def _write_table(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, restval='')
        writer.writeheader()
        writer.writerows(rows)


MODES = ('adaptive', 'baseline', 'film-beta', 'film-zero')


def condition_beta(rows, images, mode):
    """Fog value passed into FiLM. The zero arm keeps the same adapter and withholds beta."""
    if mode == 'film-zero':
        return images.new_zeros((len(rows), 1))
    if mode == 'film-beta':
        return images.new_tensor([[float(row['beta'])] for row in rows])
    raise ValueError(mode)


def apply_adapter(adapter, images, rows, mode):
    if adapter is None or mode == 'baseline':
        return images
    if mode in ('film-beta', 'film-zero'):
        return adapter(images, condition_beta(rows, images, mode))
    return adapter(images)


def _finite_grads(parameters):
    grads = [parameter.grad for parameter in parameters if parameter.grad is not None]
    return bool(grads) and all(torch.isfinite(grad).all() for grad in grads)


def train_detector(rows, output, mode='adaptive', epochs=30, image_size=640,
                   batch_size=32, seed=42, device='cuda', limit=None, fresh=False):
    """Fine-tune YOLO11n, with or without the image-adaptive filters.

    Mixed precision is bfloat16 with no gradient scaler. Float16 scaling is what
    drove the earlier YOLO11m run to a zero scale and then to NaN weights.
    """
    from fog_research import Detector, iter_image_tensors, seed_all
    from tqdm.auto import tqdm

    if mode not in MODES:
        raise ValueError(f'mode must be one of {MODES}')
    if epochs < 1 or batch_size < 1 or image_size < 32:
        raise ValueError('epochs, batch size, and image size must be positive')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    by_split = {name: [row for row in rows if row['split'] == name]
                for name in ('train', 'val', 'test')}
    if not by_split['train'] or not by_split['val'] or not by_split['test']:
        raise ValueError('Manifest needs train, val, and test rows')
    if limit:
        for name in by_split:
            by_split[name] = by_split[name][:limit]
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA is not available. This trainer is set up for the RTX 5070.')
    torch.set_num_threads(1)
    if device.startswith('cuda'):
        torch.backends.cudnn.benchmark = True
    seed_all(seed)
    checkpoint = 'yolo11n.pt'
    detector = Detector('YOLO11n', checkpoint, device, trainable=True)
    if mode == 'adaptive':
        adapter = ImageAdaptive().to(device)
    elif mode in ('film-beta', 'film-zero'):
        from fog_research import FogAdapter
        adapter = FogAdapter().to(device)
    else:
        adapter = None
    yolo_parameters = [parameter for parameter in detector.model.parameters()
                       if parameter.requires_grad]
    groups = [{'params': yolo_parameters, 'lr': 1e-4}]
    if adapter is not None:
        groups.append({'params': adapter.parameters(), 'lr': 1e-3})
    optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
    last_path, best_path = output / 'last.pt', output / 'best.pt'
    start_epoch, best_loss = 0, float('inf')
    if fresh:
        last_path.unlink(missing_ok=True)
        best_path.unlink(missing_ok=True)
    elif last_path.is_file():
        saved = torch.load(last_path, map_location=device, weights_only=True)
        if saved.get('mode') != mode or saved.get('image_size') != image_size:
            raise ValueError(f'{last_path} was saved for a different mode or image size')
        detector.model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        start_epoch = int(saved['epoch'])
        best_loss = float(saved['best_loss'])
        if adapter is not None:
            adapter.load_state_dict(saved['adapter'])
        print(f'Resume {mode} at epoch {start_epoch + 1}', flush=True)
    use_amp = device.startswith('cuda')
    run = {'mode': mode, 'precision': 'bfloat16' if use_amp else 'float32',
           'gradient_scaler': False, 'epochs': epochs, 'image_size': image_size,
           'batch_size': batch_size, 'seed': seed, 'train_images': len(by_split['train']),
           'val_images': len(by_split['val']), 'test_images': len(by_split['test']),
           'predictor_parameters': (sum(p.numel() for p in adapter.parameters())
                                    if adapter is not None else 0),
           'conditioning': {'film-beta': 'oracle beta', 'film-zero': 'zero beta',
                            'adaptive': 'image filters', 'baseline': 'none'}[mode]}
    if device.startswith('cuda'):
        run['gpu'] = torch.cuda.get_device_name(0)
    (output / 'run.json').write_text(json.dumps(run, indent=2), encoding='utf-8')
    print(json.dumps(run), flush=True)

    batch_limit = [batch_size]

    def batched(split_rows):
        group_rows, group_images = [], []
        for row, image in iter_image_tensors(split_rows, image_size, device):
            group_rows.append(row)
            group_images.append(image)
            if len(group_rows) == batch_limit[0]:
                yield group_rows, torch.cat(group_images, 0)
                group_rows, group_images = [], []
        if group_rows:
            yield group_rows, torch.cat(group_images, 0)

    def loss_of(split_rows, images):
        enhanced = apply_adapter(adapter, images, split_rows, mode)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_amp):
            return detector.loss_many(enhanced, split_rows) / len(split_rows)

    parameters = yolo_parameters + (list(adapter.parameters()) if adapter is not None else [])
    history = []
    if (output / 'history.csv').is_file() and start_epoch and not fresh:
        with (output / 'history.csv').open(newline='', encoding='utf-8') as handle:
            history = list(csv.DictReader(handle))

    def snapshot(epoch, reason):
        payload = {'mode': mode, 'image_size': image_size, 'epoch': epoch,
                   'best_loss': best_loss, 'reason': reason,
                   'model': detector.model.state_dict(),
                   'optimizer': optimizer.state_dict()}
        if adapter is not None:
            payload['adapter'] = adapter.state_dict()
        torch.save(payload, last_path)
        if reason == 'best':
            torch.save(payload, best_path)

    for epoch in range(start_epoch, epochs):
        detector.training_mode()
        if adapter is not None:
            adapter.train()
        order = random.Random(seed + epoch).sample(by_split['train'], len(by_split['train']))
        seen = skipped = 0
        total = 0.0
        progress = tqdm(total=len(order), desc=f'{mode} epoch {epoch + 1}/{epochs}', mininterval=2)

        def step(split_rows, images):
            nonlocal seen, skipped, total
            optimizer.zero_grad(set_to_none=True)
            loss = loss_of(split_rows, images)
            if not torch.isfinite(loss.detach()):
                skipped += len(split_rows)
                progress.update(len(split_rows))
                return
            loss.backward()
            if not _finite_grads(parameters):
                skipped += len(split_rows)
                progress.update(len(split_rows))
                return
            torch.nn.utils.clip_grad_norm_(parameters, 10)
            optimizer.step()
            total += float(loss.detach()) * len(split_rows)
            seen += len(split_rows)
            progress.update(len(split_rows))

        def rerun(split_rows):
            if not split_rows:
                return
            images = None
            try:
                from fog_research import image_tensor
                images = torch.cat([image_tensor(row, image_size, device) for row in split_rows])
                step(split_rows, images)
            except torch.cuda.OutOfMemoryError:
                if images is not None:
                    del images
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                if len(split_rows) == 1:
                    raise
                batch_limit[0] = max(1, len(split_rows) // 2)
                midpoint = len(split_rows) // 2
                print(f'CUDA memory full; batch size is now {batch_limit[0]}', flush=True)
                rerun(split_rows[:midpoint])
                rerun(split_rows[midpoint:])
            else:
                del images

        try:
            for split_rows, images in batched(order):
                try:
                    step(split_rows, images)
                except torch.cuda.OutOfMemoryError:
                    del images
                    optimizer.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
                    if len(split_rows) == 1:
                        raise
                    batch_limit[0] = max(1, len(split_rows) // 2)
                    print(f'CUDA memory full; batch size is now {batch_limit[0]}', flush=True)
                    rerun(split_rows)
                else:
                    del images
        except KeyboardInterrupt:
            snapshot(epoch, 'interrupt')
            raise
        finally:
            progress.close()
        if not seen:
            raise RuntimeError('Every training batch was nonfinite or empty')
        detector.training_mode()
        if adapter is not None:
            adapter.eval()
        validation = count = 0.0
        with torch.no_grad():
            for split_rows, images in batched(by_split['val']):
                validation += float(loss_of(split_rows, images).detach()) * len(split_rows)
                count += len(split_rows)
        train_loss = total / seen
        val_loss = validation / count
        record = {'epoch': epoch + 1, 'train_loss': train_loss, 'val_loss': val_loss,
                  'skipped': skipped, 'batch_size': batch_limit[0]}
        if mode == 'adaptive':
            with torch.no_grad():
                sample_rows = by_split['val'][:batch_limit[0]]
                sample = next(batched(sample_rows))[1]
                omega = torch.sigmoid(adapter.predictor(sample)[:, 0])
            record['omega_mean'] = float(omega.mean())
            record['omega_std'] = float(omega.std(unbiased=False))
        history.append(record)
        _write_table(output / 'history.csv', history)
        print(json.dumps(record), flush=True)
        reason = 'epoch'
        if np.isfinite(val_loss) and val_loss < best_loss:
            best_loss = float(val_loss)
            reason = 'best'
        snapshot(epoch + 1, reason)
    saved = torch.load(best_path, map_location=device, weights_only=True)
    detector.model.load_state_dict(saved['model'])
    if adapter is not None:
        adapter.load_state_dict(saved['adapter'])
    return detector, adapter


def evaluate_detector(detector, adapter, rows, device, image_size, output, repeats=3, mode='adaptive'):
    from fog_research import coco_metrics, image_tensor

    if not rows or repeats < 1:
        raise ValueError('Evaluation needs images and at least one timing repeat')
    output = Path(output)
    detector.model.eval()
    if adapter is not None:
        adapter.eval()

    def sync():
        if str(device).startswith('cuda'):
            torch.cuda.synchronize()

    @torch.inference_mode()
    def once(row):
        sync()
        started = time.perf_counter()
        image = image_tensor(row, image_size, device)
        sync()
        prepared = time.perf_counter()
        enhanced = apply_adapter(adapter, image, [row], mode)
        sync()
        adapted = time.perf_counter()
        prediction = detector.predict(enhanced, row)
        sync()
        finished = time.perf_counter()
        return prediction, {'total_ms': (finished - started) * 1000,
                            'adapt_ms': (adapted - prepared) * 1000}

    for _ in range(3):
        once(rows[0])
    results = []
    for row in rows:
        timings = [once(row)[1] for _ in range(repeats - 1)]
        prediction, timing = once(row)
        timings.append(timing)
        results.append({'id': row['id'], 'prediction': prediction, 'timings': timings})
    groups = [('all', list(range(len(rows))))]
    for beta in sorted({row['beta'] for row in rows}):
        groups.append((f'{beta:g}', [index for index, row in enumerate(rows) if row['beta'] == beta]))
    table = []
    for name, indices in groups:
        subset = [rows[index] for index in indices]
        paired = [results[index] for index in indices]
        metrics = coco_metrics(subset, paired)
        totals = [item['total_ms'] for result in paired for item in result['timings']]
        adapt = [item['adapt_ms'] for result in paired for item in result['timings']]
        table.append({'fog_beta': name, 'images': len(subset), **metrics,
                      'mean_ms': float(np.mean(totals)),
                      'p95_ms': float(np.percentile(totals, 95)),
                      'adapt_ms': float(np.mean(adapt))})
    _write_table(output / 'results.csv', table)
    if mode == 'adaptive':
        save_filter_previews(adapter, rows, device, image_size, output / 'previews')
    elif mode == 'film-beta':
        save_film_previews(adapter, rows, device, image_size, output / 'previews')
    return table


def save_filter_previews(adapter, rows, device, image_size, directory):
    from fog_research import image_tensor

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    adapter.eval()
    saved = set()
    with torch.inference_mode():
        for row in rows:
            beta = f'{row["beta"]:g}'
            if beta in saved:
                continue
            original = image_tensor(row, image_size, device)
            enhanced = adapter(original)
            pair = torch.cat([original, enhanced], dim=3)[0].clamp(0, 1)
            array = (pair.detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            Image.fromarray(array).save(directory / f'beta-{beta}.png')
            saved.add(beta)
            if len(saved) >= 6:
                break


def save_film_previews(adapter, rows, device, image_size, directory):
    """Original frame beside the same frame after FiLM uses that frame's beta."""
    from fog_research import image_tensor

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    adapter.eval()
    saved = set()
    with torch.inference_mode():
        for row in rows:
            beta = f'{row["beta"]:g}'
            if beta in saved:
                continue
            original = image_tensor(row, image_size, device)
            enhanced = apply_adapter(adapter, original, [row], 'film-beta')
            pair = torch.cat([original, enhanced], dim=3)[0].clamp(0, 1)
            array = (pair.detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            Image.fromarray(array).save(directory / f'beta-{beta}.png')
            saved.add(beta)
            if len(saved) >= 6:
                break
