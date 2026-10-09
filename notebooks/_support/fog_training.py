"""Full detector fine-tuning, resumable checkpoints, preflight, and persistent evaluation."""
import gc
import hashlib
import importlib.metadata
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from filelock import FileLock
from tqdm.auto import tqdm

from fog_prepare import atomic_json
from fog_research import (BETA_SCALE, CLASSES, DensityCNN, Detector, FogAdapter, beta_for,
                          digest, image_tensor, seed_all, summarize)

ENGINE_VERSION = 4


def default_config(root):
    root = Path(root).resolve()
    return {'output': str(root/'reports/fog/full-training-complete'),
            'manifest': str(root/'data/fog/nuscenes_all.json'),
            'city_manifest': str(root/'data/fog/cityscapes.json'),
            'density_checkpoint': None,
            'checkpoints': {'YOLO11n': str(root/'yolo11n.pt'), 'YOLO11m': str(root/'yolo11m.pt'),
                            'DETR': 'facebook/detr-resnet-50'},
            'arms': ['rgb_control', 'predicted_density'], 'seeds': [42, 43, 44],
            'device': 'cuda:0' if torch.cuda.is_available() else 'cpu',
            'size': 640, 'epochs': 30, 'density_epochs': 15,
            'accumulate': 4, 'save_every': 200, 'repeats': 1,
            'learning_rates': {'YOLO11n': 1e-4, 'YOLO11m': 1e-4, 'DETR': 1e-5},
            'adapter_lr': 1e-3, 'amp': True}


def fingerprint(rows):
    # Annotation/identity fingerprint plus image stat records; does not re-read every JPEG.
    value = hashlib.sha256()
    for row in sorted(rows, key=lambda r: r['id']):
        stat = Path(row['image']).stat()
        value.update(json.dumps([row, stat.st_size, stat.st_mtime_ns], sort_keys=True).encode())
    return value.hexdigest()


def partitions(rows):
    parts = {name: [r for r in rows if r['split'] == name and
                   (name == 'density_train' or r.get('detection_labeled', True)) and
                   (name not in ('detector_train', 'val') or not any(r.get('iscrowd', [])))]
             for name in ('detector_train', 'density_train', 'val', 'test')}
    parts['density_val'] = [r for r in rows if r['split'] == 'val' and r['beta'] is not None]
    parts['density_test'] = [r for r in rows if r['split'] == 'test' and r['beta'] is not None]
    return parts


def preflight(config, rows):
    """Return actionable checks without starting training or downloading anything."""
    parts = partitions(rows)
    checks = []
    def check(name, good, detail):
        checks.append({'check': name, 'ready': bool(good), 'detail': detail})
    check('training images', bool(parts['detector_train']), f'{len(parts["detector_train"])} labeled images')
    check('tuning images', bool(parts['val']), f'{len(parts["val"])} scene-held-out images for checkpoint selection')
    check('test images', bool(parts['test']), f'{len(parts["test"])} held-out images for final metrics')
    valid_arms = {'rgb_control', 'predicted_density', 'oracle_density'}
    check('arms', bool(config['arms']) and len(set(config['arms'])) == len(config['arms']) and
          set(config['arms']) <= valid_arms, ', '.join(config['arms']))
    groups = {}
    clean = True
    ids = set()
    for row in rows:
        key = (row['dataset'], row['group'])
        clean &= key not in groups or groups[key] == row['split']
        clean &= row['id'] not in ids
        groups[key] = row['split']; ids.add(row['id'])
    check('scene/ID isolation', clean, 'No scene crosses train, tuning, density-training or final-test splits')
    for key in ('size', 'epochs', 'density_epochs', 'accumulate', 'save_every', 'repeats'):
        check(key, isinstance(config[key], int) and config[key] > 0, str(config[key]))
    check('size alignment', config['size'] % 32 == 0, 'Image size must be divisible by 32')
    check('seeds', bool(config['seeds']) and len(set(config['seeds'])) == len(config['seeds']), str(config['seeds']))
    coverage_path = Path(config['manifest']).with_suffix('.coverage.json')
    if any(r['dataset'] == 'nuscenes' for r in rows):
        coverage = json.loads(coverage_path.read_text()) if coverage_path.exists() else {}
        complete = coverage.get('complete') is True
        check('nuScenes coverage', complete,
              f'complete={complete}; coverage report: {coverage_path}. All expected camera keyframes are required.')
    if 'predicted_density' in config['arms']:
        source = config.get('density_checkpoint')
        if source:
            try:
                saved = torch.load(source, map_location='cpu', weights_only=True)
                correct = saved.get('kind') == 'fog-density-v1' and saved.get('beta_scale') == BETA_SCALE
                trained_groups = {tuple(x) for x in saved.get('train_groups', [])}
                held_out = {(r['dataset'], r['group']) for r in rows if r['split'] in ('val', 'test', 'detector_train')}
                correct &= bool(trained_groups) and not (trained_groups & held_out)
                check('density checkpoint', correct, 'Requires this pipeline format, beta units, and independent training groups')
            except Exception as error:
                check('density checkpoint', False, f'{type(error).__name__}: {error}')
        else:
            train = parts['density_train']
            known_val = parts['density_val']
            check('density supervision', bool(train) and all(r['beta'] is not None for r in train)
                  and len({r['beta'] for r in train}) > 1 and bool(known_val),
                  'Add Foggy Cityscapes images (beta comes from filenames; gtFine is not required), prepare its manifest, '
                  'or set density_checkpoint to an independently trained fog-density checkpoint. '
                  'Native nuScenes has no beta labels; its visibility field is NOT fog density.')
    if 'oracle_density' in config['arms']:
        check('oracle labels', all(r['beta'] is not None for r in parts['detector_train']+parts['val']+parts['test']),
              'Oracle is optional and requires known beta for every matched train/validation/test image. '
              'Use the default rgb_control/predicted_density comparison for native nuScenes.')
    check('models', bool(config['checkpoints']) and set(config['checkpoints']) <= {'YOLO11n', 'YOLO11m', 'DETR'},
          'Three supported pretrained architectures; missing public weights download on first load')
    check('device', not config['device'].startswith('cuda') or torch.cuda.is_available(), config['device'])
    return pd.DataFrame(checks)


def save_torch(path, payload):
    temporary = path.with_suffix('.tmp')
    torch.save(payload, temporary)
    os.replace(temporary, path)


def fit_density(parts, config, directory, seed):
    """Train or resume a scalar estimator; select the lowest tuning loss checkpoint."""
    directory.mkdir(parents=True, exist_ok=True)
    seed_all(seed)
    model = DensityCNN().to(config['device'])
    source = config.get('density_checkpoint')
    if source:
        saved = torch.load(source, map_location=config['device'], weights_only=True)
        model.load_state_dict(saved['model'])
        return model.eval().requires_grad_(False)
    train = parts['density_train']
    val = parts['density_val']
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    last, best = directory/'last.pt', directory/'best.pt'
    start_epoch, start_step, best_loss = 0, 0, float('inf')
    running_sum, running_count, history = 0., 0, []
    if last.exists():
        state = torch.load(last, map_location=config['device'], weights_only=True)
        model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
        start_epoch, start_step, best_loss = state['epoch'], state['step'], state['best_loss']
        running_sum, running_count, history = state['running_sum'], state['running_count'], state['history']
    def save(epoch, step, improved=False):
        payload = {'kind': 'fog-density-v1', 'beta_scale': BETA_SCALE, 'model': model.state_dict(),
                   'optimizer': optimizer.state_dict(), 'epoch': epoch, 'step': step, 'best_loss': best_loss,
                   'running_sum': running_sum, 'running_count': running_count, 'history': history,
                   'train_groups': sorted({(r['dataset'], r['group']) for r in train})}
        if improved:
            save_torch(best, payload)
        save_torch(last, payload)
        return payload
    for epoch in range(start_epoch, config['density_epochs']):
        order = random.Random(seed+epoch).sample(range(len(train)), len(train))
        model.train()
        for step in tqdm(range(start_step, len(order)), total=len(order), initial=start_step,
                         desc=f'Density seed={seed} epoch={epoch+1}', mininterval=2):
            row = train[order[step]]
            x = image_tensor(row, config['size'], config['device'])
            target = x.new_tensor([[row['beta']]])
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(model(x)/BETA_SCALE, target/BETA_SCALE)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite density loss')
            loss.backward(); optimizer.step()
            running_sum += float(loss.detach()); running_count += 1
            if (step+1) % config['save_every'] == 0:
                save(epoch, step+1)
        model.eval()
        errors = []
        with torch.inference_mode():
            for row in tqdm(val, desc='Density tuning', mininterval=2):
                prediction = model(image_tensor(row, config['size'], config['device'])).item()
                errors.append((prediction-row['beta'])**2)
        val_loss = float(np.mean(errors))
        history.append({'epoch': epoch+1, 'train_normalized_mse': running_sum/running_count,
                        'val_beta_rmse': val_loss**.5})
        improved = val_loss < best_loss
        best_loss = min(best_loss, val_loss)
        running_sum, running_count, start_step = 0., 0, 0
        save(epoch+1, 0, improved)
        pd.DataFrame(history).to_csv(directory/'history.csv', index=False)
    model.load_state_dict(torch.load(best, map_location=config['device'], weights_only=True)['model'])
    return model.eval().requires_grad_(False)


def fit_detector(detector, adapter, density, parts, arm, config, directory, seed):
    """Train detector AND adapter, with atomic optimizer-boundary resume snapshots."""
    train, val = parts['detector_train'], parts['val']
    parameters = [p for p in detector.model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([{'params': parameters, 'lr': config['learning_rates'][detector.name]},
                                   {'params': adapter.parameters(), 'lr': config['adapter_lr']}], weight_decay=1e-4)
    use_amp = config['amp'] and config['device'].startswith('cuda')
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp, init_scale=128.)
    last, best = directory/'last.pt', directory/'best.pt'
    start_epoch, start_step, best_loss = 0, 0, float('inf')
    running_sum, running_count, history = 0., 0, []
    if last.exists():
        state = torch.load(last, map_location=config['device'], weights_only=True)
        detector.model.load_state_dict(state['model']); adapter.load_state_dict(state['adapter'])
        optimizer.load_state_dict(state['optimizer']); scaler.load_state_dict(state['scaler'])
        start_epoch, start_step, best_loss = state['epoch'], state['step'], state['best_loss']
        running_sum, running_count, history = state['running_sum'], state['running_count'], state['history']
        torch.set_rng_state(state['rng'].cpu())
        if config['device'].startswith('cuda'):
            torch.cuda.set_rng_state_all([r.cpu() for r in state['cuda_rng']])
    def snapshot(epoch, step, improved=False, reason='periodic'):
        payload = {'kind': 'fog-detector-v1', 'model': detector.model.state_dict(), 'adapter': adapter.state_dict(),
                   'optimizer': optimizer.state_dict(), 'scaler': scaler.state_dict(), 'epoch': epoch, 'step': step,
                   'best_loss': best_loss, 'history': history, 'running_sum': running_sum, 'running_count': running_count,
                   'rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all() if config['device'].startswith('cuda') else [],
                   'checkpoint_reason': reason}
        if improved:
            save_torch(best, payload)
        save_torch(last, payload)
        atomic_json(directory/'progress.json', {'model': detector.name, 'arm': arm, 'seed': seed,
                     'epoch': epoch, 'step': step,
                     'best_loss': float(best_loss) if np.isfinite(best_loss) else None,
                     'reason': reason})
        return payload
    def loss_for(x, row, beta):
        with torch.autocast(device_type=x.device.type, enabled=use_amp):
            loss = detector.loss(adapter(x, beta), row)
        if torch.isfinite(loss):
            return loss
        if use_amp:
            # Ultralytics loss can overflow in float16 on a rare large-box
            # example even when the same update is finite in float32.
            with torch.autocast(device_type=x.device.type, enabled=False):
                loss = detector.loss(adapter(x.float(), beta.float()), row)
            if torch.isfinite(loss):
                return loss
        raise ValueError(f'Nonfinite loss for {row["id"]}; AMP fallback was also nonfinite')
    for epoch in range(start_epoch, config['epochs']):
        detector.training_mode(); adapter.train()
        order = random.Random(seed+epoch).sample(range(len(train)), len(train))
        optimizer.zero_grad(set_to_none=True)
        checkpoint_step = start_step
        try:
            for step in tqdm(range(start_step, len(order)), total=len(order), initial=start_step,
                             desc=f'{detector.name}/{arm} seed={seed} epoch={epoch+1}', mininterval=2):
                row = train[order[step]]
                x = image_tensor(row, config['size'], config['device'])
                beta = beta_for(arm, x, row, density)
                # Tail microbatch uses its actual size instead of downweighting the last examples.
                window_start = (step//config['accumulate'])*config['accumulate']
                divisor = min(config['accumulate'], len(order)-window_start)
                loss = loss_for(x, row, beta)
                scaler.scale(loss/divisor).backward()
                running_sum += float(loss.detach()); running_count += 1
                boundary = (step+1) % config['accumulate'] == 0 or step+1 == len(order)
                if boundary:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(parameters+list(adapter.parameters()), 10)
                    scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
                    checkpoint_step = step+1
                    if checkpoint_step//config['accumulate'] % config['save_every'] == 0 or checkpoint_step == len(order):
                        snapshot(epoch, checkpoint_step)
        except KeyboardInterrupt:
            # Preserve the last completed optimizer boundary. A restart reruns at
            # most the current accumulation window, without losing model updates.
            optimizer.zero_grad(set_to_none=True)
            snapshot(epoch, checkpoint_step, reason='interrupt')
            raise
        detector.model.eval(); adapter.eval()
        validation_losses = []
        with torch.no_grad():
            for row in tqdm(val, desc=f'{detector.name}/{arm} tuning', mininterval=2):
                x = image_tensor(row, config['size'], config['device'])
                loss = loss_for(x, row, beta_for(arm, x, row, density))
                validation_losses.append(float(loss))
        val_loss = float(np.mean(validation_losses))
        if not np.isfinite(val_loss):
            raise ValueError('Nonfinite validation loss')
        history.append({'epoch': epoch+1, 'train_loss': running_sum/running_count, 'val_loss': val_loss})
        improved = val_loss < best_loss
        best_loss = min(best_loss, val_loss)
        running_sum, running_count, start_step = 0., 0, 0
        snapshot(epoch+1, 0, improved, reason='epoch')
        pd.DataFrame(history).to_csv(directory/'history.csv', index=False)
    saved = torch.load(best, map_location=config['device'], weights_only=True)
    detector.model.load_state_dict(saved['model']); adapter.load_state_dict(saved['adapter'])
    detector.model.eval(); adapter.eval()


def evaluate_resumable(detector, adapter, density, arm, rows, config, path):
    import time
    completed = []
    if path.exists():
        # A trailing partial line after power loss is removed; earlier lines must be valid.
        with path.open('rb+') as handle:
            end = 0
            while line := handle.readline():
                if not line.endswith(b'\n'):
                    handle.truncate(end); break
                completed.append(json.loads(line)); end = handle.tell()
        if [r['id'] for r in completed] != [r['id'] for r in rows[:len(completed)]]:
            raise ValueError('Saved predictions do not match input order; use a new run directory')
    def sync():
        if config['device'].startswith('cuda'):
            torch.cuda.synchronize()
    @torch.inference_mode()
    def predict(row):
        sync(); start = time.perf_counter()
        x = image_tensor(row, config['size'], config['device'])
        sync(); before_density = time.perf_counter()
        beta = beta_for(arm, x, row, density)
        sync(); after_density = time.perf_counter()
        predictions = detector.predict(adapter(x, beta), row)
        sync(); end = time.perf_counter()
        return predictions, float(beta.item()), {'total_ms': (end-start)*1000,
            'density_ms': (after_density-before_density)*1000,
            'detector_adapter_ms': (end-after_density)*1000}
    if len(completed) < len(rows):
        for _ in range(3):
            predict(rows[len(completed)])
    with path.open('a', encoding='utf-8') as handle:
        for row in tqdm(rows[len(completed):], desc=f'Evaluate {detector.name}/{arm}', mininterval=2):
            timings = []
            for _ in range(config['repeats']):
                prediction, beta, timing = predict(row)
                timings.append(timing)
            result = {'id': row['id'], 'prediction': prediction, 'beta_input': beta, 'timings': timings}
            handle.write(json.dumps(result)+'\n'); handle.flush()
            completed.append(result)
    return completed


def export_trained_model(detector, adapter, directory, identity, config):
    """Write a compact inference artifact without optimizer, scaler or RNG state."""
    payload = {'kind': 'fog-detector-export-v1', **identity,
               'checkpoint_source': config['checkpoints'][detector.name],
               'input_size': config['size'], 'classes': list(CLASSES),
               'model': {key: value.detach().cpu() for key, value in detector.model.state_dict().items()},
               'adapter': {key: value.detach().cpu() for key, value in adapter.state_dict().items()}}
    save_torch(directory/'trained-model.pt', payload)


def export_saved_checkpoint(name, directory, identity, config):
    """Create the model-only artifact for a run completed by an older engine."""
    checkpoint = directory/'best.pt'
    if not checkpoint.exists() or (directory/'trained-model.pt').exists():
        return
    detector = Detector(name, config['checkpoints'][name], config['device'], trainable=False)
    adapter = FogAdapter().to(config['device'])
    saved = torch.load(checkpoint, map_location=config['device'], weights_only=True)
    detector.model.load_state_dict(saved['model']); adapter.load_state_dict(saved['adapter'])
    export_trained_model(detector, adapter, directory, identity, config)
    del detector, adapter


def run_training(rows, config):
    checks = preflight(config, rows)
    if not checks.ready.all():
        raise RuntimeError('Preflight failed:\n'+checks.loc[~checks.ready, ['check','detail']].to_string(index=False))
    output = Path(config['output'])
    output.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output/'run.lock'), timeout=0):
        signature = {'engine': ENGINE_VERSION, 'config': config, 'inputs': fingerprint(rows),
                     'density_sha256': digest(config['density_checkpoint']) if config.get('density_checkpoint') else None,
                     'source_sha256': {p.name: digest(p) for p in Path(__file__).parent.glob('fog_*.py')},
                     'versions': {p: importlib.metadata.version(p) for p in ('torch','ultralytics','transformers','pycocotools')}}
        settings = output/'settings.json'
        if settings.exists():
            previous = json.loads(settings.read_text())
            if previous != signature:
                immutable = ('config', 'inputs', 'density_sha256', 'versions')
                compatible = all(previous.get(key) == signature.get(key) for key in immutable)
                if compatible and previous.get('engine') in (ENGINE_VERSION-1, ENGINE_VERSION):
                    print('Migrating compatible checkpoint metadata to the current resumable engine.', flush=True)
                else:
                    if previous.get('inputs') != signature.get('inputs'):
                        raise ValueError('The input manifest changed since this run was started. The saved checkpoint belongs to a different dataset version; use a new output folder for the complete-data run. The old checkpoint is preserved.')
                    raise ValueError('Run configuration, inputs or package versions changed. Choose a new output folder to avoid mixing experiments.')
        atomic_json(settings, signature)
        atomic_json(output/'input-manifest.json', rows)
        parts = partitions(rows)
        statuses = []
        models = list(config['checkpoints'])
        for seed in config['seeds']:
            density = None
            if 'predicted_density' in config['arms']:
                density = fit_density(parts, config, output/f'density-{seed}', seed)
                density_test_path = output/f'density-{seed}'/'test-predictions.csv'
                if not density_test_path.exists():
                    density_results = []
                    with torch.inference_mode():
                        for row in tqdm(parts['density_test'], desc='Held-out density quality', mininterval=2):
                            estimate = float(density(image_tensor(row, config['size'], config['device'])).item())
                            density_results.append({'id':row['id'],'dataset':row['dataset'],'target_beta':row['beta'],'predicted_beta':estimate})
                    pd.DataFrame(density_results, columns=['id','dataset','target_beta','predicted_beta']).to_csv(density_test_path,index=False)
            # Rotate evaluation order across seeds to reduce fixed-order timing bias.
            offset = config['seeds'].index(seed) % len(models)
            for name in models[offset:]+models[:offset]:
                for arm in config['arms']:
                    directory = output/f'{name}-{arm}-{seed}'
                    directory.mkdir(exist_ok=True)
                    identity = {'model': name, 'arm': arm, 'seed': seed}
                    status_file = directory/'status.json'
                    if status_file.exists() and json.loads(status_file.read_text()).get('status') == 'complete':
                        export_saved_checkpoint(name, directory, identity, config)
                        statuses.append(json.loads(status_file.read_text())); continue
                    detector = adapter = None
                    try:
                        seed_all(seed)
                        detector = Detector(name, config['checkpoints'][name], config['device'], trainable=True)
                        seed_all(seed)
                        adapter = FogAdapter().to(config['device'])
                        trainable = sum(p.numel() for p in detector.model.parameters() if p.requires_grad)
                        atomic_json(status_file, {**identity, 'status': 'training', 'trainable_detector_parameters': trainable})
                        fit_detector(detector, adapter, density, parts, arm, config, directory, seed)
                        export_trained_model(detector, adapter, directory, identity, config)
                        atomic_json(status_file, {**identity, 'status': 'evaluating'})
                        summaries = []
                        for dataset in sorted({r['dataset'] for r in parts['test']}):
                            test_rows = sorted([r for r in parts['test'] if r['dataset'] == dataset], key=lambda r: r['id'])
                            if config['device'].startswith('cuda'):
                                torch.cuda.reset_peak_memory_stats()
                            predictions = evaluate_resumable(detector, adapter, density, arm, test_rows, config,
                                                             directory/f'{dataset}-predictions.jsonl')
                            peak = torch.cuda.max_memory_allocated()/2**20 if config['device'].startswith('cuda') else None
                            for metric in summarize(test_rows, predictions):
                                summaries.append({**identity, 'dataset': dataset, **metric,
                                    'peak_gpu_mb': peak, 'trainable_detector_parameters': trainable,
                                    'trained_checkpoint_mb': (directory/'best.pt').stat().st_size/2**20})
                        pd.DataFrame(summaries).to_csv(directory/'results.csv', index=False)
                        atomic_json(status_file, {**identity, 'status': 'complete'})
                    except BaseException as error:
                        atomic_json(status_file, {**identity, 'status': 'interrupted' if isinstance(error, KeyboardInterrupt) else 'error',
                                                 'reason': f'{type(error).__name__}: {error}'})
                        # Fail visibly; rerunning with the same config resumes saved work.
                        raise
                    finally:
                        del detector, adapter
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    statuses.append(json.loads(status_file.read_text()))
        from fog_analytics import export_analytics
        return export_analytics(output, rows), pd.DataFrame(statuses)
