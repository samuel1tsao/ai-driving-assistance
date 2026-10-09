"""Small shared utilities; notebook settings are passed explicitly."""
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd


def select_samples(samples, limit=100, seed=42):
    ordered = sorted(samples, key=lambda x: str(x['id']))
    if not ordered:
        return []
    return random.Random(seed).sample(ordered, min(limit or len(ordered), len(ordered)))


def save_run(root, task, samples, settings, results):
    folder = Path(root) / 'reports' / task
    folder.mkdir(parents=True, exist_ok=True)
    ids = [s['id'] for s in samples]
    fingerprint = hashlib.sha256('\n'.join(ids).encode()).hexdigest()
    payload = {'settings': settings, 'sample_ids': ids, 'sample_sha256': fingerprint}
    (folder / 'run.json').write_text(json.dumps(payload, indent=2, default=str))
    pd.DataFrame(results).to_csv(folder / 'results.csv', index=False)
    return pd.DataFrame(results)


def latency_summary(times):
    values = np.asarray(times, dtype=float)
    if not len(values):
        return {'mean_ms': np.nan, 'p95_ms': np.nan, 'images_per_second': np.nan}
    return {'mean_ms': float(values.mean()), 'p95_ms': float(np.percentile(values, 95)),
            'images_per_second': float(1000 / values.mean())}


def comparison_plot(results, metric):
    import matplotlib.pyplot as plt
    if 'status' not in results:
        return
    good = results[results.status == 'ok']
    if good.empty or metric not in good:
        print('No completed model measurements yet; see the status table above.')
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].barh(good.model, good[metric], color='#176B5B')
    axes[0].set_xlabel(metric)
    speed = good.dropna(subset=['mean_ms'])
    if len(speed):
        axes[1].barh(speed.model, speed.mean_ms, color='#D97706')
    else:
        axes[1].text(.5, .5, 'Inference timing unavailable', ha='center')
    axes[1].set_xlabel('Mean model pipeline time (ms); lower is better')
    fig.tight_layout()
    plt.show()
