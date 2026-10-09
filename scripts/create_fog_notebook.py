"""Author the full-training notebook, preserving a reviewed previous version on update."""
import argparse
import shutil
from datetime import datetime
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT/'notebooks/fog_density_detection_research.ipynb'


def build():
    cells = []
    def md(text): cells.append(nbf.v4.new_markdown_cell(text))
    def code(text): cells.append(nbf.v4.new_code_cell(text))
    md('# Fog-density detection\n\nCompare YOLO11n, YOLO11m and DETR with RGB versus predicted fog density.')
    md('## 1. Configure\n\nSet `ACTION` and experiment controls. Keep the output path to resume.')
    code('''from pathlib import Path
import json
import os
import sys
import importlib
import pandas as pd
from IPython.display import display, Image


def configure_environment():
    root = Path.cwd().resolve()
    if root.name == 'notebooks':
        root = root.parent
    support = str(root / 'notebooks/_support')
    if support not in sys.path:
        sys.path.insert(0, support)
    os.environ.setdefault('YOLO_CONFIG_DIR', str(root / '.ultralytics'))
    os.environ.setdefault('HF_HOME', str(root / '.cache/huggingface'))
    os.environ.setdefault('HF_HUB_DISABLE_PROGRESS_BARS', '1')
    return root


ROOT = configure_environment()
from fog_data import cityscapes_manifest, resolve_foggy_images
from fog_analytics import export_analytics
from fog_prepare import prepare_nuscenes
from fog_research import read_manifest
import fog_training
fog_training = importlib.reload(fog_training)
from fog_training import default_config, preflight, run_training


def configure_run(root):
    config = default_config(root)
    config.update(epochs=30, seeds=[42, 43, 44], size=640, accumulate=4,
                  arms=['rgb_control', 'predicted_density'], density_checkpoint=None)
    return config


ACTION = 'train'  # train, preflight, analyze
CONFIG = configure_run(ROOT)
CITY_ROOT = ROOT / 'data/fog/cityscapes'
DOWNLOAD_FOG_IMAGES = True
NUSCENES_ROOT = ROOT / 'data/depth/datasets/nuscenes'
assert ACTION in {'train', 'preflight', 'analyze'}
display(pd.DataFrame([{'setting': key, 'value': str(value)}
                      for key, value in CONFIG.items()]))''')
    md('## 2. Prepare data\n\nBuild or reuse the complete nuScenes and Foggy Cityscapes manifests.')
    code('''def prepare_manifests(root, config):
    if ACTION == 'analyze':
        return
    if not Path(config['manifest']).exists():
        prepare_nuscenes(NUSCENES_ROOT, config['manifest'])
    if not Path(config['city_manifest']).exists():
        source = resolve_foggy_images(root, CITY_ROOT, download=DOWNLOAD_FOG_IMAGES)
        if source is not None:
            cityscapes_manifest(source, config['city_manifest'])
        else:
            print('Foggy Cityscapes not found; density training is blocked.')


def load_rows(config):
    rows = []
    for key in ('manifest', 'city_manifest'):
        path = Path(config[key])
        if path.exists():
            print(f'Validating {path.name}')
            rows.extend(read_manifest(path))
    return rows


def show_data_summary(rows, config):
    if not rows:
        return
    frame = pd.DataFrame(rows)
    display(frame.groupby(['dataset', 'split']).agg(
        images=('id', 'size'), scenes=('group', 'nunique'), known_beta=('beta', 'count')))
    coverage_path = Path(config['manifest']).with_suffix('.coverage.json')
    if coverage_path.exists():
        coverage = json.loads(coverage_path.read_text())
        display(pd.DataFrame([{
            'expected': coverage['expected_images'],
            'available': coverage['available_images'],
            'missing': len(coverage['missing_images']),
            'complete': coverage['complete']}]))


prepare_manifests(ROOT, CONFIG)
rows = load_rows(CONFIG)
show_data_summary(rows, CONFIG)''')
    md('## 3. Validate\n\nPreflight stops before training when data, models or settings are invalid.')
    code('''def validate_run(config, rows):
    checks = preflight(config, rows)
    display(checks)
    if not checks.ready.all():
        display(checks.loc[~checks.ready, ['check', 'detail']])
    return checks


checks = None if ACTION == 'analyze' else validate_run(CONFIG, rows)''')
    md('## 4. Train and evaluate\n\nInterrupt the kernel to checkpoint; rerun with the same output path to resume.')
    code('''def execute_analysis(config):
    return export_analytics(config['output'])


def execute_training(action, config, rows, checks):
    if action == 'preflight':
        return pd.DataFrame(), None
    if not checks.ready.all():
        print('Training is blocked by the checks above.')
        return pd.DataFrame(), None
    return run_training(rows, config)


def execute_run(action, config, rows, checks):
    if action == 'analyze':
        return execute_analysis(config), None
    return execute_training(action, config, rows, checks)


results, status = execute_run(ACTION, CONFIG, rows, checks)
if status is not None:
    display(status)''')
    md('## 5. Review\n\nReports include accuracy, latency, density error, paired deltas and saved checkpoints.')
    code('''def show_metrics(results, output):
    if results.empty:
        print('No completed runs yet.')
        return
    display(results[results.density == 'all'].round(4))
    paired = pd.read_csv(Path(output) / 'paired-deltas.csv', dtype={'density': str})
    display(paired.loc[paired.density == 'all',
                       ['model', 'dataset', 'seed', 'arm', 'delta_mAP', 'delta_latency_ms']].round(4))
    density = pd.read_csv(Path(output) / 'density-quality.csv')
    if not density.empty:
        display(density.round(6))


def show_figures(output, results):
    if results.empty:
        return
    for pattern in ('*-comparison.png', '*-deltas.png', '*-classes.png', '*-density.png', '*-loss.png'):
        for path in sorted((Path(output) / 'figures').glob(pattern)):
            display(Image(filename=str(path), width=1000))


show_metrics(results, CONFIG['output'])
show_figures(CONFIG['output'], results)''')
    md('## Notes\n\nnuScenes boxes are projected 2D proxies. Foggy Cityscapes beta comes from filenames. Results are saved under the configured output path.')
    notebook = nbf.v4.new_notebook(cells=cells, metadata={
        'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
        'language_info':{'name':'python','version':'3.11'}})
    nbf.validate(notebook)
    return notebook


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--update',action='store_true')
    args = parser.parse_args()
    if TARGET.exists():
        if not args.update:
            raise FileExistsError('Review existing notebook and use --update to preserve a backup before replacing it')
        backup = ROOT/'archive/legacy-before-fog/notebooks'/f'fog-pilot-{datetime.now().strftime("%Y%m%d-%H%M%S")}.ipynb'
        backup.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(TARGET,backup)
    nbf.write(build(),TARGET)
    print(TARGET)
