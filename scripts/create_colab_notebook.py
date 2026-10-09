"""Build a standalone Colab notebook from the active research notebook generator."""
import json
from pathlib import Path
import zipfile
import nbformat as nbf
from create_fog_notebook import build as build_research

ROOT = Path(__file__).resolve().parents[1]


def build():
    original = build_research()
    sources = {p.relative_to(ROOT).as_posix(): p.read_text(encoding='utf-8')
               for p in sorted((ROOT/'notebooks/_support').glob('fog_*.py'))}
    for name in ('requirements-notebook.txt', 'requirements-fog-research.txt', 'requirements-colab.txt'):
        sources[name] = (ROOT/name).read_text(encoding='utf-8')
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    cells = [md('''# Fog-density detection — standalone Google Colab

All five project modules are embedded below; no repository checkout or code ZIP is required.
Select **Runtime → Change runtime type → GPU**, then run the cells in order.
The original full study remains 3 detectors × 2 arms × 3 seeds, with 30 epochs.
This can exceed a Colab session. Checkpoints go to Google Drive by default.

**Data:** Put the complete `fog-colab-data.zip` (including Foggy Cityscapes) in `MyDrive/fog-study/`.
Low-disk mode keeps the ZIP and outputs in Drive, extracts only one dataset copy locally,
and requires 15 GiB of free working space after extraction. It blocks additional dataset downloads.
Create it on your PC with `.venv/Scripts/python.exe scripts/package_colab_data.py`.
Pretrained model weights download when needed; extra dataset downloads are blocked in low-disk mode.
nuScenes cannot be anonymously downloaded by this notebook: its official download requires an account.
The data ZIP carries your existing prepared labels and referenced camera images, excluding lidar/sweeps/archive copies.
Use `--include-fog` when packaging to carry Foggy Cityscapes too.

See the [Colab FAQ](https://research.google.com/colaboratory/faq.html) and
[nuScenes downloads](https://www.nuscenes.org/download).'''),
        md('## Configuration — edit only this cell\nPaths below are relative to MyDrive. In `zip` mode, the data folder contains the ZIP. In `directory` mode, it contains the extracted `data/` folder (and `.cache/` if present).'), code('''from pathlib import Path
import os, sys, subprocess, shutil

# Storage: do not include /content/drive/MyDrive in these relative paths.
DRIVE_DATA_DIR = "fog-study"          # Example: "Research/Fog study"
DRIVE_OUTPUT_DIR = "fog-study/runs"   # Checkpoints, metrics and exported ZIPs
DATA_MODE = "zip"                    # "zip": extract locally; "directory": read extracted data in Drive
DATA_ZIP_NAME = "fog-colab-data.zip"
DATA_ZIP_URL = ""                    # Optional HTTPS URL; requires LOW_DISK_MODE=False
USE_DRIVE = True
LOW_DISK_MODE = True
MIN_FREE_GIB = 15                    # Runtime working-space reserve
RUNTIME_ROOT = "/content/fog-study"  # Code, caches, and ZIP-mode extracted data
RUN_NAME = "colab-full-study"         # Same name/settings to resume; new name for a changed experiment

# Study controls: defaults preserve the complete matched study.
ACTION = "train"                     # "train", "preflight", or "analyze"
STUDY = {
    "epochs": 30, "density_epochs": 15,
    "seeds": [42, 43, 44],
    "arms": ["rgb_control", "predicted_density"],
    "size": 640, "accumulate": 4, "save_every": 200, "repeats": 1,
    "learning_rates": {"YOLO11n": 1e-4, "YOLO11m": 1e-4, "DETR": 1e-5},
    "adapter_lr": 1e-3, "amp": True,
    "device": "cuda:0",
    "density_checkpoint": None,
}
# Relative model paths are resolved under RUNTIME_ROOT; DETR is a model ID.
MODEL_CHECKPOINTS = {
    "YOLO11n": "yolo11n.pt", "YOLO11m": "yolo11m.pt",
    "DETR": "facebook/detr-resnet-50",
}
'''), md('## Initialize paths and storage — no settings to edit below'), code(r'''
def mydrive_path(relative):
    base = Path('/content/drive/MyDrive')
    from pathlib import PurePosixPath
    value = PurePosixPath(relative)
    if value.is_absolute() or '..' in value.parts or "\\" in str(relative) or ':' in str(relative):
        raise ValueError('Use a folder relative to MyDrive, such as Research/Fog study.')
    return base / value

if ACTION not in {'train', 'preflight', 'analyze'}:
    raise ValueError('ACTION must be train, preflight, or analyze.')
if DATA_MODE not in {'zip', 'directory'}:
    raise ValueError('DATA_MODE must be zip or directory.')
if not RUN_NAME or Path(RUN_NAME).name != RUN_NAME or RUN_NAME in {'.', '..'} or "\\" in RUN_NAME:
    raise ValueError('RUN_NAME must be a single folder name.')
if MIN_FREE_GIB < 0:
    raise ValueError('MIN_FREE_GIB cannot be negative.')
DRIVE_DATA_ROOT = mydrive_path(DRIVE_DATA_DIR)
DRIVE_RUNS_ROOT = mydrive_path(DRIVE_OUTPUT_DIR)
if Path(DATA_ZIP_NAME).name != DATA_ZIP_NAME or not DATA_ZIP_NAME or "\\" in DATA_ZIP_NAME:
    raise ValueError('DATA_ZIP_NAME must be a filename, not a path.')
if DATA_MODE == 'directory' and (not USE_DRIVE or DATA_ZIP_URL):
    raise ValueError('Directory mode requires mounted Drive and no DATA_ZIP_URL.')
ROOT = Path(RUNTIME_ROOT)
if not ROOT.is_absolute():
    raise ValueError('RUNTIME_ROOT must be an absolute runtime path.')
ROOT.mkdir(parents=True, exist_ok=True)
if LOW_DISK_MODE and (not USE_DRIVE or DATA_ZIP_URL):
    raise ValueError('Low-disk mode requires USE_DRIVE=True and DATA_ZIP_URL="".')
if USE_DRIVE:
    from google.colab import drive
    drive.mount('/content/drive')
DATA_ROOT = DRIVE_DATA_ROOT if DATA_MODE == 'directory' else ROOT
DATA_ZIP_PATH = DRIVE_DATA_ROOT / DATA_ZIP_NAME if USE_DRIVE else ROOT / DATA_ZIP_NAME
OUTPUT_ROOT = DRIVE_RUNS_ROOT if USE_DRIVE else ROOT / 'reports/fog'
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
os.chdir(ROOT)
print('Data root:', DATA_ROOT)
print('Dataset ZIP:', DATA_ZIP_PATH if DATA_MODE == 'zip' else '(not used)')
print('Checkpoints:', OUTPUT_ROOT / RUN_NAME)
print(f'Free runtime disk: {shutil.disk_usage(ROOT).free / 2**30:.1f} GiB')
'''), md('## Restore embedded project code'),
        code('EMBEDDED_FILES = ' + repr(sources) + '''
for relative, source in EMBEDDED_FILES.items():
    target = ROOT / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(source, encoding='utf-8')
print('Restored', len(EMBEDDED_FILES), 'project files.')
'''), md('## Install dependencies and verify GPU\nUses Colab’s existing PyTorch/CUDA installation. If pip requests a runtime restart, restart and rerun from the first cell.'),
        code('''subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--no-cache-dir', '-q', '-r', str(ROOT/'requirements-colab.txt')])
import torch, torchvision
from torchvision.ops import nms
assert torch.cuda.is_available(), 'Select a GPU runtime before running the full study.'
nms(torch.tensor([[0., 0., 2., 2.]], device='cuda'), torch.tensor([1.], device='cuda'), .5)
print(torch.__version__, torchvision.__version__, torch.cuda.get_device_name(0))
subprocess.run([sys.executable, '-m', 'pip', 'freeze'], stdout=(ROOT/'colab-packages.txt').open('w'), check=True)
'''), md('## Load your portable dataset ZIP\nExtraction checks paths and free disk space. A completion marker avoids repeat extraction in the same runtime. Keep the ZIP in Drive for a fresh runtime.'),
        code('''import urllib.request, zipfile, stat

if DATA_MODE == 'directory':
    for name in ('nuscenes_all.json', 'cityscapes.json'):
        if not (DATA_ROOT / 'data/fog' / name).is_file():
            raise FileNotFoundError(f'Missing {DATA_ROOT / "data/fog" / name}. Point DRIVE_DATA_DIR to the folder containing data/.')
    if shutil.disk_usage(ROOT).free < MIN_FREE_GIB * 2**30:
        raise OSError('Runtime free space is below the configured working-space reserve.')
    print('Using existing Drive data; no local extraction.')
else:
    
    if LOW_DISK_MODE:
        if DATA_ZIP_URL or not USE_DRIVE:
            raise ValueError('Low-disk mode requires the ZIP in mounted Drive, not a runtime-local download.')
        if not Path(DATA_ZIP_PATH).resolve().is_relative_to(Path('/content/drive').resolve()):
            raise ValueError('Set DATA_ZIP_PATH to the complete ZIP under /content/drive/.')
    if DATA_ZIP_URL:
        if not DATA_ZIP_URL.startswith('https://'):
            raise ValueError('Use a direct HTTPS URL.')
        bundle = ROOT/'downloaded-data.zip'
        if not bundle.exists():
            partial = bundle.with_suffix('.part')
            urllib.request.urlretrieve(DATA_ZIP_URL, partial)
            partial.replace(bundle)
    else:
        bundle = Path(DATA_ZIP_PATH)
    marker = ROOT/'.data-extracted'
    if not marker.exists():
        if not bundle.is_file():
            raise FileNotFoundError('Upload fog-colab-data.zip to the configured Drive path or set DATA_ZIP_URL. See the first cell.')
        with zipfile.ZipFile(bundle) as archive:
            members = archive.infolist()
            if LOW_DISK_MODE:
                needed = {'data/fog/nuscenes_all.json', 'data/fog/cityscapes.json'}
                if not needed.issubset(archive.namelist()):
                    raise ValueError('Low-disk mode needs both prepared datasets. Use the --include-fog bundle; no extra dataset download will be started.')
            for member in members:
                destination = (ROOT/member.filename).resolve()
                if not destination.is_relative_to(ROOT.resolve()) or not member.filename.startswith(('data/', '.cache/')) or stat.S_ISLNK(member.external_attr >> 16):
                    raise ValueError('Unsafe or unexpected ZIP member: ' + member.filename)
            required = sum(m.file_size for m in members)
            reserve = int(MIN_FREE_GIB * 2**30)
            free = shutil.disk_usage(ROOT).free
            print(f'Extraction: {required / 2**30:.2f} GiB; free: {free / 2**30:.2f} GiB; reserve: {MIN_FREE_GIB} GiB')
            if required + reserve > free:
                raise OSError(f'Need {(required + reserve) / 2**30:.2f} GiB FREE before extraction, including working space. Total disk capacity is not free space.')
            archive.extractall(ROOT)
        marker.write_text('complete')
    assert (ROOT/'data/fog/nuscenes_all.json').is_file(), 'ZIP must be produced by package_colab_data.py.'
    if LOW_DISK_MODE:
        assert (ROOT/'data/fog/cityscapes.json').is_file(), 'Complete bundle with Foggy Cityscapes required.'
    if shutil.disk_usage(ROOT).free < MIN_FREE_GIB * 2**30:
        raise OSError('Runtime free space is below the configured working-space reserve.')
    print('Portable nuScenes data is ready.')
''')]
    # Reuse the reviewed training/report cells instead of maintaining a second engine.
    for cell in original.cells[1:]:
        cell = nbf.from_dict(dict(cell))
        if cell.cell_type == 'code':
            cell.outputs, cell.execution_count = [], None
            if 'def configure_run(root):' in cell.source:
                begin = cell.source.index('def configure_run(root):')
                finish = cell.source.index("assert ACTION in", begin)
                cell.source = cell.source[:begin] + """# Resolve the configuration from the first cell; no hidden experiment overrides.
CONFIG = default_config(ROOT)
CONFIG.update(STUDY)
CONFIG['checkpoints'] = {name: value if name == 'DETR' else str(ROOT / value)
                         for name, value in MODEL_CHECKPOINTS.items()}
CONFIG['output'] = str(OUTPUT_ROOT / RUN_NAME)
CONFIG['manifest'] = str(DATA_ROOT / 'data/fog/nuscenes_all.json')
CONFIG['city_manifest'] = str(DATA_ROOT / 'data/fog/cityscapes.json')
CITY_ROOT = DATA_ROOT / 'data/fog/cityscapes'
DOWNLOAD_FOG_IMAGES = not LOW_DISK_MODE and DATA_MODE == 'zip'
NUSCENES_ROOT = DATA_ROOT / 'data/depth/datasets/nuscenes'
""" + cell.source[finish:]
            cell.source = cell.source.replace('DOWNLOAD_FOG_IMAGES = True', 'DOWNLOAD_FOG_IMAGES = not LOW_DISK_MODE')
            # The existing resume fingerprint includes mtimes. Use a stable value on
            # ephemeral Colab copies so re-extracting identical inputs can resume.
            cell.source = cell.source.replace('rows.extend(read_manifest(path))', '''portable_rows = json.loads(path.read_text(encoding='utf-8'))
            for row in portable_rows:
                image_path = (path.parent / row['image']).resolve()
                os.utime(image_path, ns=(946684800000000000, 946684800000000000))
            rows.extend(read_manifest(path))''')
        if cell.cell_type == 'markdown' and cell.source.startswith('## 1. Configure'):
            cell.source = '## Load research modules and display resolved configuration'
        cells.append(cell)
    cells.extend([md('## Export results\nRun after training, or after interrupting it. The ZIP includes checkpoints and may be large. With Drive enabled, it stays in Drive; otherwise the browser download starts.'), code('''export_path = OUTPUT_ROOT / (RUN_NAME + '-results')
Path(CONFIG['output']).mkdir(parents=True, exist_ok=True)
shutil.copy2(ROOT/'colab-packages.txt', Path(CONFIG['output'])/'colab-packages.txt')
archive_path = shutil.make_archive(str(export_path), 'zip', CONFIG['output'])
print('Saved:', archive_path)
if not USE_DRIVE:
    from google.colab import files
    files.download(archive_path)
''')])
    notebook = nbf.v4.new_notebook(cells=cells, metadata={
        'kernelspec': {'display_name': 'Python 3', 'name': 'python3', 'language': 'python'},
        'colab': {'name': 'fog_density_colab.ipynb'}, 'accelerator': 'GPU'})
    nbf.validate(notebook)
    return notebook


if __name__ == '__main__':
    notebook_path = ROOT/'notebooks/fog_density_colab.ipynb'
    nbf.write(build(), notebook_path)
    destination = ROOT/'dist/fog-colab-code.zip'
    destination.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(destination, 'w', zipfile.ZIP_DEFLATED) as archive:
        archive.write(notebook_path, notebook_path.name)
        archive.write(ROOT/'scripts/package_colab_data.py', 'scripts/package_colab_data.py')
        archive.write(ROOT/'docs/colab-guide.md', 'colab-guide.md')
    print(notebook_path)
    print(destination)
