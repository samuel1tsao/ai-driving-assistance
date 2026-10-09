"""Bounded portability checks; does not start training or download datasets."""
import ast
import argparse
import json
import posixpath
from pathlib import Path
import tempfile
import zipfile
import nbformat
from package_colab_data import package

ROOT = Path(__file__).resolve().parents[1]
notebook = nbformat.read(ROOT/'notebooks/fog_density_colab.ipynb', as_version=4)
nbformat.validate(notebook)
for cell in notebook.cells:
    if cell.cell_type == 'code':
        ast.parse(cell.source)
with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    namespace = {'ROOT': root}
    restore = next(c.source for c in notebook.cells if c.source.startswith('EMBEDDED_FILES ='))
    exec(restore, namespace)
    for source in (ROOT/'notebooks/_support').glob('fog_*.py'):
        assert (root/'notebooks/_support'/source.name).read_text(encoding='utf-8') == source.read_text(encoding='utf-8')
    data = root/'data/fog'
    data.mkdir(parents=True)
    image = root/'data/images/frame.jpg'
    image.parent.mkdir()
    image.write_bytes(b'fixture image')
    (data/'nuscenes_all.json').write_text(json.dumps([{'image': str(image)}]))
    (data/'nuscenes_all.coverage.json').write_text(json.dumps({'complete': True}))
    bundle = root/'fixture.zip'
    package(root, bundle)
    extraction = next(c.source for c in notebook.cells if c.source.startswith('import urllib.request'))
    destination = root/'colab'
    destination.mkdir()
    import shutil
    env = {'ROOT': destination, 'Path': Path, 'DATA_ZIP_URL': '', 'DATA_ZIP_PATH': str(bundle), 'shutil': shutil,
           'LOW_DISK_MODE': False, 'USE_DRIVE': False, 'MIN_FREE_GIB': 0, 'DATA_MODE': 'zip'}
    exec(extraction, env)
    row = json.loads((destination/'data/fog/nuscenes_all.json').read_text())[0]
    assert (destination/'data/fog'/row['image']).resolve().read_bytes() == b'fixture image'
    exec(extraction, env)  # completed extraction can be rerun
    try:
        package(root, bundle)
    except FileExistsError:
        pass
    else:
        raise AssertionError('Overwrote an existing archive')
    malicious = root/'malicious.zip'
    with zipfile.ZipFile(malicious, 'w') as archive:
        archive.writestr('data/../../escaped', 'no')
    (destination/'.data-extracted').unlink()
    env['DATA_ZIP_PATH'] = str(malicious)
    try:
        exec(extraction, env)
    except ValueError:
        pass
    else:
        raise AssertionError('Accepted path traversal')
    env.update(LOW_DISK_MODE=True, USE_DRIVE=True, DATA_ZIP_URL='https://example.invalid/data.zip')
    try:
        exec(extraction, env)
    except ValueError as error:
        assert 'Low-disk mode' in str(error)
    else:
        raise AssertionError('Allowed runtime-local download in low-disk mode')
    env.update(LOW_DISK_MODE=False, DATA_ZIP_URL='', DATA_ZIP_PATH=str(bundle), MIN_FREE_GIB=10**9)
    try:
        exec(extraction, env)
    except OSError as error:
        assert 'FREE before extraction' in str(error)
    else:
        raise AssertionError('Extraction ignored working-space reserve')
    # Directory mode must read the configured root without extracting locally.
    env.update(DATA_MODE='directory', DATA_ROOT=destination, MIN_FREE_GIB=0)
    city_manifest = destination/'data/fog/cityscapes.json'
    city_manifest.write_text('[]')
    exec(extraction, env)
    city_manifest.unlink()
    try:
        exec(extraction, env)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError('Accepted an incomplete Drive data directory')

# Exercise path parsing without mounting Drive, installing packages or downloading data.
config_cell = next(c.source for c in notebook.cells if 'DRIVE_DATA_DIR = ' in c.source)
init_cell = next(c.source for c in notebook.cells if 'def mydrive_path' in c.source)
config_env = {}
exec(config_cell, config_env)
path_function = next(n for n in ast.parse(init_cell).body if isinstance(n, ast.FunctionDef))
exec(compile(ast.Module(body=[path_function], type_ignores=[]), '<paths>', 'exec'), config_env)
resolve = config_env['mydrive_path']
assert str(resolve('Research/Fog study')).replace('\\', '/') == '/content/drive/MyDrive/Research/Fog study'
for invalid in ('/outside', '../outside', 'folder/../../outside', 'C:/outside', 'folder\\outside'):
    try:
        resolve(invalid)
    except ValueError:
        pass
    else:
        raise AssertionError(f'Accepted invalid MyDrive path: {invalid}')
assert config_env['STUDY']['epochs'] == 30
assert config_env['STUDY']['seeds'] == [42, 43, 44]
assert config_env['STUDY']['arms'] == ['rgb_control', 'predicted_density']
assert set(config_env['MODEL_CHECKPOINTS']) == {'YOLO11n', 'YOLO11m', 'DETR'}
print('PASS: centralized study defaults, MyDrive paths, directory mode and missing-data rejection')
print('PASS: cell syntax, notebook format, embedded sources, portable extraction, rerun, overwrite and traversal checks')

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--data-zip', type=Path)
args = parser.parse_args()
if args.data_zip:
    with zipfile.ZipFile(args.data_zip) as archive:
        names = set(archive.namelist())
        assert len(names) == len(archive.infolist()), 'Duplicate archive entries'
        count = 0
        for manifest in ('data/fog/nuscenes_all.json', 'data/fog/cityscapes.json'):
            if manifest not in names:
                assert manifest.endswith('cityscapes.json'), 'nuScenes manifest missing'
                continue
            rows = json.loads(archive.read(manifest))
            for row in rows:
                name = posixpath.normpath(posixpath.join('data/fog', row['image']))
                assert name in names, name
                assert not name.startswith(('/', '../')) and '\\' not in name, name
            # Read samples throughout each dataset, checking ZIP CRC and source bytes.
            for row in rows[::max(1, len(rows)//12)]:
                name = posixpath.normpath(posixpath.join('data/fog', row['image']))
                assert archive.read(name) == (ROOT/name).read_bytes(), name
            count += len(rows)
        assert json.loads(archive.read('data/fog/nuscenes_all.coverage.json'))['complete'] is True
        print(f'PASS: {count:,} portable image references, coverage, and sampled source-byte/CRC checks; ZIP {args.data_zip.stat().st_size / 2**30:.2f} GiB')
