"""Run either deployment compute profile, or both, on this computer."""

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

from filelock import FileLock, Timeout

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / 'configs/deployment.json'
PROFILES = json.loads(CONFIG.read_text(encoding='utf-8'))
OPTIONS = {key: ['--models', p['model'], '--device', p['device']] for key, p in PROFILES.items()}
RESULTS = ROOT / 'reports/deployment'
LOCK = ROOT / '.deployment.lock'
TIMEOUT = 600
INFO = logging.INFO
ERROR = logging.ERROR
DEFAULT_FRAMES = 100
IMAGE_SIZE = '640'
BASE = [sys.executable, str(ROOT / 'src/scripts/profile_models.py'), '--imgsz', IMAGE_SIZE]
PARSER = argparse.ArgumentParser(description=__doc__)
PARSER.add_argument('--mode', choices=[*PROFILES, 'both'], default='both')
PARSER.add_argument('--image', type=Path, required=True)
PARSER.add_argument('--frames', type=int, default=DEFAULT_FRAMES)
logging.basicConfig(level=INFO, format='%(levelname)s %(message)s')
LOGGER = logging.getLogger('driving')
FAILURES = (ValueError, OSError, subprocess.SubprocessError, Timeout)


def log(level, message, fields=None):
    LOGGER.log(level, '%s %s', message, json.dumps(fields or {}))


def selected_modes(mode):
    return list(PROFILES) if mode == 'both' else [mode]


def validate_inputs(args):
    if args.frames < 1 or not args.image.is_file():
        raise ValueError('An image and positive frame count are required')
    for mode in selected_modes(args.mode):
        require_model(PROFILES[mode]['model'])


def require_model(model):
    if not (ROOT / model).is_file():
        raise FileNotFoundError(f'Missing {model}; see docs/deployment-modes.md')


def prepare_commands(args, output):
    tail = ['--image', str(args.image.resolve()), '--frames', str(args.frames)]
    return [BASE + OPTIONS[m] + tail + ['--output', str(output / f'{m}.json')] for m in args.modes]


def execute(commands):
    with FileLock(str(LOCK), timeout=0):
        for command in commands:
            subprocess.run(command, check=True, cwd=ROOT, timeout=TIMEOUT)


def prepare_run(args):
    output = RESULTS / uuid4().hex
    output.mkdir(parents=True)
    log(INFO, 'Results', {'run': output.name})
    args.modes = selected_modes(args.mode)
    return prepare_commands(args, output)


def main():
    args = PARSER.parse_args()
    log(INFO, 'Validate inputs')
    validate_inputs(args)
    log(INFO, 'Prepare run')
    commands = prepare_run(args)
    log(INFO, 'Run profiles sequentially')
    execute(commands)


if __name__ == '__main__':
    try:
        main()
    except FAILURES as error:
        log(ERROR, 'Deployment run failed', {'type': type(error).__name__})
        sys.exit(1)
