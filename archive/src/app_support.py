"""Bounded local inputs and report handling for the guided interface."""

import io
import json
import re
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / '.venv/Lib/site-packages/ultralytics/assets/bus.jpg'
UPLOADS = ROOT / 'data/local-inputs'
REPORTS = ROOT / 'reports/deployment'
MAX_BYTES = 10 * 1024 * 1024
MAX_PIXELS = 12_000_000
Image.MAX_IMAGE_PIXELS = MAX_PIXELS
TIMEOUT = 1250
RUNNER = [sys.executable, str(ROOT / 'src/scripts/run_deployment.py')]
RUN_PATTERN = re.compile(r'INFO Results \{"run": "([a-f0-9]{32})"\}')
MODES = {
    'Compare both': 'both',
    'Phone candidate (desktop test)': 'phone',
    'Backend (this PC)': 'backend',
}
TITLES = {'phone': 'Phone candidate · desktop CPU', 'backend': 'Backend · local GPU'}
INPUT_ERROR = 'Choose a JPEG or PNG under 10 MB and 12 million pixels.'
RUN_ERROR = 'Run did not complete. Check model setup, GPU availability, or another active run.'
REPORT_ERROR = 'Incomplete results. Run the comparison again.'
ROW_KEYS = ['Mean (ms)', 'p95 (ms)', 'Predictions/sec']
METRIC_KEYS = ['mean_ms', 'p95_ms', 'serial_prediction_fps']


def decode_image(data):
    if not data or len(data) > MAX_BYTES:
        raise ValueError(INPUT_ERROR)
    image = Image.open(io.BytesIO(data))
    validate_size(image)
    image.load()
    return image.convert('RGB')


def validate_size(image):
    if image.format not in {'JPEG', 'PNG'} or image.width * image.height > MAX_PIXELS:
        raise ValueError(INPUT_ERROR)


def save_input(image):
    UPLOADS.mkdir(parents=True, exist_ok=True)
    path = UPLOADS / f'{uuid4().hex}.png'
    image.save(path)
    return path


def run_comparison(image, mode, frames):
    command = RUNNER + ['--image', str(image), '--mode', mode, '--frames', str(frames)]
    return subprocess.run(
        command, cwd=ROOT, capture_output=True, text=True, timeout=TIMEOUT, check=False
    )


def locate_results(process):
    match = RUN_PATTERN.search(process.stderr)
    if process.returncode or not match:
        raise ValueError(RUN_ERROR)
    return REPORTS / match.group(1)


def read_reports(directory):
    return {
        path.stem: json.loads(path.read_text())
        for path in directory.glob('*.json')
        if path.stem in TITLES
    }


def validate_reports(reports, mode):
    expected = set(TITLES) if mode == 'both' else {mode}
    if set(reports) != expected or any(not item.get('results') for item in reports.values()):
        raise ValueError(REPORT_ERROR)


def comparison_rows(reports):
    return [
        {'Mode': TITLES[key], **dict(zip(ROW_KEYS, [r[k] for k in METRIC_KEYS]))}
        for key, item in reports.items()
        for r in item['results']
    ]
