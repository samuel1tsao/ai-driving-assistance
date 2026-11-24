"""Live screen demo for the fog YOLO11n engines.

The script continuously captures the upper-right quarter of the primary
monitor with mss, runs the TensorRT engine directly, and draws boxes in a
separate window. The next screen grab happens while the current frame is
on the GPU.

Keys:
  t  toggle baseline and the image-adaptive filters
  r  crop to a different rectangle
  m  return to the upper-right quarter
  [  ]  lower or raise the confidence threshold
  q or Esc  quit
"""
import argparse
import ctypes
import json
import sys
import threading
import time
from pathlib import Path
from queue import Empty, Full, Queue

import cv2
import mss
import numpy as np
import torch
import torchvision
from torch.nn import functional as F
from ultralytics.nn.backends.tensorrt import TensorRTBackend


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'notebooks' / '_support'))
from ia_yolo import ImageAdaptive  # noqa: E402


ENGINES = ROOT / 'reports' / 'fog' / 'ia-yolo-5070' / 'engines'
REGION_FILE = ENGINES / 'capture-region.json'
STUDY_CLASSES = (0, 1, 2, 3, 5, 6, 7)
NAMES = {
    0: 'person',
    1: 'bicycle',
    2: 'car',
    3: 'motorcycle',
    5: 'bus',
    6: 'train',
    7: 'truck',
}
COLORS = {
    0: (80, 180, 255),
    1: (255, 180, 40),
    2: (60, 220, 80),
    3: (200, 80, 255),
    5: (40, 140, 255),
    6: (180, 180, 180),
    7: (40, 80, 255),
}
EMPTY_BOXES = np.zeros((0, 4), np.float32)
EMPTY_SCORES = np.zeros((0,), np.float32)
EMPTY_CLASSES = np.zeros((0,), np.int32)


def parse_args():
    parser = argparse.ArgumentParser(description='Screen-capture demo for the fog TensorRT engines.')
    parser.add_argument('--engines', type=Path, default=ENGINES)
    parser.add_argument('--conf', type=float, default=0.25)
    parser.add_argument('--monitor', type=int, default=1, help='mss monitor index. 1 is the primary display.')
    parser.add_argument('--select', action='store_true', help='Crop to a rectangle before the live loop starts.')
    return parser.parse_args()


def load_adapter(path):
    adapter = ImageAdaptive().cuda().eval()
    adapter.load_state_dict(torch.load(path, map_location='cuda', weights_only=True))
    return adapter


def load_engine(path):
    return TensorRTBackend(str(path), torch.device('cuda:0'))


class ScreenGrabber:
    """Grab the latest screen rectangle on a side thread."""

    def __init__(self, region):
        self._region = dict(region)
        self._lock = threading.Lock()
        self._queue = Queue(maxsize=1)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name='screen-grab', daemon=True)
        self._thread.start()

    def set_region(self, region):
        with self._lock:
            self._region = dict(region)

    def latest(self):
        return self._queue.get()

    def close(self):
        self._stop.set()
        self._thread.join(timeout=1)

    def _run(self):
        with mss.MSS() as capture:
            while not self._stop.is_set():
                with self._lock:
                    region = dict(self._region)
                frame = np.ascontiguousarray(capture.grab(region))
                try:
                    self._queue.put_nowait(frame)
                except Full:
                    try:
                        self._queue.get_nowait()
                    except Empty:
                        pass
                    try:
                        self._queue.put_nowait(frame)
                    except Full:
                        pass


def monitor_region(capture, index):
    if index < 1 or index >= len(capture.monitors):
        raise SystemExit(f'Monitor {index} is not available. Choose 1..{len(capture.monitors) - 1}.')
    monitor = capture.monitors[index]
    return {
        'left': int(monitor['left']),
        'top': int(monitor['top']),
        'width': int(monitor['width']),
        'height': int(monitor['height']),
    }


def upper_right_quarter(monitor):
    """Upper-right quarter. On a 2560x1440 screen this is 1280x720 at x=1280, y=0."""
    width = monitor['width'] // 2
    height = monitor['height'] // 2
    return {
        'left': monitor['left'] + monitor['width'] - width,
        'top': monitor['top'],
        'width': width,
        'height': height,
    }


def monitor_image(capture, monitor):
    raw = np.asarray(capture.grab(monitor))
    return cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)


def choose_region(capture):
    monitor = capture.monitors[0]
    preview = monitor_image(capture, monitor)
    window = 'Select the photo area, then press Enter'
    rectangle = cv2.selectROI(window, preview, showCrosshair=False, fromCenter=False)
    cv2.destroyWindow(window)
    left, top, width, height = [int(value) for value in rectangle]
    if width < 20 or height < 20:
        raise SystemExit('No capture area selected.')
    region = {
        'left': monitor['left'] + left,
        'top': monitor['top'] + top,
        'width': width,
        'height': height,
    }
    REGION_FILE.write_text(json.dumps(region), encoding='utf-8')
    return region


def upload_bgra(raw, cache):
    height, width = raw.shape[:2]
    if cache.get('shape') != (height, width):
        cache['pinned'] = torch.empty((height, width, 4), dtype=torch.uint8, pin_memory=True)
        cache['gpu'] = torch.empty((height, width, 4), dtype=torch.uint8, device='cuda')
        cache['shape'] = (height, width)
    cache['pinned'].copy_(torch.from_numpy(raw))
    cache['gpu'].copy_(cache['pinned'], non_blocking=True)
    return cache['gpu']


def model_view(bgra, adapter):
    """Stretch the captured BGRA frame to the 640 square the detector was trained on."""
    rgb = bgra[..., [2, 1, 0]].permute(2, 0, 1).unsqueeze(0).float().div_(255)
    view = F.interpolate(rgb, (640, 640), mode='bilinear', align_corners=False)
    if adapter is not None:
        view = adapter(view).clamp_(0, 1)
    return view.contiguous()


def xywh_to_xyxy(xywh):
    half = xywh[..., 2:] / 2
    center = xywh[..., :2]
    return torch.cat((center - half, center + half), dim=-1)


def decode_boxes(prediction, confidence, class_ids):
    pred = prediction[0]
    scores, classes = pred[4:].max(dim=0)
    keep = scores > confidence
    keep &= (classes.unsqueeze(1) == class_ids).any(dim=1)
    index = keep.nonzero(as_tuple=False).view(-1)
    if index.numel() == 0:
        return EMPTY_BOXES, EMPTY_SCORES, EMPTY_CLASSES
    xywh = pred[:4].index_select(1, index).T
    scores = scores.index_select(0, index)
    classes = classes.index_select(0, index)
    boxes = xywh_to_xyxy(xywh)
    order = torchvision.ops.nms(boxes + classes[:, None] * 7680, scores, 0.7)[:300]
    return (
        boxes.index_select(0, order).float().cpu().numpy(),
        scores.index_select(0, order).float().cpu().numpy(),
        classes.index_select(0, order).int().cpu().numpy(),
    )


def preview_bgr(view, width, height):
    image = F.interpolate(view, (height, width), mode='bilinear', align_corners=False)
    channels = image[0].mul(255).clamp_(0, 255).byte()
    return channels[[2, 1, 0]].permute(1, 2, 0).contiguous().cpu().numpy()


def draw_detections(frame, boxes, scores, classes, mode, confidence, fps):
    height, width = frame.shape[:2]
    scale_x = width / 640
    scale_y = height / 640
    for xyxy, score, class_id in zip(boxes, scores, classes):
        class_id = int(class_id)
        x1, y1, x2, y2 = xyxy
        x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
        y1, y2 = int(y1 * scale_y), int(y2 * scale_y)
        color = COLORS.get(class_id, (255, 255, 255))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{NAMES.get(class_id, class_id)} {score:.2f}"
        cv2.putText(frame, label, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    label = 'adaptive filters' if mode == 'adaptive' else 'baseline'
    banner = f'{label}  conf {confidence:.2f}  {len(boxes)} boxes'
    cv2.rectangle(frame, (0, 0), (width, 36), (0, 0, 0), -1)
    cv2.putText(frame, banner, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    rate = f'{fps:.1f} FPS'
    text_size, _ = cv2.getTextSize(rate, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
    cv2.putText(
        frame, rate, (width - text_size[0] - 10, 26),
        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (80, 255, 120), 2, cv2.LINE_AA,
    )
    return frame


def display_size(width, height, max_width=1280):
    if width <= max_width:
        return width, height
    scale = max_width / width
    return max_width, max(1, int(height * scale))


def apply_region_change(grabber, region):
    grabber.set_region(region)
    preview_width, preview_height = display_size(region['width'], region['height'])
    cv2.resizeWindow('Fog demo — upper right', preview_width, preview_height)
    return preview_width, preview_height


def main():
    args = parse_args()
    engines = args.engines
    baseline_path = engines / 'baseline.engine'
    adaptive_path = engines / 'adaptive.engine'
    adapter_path = engines / 'adaptive-adapter.pt'
    for path in (baseline_path, adaptive_path, adapter_path):
        if not path.is_file():
            raise SystemExit(f'Missing {path}. Run scripts/export_fog_engines.py first.')

    torch.set_num_threads(1)
    ctypes.windll.winmm.timeBeginPeriod(1)
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        ctypes.windll.user32.SetProcessDPIAware()

    baseline = load_engine(baseline_path)
    adaptive_engine = load_engine(adaptive_path)
    adapter = load_adapter(adapter_path)
    class_ids = torch.tensor(STUDY_CLASSES, device='cuda')
    confidence = args.conf
    use_adaptive = True
    upload = {}

    with torch.inference_mode():
        blank = torch.zeros((1, 3, 640, 640), device='cuda')
        baseline.forward(blank)
        adaptive_engine.forward(blank)
        adapter(blank)

    try:
        with mss.MSS() as capture:
            full_monitor = monitor_region(capture, args.monitor)
            quarter = upper_right_quarter(full_monitor)
            region = choose_region(capture) if args.select else quarter
        grabber = ScreenGrabber(region)
        window = 'Fog demo — upper right'
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        preview_width, preview_height = display_size(region['width'], region['height'])
        cv2.resizeWindow(window, preview_width, preview_height)
        cv2.moveWindow(window, full_monitor['left'], full_monitor['top'] + full_monitor['height'] // 2)
        smoothed_fps = None
        while True:
            started = time.perf_counter()
            raw = grabber.latest()
            mode = 'adaptive' if use_adaptive else 'baseline'
            engine = adaptive_engine if use_adaptive else baseline
            with torch.inference_mode():
                view = model_view(upload_bgra(raw, upload), adapter if use_adaptive else None)
                prediction = engine.forward(view)[0]
                boxes, scores, classes = decode_boxes(prediction, confidence, class_ids)
                shown = preview_bgr(view, preview_width, preview_height)
            shown = draw_detections(shown, boxes, scores, classes, mode, confidence, smoothed_fps or 0)
            cv2.imshow(window, shown)
            key = cv2.waitKey(1) & 0xFF
            elapsed = time.perf_counter() - started
            instant_fps = 0 if elapsed <= 0 else 1 / elapsed
            smoothed_fps = instant_fps if smoothed_fps is None else smoothed_fps * 0.9 + instant_fps * 0.1
            if key in (ord('q'), 27):
                break
            if key == ord('t'):
                use_adaptive = not use_adaptive
            elif key == ord('r'):
                with mss.MSS() as capture:
                    region = choose_region(capture)
                preview_width, preview_height = apply_region_change(grabber, region)
            elif key == ord('m'):
                region = quarter
                preview_width, preview_height = apply_region_change(grabber, region)
            elif key == ord('['):
                confidence = max(0.05, round(confidence - 0.05, 2))
            elif key == ord(']'):
                confidence = min(0.95, round(confidence + 0.05, 2))
    finally:
        if 'grabber' in locals():
            grabber.close()
        ctypes.windll.winmm.timeEndPeriod(1)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
