# Two deployment modes and the first local run

Both modes are retained. `scripts/run_deployment.py --mode phone|backend|both` selects a compute profile in `configs/deployment.json`. This is the offline deployment spike: it is not a native mobile application or a listening network server.

| Mode | Current artifact | Where this run executes | Intended deployment |
| --- | --- | --- | --- |
| phone | YOLO26n ONNX, FP32, 640 square, NMS-free head | Desktop ONNX Runtime CPU | Local phone inference, no uploaded frames |
| backend | YOLO26s PyTorch, FP32, 640 square, default NMS | Local RTX 2070 | Phone sends frames; backend returns detections |

The eventual app will offer a mode selector before capture starts. Switching requires stopping the current session. Phone mode must not upload frames; backend mode will display connection state and stale-frame status. No automatic remote fallback. A native runtime, camera integration and authenticated transport are future work after the phone OS is known.

## Reproduce locally

From the workspace root, after the base setup:

```powershell
$env:UV_CACHE_DIR = Join-Path (Get-Location) '.uv-cache'
uv pip install --python .venv/Scripts/python.exe -r requirements-deployment.txt
$env:YOLO_CONFIG_DIR = Join-Path (Get-Location) '.ultralytics'
.venv/Scripts/yolo.exe export model=yolo26n.pt format=onnx imgsz=640 device=cpu opset=17 simplify=False dynamic=False nms=False
.venv/Scripts/python.exe scripts/run_deployment.py --mode both --image .venv/Lib/site-packages/ultralytics/assets/bus.jpg --frames 100
```

Export is a separate preparation step; rerunning it replaces `yolo26n.onnx`. The runner requires prepared local model files. Use `--mode phone` or `--mode backend` independently, and `--image` to select your own image. Backend mode requires CUDA and fails explicitly if unavailable. The runner uses an exclusive lock, fresh result directory per invocation, sequential child processes and a 600-second timeout per profile. Failed runs can retain partial results; a nonzero exit means the invocation did not complete. An empty result directory may remain when another invocation holds the lock.

## Observed results

Run: `reports/deployment/cdd700badf934272b513456a55e55135/`. Ten warmups plus 100 predictions of the bundled Ultralytics bus/street image, per model, sequentially.

| Profile | Mean prediction | p95 | Serial prediction rate |
| --- | ---: | ---: | ---: |
| Phone candidate on desktop CPU | 45.51 ms | 54.97 ms | 21.97 FPS |
| Backend on RTX 2070 | 26.37 ms | 32.89 ms | 37.92 FPS |

These exclude camera, transport, rendering, depth and tracking. They are neither handset measurements nor sustained video results. Different models, runtimes and heads make this a profile feasibility check, not a controlled architecture ranking. Both detected one bus and four people on this sample; there is no labeled accuracy score. Small distant objects and adverse weather remain untested.

The ONNX output matched the source nano PyTorch output on this image with absolute tolerance 0.002 and relative tolerance 0.0001, using the same NMS-free head. This checks export parity on one input, not dataset-wide accuracy. The 9.4 MB export uses opset 17, fixed input `[1,3,640,640]`, and output `[1,300,6]`. Phone integration still needs matching preprocessing, confidence filtering, coordinate restoration and device execution-provider tests.

The run directory contains `phone.json`, `backend.json`, `sample-validation.json` and annotated `phone-preview.jpg` / `backend-preview.jpg`. JSON records artifact hashes and versions. CPU profile memory is process RSS, not phone RAM; PyTorch allocation fields do not measure ONNX Runtime memory. No recording or personal camera feed was accessed.

## Validation and next steps

Nine automated tests cover mode selection, invalid mode/frame count, missing image/model, child failure, lock contention and unique runs. Format/lint checks apply to the new files; existing profiler was reused with only a head-description correction. Next: target-phone integration and replay benchmarks, then labeled BDD evaluation/fine-tuning. Neither fine-tuning nor a mobile/streaming service was run here.

References: [Ultralytics export contracts](https://docs.ultralytics.com/modes/export/), [ONNX Runtime mobile deployment](https://onnxruntime.ai/docs/tutorials/mobile/). ONNX is a portable starting point; successful desktop execution does not establish handset acceleration or compatibility.
