# Setup and validation — 19 September 2026

## Completed

- Created a project-local `.venv` using Python 3.11.0, CUDA PyTorch 2.7.1+cu128, torchvision 0.22.1+cu128, Ultralytics 8.4.155 and profiling dependencies. Exact installed versions are in `requirements-lock.txt`.
- Verified CUDA inference on the NVIDIA RTX 2070 (8 GiB). Downloaded YOLOv8n, YOLO11n, YOLO26n and YOLO26s public checkpoints into the workspace.
- Added PowerShell setup, reusable model profiling and fine-tuning scripts, plus a ten-class BDD100K configuration template.
- Syntax compilation passed. Missing training data and a zero-frame benchmark are rejected with argument errors, as expected.
- All four models completed the revised profiler with 10 warmups and 100 measured predictions each. Updated precision syntax eliminated repeated deprecation warnings. Created the local Ultralytics settings directory to prevent fallback outside the project.

## Local smoke measurements

Source: `reports/local_profile.json`. Batch 1, FP32, square 640 input, synthetic black 1280x720 source, PyTorch, default external NMS. Timings include prediction preprocessing and postprocessing. One sequential pass, with other desktop applications running.

| Model | Mean prediction ms | p95 ms |
| --- | ---: | ---: |
| YOLOv8n | 17.98 | 21.29 |
| YOLO11n | 21.39 | 24.11 |
| YOLO26n | 22.07 | 26.97 |
| YOLO26s | 20.91 | 25.32 |

The small model's lower observed mean than the nano model is not evidence it is inherently faster: these short sequential runs are sensitive to GPU scheduling, clock state and background workload. Repeat in randomized order on road imagery before drawing performance conclusions. Blank images underexercise postprocessing. The JSON records checkpoint hashes, stage times and memory; PyTorch peak memory excludes driver/runtime allocations and is not total GPU usage.

## Still needed

Labeled BDD100K images/annotations, explicit common-class mapping, dataset conversion/audit, fixed scene splits, and an accuracy evaluator are needed before baseline accuracy or fine-tuning claims. No training has been started. The training entry point has syntax and invalid-input validation only; actual training awaits data. RF-DETR and depth candidates have been researched but are not installed or benchmarked.

The phone model/OS and backend deployment budget remain open. Both deployment tracks are preserved. There is no mobile UI, streaming server, ROS 2 pipeline or collision-warning implementation yet. This is the model-analysis foundation requested for the first stage.

## Reproduce

From the workspace root in PowerShell:

```powershell
.venv/Scripts/python.exe -m compileall -q scripts
.venv/Scripts/python.exe scripts/profile_models.py --frames 100
```

Use a unique `--output` path to preserve previous runs. Avoid concurrent GPU benchmarks. Setup requires downloading packages and checkpoints; no cloud compute or paid services were provisioned.
