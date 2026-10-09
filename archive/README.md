# AI Driving Assistance — model research starter

## Start with the guided app

```powershell
./scripts/start_app.ps1
```

Open **http://127.0.0.1:8501**, choose **Compare both**, keep the included sample, and click **Run local test**. The interface explains each result. For first-time setup and troubleshooting, read the [user guide](docs/user-guide.md). The phone candidate runs on desktop CPU in this demo; actual phone deployment is still pending.

This workspace provides an isolated Python environment, local detector profiler, and fine-tuning entry point for comparing on-phone inference with a streamed backend. It does not yet implement the mobile app, streaming, ROS 2, depth fusion or collision alerts.

## Setup (PowerShell)

Run commands from this directory. `uv` must be installed.

```powershell
./scripts/setup.ps1
.venv/Scripts/python.exe scripts/profile_models.py
```

The first profile downloads public pretrained checkpoints. Default input is a synthetic black frame: results are execution smoke timings, not driving accuracy or full application FPS. Use a representative local image for a more useful prediction benchmark:

```powershell
.venv/Scripts/python.exe scripts/profile_models.py --image data/sample-road.jpg --frames 200 --output reports/road-profile.json
```

The profiler warms up each model, synchronizes CUDA, and records batch-one FP32 wall-clock p50/p95, stage times, process RSS, peak allocated/reserved PyTorch VRAM, versions and checkpoint hashes. Video, network, decode and alert delays are excluded. It saves after each model; use distinct output paths for concurrent runs. Models run sequentially to avoid GPU contention.

## Fine-tuning

Obtain BDD100K images and detection annotations through the [official project](https://github.com/bdd100k/bdd100k), prepare normalized YOLO labels, and copy `configs/bdd100k.example.yaml` with an absolute dataset path. Each label row is `class x_center y_center width height` normalized to [0,1]. The example lists all ten BDD detection classes. Dataset conversion and taxonomy-aware baseline evaluation are next steps once source data is available.

```powershell
.venv/Scripts/python.exe scripts/train.py --data configs/bdd100k.local.yaml --model yolo26n.pt --epochs 30 --batch 4
```

Do not evaluate unchanged COCO weights directly against BDD class IDs. See the analysis for the required mapping and split rules. Training has not been run without labeled data. `runs/`, checkpoints, datasets and local environments are ignored by Git.

## Select phone or backend compute

Both deployment tracks are supported by a local compute runner. Phone mode tests the portable ONNX candidate on desktop CPU; backend mode uses the local GPU. See [deployment setup and results](docs/deployment-modes.md) for the export command and limits.

```powershell
.venv/Scripts/python.exe scripts/run_deployment.py --mode both --image .venv/Lib/site-packages/ultralytics/assets/bus.jpg
```

Use `--mode phone` or `--mode backend` to run just one. This does not yet start a mobile app or streaming service.

## Documentation and results

- [Basic model evaluation notebook](notebooks/model_evaluation_basics.ipynb)
- [Notebook instructions](docs/notebook-evaluation.md)
- [3D occupancy evaluation track](docs/occupancy-evaluation-plan.md)
- [Occupancy dataset adapter](scripts/occupancy_data.py)
- [Occupancy dataset setup](docs/occupancy-dataset-setup.md)
- [Monocular depth evaluation](docs/monocular-depth-evaluation.md)
- [Evaluation notebook index](docs/evaluation-notebooks.md)
- [Object-distance fusion notebook](notebooks/object_distance_fusion_evaluation.ipynb)
- [Step-by-step user guide](docs/user-guide.md)
- [Detailed architecture](docs/architecture.md)
- [Draft implementation plan](docs/implementation-plan.md)
- [Deployment modes and local results](docs/deployment-modes.md)
- [Implementation progress](progress.md)
- [Initial model analysis and experiment design](docs/model-analysis.md)
- [Setup status and validation](docs/setup-status.md)
- `reports/local_profile.json`: generated local timing results, when a profile succeeds.
- `requirements-lock.txt`: installed dependency snapshot, when setup completes; CUDA PyTorch uses the index in `scripts/setup.ps1`.

Source references: the user's `Focused Project Proposal Overview.md` and `model_analysis_profiling_blueprint.md` in Downloads; originals are untouched.
