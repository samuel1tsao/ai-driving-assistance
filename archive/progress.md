# Deployment modes progress

Source: docs/implementation-plan.md and user's request to retain both phone/backend options and run locally.

Scope: two selectable execution profiles; portable phone-candidate export tested on desktop, backend model tested on local GPU. Native phone app and streaming service remain later phases. No handset/OS or labeled data is available.

| Work item | Status | Acceptance / verification |
| --- | --- | --- |
| Define both deployment profiles and runner | DONE | configs/deployment.json and scripts/run_deployment.py; explicit phone/backend/both, no silent fallback |
| Export phone candidate and run both locally | DONE | ONNX export and CPU inference; GPU backend; source/export parity passed on one image |
| Positive and negative tests | DONE | `python -m unittest discover -s tests -v`: 9 passed |
| Performance | DONE | 10 warmups + 100 predictions/profile; CPU 45.51 ms mean, GPU 26.37 ms mean; not end-to-end |
| Race/idempotency | DONE | Lock contention and unique-output tests passed; no existing results overwritten |
| Documentation and coding standard | DONE | Ruff format/check passed; new functions <=250 characters; logs and helper depth reviewed |

Installed ONNX 1.23.0 and ONNX Runtime 1.30.0; refreshed requirements-lock.txt. Sample: bundled street/bus image, not labeled driving evaluation. Results and previews: reports/deployment/cdd700badf934272b513456a55e55135/. Commands and boundaries: docs/deployment-modes.md.

Verification fixes: initial style checks found two over-length functions, root logging, broad exception catch and nested contexts. Shortened functions, used a named logger, narrowed startup exceptions and combined contexts; rerun passed. No coding-standard exceptions. Existing profiler reused; only its head metadata text changed to describe exported artifacts accurately.

Complete for this local compute spike. Remaining: phone OS/device selection, native app and streaming integration, sustained device/network tests, labeled driving evaluation and fine-tuning. No full deployment or driving-accuracy claim. Next: integrate both modes into the chosen mobile platform.

## Guided local application

User requested implementation with clear instructions and a separate detailed architecture document. Scope: a local browser interface around the existing two-mode runner, sample/upload input, readable comparison and downloadable results. Actual handset/runtime and remote streaming remain explicitly pending.

| Work item | Status | Verification |
| --- | --- | --- |
| Guided UI and one-command launcher | DONE | app.py, app_support.py, scripts/start_app.ps1; sample/photo and three mode choices |
| Positive/negative input and report tests | DONE | 13 new tests: valid/corrupt/oversized inputs, partial reports, subprocess failure/timeout and UI messages |
| Performance and race checks | DONE | Real 100-prediction UI submission; exclusive lock tests retained; UUID upload/output isolation verified |
| User guide and detailed architecture | DONE | docs/user-guide.md and docs/architecture.md linked in README and embedded in UI |
| UI, formatting and regression checks | DONE | 22 tests passed; Ruff lint/format and <=250-character new-function checks passed; browser layout inspected |

Plan: reuse run_deployment.py/profile_models.py; add a Streamlit entry point and support module. Local-only server binding. No raw image upload to external services. Progress/status messages explain CPU surrogate versus GPU backend and avoid calling timing an accuracy score.

Implemented and verified 2026-09-19. App is served at http://127.0.0.1:8501. `python -m unittest discover -s tests -q`: 22 passed. `ruff check` and `ruff format --check` passed on app.py, app_support.py and tests/test_guided_app.py. Initial function-length checks identified four oversized support/test functions; simplified row construction and fixtures, then reran successfully. No standard exception required; Streamlit's page is declarative module code and the existing runner owns CLI main/logging.

Real UI integration run: reports/deployment/bb859d581b4c450da31b98ae3aad5cc2; evidence in reports/guided-ui-validation.json. Backend mean/p95 25.15/30.53 ms; phone candidate on desktop CPU 44.75/57.61 ms. 100 predictions per mode. Completed results retained their original mode after controls changed. These are per-image compute timings, not accuracy or device/network measurements. Browser checked at a narrow viewport: headings, radio labels and upload control wrap without overlap; guides and explanations are available as expanders.

Current limits: no camera, new box rendering, actual handset execution, remote streaming or labeled-data evaluation. Image/report retention is manual, OneDrive may sync the workspace, and parent timeout does not guarantee child cancellation; all explained in the guides. Both deployment options remain supported by the local compute layer. Dependency snapshot updated for UI packages. Server intentionally remains running for the user; stop with Ctrl+C in its terminal or end its process.

## Basic evaluation notebook

User requested a simpler starting point: one Jupyter notebook that loads each checkpoint, evaluates labeled accuracy, and measures speed over the same validation images.

| Work item | Status | Verification |
| --- | --- | --- |
| Notebook structure and model explanations | DONE | `notebooks/model_evaluation_basics.ipynb`; eight model descriptions and metric definitions |
| Labeled accuracy/speed function | DONE | Shared `evaluate_model()` routes Ultralytics and torchvision models through compatible adapters and records mAP50, mAP50-95, precision, recall and stage timings |
| Smoke execution | DONE | Executed top-to-bottom with registered `driving-env` kernel; COCO8 smoke outputs saved |
| Full dataset guidance | DONE | COCO validation and BDD100K instructions in notebook and docs/notebook-evaluation.md |

Smoke results are refreshed in the later adapter section after the expanded eight-model run. These are not final rankings because the smoke set is too small and GPU timings vary with warmup and contention. The notebook writes reports/notebook/model_eval_results.csv and .json.

Notebook validation: JSON parsed, nbconvert execution completed without cell errors, outputs and plots were saved in the notebook, and the project kernel used CUDA. The execution emitted normal Windows Jupyter transport warnings; no analysis cell failed. Full COCO/BDD evaluation was not run because no large labeled driving dataset is present and standard Ultralytics COCO setup can require substantial disk space. Next step is to choose/download the dataset before changing `USE_FULL_DATASET` or `DATA_YAML`.

## Notebook self-download and model adapters

User requested that the notebook download missing models/data, use one reusable evaluator, and print a per-model summary. Scope: update the notebook only, add a model-specific adapter for RT-DETR, preserve the simple smoke/full-dataset switch, and keep external model families optional until their dependencies are explicitly installed.

| Work item | Status | Verification |
| --- | --- | --- |
| Download/checkpoint resolver | DONE | Missing local checkpoints invoke the model family's official loader; `rtdetr-l.pt` downloaded during verification |
| Dataset resolver | DONE | Dataset YAML checked and missing configured data downloaded by Ultralytics |
| Reusable evaluator and summaries | DONE | One loop prints description, model family, accuracy, recall and speed; results saved as CSV/JSON |
| Adapter coverage and validation | DONE | YOLO/RT-DETR plus torchvision Faster R-CNN, RetinaNet and SSD adapters; notebook rerun with eight successful models and three random visual prediction panels |

Smoke results on COCO8: RT-DETR-L mAP50-95 0.731 / 1484.71 ms; YOLO26s 0.684 / 112.99 ms; YOLO26n 0.654 / 130.69 ms; YOLOv8n 0.637 / 224.09 ms; Faster R-CNN 0.610 / 1839.41 ms; YOLO11n 0.608 / 186.73 ms; RetinaNet 0.576 / 3597.45 ms; SSD300 0.409 / 607.88 ms. These are smoke checks on four images, and the torchvision metrics use the notebook's shared YOLO-label evaluator; repeat on a full held-out set before comparing families. No labeled driving data is present, so full BDD100K execution remains blocked on dataset acquisition. The notebook will continue to distinguish smoke results from reportable full-set results.

## 3D occupancy track

Occupancy models are not included in the 2D box notebook because they require calibrated multi-camera/temporal inputs and voxel occupancy labels. The separate scope and dataset/metric plan is documented in [docs/occupancy-evaluation-plan.md](docs/occupancy-evaluation-plan.md). The notebook now has adapters for Occ3D-nuScenes, Occ3D-Waymo, OpenOccupancy/OpenOcc v2, and nuScenes-Occupancy v0.1, including dataset-specific free-space IDs and visibility masks. FlashOcc, OccFormer, and GaussianOcc still require their official checkpoints/configs to export prediction `.npz` files; the shared evaluator is ready for those outputs.

## Monocular depth track

The depth notebook and manifest contract are now scaffolded in [notebooks/monocular_depth_evaluation.ipynb](notebooks/monocular_depth_evaluation.ipynb) and [docs/monocular-depth-evaluation.md](docs/monocular-depth-evaluation.md). It compares Depth Anything V2 Metric Outdoor Small with MiDaS DPT-Hybrid, reports depth errors and speed, and renders actual/predicted/error maps. Accuracy execution is pending a real image-to-depth manifest; BDD100K detection images alone are not depth labels. The notebook now prints a `not_ready` row for each model when that manifest is absent, so a run always shows its state.

## Separate evaluation notebooks

The three task tracks are now indexed in [docs/evaluation-notebooks.md](docs/evaluation-notebooks.md): object detection, monocular distance, and 3D occupancy. The [notebooks/occupancy_evaluation.ipynb](notebooks/occupancy_evaluation.ipynb) now supports Occ3D-nuScenes, Occ3D-Waymo, OpenOccupancy/OpenOcc v2, and nuScenes-Occupancy v0.1 through [scripts/occupancy_data.py](scripts/occupancy_data.py). It uses direct relative folders under `data/occupancy/datasets/` rather than environment variables, normalizes dataset-specific label keys, free-space IDs, and visibility masks, then prints one status row per model even before prediction files are present. Real scores remain pending the official dataset archives and model prediction exports.

The app-level association notebook [notebooks/object_distance_fusion_evaluation.ipynb](notebooks/object_distance_fusion_evaluation.ipynb) now measures robust depth sampled inside detected boxes against ground-truth object distances, with MAE/RMSE/within-1m metrics and visual overlays. It is validated in no-data mode and awaits a projected-LiDAR fusion manifest. It now prints a `not_ready` row when that manifest is absent.
