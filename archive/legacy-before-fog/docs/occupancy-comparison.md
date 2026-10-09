# Occupancy comparison: setup and interpretation

Open `notebooks/occupancy_evaluation.ipynb`, start Docker Desktop, and run all cells using the project's normal Python kernel. The notebook calls the isolated Linux GPU model runner and then scores its saved predictions in the notebook environment. No host environment variables are required or changed.

## Models

| Model | What differs | Camera input |
| --- | --- | --- |
| FlashOCC M0 | Supervised ResNet-50; 128-channel BEV neck/head, 1 m depth bins | Six 256 × 704 images |
| FlashOCC M1 | Same backbone; 256-channel BEV neck/head, 0.5 m depth bins | Six 256 × 704 images |
| GaussianOcc | ResNet-101 with a 3D volume decoder; Gaussian rendering provides self-supervision during training | Six 384 × 640 images |

M0 and M1 are variants of one family. GaussianOcc has a different architecture and training supervision. Comparing released checkpoints is useful for selecting a system, but it does not isolate architecture from training differences. Original OccFormer nuScenes weights target LiDAR segmentation; they are not a verified drop-in Occ3D checkpoint and are excluded from the runnable list. Waymo labels are present, but these nuScenes models are not validated Waymo runners.

Sources: [FlashOCC model zoo](https://github.com/Yzichen/FlashOCC), [M0 configuration](https://github.com/Yzichen/FlashOCC/blob/master/projects/configs/flashocc/flashocc-r50-M0.py), [GaussianOcc](https://github.com/GANWANSHUI/GaussianOcc), [OccFormer model zoo](https://github.com/zhangyp15/OccFormer#model-zoo).

## Data and architecture

1. `scripts/prepare_occupancy_inputs.py` selects seeded official nuScenes validation labels and joins their sample IDs to original key-frame camera metadata. It extracts only the needed images from the downloaded archive.
2. `data/occupancy/inputs/nuscenes/frames.json` stores six image paths, camera intrinsics, camera-to-ego transforms and ego poses per frame. Paths are project-relative. Missing views prevent writing a complete manifest.
3. The notebook passes the same ordered frame list to each model runner. Each runner uses its own official image preprocessing and calibration convention. Labels are used only for scoring, not model input.
4. `scripts/run_flashocc.py` uses the official config, strict checkpoint loading, image pipeline and prediction method. `scripts/run_gaussianocc.py` uses the official encoder, volume decoder and occupancy sampling. Its subclass skips rendered RGB/depth images, which do not feed the occupancy logits, and streams the six camera volumes through the same masked-mean aggregation to fit 8 GB VRAM. Before inference it checks streamed versus original aggregation on a reduced grid with real features (`allclose`, absolute/relative tolerance 1e-5). Camera ordering, weights, full inference grid and input resolution remain unchanged. Floating-point accumulation order can differ slightly.
5. Predictions are atomic NPZ files under `data/occupancy/predictions/occ3d_nuscenes/<model>/<scene>/<frame>.npz`. Integer `occupancy` arrays use the original 200 × 200 × 16 label coordinates and 18 classes. The files also contain measured latency and cache provenance.
6. `notebooks/_support/occupancy.py` applies one confusion-matrix evaluator and visibility mask to all models. `occupancy_inference.py` keeps container commands out of the notebook.

There are 6,019 available labels across all 150 official validation scenes. The prepared benchmark uses 100 seeded frames, not all 6,019. Six calibrated surround cameras are not equivalent to a single phone camera.

## Read the results

- Semantic mIoU averages occupied-class IoUs after pooling confusion counts across visible voxels. Free space is excluded from that average. Absent classes are excluded rather than scored perfectly.
- Occupied precision measures false occupied-space predictions; occupied recall measures missed occupied space. Free-space recall checks free-space recovery.
- All models use the camera visibility mask and the same labels. Visuals show a horizontal slice at z = 0.8 m; scores cover all labeled heights.
- Mean and p95 latency describe one six-camera frame after three warm-ups, with explicit CUDA synchronization. Timing starts with prepared GPU inputs and ends with the occupancy output. It excludes resizing/decoding, setup, downloads, saving and app/network latency. GaussianOcc image rendering is excluded because this benchmark targets occupancy.
- This is a local PyTorch benchmark on an RTX 2070, not the authors' optimized TensorRT benchmark. Do not compare these milliseconds directly with published FPS on other hardware.

## Repeat and extend

In the notebook, `RUN_MODELS = True` runs missing/changed model outputs; `REFRESH = True` explicitly regenerates outputs and timings. `RUN_MODELS = False` is useful for quickly inspecting existing results. Errors have no accuracy score. A notebook lock prevents concurrent benchmark runs.

To recreate the research runtime after deleting its container:

```powershell
./scripts/setup_occupancy.ps1
```

The script downloads missing checkpoints and pinned sources into the project. It creates `driving-flashocc` from the published CUDA 11.3/PyTorch 1.10 container and compiles the research extensions. This image uses Python 3.7; one starred tuple return in FlashOCC is parenthesized for compatibility. GaussianOcc's unused pose-refinement import is deferred until that feature is called. Neither patch changes occupancy calculations. Headless OpenCV avoids a desktop graphics dependency.

Prepare more images using explicit archive paths; this can take time because gzip archives must be scanned:

```powershell
.venv/Scripts/python.exe scripts/prepare_occupancy_inputs.py --metadata C:/Users/tsao1/Downloads/v1.0-trainval_meta.tgz --images data/occupancy/downloads/imgs-001.tar.gz --limit 100 --seed 42
```

Increase `--limit` and then the notebook's `MAX_SAMPLES`. All models must be rerun on that same prepared sample list. Existing extracted files are reused; original downloaded archives remain intact. Checkpoint and manifest fingerprints prevent reusing predictions from different weights or input manifests. After changing runner code or model configs, explicitly use `REFRESH = True`. If a notebook or container is forcibly terminated, check that no inference process remains before removing a stale `.running` or `.notebook-running` lock.

## Limits for driving assistance

These are useful component tests for semantic scene reconstruction and runtime cost. They do not validate collision warning or establish safe deployment. A later sequence benchmark needs distance error, dangerous obstacle misses, false alerts, time-to-collision/warning lead time, temporal stability, weather/night slices and complete device-to-backend latency. GaussianOcc also carries research/noncommercial licensing restrictions in parts of its source; review licenses before product use.
