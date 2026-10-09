# Model comparison notebooks

Occupancy model setup and interpretation are in the [comparison guide](occupancy-comparison.md), with the [implementation plan](occupancy-inference-implementation.md) and [progress](../progress.md).

Start with these three notebooks. Each has settings, labeled samples, a model loop, a results table and visual examples. Longer implementation code lives in `notebooks/_support/`.

| Notebook | Labels | What runs now |
| --- | --- | --- |
| [Object detection](../notebooks/model_evaluation_basics.ipynb) | Local BDD100K validation JSON and images | Eight pretrained detectors; shared-class accuracy, mean/p95 latency, actual/predicted boxes |
| [Occupancy](../notebooks/occupancy_evaluation.ipynb) | Occ3D-nuScenes validation cameras and labels | FlashOCC M0/M1 and GaussianOcc runners; shared accuracy/speed scoring, source images and label/prediction slices. Waymo supports label/export inspection only. |
| [Distance / depth](../notebooks/monocular_depth_evaluation.ipynb) | RGB paired with depth labels in meters | Metric-model loading, scoring and plots once the depth manifest exists; the local manifest is not prepared yet |

## Run

From the project directory:

```powershell
.venv/Scripts/python.exe -m jupyter lab notebooks
```

Choose the project Python kernel and **Restart Kernel and Run All Cells**. Begin with `MAX_SAMPLES = 100`. For occupancy, start Docker Desktop first; 100 matched camera frames are prepared, and `None` means all prepared frames. Models run sequentially on the same samples. An error or missing-data row is not an accuracy result.

The project environment's missing Python base was replaced with a project-local Python 3.11 runtime under `.runtime/python/`. No environment variables are required or changed by the new notebooks. If creating a new environment, install the packages in [requirements-notebook.txt](../requirements-notebook.txt), choosing the PyTorch build suitable for your GPU.

## Fair comparisons

- Same frame IDs within each task; the tasks have different label requirements and do not share one universal dataset.
- Detection uses original BDD boxes, maps pedestrian/rider to person and excludes unsupported traffic signs. Every detector uses the same simplified 101-point AP calculation, not a mixture of library validators. This is not an official BDD leaderboard score.
- Depth compares metric models separately from oracle-aligned MiDaS. MiDaS fitting uses ground truth and therefore cannot establish real-world distance accuracy.
- Occupancy pools confusion counts across camera-visible voxels. Semantic mIoU covers occupied classes 0–16; missing classes are not counted as perfect. The old shared `data/occupancy/manifest.csv` is not trusted or reused.
- Waymo uses its own 15 occupied classes plus free space. Raw ground-truth ID 23 is mapped to 15; its valid region is `final_voxel_state & infov`. Only 0.4 m validation files are loaded. nuScenes and Waymo scores are saved separately and are not directly comparable rankings.
- Detector timing includes its native transformations. Occupancy timing starts with prepared GPU inputs and includes model/output computation after three warm-ups with CUDA synchronization. Image decoding/resizing, downloads, loading and app/network overhead are excluded for occupancy; file-read speed is never reported as inference speed.
- Models retain documented native transforms. SSD uses 300-pixel inputs; other detector adapters use a 640-pixel limit. Compare these as deployed pipelines rather than claiming equal internal resolutions.

Outputs go to `reports/<task>/results.csv` and `run.json`, including selected IDs, a sample-list hash and settings. A 100-image run is a preliminary comparison, not driving-safety validation.

## Occupancy inputs and model execution

The labels are at `data/occupancy/datasets/occ3d_nuscenes/trainval/gts/<scene>/<frame>/labels.npz`. The official validation scene list is cached in `data/occupancy/nuscenes_val_scenes.json`. The downloaded image archive alone does not supply a configured model runner: each model also needs the matching checkpoint, camera calibration/poses, preprocessing and compatible dependencies.

The matching cameras and calibration are prepared under `data/occupancy/inputs/nuscenes/`. The notebook runs [FlashOCC](https://github.com/Yzichen/FlashOCC) M0/M1 and [GaussianOcc](https://github.com/GANWANSHUI/GaussianOcc) in an isolated Docker GPU runtime, then evaluates their predictions. GaussianOcc streams camera aggregation to fit 8 GB VRAM and excludes unused image rendering. [OccFormer](https://github.com/zhangyp15/OccFormer)'s original nuScenes checkpoint is LiDAR segmentation, so it is excluded until compatible occupancy weights are available. See the [guide](occupancy-comparison.md) for setup and exact timing scope.

Each runner should export:

```python
np.savez_compressed(output_path, occupancy=prediction, inference_ms=measured_ms)
```

Path: `data/occupancy/predictions/occ3d_nuscenes/<flashocc_m0|flashocc_m1|gaussianocc>/<scene>/<frame>.npz`. `occupancy` is an integer `(200, 200, 16)` grid in the original Occ3D axes, with IDs 0–17 and free ID 17. All selected frames must exist for a model to be scored. The supplied runners save real timings and weight/input fingerprints.

## Waymo download setup

The downloaded Drive ZIPs contain Occ3D occupancy annotations at two resolutions, not a complete camera/LiDAR sensor release. The imported 0.4 m validation labels live at `data/occupancy/datasets/occ3d_waymo/validation/<scene>/<frame>_04.npz`. Large archives remain in Downloads; only the needed validation NPZs are extracted into the project. Existing 0.1 m data is preserved but ignored by this notebook.

The current selection is nuScenes for real model comparison. To inspect Waymo labels or compatible externally generated predictions, use `DATASET = 'occ3d_waymo'` and `RUN_MODELS = False`. The supplied nuScenes checkpoints cannot run this Waymo benchmark. Reports go to `reports/occupancy/<dataset>/`, so datasets remain separate.

The new import contains **25,355 verified 0.4 m frames from 128 scenes**. Including 198 verified frames from an already-extracted scene, the notebook has **25,553 frames from 129 of the expected 202 validation scenes**. This is a partial local benchmark. The import report at `reports/occupancy/waymo_import.json` records imported and total local coverage separately, missing scenes, archive provenance and file hashes. ZIP files in the latest download batch are numbered 001–099 with 003 absent; the missing scene list, rather than archive numbering alone, determines coverage.

Waymo predictions belong under `data/occupancy/predictions/occ3d_waymo/<model>/<scene>/<frame>.npz`. They must already use normalized IDs 0–15 and the `(200,200,16)` grid. A nuScenes checkpoint/export is not automatically compatible with Waymo cameras, coordinates and classes. Model inference is still pending; labels alone do not produce model accuracy or speed.

The raw-label conversion follows the authors' [CVT-Occ dataset instructions](https://github.com/Tsinghua-MARS-Lab/CVT-Occ/blob/main/docs/dataset.md). The notebook explicitly scores the intersection of camera visibility and field of view for Waymo.

## Depth labels still needed

Create `data/depth/manifest.csv` with `image,depth,depth_scale,split` columns. Paths are relative to that CSV. `depth` is a 2D `.npy` or depth PNG aligned to the RGB image; divide values by `depth_scale` for meters. Zero is invalid. Use `split=val` for independently held-out scenes.

nuScenes provides LiDAR, camera intrinsics/extrinsics and timestamped ego poses. A preparation step must transform LiDAR points into the camera frame at the image timestamp, project them into pixels and resolve overlaps with the nearest positive camera-axis depth. Ordinary BDD boxes and Occ3D voxels are not substitute depth maps. Sparse LiDAR evaluation only assesses labeled pixels.

[Object-distance fusion](../archive/notebooks/object_distance_fusion_evaluation.ipynb) is archived as an older experiment, leaving three active notebooks. Its paths need updating before reuse. Surface depth and object-center range are different targets; its outputs should not be presented as part of the three model benchmarks.

The previous app and documentation remain under `archive/`.

## Verification in this workspace

The notebook refactor was exercised on the local RTX 2070: all eight detection models completed a 100-image BDD100K comparison, with saved tables and 24 label/prediction example figures. The occupancy notebook discovered 6,019 labeled frames in the 150 official validation scenes and rendered three ground-truth examples. The earlier 34,149-frame manifest mixed training and validation and is no longer used.

Depth executed its missing-data path; no depth accuracy or real prediction plots have been verified yet. Occupancy now has real pretrained inference; see the [progress record](../progress.md) for completed runs and validation evidence.

Metric checks cover perfect/wrong boxes, masked occupancy, invalid class IDs, exact metric depth, inverse-depth alignment and non-finite predictions. To repeat checks and execute the notebooks, run `.venv/Scripts/python.exe scripts/check_notebooks.py`. Model downloads may need network access on the first run.
