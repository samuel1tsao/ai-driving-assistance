# 3D occupancy evaluation track

The current model notebook evaluates 2D object detectors. That is useful for identifying visible cars, people, bicycles, and signs, but it does not tell us whether the space around the vehicle is occupied, free, behind an occluder, or at a safe distance. Occupancy models should therefore be evaluated in a separate track rather than inserted into the 2D box leaderboard.

## Models to add

- [FlashOcc](https://github.com/Yzichen/FlashOCC): an efficient camera-based occupancy model. Its official setup uses nuScenes metadata and occupancy ground truth, with optional Occ3D and TensorRT paths.
- [OccFormer](https://github.com/zhangyp15/OccFormer): a dual-path transformer for vision-based 3D semantic occupancy. It is evaluated with occupancy volumes, not 2D detection boxes.
- [GaussianOcc](https://github.com/GANWANSHUI/GaussianOcc): a Gaussian-splatting occupancy model. Its official repository uses DDAD or nuScenes data and an older, separate Python/PyTorch environment, so it should run in an isolated environment.

These models cannot be evaluated from the current COCO8 or BDD100K 2D detection labels. The occupancy notebook now supports four label formats through `scripts/occupancy_data.py`:

| Dataset | Official label layout | Free-space ID | Evaluation mask | Geometry |
|---|---|---:|---|---|
| [Occ3D-nuScenes](https://github.com/Tsinghua-MARS-Lab/Occ3D) | `trainval/gts/<scene>/<frame>/labels.npz` with `semantics` | 17 | `mask_camera` | 0.4 m, 200×200×16 |
| [Occ3D-Waymo](https://github.com/Tsinghua-MARS-Lab/Occ3D) | `validation/<scene>/*_04.npz` with `voxel_label` | 15 | `final_voxel_state` | 0.4 m, 200×200×16 |
| [OpenOccupancy / OpenOcc v2](https://github.com/JeffWang987/OpenOccupancy) | `openocc_v2/<scene>/<token>/labels.npz` with `semantics` | 0 | dense labels | 0.4 m, 200×200×16 |
| [nuScenes-Occupancy v0.1](https://github.com/JeffWang987/OpenOccupancy) | extracted `nuScenes-Occupancy-v0.1` scene labels | 0 | dense labels | 0.2 m, 512×512×40 |

The notebook normalizes each label file to `occupancy` plus `valid_mask`, writes `data/occupancy/manifest.csv`, and evaluates saved predictions under `data/occupancy/predictions/<dataset>/<model>/<scene>/<frame>.npz`. This keeps the official camera-visible mask and free-space convention instead of treating every dataset as if class 0 were free. Each model still needs its own checkpoint/config loader, camera calibration handling, temporal-frame handling, and output-to-voxel adapter. The notebook deliberately evaluates those adapters in the common environment because the official model repositories require incompatible legacy environments.

The official datasets are gated or large. Occ3D requires the original nuScenes or Waymo data and the respective terms of use; OpenOccupancy requires the original nuScenes data plus its released annotations. The notebook does not silently download these archives or bypass registration. Set `OCCUPANCY_DATA_ROOT` after downloading and extracting the dataset, then run the notebook with `OCCUPANCY_DATASET=occ3d_nuscenes` (or another supported key) and `OCCUPANCY_SPLIT=val`.

## Metrics

The occupancy track should report:

- semantic occupancy mIoU and per-class IoU;
- occupied/free-space precision and recall;
- near-ego obstacle recall, where a miss is especially dangerous;
- ray IoU or depth consistency when supported by the dataset;
- inference latency, peak GPU memory, and input frame count;
- visualization of predicted occupied voxels beside ground truth and camera views.

For collision assistance, aggregate mIoU is not enough. A follow-up scenario evaluation should measure false-free-space rate along the planned path, obstacle recall within a configurable stopping distance, temporal stability across consecutive frames, and time-to-collision warning performance. Occupancy is a perception input to that module; it is not itself a collision decision.

## Recommended order

1. Keep the existing 2D notebook for box detection and visual box inspection.
2. Prepare one supported occupancy dataset and verify camera calibration, ego pose, and voxel coordinate conventions.
3. Add FlashOcc first as the practical baseline, then OccFormer as a transformer comparison.
4. Add GaussianOcc in its own environment after the first two models produce valid occupancy outputs.
5. Compare occupancy metrics, visual slices, memory, and end-to-end warning behavior separately from the 2D detector results.

This separation prevents a high 2D mAP score from being mistaken for reliable 3D free-space or collision reasoning.
