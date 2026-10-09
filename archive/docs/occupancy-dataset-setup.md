# Occupancy dataset setup

The occupancy notebook evaluates saved outputs from FlashOcc, OccFormer, and GaussianOcc against four supported label formats. It does not bypass dataset registration or download large gated archives automatically.

## Choose one dataset

| Notebook key | Dataset root | Official labels | Notes |
|---|---|---|---|
| `occ3d_nuscenes` | extracted Occ3D-nuScenes root | `trainval/gts/<scene>/<frame>/labels.npz` | Uses `semantics` and `mask_camera`; free class is 17. |
| `occ3d_waymo` | extracted Occ3D-Waymo root | `validation/<scene>/*_04.npz` | Uses the 0.4 m `voxel_label` and `final_voxel_state`; free class is 15. |
| `openoccupancy` | OpenOccupancy `data/nuscenes` or `openocc_v2` root | `openocc_v2/<scene>/<token>/labels.npz` | Uses `semantics`; class 0 is empty/noise for semantic mIoU. |
| `nuscenes_occupancy` | extracted `nuScenes-Occupancy-v0.1` root | scene label files | Uses the dense 0.2 m label grid; class 0 is empty/noise for semantic mIoU. |

Official references: [Occ3D](https://github.com/Tsinghua-MARS-Lab/Occ3D), [OpenOccupancy](https://github.com/JeffWang987/OpenOccupancy), and [nuScenes](https://www.nuscenes.org/nuscenes). Occ3D requires the original nuScenes or Waymo data and their terms of use. OpenOccupancy requires the original nuScenes data plus its released annotations.

The project currently has an extracted Occ3D-nuScenes ground-truth package at `data/occupancy/datasets/occ3d_nuscenes/trainval/gts/`. The original archives are retained in `data/occupancy/downloads/`; the 30 GB image archive remains compressed until image-level model inference is needed.

For a first smoke run, the public [UniOcc Hugging Face mirror](https://huggingface.co/datasets/tasl-lab/uniocc) provides mini labeled packages for Occ3D-nuScenes, Occ3D-Waymo, and OpenOccupancy. Those files use `occ_label` and `occ_mask_camera`; the project adapter supports that mirror format as well as the original layouts.

## Run the notebook

Place the extracted dataset in the matching project folder:

```text
data/occupancy/datasets/occ3d_nuscenes/
data/occupancy/datasets/occ3d_waymo/
data/occupancy/datasets/openoccupancy/
data/occupancy/datasets/nuscenes_occupancy/
```

In the configuration cell of [occupancy_evaluation.ipynb](../notebooks/occupancy_evaluation.ipynb), edit the direct relative paths by changing `DATASET` and `DATASET_SPLIT`. Run all cells. When the folder is valid, the notebook normalizes labels into `data/occupancy/normalized/` and writes `data/occupancy/manifest.csv`. The manifest keeps the source label path, dataset, split, voxel size, and free-space ID.

For OpenOccupancy or nuScenes-Occupancy, use `DATASET_SPLIT = 'val'` only when the official `nuscenes_infos_val_occ.pkl` metadata is present. Otherwise use `DATASET_SPLIT = 'all'` and create a held-out manifest before reporting accuracy; do not mix training and validation frames.

## Add model predictions

Each model runner must save one `.npz` per manifest frame here:

```text
data/occupancy/predictions/<dataset>/<model>/<scene>/<frame>.npz
```

Use model folders `flashocc`, `occformer`, and `gaussianocc`. Each file must contain an integer `occupancy` array with the exact shape, voxel order, coordinate range, and class IDs expected by the selected dataset. After prediction export, rerun all notebook cells. The notebook will then report masked semantic mIoU, occupied precision/recall, free-space recall, and voxel-slice comparisons.

The official model repositories use different legacy PyTorch/MMCV environments. Keep those runners isolated and use this notebook as the common scoring layer. A prediction array with a different voxel resolution or coordinate order is invalid until it has been explicitly resampled and transformed.
