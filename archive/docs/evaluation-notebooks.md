# Evaluation notebooks

The project now separates three perception tasks because their labels and metrics are different:

| Task | Notebook | Ground truth | Main metrics |
|---|---|---|---|
| Objects | [model_evaluation_basics.ipynb](../notebooks/model_evaluation_basics.ipynb) | 2D boxes | mAP50, mAP50-95, precision, recall, latency, visual boxes |
| Distance | [monocular_depth_evaluation.ipynb](../notebooks/monocular_depth_evaluation.ipynb) | Dense or sparse depth in meters | AbsRel, RMSE, log RMSE, delta accuracy, latency, error maps |
| Occupancy | [occupancy_evaluation.ipynb](../notebooks/occupancy_evaluation.ipynb) | Occ3D-nuScenes, Occ3D-Waymo, OpenOccupancy, or nuScenes-Occupancy voxel labels | masked mIoU, occupied/free-space precision and recall, near-ego recall, voxel-slice visual checks |
| Object-distance fusion | [object_distance_fusion_evaluation.ipynb](../notebooks/object_distance_fusion_evaluation.ipynb) | Detector boxes plus metric object distances | distance MAE, RMSE, within-1m accuracy, fused visual overlays |

Run each notebook on a held-out scene split. Do not merge the scores into one leaderboard: a detector can find a car without estimating its distance, and a depth map can estimate distance without understanding free space or object identity.

Use the fusion notebook for the app-level question: “How accurately can we assign a distance to each detected object?” Keep its results separate from the standalone detector and depth results so failures can be diagnosed.

The object notebook can run on its COCO8 smoke set. The distance notebook needs an image-to-depth manifest. The occupancy notebook can build its manifest from the four supported official layouts, but it still needs model prediction `.npz` files in the same voxel convention. Occ3D-nuScenes is the recommended first run because its validation layout and camera visibility mask are explicit; Occ3D-Waymo, OpenOccupancy, and nuScenes-Occupancy are separate dataset rows rather than scores merged into one leaderboard.
