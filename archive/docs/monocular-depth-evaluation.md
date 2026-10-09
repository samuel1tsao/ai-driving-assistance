# Monocular depth evaluation

The notebook at [notebooks/monocular_depth_evaluation.ipynb](../notebooks/monocular_depth_evaluation.ipynb) compares two complementary monocular depth models:

- **Depth Anything V2 Metric Outdoor Small** produces outdoor metric-depth predictions in meters. The [official model card](https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf) documents its Transformers loader and Virtual KITTI 2 metric-depth fine-tuning.
- **MiDaS DPT-Hybrid** is a relative-depth baseline. The [official model card](https://huggingface.co/Intel/dpt-hybrid-midas) describes it as monocular relative depth; the notebook applies per-image scale/shift alignment before computing depth errors.

The notebook downloads missing Hugging Face checkpoints, evaluates the same image/depth pairs, reports AbsRel, RMSE in meters, log RMSE, delta accuracy, and inference speed, then renders actual depth, predictions, and absolute-error heatmaps for up to three random samples.

## Dataset choice

The manifest must pair each RGB image with a depth map. The depth map can be a floating-point `.npy` array in meters or a 16-bit PNG with a `depth_scale` column. Invalid or missing pixels should be zero. See [data/depth/manifest.example.csv](../data/depth/manifest.example.csv).

- **nuScenes:** good first driving source because it provides synchronized camera/LiDAR data, calibration, and poses. Project LiDAR points into each camera to create sparse depth maps.
- **Argoverse 2 Sensor:** also suitable because it provides synchronized ring/stereo cameras, LiDAR, calibration, and poses. Its scale is large, so start with a small held-out subset.
- **Waymo Open:** suitable but requires its own access, file format, and conversion tooling.
- **BDD100K:** useful for visual domain checks, but ordinary BDD100K detection labels do not provide metric depth ground truth. Do not report depth accuracy from BDD images alone.

## Metrics and limits

Lower AbsRel, RMSE, and log RMSE are better; higher delta values are better. MiDaS scores are scale/shift aligned and therefore do not prove calibrated meters. Depth accuracy should be evaluated on held-out scenes and reported separately for near range, far range, night, rain, glare, and occlusion. Collision assistance additionally needs temporal stability, object association, calibrated camera geometry, and time-to-collision tests.

Run the notebook after creating `data/depth/manifest.csv`:

```powershell
uv pip install --python .venv/Scripts/python.exe -r requirements-notebook.txt
.venv/Scripts/python.exe -m jupyter lab
```
